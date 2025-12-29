"""
Stage 6: Graph Materialization Pipeline

Creates deterministic, read-only graph-shaped tables for visualization/export,
using Stage 5 (assertion_temporalized, conflicts) as the authoritative lifecycle
+ valid-time source.

Additionally generates temporal profile structures that describe entities through
clustered semantic categories partitioned by configurable time windows.
"""
import hashlib
import json
import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Tuple, Set, Callable
import pendulum
import yaml


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = get_logger(__name__)


# ===| UTILITIES |===

class JCS:
    """
    JSON Canonicalization Scheme (RFC 8785) implementation.

    Provides deterministic JSON serialization for hashing and ID generation.
    All JSON stored in the database MUST pass through this canonicalizer.
    """

    @staticmethod
    def canonicalize(obj: Any) -> str:
        """Convert Python object to JCS-canonical JSON string."""
        return JCS._serialize(obj)

    @staticmethod
    def canonicalize_bytes(obj: Any) -> bytes:
        """Convert to canonical JSON and encode as UTF-8."""
        return JCS.canonicalize(obj).encode('utf-8')

    @staticmethod
    def _serialize(obj: Any) -> str:
        """Recursively serialize object to JCS-canonical form."""
        if obj is None:
            return 'null'
        elif isinstance(obj, bool):
            return 'true' if obj else 'false'
        elif isinstance(obj, int):
            return str(obj)
        elif isinstance(obj, float):
            if obj != obj:  # NaN
                raise ValueError("NaN is not allowed in JCS")
            if obj == float('inf') or obj == float('-inf'):
                raise ValueError("Infinity is not allowed in JCS")
            s = repr(obj)
            if 'e' in s or 'E' in s:
                s = s.lower()
            return s
        elif isinstance(obj, str):
            return JCS._escape_string(obj)
        elif isinstance(obj, (list, tuple)):
            items = ','.join(JCS._serialize(item) for item in obj)
            return f'[{items}]'
        elif isinstance(obj, dict):
            sorted_keys = sorted(obj.keys(), key=lambda k: k.encode('utf-16-be'))
            items = ','.join(
                f'{JCS._escape_string(k)}:{JCS._serialize(obj[k])}'
                for k in sorted_keys
            )
            return '{' + items + '}'
        else:
            raise TypeError(f"Cannot serialize {type(obj)} to JCS")

    @staticmethod
    def _escape_string(s: str) -> str:
        """Escape a string for JSON according to RFC 8785."""
        result = ['"']
        for char in s:
            code = ord(char)
            if char == '"':
                result.append('\\"')
            elif char == '\\':
                result.append('\\\\')
            elif char == '\b':
                result.append('\\b')
            elif char == '\f':
                result.append('\\f')
            elif char == '\n':
                result.append('\\n')
            elif char == '\r':
                result.append('\\r')
            elif char == '\t':
                result.append('\\t')
            elif code < 0x20:
                result.append(f'\\u{code:04x}')
            else:
                result.append(char)
        result.append('"')
        return ''.join(result)


class IDGenerator:
    """Basic deterministic UUID generation using namespace-based UUIDv5."""

    def __init__(self, namespace: uuid.UUID):
        """Initialize with namespace UUID."""
        self.namespace = namespace

    def generate(self, components: List[Any]) -> str:
        """Generate UUIDv5 from component array."""
        encoded = []
        for c in components:
            if c is None:
                encoded.append("__NULL__")
            elif c == "":
                encoded.append("__EMPTY__")
            else:
                encoded.append(c)

        name = JCS.canonicalize(encoded)
        return str(uuid.uuid5(self.namespace, name))


class HashUtils:
    """SHA-256 hashing utilities for fingerprints and content hashing."""

    @staticmethod
    def sha256_hex(data: bytes) -> str:
        """Compute SHA-256 hash, return lowercase hex string."""
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def sha256_string(s: str) -> str:
        """Compute SHA-256 of UTF-8 encoded string."""
        return HashUtils.sha256_hex(s.encode('utf-8'))


class TimestampUtils:
    """
    Store timestamps as canonical UTC ISO-8601 strings with milliseconds:
    YYYY-MM-DDTHH:MM:SS.sssZ
    """
    ISO_UTC_MILLIS = "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]"

    @staticmethod
    def now_utc() -> str:
        return pendulum.now("UTC").format(TimestampUtils.ISO_UTC_MILLIS)

    @staticmethod
    def normalize_to_utc(timestamp: Any, source_tz: str | None = None) -> str | None:
        if timestamp is None:
            return None

        try:
            if isinstance(timestamp, (int, float)):
                dt = pendulum.from_timestamp(timestamp, tz="UTC")
                return dt.format(TimestampUtils.ISO_UTC_MILLIS)

            if isinstance(timestamp, datetime):
                dt = pendulum.instance(timestamp)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pendulum.timezone(source_tz or "UTC"))
                return dt.in_timezone("UTC").format(TimestampUtils.ISO_UTC_MILLIS)

            if isinstance(timestamp, str):
                s = timestamp.strip()
                dt = pendulum.parse(s, tz=source_tz or "UTC", strict=False)
                return dt.in_timezone("UTC").format(TimestampUtils.ISO_UTC_MILLIS)

            return None
        except Exception:
            return None

    @staticmethod
    def parse_iso(iso_string: str) -> datetime | None:
        try:
            return pendulum.parse(iso_string, strict=False)
        except Exception:
            return None

    @staticmethod
    def compare(ts1: str | None, ts2: str | None) -> int:
        """Compare two UTC ISO strings. Returns -1, 0, or 1."""
        if ts1 is None and ts2 is None:
            return 0
        if ts1 is None:
            return 1
        if ts2 is None:
            return -1
        return -1 if ts1 < ts2 else (1 if ts1 > ts2 else 0)

    @staticmethod
    def min_time(ts1: str | None, ts2: str | None) -> str | None:
        """Return the earlier of two timestamps, handling NULLs."""
        if ts1 is None:
            return ts2
        if ts2 is None:
            return ts1
        return ts1 if ts1 < ts2 else ts2


# ===| ENUMS |===

class NodeType(StrEnum):
    """Graph node types."""
    ENTITY = "Entity"
    ASSERTION = "Assertion"
    PREDICATE = "Predicate"
    TIME_INTERVAL = "TimeInterval"
    VALUE = "Value"
    MESSAGE = "Message"
    RETRACTION = "Retraction"
    CONFLICT_GROUP = "ConflictGroup"
    LEXICON_TERM = "LexiconTerm"
    TEMPORAL_WINDOW = "TemporalWindow"
    SEMANTIC_CLUSTER = "SemanticCluster"
    TEMPORAL_PROFILE = "TemporalProfile"


class EdgeType(StrEnum):
    """Graph edge types."""
    # Core semantic edges
    HAS_SUBJECT = "HAS_SUBJECT"
    HAS_PREDICATE = "HAS_PREDICATE"
    HAS_OBJECT = "HAS_OBJECT"
    # Temporal edges
    VALID_IN = "VALID_IN"
    VALID_UNTIL_HINT = "VALID_UNTIL_HINT"
    # Message anchoring
    ASSERTED_IN = "ASSERTED_IN"
    # Lifecycle edges
    SUPERSEDES = "SUPERSEDES"
    RETRACTED_BY = "RETRACTED_BY"
    RETRACTS = "RETRACTS"
    NEGATED_BY = "NEGATED_BY"
    NEGATES = "NEGATES"
    # Conflict edges
    HAS_CONFLICT_MEMBER = "HAS_CONFLICT_MEMBER"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    # Lexicon provenance
    DERIVED_FROM_LEXICON = "DERIVED_FROM_LEXICON"
    # Temporal profile edges
    HAS_WINDOW = "HAS_WINDOW"
    HAS_CLUSTER = "HAS_CLUSTER"
    CLUSTER_CONTAINS = "CLUSTER_CONTAINS"
    WINDOW_INCLUDES = "WINDOW_INCLUDES"
    EVOLVES_TO = "EVOLVES_TO"


class AssertionStatus(StrEnum):
    """Assertion lifecycle status."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    NEGATED = "negated"
    CONFLICTED = "conflicted"
    INELIGIBLE = "ineligible"


class WindowGranularity(StrEnum):
    """Temporal window granularity options."""
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class SemanticCategory(StrEnum):
    """Semantic cluster categories."""
    PROJECT = "project"
    INTEREST = "interest"
    PROBLEM = "problem"
    TASK = "task"
    DESIRE = "desire"
    PLAN = "plan"
    UNCLASSIFIED = "unclassified"


class ClassificationTier(IntEnum):
    """Classification tier levels."""
    STRONG_RULES = 1
    PREDICATE_CLUSTERING = 2
    LLM_CONSENSUS = 3
    DEFAULT = 0


# ===| CONFIGURATION |===

@dataclass
class TemporalProfileConfig:
    """Configuration for temporal profile generation."""
    # Strategy selection
    classification_strategy: str = "hybrid"  # rules_only, hybrid, llm_only

    # Window partitioning
    window_granularities: List[str] = field(default_factory=lambda: ["month", "quarter", "year"])
    default_granularity: str = "month"
    window_min_assertions: int = 3

    # Tier 1: Strong deterministic rules
    strong_rule_keywords: Dict[str, List[str]] = field(default_factory=lambda: {
        "project": ["work", "build", "develop", "create", "start", "implement", "design", "launch"],
        "interest": ["like", "enjoy", "love", "prefer", "interest", "curious", "fascinate", "hobby"],
        "problem": ["issue", "problem", "bug", "error", "fail", "broken", "stuck", "trouble", "difficult"],
        "task": ["todo", "need to", "must", "should", "plan to", "going to", "will", "deadline"],
        "desire": ["want", "wish", "hope", "dream", "aspire", "goal", "aim"],
        "plan": ["schedule", "planning", "intend", "strategy", "roadmap", "milestone"],
    })

    # Tier 2: Predicate clustering
    enable_predicate_clustering: bool = True
    predicate_cluster_model: str = "all-MiniLM-L6-v2"
    predicate_cluster_seed: int = 42
    predicate_cluster_threshold: float = 0.75
    predicate_cluster_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        "project": ["works_on", "develops", "maintains", "contributes_to"],
        "interest": ["interested_in", "enjoys", "follows", "watches"],
        "problem": ["struggles_with", "blocked_by", "debugging", "investigating"],
    })

    # Tier 3: LLM consensus fallback
    enable_llm_cluster_fallback: bool = False
    llm_cluster_model: Optional[str] = None
    llm_cluster_api_base: Optional[str] = None
    llm_cluster_runs: int = 5
    llm_cluster_consensus_threshold: int = 4
    llm_cluster_min_salience: float = 0.5
    llm_cluster_temperature: float = 0.0
    llm_cluster_seed: int = 42

    # Salience computation
    window_salience_decay: float = 0.9
    window_salience_frequency_weight: float = 0.4
    window_salience_recency_weight: float = 0.3
    window_salience_confidence_weight: float = 0.3

    # Output control
    top_k_per_cluster: int = 10
    include_unclassified_cluster: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "TemporalProfileConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        config_data = data.get('temporal_profile_config', data)
        return cls(
            classification_strategy=config_data.get('classification_strategy', cls.classification_strategy),
            window_granularities=config_data.get('window_granularities', ["month", "quarter", "year"]),
            default_granularity=config_data.get('default_granularity', cls.default_granularity),
            window_min_assertions=config_data.get('window_min_assertions', cls.window_min_assertions),
            strong_rule_keywords=config_data.get('strong_rule_keywords', cls().strong_rule_keywords),
            enable_predicate_clustering=config_data.get('enable_predicate_clustering', cls.enable_predicate_clustering),
            predicate_cluster_model=config_data.get('predicate_cluster_model', cls.predicate_cluster_model),
            predicate_cluster_seed=config_data.get('predicate_cluster_seed', cls.predicate_cluster_seed),
            predicate_cluster_threshold=config_data.get('predicate_cluster_threshold', cls.predicate_cluster_threshold),
            predicate_cluster_categories=config_data.get('predicate_cluster_categories', cls().predicate_cluster_categories),
            enable_llm_cluster_fallback=config_data.get('enable_llm_cluster_fallback', cls.enable_llm_cluster_fallback),
            llm_cluster_model=config_data.get('llm_cluster_model'),
            llm_cluster_api_base=config_data.get('llm_cluster_api_base'),
            llm_cluster_runs=config_data.get('llm_cluster_runs', cls.llm_cluster_runs),
            llm_cluster_consensus_threshold=config_data.get('llm_cluster_consensus_threshold', cls.llm_cluster_consensus_threshold),
            llm_cluster_min_salience=config_data.get('llm_cluster_min_salience', cls.llm_cluster_min_salience),
            llm_cluster_temperature=config_data.get('llm_cluster_temperature', cls.llm_cluster_temperature),
            llm_cluster_seed=config_data.get('llm_cluster_seed', cls.llm_cluster_seed),
            window_salience_decay=config_data.get('window_salience_decay', cls.window_salience_decay),
            window_salience_frequency_weight=config_data.get('window_salience_frequency_weight', cls.window_salience_frequency_weight),
            window_salience_recency_weight=config_data.get('window_salience_recency_weight', cls.window_salience_recency_weight),
            window_salience_confidence_weight=config_data.get('window_salience_confidence_weight', cls.window_salience_confidence_weight),
            top_k_per_cluster=config_data.get('top_k_per_cluster', cls.top_k_per_cluster),
            include_unclassified_cluster=config_data.get('include_unclassified_cluster', cls.include_unclassified_cluster),
        )


@dataclass
class Stage6Config:
    """Configuration for Stage 6 pipeline."""
    output_file_path: Path
    id_namespace: str = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"  # Default namespace UUID

    # Core graph generation
    include_message_nodes: bool = True
    include_lexicon_nodes: bool = False
    include_detection_tier_metadata: bool = True
    conflict_pairwise_max_n: int = 25
    include_inverse_lifecycle_edges: bool = False

    # Temporal profile generation
    enable_temporal_profiles: bool = False
    temporal_profile_config_path: Optional[Path] = None
    temporal_profile_focal_entities: Optional[List[str]] = None
    temporal_profile_min_salience: float = 0.8
    include_window_assertion_edges: bool = False
    include_cluster_evolution_edges: bool = False


# ===| DATA CLASSES |===

@dataclass
class GraphNode:
    """Represents a graph node."""
    node_id: str
    node_type: str
    source_id: str
    label: str
    metadata_json: str


@dataclass
class GraphEdge:
    """Represents a graph edge."""
    edge_id: str
    edge_type: str
    src_node_id: str
    dst_node_id: str
    metadata_json: Optional[str] = None


@dataclass
class TemporalWindowRecord:
    """Represents a temporal window for profile generation."""
    entity_id: str
    granularity: str
    window_start_utc: str
    window_end_utc: str
    assertion_ids: List[str]
    window_key: str


@dataclass
class ClassificationRecord:
    """Represents a classification result."""
    category: str
    tier: int
    confidence: float
    evidence: List[str]
    raw_classification_json: str


# ===| DATABASE |===

class Database:
    """Interface for the SQLite database."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        logger.info(f"Opening database {str(database_path)}")
        self.connection = sqlite3.connect(str(database_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._in_transaction = False

    def begin(self):
        """Begin transaction."""
        self._in_transaction = True
        self.connection.execute("BEGIN TRANSACTION")

    def rollback(self):
        """Rollback current transaction."""
        self.connection.rollback()
        self._in_transaction = False

    def commit(self):
        """Commit transaction."""
        if not self._in_transaction:
            raise RuntimeError("Cannot commit if not in transaction")
        self.connection.commit()
        self._in_transaction = False

    def close(self):
        """Close database connection."""
        self.connection.close()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute SQL statement."""
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """Execute SQL statement with multiple parameter sets."""
        return self.connection.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute and fetch one row."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute and fetch all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        result = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return result is not None


class Stage6Database(Database):
    """Stage 6 specific database operations."""

    REQUIRED_TABLES = [
        "messages",
        "entities",
        "predicates",
        "assertions",
        "assertion_temporalized",
        "retractions",
        "conflict_groups",
        "conflict_members",
    ]

    def check_required_tables(self):
        """Verify that all required tables from previous stages exist."""
        missing = []
        for table in self.REQUIRED_TABLES:
            if not self.table_exists(table):
                missing.append(table)

        if missing:
            raise RuntimeError(
                f"Missing required tables from previous stages: {', '.join(missing)}"
            )
        logger.info("All required tables present")

    def initialize_stage6_schema(self):
        """Create or recreate Stage 6 tables."""
        logger.info("Initializing Stage 6 schema")

        # Drop existing tables if they exist (overwrite mode)
        self.execute("DROP TABLE IF EXISTS graph_edges")
        self.execute("DROP TABLE IF EXISTS graph_nodes")

        # Create graph_nodes table
        self.execute("""
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                label TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)

        # Create indices for graph_nodes
        self.execute("CREATE INDEX idx_graph_nodes_type ON graph_nodes(node_type, node_id)")
        self.execute("CREATE INDEX idx_graph_nodes_source ON graph_nodes(node_type, source_id)")

        # Create graph_edges table
        self.execute("""
            CREATE TABLE graph_edges (
                edge_id TEXT PRIMARY KEY,
                edge_type TEXT NOT NULL,
                src_node_id TEXT NOT NULL,
                dst_node_id TEXT NOT NULL,
                metadata_json TEXT,
                FOREIGN KEY (src_node_id) REFERENCES graph_nodes(node_id),
                FOREIGN KEY (dst_node_id) REFERENCES graph_nodes(node_id)
            )
        """)

        # Create indices for graph_edges
        self.execute("CREATE INDEX idx_graph_edges_src ON graph_edges(src_node_id)")
        self.execute("CREATE INDEX idx_graph_edges_dst ON graph_edges(dst_node_id)")
        self.execute("CREATE INDEX idx_graph_edges_type ON graph_edges(edge_type, edge_id)")

        logger.info("Stage 6 schema initialized")

    def insert_nodes(self, nodes: List[GraphNode]):
        """Insert multiple nodes."""
        if not nodes:
            return
        self.executemany(
            "INSERT INTO graph_nodes (node_id, node_type, source_id, label, metadata_json) VALUES (?, ?, ?, ?, ?)",
            [(n.node_id, n.node_type, n.source_id, n.label, n.metadata_json) for n in nodes]
        )

    def insert_edges(self, edges: List[GraphEdge]):
        """Insert multiple edges."""
        if not edges:
            return
        self.executemany(
            "INSERT INTO graph_edges (edge_id, edge_type, src_node_id, dst_node_id, metadata_json) VALUES (?, ?, ?, ?, ?)",
            [(e.edge_id, e.edge_type, e.src_node_id, e.dst_node_id, e.metadata_json) for e in edges]
        )

    # === Data Stream Methods ===

    def stream_active_entities(self) -> Iterator[sqlite3.Row]:
        """Stream active entities in deterministic order."""
        return iter(self.fetchall(
            "SELECT * FROM entities WHERE status='active' ORDER BY entity_id ASC"
        ))

    def stream_predicates(self) -> Iterator[sqlite3.Row]:
        """Stream predicates in deterministic order."""
        return iter(self.fetchall(
            "SELECT * FROM predicates ORDER BY predicate_id ASC"
        ))

    def stream_messages(self) -> Iterator[sqlite3.Row]:
        """Stream messages in deterministic order."""
        return iter(self.fetchall(
            "SELECT * FROM messages ORDER BY conversation_id ASC, order_index ASC, message_id ASC"
        ))

    def stream_assertions_joined(self) -> Iterator[sqlite3.Row]:
        """Stream assertions joined with temporalized data and messages."""
        return iter(self.fetchall("""
            SELECT 
                A.*,
                T.valid_time_type,
                T.valid_from_utc,
                T.valid_to_utc,
                T.valid_until_hint_utc,
                T.status,
                T.temporal_superseded_by_assertion_id,
                T.retracted_by_retraction_id,
                T.negated_by_assertion_id,
                T.rule_id_applied,
                T.raw_temporalize_json,
                M.conversation_id,
                M.order_index
            FROM assertions A
            JOIN assertion_temporalized T ON A.assertion_id = T.assertion_id
            JOIN messages M ON A.message_id = M.message_id
            ORDER BY M.conversation_id ASC, M.order_index ASC, A.message_id ASC, A.assertion_id ASC
        """))

    def stream_retractions(self) -> Iterator[sqlite3.Row]:
        """Stream retractions in deterministic order."""
        return iter(self.fetchall("""
            SELECT * FROM retractions 
            ORDER BY retraction_message_id ASC, 
                     CASE WHEN char_start IS NULL THEN 1 ELSE 0 END,
                     char_start ASC,
                     retraction_id ASC
        """))

    def stream_conflict_groups(self) -> Iterator[sqlite3.Row]:
        """Stream conflict groups in deterministic order."""
        return iter(self.fetchall(
            "SELECT * FROM conflict_groups ORDER BY conflict_type ASC, conflict_group_id ASC"
        ))

    def stream_conflict_members(self) -> Iterator[sqlite3.Row]:
        """Stream conflict members in deterministic order."""
        return iter(self.fetchall(
            "SELECT * FROM conflict_members ORDER BY conflict_group_id ASC, assertion_id ASC"
        ))

    def stream_lexicon_terms(self) -> Iterator[sqlite3.Row]:
        """Stream lexicon terms if table exists."""
        if not self.table_exists("lexicon_terms"):
            return iter([])
        return iter(self.fetchall(
            "SELECT * FROM lexicon_terms ORDER BY build_id ASC, term_id ASC"
        ))

    def get_entity_by_id(self, entity_id: str) -> Optional[sqlite3.Row]:
        """Get entity by ID."""
        return self.fetchone("SELECT * FROM entities WHERE entity_id = ?", (entity_id,))

    def get_predicate_by_id(self, predicate_id: str) -> Optional[sqlite3.Row]:
        """Get predicate by ID."""
        return self.fetchone("SELECT * FROM predicates WHERE predicate_id = ?", (predicate_id,))

    def get_best_detector_tier_for_entity(self, entity_id: str) -> Optional[int]:
        """Get the best (lowest) detector tier for an entity."""
        if not self.table_exists("entity_mentions"):
            return None
        result = self.fetchone("""
            SELECT MIN(
                CASE 
                    WHEN detector LIKE 'PATTERN:%' THEN 1
                    WHEN detector LIKE 'LEXICON:%' THEN 2
                    WHEN detector LIKE 'NER:%' THEN 3
                    ELSE 4
                END
            ) as best_tier
            FROM entity_mentions 
            WHERE entity_id = ?
        """, (entity_id,))
        return result['best_tier'] if result and result['best_tier'] else None

    def get_conflict_group_member_count(self, conflict_group_id: str) -> int:
        """Get member count for a conflict group."""
        result = self.fetchone(
            "SELECT COUNT(*) as cnt FROM conflict_members WHERE conflict_group_id = ?",
            (conflict_group_id,)
        )
        return result['cnt'] if result else 0


# ===| GRAPH NODE BUILDER |===

class GraphNodeBuilder:
    """Builds graph nodes with deterministic IDs and canonical metadata."""

    def __init__(self, id_generator: IDGenerator, db: Stage6Database):
        self.id_generator = id_generator
        self.db = db
        self.schema_version = "1.0"

    def _compute_node_id(self, node_type: str, source_id: str) -> str:
        """Compute node ID using uuid5."""
        return self.id_generator.generate(["node", node_type, source_id])

    def _make_metadata(self, data: Dict[str, Any]) -> str:
        """Create JCS-canonical metadata JSON with schema version."""
        data["schema_version"] = self.schema_version
        return JCS.canonicalize(data)

    def build_entity_node(self, entity: sqlite3.Row) -> GraphNode:
        """Build an Entity node."""
        source_id = entity['entity_id']
        node_id = self._compute_node_id(NodeType.ENTITY, source_id)

        label = entity['canonical_name'] or entity['entity_key']

        # Get best detector tier
        best_tier = self.db.get_best_detector_tier_for_entity(entity['entity_id'])

        metadata = {
            "entity_id": entity['entity_id'],
            "entity_type": entity['entity_type'],
            "entity_key": entity['entity_key'],
            "aliases_json": entity['aliases_json'] if 'aliases_json' in entity.keys() else None,
            "first_seen_at_utc": entity['first_seen_at_utc'] if 'first_seen_at_utc' in entity.keys() else None,
            "last_seen_at_utc": entity['last_seen_at_utc'] if 'last_seen_at_utc' in entity.keys() else None,
            "mention_count": entity['mention_count'] if 'mention_count' in entity.keys() else None,
            "conversation_count": entity['conversation_count'] if 'conversation_count' in entity.keys() else None,
            "salience_score": entity['salience_score'] if 'salience_score' in entity.keys() else None,
        }
        if best_tier is not None:
            metadata["best_detector_tier"] = best_tier

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.ENTITY,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_predicate_node(self, predicate: sqlite3.Row) -> GraphNode:
        """Build a Predicate node."""
        source_id = predicate['predicate_id']
        node_id = self._compute_node_id(NodeType.PREDICATE, source_id)

        label = predicate['canonical_label']

        metadata = {
            "predicate_id": predicate['predicate_id'],
            "canonical_label": predicate['canonical_label'],
            "inverse_label": predicate['inverse_label'] if 'inverse_label' in predicate.keys() else None,
            "category": predicate['category'] if 'category' in predicate.keys() else None,
            "arity": predicate['arity'] if 'arity' in predicate.keys() else None,
            "value_type_constraint": predicate['value_type_constraint'] if 'value_type_constraint' in predicate.keys() else None,
            "assertion_count": predicate['assertion_count'] if 'assertion_count' in predicate.keys() else None,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.PREDICATE,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_message_node(self, message: sqlite3.Row) -> GraphNode:
        """Build a Message node."""
        source_id = message['message_id']
        node_id = self._compute_node_id(NodeType.MESSAGE, source_id)

        # Label: conversation_id + "#" + order_index
        label = f"{message['conversation_id']}#{message['order_index']}"

        metadata = {
            "message_id": message['message_id'],
            "conversation_id": message['conversation_id'],
            "role": message['role'] if 'role' in message.keys() else None,
            "created_at_utc": message['created_at_utc'] if 'created_at_utc' in message.keys() else None,
            "timestamp_quality": message['timestamp_quality'] if 'timestamp_quality' in message.keys() else None,
            "parent_id": message['parent_id'] if 'parent_id' in message.keys() else None,
            "tree_path": message['tree_path'] if 'tree_path' in message.keys() else None,
            "order_index": message['order_index'],
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.MESSAGE,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_assertion_node(self, assertion: sqlite3.Row) -> GraphNode:
        """Build an Assertion node from joined assertion + temporalized data."""
        source_id = assertion['assertion_id']
        node_id = self._compute_node_id(NodeType.ASSERTION, source_id)

        # Build label: subject_entity_id predicate_id object_signature
        label = f"{assertion['subject_entity_id']} {assertion['predicate_id']} {assertion['object_signature']}"

        # Compute detector reliability label
        subj_tier = assertion['subject_detection_tier'] if 'subject_detection_tier' in assertion.keys() else None
        obj_tier = assertion['object_detection_tier'] if 'object_detection_tier' in assertion.keys() else None
        min_tier = min(t for t in [subj_tier, obj_tier, 4] if t is not None)
        tier_labels = {1: "high", 2: "medium-high", 3: "medium", 4: "low"}
        detector_reliability_label = tier_labels.get(min_tier, "unknown")

        # Compute hashes for provenance
        raw_assertion_json = assertion['raw_assertion_json'] if 'raw_assertion_json' in assertion.keys() else None
        raw_temporalize_json = assertion['raw_temporalize_json'] if 'raw_temporalize_json' in assertion.keys() else None

        metadata = {
            # Assertion core
            "assertion_id": assertion['assertion_id'],
            "message_id": assertion['message_id'],
            "subject_entity_id": assertion['subject_entity_id'],
            "subject_detection_tier": subj_tier,
            "predicate_id": assertion['predicate_id'],
            "object_entity_id": assertion['object_entity_id'] if 'object_entity_id' in assertion.keys() else None,
            "object_detection_tier": obj_tier,
            "object_value_type": assertion['object_value_type'] if 'object_value_type' in assertion.keys() else None,
            "object_value": assertion['object_value'] if 'object_value' in assertion.keys() else None,
            "object_signature": assertion['object_signature'],
            "modality": assertion['modality'] if 'modality' in assertion.keys() else None,
            "polarity": assertion['polarity'] if 'polarity' in assertion.keys() else None,
            "asserted_role": assertion['asserted_role'] if 'asserted_role' in assertion.keys() else None,
            "asserted_at_utc": assertion['asserted_at_utc'] if 'asserted_at_utc' in assertion.keys() else None,
            "confidence_extraction": assertion['confidence_extraction'] if 'confidence_extraction' in assertion.keys() else None,
            "confidence_grounding": assertion['confidence_grounding'] if 'confidence_grounding' in assertion.keys() else None,
            "confidence_final": assertion['confidence_final'] if 'confidence_final' in assertion.keys() else None,
            "has_user_corroboration": assertion['has_user_corroboration'] if 'has_user_corroboration' in assertion.keys() else None,
            # Temporalized
            "status": assertion['status'],
            "valid_time_type": assertion['valid_time_type'],
            "valid_from_utc": assertion['valid_from_utc'],
            "valid_to_utc": assertion['valid_to_utc'],
            "valid_until_hint_utc": assertion['valid_until_hint_utc'],
            "temporal_superseded_by_assertion_id": assertion['temporal_superseded_by_assertion_id'],
            "retracted_by_retraction_id": assertion['retracted_by_retraction_id'],
            "negated_by_assertion_id": assertion['negated_by_assertion_id'],
            "rule_id_applied": assertion['rule_id_applied'],
            # Provenance
            "raw_assertion_json_hash": HashUtils.sha256_string(raw_assertion_json) if raw_assertion_json else None,
            "raw_temporalize_json_hash": HashUtils.sha256_string(raw_temporalize_json) if raw_temporalize_json else None,
            # Derived
            "detector_reliability_label": detector_reliability_label,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.ASSERTION,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_value_node(self, object_value_type: str, object_value: str) -> GraphNode:
        """Build a Value node for literal objects."""
        # source_id is hash of type and value
        source_id = HashUtils.sha256_string(JCS.canonicalize([object_value_type, object_value]))
        node_id = self._compute_node_id(NodeType.VALUE, source_id)

        # Label: truncated string or canonical value
        if object_value_type == 'string':
            label = object_value[:100] if len(object_value) > 100 else object_value
        else:
            label = object_value

        metadata = {
            "object_value_type": object_value_type,
            "object_value": object_value,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.VALUE,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_time_interval_node(
        self,
        valid_from_utc: Optional[str],
        valid_to_utc: Optional[str],
        is_hint_only: bool = False
    ) -> GraphNode:
        """Build a TimeInterval node."""
        # source_id is hash of the interval
        source_id = HashUtils.sha256_string(JCS.canonicalize([valid_from_utc, valid_to_utc]))
        node_id = self._compute_node_id(NodeType.TIME_INTERVAL, source_id)

        # Build label
        if is_hint_only:
            label = f"… until {valid_to_utc}"
        else:
            from_str = valid_from_utc if valid_from_utc else "∅"
            to_str = valid_to_utc if valid_to_utc else "∞"
            label = f"{from_str} … {to_str}"

        metadata = {
            "valid_from_utc": valid_from_utc,
            "valid_to_utc": valid_to_utc,
            "is_hint_only": is_hint_only,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.TIME_INTERVAL,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_retraction_node(self, retraction: sqlite3.Row) -> GraphNode:
        """Build a Retraction node."""
        source_id = retraction['retraction_id']
        node_id = self._compute_node_id(NodeType.RETRACTION, source_id)

        label = f"retraction:{retraction['retraction_type']}"

        metadata = {
            "retraction_id": retraction['retraction_id'],
            "retraction_message_id": retraction['retraction_message_id'],
            "target_assertion_id": retraction['target_assertion_id'] if 'target_assertion_id' in retraction.keys() else None,
            "target_fact_key": retraction['target_fact_key'] if 'target_fact_key' in retraction.keys() else None,
            "retraction_type": retraction['retraction_type'],
            "replacement_assertion_id": retraction['replacement_assertion_id'] if 'replacement_assertion_id' in retraction.keys() else None,
            "confidence": retraction['confidence'] if 'confidence' in retraction.keys() else None,
            "char_start": retraction['char_start'] if 'char_start' in retraction.keys() else None,
            "char_end": retraction['char_end'] if 'char_end' in retraction.keys() else None,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.RETRACTION,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_conflict_group_node(self, conflict_group: sqlite3.Row) -> GraphNode:
        """Build a ConflictGroup node."""
        source_id = conflict_group['conflict_group_id']
        node_id = self._compute_node_id(NodeType.CONFLICT_GROUP, source_id)

        label = conflict_group['conflict_type']

        # Get member count
        member_count = self.db.get_conflict_group_member_count(source_id)

        raw_conflict_json = conflict_group['raw_conflict_json'] if 'raw_conflict_json' in conflict_group.keys() else None

        metadata = {
            "conflict_group_id": conflict_group['conflict_group_id'],
            "conflict_type": conflict_group['conflict_type'],
            "conflict_key": conflict_group['conflict_key'],
            "detected_at_utc": conflict_group['detected_at_utc'] if 'detected_at_utc' in conflict_group.keys() else None,
            "member_count": member_count,
            "raw_conflict_json_hash": HashUtils.sha256_string(raw_conflict_json) if raw_conflict_json else None,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.CONFLICT_GROUP,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_lexicon_term_node(self, term: sqlite3.Row) -> GraphNode:
        """Build a LexiconTerm node."""
        source_id = term['term_id']
        node_id = self._compute_node_id(NodeType.LEXICON_TERM, source_id)

        label = term['canonical_surface']

        metadata = {
            "term_id": term['term_id'],
            "build_id": term['build_id'] if 'build_id' in term.keys() else None,
            "term_key": term['term_key'] if 'term_key' in term.keys() else None,
            "canonical_surface": term['canonical_surface'],
            "aliases_json": term['aliases_json'] if 'aliases_json' in term.keys() else None,
            "score": term['score'] if 'score' in term.keys() else None,
            "entity_type_hint": term['entity_type_hint'] if 'entity_type_hint' in term.keys() else None,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.LEXICON_TERM,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_temporal_profile_node(
        self,
        entity_id: str,
        granularity: str,
        window_count: int,
        total_assertions: int,
        date_range_start: Optional[str],
        date_range_end: Optional[str],
        entity_name: str
    ) -> GraphNode:
        """Build a TemporalProfile node."""
        profile_version = "1.0"
        source_id = HashUtils.sha256_string(JCS.canonicalize([entity_id, granularity, profile_version]))
        node_id = self._compute_node_id(NodeType.TEMPORAL_PROFILE, source_id)

        label = f"{entity_name} ({granularity} profile)"

        metadata = {
            "entity_id": entity_id,
            "granularity": granularity,
            "window_count": window_count,
            "total_assertions": total_assertions,
            "date_range_start": date_range_start,
            "date_range_end": date_range_end,
            "profile_version": profile_version,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.TEMPORAL_PROFILE,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_temporal_window_node(
        self,
        entity_id: str,
        granularity: str,
        window_start_utc: str,
        window_end_utc: str,
        assertion_count: int,
        cluster_count: int,
        top_entities: List[str]
    ) -> GraphNode:
        """Build a TemporalWindow node."""
        source_id = HashUtils.sha256_string(JCS.canonicalize([entity_id, window_start_utc, window_end_utc, granularity]))
        node_id = self._compute_node_id(NodeType.TEMPORAL_WINDOW, source_id)

        # Format label based on granularity
        label = self._format_window_label(window_start_utc, granularity)

        metadata = {
            "entity_id": entity_id,
            "granularity": granularity,
            "window_start_utc": window_start_utc,
            "window_end_utc": window_end_utc,
            "assertion_count": assertion_count,
            "cluster_count": cluster_count,
            "top_entities": top_entities,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.TEMPORAL_WINDOW,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    def build_semantic_cluster_node(
        self,
        window_source_id: str,
        category: str,
        classification_tier: int,
        window_label: str,
        member_count: int,
        avg_confidence: float,
        avg_salience: float,
        tier_distribution: Dict[str, int]
    ) -> GraphNode:
        """Build a SemanticCluster node."""
        source_id = HashUtils.sha256_string(JCS.canonicalize([window_source_id, category, classification_tier]))
        node_id = self._compute_node_id(NodeType.SEMANTIC_CLUSTER, source_id)

        label = f"{category} ({window_label})"

        metadata = {
            "category": category,
            "window_key": window_source_id,
            "member_count": member_count,
            "avg_confidence": avg_confidence,
            "avg_salience": avg_salience,
            "classification_tier_distribution": tier_distribution,
        }

        return GraphNode(
            node_id=node_id,
            node_type=NodeType.SEMANTIC_CLUSTER,
            source_id=source_id,
            label=label,
            metadata_json=self._make_metadata(metadata)
        )

    @staticmethod
    def _format_window_label(window_start_utc: str, granularity: str) -> str:
        """Format window label based on granularity."""
        try:
            dt = pendulum.parse(window_start_utc, strict=False)
            if granularity == "month":
                return dt.format("YYYY-MM")
            elif granularity == "quarter":
                quarter = (dt.month - 1) // 3 + 1
                return f"{dt.year}-Q{quarter}"
            elif granularity == "year":
                return str(dt.year)
            else:
                return window_start_utc
        except Exception:
            return window_start_utc


# ===| GRAPH EDGE BUILDER |===

class GraphEdgeBuilder:
    """Builds graph edges with deterministic IDs."""

    def __init__(self, id_generator: IDGenerator):
        self.id_generator = id_generator

    def _compute_edge_id(self, edge_type: str, src_node_id: str, dst_node_id: str) -> str:
        """Compute edge ID using uuid5."""
        return self.id_generator.generate(["edge", edge_type, src_node_id, dst_node_id])

    def _make_metadata(self, data: Optional[Dict[str, Any]]) -> Optional[str]:
        """Create JCS-canonical metadata JSON."""
        if data is None:
            return None
        return JCS.canonicalize(data)

    def build_edge(
        self,
        edge_type: str,
        src_node_id: str,
        dst_node_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> GraphEdge:
        """Build a graph edge."""
        edge_id = self._compute_edge_id(edge_type, src_node_id, dst_node_id)
        return GraphEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            src_node_id=src_node_id,
            dst_node_id=dst_node_id,
            metadata_json=self._make_metadata(metadata)
        )


# ===| TEMPORAL PROFILE GENERATOR |===

class TemporalProfileGenerator:
    """Generates temporal profiles with semantic clustering."""

    def __init__(
        self,
        config: TemporalProfileConfig,
        db: Stage6Database,
        node_builder: GraphNodeBuilder,
        edge_builder: GraphEdgeBuilder,
        id_generator: IDGenerator
    ):
        self.config = config
        self.db = db
        self.node_builder = node_builder
        self.edge_builder = edge_builder
        self.id_generator = id_generator

        # Caches
        self.predicate_embeddings: Dict[str, List[float]] = {}
        self.category_centroids: Dict[str, List[float]] = {}

        # Statistics
        self.stats = {
            "profiles_generated": 0,
            "windows_generated": 0,
            "clusters_by_category": {},
            "classifications_by_tier": {0: 0, 1: 0, 2: 0, 3: 0},
            "llm_classification_calls": 0,
        }

    def generate(self, focal_entities: List[sqlite3.Row]) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Generate temporal profile nodes and edges."""
        if len(self.config.window_granularities) == 0:
            logger.error("No window granularities configured -> no nodes or edges generated.")
            return [], []

        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        for entity in focal_entities:
            entity_id = entity['entity_id']
            entity_name = entity['canonical_name'] or entity['entity_key']

            for granularity in self.config.window_granularities:
                profile_nodes, profile_edges = self._generate_profile_for_entity(
                    entity_id, entity_name, granularity
                )
                nodes.extend(profile_nodes)
                edges.extend(profile_edges)

        return nodes, edges

    def _generate_profile_for_entity(
        self,
        entity_id: str,
        entity_name: str,
        granularity: str
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Generate profile for a single entity and granularity."""
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # Phase T1: Time window partitioning
        windows = self._partition_time_windows(entity_id, granularity)

        if not windows:
            return nodes, edges

        # Phase T2: Window salience recomputation
        window_saliences = self._compute_window_saliences(windows)

        # Phase T3: Semantic clustering
        window_classifications = self._classify_entities_in_windows(windows, window_saliences)

        # Phase T4: Profile node and edge generation
        profile_nodes, profile_edges = self._build_profile_nodes_and_edges(
            entity_id, entity_name, granularity, windows, window_saliences, window_classifications
        )

        nodes.extend(profile_nodes)
        edges.extend(profile_edges)

        return nodes, edges

    def _partition_time_windows(
        self,
        entity_id: str,
        granularity: str
    ) -> List[TemporalWindowRecord]:
        """Phase T1: Partition assertions into time windows."""
        # Get relevant assertions
        rows = self.db.fetchall("""
            SELECT A.assertion_id, T.valid_from_utc
            FROM assertions A
            JOIN assertion_temporalized T ON A.assertion_id = T.assertion_id
            WHERE A.subject_entity_id = ?
              AND T.valid_from_utc IS NOT NULL
            ORDER BY T.valid_from_utc ASC
        """, (entity_id,))

        if not rows:
            return []

        # Determine time range
        valid_times = [r['valid_from_utc'] for r in rows]
        range_start = min(valid_times)
        range_end = max(valid_times)

        # Generate window boundaries
        boundaries = self._generate_window_boundaries(range_start, range_end, granularity)

        # Assign assertions to windows
        windows: List[TemporalWindowRecord] = []
        for window_start, window_end in boundaries:
            assertion_ids = [
                r['assertion_id'] for r in rows
                if window_start <= r['valid_from_utc'] < window_end
            ]

            if len(assertion_ids) >= self.config.window_min_assertions:
                window_key = JCS.canonicalize([entity_id, granularity, window_start, window_end])
                windows.append(TemporalWindowRecord(
                    entity_id=entity_id,
                    granularity=granularity,
                    window_start_utc=window_start,
                    window_end_utc=window_end,
                    assertion_ids=sorted(assertion_ids),
                    window_key=window_key
                ))

        return windows

    def _generate_window_boundaries(
        self,
        range_start: str,
        range_end: str,
        granularity: str
    ) -> List[Tuple[str, str]]:
        """Generate window boundaries for the given time range."""
        boundaries = []

        try:
            start_dt = pendulum.parse(range_start, strict=False)
            end_dt = pendulum.parse(range_end, strict=False)

            # Align to boundary
            if granularity == "month":
                current = start_dt.start_of("month")
                delta = lambda d: d.add(months=1)
            elif granularity == "quarter":
                quarter_month = ((start_dt.month - 1) // 3) * 3 + 1
                current = start_dt.replace(month=quarter_month, day=1).start_of("day")
                delta = lambda d: d.add(months=3)
            elif granularity == "year":
                current = start_dt.start_of("year")
                delta = lambda d: d.add(years=1)
            else:
                return boundaries

            while current <= end_dt:
                window_start = current.format(TimestampUtils.ISO_UTC_MILLIS)
                current = delta(current)
                window_end = current.format(TimestampUtils.ISO_UTC_MILLIS)
                boundaries.append((window_start, window_end))

        except Exception as e:
            logger.warning(f"Failed to generate window boundaries: {e}")

        return boundaries

    def _compute_window_saliences(
        self,
        windows: List[TemporalWindowRecord]
    ) -> Dict[str, Dict[str, float]]:
        """Phase T2: Compute per-window salience scores for object entities."""
        window_saliences: Dict[str, Dict[str, float]] = {}

        for window in windows:
            entity_assertions: Dict[str, List[Dict]] = {}

            # Get assertions with object entities
            for assertion_id in window.assertion_ids:
                row = self.db.fetchone("""
                    SELECT A.object_entity_id, T.valid_from_utc, A.confidence_final
                    FROM assertions A
                    JOIN assertion_temporalized T ON A.assertion_id = T.assertion_id
                    WHERE A.assertion_id = ? AND A.object_entity_id IS NOT NULL
                """, (assertion_id,))

                if row:
                    obj_id = row['object_entity_id']
                    if obj_id not in entity_assertions:
                        entity_assertions[obj_id] = []
                    entity_assertions[obj_id].append({
                        'valid_from_utc': row['valid_from_utc'],
                        'confidence_final': row['confidence_final'] or 0.5
                    })

            # Compute salience per entity
            saliences: Dict[str, float] = {}
            total_assertions = len(window.assertion_ids)

            for entity_id, assertions in entity_assertions.items():
                # Frequency component
                freq_score = len(assertions) / total_assertions if total_assertions > 0 else 0

                # Recency component (within window)
                sorted_assertions = sorted(assertions, key=lambda a: a['valid_from_utc'], reverse=True)
                recency_scores = [
                    self.config.window_salience_decay ** i
                    for i in range(len(sorted_assertions))
                ]
                recency_score = sum(recency_scores) / len(recency_scores) if recency_scores else 0

                # Confidence component
                confidence_score = sum(a['confidence_final'] for a in assertions) / len(assertions)

                # Composite
                saliences[entity_id] = (
                    freq_score * self.config.window_salience_frequency_weight +
                    recency_score * self.config.window_salience_recency_weight +
                    confidence_score * self.config.window_salience_confidence_weight
                )

            window_saliences[window.window_key] = saliences

        return window_saliences

    def _classify_entities_in_windows(
        self,
        windows: List[TemporalWindowRecord],
        window_saliences: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, ClassificationRecord]]:
        """Phase T3: Semantic clustering with three-tier classification."""
        classifications: Dict[str, Dict[str, ClassificationRecord]] = {}

        for window in windows:
            window_classifications: Dict[str, ClassificationRecord] = {}
            saliences = window_saliences.get(window.window_key, {})

            for entity_id, salience in saliences.items():
                if salience <= 0:
                    continue

                # Get predicates for this entity in window
                predicates = self._get_predicates_for_entity_in_window(
                    window.entity_id, entity_id, window.assertion_ids
                )

                # Three-tier classification
                classification = self._classify_entity(
                    entity_id, predicates, window, salience
                )

                window_classifications[entity_id] = classification
                self.stats["classifications_by_tier"][classification.tier] += 1

            classifications[window.window_key] = window_classifications

        return classifications

    def _get_predicates_for_entity_in_window(
        self,
        focal_entity_id: str,
        target_entity_id: str,
        assertion_ids: List[str]
    ) -> List[str]:
        """Get predicate labels for assertions linking focal entity to target entity."""
        if not assertion_ids:
            return []

        placeholders = ','.join(['?' for _ in assertion_ids])
        rows = self.db.fetchall(f"""
            SELECT DISTINCT P.canonical_label
            FROM assertions A
            JOIN predicates P ON A.predicate_id = P.predicate_id
            WHERE A.assertion_id IN ({placeholders})
              AND A.subject_entity_id = ?
              AND A.object_entity_id = ?
        """, tuple(assertion_ids) + (focal_entity_id, target_entity_id))

        return [r['canonical_label'] for r in rows]

    def _classify_entity(
        self,
        entity_id: str,
        predicates: List[str],
        window: TemporalWindowRecord,
        salience: float
    ) -> ClassificationRecord:
        """Apply three-tier classification to an entity."""
        evidence: List[str] = []
        raw_log: Dict[str, Any] = {"entity_id": entity_id, "predicates": predicates}

        # Tier 1: Strong deterministic rules
        tier1_result = self._apply_tier1_rules(predicates)
        if tier1_result:
            category, matches = tier1_result
            evidence.extend(matches)
            raw_log["tier1_category"] = category
            raw_log["tier1_matches"] = matches
            return ClassificationRecord(
                category=category,
                tier=ClassificationTier.STRONG_RULES,
                confidence=1.0,
                evidence=evidence,
                raw_classification_json=JCS.canonicalize(raw_log)
            )

        # Tier 2: Predicate clustering (if enabled)
        if self.config.enable_predicate_clustering and self.config.classification_strategy in ("hybrid", "llm_only"):
            tier2_result = self._apply_tier2_clustering(predicates)
            if tier2_result:
                category, confidence, matched = tier2_result
                evidence.extend(matched)
                raw_log["tier2_category"] = category
                raw_log["tier2_confidence"] = confidence
                return ClassificationRecord(
                    category=category,
                    tier=ClassificationTier.PREDICATE_CLUSTERING,
                    confidence=confidence,
                    evidence=evidence,
                    raw_classification_json=JCS.canonicalize(raw_log)
                )

        # Tier 3: LLM consensus fallback (if enabled)
        if self.config.enable_llm_cluster_fallback and salience >= self.config.llm_cluster_min_salience:
            tier3_result = self._apply_tier3_llm(entity_id, predicates, window)
            if tier3_result:
                category, confidence = tier3_result
                raw_log["tier3_category"] = category
                raw_log["tier3_confidence"] = confidence
                return ClassificationRecord(
                    category=category,
                    tier=ClassificationTier.LLM_CONSENSUS,
                    confidence=confidence,
                    evidence=evidence,
                    raw_classification_json=JCS.canonicalize(raw_log)
                )

        # Default fallback
        return ClassificationRecord(
            category=SemanticCategory.UNCLASSIFIED,
            tier=ClassificationTier.DEFAULT,
            confidence=0.0,
            evidence=evidence,
            raw_classification_json=JCS.canonicalize(raw_log)
        )

    def _apply_tier1_rules(self, predicates: List[str]) -> Optional[Tuple[str, List[str]]]:
        """Apply Tier 1 strong deterministic rules."""
        category_matches: Dict[str, List[str]] = {}

        for predicate in predicates:
            predicate_lower = predicate.lower().strip()
            for category, keywords in self.config.strong_rule_keywords.items():
                for keyword in keywords:
                    if keyword in predicate_lower:
                        if category not in category_matches:
                            category_matches[category] = []
                        category_matches[category].append(f"{predicate}:{keyword}")
                        break

        if not category_matches:
            return None

        # Find winner
        max_count = max(len(m) for m in category_matches.values())
        winners = [c for c, m in category_matches.items() if len(m) == max_count]

        # Deterministic tie-break: alphabetical
        winner = sorted(winners)[0]
        return winner, category_matches[winner]

    def _apply_tier2_clustering(self, predicates: List[str]) -> Optional[Tuple[str, float, List[str]]]:
        """Apply Tier 2 predicate cluster matching."""
        # This would require an embedding model - simplified stub
        # In production, load model and compute embeddings
        logger.debug("Tier 2 clustering not fully implemented (requires embedding model)")
        return None

    def _apply_tier3_llm(
        self,
        entity_id: str,
        predicates: List[str],
        window: TemporalWindowRecord
    ) -> Optional[Tuple[str, float]]:
        """Apply Tier 3 LLM consensus fallback."""
        if not self.config.llm_cluster_model:
            return None

        # This would make LLM API calls - simplified stub
        logger.debug("Tier 3 LLM classification not fully implemented (requires API calls)")
        self.stats["llm_classification_calls"] += self.config.llm_cluster_runs
        return None

    def _build_profile_nodes_and_edges(
        self,
        entity_id: str,
        entity_name: str,
        granularity: str,
        windows: List[TemporalWindowRecord],
        window_saliences: Dict[str, Dict[str, float]],
        window_classifications: Dict[str, Dict[str, ClassificationRecord]]
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Phase T4: Build profile nodes and edges."""
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        # Get date range
        if windows:
            date_range_start = windows[0].window_start_utc
            date_range_end = windows[-1].window_end_utc
        else:
            date_range_start = None
            date_range_end = None

        total_assertions = sum(len(w.assertion_ids) for w in windows)

        # Create TemporalProfile node
        profile_node = self.node_builder.build_temporal_profile_node(
            entity_id=entity_id,
            granularity=granularity,
            window_count=len(windows),
            total_assertions=total_assertions,
            date_range_start=date_range_start,
            date_range_end=date_range_end,
            entity_name=entity_name
        )
        nodes.append(profile_node)
        self.stats["profiles_generated"] += 1

        prev_cluster_nodes: Dict[str, GraphNode] = {}  # category -> previous window's cluster node

        for window_idx, window in enumerate(windows):
            saliences = window_saliences.get(window.window_key, {})
            classifications = window_classifications.get(window.window_key, {})

            # Get top entities by salience
            sorted_entities = sorted(
                saliences.items(),
                key=lambda x: (-x[1], x[0])  # salience DESC, entity_id ASC
            )
            top_entities = [e[0] for e in sorted_entities[:self.config.top_k_per_cluster]]

            # Group by category for cluster count
            category_entities: Dict[str, List[str]] = {}
            for eid, classification in classifications.items():
                cat = classification.category
                if cat not in category_entities:
                    category_entities[cat] = []
                category_entities[cat].append(eid)

            # Create TemporalWindow node
            window_source_id = HashUtils.sha256_string(JCS.canonicalize([
                entity_id, window.window_start_utc, window.window_end_utc, granularity
            ]))
            window_node = self.node_builder.build_temporal_window_node(
                entity_id=entity_id,
                granularity=granularity,
                window_start_utc=window.window_start_utc,
                window_end_utc=window.window_end_utc,
                assertion_count=len(window.assertion_ids),
                cluster_count=len(category_entities),
                top_entities=top_entities
            )
            nodes.append(window_node)
            self.stats["windows_generated"] += 1

            # HAS_WINDOW edge
            edges.append(self.edge_builder.build_edge(
                EdgeType.HAS_WINDOW,
                profile_node.node_id,
                window_node.node_id,
                {"window_index": window_idx}
            ))

            window_label = self.node_builder._format_window_label(window.window_start_utc, granularity)
            current_cluster_nodes: Dict[str, GraphNode] = {}

            # Create SemanticCluster nodes per category
            for category, entity_ids in category_entities.items():
                if category == SemanticCategory.UNCLASSIFIED and not self.config.include_unclassified_cluster:
                    continue

                # Compute cluster stats
                cluster_classifications = [classifications[eid] for eid in entity_ids]
                avg_confidence = sum(c.confidence for c in cluster_classifications) / len(cluster_classifications)
                avg_salience = sum(saliences.get(eid, 0) for eid in entity_ids) / len(entity_ids)

                tier_dist: Dict[str, int] = {}
                for c in cluster_classifications:
                    tier_key = str(int(c.tier) if hasattr(c.tier, '__int__') else c.tier)
                    tier_dist[tier_key] = tier_dist.get(tier_key, 0) + 1

                max_tier = max(int(c.tier) if hasattr(c.tier, '__int__') else c.tier for c in cluster_classifications)

                cluster_node = self.node_builder.build_semantic_cluster_node(
                    window_source_id=window_source_id,
                    category=category,
                    classification_tier=max_tier,
                    window_label=window_label,
                    member_count=len(entity_ids),
                    avg_confidence=avg_confidence,
                    avg_salience=avg_salience,
                    tier_distribution=tier_dist
                )
                nodes.append(cluster_node)
                current_cluster_nodes[category] = cluster_node

                # Update stats
                if category not in self.stats["clusters_by_category"]:
                    self.stats["clusters_by_category"][category] = 0
                self.stats["clusters_by_category"][category] += 1

                # HAS_CLUSTER edge
                edges.append(self.edge_builder.build_edge(
                    EdgeType.HAS_CLUSTER,
                    window_node.node_id,
                    cluster_node.node_id,
                    {"category": category, "member_count": len(entity_ids)}
                ))

                # CLUSTER_CONTAINS edges (limited to top_k)
                sorted_cluster_entities = sorted(
                    entity_ids,
                    key=lambda eid: (-saliences.get(eid, 0), eid)
                )[:self.config.top_k_per_cluster]

                for rank, eid in enumerate(sorted_cluster_entities):
                    # Get entity node ID
                    entity_node_id = self.node_builder._compute_node_id(NodeType.ENTITY, eid)
                    classification = classifications[eid]

                    edges.append(self.edge_builder.build_edge(
                        EdgeType.CLUSTER_CONTAINS,
                        cluster_node.node_id,
                        entity_node_id,
                        {
                            "window_salience": saliences.get(eid, 0),
                            "classification_tier": classification.tier,
                            "classification_confidence": classification.confidence,
                            "rank": rank
                        }
                    ))

                # EVOLVES_TO edge (connect to previous window's same-category cluster)
                if category in prev_cluster_nodes:
                    prev_cluster = prev_cluster_nodes[category]
                    # Compute overlap
                    prev_window = windows[window_idx - 1]
                    prev_classifications = window_classifications.get(prev_window.window_key, {})
                    prev_entities = set(
                        eid for eid, c in prev_classifications.items()
                        if c.category == category
                    )
                    current_entities = set(entity_ids)
                    overlap_count = len(prev_entities & current_entities)
                    union_count = len(prev_entities | current_entities)
                    jaccard = overlap_count / union_count if union_count > 0 else 0

                    edges.append(self.edge_builder.build_edge(
                        EdgeType.EVOLVES_TO,
                        prev_cluster.node_id,
                        cluster_node.node_id,
                        {
                            "entity_overlap_count": overlap_count,
                            "entity_overlap_jaccard": jaccard
                        }
                    ))

            prev_cluster_nodes = current_cluster_nodes

        return nodes, edges


# ===| MAIN PIPELINE |===

class GraphMaterializationPipeline:
    """
    Stage 6: Temporal Reasoning Layer pipeline.

    Creates deterministic, read-only graph-shaped tables for visualization/export.
    """

    def __init__(self, config: Stage6Config):
        self.config = config
        self.db = Stage6Database(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.node_builder = GraphNodeBuilder(self.id_generator, self.db)
        self.edge_builder = GraphEdgeBuilder(self.id_generator)
        self.stage_started_at_utc = TimestampUtils.now_utc()

        # Node ID lookups (built during node generation)
        self.entity_node_ids: Dict[str, str] = {}  # entity_id -> node_id
        self.predicate_node_ids: Dict[str, str] = {}  # predicate_id -> node_id
        self.message_node_ids: Dict[str, str] = {}  # message_id -> node_id
        self.assertion_node_ids: Dict[str, str] = {}  # assertion_id -> node_id
        self.value_node_ids: Dict[str, str] = {}  # value_signature -> node_id
        self.time_interval_node_ids: Dict[str, str] = {}  # interval_key -> node_id
        self.retraction_node_ids: Dict[str, str] = {}  # retraction_id -> node_id
        self.conflict_group_node_ids: Dict[str, str] = {}  # conflict_group_id -> node_id
        self.lexicon_term_node_ids: Dict[str, str] = {}  # term_id -> node_id

        # Statistics
        self.stats: Dict[str, Any] = {
            "nodes_by_type": {},
            "edges_by_type": {},
            "entities_with_salience": 0,
            "assertions_by_status": {},
            "assertions_by_detection_tier": {},
            "conflict_groups_by_type": {},
            "lexicon_terms_linked": 0,
        }

    def run(self) -> Dict[str, Any]:
        """Execute Stage 6 pipeline. Returns statistics."""
        logger.info("Starting Stage 6: Graph Materialization")

        # Check prerequisites
        self.db.check_required_tables()

        # Initialize schema
        self.db.initialize_stage6_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Phase A-C: Generate core nodes
            logger.info("Phase A-C: Generating nodes...")
            self._generate_nodes()

            # Generate edges
            logger.info("Generating edges...")
            self._generate_edges()

            # Temporal profiles (if enabled)
            if self.config.enable_temporal_profiles:
                logger.info("Generating temporal profiles...")
                self._generate_temporal_profiles()
            else:
                logger.info("Generating temporal profiles was skipped")

            # Commit transaction
            self.db.commit()
            logger.info("Stage 6 completed successfully")

        except Exception as e:
            logger.error(f"Stage 6 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return self.stats

    def _generate_nodes(self):
        """Generate all graph nodes in type order."""
        all_nodes: List[GraphNode] = []

        # Entity nodes
        logger.info("  Generating Entity nodes...")
        entity_nodes = self._generate_entity_nodes()
        logger.info(f"  Created {len(entity_nodes)} Entity nodes")
        all_nodes.extend(entity_nodes)

        # Predicate nodes
        logger.info("  Generating Predicate nodes...")
        predicate_nodes = self._generate_predicate_nodes()
        logger.info(f"  Created {len(predicate_nodes)} Predicate nodes")
        all_nodes.extend(predicate_nodes)

        # Message nodes (optional)
        if self.config.include_message_nodes:
            logger.info("  Generating Message nodes...")
            message_nodes = self._generate_message_nodes()
            logger.info(f"  Created {len(message_nodes)} Message nodes")
            all_nodes.extend(message_nodes)
        else:
            logger.info("  Message nodes generation was skipped")

        # Assertion nodes (and collect Value/TimeInterval requirements)
        logger.info("  Generating Assertion nodes...")
        assertion_nodes, value_specs, time_interval_specs = self._generate_assertion_nodes()
        logger.info(f"  Created {len(assertion_nodes)} Assertion nodes")
        all_nodes.extend(assertion_nodes)

        # Value nodes
        logger.info("  Generating Value nodes...")
        value_nodes = self._generate_value_nodes(value_specs)
        logger.info(f"  Created {len(value_nodes)} Value nodes")
        all_nodes.extend(value_nodes)

        # TimeInterval nodes
        logger.info("  Generating TimeInterval nodes...")
        time_interval_nodes = self._generate_time_interval_nodes(time_interval_specs)
        logger.info(f"  Created {len(time_interval_nodes)} Time Interval nodes")
        all_nodes.extend(time_interval_nodes)

        # Retraction nodes
        logger.info("  Generating Retraction nodes...")
        retraction_nodes = self._generate_retraction_nodes()
        logger.info(f"  Created {len(retraction_nodes)} Retraction nodes")
        all_nodes.extend(retraction_nodes)

        # ConflictGroup nodes
        logger.info("  Generating ConflictGroup nodes...")
        conflict_group_nodes = self._generate_conflict_group_nodes()
        logger.info(f"  Created {len(conflict_group_nodes)} ConflictGroup nodes")
        all_nodes.extend(conflict_group_nodes)

        # LexiconTerm nodes (optional)
        if self.config.include_lexicon_nodes:
            logger.info("  Generating LexiconTerm nodes...")
            lexicon_term_nodes = self._generate_lexicon_term_nodes()
            logger.info(f"  Created {len(lexicon_term_nodes)} LexiconTerm nodes")
            all_nodes.extend(lexicon_term_nodes)
        else:
            logger.info(" LexiconTerm nodes generation skipped")

        # Sort nodes by (node_type ASC, node_id ASC) for determinism
        all_nodes.sort(key=lambda n: (n.node_type, n.node_id))

        # Insert nodes
        logger.info(f"  Inserting {len(all_nodes)} nodes...")
        self.db.insert_nodes(all_nodes)

    def _generate_entity_nodes(self) -> List[GraphNode]:
        """Generate Entity nodes."""
        nodes = []
        for entity in self.db.stream_active_entities():
            node = self.node_builder.build_entity_node(entity)
            nodes.append(node)
            self.entity_node_ids[entity['entity_id']] = node.node_id

            # Stats
            if entity['salience_score'] is not None:
                self.stats["entities_with_salience"] += 1

        self.stats["nodes_by_type"][NodeType.ENTITY] = len(nodes)
        return nodes

    def _generate_predicate_nodes(self) -> List[GraphNode]:
        """Generate Predicate nodes."""
        nodes = []
        for predicate in self.db.stream_predicates():
            node = self.node_builder.build_predicate_node(predicate)
            nodes.append(node)
            self.predicate_node_ids[predicate['predicate_id']] = node.node_id

        self.stats["nodes_by_type"][NodeType.PREDICATE] = len(nodes)
        return nodes

    def _generate_message_nodes(self) -> List[GraphNode]:
        """Generate Message nodes."""
        nodes = []
        for message in self.db.stream_messages():
            node = self.node_builder.build_message_node(message)
            nodes.append(node)
            self.message_node_ids[message['message_id']] = node.node_id

        self.stats["nodes_by_type"][NodeType.MESSAGE] = len(nodes)
        return nodes

    def _generate_assertion_nodes(self) -> Tuple[List[GraphNode], Set[Tuple[str, str]], Set[Tuple[Optional[str], Optional[str], bool]]]:
        """Generate Assertion nodes and collect Value/TimeInterval specs."""
        nodes = []
        value_specs: Set[Tuple[str, str]] = set()
        time_interval_specs: Set[Tuple[Optional[str], Optional[str], bool]] = set()

        for assertion in self.db.stream_assertions_joined():
            node = self.node_builder.build_assertion_node(assertion)
            nodes.append(node)
            self.assertion_node_ids[assertion['assertion_id']] = node.node_id

            # Collect Value specs
            obj_value_type = assertion['object_value_type'] if 'object_value_type' in assertion.keys() else None
            obj_value = assertion['object_value'] if 'object_value' in assertion.keys() else None
            if obj_value_type is not None and obj_value is not None:
                value_specs.add((obj_value_type, obj_value))

            # Collect TimeInterval specs
            valid_from = assertion['valid_from_utc']
            valid_to = assertion['valid_to_utc']
            valid_until_hint = assertion['valid_until_hint_utc']

            if valid_from is not None:
                time_interval_specs.add((valid_from, valid_to, False))

            if valid_until_hint is not None:
                time_interval_specs.add((None, valid_until_hint, True))

            # Stats
            status = assertion['status']
            self.stats["assertions_by_status"][status] = self.stats["assertions_by_status"].get(status, 0) + 1

            subj_tier = assertion['subject_detection_tier'] if 'subject_detection_tier' in assertion.keys() else None
            obj_tier = assertion['object_detection_tier'] if 'object_detection_tier' in assertion.keys() else None
            min_tier = min(t for t in [subj_tier, obj_tier, 5] if t is not None)
            if min_tier < 5:
                self.stats["assertions_by_detection_tier"][min_tier] = self.stats["assertions_by_detection_tier"].get(min_tier, 0) + 1

        self.stats["nodes_by_type"][NodeType.ASSERTION] = len(nodes)
        return nodes, value_specs, time_interval_specs

    def _generate_value_nodes(self, value_specs: Set[Tuple[str, str]]) -> List[GraphNode]:
        """Generate Value nodes."""
        nodes = []
        for value_type, value in value_specs:
            node = self.node_builder.build_value_node(value_type, value)
            nodes.append(node)
            value_sig = f"V:{HashUtils.sha256_string(JCS.canonicalize([value_type, value]))}"
            self.value_node_ids[value_sig] = node.node_id

        self.stats["nodes_by_type"][NodeType.VALUE] = len(nodes)
        return nodes

    def _generate_time_interval_nodes(self, time_interval_specs: Set[Tuple[Optional[str], Optional[str], bool]]) -> List[GraphNode]:
        """Generate TimeInterval nodes."""
        nodes = []
        for valid_from, valid_to, is_hint in time_interval_specs:
            # Skip (NULL, NULL) intervals
            if valid_from is None and valid_to is None:
                continue

            node = self.node_builder.build_time_interval_node(valid_from, valid_to, is_hint)
            nodes.append(node)
            interval_key = JCS.canonicalize([valid_from, valid_to])
            self.time_interval_node_ids[interval_key] = node.node_id

        self.stats["nodes_by_type"][NodeType.TIME_INTERVAL] = len(nodes)
        return nodes

    def _generate_retraction_nodes(self) -> List[GraphNode]:
        """Generate Retraction nodes."""
        nodes = []
        for retraction in self.db.stream_retractions():
            node = self.node_builder.build_retraction_node(retraction)
            nodes.append(node)
            self.retraction_node_ids[retraction['retraction_id']] = node.node_id

        self.stats["nodes_by_type"][NodeType.RETRACTION] = len(nodes)
        return nodes

    def _generate_conflict_group_nodes(self) -> List[GraphNode]:
        """Generate ConflictGroup nodes."""
        nodes = []
        for conflict_group in self.db.stream_conflict_groups():
            node = self.node_builder.build_conflict_group_node(conflict_group)
            nodes.append(node)
            self.conflict_group_node_ids[conflict_group['conflict_group_id']] = node.node_id

            # Stats
            conflict_type = conflict_group['conflict_type']
            self.stats["conflict_groups_by_type"][conflict_type] = self.stats["conflict_groups_by_type"].get(conflict_type, 0) + 1

        self.stats["nodes_by_type"][NodeType.CONFLICT_GROUP] = len(nodes)
        return nodes

    def _generate_lexicon_term_nodes(self) -> List[GraphNode]:
        """Generate LexiconTerm nodes."""
        nodes = []
        for term in self.db.stream_lexicon_terms():
            node = self.node_builder.build_lexicon_term_node(term)
            nodes.append(node)
            self.lexicon_term_node_ids[term['term_id']] = node.node_id

        self.stats["nodes_by_type"][NodeType.LEXICON_TERM] = len(nodes)
        return nodes

    def _generate_edges(self):
        """Generate all graph edges."""
        all_edges: List[GraphEdge] = []

        # Core semantic edges (from assertions)
        logger.info("  Generating semantic edges...")
        semantic_edges = self._generate_semantic_edges()
        logger.info(f"  Created {len(semantic_edges)} Semantic edges")
        all_edges.extend(semantic_edges)

        # Temporal edges
        logger.info("  Generating temporal edges...")
        temporal_edges = self._generate_temporal_edges()
        logger.info(f"  Created {len(temporal_edges)} Temporal edges")
        all_edges.extend(temporal_edges)

        # Message anchoring edges
        if self.config.include_message_nodes:
            logger.info("  Generating message anchoring edges...")
            message_edges = self._generate_message_anchoring_edges()
            logger.info(f"  Created {len(temporal_edges)} Temporal edges")
            all_edges.extend(message_edges)
        else:
            logger.info(f"  Temporal edge generation was skipped")

        # Lifecycle edges
        logger.info("  Generating lifecycle edges...")
        lifecycle_edges = self._generate_lifecycle_edges()
        logger.info(f"  Created {len(lifecycle_edges)} Lifecycle edges")
        all_edges.extend(lifecycle_edges)

        # Conflict edges
        logger.info("  Generating conflict edges...")
        conflict_edges = self._generate_conflict_edges()
        logger.info(f"  Created {len(conflict_edges)} Conflict edges")
        all_edges.extend(conflict_edges)

        # Lexicon provenance edges (optional)
        if self.config.include_lexicon_nodes:
            logger.info("  Generating lexicon provenance edges...")
            lexicon_edges = self._generate_lexicon_provenance_edges()
            logger.info(f"  Created {len(lexicon_edges)} Lexicon edges")
            all_edges.extend(lexicon_edges)
        else:
            logger.info(f"  Lexicon edge generation was skipped")

        # Sort edges by (edge_type ASC, edge_id ASC) for determinism
        all_edges.sort(key=lambda e: (e.edge_type, e.edge_id))

        # Insert edges
        logger.info(f"  Inserting {len(all_edges)} edges...")
        self.db.insert_edges(all_edges)

    def _generate_semantic_edges(self) -> List[GraphEdge]:
        """Generate HAS_SUBJECT, HAS_PREDICATE, HAS_OBJECT edges."""
        edges = []

        for assertion in self.db.stream_assertions_joined():
            assertion_node_id = self.assertion_node_ids.get(assertion['assertion_id'])
            if not assertion_node_id:
                continue

            # HAS_SUBJECT
            subject_node_id = self.entity_node_ids.get(assertion['subject_entity_id'])
            if subject_node_id:
                metadata = None
                if self.config.include_detection_tier_metadata:
                    subj_tier = assertion['subject_detection_tier'] if 'subject_detection_tier' in assertion.keys() else None
                    if subj_tier is not None:
                        metadata = {"detection_tier": subj_tier}

                edges.append(self.edge_builder.build_edge(
                    EdgeType.HAS_SUBJECT,
                    assertion_node_id,
                    subject_node_id,
                    metadata
                ))

            # HAS_PREDICATE
            predicate_node_id = self.predicate_node_ids.get(assertion['predicate_id'])
            if predicate_node_id:
                edges.append(self.edge_builder.build_edge(
                    EdgeType.HAS_PREDICATE,
                    assertion_node_id,
                    predicate_node_id
                ))

            # HAS_OBJECT
            obj_entity_id = assertion['object_entity_id'] if 'object_entity_id' in assertion.keys() else None
            obj_value_type = assertion['object_value_type'] if 'object_value_type' in assertion.keys() else None
            obj_value = assertion['object_value'] if 'object_value' in assertion.keys() else None

            if obj_entity_id is not None:
                # Entity object
                object_node_id = self.entity_node_ids.get(obj_entity_id)
                if object_node_id:
                    metadata = None
                    if self.config.include_detection_tier_metadata:
                        obj_tier = assertion['object_detection_tier'] if 'object_detection_tier' in assertion.keys() else None
                        if obj_tier is not None:
                            metadata = {"detection_tier": obj_tier}

                    edges.append(self.edge_builder.build_edge(
                        EdgeType.HAS_OBJECT,
                        assertion_node_id,
                        object_node_id,
                        metadata
                    ))
            elif obj_value_type is not None and obj_value is not None:
                # Literal object
                value_sig = f"V:{HashUtils.sha256_string(JCS.canonicalize([obj_value_type, obj_value]))}"
                value_node_id = self.value_node_ids.get(value_sig)
                if value_node_id:
                    edges.append(self.edge_builder.build_edge(
                        EdgeType.HAS_OBJECT,
                        assertion_node_id,
                        value_node_id
                    ))
            # Else unary - no HAS_OBJECT edge

        self.stats["edges_by_type"][EdgeType.HAS_SUBJECT] = sum(1 for e in edges if e.edge_type == EdgeType.HAS_SUBJECT)
        self.stats["edges_by_type"][EdgeType.HAS_PREDICATE] = sum(1 for e in edges if e.edge_type == EdgeType.HAS_PREDICATE)
        self.stats["edges_by_type"][EdgeType.HAS_OBJECT] = sum(1 for e in edges if e.edge_type == EdgeType.HAS_OBJECT)

        return edges

    def _generate_temporal_edges(self) -> List[GraphEdge]:
        """Generate VALID_IN and VALID_UNTIL_HINT edges."""
        edges = []

        for assertion in self.db.stream_assertions_joined():
            assertion_node_id = self.assertion_node_ids.get(assertion['assertion_id'])
            if not assertion_node_id:
                continue

            valid_from = assertion['valid_from_utc']
            valid_to = assertion['valid_to_utc']
            valid_until_hint = assertion['valid_until_hint_utc']

            # VALID_IN
            if valid_from is not None:
                interval_key = JCS.canonicalize([valid_from, valid_to])
                time_node_id = self.time_interval_node_ids.get(interval_key)
                if time_node_id:
                    edges.append(self.edge_builder.build_edge(
                        EdgeType.VALID_IN,
                        assertion_node_id,
                        time_node_id
                    ))

            # VALID_UNTIL_HINT
            if valid_until_hint is not None:
                hint_interval_key = JCS.canonicalize([None, valid_until_hint])
                hint_node_id = self.time_interval_node_ids.get(hint_interval_key)
                if hint_node_id:
                    edges.append(self.edge_builder.build_edge(
                        EdgeType.VALID_UNTIL_HINT,
                        assertion_node_id,
                        hint_node_id
                    ))

        self.stats["edges_by_type"][EdgeType.VALID_IN] = sum(1 for e in edges if e.edge_type == EdgeType.VALID_IN)
        self.stats["edges_by_type"][EdgeType.VALID_UNTIL_HINT] = sum(1 for e in edges if e.edge_type == EdgeType.VALID_UNTIL_HINT)

        return edges

    def _generate_message_anchoring_edges(self) -> List[GraphEdge]:
        """Generate ASSERTED_IN edges."""
        edges = []

        for assertion in self.db.stream_assertions_joined():
            assertion_node_id = self.assertion_node_ids.get(assertion['assertion_id'])
            message_node_id = self.message_node_ids.get(assertion['message_id'])

            if assertion_node_id and message_node_id:
                edges.append(self.edge_builder.build_edge(
                    EdgeType.ASSERTED_IN,
                    assertion_node_id,
                    message_node_id
                ))
            elif assertion_node_id and not message_node_id:
                logger.warning(f"Missing message node for assertion {assertion['assertion_id']}")

        self.stats["edges_by_type"][EdgeType.ASSERTED_IN] = len(edges)
        return edges

    def _generate_lifecycle_edges(self) -> List[GraphEdge]:
        """Generate SUPERSEDES, RETRACTED_BY, RETRACTS, NEGATED_BY, NEGATES edges."""
        edges = []

        for assertion in self.db.stream_assertions_joined():
            assertion_node_id = self.assertion_node_ids.get(assertion['assertion_id'])
            if not assertion_node_id:
                continue

            # SUPERSEDES (superseder → superseded)
            superseded_by = assertion['temporal_superseded_by_assertion_id']
            if superseded_by:
                superseder_node_id = self.assertion_node_ids.get(superseded_by)
                if superseder_node_id:
                    metadata = {}
                    rule_id = assertion['rule_id_applied']
                    if rule_id:
                        metadata["rule_id_applied"] = rule_id

                    edges.append(self.edge_builder.build_edge(
                        EdgeType.SUPERSEDES,
                        superseder_node_id,
                        assertion_node_id,
                        metadata if metadata else None
                    ))

            # RETRACTED_BY
            retracted_by = assertion['retracted_by_retraction_id']
            if retracted_by:
                retraction_node_id = self.retraction_node_ids.get(retracted_by)
                if retraction_node_id:
                    edges.append(self.edge_builder.build_edge(
                        EdgeType.RETRACTED_BY,
                        assertion_node_id,
                        retraction_node_id
                    ))

                    # Optional inverse: RETRACTS
                    if self.config.include_inverse_lifecycle_edges:
                        edges.append(self.edge_builder.build_edge(
                            EdgeType.RETRACTS,
                            retraction_node_id,
                            assertion_node_id
                        ))

            # NEGATED_BY
            negated_by = assertion['negated_by_assertion_id']
            if negated_by:
                negator_node_id = self.assertion_node_ids.get(negated_by)
                if negator_node_id:
                    edges.append(self.edge_builder.build_edge(
                        EdgeType.NEGATED_BY,
                        assertion_node_id,
                        negator_node_id
                    ))

                    # Optional inverse: NEGATES
                    if self.config.include_inverse_lifecycle_edges:
                        edges.append(self.edge_builder.build_edge(
                            EdgeType.NEGATES,
                            negator_node_id,
                            assertion_node_id
                        ))

        self.stats["edges_by_type"][EdgeType.SUPERSEDES] = sum(1 for e in edges if e.edge_type == EdgeType.SUPERSEDES)
        self.stats["edges_by_type"][EdgeType.RETRACTED_BY] = sum(1 for e in edges if e.edge_type == EdgeType.RETRACTED_BY)
        self.stats["edges_by_type"][EdgeType.NEGATED_BY] = sum(1 for e in edges if e.edge_type == EdgeType.NEGATED_BY)

        if self.config.include_inverse_lifecycle_edges:
            self.stats["edges_by_type"][EdgeType.RETRACTS] = sum(1 for e in edges if e.edge_type == EdgeType.RETRACTS)
            self.stats["edges_by_type"][EdgeType.NEGATES] = sum(1 for e in edges if e.edge_type == EdgeType.NEGATES)

        return edges

    def _generate_conflict_edges(self) -> List[GraphEdge]:
        """Generate HAS_CONFLICT_MEMBER and optionally CONFLICTS_WITH edges."""
        edges = []

        # Build conflict group membership
        conflict_members: Dict[str, List[str]] = {}  # group_id -> [assertion_ids]

        for member in self.db.stream_conflict_members():
            group_id = member['conflict_group_id']
            assertion_id = member['assertion_id']

            if group_id not in conflict_members:
                conflict_members[group_id] = []
            conflict_members[group_id].append(assertion_id)

        for group_id, assertion_ids in conflict_members.items():
            group_node_id = self.conflict_group_node_ids.get(group_id)
            if not group_node_id:
                continue

            # Get conflict type for metadata
            conflict_group = self.db.fetchone(
                "SELECT conflict_type FROM conflict_groups WHERE conflict_group_id = ?",
                (group_id,)
            )
            conflict_type = conflict_group['conflict_type'] if conflict_group else None

            # HAS_CONFLICT_MEMBER edges
            for assertion_id in sorted(assertion_ids):
                assertion_node_id = self.assertion_node_ids.get(assertion_id)
                if assertion_node_id:
                    metadata = {"conflict_type": conflict_type} if conflict_type else None
                    edges.append(self.edge_builder.build_edge(
                        EdgeType.HAS_CONFLICT_MEMBER,
                        group_node_id,
                        assertion_node_id,
                        metadata
                    ))

            # Optional pairwise CONFLICTS_WITH edges
            n = len(assertion_ids)
            if n <= self.config.conflict_pairwise_max_n:
                # Get node IDs and sort
                member_node_ids = []
                for aid in assertion_ids:
                    node_id = self.assertion_node_ids.get(aid)
                    if node_id:
                        member_node_ids.append(node_id)

                member_node_ids.sort()

                # Create directed edges for ordered pairs (i < j)
                for i in range(len(member_node_ids)):
                    for j in range(i + 1, len(member_node_ids)):
                        edges.append(self.edge_builder.build_edge(
                            EdgeType.CONFLICTS_WITH,
                            member_node_ids[i],
                            member_node_ids[j]
                        ))

        self.stats["edges_by_type"][EdgeType.HAS_CONFLICT_MEMBER] = sum(1 for e in edges if e.edge_type == EdgeType.HAS_CONFLICT_MEMBER)
        self.stats["edges_by_type"][EdgeType.CONFLICTS_WITH] = sum(1 for e in edges if e.edge_type == EdgeType.CONFLICTS_WITH)

        return edges

    def _generate_lexicon_provenance_edges(self) -> List[GraphEdge]:
        """Generate DERIVED_FROM_LEXICON edges."""
        edges = []

        if not self.db.table_exists("lexicon_terms"):
            return edges

        # Query entities with type CUSTOM_TERM and match to lexicon terms
        rows = self.db.fetchall("""
            SELECT E.entity_id, E.entity_key, L.term_id, L.build_id, L.score
            FROM entities E
            JOIN lexicon_terms L ON E.entity_key = L.term_key
            WHERE E.entity_type = 'CUSTOM_TERM' AND E.status = 'active'
        """)

        for row in rows:
            entity_node_id = self.entity_node_ids.get(row['entity_id'])
            term_node_id = self.lexicon_term_node_ids.get(row['term_id'])

            if entity_node_id and term_node_id:
                metadata = {
                    "build_id": row['build_id'],
                    "score": row['score']
                }
                edges.append(self.edge_builder.build_edge(
                    EdgeType.DERIVED_FROM_LEXICON,
                    entity_node_id,
                    term_node_id,
                    metadata
                ))

        self.stats["edges_by_type"][EdgeType.DERIVED_FROM_LEXICON] = len(edges)
        self.stats["lexicon_terms_linked"] = len(edges)

        return edges

    def _generate_temporal_profiles(self):
        """Generate temporal profile nodes and edges."""
        if not self.config.temporal_profile_config_path:
            raise RuntimeError("CONFIG_MISSING_ERROR: temporal_profile_config_path not specified")

        if not self.config.temporal_profile_config_path.exists():
            raise RuntimeError(f"CONFIG_MISSING_ERROR: {self.config.temporal_profile_config_path} not found")

        # Load temporal profile config
        tp_config = TemporalProfileConfig.from_yaml(self.config.temporal_profile_config_path)

        # Validate config
        if not tp_config.strong_rule_keywords:
            raise RuntimeError("EMPTY_STRONG_RULES: strong_rule_keywords is empty or missing")

        for granularity in tp_config.window_granularities:
            if granularity not in [g.value for g in WindowGranularity]:
                raise RuntimeError(f"INVALID_GRANULARITY: {granularity}")

        # Determine focal entities
        focal_entities = self._get_focal_entities(tp_config)

        if not focal_entities:
            logger.info("No focal entities found for temporal profiles")
            return

        logger.info(f"Generating temporal profiles for {len(focal_entities)} focal entities")

        # Create generator
        generator = TemporalProfileGenerator(
            tp_config, self.db, self.node_builder, self.edge_builder, self.id_generator
        )

        # Generate nodes and edges
        profile_nodes, profile_edges = generator.generate(focal_entities)

        # Sort and insert
        profile_nodes.sort(key=lambda n: (n.node_type, n.node_id))
        profile_edges.sort(key=lambda e: (e.edge_type, e.edge_id))

        logger.info(f"  Inserting {len(profile_nodes)} temporal profile nodes...")
        self.db.insert_nodes(profile_nodes)

        logger.info(f"  Inserting {len(profile_edges)} temporal profile edges...")
        self.db.insert_edges(profile_edges)

        # Update stats
        for node in profile_nodes:
            self.stats["nodes_by_type"][node.node_type] = self.stats["nodes_by_type"].get(node.node_type, 0) + 1

        for edge in profile_edges:
            self.stats["edges_by_type"][edge.edge_type] = self.stats["edges_by_type"].get(edge.edge_type, 0) + 1

        self.stats["temporal_profiles_generated"] = generator.stats["profiles_generated"]
        self.stats["temporal_windows_generated"] = generator.stats["windows_generated"]
        self.stats["semantic_clusters_by_category"] = generator.stats["clusters_by_category"]
        self.stats["classifications_by_tier"] = generator.stats["classifications_by_tier"]
        self.stats["llm_classification_calls"] = generator.stats["llm_classification_calls"]

    def _get_focal_entities(self, tp_config: TemporalProfileConfig) -> List[sqlite3.Row]:
        """Get focal entities for temporal profile generation."""
        if self.config.temporal_profile_focal_entities:
            # Use explicitly specified entities
            placeholders = ','.join(['?' for _ in self.config.temporal_profile_focal_entities])
            return self.db.fetchall(f"""
                SELECT * FROM entities 
                WHERE entity_id IN ({placeholders}) AND status = 'active'
                ORDER BY salience_score DESC NULLS LAST, entity_id ASC
            """, tuple(self.config.temporal_profile_focal_entities))

        # Use entities with SELF type or high salience
        return self.db.fetchall("""
            SELECT * FROM entities 
            WHERE status = 'active' 
              AND (entity_type = 'SELF' OR salience_score >= ?)
            ORDER BY salience_score DESC NULLS LAST, entity_id ASC
        """, (self.config.temporal_profile_min_salience,))


def run_stage6(config: Stage6Config) -> Dict[str, Any]:
    """Run Stage 6 pipeline on existing database."""
    pipeline = GraphMaterializationPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 6: Graph Materialization")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: ../data/output/kg.db)"
    )
    parser.add_argument(
        "--id-namespace",
        type=str,
        default="6ba7b810-9dad-11d1-80b4-00c04fd430c8",
        help="UUID namespace for ID generation"
    )
    parser.add_argument(
        "--include-message-nodes",
        action="store_true",
        default=True,
        help="Generate Message nodes (default: True)"
    )
    parser.add_argument(
        "--no-message-nodes",
        action="store_true",
        help="Do not generate Message nodes"
    )
    parser.add_argument(
        "--include-lexicon-nodes",
        default=True,
        action="store_true",
        help="Generate LexiconTerm nodes"
    )
    parser.add_argument(
        "--include-detection-tier-metadata",
        action="store_true",
        default=True,
        help="Include detection_tier in edge metadata (default: True)"
    )
    parser.add_argument(
        "--conflict-pairwise-max-n",
        type=int,
        default=25,
        help="Max group size for pairwise CONFLICTS_WITH edges (default: 25)"
    )
    parser.add_argument(
        "--include-inverse-lifecycle-edges",
        action="store_true",
        help="Generate both directions for lifecycle edges"
    )
    parser.add_argument(
        "--enable-temporal-profiles",
        default=True,
        action="store_true",
        help="Enable temporal profile visualization"
    )
    parser.add_argument(
        "--temporal-profile-config",
        default="../data/metadata/temporal_profile_config.yaml",
        type=Path,
        help="Path to temporal_profile_config.yaml"
    )
    parser.add_argument(
        "--temporal-profile-min-salience",
        type=float,
        default=0.5,
        help="Minimum salience for auto-selected focal entities (default: 0.5)"
    )

    args = parser.parse_args()

    config = Stage6Config(
        output_file_path=args.db,
        id_namespace=args.id_namespace,
        include_message_nodes=not args.no_message_nodes,
        include_lexicon_nodes=args.include_lexicon_nodes,
        include_detection_tier_metadata=args.include_detection_tier_metadata,
        conflict_pairwise_max_n=args.conflict_pairwise_max_n,
        include_inverse_lifecycle_edges=args.include_inverse_lifecycle_edges,
        enable_temporal_profiles=args.enable_temporal_profiles,
        temporal_profile_config_path=args.temporal_profile_config,
        temporal_profile_min_salience=args.temporal_profile_min_salience,
    )

    stats = run_stage6(config)

    logger.info("\n=== Stage 6 Summary ===")
    logger.info(f"Nodes by type: {stats.get('nodes_by_type', {})}")
    logger.info(f"Edges by type: {stats.get('edges_by_type', {})}")
    logger.info(f"Entities with salience: {stats.get('entities_with_salience', 0)}")
    logger.info(f"Assertions by status: {stats.get('assertions_by_status', {})}")
    logger.info(f"Assertions by detection tier: {stats.get('assertions_by_detection_tier', {})}")
    logger.info(f"Conflict groups by type: {stats.get('conflict_groups_by_type', {})}")

    if args.include_lexicon_nodes:
        logger.info(f"Lexicon terms linked: {stats.get('lexicon_terms_linked', 0)}")

    if args.enable_temporal_profiles:
        logger.info(f"Temporal profiles generated: {stats.get('temporal_profiles_generated', 0)}")
        logger.info(f"Temporal windows generated: {stats.get('temporal_windows_generated', 0)}")
        logger.info(f"Semantic clusters by category: {stats.get('semantic_clusters_by_category', {})}")
        logger.info(f"Classifications by tier: {stats.get('classifications_by_tier', {})}")
        logger.info(f"LLM classification calls: {stats.get('llm_classification_calls', 0)}")