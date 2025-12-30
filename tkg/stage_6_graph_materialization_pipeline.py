"""
Stage 6: Graph Materialization

Materializes deterministic, read-only graph tables (graph_nodes, graph_edges) for
visualization and export. Projects data from Stages 1-5 into a stable graph shape.

Primary visualization target: a SELF-centered temporal profile suitable for
"monthly clusters of what mattered".
"""
import hashlib
import json
import logging
import sqlite3
import uuid
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Tuple, Set
import pendulum


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


# ===| CONSTANTS |===

DEFAULT_KG_NAMESPACE = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"  # Example namespace UUID


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

    def generate_node_id(self, node_type: str, source_id: str) -> str:
        """Generate deterministic node ID."""
        return self.generate(["node", node_type, source_id])

    def generate_edge_id(self, edge_type: str, src_node_id: str, dst_node_id: str) -> str:
        """Generate deterministic edge ID."""
        return self.generate(["edge", edge_type, src_node_id, dst_node_id])


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

    @staticmethod
    def sha256_jcs(obj: Any) -> str:
        """Compute SHA-256 of JCS-canonical representation."""
        return HashUtils.sha256_hex(JCS.canonicalize_bytes(obj))


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
    def parse_iso(iso_string: str) -> Optional[pendulum.DateTime]:
        """Parse ISO string to pendulum DateTime."""
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

    @staticmethod
    def get_month_window(timestamp: str) -> Tuple[str, str]:
        """Get the start and end of the month containing the timestamp."""
        dt = TimestampUtils.parse_iso(timestamp)
        if dt is None:
            raise ValueError(f"Invalid timestamp: {timestamp}")
        start = dt.start_of('month')
        end = dt.end_of('month').add(microseconds=1)  # Exclusive end
        return (
            start.format(TimestampUtils.ISO_UTC_MILLIS),
            end.format(TimestampUtils.ISO_UTC_MILLIS)
        )

    @staticmethod
    def format_window_label(start: str, granularity: str) -> str:
        """Format a window start time into a human-readable label."""
        dt = TimestampUtils.parse_iso(start)
        if dt is None:
            return start
        if granularity == "month":
            return dt.format("MMMM YYYY")
        elif granularity == "quarter":
            q = (dt.month - 1) // 3 + 1
            return f"Q{q} {dt.year}"
        elif granularity == "year":
            return str(dt.year)
        return start[:10]


# ===| ENUMS |===

class NodeType(StrEnum):
    """Graph node types."""
    # Core
    ENTITY = "Entity"
    PREDICATE = "Predicate"
    ASSERTION = "Assertion"
    VALUE = "Value"
    TIME_INTERVAL = "TimeInterval"
    # Optional provenance
    MESSAGE = "Message"
    RETRACTION = "Retraction"
    CONFLICT_GROUP = "ConflictGroup"
    LEXICON_TERM = "LexiconTerm"
    TIME_MENTION = "TimeMention"
    # Temporal profile
    TEMPORAL_PROFILE = "TemporalProfile"
    TEMPORAL_WINDOW = "TemporalWindow"
    SEMANTIC_CLUSTER = "SemanticCluster"
    SEMANTIC_CATEGORY = "SemanticCategory"
    # Derived topic nodes (for "what mattered" visualization)
    TOPIC = "Topic"


class EdgeType(StrEnum):
    """Graph edge types."""
    # Assertion semantic
    HAS_SUBJECT = "HAS_SUBJECT"
    HAS_PREDICATE = "HAS_PREDICATE"
    HAS_OBJECT = "HAS_OBJECT"
    # Temporal
    VALID_IN = "VALID_IN"
    VALID_UNTIL_HINT = "VALID_UNTIL_HINT"
    QUALIFIED_BY_TIME = "QUALIFIED_BY_TIME"
    # Message anchoring
    ASSERTED_IN = "ASSERTED_IN"
    # Lifecycle
    SUPERSEDES = "SUPERSEDES"
    RETRACTED_BY = "RETRACTED_BY"
    NEGATED_BY = "NEGATED_BY"
    RETRACTS = "RETRACTS"
    NEGATES = "NEGATES"
    # Conflicts
    HAS_CONFLICT_MEMBER = "HAS_CONFLICT_MEMBER"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    # Lexicon
    DERIVED_FROM_LEXICON = "DERIVED_FROM_LEXICON"
    # Temporal profile
    HAS_PROFILE = "HAS_PROFILE"
    HAS_WINDOW = "HAS_WINDOW"
    HAS_CLUSTER = "HAS_CLUSTER"
    CLUSTER_CONTAINS = "CLUSTER_CONTAINS"
    WINDOW_INCLUDES = "WINDOW_INCLUDES"
    WINDOW_PRECEDES = "WINDOW_PRECEDES"
    WINDOW_TOP_MEMBER = "WINDOW_TOP_MEMBER"
    WINDOW_TOP_CLUSTER = "WINDOW_TOP_CLUSTER"
    CLUSTER_OF_CATEGORY = "CLUSTER_OF_CATEGORY"
    MEMBER_EVIDENCED_BY = "MEMBER_EVIDENCED_BY"
    EVOLVES_TO = "EVOLVES_TO"
    # Topic edges
    TOPIC_SUPPORTED_BY = "TOPIC_SUPPORTED_BY"


class WindowingMode(StrEnum):
    """Windowing modes for temporal profile generation."""
    MENTION_FIRST = "mention_first"    # Each item assigned to exactly one month (first mention)
    MENTION_ANY = "mention_any"        # Items assigned to all months they're mentioned in
    VALIDITY_OVERLAP = "validity_overlap"  # Items assigned to all months their validity overlaps


class EventTimeSource(StrEnum):
    """Event time source for windowing."""
    MESSAGE_CREATED_AT = "message_created_at"
    ASSERTED_AT = "asserted_at"
    VALID_FROM = "valid_from"
    COALESCE = "coalesce"  # COALESCE(valid_from, asserted_at, message.created_at)


class AssertionStatus(StrEnum):
    """Assertion lifecycle statuses."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    NEGATED = "negated"
    CONFLICTED = "conflicted"
    INELIGIBLE = "ineligible"


class ClassificationTier(IntEnum):
    """Classification tiers for semantic clustering."""
    KEYWORD_RULES = 1
    EMBEDDING_CLUSTER = 2
    LLM_CONSENSUS = 3
    UNCLASSIFIED = 0


# ===| CONFIGURATION |===

@dataclass
class TemporalProfileConfig:
    """Configuration for temporal profile generation."""
    granularity: str = "month"
    focal_entities: Optional[List[str]] = None
    allowed_assertion_statuses: List[str] = field(
        default_factory=lambda: ["ineligible"]
    )
    window_min_assertions: int = 1
    salience_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "frequency": 0.4,
            "recency": 0.2,
            "confidence": 0.4
        }
    )
    strong_rule_keywords: Dict[str, List[str]] = field(default_factory=dict)
    predicate_cluster_categories: Dict[str, List[str]] = field(default_factory=dict)
    category_keys: List[str] = field(
        default_factory=lambda: [
            "project", "interest", "problem", "task", "desire", "plan", "unclassified"
        ]
    )
    profile_version: str = "1.0"

    # Windowing mode controls how items are assigned to time windows
    # - "mention_first": each item assigned to exactly one month (first event_time)
    # - "mention_any": items assigned to all months they're mentioned in
    # - "validity_overlap": items assigned to all months their validity interval overlaps
    windowing_mode: str = "mention_first"

    # Event time source determines which timestamp to use for windowing
    # - "coalesce": COALESCE(valid_from, asserted_at, message.created_at)
    # - "message_created_at": use message timestamp only
    # - "asserted_at": use assertion timestamp only
    # - "valid_from": use valid_from only (may miss items without temporal data)
    event_time_source: str = "coalesce"

    # Evergreen penalty (IDF-like) penalizes items that appear across many months
    # 0.0 = disabled, 1.0 = full IDF weight
    # Formula: idf = log((1 + num_months) / (1 + months_with_item))
    evergreen_penalty_strength: float = 0.3

    # Topic node generation settings
    enable_topic_nodes: bool = True  # Generate derived Topic nodes
    topic_label_max_length: int = 60  # Max length of topic label

    @classmethod
    def from_yaml(cls, path: Path) -> "TemporalProfileConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            logger.warning(f"Temporal profile config not found at {path}, using defaults")
            return cls()

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if data is None:
            return cls()

        config_data = data.get('temporal_profile_config', data)
        return cls(
            granularity=config_data.get('granularity', 'month'),
            focal_entities=config_data.get('focal_entities'),
            allowed_assertion_statuses=config_data.get(
                'allowed_assertion_statuses', ['ineligible']
            ),
            window_min_assertions=config_data.get('window_min_assertions', 1),
            salience_weights=config_data.get('salience_weights', {
                "frequency": 0.4, "recency": 0.2, "confidence": 0.4
            }),
            strong_rule_keywords=config_data.get('strong_rule_keywords', {}),
            predicate_cluster_categories=config_data.get('predicate_cluster_categories', {}),
            category_keys=config_data.get('category_keys', [
                "project", "interest", "problem", "task", "desire", "plan", "unclassified"
            ]),
            profile_version=config_data.get('profile_version', '1.0'),
            windowing_mode=config_data.get('windowing_mode', 'mention_first'),
            event_time_source=config_data.get('event_time_source', 'coalesce'),
            evergreen_penalty_strength=config_data.get('evergreen_penalty_strength', 0.3),
            enable_topic_nodes=config_data.get('enable_topic_nodes', True),
            topic_label_max_length=config_data.get('topic_label_max_length', 60)
        )


@dataclass
class Stage6Config:
    """Configuration for Stage 6 pipeline."""
    output_file_path: Path
    id_namespace: str = DEFAULT_KG_NAMESPACE

    # Core graph generation
    include_message_nodes: bool = True
    include_lexicon_nodes: bool = True
    include_detection_tier_metadata: bool = True
    conflict_pairwise_max_n: int = 10
    include_inverse_lifecycle_edges: bool = False

    # Temporal profiles
    enable_temporal_profiles: bool = True
    temporal_profile_config_path: Optional[Path] = None
    include_window_assertion_edges: bool = False
    include_cluster_evolution_edges: bool = False

    # Visualization-focused optional expansions
    include_value_members_in_profiles: bool = True
    include_category_nodes: bool = True
    include_window_sequence_edges: bool = True
    include_window_top_edges: bool = True
    window_top_n_members: int = 10
    window_top_n_clusters: int = 6
    cluster_top_n_members: int = 50  # Limit CLUSTER_CONTAINS edges for clutter reduction
    include_time_mention_nodes: bool = False
    include_time_qualifier_edges: bool = False
    include_member_evidence_edges: bool = False
    member_evidence_edge_cap_per_window: int = 50  # Cap evidence edges per member per window

    # Topic node generation (derived "what mattered" themes)
    include_topic_nodes: bool = True
    use_topics_as_primary_members: bool = True  # Use topics instead of raw entities/values as cluster members


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
class TemporalWindow:
    """Represents a temporal window for profiles."""
    window_id: str
    entity_id: str
    granularity: str
    window_start_utc: str
    window_end_utc: str
    assertion_ids: List[str] = field(default_factory=list)

    @property
    def source_id(self) -> str:
        return HashUtils.sha256_jcs([
            self.entity_id,
            self.window_start_utc,
            self.window_end_utc,
            self.granularity
        ])


@dataclass
class SemanticCluster:
    """Represents a semantic cluster within a window."""
    cluster_id: str
    window_source_id: str
    category_key: str
    members: List[Tuple[str, str, float, int]]  # (node_id, node_type, salience, tier)

    @property
    def source_id(self) -> str:
        return HashUtils.sha256_jcs([
            self.window_source_id,
            self.category_key,
            0  # classification_tier placeholder
        ])


@dataclass
class DataQualityReport:
    """Data quality check results."""
    entity_count: int = 0
    predicate_count: int = 0
    assertion_count: int = 0
    temporalized_count: int = 0
    temporalized_coverage: float = 0.0
    status_distribution: Dict[str, int] = field(default_factory=dict)
    valid_from_rate: float = 0.0
    date_span: Optional[Tuple[str, str]] = None
    literal_object_rate: float = 0.0
    self_entity_id: Optional[str] = None
    time_mention_count: int = 0
    issues: List[str] = field(default_factory=list)


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

    def executemany(self, sql: str, params_list: List[tuple]):
        """Execute SQL statement with multiple parameter sets."""
        self.connection.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute and fetch one result."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute and fetch all results."""
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
        "conversations",
        "messages",
        "entities",
        "entity_mentions",
        "predicates",
        "assertions",
        "assertion_temporalized"
    ]

    OPTIONAL_TABLES = [
        "time_mentions",
        "retractions",
        "conflict_groups",
        "conflict_members",
        "lexicon_terms",
        "message_parts"
    ]

    def check_required_tables(self) -> Dict[str, bool]:
        """Check for required tables from previous stages."""
        table_status = {}
        missing = []

        for table in self.REQUIRED_TABLES:
            exists = self.table_exists(table)
            table_status[table] = exists
            if not exists:
                missing.append(table)

        for table in self.OPTIONAL_TABLES:
            table_status[table] = self.table_exists(table)

        if missing:
            raise RuntimeError(
                f"Missing required tables from previous stages: {missing}"
            )

        logger.info(f"Table status: {table_status}")
        return table_status

    def initialize_stage6_schema(self):
        """Create Stage 6 tables."""
        # Drop existing tables if they exist
        self.execute("DROP TABLE IF EXISTS graph_edges")
        self.execute("DROP TABLE IF EXISTS graph_nodes")

        # Create graph_nodes
        self.execute("""
            CREATE TABLE graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                label TEXT NOT NULL,
                metadata_json TEXT
            )
        """)

        # Create graph_edges
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

        # Create indices
        self.execute("CREATE INDEX idx_graph_nodes_type ON graph_nodes(node_type, node_id)")
        self.execute("CREATE INDEX idx_graph_nodes_source ON graph_nodes(node_type, source_id)")
        self.execute("CREATE INDEX idx_graph_edges_src ON graph_edges(src_node_id)")
        self.execute("CREATE INDEX idx_graph_edges_dst ON graph_edges(dst_node_id)")
        self.execute("CREATE INDEX idx_graph_edges_type ON graph_edges(edge_type, edge_id)")

        logger.info("Stage 6 schema initialized")

    def insert_node(self, node: GraphNode):
        """Insert a single node."""
        self.execute(
            """INSERT OR REPLACE INTO graph_nodes 
               (node_id, node_type, source_id, label, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (node.node_id, node.node_type, node.source_id, node.label, node.metadata_json)
        )

    def insert_nodes_batch(self, nodes: List[GraphNode]):
        """Insert nodes in batch."""
        if not nodes:
            return
        self.executemany(
            """INSERT OR REPLACE INTO graph_nodes 
               (node_id, node_type, source_id, label, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            [(n.node_id, n.node_type, n.source_id, n.label, n.metadata_json) for n in nodes]
        )

    def insert_edge(self, edge: GraphEdge):
        """Insert a single edge."""
        self.execute(
            """INSERT OR REPLACE INTO graph_edges 
               (edge_id, edge_type, src_node_id, dst_node_id, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            (edge.edge_id, edge.edge_type, edge.src_node_id, edge.dst_node_id, edge.metadata_json)
        )

    def insert_edges_batch(self, edges: List[GraphEdge]):
        """Insert edges in batch."""
        if not edges:
            return
        self.executemany(
            """INSERT OR REPLACE INTO graph_edges 
               (edge_id, edge_type, src_node_id, dst_node_id, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            [(e.edge_id, e.edge_type, e.src_node_id, e.dst_node_id, e.metadata_json) for e in edges]
        )

    def get_node_by_source(self, node_type: str, source_id: str) -> Optional[str]:
        """Get node_id by type and source_id."""
        result = self.fetchone(
            "SELECT node_id FROM graph_nodes WHERE node_type = ? AND source_id = ?",
            (node_type, source_id)
        )
        return result['node_id'] if result else None

    def node_exists(self, node_id: str) -> bool:
        """Check if a node exists."""
        result = self.fetchone(
            "SELECT 1 FROM graph_nodes WHERE node_id = ?",
            (node_id,)
        )
        return result is not None


# ===| NODE GENERATORS |===

class NodeGenerator:
    """Generates graph nodes from source data."""

    def __init__(self, db: Stage6Database, config: Stage6Config, id_gen: IDGenerator):
        self.db = db
        self.config = config
        self.id_gen = id_gen
        self.node_registry: Dict[str, str] = {}  # "NodeType:source_id" -> node_id
        self.value_source_map: Dict[str, Tuple[str, str]] = {}  # source_id -> (value_type, value)
        self.topic_registry: Dict[str, Dict[str, Any]] = {}  # topic_source_id -> topic_data

    def _make_metadata(self, data: Dict[str, Any]) -> str:
        """Create JCS-canonical metadata JSON with schema version."""
        data["schema_version"] = "1.0"
        return JCS.canonicalize(data)

    def _register_node(self, node_type: str, source_id: str) -> str:
        """Register and return node_id."""
        node_id = self.id_gen.generate_node_id(node_type, source_id)
        key = f"{node_type}:{source_id}"
        self.node_registry[key] = node_id
        return node_id

    def get_node_id(self, node_type: str, source_id: str) -> Optional[str]:
        """Get registered node_id."""
        key = f"{node_type}:{source_id}"
        return self.node_registry.get(key)

    def generate_entity_nodes(self) -> List[GraphNode]:
        """Generate Entity nodes from active entities."""
        logger.info("Generating Entity nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT entity_id, entity_type, entity_key, canonical_name, 
                   aliases_json, status, first_seen_at_utc, last_seen_at_utc,
                   mention_count, conversation_count, salience_score
            FROM entities
            ORDER BY entity_id ASC
        """)

        for row in rows:
            source_id = row['entity_id']
            node_id = self._register_node(NodeType.ENTITY, source_id)

            metadata = {
                "entity_id": row['entity_id'],
                "entity_type": row['entity_type'],
                "entity_key": row['entity_key'],
                "aliases": json.loads(row['aliases_json']) if row['aliases_json'] else [],
                "first_seen_at_utc": row['first_seen_at_utc'],
                "last_seen_at_utc": row['last_seen_at_utc'],
                "mention_count": row['mention_count'],
                "conversation_count": row['conversation_count'],
                "salience_score": row['salience_score']
            }

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.ENTITY,
                source_id=source_id,
                label=row['canonical_name'] or row['entity_key'],
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} Entity nodes")
        return nodes

    def generate_predicate_nodes(self) -> List[GraphNode]:
        """Generate Predicate nodes."""
        logger.info("Generating Predicate nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT predicate_id, canonical_label, canonical_label_norm,
                   inverse_label, category, arity, value_type_constraint,
                   first_seen_at_utc, assertion_count
            FROM predicates
            ORDER BY predicate_id ASC
        """)

        for row in rows:
            source_id = row['predicate_id']
            node_id = self._register_node(NodeType.PREDICATE, source_id)

            metadata = {
                "predicate_id": row['predicate_id'],
                "canonical_label_norm": row['canonical_label_norm'],
                "inverse_label": row['inverse_label'],
                "category": row['category'],
                "arity": row['arity'],
                "value_type_constraint": row['value_type_constraint'],
                "first_seen_at_utc": row['first_seen_at_utc'],
                "assertion_count": row['assertion_count']
            }

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.PREDICATE,
                source_id=source_id,
                label=row['canonical_label'],
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} Predicate nodes")
        return nodes

    def generate_message_nodes(self) -> List[GraphNode]:
        """Generate Message nodes (optional)."""
        if not self.config.include_message_nodes:
            return []

        logger.info("Generating Message nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT message_id, conversation_id, role, order_index,
                   created_at_utc, content_type
            FROM messages
            ORDER BY conversation_id ASC, order_index ASC, message_id ASC
        """)

        for row in rows:
            source_id = row['message_id']
            node_id = self._register_node(NodeType.MESSAGE, source_id)

            metadata = {
                "message_id": row['message_id'],
                "conversation_id": row['conversation_id'],
                "role": row['role'],
                "order_index": row['order_index'],
                "created_at_utc": row['created_at_utc'],
                "content_type": row['content_type']
            }

            label = f"Message {row['order_index']} ({row['role']})"

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.MESSAGE,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} Message nodes")
        return nodes

    def generate_assertion_nodes(self) -> List[GraphNode]:
        """Generate Assertion nodes with temporal data."""
        logger.info("Generating Assertion nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT 
                a.assertion_id, a.message_id, a.assertion_key, a.fact_key,
                a.subject_entity_id, a.subject_detection_tier,
                a.predicate_id, a.object_entity_id, a.object_detection_tier,
                a.object_value_type, a.object_value, a.object_signature,
                a.temporal_qualifier_type, a.temporal_qualifier_id,
                a.modality, a.polarity, a.asserted_role, a.asserted_at_utc,
                a.confidence_extraction, a.confidence_grounding, a.confidence_final,
                a.has_user_corroboration, a.char_start, a.char_end, a.surface_text,
                a.extraction_method, a.extraction_model,
                t.valid_time_type, t.valid_from_utc, t.valid_to_utc,
                t.valid_until_hint_utc, t.status,
                t.temporal_superseded_by_assertion_id,
                t.retracted_by_retraction_id, t.negated_by_assertion_id,
                t.rule_id_applied
            FROM assertions a
            LEFT JOIN assertion_temporalized t ON a.assertion_id = t.assertion_id
            ORDER BY a.assertion_id ASC
        """)

        for row in rows:
            source_id = row['assertion_id']
            node_id = self._register_node(NodeType.ASSERTION, source_id)

            metadata = {
                "assertion_id": row['assertion_id'],
                "message_id": row['message_id'],
                "assertion_key": row['assertion_key'],
                "fact_key": row['fact_key'],
                "subject_entity_id": row['subject_entity_id'],
                "predicate_id": row['predicate_id'],
                "object_entity_id": row['object_entity_id'],
                "object_value_type": row['object_value_type'],
                "object_value": row['object_value'],
                "object_signature": row['object_signature'],
                "temporal_qualifier_type": row['temporal_qualifier_type'],
                "temporal_qualifier_id": row['temporal_qualifier_id'],
                "modality": row['modality'],
                "polarity": row['polarity'],
                "asserted_role": row['asserted_role'],
                "asserted_at_utc": row['asserted_at_utc'],
                "confidence_extraction": row['confidence_extraction'],
                "confidence_grounding": row['confidence_grounding'],
                "confidence_final": row['confidence_final'],
                "has_user_corroboration": bool(row['has_user_corroboration']),
                "extraction_method": row['extraction_method'],
                # Temporal fields from assertion_temporalized
                "valid_time_type": row['valid_time_type'],
                "valid_from_utc": row['valid_from_utc'],
                "valid_to_utc": row['valid_to_utc'],
                "valid_until_hint_utc": row['valid_until_hint_utc'],
                "status": row['status'] or "unknown",
                "temporal_superseded_by": row['temporal_superseded_by_assertion_id'],
                "retracted_by": row['retracted_by_retraction_id'],
                "negated_by": row['negated_by_assertion_id']
            }

            if self.config.include_detection_tier_metadata:
                metadata["subject_detection_tier"] = row['subject_detection_tier']
                metadata["object_detection_tier"] = row['object_detection_tier']

            # Create label from surface text or components
            if row['surface_text']:
                label = row['surface_text'][:100]
            else:
                label = f"Assertion ({row['modality']}, {row['polarity']})"

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.ASSERTION,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} Assertion nodes")
        return nodes

    def generate_value_nodes(self) -> List[GraphNode]:
        """Generate Value nodes for literal assertion objects.

        Also populates self.value_source_map for efficient lookups in classify_member.
        """
        logger.info("Generating Value nodes")
        nodes = []
        seen_values: Set[str] = set()

        rows = self.db.fetchall("""
            SELECT DISTINCT object_value_type, object_value
            FROM assertions
            WHERE object_value_type IS NOT NULL AND object_value IS NOT NULL
            ORDER BY object_value_type ASC, object_value ASC
        """)

        for row in rows:
            value_type = row['object_value_type']
            value = row['object_value']

            source_id = HashUtils.sha256_jcs([value_type, value])
            if source_id in seen_values:
                continue
            seen_values.add(source_id)

            # Store in value_source_map for efficient classify_member lookups
            self.value_source_map[source_id] = (value_type, value)

            node_id = self._register_node(NodeType.VALUE, source_id)

            # Parse value for display
            try:
                parsed_value = json.loads(value) if value else value
            except (json.JSONDecodeError, TypeError):
                parsed_value = value

            metadata = {
                "value_type": value_type,
                "value": parsed_value
            }

            # Create readable label
            if isinstance(parsed_value, str):
                label = parsed_value[:50] if len(parsed_value) > 50 else parsed_value
            else:
                label = str(parsed_value)[:50]

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.VALUE,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} Value nodes")
        return nodes

    def generate_time_interval_nodes(self) -> List[GraphNode]:
        """Generate TimeInterval nodes from assertion valid times."""
        logger.info("Generating TimeInterval nodes")
        nodes = []
        seen_intervals: Set[str] = set()

        # Get intervals from valid_from/valid_to
        rows = self.db.fetchall("""
            SELECT DISTINCT valid_from_utc, valid_to_utc
            FROM assertion_temporalized
            WHERE valid_from_utc IS NOT NULL
            ORDER BY valid_from_utc ASC, valid_to_utc ASC
        """)

        for row in rows:
            valid_from = row['valid_from_utc']
            valid_to = row['valid_to_utc']

            source_id = HashUtils.sha256_jcs([valid_from, valid_to])
            if source_id in seen_intervals:
                continue
            seen_intervals.add(source_id)

            node_id = self._register_node(NodeType.TIME_INTERVAL, source_id)

            metadata = {
                "valid_from_utc": valid_from,
                "valid_to_utc": valid_to,
                "is_open_ended": valid_to is None,
                "is_hint_only": False
            }

            # Create label
            if valid_to:
                label = f"{valid_from[:10]} to {valid_to[:10]}"
            else:
                label = f"Since {valid_from[:10]}"

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.TIME_INTERVAL,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        # Also get hint-only intervals
        hint_rows = self.db.fetchall("""
            SELECT DISTINCT valid_until_hint_utc
            FROM assertion_temporalized
            WHERE valid_until_hint_utc IS NOT NULL 
              AND valid_from_utc IS NULL
            ORDER BY valid_until_hint_utc ASC
        """)

        for row in hint_rows:
            hint = row['valid_until_hint_utc']
            source_id = HashUtils.sha256_jcs([None, hint])
            if source_id in seen_intervals:
                continue
            seen_intervals.add(source_id)

            node_id = self._register_node(NodeType.TIME_INTERVAL, source_id)

            metadata = {
                "valid_from_utc": None,
                "valid_to_utc": hint,
                "is_open_ended": False,
                "is_hint_only": True
            }

            label = f"Until {hint[:10]} (hint)"

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.TIME_INTERVAL,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} TimeInterval nodes")
        return nodes

    def generate_retraction_nodes(self) -> List[GraphNode]:
        """Generate Retraction nodes."""
        if not self.db.table_exists("retractions"):
            return []

        logger.info("Generating Retraction nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT retraction_id, retraction_message_id, target_assertion_id,
                   target_fact_key, retraction_type, replacement_assertion_id,
                   confidence, char_start, char_end, surface_text
            FROM retractions
            ORDER BY retraction_id ASC
        """)

        for row in rows:
            source_id = row['retraction_id']
            node_id = self._register_node(NodeType.RETRACTION, source_id)

            metadata = {
                "retraction_id": row['retraction_id'],
                "retraction_message_id": row['retraction_message_id'],
                "target_assertion_id": row['target_assertion_id'],
                "target_fact_key": row['target_fact_key'],
                "retraction_type": row['retraction_type'],
                "replacement_assertion_id": row['replacement_assertion_id'],
                "confidence": row['confidence']
            }

            label = f"Retraction ({row['retraction_type']})"
            if row['surface_text']:
                label = row['surface_text'][:50]

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.RETRACTION,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} Retraction nodes")
        return nodes

    def generate_conflict_group_nodes(self) -> List[GraphNode]:
        """Generate ConflictGroup nodes."""
        if not self.db.table_exists("conflict_groups"):
            return []

        logger.info("Generating ConflictGroup nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT conflict_group_id, conflict_type, conflict_key, detected_at_utc
            FROM conflict_groups
            ORDER BY conflict_group_id ASC
        """)

        for row in rows:
            source_id = row['conflict_group_id']
            node_id = self._register_node(NodeType.CONFLICT_GROUP, source_id)

            # Count members
            member_count = self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM conflict_members WHERE conflict_group_id = ?",
                (source_id,)
            )['cnt']

            metadata = {
                "conflict_group_id": row['conflict_group_id'],
                "conflict_type": row['conflict_type'],
                "conflict_key": row['conflict_key'],
                "detected_at_utc": row['detected_at_utc'],
                "member_count": member_count
            }

            label = f"{row['conflict_type']} ({member_count} members)"

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.CONFLICT_GROUP,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} ConflictGroup nodes")
        return nodes

    def generate_lexicon_term_nodes(self) -> List[GraphNode]:
        """Generate LexiconTerm nodes (optional)."""
        if not self.config.include_lexicon_nodes:
            return []
        if not self.db.table_exists("lexicon_terms"):
            return []

        logger.info("Generating LexiconTerm nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT term_id, build_id, candidate_id, term_key,
                   canonical_surface, aliases_json, score, entity_type_hint
            FROM lexicon_terms
            ORDER BY term_id ASC
        """)

        for row in rows:
            source_id = row['term_id']
            node_id = self._register_node(NodeType.LEXICON_TERM, source_id)

            metadata = {
                "term_id": row['term_id'],
                "build_id": row['build_id'],
                "term_key": row['term_key'],
                "canonical_surface": row['canonical_surface'],
                "aliases": json.loads(row['aliases_json']) if row['aliases_json'] else [],
                "score": row['score'],
                "entity_type_hint": row['entity_type_hint']
            }

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.LEXICON_TERM,
                source_id=source_id,
                label=row['canonical_surface'],
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} LexiconTerm nodes")
        return nodes

    def generate_time_mention_nodes(self) -> List[GraphNode]:
        """Generate TimeMention nodes (optional)."""
        if not self.config.include_time_mention_nodes:
            return []
        if not self.db.table_exists("time_mentions"):
            return []

        logger.info("Generating TimeMention nodes")
        nodes = []

        rows = self.db.fetchall("""
            SELECT time_mention_id, message_id, char_start, char_end,
                   surface_text, pattern_id, anchor_time_utc,
                   resolved_type, valid_from_utc, valid_to_utc,
                   resolution_granularity, timezone_assumed, confidence
            FROM time_mentions
            ORDER BY time_mention_id ASC
        """)

        for row in rows:
            source_id = row['time_mention_id']
            node_id = self._register_node(NodeType.TIME_MENTION, source_id)

            metadata = {
                "time_mention_id": row['time_mention_id'],
                "message_id": row['message_id'],
                "surface_text": row['surface_text'],
                "resolved_type": row['resolved_type'],
                "valid_from_utc": row['valid_from_utc'],
                "valid_to_utc": row['valid_to_utc'],
                "resolution_granularity": row['resolution_granularity'],
                "timezone_assumed": row['timezone_assumed'],
                "confidence": row['confidence'],
                "anchor_time_utc": row['anchor_time_utc'],
                "pattern_id": row['pattern_id']
            }

            label = row['surface_text'][:30] if row['surface_text'] else "Time"

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.TIME_MENTION,
                source_id=source_id,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} TimeMention nodes")
        return nodes

    def generate_semantic_category_nodes(self, categories: List[str]) -> List[GraphNode]:
        """Generate SemanticCategory nodes for temporal profiles."""
        if not self.config.include_category_nodes:
            return []

        logger.info("Generating SemanticCategory nodes")
        nodes = []

        for category_key in sorted(categories):
            source_id = category_key
            node_id = self._register_node(NodeType.SEMANTIC_CATEGORY, source_id)

            metadata = {
                "category_key": category_key,
                "display_label": category_key.replace("_", " ").title()
            }

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.SEMANTIC_CATEGORY,
                source_id=source_id,
                label=metadata["display_label"],
                metadata_json=self._make_metadata(metadata)
            ))

        logger.info(f"Generated {len(nodes)} SemanticCategory nodes")
        return nodes

    def generate_topic_nodes(
        self,
        focal_entity_id: str,
        profile_config: 'TemporalProfileConfig'
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Generate Topic nodes from SELF-anchored assertions.

        Topics are derived from (predicate, normalized_object, modality) tuples,
        providing a more semantic view than raw entity/value objects.

        Labels are human-readable phrases like "work on: dissertation",
        "want: travel to Japan", "has task: fix login bug".

        Also generates TOPIC_SUPPORTED_BY edges linking topics to assertions.

        Returns:
            Tuple of (topic_nodes, topic_edges)
        """
        if not self.config.include_topic_nodes:
            return [], []

        if not profile_config.enable_topic_nodes:
            return [], []

        logger.info(f"Generating Topic nodes for entity {focal_entity_id}")
        nodes = []
        edges = []
        seen_topics: Set[str] = set()

        # Get SELF-anchored assertions with their predicates and objects
        status_placeholders = ','.join('?' * len(profile_config.allowed_assertion_statuses))
        rows = self.db.fetchall(f"""
            SELECT a.assertion_id, a.predicate_id, p.canonical_label, p.category,
                   a.object_entity_id, e.canonical_name as object_entity_name,
                   a.object_value_type, a.object_value, a.object_signature,
                   a.modality, a.polarity, a.confidence_final,
                   COALESCE(t.valid_from_utc, a.asserted_at_utc, m.created_at_utc) as event_time_utc
            FROM assertions a
            JOIN predicates p ON a.predicate_id = p.predicate_id
            JOIN assertion_temporalized t ON a.assertion_id = t.assertion_id
            LEFT JOIN messages m ON a.message_id = m.message_id
            LEFT JOIN entities e ON a.object_entity_id = e.entity_id
            WHERE a.subject_entity_id = ?
              AND t.status IN ({status_placeholders})
            ORDER BY event_time_utc ASC, a.assertion_id ASC
        """, (focal_entity_id, *profile_config.allowed_assertion_statuses))

        # Group assertions by topic_key
        topic_assertions: Dict[str, List[Dict]] = {}

        for row in rows:
            # Determine normalized object
            if row['object_entity_id']:
                normalized_object = row['object_entity_name'] or row['object_entity_id']
                object_type = "entity"
            elif row['object_value']:
                normalized_object = row['object_value']
                object_type = "value"
            else:
                continue

            # Create topic key: deterministic hash of (predicate_id, normalized_object, modality, polarity)
            topic_key = HashUtils.sha256_jcs([
                row['predicate_id'],
                normalized_object,
                row['modality'] or "",
                row['polarity'] or ""
            ])

            if topic_key not in topic_assertions:
                topic_assertions[topic_key] = []

            topic_assertions[topic_key].append({
                'assertion_id': row['assertion_id'],
                'predicate_label': row['canonical_label'],
                'predicate_category': row['category'],
                'normalized_object': normalized_object,
                'object_type': object_type,
                'modality': row['modality'],
                'polarity': row['polarity'],
                'confidence': row['confidence_final'],
                'event_time_utc': row['event_time_utc']
            })

        # Generate Topic nodes
        max_label_len = profile_config.topic_label_max_length

        for topic_key, assertions in sorted(topic_assertions.items()):
            if topic_key in seen_topics:
                continue
            seen_topics.add(topic_key)

            first_assertion = assertions[0]
            node_id = self._register_node(NodeType.TOPIC, topic_key)

            # Create human-readable label
            pred_label = first_assertion['predicate_label']
            obj_label = first_assertion['normalized_object']
            if len(obj_label) > 40:
                obj_label = obj_label[:37] + "..."

            label = f"{pred_label}: {obj_label}"
            if len(label) > max_label_len:
                label = label[:max_label_len-3] + "..."

            # First mention time (earliest event_time_utc)
            first_mention_utc = min(a['event_time_utc'] for a in assertions if a['event_time_utc'])

            metadata = {
                "topic_key": topic_key,
                "predicate_label": first_assertion['predicate_label'],
                "predicate_category": first_assertion['predicate_category'],
                "normalized_object": first_assertion['normalized_object'],
                "object_type": first_assertion['object_type'],
                "modality": first_assertion['modality'],
                "polarity": first_assertion['polarity'],
                "assertion_count": len(assertions),
                "first_mention_utc": first_mention_utc,
                "avg_confidence": sum(a['confidence'] or 0.5 for a in assertions) / len(assertions)
            }

            # Store in topic_registry for later use
            self.topic_registry[topic_key] = {
                **metadata,
                'node_id': node_id,
                'assertion_ids': [a['assertion_id'] for a in assertions]
            }

            nodes.append(GraphNode(
                node_id=node_id,
                node_type=NodeType.TOPIC,
                source_id=topic_key,
                label=label,
                metadata_json=self._make_metadata(metadata)
            ))

            # Generate TOPIC_SUPPORTED_BY edges
            for rank, assertion in enumerate(assertions):
                assertion_node_id = self.get_node_id(NodeType.ASSERTION, assertion['assertion_id'])
                if assertion_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.TOPIC_SUPPORTED_BY, node_id, assertion_node_id
                    )
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.TOPIC_SUPPORTED_BY,
                        src_node_id=node_id,
                        dst_node_id=assertion_node_id,
                        metadata_json=self._make_metadata({
                            "support_rank": rank,
                            "event_time_utc": assertion['event_time_utc']
                        })
                    ))

        logger.info(f"Generated {len(nodes)} Topic nodes and {len(edges)} TOPIC_SUPPORTED_BY edges")
        return nodes, edges


# ===| EDGE GENERATORS |===

class EdgeGenerator:
    """Generates graph edges from source data."""

    def __init__(
        self,
        db: Stage6Database,
        config: Stage6Config,
        id_gen: IDGenerator,
        node_gen: NodeGenerator
    ):
        self.db = db
        self.config = config
        self.id_gen = id_gen
        self.node_gen = node_gen

    def _make_metadata(self, data: Dict[str, Any]) -> str:
        """Create JCS-canonical metadata JSON."""
        return JCS.canonicalize(data)

    def _get_node_id(self, node_type: str, source_id: str) -> Optional[str]:
        """Get node_id from registry."""
        return self.node_gen.get_node_id(node_type, source_id)

    def generate_assertion_semantic_edges(self) -> List[GraphEdge]:
        """Generate HAS_SUBJECT, HAS_PREDICATE, HAS_OBJECT edges."""
        logger.info("Generating assertion semantic edges")
        edges = []

        rows = self.db.fetchall("""
            SELECT assertion_id, subject_entity_id, subject_detection_tier,
                   predicate_id, object_entity_id, object_detection_tier,
                   object_value_type, object_value
            FROM assertions
            ORDER BY assertion_id ASC
        """)

        for row in rows:
            assertion_node_id = self._get_node_id(NodeType.ASSERTION, row['assertion_id'])
            if not assertion_node_id:
                continue

            # HAS_SUBJECT
            subject_node_id = self._get_node_id(NodeType.ENTITY, row['subject_entity_id'])
            if subject_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.HAS_SUBJECT, assertion_node_id, subject_node_id
                )
                metadata = {}
                if self.config.include_detection_tier_metadata and row['subject_detection_tier']:
                    metadata["detection_tier"] = row['subject_detection_tier']

                edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.HAS_SUBJECT,
                    src_node_id=assertion_node_id,
                    dst_node_id=subject_node_id,
                    metadata_json=self._make_metadata(metadata) if metadata else None
                ))

            # HAS_PREDICATE
            predicate_node_id = self._get_node_id(NodeType.PREDICATE, row['predicate_id'])
            if predicate_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.HAS_PREDICATE, assertion_node_id, predicate_node_id
                )
                edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.HAS_PREDICATE,
                    src_node_id=assertion_node_id,
                    dst_node_id=predicate_node_id,
                    metadata_json=None
                ))

            # HAS_OBJECT (entity or value)
            if row['object_entity_id']:
                object_node_id = self._get_node_id(NodeType.ENTITY, row['object_entity_id'])
                if object_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.HAS_OBJECT, assertion_node_id, object_node_id
                    )
                    metadata = {}
                    if self.config.include_detection_tier_metadata and row['object_detection_tier']:
                        metadata["detection_tier"] = row['object_detection_tier']

                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.HAS_OBJECT,
                        src_node_id=assertion_node_id,
                        dst_node_id=object_node_id,
                        metadata_json=self._make_metadata(metadata) if metadata else None
                    ))
            elif row['object_value_type'] and row['object_value']:
                value_source_id = HashUtils.sha256_jcs([
                    row['object_value_type'], row['object_value']
                ])
                value_node_id = self._get_node_id(NodeType.VALUE, value_source_id)
                if value_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.HAS_OBJECT, assertion_node_id, value_node_id
                    )
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.HAS_OBJECT,
                        src_node_id=assertion_node_id,
                        dst_node_id=value_node_id,
                        metadata_json=None
                    ))

        logger.info(f"Generated {len(edges)} assertion semantic edges")
        return edges

    def generate_temporal_edges(self) -> List[GraphEdge]:
        """Generate VALID_IN and VALID_UNTIL_HINT edges."""
        logger.info("Generating temporal edges")
        edges = []

        rows = self.db.fetchall("""
            SELECT assertion_id, valid_from_utc, valid_to_utc, valid_until_hint_utc
            FROM assertion_temporalized
            WHERE valid_from_utc IS NOT NULL OR valid_until_hint_utc IS NOT NULL
            ORDER BY assertion_id ASC
        """)

        for row in rows:
            assertion_node_id = self._get_node_id(NodeType.ASSERTION, row['assertion_id'])
            if not assertion_node_id:
                continue

            # VALID_IN edge
            if row['valid_from_utc']:
                interval_source_id = HashUtils.sha256_jcs([
                    row['valid_from_utc'], row['valid_to_utc']
                ])
                interval_node_id = self._get_node_id(NodeType.TIME_INTERVAL, interval_source_id)
                if interval_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.VALID_IN, assertion_node_id, interval_node_id
                    )
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.VALID_IN,
                        src_node_id=assertion_node_id,
                        dst_node_id=interval_node_id,
                        metadata_json=None
                    ))

            # VALID_UNTIL_HINT edge
            if row['valid_until_hint_utc'] and not row['valid_from_utc']:
                hint_source_id = HashUtils.sha256_jcs([None, row['valid_until_hint_utc']])
                hint_node_id = self._get_node_id(NodeType.TIME_INTERVAL, hint_source_id)
                if hint_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.VALID_UNTIL_HINT, assertion_node_id, hint_node_id
                    )
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.VALID_UNTIL_HINT,
                        src_node_id=assertion_node_id,
                        dst_node_id=hint_node_id,
                        metadata_json=None
                    ))

        logger.info(f"Generated {len(edges)} temporal edges")
        return edges

    def generate_time_qualifier_edges(self) -> List[GraphEdge]:
        """Generate QUALIFIED_BY_TIME edges (optional)."""
        if not self.config.include_time_qualifier_edges:
            return []
        if not self.db.table_exists("time_mentions"):
            return []

        logger.info("Generating time qualifier edges")
        edges = []

        rows = self.db.fetchall("""
            SELECT assertion_id, temporal_qualifier_type, temporal_qualifier_id
            FROM assertions
            WHERE temporal_qualifier_id IS NOT NULL
            ORDER BY assertion_id ASC
        """)

        for row in rows:
            assertion_node_id = self._get_node_id(NodeType.ASSERTION, row['assertion_id'])
            time_node_id = self._get_node_id(NodeType.TIME_MENTION, row['temporal_qualifier_id'])

            if assertion_node_id and time_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.QUALIFIED_BY_TIME, assertion_node_id, time_node_id
                )
                metadata = {"temporal_qualifier_type": row['temporal_qualifier_type']}
                edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.QUALIFIED_BY_TIME,
                    src_node_id=assertion_node_id,
                    dst_node_id=time_node_id,
                    metadata_json=self._make_metadata(metadata)
                ))

        logger.info(f"Generated {len(edges)} time qualifier edges")
        return edges

    def generate_message_anchoring_edges(self) -> List[GraphEdge]:
        """Generate ASSERTED_IN edges (optional)."""
        if not self.config.include_message_nodes:
            return []

        logger.info("Generating message anchoring edges")
        edges = []

        rows = self.db.fetchall("""
            SELECT assertion_id, message_id
            FROM assertions
            ORDER BY assertion_id ASC
        """)

        for row in rows:
            assertion_node_id = self._get_node_id(NodeType.ASSERTION, row['assertion_id'])
            message_node_id = self._get_node_id(NodeType.MESSAGE, row['message_id'])

            if assertion_node_id and message_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.ASSERTED_IN, assertion_node_id, message_node_id
                )
                edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.ASSERTED_IN,
                    src_node_id=assertion_node_id,
                    dst_node_id=message_node_id,
                    metadata_json=None
                ))
            elif assertion_node_id and not message_node_id:
                logger.debug(f"Warning: Could not create ASSERTED_IN edge for assertion {row['assertion_id']}")

        logger.info(f"Generated {len(edges)} message anchoring edges")
        return edges

    def generate_lifecycle_edges(self) -> List[GraphEdge]:
        """Generate SUPERSEDES, RETRACTED_BY, NEGATED_BY edges."""
        logger.info("Generating lifecycle edges")
        edges = []

        rows = self.db.fetchall("""
            SELECT assertion_id, temporal_superseded_by_assertion_id,
                   retracted_by_retraction_id, negated_by_assertion_id,
                   rule_id_applied
            FROM assertion_temporalized
            WHERE temporal_superseded_by_assertion_id IS NOT NULL
               OR retracted_by_retraction_id IS NOT NULL
               OR negated_by_assertion_id IS NOT NULL
            ORDER BY assertion_id ASC
        """)

        for row in rows:
            assertion_node_id = self._get_node_id(NodeType.ASSERTION, row['assertion_id'])
            if not assertion_node_id:
                continue

            # SUPERSEDES edge (newer → older)
            if row['temporal_superseded_by_assertion_id']:
                newer_node_id = self._get_node_id(
                    NodeType.ASSERTION, row['temporal_superseded_by_assertion_id']
                )
                if newer_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.SUPERSEDES, newer_node_id, assertion_node_id
                    )
                    metadata = {}
                    if row['rule_id_applied']:
                        metadata["rule_id_applied"] = row['rule_id_applied']
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.SUPERSEDES,
                        src_node_id=newer_node_id,
                        dst_node_id=assertion_node_id,
                        metadata_json=self._make_metadata(metadata) if metadata else None
                    ))

            # RETRACTED_BY edge
            if row['retracted_by_retraction_id']:
                retraction_node_id = self._get_node_id(
                    NodeType.RETRACTION, row['retracted_by_retraction_id']
                )
                if retraction_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.RETRACTED_BY, assertion_node_id, retraction_node_id
                    )
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.RETRACTED_BY,
                        src_node_id=assertion_node_id,
                        dst_node_id=retraction_node_id,
                        metadata_json=None
                    ))

                    # Inverse: RETRACTS
                    if self.config.include_inverse_lifecycle_edges:
                        inv_edge_id = self.id_gen.generate_edge_id(
                            EdgeType.RETRACTS, retraction_node_id, assertion_node_id
                        )
                        edges.append(GraphEdge(
                            edge_id=inv_edge_id,
                            edge_type=EdgeType.RETRACTS,
                            src_node_id=retraction_node_id,
                            dst_node_id=assertion_node_id,
                            metadata_json=None
                        ))

            # NEGATED_BY edge
            if row['negated_by_assertion_id']:
                negator_node_id = self._get_node_id(
                    NodeType.ASSERTION, row['negated_by_assertion_id']
                )
                if negator_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.NEGATED_BY, assertion_node_id, negator_node_id
                    )
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.NEGATED_BY,
                        src_node_id=assertion_node_id,
                        dst_node_id=negator_node_id,
                        metadata_json=None
                    ))

                    # Inverse: NEGATES
                    if self.config.include_inverse_lifecycle_edges:
                        inv_edge_id = self.id_gen.generate_edge_id(
                            EdgeType.NEGATES, negator_node_id, assertion_node_id
                        )
                        edges.append(GraphEdge(
                            edge_id=inv_edge_id,
                            edge_type=EdgeType.NEGATES,
                            src_node_id=negator_node_id,
                            dst_node_id=assertion_node_id,
                            metadata_json=None
                        ))

        logger.info(f"Generated {len(edges)} lifecycle edges")
        return edges

    def generate_conflict_edges(self) -> List[GraphEdge]:
        """Generate HAS_CONFLICT_MEMBER and optional CONFLICTS_WITH edges."""
        if not self.db.table_exists("conflict_groups"):
            return []
        if not self.db.table_exists("conflict_members"):
            return []

        logger.info("Generating conflict edges")
        edges = []

        # Get all conflict groups with their members
        groups = self.db.fetchall("""
            SELECT cg.conflict_group_id, cg.conflict_type
            FROM conflict_groups cg
            ORDER BY cg.conflict_group_id ASC
        """)

        for group in groups:
            group_node_id = self._get_node_id(NodeType.CONFLICT_GROUP, group['conflict_group_id'])
            if not group_node_id:
                continue

            members = self.db.fetchall("""
                SELECT assertion_id FROM conflict_members
                WHERE conflict_group_id = ?
                ORDER BY assertion_id ASC
            """, (group['conflict_group_id'],))

            member_node_ids = []
            for member in members:
                member_node_id = self._get_node_id(NodeType.ASSERTION, member['assertion_id'])
                if member_node_id:
                    member_node_ids.append(member_node_id)

                    # HAS_CONFLICT_MEMBER edge
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.HAS_CONFLICT_MEMBER, group_node_id, member_node_id
                    )
                    metadata = {"conflict_type": group['conflict_type']}
                    edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.HAS_CONFLICT_MEMBER,
                        src_node_id=group_node_id,
                        dst_node_id=member_node_id,
                        metadata_json=self._make_metadata(metadata)
                    ))

            # Optional pairwise CONFLICTS_WITH edges
            if len(member_node_ids) <= self.config.conflict_pairwise_max_n:
                for i, src_id in enumerate(member_node_ids):
                    for dst_id in member_node_ids[i+1:]:
                        # Deterministic direction: lower ID → higher ID
                        if src_id > dst_id:
                            src_id, dst_id = dst_id, src_id

                        edge_id = self.id_gen.generate_edge_id(
                            EdgeType.CONFLICTS_WITH, src_id, dst_id
                        )
                        edges.append(GraphEdge(
                            edge_id=edge_id,
                            edge_type=EdgeType.CONFLICTS_WITH,
                            src_node_id=src_id,
                            dst_node_id=dst_id,
                            metadata_json=None
                        ))

        logger.info(f"Generated {len(edges)} conflict edges")
        return edges

    def generate_lexicon_edges(self) -> List[GraphEdge]:
        """Generate DERIVED_FROM_LEXICON edges (optional)."""
        if not self.config.include_lexicon_nodes:
            return []
        if not self.db.table_exists("lexicon_terms"):
            return []
        if not self.db.table_exists("entity_mentions"):
            return []

        logger.info("Generating lexicon edges")
        edges = []

        # Find entity mentions that came from lexicon detector
        rows = self.db.fetchall("""
            SELECT DISTINCT em.entity_id, lt.term_id, lt.build_id, em.confidence
            FROM entity_mentions em
            JOIN lexicon_terms lt ON em.detector LIKE 'LEXICON:%'
            JOIN entities e ON em.entity_id = e.entity_id
            WHERE em.entity_id IS NOT NULL
            ORDER BY em.entity_id ASC, lt.term_id ASC
        """)

        for row in rows:
            entity_node_id = self._get_node_id(NodeType.ENTITY, row['entity_id'])
            term_node_id = self._get_node_id(NodeType.LEXICON_TERM, row['term_id'])

            if entity_node_id and term_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.DERIVED_FROM_LEXICON, entity_node_id, term_node_id
                )
                metadata = {
                    "build_id": row['build_id'],
                    "confidence": row['confidence']
                }
                edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.DERIVED_FROM_LEXICON,
                    src_node_id=entity_node_id,
                    dst_node_id=term_node_id,
                    metadata_json=self._make_metadata(metadata)
                ))

        logger.info(f"Generated {len(edges)} lexicon edges")
        return edges


# ===| TEMPORAL PROFILE GENERATOR |===

class TemporalProfileGenerator:
    """Generates temporal profile nodes and edges."""

    def __init__(
        self,
        db: Stage6Database,
        config: Stage6Config,
        profile_config: TemporalProfileConfig,
        id_gen: IDGenerator,
        node_gen: NodeGenerator
    ):
        self.db = db
        self.config = config
        self.profile_config = profile_config
        self.id_gen = id_gen
        self.node_gen = node_gen

    def _make_metadata(self, data: Dict[str, Any]) -> str:
        """Create JCS-canonical metadata JSON with schema version."""
        data["schema_version"] = "1.0"
        return JCS.canonicalize(data)

    def resolve_focal_entities(self) -> List[Tuple[str, str]]:
        """
        Phase T0: Resolve focal entity list.
        Returns list of (entity_id, resolution_mode) tuples.
        """
        logger.info("Phase T0: Resolving focal entities")

        # If explicitly configured
        if self.profile_config.focal_entities:
            result = []
            for entity_id in self.profile_config.focal_entities:
                exists = self.db.fetchone(
                    "SELECT entity_id FROM entities WHERE entity_id = ?",
                    (entity_id,)
                )
                if exists:
                    result.append((entity_id, "config_explicit"))
            if result:
                logger.info(f"Using {len(result)} configured focal entities")
                return result

        # Try SELF entity
        self_entity = self.db.fetchone("""
            SELECT entity_id FROM entities
            WHERE entity_key = '__SELF__' AND entity_type = 'PERSON'
            ORDER BY entity_id ASC
            LIMIT 1
        """)
        if self_entity:
            logger.info(f"Using SELF entity: {self_entity['entity_id']}")
            return [(self_entity['entity_id'], "self_entity")]

        # Fallback: entity with most eligible assertions
        fallback = self.db.fetchone("""
            SELECT a.subject_entity_id, COUNT(*) as cnt
            FROM assertions a
            JOIN assertion_temporalized t ON a.assertion_id = t.assertion_id
            WHERE t.status IN ({})
              AND t.valid_from_utc IS NOT NULL
            GROUP BY a.subject_entity_id
            ORDER BY cnt DESC, a.subject_entity_id ASC
            LIMIT 1
        """.format(','.join('?' * len(self.profile_config.allowed_assertion_statuses))),
            tuple(self.profile_config.allowed_assertion_statuses)
        )

        if fallback:
            logger.info(f"Using fallback focal entity: {fallback['subject_entity_id']} ({fallback['cnt']} assertions)")
            return [(fallback['subject_entity_id'], "fallback_most_assertions")]

        logger.warning("No focal entity found")
        return []

    def partition_into_windows(
        self,
        entity_id: str,
        granularity: str
    ) -> List[TemporalWindow]:
        """
        Phase T1: Partition assertions into time windows.

        Supports three windowing modes (via profile_config.windowing_mode):

        - "mention_first": Each assertion assigned to exactly one window (first event_time month)
          This shows "what was new this month" by avoiding repeating long-lived items.

        - "mention_any": Assertions assigned to all windows where they have mentions
          Useful when tracking all activity per time period.

        - "validity_overlap": Assertions assigned using interval overlap logic
          Include assertion in window if: valid_from < window_end AND (valid_to IS NULL OR valid_to >= window_start)
          This shows all currently-valid items per window (can repeat items across many months).

        Event time is determined by profile_config.event_time_source:
        - "coalesce": COALESCE(valid_from, asserted_at, message.created_at)
        - "message_created_at", "asserted_at", "valid_from": use specific field
        """
        windowing_mode = self.profile_config.windowing_mode
        logger.info(f"Phase T1: Partitioning into {granularity} windows for entity {entity_id} (mode={windowing_mode})")

        # Build event_time SQL based on config
        event_time_source = self.profile_config.event_time_source
        if event_time_source == "message_created_at":
            event_time_sql = "m.created_at_utc"
        elif event_time_source == "asserted_at":
            event_time_sql = "a.asserted_at_utc"
        elif event_time_source == "valid_from":
            event_time_sql = "t.valid_from_utc"
        else:  # coalesce (default)
            event_time_sql = "COALESCE(t.valid_from_utc, a.asserted_at_utc, m.created_at_utc)"

        # Get eligible assertions with temporal data
        status_placeholders = ','.join('?' * len(self.profile_config.allowed_assertion_statuses))
        assertions = self.db.fetchall(f"""
            SELECT 
                a.assertion_id, 
                t.valid_from_utc, 
                t.valid_to_utc,
                {event_time_sql} as event_time_utc
            FROM assertions a
            JOIN assertion_temporalized t ON a.assertion_id = t.assertion_id
            LEFT JOIN messages m ON a.message_id = m.message_id
            WHERE a.subject_entity_id = ?
              AND t.status IN ({status_placeholders})
            ORDER BY event_time_utc ASC, a.assertion_id ASC
        """, (entity_id, *self.profile_config.allowed_assertion_statuses))

        if not assertions:
            logger.warning(f"No eligible assertions found for entity {entity_id}")
            return []

        # Parse timestamps and find range using event_time_utc for bounds
        parsed_assertions = []
        for a in assertions:
            event_time = a['event_time_utc']
            if not event_time:
                continue
            dt = TimestampUtils.parse_iso(event_time)
            if dt:
                parsed_assertions.append({
                    'assertion_id': a['assertion_id'],
                    'valid_from_utc': a['valid_from_utc'],
                    'valid_to_utc': a['valid_to_utc'],
                    'event_time_utc': event_time,
                    'event_dt': dt
                })

        if not parsed_assertions:
            return []

        parsed_assertions.sort(key=lambda x: (x['event_dt'], x['assertion_id']))
        min_dt = parsed_assertions[0]['event_dt']
        max_dt = parsed_assertions[-1]['event_dt']

        # Generate aligned windows
        windows: List[TemporalWindow] = []

        if granularity == "month":
            current = min_dt.start_of('month')
            while current <= max_dt:
                window_start = current.format(TimestampUtils.ISO_UTC_MILLIS)
                window_end = current.end_of('month').add(microseconds=1).format(TimestampUtils.ISO_UTC_MILLIS)

                window = TemporalWindow(
                    window_id="",  # Will be set from source_id
                    entity_id=entity_id,
                    granularity=granularity,
                    window_start_utc=window_start,
                    window_end_utc=window_end
                )
                window.window_id = self.id_gen.generate_node_id(
                    NodeType.TEMPORAL_WINDOW, window.source_id
                )
                windows.append(window)
                current = current.add(months=1)

        elif granularity == "quarter":
            current = min_dt.start_of('quarter')
            while current <= max_dt:
                window_start = current.format(TimestampUtils.ISO_UTC_MILLIS)
                window_end = current.end_of('quarter').add(microseconds=1).format(TimestampUtils.ISO_UTC_MILLIS)

                window = TemporalWindow(
                    window_id="",
                    entity_id=entity_id,
                    granularity=granularity,
                    window_start_utc=window_start,
                    window_end_utc=window_end
                )
                window.window_id = self.id_gen.generate_node_id(
                    NodeType.TEMPORAL_WINDOW, window.source_id
                )
                windows.append(window)
                current = current.add(months=3)

        elif granularity == "year":
            current = min_dt.start_of('year')
            while current <= max_dt:
                window_start = current.format(TimestampUtils.ISO_UTC_MILLIS)
                window_end = current.end_of('year').add(microseconds=1).format(TimestampUtils.ISO_UTC_MILLIS)

                window = TemporalWindow(
                    window_id="",
                    entity_id=entity_id,
                    granularity=granularity,
                    window_start_utc=window_start,
                    window_end_utc=window_end
                )
                window.window_id = self.id_gen.generate_node_id(
                    NodeType.TEMPORAL_WINDOW, window.source_id
                )
                windows.append(window)
                current = current.add(years=1)

        # Assign assertions to windows based on windowing mode
        if windowing_mode == WindowingMode.MENTION_FIRST:
            # Each assertion assigned to exactly one window (first mention month)
            for assertion in parsed_assertions:
                event_time = assertion['event_time_utc']
                for window in windows:
                    if window.window_start_utc <= event_time < window.window_end_utc:
                        if assertion['assertion_id'] not in window.assertion_ids:
                            window.assertion_ids.append(assertion['assertion_id'])
                        break  # Only assign to first matching window

        elif windowing_mode == WindowingMode.MENTION_ANY:
            # Assign to all windows where assertion has mentions (event_time within window)
            for assertion in parsed_assertions:
                event_time = assertion['event_time_utc']
                for window in windows:
                    if window.window_start_utc <= event_time < window.window_end_utc:
                        if assertion['assertion_id'] not in window.assertion_ids:
                            window.assertion_ids.append(assertion['assertion_id'])

        else:  # validity_overlap (default legacy behavior)
            # Interval overlap logic
            for assertion in parsed_assertions:
                valid_from = assertion['valid_from_utc']
                valid_to = assertion['valid_to_utc']
                event_time = assertion['event_time_utc']

                for window in windows:
                    # Include assertion in window if valid_from < window_end AND (valid_to IS NULL OR valid_to >= window_start)
                    if valid_from:
                        # Has explicit valid time - use interval overlap
                        if valid_from < window.window_end_utc and (valid_to is None or valid_to >= window.window_start_utc):
                            if assertion['assertion_id'] not in window.assertion_ids:
                                window.assertion_ids.append(assertion['assertion_id'])
                    else:
                        # No valid_from - use event_time_utc within window
                        if window.window_start_utc <= event_time < window.window_end_utc:
                            if assertion['assertion_id'] not in window.assertion_ids:
                                window.assertion_ids.append(assertion['assertion_id'])

        # Filter windows with minimum assertions
        windows = [w for w in windows if len(w.assertion_ids) >= self.profile_config.window_min_assertions]

        logger.info(f"Created {len(windows)} windows with {sum(len(w.assertion_ids) for w in windows)} assertion assignments (mode={windowing_mode})")
        return windows

    def compute_member_salience(
        self,
        window: TemporalWindow,
        member_window_counts: Optional[Dict[Tuple[str, str], int]] = None,
        total_windows: int = 1
    ) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], List[str]]]:
        """
        Phase T2: Compute salience for members (entities, values, and topics) in window.

        Returns tuple of:
          - dict of (node_id, node_type) -> salience
          - dict of (node_id, node_type) -> [assertion_ids] (for evidence edges)

        Salience formula:
          base_salience = frequency * w_f + recency * w_r + confidence * w_c
          evergreen_penalty = log((1 + total_windows) / (1 + windows_with_member))
          final_salience = base_salience * (1 - penalty_strength + penalty_strength * evergreen_penalty / log(1 + total_windows))

        Args:
            window: The temporal window being processed
            member_window_counts: Dict mapping (node_id, node_type) -> number of windows member appears in
            total_windows: Total number of windows for IDF computation
        """
        if not window.assertion_ids:
            return {}, {}

        placeholders = ','.join('?' * len(window.assertion_ids))

        # Get assertions with their objects
        assertions = self.db.fetchall(f"""
            SELECT a.assertion_id, a.predicate_id, p.canonical_label,
                   a.object_entity_id, e.canonical_name as object_entity_name,
                   a.object_value_type, a.object_value, a.object_signature,
                   a.modality, a.polarity,
                   COALESCE(t.valid_from_utc, a.asserted_at_utc) as event_time_utc,
                   a.confidence_final
            FROM assertions a
            JOIN assertion_temporalized t ON a.assertion_id = t.assertion_id
            JOIN predicates p ON a.predicate_id = p.predicate_id
            LEFT JOIN entities e ON a.object_entity_id = e.entity_id
            WHERE a.assertion_id IN ({placeholders})
            ORDER BY a.assertion_id ASC
        """, tuple(window.assertion_ids))

        # Collect member mentions - support entities, values, AND topics
        member_mentions: Dict[Tuple[str, str], List[Tuple[str, float, str]]] = {}  # (source_id, type) -> [(event_time, confidence, assertion_id)]

        for a in assertions:
            # Check for Topic members first (if enabled)
            if self.config.include_topic_nodes and self.config.use_topics_as_primary_members:
                # Build topic key
                if a['object_entity_id']:
                    normalized_object = a['object_entity_name'] or a['object_entity_id']
                elif a['object_value']:
                    normalized_object = a['object_value']
                else:
                    normalized_object = None

                if normalized_object:
                    topic_key = HashUtils.sha256_jcs([
                        a['predicate_id'],
                        normalized_object,
                        a['modality'] or "",
                        a['polarity'] or ""
                    ])

                    # Check if this topic exists in registry
                    topic_node_id = self.node_gen.get_node_id(NodeType.TOPIC, topic_key)
                    if topic_node_id:
                        key = (topic_key, NodeType.TOPIC)
                        if key not in member_mentions:
                            member_mentions[key] = []
                        member_mentions[key].append((
                            a['event_time_utc'],
                            a['confidence_final'] or 0.5,
                            a['assertion_id']
                        ))
                        continue  # Topic found, skip entity/value extraction

            # Fall back to entity/value members
            if a['object_entity_id']:
                key = (a['object_entity_id'], NodeType.ENTITY)
            elif a['object_value_type'] and a['object_value']:
                # Skip value members if disabled
                if not self.config.include_value_members_in_profiles:
                    continue
                value_source_id = HashUtils.sha256_jcs([a['object_value_type'], a['object_value']])
                key = (value_source_id, NodeType.VALUE)
            else:
                continue

            if key not in member_mentions:
                member_mentions[key] = []
            member_mentions[key].append((
                a['event_time_utc'],
                a['confidence_final'] or 0.5,
                a['assertion_id']
            ))

        # Compute salience for each member
        total_assertions = len(window.assertion_ids)
        weights = self.profile_config.salience_weights
        penalty_strength = self.profile_config.evergreen_penalty_strength

        member_salience: Dict[Tuple[str, str], float] = {}
        member_assertions: Dict[Tuple[str, str], List[str]] = {}  # For evidence edges

        import math
        max_idf = math.log(1 + total_windows) if total_windows > 0 else 1.0

        for (source_id, member_type), mentions in member_mentions.items():
            # Frequency
            frequency = len(mentions) / total_assertions if total_assertions > 0 else 0

            # Recency (simple: more recent = higher)
            recency_scores = []
            for event_time, _, _ in mentions:
                if event_time:
                    dt = TimestampUtils.parse_iso(event_time)
                    window_end = TimestampUtils.parse_iso(window.window_end_utc)
                    if dt and window_end:
                        days_ago = (window_end - dt).days
                        recency_scores.append(1.0 / (1.0 + days_ago * 0.1))
            recency = max(recency_scores) if recency_scores else 0

            # Confidence (mean)
            confidence = sum(c for _, c, _ in mentions) / len(mentions) if mentions else 0

            # Base weighted combination
            base_salience = (
                weights.get("frequency", 0.4) * frequency +
                weights.get("recency", 0.2) * recency +
                weights.get("confidence", 0.4) * confidence
            )

            # Get node_id
            node_id = self.node_gen.get_node_id(member_type, source_id)
            if not node_id:
                continue

            # Apply evergreen penalty (IDF-like)
            if penalty_strength > 0 and member_window_counts and total_windows > 1:
                windows_with_member = member_window_counts.get((node_id, member_type), 1)
                # IDF formula: log((1 + total_windows) / (1 + windows_with_member))
                # Normalized to [0, 1] range
                idf = math.log((1 + total_windows) / (1 + windows_with_member))
                normalized_idf = idf / max_idf if max_idf > 0 else 1.0

                # Blend: penalty_strength=0 means no penalty, penalty_strength=1 means full IDF
                salience = base_salience * (1 - penalty_strength + penalty_strength * normalized_idf)
            else:
                salience = base_salience

            member_salience[(node_id, member_type)] = salience
            # Collect assertion_ids for evidence edges (sorted for determinism)
            assertion_ids = sorted(set(aid for _, _, aid in mentions))
            member_assertions[(node_id, member_type)] = assertion_ids

        return member_salience, member_assertions

    def compute_member_window_counts(
        self,
        windows: List[TemporalWindow]
    ) -> Dict[Tuple[str, str], int]:
        """
        Pre-compute how many windows each member appears in for evergreen penalty.

        Returns dict of (node_id, node_type) -> count of windows containing this member.
        """
        member_counts: Dict[Tuple[str, str], int] = {}

        for window in windows:
            if not window.assertion_ids:
                continue

            placeholders = ','.join('?' * len(window.assertion_ids))

            # Get objects in this window
            rows = self.db.fetchall(f"""
                SELECT DISTINCT a.predicate_id, a.object_entity_id, 
                       a.object_value_type, a.object_value,
                       a.modality, a.polarity,
                       e.canonical_name as object_entity_name
                FROM assertions a
                LEFT JOIN entities e ON a.object_entity_id = e.entity_id
                WHERE a.assertion_id IN ({placeholders})
            """, tuple(window.assertion_ids))

            window_members: Set[Tuple[str, str]] = set()

            for row in rows:
                # Check for Topic
                if self.config.include_topic_nodes and self.config.use_topics_as_primary_members:
                    if row['object_entity_id']:
                        normalized_object = row['object_entity_name'] or row['object_entity_id']
                    elif row['object_value']:
                        normalized_object = row['object_value']
                    else:
                        normalized_object = None

                    if normalized_object:
                        topic_key = HashUtils.sha256_jcs([
                            row['predicate_id'],
                            normalized_object,
                            row['modality'] or "",
                            row['polarity'] or ""
                        ])
                        topic_node_id = self.node_gen.get_node_id(NodeType.TOPIC, topic_key)
                        if topic_node_id:
                            window_members.add((topic_node_id, NodeType.TOPIC))
                            continue

                # Entity/value members
                if row['object_entity_id']:
                    node_id = self.node_gen.get_node_id(NodeType.ENTITY, row['object_entity_id'])
                    if node_id:
                        window_members.add((node_id, NodeType.ENTITY))
                elif row['object_value_type'] and row['object_value']:
                    if self.config.include_value_members_in_profiles:
                        value_source_id = HashUtils.sha256_jcs([row['object_value_type'], row['object_value']])
                        node_id = self.node_gen.get_node_id(NodeType.VALUE, value_source_id)
                        if node_id:
                            window_members.add((node_id, NodeType.VALUE))

            # Increment counts for members in this window
            for member_key in window_members:
                member_counts[member_key] = member_counts.get(member_key, 0) + 1

        return member_counts

    def classify_member(
        self,
        node_id: str,
        node_type: str,
        window: TemporalWindow,
        entity_id: str
    ) -> Tuple[str, int]:
        """
        Phase T3: Classify a member into a semantic category.
        Returns (category_key, classification_tier).

        Supports Entity, Value, and Topic node types.

        Priority order:
        1. predicate_cluster_categories (explicit config mapping)
        2. predicates.category (if matches configured category_keys)
        3. strong_rule_keywords
        4. modality hints (intention→plan, preference→interest)
        5. fallback to "unclassified"

        Tie-breaking: highest vote count, then alphabetical ascending.
        """
        # Get predicates linking focal entity to this member in this window
        if not window.assertion_ids:
            return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

        placeholders = ','.join('?' * len(window.assertion_ids))

        # Handle Topic nodes - extract category from topic_registry
        if node_type == NodeType.TOPIC:
            # Find topic_key from node_id
            topic_key = None
            for key, nid in self.node_gen.node_registry.items():
                if nid == node_id and key.startswith(f"{NodeType.TOPIC}:"):
                    topic_key = key.split(":", 1)[1]
                    break

            if topic_key and topic_key in self.node_gen.topic_registry:
                topic_data = self.node_gen.topic_registry[topic_key]
                # Use predicate_category from topic if it matches configured categories
                pred_category = topic_data.get('predicate_category')
                valid_categories = set(self.profile_config.category_keys)
                if pred_category and pred_category.lower() in valid_categories:
                    return (pred_category.lower(), int(ClassificationTier.KEYWORD_RULES))

                # Use modality hints
                modality = topic_data.get('modality')
                if modality == "intention" and "plan" in valid_categories:
                    return ("plan", int(ClassificationTier.KEYWORD_RULES))
                elif modality == "preference" and "interest" in valid_categories:
                    return ("interest", int(ClassificationTier.KEYWORD_RULES))
                elif modality == "desire" and "desire" in valid_categories:
                    return ("desire", int(ClassificationTier.KEYWORD_RULES))

            return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

        if node_type == NodeType.ENTITY:
            # Find entity source_id from node_id (reverse lookup)
            entity_source = None
            for key, nid in self.node_gen.node_registry.items():
                if nid == node_id and key.startswith(f"{NodeType.ENTITY}:"):
                    entity_source = key.split(":", 1)[1]
                    break

            if not entity_source:
                return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

            predicates = self.db.fetchall(f"""
                SELECT DISTINCT p.predicate_id, p.canonical_label, p.canonical_label_norm, 
                       p.category as predicate_category, a.modality
                FROM assertions a
                JOIN predicates p ON a.predicate_id = p.predicate_id
                WHERE a.assertion_id IN ({placeholders})
                  AND a.object_entity_id = ?
                ORDER BY p.canonical_label_norm ASC
            """, (*window.assertion_ids, entity_source))

        elif node_type == NodeType.VALUE:
            # Use value_source_map for efficient O(1) lookup
            value_source_id = None
            for key, nid in self.node_gen.node_registry.items():
                if nid == node_id and key.startswith(f"{NodeType.VALUE}:"):
                    value_source_id = key.split(":", 1)[1]
                    break

            if not value_source_id:
                return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

            # Direct lookup using value_source_map (O(1) instead of O(n))
            value_info = self.node_gen.value_source_map.get(value_source_id)
            if not value_info:
                return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

            value_type, value = value_info

            # Query predicates for this specific value only
            predicates = self.db.fetchall(f"""
                SELECT DISTINCT p.predicate_id, p.canonical_label, p.canonical_label_norm,
                       p.category as predicate_category, a.modality
                FROM assertions a
                JOIN predicates p ON a.predicate_id = p.predicate_id
                WHERE a.assertion_id IN ({placeholders})
                  AND a.object_value_type = ?
                  AND a.object_value = ?
                ORDER BY p.canonical_label_norm ASC
            """, (*window.assertion_ids, value_type, value))
        else:
            return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

        if not predicates:
            return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

        predicate_ids = [p['predicate_id'] for p in predicates]
        predicate_labels = [p['canonical_label_norm'] for p in predicates]
        predicate_categories = [p['predicate_category'] for p in predicates if p['predicate_category']]
        modalities = [p['modality'] for p in predicates if p['modality']]

        category_votes: Dict[str, int] = {}
        valid_categories = set(self.profile_config.category_keys)

        # Priority 1: predicate_cluster_categories (explicit config mapping)
        predicate_cluster_map = self.profile_config.predicate_cluster_categories
        for pred_id in predicate_ids:
            for category, predicate_list in predicate_cluster_map.items():
                if pred_id in predicate_list and category in valid_categories:
                    category_votes[category] = category_votes.get(category, 0) + 10  # High weight for explicit mapping

        # If explicit mapping gave votes, return immediately (highest priority)
        if category_votes:
            winner = self._select_category_winner(category_votes)
            return (winner, int(ClassificationTier.KEYWORD_RULES))

        # Priority 2: predicates.category (if matches configured category_keys)
        for pred_category in predicate_categories:
            if pred_category and pred_category.lower() in valid_categories:
                category_votes[pred_category.lower()] = category_votes.get(pred_category.lower(), 0) + 5

        if category_votes:
            winner = self._select_category_winner(category_votes)
            return (winner, int(ClassificationTier.KEYWORD_RULES))

        # Priority 3: strong_rule_keywords
        keyword_rules = self.profile_config.strong_rule_keywords
        for label in predicate_labels:
            for category, keywords in keyword_rules.items():
                if category not in valid_categories:
                    continue
                for keyword in keywords:
                    if keyword.lower() in label.lower():
                        category_votes[category] = category_votes.get(category, 0) + 1

        if category_votes:
            winner = self._select_category_winner(category_votes)
            return (winner, int(ClassificationTier.KEYWORD_RULES))

        # Priority 4: Modality-based hints
        for modality in modalities:
            if modality == "intention" and "plan" in valid_categories:
                category_votes["plan"] = category_votes.get("plan", 0) + 1
            elif modality == "preference" and "interest" in valid_categories:
                category_votes["interest"] = category_votes.get("interest", 0) + 1
            elif modality == "desire" and "desire" in valid_categories:
                category_votes["desire"] = category_votes.get("desire", 0) + 1
            elif modality == "state" and "interest" in valid_categories:
                # States often indicate interests
                category_votes["interest"] = category_votes.get("interest", 0) + 1

        if category_votes:
            winner = self._select_category_winner(category_votes)
            return (winner, int(ClassificationTier.KEYWORD_RULES))

        # Priority 5: Fallback
        return ("unclassified", int(ClassificationTier.UNCLASSIFIED))

    def _select_category_winner(self, votes: Dict[str, int]) -> str:
        """
        Select winning category from votes.
        Tie-breaking: highest vote count, then alphabetical ascending.
        """
        if not votes:
            return "unclassified"

        # Sort by (-count, alphabetical) and return first
        sorted_categories = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
        return sorted_categories[0][0]

    def _generate_member_evidence_edges(
        self,
        member_node_id: str,
        member_type: str,
        member_assertions: Dict[Tuple[str, str], List[str]],
        window: TemporalWindow
    ) -> List[GraphEdge]:
        """
        Generate MEMBER_EVIDENCED_BY edges from member to assertions that mention it.

        Links members to assertions within the window that evidence their presence,
        capped by member_evidence_edge_cap_per_window for clutter reduction.

        Args:
            member_node_id: Node ID of the member (entity or value)
            member_type: NodeType.ENTITY or NodeType.VALUE
            member_assertions: Dict mapping (node_id, type) -> [assertion_ids]
            window: The temporal window being processed

        Returns:
            List of MEMBER_EVIDENCED_BY edges
        """
        edges: List[GraphEdge] = []

        key = (member_node_id, member_type)
        assertion_ids = member_assertions.get(key, [])

        if not assertion_ids:
            return edges

        # Cap the number of evidence edges per member per window
        cap = self.config.member_evidence_edge_cap_per_window
        capped_assertion_ids = assertion_ids[:cap]

        for rank, assertion_id in enumerate(capped_assertion_ids):
            assertion_node_id = self.node_gen.get_node_id(NodeType.ASSERTION, assertion_id)
            if assertion_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.MEMBER_EVIDENCED_BY, member_node_id, assertion_node_id
                )
                edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.MEMBER_EVIDENCED_BY,
                    src_node_id=member_node_id,
                    dst_node_id=assertion_node_id,
                    metadata_json=self._make_metadata({
                        "evidence_rank": rank,
                        "window_start_utc": window.window_start_utc,
                        "total_evidence_count": len(assertion_ids),
                        "capped": len(assertion_ids) > cap
                    })
                ))

        return edges

    def generate_profile_nodes_and_edges(
        self,
        focal_entities: List[Tuple[str, str]]
    ) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        Phase T4: Generate all temporal profile nodes and edges.

        Includes Topic node generation (if enabled) and evergreen penalty computation.
        """
        logger.info("Phase T4: Generating profile nodes and edges")

        all_nodes: List[GraphNode] = []
        all_edges: List[GraphEdge] = []

        # Generate category nodes first if enabled
        if self.config.include_category_nodes:
            category_nodes = self.node_gen.generate_semantic_category_nodes(
                self.profile_config.category_keys
            )
            all_nodes.extend(category_nodes)

        for entity_id, resolution_mode in focal_entities:
            # Generate Topic nodes for this focal entity (if enabled)
            if self.config.include_topic_nodes and self.profile_config.enable_topic_nodes:
                topic_nodes, topic_edges = self.node_gen.generate_topic_nodes(
                    entity_id, self.profile_config
                )
                all_nodes.extend(topic_nodes)
                all_edges.extend(topic_edges)

            # Partition into windows
            windows = self.partition_into_windows(
                entity_id, self.profile_config.granularity
            )

            if not windows:
                continue

            # Pre-compute member window counts for evergreen penalty
            member_window_counts = {}
            if self.profile_config.evergreen_penalty_strength > 0:
                member_window_counts = self.compute_member_window_counts(windows)
            total_windows = len(windows)

            # Create profile node
            profile_source_id = HashUtils.sha256_jcs([
                entity_id,
                self.profile_config.granularity,
                self.profile_config.profile_version
            ])
            profile_node_id = self.id_gen.generate_node_id(
                NodeType.TEMPORAL_PROFILE, profile_source_id
            )
            self.node_gen.node_registry[f"{NodeType.TEMPORAL_PROFILE}:{profile_source_id}"] = profile_node_id

            # Compute date range
            date_range = (
                windows[0].window_start_utc[:10],
                windows[-1].window_end_utc[:10]
            )

            profile_metadata = {
                "entity_id": entity_id,
                "granularity": self.profile_config.granularity,
                "windowing_mode": self.profile_config.windowing_mode,
                "evergreen_penalty_strength": self.profile_config.evergreen_penalty_strength,
                "window_count": len(windows),
                "total_assertions": sum(len(w.assertion_ids) for w in windows),
                "date_range": date_range,
                "profile_version": self.profile_config.profile_version,
                "resolution_mode": resolution_mode,
                "topics_enabled": self.config.include_topic_nodes and self.profile_config.enable_topic_nodes,
                "topic_count": len(self.node_gen.topic_registry) if self.config.include_topic_nodes else 0
            }

            # Get entity label
            entity_row = self.db.fetchone(
                "SELECT canonical_name FROM entities WHERE entity_id = ?",
                (entity_id,)
            )
            entity_label = entity_row['canonical_name'] if entity_row else entity_id

            all_nodes.append(GraphNode(
                node_id=profile_node_id,
                node_type=NodeType.TEMPORAL_PROFILE,
                source_id=profile_source_id,
                label=f"Profile: {entity_label}",
                metadata_json=self._make_metadata(profile_metadata)
            ))

            # HAS_PROFILE edge
            entity_node_id = self.node_gen.get_node_id(NodeType.ENTITY, entity_id)
            if entity_node_id:
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.HAS_PROFILE, entity_node_id, profile_node_id
                )
                all_edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.HAS_PROFILE,
                    src_node_id=entity_node_id,
                    dst_node_id=profile_node_id,
                    metadata_json=self._make_metadata({
                        "granularity": self.profile_config.granularity,
                        "resolution_mode": resolution_mode
                    })
                ))

            # Process each window
            prev_window_node_id = None
            prev_clusters_by_category: Dict[str, str] = {}  # category -> cluster_node_id

            for window_idx, window in enumerate(windows):
                # Window node
                window_node_id = window.window_id
                self.node_gen.node_registry[f"{NodeType.TEMPORAL_WINDOW}:{window.source_id}"] = window_node_id

                # Compute member salience and assertion mapping (for evidence edges)
                member_salience, member_assertions = self.compute_member_salience(
                    window, member_window_counts, total_windows
                )

                # Classify members into clusters
                clusters: Dict[str, List[Tuple[str, str, float, int]]] = {}  # category -> [(node_id, type, salience, tier)]

                for (member_node_id, member_type), salience in member_salience.items():
                    if salience > 0:
                        category, tier = self.classify_member(
                            member_node_id, member_type, window, entity_id
                        )
                        if category not in clusters:
                            clusters[category] = []
                        clusters[category].append((member_node_id, member_type, salience, tier))

                # Sort members within each cluster by salience
                for category in clusters:
                    clusters[category].sort(key=lambda x: (-x[2], x[0]))

                window_metadata = {
                    "entity_id": entity_id,
                    "granularity": self.profile_config.granularity,
                    "window_start_utc": window.window_start_utc,
                    "window_end_utc": window.window_end_utc,
                    "assertion_count": len(window.assertion_ids),
                    "cluster_count": len(clusters),
                    "top_members": [
                        {"node_id": m[0], "category": cat, "salience": m[2]}
                        for cat, members in clusters.items()
                        for m in members[:3]
                    ][:self.config.window_top_n_members]
                }

                window_label = TimestampUtils.format_window_label(
                    window.window_start_utc, self.profile_config.granularity
                )

                all_nodes.append(GraphNode(
                    node_id=window_node_id,
                    node_type=NodeType.TEMPORAL_WINDOW,
                    source_id=window.source_id,
                    label=window_label,
                    metadata_json=self._make_metadata(window_metadata)
                ))

                # HAS_WINDOW edge
                edge_id = self.id_gen.generate_edge_id(
                    EdgeType.HAS_WINDOW, profile_node_id, window_node_id
                )
                all_edges.append(GraphEdge(
                    edge_id=edge_id,
                    edge_type=EdgeType.HAS_WINDOW,
                    src_node_id=profile_node_id,
                    dst_node_id=window_node_id,
                    metadata_json=self._make_metadata({"window_index": window_idx})
                ))

                # WINDOW_PRECEDES edge
                if self.config.include_window_sequence_edges and prev_window_node_id:
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.WINDOW_PRECEDES, prev_window_node_id, window_node_id
                    )
                    all_edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.WINDOW_PRECEDES,
                        src_node_id=prev_window_node_id,
                        dst_node_id=window_node_id,
                        metadata_json=self._make_metadata({"granularity": self.profile_config.granularity})
                    ))

                # Optional WINDOW_INCLUDES edges
                if self.config.include_window_assertion_edges:
                    for assertion_id in window.assertion_ids:
                        assertion_node_id = self.node_gen.get_node_id(NodeType.ASSERTION, assertion_id)
                        if assertion_node_id:
                            edge_id = self.id_gen.generate_edge_id(
                                EdgeType.WINDOW_INCLUDES, window_node_id, assertion_node_id
                            )
                            all_edges.append(GraphEdge(
                                edge_id=edge_id,
                                edge_type=EdgeType.WINDOW_INCLUDES,
                                src_node_id=window_node_id,
                                dst_node_id=assertion_node_id,
                                metadata_json=None
                            ))

                # Create cluster nodes and edges
                current_clusters_by_category: Dict[str, str] = {}

                for category, members in clusters.items():
                    if not members:
                        continue

                    cluster_source_id = HashUtils.sha256_jcs([
                        window.source_id, category, 0
                    ])
                    cluster_node_id = self.id_gen.generate_node_id(
                        NodeType.SEMANTIC_CLUSTER, cluster_source_id
                    )
                    self.node_gen.node_registry[f"{NodeType.SEMANTIC_CLUSTER}:{cluster_source_id}"] = cluster_node_id
                    current_clusters_by_category[category] = cluster_node_id

                    # Tier distribution (convert enum/int keys to strings for JCS)
                    tier_dist = {}
                    for _, _, _, tier in members:
                        tier_key = str(int(tier)) if isinstance(tier, (int, ClassificationTier)) else str(tier)
                        tier_dist[tier_key] = tier_dist.get(tier_key, 0) + 1

                    cluster_metadata = {
                        "category_key": category,
                        "member_count": len(members),
                        "avg_salience": sum(m[2] for m in members) / len(members) if members else 0,
                        "tier_distribution": tier_dist
                    }

                    all_nodes.append(GraphNode(
                        node_id=cluster_node_id,
                        node_type=NodeType.SEMANTIC_CLUSTER,
                        source_id=cluster_source_id,
                        label=f"{category.title()} ({len(members)})",
                        metadata_json=self._make_metadata(cluster_metadata)
                    ))

                    # HAS_CLUSTER edge
                    edge_id = self.id_gen.generate_edge_id(
                        EdgeType.HAS_CLUSTER, window_node_id, cluster_node_id
                    )
                    all_edges.append(GraphEdge(
                        edge_id=edge_id,
                        edge_type=EdgeType.HAS_CLUSTER,
                        src_node_id=window_node_id,
                        dst_node_id=cluster_node_id,
                        metadata_json=None
                    ))

                    # CLUSTER_OF_CATEGORY edge
                    if self.config.include_category_nodes:
                        category_node_id = self.node_gen.get_node_id(
                            NodeType.SEMANTIC_CATEGORY, category
                        )
                        if category_node_id:
                            edge_id = self.id_gen.generate_edge_id(
                                EdgeType.CLUSTER_OF_CATEGORY, cluster_node_id, category_node_id
                            )
                            all_edges.append(GraphEdge(
                                edge_id=edge_id,
                                edge_type=EdgeType.CLUSTER_OF_CATEGORY,
                                src_node_id=cluster_node_id,
                                dst_node_id=category_node_id,
                                metadata_json=None
                            ))

                    # CLUSTER_CONTAINS edges (limited by cluster_top_n_members for clutter reduction)
                    members_to_include = members[:self.config.cluster_top_n_members]
                    for rank, (member_node_id, member_type, salience, tier) in enumerate(members_to_include):
                        edge_id = self.id_gen.generate_edge_id(
                            EdgeType.CLUSTER_CONTAINS, cluster_node_id, member_node_id
                        )
                        all_edges.append(GraphEdge(
                            edge_id=edge_id,
                            edge_type=EdgeType.CLUSTER_CONTAINS,
                            src_node_id=cluster_node_id,
                            dst_node_id=member_node_id,
                            metadata_json=self._make_metadata({
                                "rank": rank,
                                "salience": salience,
                                "classification_tier": int(tier) if isinstance(tier, (int, ClassificationTier)) else tier
                            })
                        ))

                        # MEMBER_EVIDENCED_BY edges (evidence links)
                        if self.config.include_member_evidence_edges:
                            evidence_edges = self._generate_member_evidence_edges(
                                member_node_id, member_type, member_assertions, window
                            )
                            all_edges.extend(evidence_edges)

                    # EVOLVES_TO edges (cluster evolution)
                    if self.config.include_cluster_evolution_edges:
                        prev_cluster_id = prev_clusters_by_category.get(category)
                        if prev_cluster_id:
                            edge_id = self.id_gen.generate_edge_id(
                                EdgeType.EVOLVES_TO, prev_cluster_id, cluster_node_id
                            )
                            all_edges.append(GraphEdge(
                                edge_id=edge_id,
                                edge_type=EdgeType.EVOLVES_TO,
                                src_node_id=prev_cluster_id,
                                dst_node_id=cluster_node_id,
                                metadata_json=None
                            ))

                # WINDOW_TOP_MEMBER edges
                if self.config.include_window_top_edges:
                    all_members = [
                        (node_id, node_type, salience, cat)
                        for cat, members in clusters.items()
                        for node_id, node_type, salience, _ in members
                    ]
                    all_members.sort(key=lambda x: (-x[2], x[0]))

                    for rank, (member_node_id, member_type, salience, category) in enumerate(
                        all_members[:self.config.window_top_n_members]
                    ):
                        edge_id = self.id_gen.generate_edge_id(
                            EdgeType.WINDOW_TOP_MEMBER, window_node_id, member_node_id
                        )
                        all_edges.append(GraphEdge(
                            edge_id=edge_id,
                            edge_type=EdgeType.WINDOW_TOP_MEMBER,
                            src_node_id=window_node_id,
                            dst_node_id=member_node_id,
                            metadata_json=self._make_metadata({
                                "rank": rank,
                                "salience": salience,
                                "member_node_type": member_type,
                                "category_key": category
                            })
                        ))

                    # WINDOW_TOP_CLUSTER edges
                    cluster_scores = [
                        (cluster_id, cat, sum(m[2] for m in members) / len(members) if members else 0)
                        for cat, members in clusters.items()
                        for cluster_id in [current_clusters_by_category.get(cat)]
                        if cluster_id
                    ]
                    cluster_scores.sort(key=lambda x: (-x[2], x[0]))

                    for rank, (cluster_id, category, score) in enumerate(
                        cluster_scores[:self.config.window_top_n_clusters]
                    ):
                        edge_id = self.id_gen.generate_edge_id(
                            EdgeType.WINDOW_TOP_CLUSTER, window_node_id, cluster_id
                        )
                        all_edges.append(GraphEdge(
                            edge_id=edge_id,
                            edge_type=EdgeType.WINDOW_TOP_CLUSTER,
                            src_node_id=window_node_id,
                            dst_node_id=cluster_id,
                            metadata_json=self._make_metadata({
                                "rank": rank,
                                "score": score,
                                "category_key": category
                            })
                        ))

                prev_window_node_id = window_node_id
                prev_clusters_by_category = current_clusters_by_category

        logger.info(f"Generated {len(all_nodes)} profile nodes and {len(all_edges)} profile edges")
        return all_nodes, all_edges


# ===| DATA QUALITY CHECK |===

class DataQualityChecker:
    """Performs data quality checks before graph materialization."""

    def __init__(self, db: Stage6Database):
        self.db = db

    def run_checks(self) -> DataQualityReport:
        """Run all data quality checks."""
        logger.info("Running data quality checks")
        report = DataQualityReport()

        # Entity count
        result = self.db.fetchone("SELECT COUNT(*) as cnt FROM entities")
        report.entity_count = result['cnt'] if result else 0

        # Predicate count
        result = self.db.fetchone("SELECT COUNT(*) as cnt FROM predicates")
        report.predicate_count = result['cnt'] if result else 0

        # Assertion count
        result = self.db.fetchone("SELECT COUNT(*) as cnt FROM assertions")
        report.assertion_count = result['cnt'] if result else 0

        # Temporalized count and coverage
        result = self.db.fetchone("SELECT COUNT(*) as cnt FROM assertion_temporalized")
        report.temporalized_count = result['cnt'] if result else 0
        report.temporalized_coverage = (
            report.temporalized_count / report.assertion_count
            if report.assertion_count > 0 else 0
        )

        # Status distribution
        rows = self.db.fetchall("""
            SELECT status, COUNT(*) as cnt 
            FROM assertion_temporalized 
            GROUP BY status
            ORDER BY status ASC
        """)
        report.status_distribution = {row['status']: row['cnt'] for row in rows}

        # Valid-time availability
        result = self.db.fetchone("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN valid_from_utc IS NOT NULL THEN 1 ELSE 0 END) as with_valid_from
            FROM assertion_temporalized
        """)
        if result and result['total'] > 0:
            report.valid_from_rate = result['with_valid_from'] / result['total']

        # Date span
        result = self.db.fetchone("""
            SELECT MIN(valid_from_utc) as min_date, MAX(valid_from_utc) as max_date
            FROM assertion_temporalized
            WHERE valid_from_utc IS NOT NULL
        """)
        if result and result['min_date'] and result['max_date']:
            report.date_span = (result['min_date'], result['max_date'])

        # Literal object rate
        result = self.db.fetchone("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN object_value_type IS NOT NULL THEN 1 ELSE 0 END) as with_literal
            FROM assertions
        """)
        if result and result['total'] > 0:
            report.literal_object_rate = result['with_literal'] / result['total']

        # SELF presence
        result = self.db.fetchone("""
            SELECT entity_id FROM entities
            WHERE entity_key = '__SELF__' AND entity_type = 'PERSON'
            LIMIT 1
        """)
        report.self_entity_id = result['entity_id'] if result else None

        # Time mention count (if table exists)
        if self.db.table_exists("time_mentions"):
            result = self.db.fetchone("SELECT COUNT(*) as cnt FROM time_mentions")
            report.time_mention_count = result['cnt'] if result else 0

        # Check for issues
        if report.temporalized_coverage < 1.0:
            report.issues.append(
                f"Incomplete temporalized coverage: {report.temporalized_coverage:.1%}"
            )
        if report.valid_from_rate < 0.5:
            report.issues.append(
                f"Low valid_from_utc rate: {report.valid_from_rate:.1%} - sparse temporal windows expected"
            )
        if not report.self_entity_id:
            report.issues.append("No SELF entity found - profile may use fallback focal entity")

        # Log report
        logger.info(f"Data Quality Report:")
        logger.info(f"  Entities: {report.entity_count}")
        logger.info(f"  Predicates: {report.predicate_count}")
        logger.info(f"  Assertions: {report.assertion_count}")
        logger.info(f"  Temporalized: {report.temporalized_count} ({report.temporalized_coverage:.1%})")
        logger.info(f"  Status distribution: {report.status_distribution}")
        logger.info(f"  Valid-from rate: {report.valid_from_rate:.1%}")
        logger.info(f"  Date span: {report.date_span}")
        logger.info(f"  Literal object rate: {report.literal_object_rate:.1%}")
        logger.info(f"  SELF entity: {report.self_entity_id}")
        logger.info(f"  Time mentions: {report.time_mention_count}")
        if report.issues:
            logger.warning(f"  Issues: {report.issues}")

        return report


# ===| MAIN PIPELINE |===

class GraphMaterializationPipeline:
    """
    Stage 6: Graph Materialization pipeline.
    """

    def __init__(self, config: Stage6Config):
        self.config = config
        self.db = Stage6Database(config.output_file_path)
        self.id_gen = IDGenerator(uuid.UUID(config.id_namespace))
        self.stage_started_at_utc = TimestampUtils.now_utc()

    def run(self) -> Dict[str, Any]:
        """Execute Stage 6 pipeline. Returns statistics."""
        logger.info("Starting Stage 6: Graph Materialization")
        stats: Dict[str, Any] = {
            "stage_started_at_utc": self.stage_started_at_utc,
            "nodes_by_type": {},
            "edges_by_type": {},
            "total_nodes": 0,
            "total_edges": 0
        }

        try:
            # Check prerequisites
            table_status = self.db.check_required_tables()

            # Run data quality checks
            quality_checker = DataQualityChecker(self.db)
            quality_report = quality_checker.run_checks()
            stats["quality_report"] = {
                "entity_count": quality_report.entity_count,
                "predicate_count": quality_report.predicate_count,
                "assertion_count": quality_report.assertion_count,
                "temporalized_coverage": quality_report.temporalized_coverage,
                "valid_from_rate": quality_report.valid_from_rate,
                "issues": quality_report.issues
            }

            # Initialize schema
            self.db.initialize_stage6_schema()

            # Begin transaction
            self.db.begin()

            # Initialize generators
            node_gen = NodeGenerator(self.db, self.config, self.id_gen)
            edge_gen = EdgeGenerator(self.db, self.config, self.id_gen, node_gen)

            # ===| NODE GENERATION |===
            logger.info("Starting node generation")
            all_nodes: List[GraphNode] = []

            # Generate nodes in type order (deterministic)
            entity_nodes = node_gen.generate_entity_nodes()
            all_nodes.extend(entity_nodes)
            stats["nodes_by_type"]["Entity"] = len(entity_nodes)

            predicate_nodes = node_gen.generate_predicate_nodes()
            all_nodes.extend(predicate_nodes)
            stats["nodes_by_type"]["Predicate"] = len(predicate_nodes)

            message_nodes = node_gen.generate_message_nodes()
            all_nodes.extend(message_nodes)
            stats["nodes_by_type"]["Message"] = len(message_nodes)

            assertion_nodes = node_gen.generate_assertion_nodes()
            all_nodes.extend(assertion_nodes)
            stats["nodes_by_type"]["Assertion"] = len(assertion_nodes)

            value_nodes = node_gen.generate_value_nodes()
            all_nodes.extend(value_nodes)
            stats["nodes_by_type"]["Value"] = len(value_nodes)

            time_interval_nodes = node_gen.generate_time_interval_nodes()
            all_nodes.extend(time_interval_nodes)
            stats["nodes_by_type"]["TimeInterval"] = len(time_interval_nodes)

            retraction_nodes = node_gen.generate_retraction_nodes()
            all_nodes.extend(retraction_nodes)
            stats["nodes_by_type"]["Retraction"] = len(retraction_nodes)

            conflict_group_nodes = node_gen.generate_conflict_group_nodes()
            all_nodes.extend(conflict_group_nodes)
            stats["nodes_by_type"]["ConflictGroup"] = len(conflict_group_nodes)

            lexicon_term_nodes = node_gen.generate_lexicon_term_nodes()
            all_nodes.extend(lexicon_term_nodes)
            stats["nodes_by_type"]["LexiconTerm"] = len(lexicon_term_nodes)

            time_mention_nodes = node_gen.generate_time_mention_nodes()
            all_nodes.extend(time_mention_nodes)
            stats["nodes_by_type"]["TimeMention"] = len(time_mention_nodes)

            # Sort nodes for deterministic insertion
            all_nodes.sort(key=lambda n: (n.node_type, n.node_id))

            # Insert all nodes
            logger.info(f"Inserting {len(all_nodes)} nodes")
            self.db.insert_nodes_batch(all_nodes)
            stats["total_nodes"] = len(all_nodes)

            # ===| EDGE GENERATION |===
            logger.info("Starting edge generation")
            all_edges: List[GraphEdge] = []

            semantic_edges = edge_gen.generate_assertion_semantic_edges()
            all_edges.extend(semantic_edges)
            stats["edges_by_type"]["semantic"] = len(semantic_edges)

            temporal_edges = edge_gen.generate_temporal_edges()
            all_edges.extend(temporal_edges)
            stats["edges_by_type"]["temporal"] = len(temporal_edges)

            time_qualifier_edges = edge_gen.generate_time_qualifier_edges()
            all_edges.extend(time_qualifier_edges)
            stats["edges_by_type"]["time_qualifier"] = len(time_qualifier_edges)

            message_edges = edge_gen.generate_message_anchoring_edges()
            all_edges.extend(message_edges)
            stats["edges_by_type"]["message_anchoring"] = len(message_edges)

            lifecycle_edges = edge_gen.generate_lifecycle_edges()
            all_edges.extend(lifecycle_edges)
            stats["edges_by_type"]["lifecycle"] = len(lifecycle_edges)

            conflict_edges = edge_gen.generate_conflict_edges()
            all_edges.extend(conflict_edges)
            stats["edges_by_type"]["conflict"] = len(conflict_edges)

            lexicon_edges = edge_gen.generate_lexicon_edges()
            all_edges.extend(lexicon_edges)
            stats["edges_by_type"]["lexicon"] = len(lexicon_edges)

            # ===| TEMPORAL PROFILES (Optional) |===
            if self.config.enable_temporal_profiles:
                logger.info("Generating temporal profiles")

                # Load profile config
                if self.config.temporal_profile_config_path:
                    profile_config = TemporalProfileConfig.from_yaml(
                        self.config.temporal_profile_config_path
                    )
                else:
                    profile_config = TemporalProfileConfig()

                # Apply CLI overrides if present
                if hasattr(self.config, '_cli_windowing_mode'):
                    profile_config.windowing_mode = self.config._cli_windowing_mode
                if hasattr(self.config, '_cli_evergreen_penalty'):
                    profile_config.evergreen_penalty_strength = self.config._cli_evergreen_penalty
                # Sync topic settings from Stage6Config
                profile_config.enable_topic_nodes = self.config.include_topic_nodes

                profile_gen = TemporalProfileGenerator(
                    self.db, self.config, profile_config, self.id_gen, node_gen
                )

                # Resolve focal entities
                focal_entities = profile_gen.resolve_focal_entities()
                stats["focal_entities"] = [
                    {"entity_id": e, "resolution_mode": m} for e, m in focal_entities
                ]

                if focal_entities:
                    # Generate profile nodes and edges
                    profile_nodes, profile_edges = profile_gen.generate_profile_nodes_and_edges(
                        focal_entities
                    )

                    # Add profile nodes (sorted)
                    profile_nodes.sort(key=lambda n: (n.node_type, n.node_id))
                    all_nodes.extend(profile_nodes)
                    self.db.insert_nodes_batch(profile_nodes)

                    # Count profile node types
                    for node in profile_nodes:
                        key = f"profile_{node.node_type}"
                        stats["nodes_by_type"][key] = stats["nodes_by_type"].get(key, 0) + 1

                    # Add profile edges
                    all_edges.extend(profile_edges)
                    stats["edges_by_type"]["profile"] = len(profile_edges)

                    stats["total_nodes"] = len(all_nodes) - len(profile_nodes) + len(profile_nodes)

            # Sort edges for deterministic insertion
            all_edges.sort(key=lambda e: (e.edge_type, e.edge_id))

            # Insert all edges
            logger.info(f"Inserting {len(all_edges)} edges")
            self.db.insert_edges_batch(all_edges)
            stats["total_edges"] = len(all_edges)

            # Commit transaction
            self.db.commit()
            stats["stage_completed_at_utc"] = TimestampUtils.now_utc()
            stats["success"] = True
            logger.info("Stage 6 completed successfully")

        except Exception as e:
            logger.error(f"Stage 6 failed: {e}")
            self.db.rollback()
            stats["success"] = False
            stats["error"] = str(e)
            raise

        finally:
            self.db.close()

        return stats


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
        "--temporal-profile-config",
        type=Path,
        default=Path("../data/metadata/temporal_profile_config.yaml"),
        help="Path to temporal_profile_config.yaml"
    )
    parser.add_argument(
        "--id-namespace",
        type=str,
        default=DEFAULT_KG_NAMESPACE,
        help="UUID namespace for ID generation"
    )
    parser.add_argument(
        "--no-message-nodes",
        action="store_true",
        help="Disable message node generation"
    )
    parser.add_argument(
        "--no-lexicon-nodes",
        action="store_true",
        help="Disable lexicon term node generation"
    )
    parser.add_argument(
        "--no-temporal-profiles",
        action="store_true",
        help="Disable temporal profile generation"
    )
    parser.add_argument(
        "--include-time-mentions",
        action="store_true",
        help="Include time mention nodes"
    )
    parser.add_argument(
        "--include-time-qualifier-edges",
        action="store_true",
        help="Include time qualifier edges"
    )
    parser.add_argument(
        "--include-member-evidence-edges",
        action="store_true",
        help="Include member evidence edges in profiles"
    )
    parser.add_argument(
        "--include-cluster-evolution",
        action="store_true",
        help="Include cluster evolution edges in profiles"
    )
    parser.add_argument(
        "--include-window-assertions",
        action="store_true",
        help="Include window-to-assertion edges in profiles"
    )
    parser.add_argument(
        "--no-value-members",
        action="store_true",
        help="Exclude value members from temporal profiles (only include entities)"
    )
    parser.add_argument(
        "--window-top-n-members",
        type=int,
        default=10,
        help="Number of top members per window for visualization (default: 10)"
    )
    parser.add_argument(
        "--cluster-top-n-members",
        type=int,
        default=50,
        help="Max members per cluster for CLUSTER_CONTAINS edges (default: 50)"
    )
    parser.add_argument(
        "--member-evidence-cap",
        type=int,
        default=50,
        help="Max evidence edges per member per window (default: 50)"
    )
    parser.add_argument(
        "--windowing-mode",
        type=str,
        choices=["mention_first", "mention_any", "validity_overlap"],
        default="mention_first",
        help="Window assignment mode: mention_first (each item in one month), mention_any, or validity_overlap (default: mention_first)"
    )
    parser.add_argument(
        "--evergreen-penalty",
        type=float,
        default=0.3,
        help="Evergreen penalty strength 0.0-1.0 (default: 0.3). Penalizes items appearing in many months."
    )
    parser.add_argument(
        "--no-topic-nodes",
        action="store_true",
        help="Disable Topic node generation (derived themes like 'work on: project')"
    )

    args = parser.parse_args()

    config = Stage6Config(
        output_file_path=args.db,
        id_namespace=args.id_namespace,
        include_message_nodes=not args.no_message_nodes,
        include_lexicon_nodes=not args.no_lexicon_nodes,
        enable_temporal_profiles=not args.no_temporal_profiles,
        temporal_profile_config_path=args.temporal_profile_config,
        include_time_mention_nodes=args.include_time_mentions,
        include_time_qualifier_edges=args.include_time_qualifier_edges,
        include_member_evidence_edges=args.include_member_evidence_edges,
        include_cluster_evolution_edges=args.include_cluster_evolution,
        include_window_assertion_edges=args.include_window_assertions,
        include_value_members_in_profiles=not args.no_value_members,
        window_top_n_members=args.window_top_n_members,
        cluster_top_n_members=args.cluster_top_n_members,
        member_evidence_edge_cap_per_window=args.member_evidence_cap,
        include_topic_nodes=not args.no_topic_nodes
    )

    # Override profile config settings from CLI
    # These will be applied when the profile config is loaded
    config._cli_windowing_mode = args.windowing_mode
    config._cli_evergreen_penalty = args.evergreen_penalty

    stats = run_stage6(config)

    logger.info("\n=== Stage 6 Summary ===")
    logger.info(f"Success: {stats.get('success', False)}")
    logger.info(f"Total nodes: {stats.get('total_nodes', 0)}")
    logger.info(f"Total edges: {stats.get('total_edges', 0)}")
    logger.info(f"Nodes by type: {stats.get('nodes_by_type', {})}")
    logger.info(f"Edges by type: {stats.get('edges_by_type', {})}")
    if stats.get('focal_entities'):
        logger.info(f"Focal entities: {stats['focal_entities']}")
    if stats.get('quality_report', {}).get('issues'):
        logger.warning(f"Quality issues: {stats['quality_report']['issues']}")