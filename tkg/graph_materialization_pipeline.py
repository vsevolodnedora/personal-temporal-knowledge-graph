"""
Stage 5: Graph Materialization

Creates deterministic, read-only graph-shaped tables for visualization/export,
using Stage 4 (assertion_temporalized, conflicts) as the authoritative
lifecycle + valid-time source.
"""
import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
        return JCS.canonicalize(obj).encode("utf-8")

    @staticmethod
    def _serialize(obj: Any) -> str:
        """Recursively serialize object to JCS-canonical form."""
        if obj is None:
            return "null"
        elif isinstance(obj, bool):
            return "true" if obj else "false"
        elif isinstance(obj, int):
            return str(obj)
        elif isinstance(obj, float):
            if obj != obj:  # NaN
                raise ValueError("NaN is not allowed in JCS")
            if obj == float("inf") or obj == float("-inf"):
                raise ValueError("Infinity is not allowed in JCS")
            s = repr(obj)
            if "e" in s or "E" in s:
                s = s.lower()
            return s
        elif isinstance(obj, str):
            return JCS._escape_string(obj)
        elif isinstance(obj, (list, tuple)):
            items = ",".join(JCS._serialize(item) for item in obj)
            return f"[{items}]"
        elif isinstance(obj, dict):
            sorted_keys = sorted(obj.keys(), key=lambda k: k.encode("utf-16-be"))
            items = ",".join(
                f"{JCS._escape_string(k)}:{JCS._serialize(obj[k])}"
                for k in sorted_keys
            )
            return "{" + items + "}"
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
            elif char == "\\":
                result.append("\\\\")
            elif char == "\b":
                result.append("\\b")
            elif char == "\f":
                result.append("\\f")
            elif char == "\n":
                result.append("\\n")
            elif char == "\r":
                result.append("\\r")
            elif char == "\t":
                result.append("\\t")
            elif code < 0x20:
                result.append(f"\\u{code:04x}")
            else:
                result.append(char)
        result.append('"')
        return "".join(result)


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
        return HashUtils.sha256_hex(s.encode("utf-8"))


class TimestampUtils:
    """
    Store timestamps as canonical UTC ISO-8601 strings with milliseconds:
    YYYY-MM-DDTHH:MM:SS.sssZ
    """

    ISO_UTC_MILLIS = "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]"

    @staticmethod
    def now_utc() -> str:
        return pendulum.now("UTC").format(TimestampUtils.ISO_UTC_MILLIS)


# ===| ENUMS |===

class NodeType(StrEnum):
    """Types of nodes in the graph."""

    ENTITY = "Entity"
    PREDICATE = "Predicate"
    MESSAGE = "Message"
    ASSERTION = "Assertion"
    VALUE = "Value"
    TIME_INTERVAL = "TimeInterval"
    RETRACTION = "Retraction"
    CONFLICT_GROUP = "ConflictGroup"


class EdgeType(StrEnum):
    """Types of edges in the graph."""

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
    IN_CONFLICT = "IN_CONFLICT"
    HAS_CONFLICT_MEMBER = "HAS_CONFLICT_MEMBER"
    CONFLICTS_WITH = "CONFLICTS_WITH"


# ===| CONFIGURATION |===

@dataclass
class Stage5Config:
    """Configuration for Stage 5 pipeline."""

    output_file_path: Path = field(default_factory=lambda: Path("../data/output/kg.db"))
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"

    # Node generation options
    emit_message_nodes: bool = True

    # Edge generation options
    emit_inverse_retraction_edges: bool = True
    emit_inverse_negation_edges: bool = True
    emit_conflict_membership_edges: bool = True
    emit_pairwise_conflict_edges: bool = False
    conflict_pairwise_max_n: int = 25

    # Metadata schema version
    schema_version: str = "1.0"


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


class Stage5Database(Database):
    """Stage 5 specific database operations."""

    REQUIRED_TABLES = [
        "conversations",
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
        """Verify all required tables from previous stages exist."""
        cursor = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing_tables = {row["name"] for row in cursor.fetchall()}

        missing = []
        for table in self.REQUIRED_TABLES:
            if table not in existing_tables:
                missing.append(table)

        if missing:
            raise RuntimeError(
                f"Missing required tables from previous stages: {missing}"
            )
        logger.info("All required tables present")

    def initialize_stage5_schema(self):
        """Create Stage 5 tables (drop if exist for overwrite)."""
        cursor = self.connection.cursor()

        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS graph_edges")
        cursor.execute("DROP TABLE IF EXISTS graph_nodes")


        # Create graph_nodes table
        cursor.execute("""
                       CREATE TABLE graph_nodes (
                                                    node_id TEXT PRIMARY KEY,
                                                    node_type TEXT NOT NULL,
                                                    source_id TEXT NOT NULL,
                                                    label TEXT,
                                                    metadata_json TEXT
                       )
                       """)

        # Create indices for graph_nodes
        cursor.execute(
            "CREATE INDEX idx_graph_nodes_type ON graph_nodes(node_type, node_id)"
        )
        cursor.execute(
            "CREATE INDEX idx_graph_nodes_source ON graph_nodes(node_type, source_id)"
        )

        # Create graph_edges table
        cursor.execute("""
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
        cursor.execute(
            "CREATE INDEX idx_graph_edges_src ON graph_edges(src_node_id)"
        )
        cursor.execute(
            "CREATE INDEX idx_graph_edges_dst ON graph_edges(dst_node_id)"
        )
        cursor.execute(
            "CREATE INDEX idx_graph_edges_type ON graph_edges(edge_type, edge_id)"
        )

        logger.info("Stage 5 schema initialized")

    # === Data Loading Streams ===

    def load_active_entities(self) -> Iterator[sqlite3.Row]:
        """Load active entities ordered by entity_id."""
        cursor = self.connection.execute("""
                                         SELECT entity_id, entity_type, entity_key, canonical_name,
                                                aliases_json, first_seen_at_utc, last_seen_at_utc,
                                                mention_count, conversation_count
                                         FROM entities
                                         WHERE status = 'active'
                                         ORDER BY entity_id ASC
                                         """)
        return cursor

    def load_predicates(self) -> Iterator[sqlite3.Row]:
        """Load predicates ordered by predicate_id."""
        cursor = self.connection.execute("""
                                         SELECT predicate_id, canonical_label, inverse_label, category,
                                                arity, value_type_constraint
                                         FROM predicates
                                         ORDER BY predicate_id ASC
                                         """)
        return cursor

    def load_messages(self) -> Iterator[sqlite3.Row]:
        """Load messages ordered by conversation_id, order_index, message_id."""
        cursor = self.connection.execute("""
                                         SELECT m.message_id, m.conversation_id, m.role, m.created_at_utc,
                                                m.timestamp_quality, m.parent_id, m.tree_path, m.order_index,
                                                c.title as conversation_title
                                         FROM messages m
                                                  LEFT JOIN conversations c ON m.conversation_id = c.conversation_id
                                         ORDER BY m.conversation_id ASC, m.order_index ASC, m.message_id ASC
                                         """)
        return cursor

    def load_assertions_joined(self) -> Iterator[sqlite3.Row]:
        """Load assertions joined with temporalized and messages."""
        cursor = self.connection.execute("""
                                         SELECT
                                             a.assertion_id, a.message_id, a.subject_entity_id, a.predicate_id,
                                             a.object_entity_id, a.object_value_type, a.object_value,
                                             a.object_signature, a.modality, a.polarity, a.asserted_role,
                                             a.asserted_at_utc, a.confidence_final, a.raw_assertion_json,
                                             t.status, t.valid_time_type, t.valid_from_utc, t.valid_to_utc,
                                             t.valid_until_hint_utc, t.temporal_superseded_by_assertion_id,
                                             t.retracted_by_retraction_id, t.negated_by_assertion_id,
                                             t.rule_id_applied, t.raw_temporalize_json,
                                             m.conversation_id, m.order_index
                                         FROM assertions a
                                                  JOIN assertion_temporalized t ON a.assertion_id = t.assertion_id
                                                  JOIN messages m ON a.message_id = m.message_id
                                         ORDER BY m.conversation_id ASC, m.order_index ASC,
                                                  a.message_id ASC, a.assertion_id ASC
                                         """)
        return cursor

    def load_retractions(self) -> Iterator[sqlite3.Row]:
        """Load retractions ordered deterministically."""
        cursor = self.connection.execute("""
                                         SELECT retraction_id, retraction_message_id, target_assertion_id,
                                                target_fact_key, retraction_type, replacement_assertion_id,
                                                confidence, char_start, char_end
                                         FROM retractions
                                         ORDER BY retraction_message_id ASC,
                                                  CASE WHEN char_start IS NULL THEN 1 ELSE 0 END,
                                                  char_start ASC,
                                                  retraction_id ASC
                                         """)
        return cursor

    def load_conflict_groups(self) -> Iterator[sqlite3.Row]:
        """Load conflict groups ordered by type and id."""
        cursor = self.connection.execute("""
                                         SELECT conflict_group_id, conflict_type, conflict_key,
                                                detected_at_utc, raw_conflict_json
                                         FROM conflict_groups
                                         ORDER BY conflict_type ASC, conflict_group_id ASC
                                         """)
        return cursor

    def load_conflict_members(self) -> Iterator[sqlite3.Row]:
        """Load conflict members ordered by group and assertion."""
        cursor = self.connection.execute("""
                                         SELECT conflict_group_id, assertion_id
                                         FROM conflict_members
                                         ORDER BY conflict_group_id ASC, assertion_id ASC
                                         """)
        return cursor

    def get_entity_canonical_name(self, entity_id: str) -> Optional[str]:
        """Get canonical name for an entity."""
        cursor = self.connection.execute(
            "SELECT canonical_name FROM entities WHERE entity_id = ?",
            (entity_id,)
        )
        row = cursor.fetchone()
        return row["canonical_name"] if row else None

    def get_predicate_label(self, predicate_id: str) -> Optional[str]:
        """Get canonical label for a predicate."""
        cursor = self.connection.execute(
            "SELECT canonical_label FROM predicates WHERE predicate_id = ?",
            (predicate_id,)
        )
        row = cursor.fetchone()
        return row["canonical_label"] if row else None

    # === Batch Insert Methods ===

    def insert_nodes_batch(self, nodes: List[Tuple[str, str, str, str, str]]):
        """Insert multiple nodes at once."""
        self.connection.executemany(
            """INSERT INTO graph_nodes
                   (node_id, node_type, source_id, label, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            nodes
        )

    def insert_edges_batch(self, edges: List[Tuple[str, str, str, str, str]]):
        """Insert multiple edges at once."""
        self.connection.executemany(
            """INSERT INTO graph_edges
                   (edge_id, edge_type, src_node_id, dst_node_id, metadata_json)
               VALUES (?, ?, ?, ?, ?)""",
            edges
        )

    def update_build_meta(self, stats: Dict[str, int]):
        """Update build_meta if exists with stage completion."""
        try:
            cursor = self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='build_meta'"
            )
            if cursor.fetchone():
                self.connection.execute(
                    "UPDATE build_meta SET stage_completed = 5"
                )
                logger.info("Updated build_meta.stage_completed = 5")
        except Exception as e:
            logger.warning(f"Could not update build_meta: {e}")


# ===| NODE GENERATORS |===

class NodeGenerator:
    """Generates graph nodes from source data."""

    def __init__(self, config: Stage5Config, id_generator: IDGenerator, db: Stage5Database):
        self.config = config
        self.id_gen = id_generator
        self.db = db

        # Track generated nodes for edge generation
        self.node_ids: Dict[str, str] = {}  # (node_type, source_id) -> node_id
        self.time_interval_nodes: Dict[str, str] = {}  # source_id -> node_id
        self.value_nodes: Dict[str, str] = {}  # source_id -> node_id

    def generate_node_id(self, node_type: str, source_id: str) -> str:
        """Generate deterministic node ID."""
        return self.id_gen.generate(["node", node_type, source_id])

    def register_node(self, node_type: str, source_id: str, node_id: str):
        """Register a node for later edge generation."""
        key = (node_type, source_id)
        self.node_ids[key] = node_id

    def get_node_id(self, node_type: str, source_id: str) -> Optional[str]:
        """Get registered node ID."""
        return self.node_ids.get((node_type, source_id))

    def build_metadata_json(self, data: Dict[str, Any]) -> str:
        """Build JCS-canonical metadata JSON with schema version."""
        data["schema_version"] = self.config.schema_version
        return JCS.canonicalize(data)

    # --- Entity Nodes ---

    def generate_entity_nodes(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate Entity nodes from active entities."""
        nodes = []
        for row in self.db.load_active_entities():
            source_id = row["entity_id"]
            node_id = self.generate_node_id(NodeType.ENTITY, source_id)

            # Label: canonical_name or fallback to entity_key
            label = row["canonical_name"] or row["entity_key"]

            # Metadata
            metadata = {
                "entity_id": row["entity_id"],
                "entity_type": row["entity_type"],
                "entity_key": row["entity_key"],
                "aliases_json": row["aliases_json"],
                "first_seen_at_utc": row["first_seen_at_utc"],
                "last_seen_at_utc": row["last_seen_at_utc"],
                "mention_count": row["mention_count"],
                "conversation_count": row["conversation_count"],
            }

            nodes.append((
                node_id,
                NodeType.ENTITY,
                source_id,
                label,
                self.build_metadata_json(metadata)
            ))
            self.register_node(NodeType.ENTITY, source_id, node_id)

        return sorted(nodes, key=lambda x: (x[1], x[0]))

    # --- Predicate Nodes ---

    def generate_predicate_nodes(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate Predicate nodes from predicates table."""
        nodes = []
        for row in self.db.load_predicates():
            source_id = row["predicate_id"]
            node_id = self.generate_node_id(NodeType.PREDICATE, source_id)

            label = row["canonical_label"]

            metadata = {
                "predicate_id": row["predicate_id"],
                "canonical_label": row["canonical_label"],
                "inverse_label": row["inverse_label"],
                "category": row["category"],
                "arity": row["arity"],
                "value_type_constraint": row["value_type_constraint"],
            }

            nodes.append((
                node_id,
                NodeType.PREDICATE,
                source_id,
                label,
                self.build_metadata_json(metadata)
            ))
            self.register_node(NodeType.PREDICATE, source_id, node_id)

        return sorted(nodes, key=lambda x: (x[1], x[0]))

    # --- Message Nodes ---

    def generate_message_nodes(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate Message nodes from messages table."""
        if not self.config.emit_message_nodes:
            return []

        nodes = []
        for row in self.db.load_messages():
            source_id = row["message_id"]
            node_id = self.generate_node_id(NodeType.MESSAGE, source_id)

            # Label: conversation_id#order_index or with title
            conv_title = row["conversation_title"]
            if conv_title:
                label = f"{conv_title}#{row['order_index']}"
            else:
                label = f"{row['conversation_id']}#{row['order_index']}"

            metadata = {
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "role": row["role"],
                "created_at_utc": row["created_at_utc"],
                "timestamp_quality": row["timestamp_quality"],
                "parent_id": row["parent_id"],
                "tree_path": row["tree_path"],
                "order_index": row["order_index"],
            }

            nodes.append((
                node_id,
                NodeType.MESSAGE,
                source_id,
                label,
                self.build_metadata_json(metadata)
            ))
            self.register_node(NodeType.MESSAGE, source_id, node_id)

        return sorted(nodes, key=lambda x: (x[1], x[0]))

    # --- TimeInterval Nodes ---

    def _compute_time_interval_source_id(
            self, valid_from: Optional[str], valid_to: Optional[str]
    ) -> str:
        """Compute deterministic source_id for TimeInterval."""
        # Use __NULL__ encoding for null values
        from_val = valid_from if valid_from is not None else "__NULL__"
        to_val = valid_to if valid_to is not None else "__NULL__"
        return HashUtils.sha256_string(JCS.canonicalize([from_val, to_val]))

    def _compute_time_interval_label(
            self, valid_from: Optional[str], valid_to: Optional[str], is_hint: bool = False
    ) -> str:
        """Compute human-readable label for TimeInterval."""
        if is_hint:
            return f"… until {valid_to}"
        from_str = valid_from if valid_from else "∅"
        to_str = valid_to if valid_to else "∞"
        return f"{from_str} … {to_str}"

    def generate_time_interval_node(
            self, valid_from: Optional[str], valid_to: Optional[str], is_hint: bool = False
    ) -> Optional[Tuple[str, str, str, str, str]]:
        """Generate a TimeInterval node if not already generated."""
        # Skip (NULL, NULL) unless it's a hint
        if valid_from is None and valid_to is None and not is_hint:
            return None

        source_id = self._compute_time_interval_source_id(valid_from, valid_to)

        # Check if already generated
        if source_id in self.time_interval_nodes:
            return None

        node_id = self.generate_node_id(NodeType.TIME_INTERVAL, source_id)
        label = self._compute_time_interval_label(valid_from, valid_to, is_hint)

        metadata = {
            "valid_from_utc": valid_from,
            "valid_to_utc": valid_to,
            "is_hint_only": is_hint,
        }

        self.time_interval_nodes[source_id] = node_id
        self.register_node(NodeType.TIME_INTERVAL, source_id, node_id)

        return (
            node_id,
            NodeType.TIME_INTERVAL,
            source_id,
            label,
            self.build_metadata_json(metadata)
        )

    # --- Value Nodes ---

    def _compute_value_source_id(
            self, value_type: str, value: str
    ) -> str:
        """Compute deterministic source_id for Value node."""
        return HashUtils.sha256_string(JCS.canonicalize([value_type, value]))

    def _compute_value_label(self, value_type: str, value: str) -> str:
        """Compute human-readable label for Value node."""
        if value_type == "string":
            # Parse the JSON string to get actual string
            try:
                parsed = json.loads(value)
                if isinstance(parsed, str):
                    # Truncate if too long
                    return parsed[:100] + "..." if len(parsed) > 100 else parsed
            except (json.JSONDecodeError, TypeError):
                pass
        return value

    def generate_value_node(
            self, value_type: str, value: str
    ) -> Optional[Tuple[str, str, str, str, str]]:
        """Generate a Value node if not already generated."""
        source_id = self._compute_value_source_id(value_type, value)

        if source_id in self.value_nodes:
            return None

        node_id = self.generate_node_id(NodeType.VALUE, source_id)
        label = self._compute_value_label(value_type, value)

        metadata = {
            "object_value_type": value_type,
            "object_value": value,
        }

        self.value_nodes[source_id] = node_id
        self.register_node(NodeType.VALUE, source_id, node_id)

        return (
            node_id,
            NodeType.VALUE,
            source_id,
            label,
            self.build_metadata_json(metadata)
        )

    # --- Assertion Nodes ---

    def generate_assertion_nodes(
            self
    ) -> Tuple[List[Tuple[str, str, str, str, str]], List[Tuple[str, str, str, str, str]], List[Tuple[str, str, str, str, str]]]:
        """
        Generate Assertion nodes and collect TimeInterval/Value nodes.
        Returns (assertion_nodes, time_interval_nodes, value_nodes).
        """
        assertion_nodes = []
        time_interval_nodes = []
        value_nodes = []

        for row in self.db.load_assertions_joined():
            source_id = row["assertion_id"]
            node_id = self.generate_node_id(NodeType.ASSERTION, source_id)

            # Build deterministic label
            subject_name = self.db.get_entity_canonical_name(row["subject_entity_id"]) or row["subject_entity_id"]
            pred_label = self.db.get_predicate_label(row["predicate_id"]) or row["predicate_id"]
            obj_sig = row["object_signature"]
            label = f"{subject_name} {pred_label} {obj_sig}"

            # Compute raw JSON hashes
            raw_assertion_hash = None
            if row["raw_assertion_json"]:
                raw_assertion_hash = HashUtils.sha256_string(row["raw_assertion_json"])

            raw_temporalize_hash = None
            if row["raw_temporalize_json"]:
                raw_temporalize_hash = HashUtils.sha256_string(row["raw_temporalize_json"])

            # Metadata
            metadata = {
                # Assertion core
                "assertion_id": row["assertion_id"],
                "message_id": row["message_id"],
                "subject_entity_id": row["subject_entity_id"],
                "predicate_id": row["predicate_id"],
                "object_entity_id": row["object_entity_id"],
                "object_value_type": row["object_value_type"],
                "object_value": row["object_value"],
                "object_signature": row["object_signature"],
                "modality": row["modality"],
                "polarity": row["polarity"],
                "asserted_role": row["asserted_role"],
                "asserted_at_utc": row["asserted_at_utc"],
                "confidence_final": row["confidence_final"],
                # Temporalized
                "status": row["status"],
                "valid_time_type": row["valid_time_type"],
                "valid_from_utc": row["valid_from_utc"],
                "valid_to_utc": row["valid_to_utc"],
                "valid_until_hint_utc": row["valid_until_hint_utc"],
                "temporal_superseded_by_assertion_id": row["temporal_superseded_by_assertion_id"],
                "retracted_by_retraction_id": row["retracted_by_retraction_id"],
                "negated_by_assertion_id": row["negated_by_assertion_id"],
                "rule_id_applied": row["rule_id_applied"],
                # Provenance hashes
                "raw_assertion_json_hash": raw_assertion_hash,
                "raw_temporalize_json_hash": raw_temporalize_hash,
            }

            assertion_nodes.append((
                node_id,
                NodeType.ASSERTION,
                source_id,
                label,
                self.build_metadata_json(metadata)
            ))
            self.register_node(NodeType.ASSERTION, source_id, node_id)

            # Generate TimeInterval node if valid_from_utc is non-null
            if row["valid_from_utc"] is not None:
                ti_node = self.generate_time_interval_node(
                    row["valid_from_utc"], row["valid_to_utc"]
                )
                if ti_node:
                    time_interval_nodes.append(ti_node)

            # Generate hint-only TimeInterval node if valid_until_hint_utc is non-null
            if row["valid_until_hint_utc"] is not None:
                ti_hint_node = self.generate_time_interval_node(
                    None, row["valid_until_hint_utc"], is_hint=True
                )
                if ti_hint_node:
                    time_interval_nodes.append(ti_hint_node)

            # Generate Value node if literal object
            if row["object_value_type"] is not None and row["object_value"] is not None:
                value_node = self.generate_value_node(
                    row["object_value_type"], row["object_value"]
                )
                if value_node:
                    value_nodes.append(value_node)

        return (
            sorted(assertion_nodes, key=lambda x: (x[1], x[0])),
            sorted(time_interval_nodes, key=lambda x: (x[1], x[0])),
            sorted(value_nodes, key=lambda x: (x[1], x[0]))
        )

    # --- Retraction Nodes ---

    def generate_retraction_nodes(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate Retraction nodes from retractions table."""
        nodes = []
        for row in self.db.load_retractions():
            source_id = row["retraction_id"]
            node_id = self.generate_node_id(NodeType.RETRACTION, source_id)

            # Label: "retraction" optionally with type
            label = f"retraction:{row['retraction_type']}" if row["retraction_type"] else "retraction"

            metadata = {
                "retraction_id": row["retraction_id"],
                "retraction_message_id": row["retraction_message_id"],
                "target_assertion_id": row["target_assertion_id"],
                "target_fact_key": row["target_fact_key"],
                "retraction_type": row["retraction_type"],
                "replacement_assertion_id": row["replacement_assertion_id"],
                "confidence": row["confidence"],
                "char_start": row["char_start"],
                "char_end": row["char_end"],
            }

            nodes.append((
                node_id,
                NodeType.RETRACTION,
                source_id,
                label,
                self.build_metadata_json(metadata)
            ))
            self.register_node(NodeType.RETRACTION, source_id, node_id)

        return sorted(nodes, key=lambda x: (x[1], x[0]))

    # --- ConflictGroup Nodes ---

    def generate_conflict_group_nodes(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate ConflictGroup nodes from conflict_groups table."""
        nodes = []
        for row in self.db.load_conflict_groups():
            source_id = row["conflict_group_id"]
            node_id = self.generate_node_id(NodeType.CONFLICT_GROUP, source_id)

            label = row["conflict_type"]

            raw_hash = None
            if row["raw_conflict_json"]:
                raw_hash = HashUtils.sha256_string(row["raw_conflict_json"])

            metadata = {
                "conflict_group_id": row["conflict_group_id"],
                "conflict_type": row["conflict_type"],
                "conflict_key": row["conflict_key"],
                "detected_at_utc": row["detected_at_utc"],
                "raw_conflict_json_hash": raw_hash,
            }

            nodes.append((
                node_id,
                NodeType.CONFLICT_GROUP,
                source_id,
                label,
                self.build_metadata_json(metadata)
            ))
            self.register_node(NodeType.CONFLICT_GROUP, source_id, node_id)

        return sorted(nodes, key=lambda x: (x[1], x[0]))


# ===| EDGE GENERATORS |===

class EdgeGenerator:
    """Generates graph edges from source data."""

    def __init__(
            self,
            config: Stage5Config,
            id_generator: IDGenerator,
            db: Stage5Database,
            node_generator: NodeGenerator
    ):
        self.config = config
        self.id_gen = id_generator
        self.db = db
        self.node_gen = node_generator

    def generate_edge_id(self, edge_type: str, src_node_id: str, dst_node_id: str) -> str:
        """Generate deterministic edge ID."""
        return self.id_gen.generate(["edge", edge_type, src_node_id, dst_node_id])

    def build_metadata_json(self, data: Dict[str, Any]) -> Optional[str]:
        """Build JCS-canonical metadata JSON, or None if empty."""
        if not data:
            return None
        data["schema_version"] = self.config.schema_version
        return JCS.canonicalize(data)

    def generate_semantic_edges(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate HAS_SUBJECT, HAS_PREDICATE, HAS_OBJECT edges."""
        edges = []

        for row in self.db.load_assertions_joined():
            assertion_id = row["assertion_id"]
            assertion_node_id = self.node_gen.get_node_id(NodeType.ASSERTION, assertion_id)

            if not assertion_node_id:
                logger.warning(f"Assertion node not found: {assertion_id}")
                continue

            # HAS_SUBJECT edge
            subject_node_id = self.node_gen.get_node_id(
                NodeType.ENTITY, row["subject_entity_id"]
            )
            if subject_node_id:
                edge_id = self.generate_edge_id(
                    EdgeType.HAS_SUBJECT, assertion_node_id, subject_node_id
                )
                edges.append((
                    edge_id, EdgeType.HAS_SUBJECT,
                    assertion_node_id, subject_node_id, None
                ))
            else:
                logger.warning(
                    f"Subject entity node not found: {row['subject_entity_id']}"
                )

            # HAS_PREDICATE edge
            predicate_node_id = self.node_gen.get_node_id(
                NodeType.PREDICATE, row["predicate_id"]
            )
            if predicate_node_id:
                edge_id = self.generate_edge_id(
                    EdgeType.HAS_PREDICATE, assertion_node_id, predicate_node_id
                )
                edges.append((
                    edge_id, EdgeType.HAS_PREDICATE,
                    assertion_node_id, predicate_node_id, None
                ))
            else:
                logger.warning(
                    f"Predicate node not found: {row['predicate_id']}"
                )

            # HAS_OBJECT edge (exclusive: entity OR value OR unary)
            if row["object_entity_id"] is not None:
                object_node_id = self.node_gen.get_node_id(
                    NodeType.ENTITY, row["object_entity_id"]
                )
                if object_node_id:
                    edge_id = self.generate_edge_id(
                        EdgeType.HAS_OBJECT, assertion_node_id, object_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.HAS_OBJECT,
                        assertion_node_id, object_node_id, None
                    ))
                else:
                    logger.warning(
                        f"Object entity node not found: {row['object_entity_id']}"
                    )
            elif row["object_value_type"] is not None and row["object_value"] is not None:
                # Literal object - link to Value node
                value_source_id = self.node_gen._compute_value_source_id(
                    row["object_value_type"], row["object_value"]
                )
                value_node_id = self.node_gen.value_nodes.get(value_source_id)
                if value_node_id:
                    edge_id = self.generate_edge_id(
                        EdgeType.HAS_OBJECT, assertion_node_id, value_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.HAS_OBJECT,
                        assertion_node_id, value_node_id, None
                    ))
            # else: unary assertion, no HAS_OBJECT edge

        return sorted(edges, key=lambda x: (x[1], x[0]))

    def generate_temporal_edges(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate VALID_IN and VALID_UNTIL_HINT edges."""
        edges = []

        for row in self.db.load_assertions_joined():
            assertion_id = row["assertion_id"]
            assertion_node_id = self.node_gen.get_node_id(NodeType.ASSERTION, assertion_id)

            if not assertion_node_id:
                continue

            # VALID_IN edge (if valid_from_utc is non-null)
            if row["valid_from_utc"] is not None:
                ti_source_id = self.node_gen._compute_time_interval_source_id(
                    row["valid_from_utc"], row["valid_to_utc"]
                )
                ti_node_id = self.node_gen.time_interval_nodes.get(ti_source_id)
                if ti_node_id:
                    edge_id = self.generate_edge_id(
                        EdgeType.VALID_IN, assertion_node_id, ti_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.VALID_IN,
                        assertion_node_id, ti_node_id, None
                    ))

            # VALID_UNTIL_HINT edge (if valid_until_hint_utc is non-null)
            if row["valid_until_hint_utc"] is not None:
                hint_source_id = self.node_gen._compute_time_interval_source_id(
                    None, row["valid_until_hint_utc"]
                )
                hint_node_id = self.node_gen.time_interval_nodes.get(hint_source_id)
                if hint_node_id:
                    edge_id = self.generate_edge_id(
                        EdgeType.VALID_UNTIL_HINT, assertion_node_id, hint_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.VALID_UNTIL_HINT,
                        assertion_node_id, hint_node_id, None
                    ))

        return sorted(edges, key=lambda x: (x[1], x[0]))

    def generate_message_anchoring_edges(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate ASSERTED_IN edges linking assertions to messages."""
        if not self.config.emit_message_nodes:
            return []

        edges = []

        for row in self.db.load_assertions_joined():
            assertion_id = row["assertion_id"]
            message_id = row["message_id"]

            assertion_node_id = self.node_gen.get_node_id(NodeType.ASSERTION, assertion_id)
            message_node_id = self.node_gen.get_node_id(NodeType.MESSAGE, message_id)

            if assertion_node_id and message_node_id:
                edge_id = self.generate_edge_id(
                    EdgeType.ASSERTED_IN, assertion_node_id, message_node_id
                )
                edges.append((
                    edge_id, EdgeType.ASSERTED_IN,
                    assertion_node_id, message_node_id, None
                ))
            elif assertion_node_id and not message_node_id:
                logger.error(
                    f"Cannot create ASSERTED_IN edge: message node missing for {message_id}"
                )

        return sorted(edges, key=lambda x: (x[1], x[0]))

    def generate_lifecycle_edges(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate SUPERSEDES, RETRACTED_BY, NEGATED_BY (and inverses) edges."""
        edges = []

        for row in self.db.load_assertions_joined():
            assertion_id = row["assertion_id"]
            assertion_node_id = self.node_gen.get_node_id(NodeType.ASSERTION, assertion_id)

            if not assertion_node_id:
                continue

            # SUPERSEDES edge (this assertion was superseded by another)
            # The edge direction is: superseded -> superseder
            # But per spec §5.5.4: SUPERSEDES: Assertion(A) → Assertion(temporal_superseded_by)
            # This seems like: the superseded assertion points to its superseder
            if row["temporal_superseded_by_assertion_id"]:
                superseder_node_id = self.node_gen.get_node_id(
                    NodeType.ASSERTION, row["temporal_superseded_by_assertion_id"]
                )
                if superseder_node_id:
                    metadata = {}
                    if row["rule_id_applied"]:
                        metadata["rule_id_applied"] = row["rule_id_applied"]

                    edge_id = self.generate_edge_id(
                        EdgeType.SUPERSEDES, assertion_node_id, superseder_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.SUPERSEDES,
                        assertion_node_id, superseder_node_id,
                        self.build_metadata_json(metadata) if metadata else None
                    ))

            # RETRACTED_BY edge
            if row["retracted_by_retraction_id"]:
                retraction_node_id = self.node_gen.get_node_id(
                    NodeType.RETRACTION, row["retracted_by_retraction_id"]
                )
                if retraction_node_id:
                    edge_id = self.generate_edge_id(
                        EdgeType.RETRACTED_BY, assertion_node_id, retraction_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.RETRACTED_BY,
                        assertion_node_id, retraction_node_id, None
                    ))

                    # Optional inverse: RETRACTS
                    if self.config.emit_inverse_retraction_edges:
                        inv_edge_id = self.generate_edge_id(
                            EdgeType.RETRACTS, retraction_node_id, assertion_node_id
                        )
                        edges.append((
                            inv_edge_id, EdgeType.RETRACTS,
                            retraction_node_id, assertion_node_id, None
                        ))

            # NEGATED_BY edge
            if row["negated_by_assertion_id"]:
                negator_node_id = self.node_gen.get_node_id(
                    NodeType.ASSERTION, row["negated_by_assertion_id"]
                )
                if negator_node_id:
                    edge_id = self.generate_edge_id(
                        EdgeType.NEGATED_BY, assertion_node_id, negator_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.NEGATED_BY,
                        assertion_node_id, negator_node_id, None
                    ))

                    # Optional inverse: NEGATES
                    if self.config.emit_inverse_negation_edges:
                        inv_edge_id = self.generate_edge_id(
                            EdgeType.NEGATES, negator_node_id, assertion_node_id
                        )
                        edges.append((
                            inv_edge_id, EdgeType.NEGATES,
                            negator_node_id, assertion_node_id, None
                        ))

        return sorted(edges, key=lambda x: (x[1], x[0]))

    def generate_conflict_edges(self) -> List[Tuple[str, str, str, str, str]]:
        """Generate conflict-related edges (IN_CONFLICT, HAS_CONFLICT_MEMBER, CONFLICTS_WITH)."""
        edges = []

        # Build conflict group memberships
        group_members: Dict[str, List[str]] = {}  # group_id -> [assertion_ids]

        for row in self.db.load_conflict_members():
            group_id = row["conflict_group_id"]
            assertion_id = row["assertion_id"]

            if group_id not in group_members:
                group_members[group_id] = []
            group_members[group_id].append(assertion_id)

        # Generate membership edges
        for group_id, assertion_ids in group_members.items():
            group_node_id = self.node_gen.get_node_id(NodeType.CONFLICT_GROUP, group_id)

            if not group_node_id:
                logger.warning(f"Conflict group node not found: {group_id}")
                continue

            for assertion_id in assertion_ids:
                assertion_node_id = self.node_gen.get_node_id(
                    NodeType.ASSERTION, assertion_id
                )

                if not assertion_node_id:
                    logger.warning(
                        f"Assertion node not found for conflict member: {assertion_id}"
                    )
                    continue

                # IN_CONFLICT: Assertion -> ConflictGroup
                if self.config.emit_conflict_membership_edges:
                    edge_id = self.generate_edge_id(
                        EdgeType.IN_CONFLICT, assertion_node_id, group_node_id
                    )
                    edges.append((
                        edge_id, EdgeType.IN_CONFLICT,
                        assertion_node_id, group_node_id, None
                    ))

                # HAS_CONFLICT_MEMBER: ConflictGroup -> Assertion (preferred per spec)
                edge_id = self.generate_edge_id(
                    EdgeType.HAS_CONFLICT_MEMBER, group_node_id, assertion_node_id
                )
                edges.append((
                    edge_id, EdgeType.HAS_CONFLICT_MEMBER,
                    group_node_id, assertion_node_id, None
                ))

            # Optional pairwise CONFLICTS_WITH edges
            if self.config.emit_pairwise_conflict_edges:
                n = len(assertion_ids)
                if n <= self.config.conflict_pairwise_max_n:
                    # Sort assertion_ids for deterministic ordering
                    sorted_ids = sorted(assertion_ids)
                    for i in range(n):
                        for j in range(i + 1, n):
                            a1_id = sorted_ids[i]
                            a2_id = sorted_ids[j]

                            a1_node_id = self.node_gen.get_node_id(
                                NodeType.ASSERTION, a1_id
                            )
                            a2_node_id = self.node_gen.get_node_id(
                                NodeType.ASSERTION, a2_id
                            )

                            if a1_node_id and a2_node_id:
                                # Direction: min(node_id) -> max(node_id)
                                if a1_node_id < a2_node_id:
                                    src, dst = a1_node_id, a2_node_id
                                else:
                                    src, dst = a2_node_id, a1_node_id

                                edge_id = self.generate_edge_id(
                                    EdgeType.CONFLICTS_WITH, src, dst
                                )
                                edges.append((
                                    edge_id, EdgeType.CONFLICTS_WITH,
                                    src, dst, None
                                ))

        return sorted(edges, key=lambda x: (x[1], x[0]))


# ===| MAIN PIPELINE |===

class GraphMaterializationPipeline:
    """
    Stage 5: Graph Materialization pipeline.

    Phases:
    1. Initialize schema
    2. Generate nodes (Phase A→C)
    3. Generate edges (§5.5)
    4. Insert nodes and edges
    5. Write stats and commit
    """

    def __init__(self, config: Stage5Config):
        self.config = config
        self.db = Stage5Database(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.stage_started_at_utc = TimestampUtils.now_utc()

    def run(self) -> Dict[str, int]:
        """Execute Stage 5 pipeline. Returns statistics."""
        logger.info("Starting Stage 5: Graph Materialization")
        stats: Dict[str, int] = {
            "nodes_total": 0,
            "edges_total": 0,
        }

        # Check prerequisites
        self.db.check_required_tables()

        # Initialize schema
        self.db.initialize_stage5_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Initialize generators
            node_gen = NodeGenerator(self.config, self.id_generator, self.db)
            edge_gen = EdgeGenerator(self.config, self.id_generator, self.db, node_gen)

            # === Phase A-C: Generate Nodes ===
            logger.info("Phase A-C: Generating nodes...")

            all_nodes = []

            # Entity nodes
            entity_nodes = node_gen.generate_entity_nodes()
            all_nodes.extend(entity_nodes)
            stats["nodes_entity"] = len(entity_nodes)
            logger.info(f"  Generated {len(entity_nodes)} Entity nodes")

            # Predicate nodes
            predicate_nodes = node_gen.generate_predicate_nodes()
            all_nodes.extend(predicate_nodes)
            stats["nodes_predicate"] = len(predicate_nodes)
            logger.info(f"  Generated {len(predicate_nodes)} Predicate nodes")

            # Message nodes (if enabled)
            message_nodes = node_gen.generate_message_nodes()
            all_nodes.extend(message_nodes)
            stats["nodes_message"] = len(message_nodes)
            logger.info(f"  Generated {len(message_nodes)} Message nodes")

            # Assertion nodes (+ TimeInterval + Value nodes)
            assertion_nodes, ti_nodes, value_nodes = node_gen.generate_assertion_nodes()
            all_nodes.extend(assertion_nodes)
            all_nodes.extend(ti_nodes)
            all_nodes.extend(value_nodes)
            stats["nodes_assertion"] = len(assertion_nodes)
            stats["nodes_time_interval"] = len(ti_nodes)
            stats["nodes_value"] = len(value_nodes)
            logger.info(f"  Generated {len(assertion_nodes)} Assertion nodes")
            logger.info(f"  Generated {len(ti_nodes)} TimeInterval nodes")
            logger.info(f"  Generated {len(value_nodes)} Value nodes")

            # Retraction nodes
            retraction_nodes = node_gen.generate_retraction_nodes()
            all_nodes.extend(retraction_nodes)
            stats["nodes_retraction"] = len(retraction_nodes)
            logger.info(f"  Generated {len(retraction_nodes)} Retraction nodes")

            # ConflictGroup nodes
            conflict_nodes = node_gen.generate_conflict_group_nodes()
            all_nodes.extend(conflict_nodes)
            stats["nodes_conflict_group"] = len(conflict_nodes)
            logger.info(f"  Generated {len(conflict_nodes)} ConflictGroup nodes")

            # Insert all nodes
            logger.info(f"Inserting {len(all_nodes)} nodes...")
            self.db.insert_nodes_batch(all_nodes)
            stats["nodes_total"] = len(all_nodes)

            # === Generate Edges ===
            logger.info("Generating edges...")

            all_edges = []

            # Semantic edges (HAS_SUBJECT, HAS_PREDICATE, HAS_OBJECT)
            semantic_edges = edge_gen.generate_semantic_edges()
            all_edges.extend(semantic_edges)
            stats["edges_semantic"] = len(semantic_edges)
            logger.info(f"  Generated {len(semantic_edges)} semantic edges")

            # Temporal edges (VALID_IN, VALID_UNTIL_HINT)
            temporal_edges = edge_gen.generate_temporal_edges()
            all_edges.extend(temporal_edges)
            stats["edges_temporal"] = len(temporal_edges)
            logger.info(f"  Generated {len(temporal_edges)} temporal edges")

            # Message anchoring edges (ASSERTED_IN)
            anchoring_edges = edge_gen.generate_message_anchoring_edges()
            all_edges.extend(anchoring_edges)
            stats["edges_asserted_in"] = len(anchoring_edges)
            logger.info(f"  Generated {len(anchoring_edges)} ASSERTED_IN edges")

            # Lifecycle edges (SUPERSEDES, RETRACTED_BY, NEGATED_BY + inverses)
            lifecycle_edges = edge_gen.generate_lifecycle_edges()
            all_edges.extend(lifecycle_edges)
            stats["edges_lifecycle"] = len(lifecycle_edges)
            logger.info(f"  Generated {len(lifecycle_edges)} lifecycle edges")

            # Conflict edges
            conflict_edges = edge_gen.generate_conflict_edges()
            all_edges.extend(conflict_edges)
            stats["edges_conflict"] = len(conflict_edges)
            logger.info(f"  Generated {len(conflict_edges)} conflict edges")

            # Insert all edges
            logger.info(f"Inserting {len(all_edges)} edges...")
            self.db.insert_edges_batch(all_edges)
            stats["edges_total"] = len(all_edges)

            # Update build_meta if present
            self.db.update_build_meta(stats)

            # Commit transaction
            self.db.commit()
            logger.info("Stage 5 completed successfully")

        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return stats


def run_stage5(config: Stage5Config) -> Dict[str, int]:
    """Run Stage 5 pipeline on existing database."""
    pipeline = GraphMaterializationPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 5: Graph Materialization")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: ../data/output/kg.db)"
    )
    parser.add_argument(
        "--no-message-nodes",
        action="store_true",
        help="Do not emit Message nodes"
    )
    parser.add_argument(
        "--no-inverse-retraction",
        action="store_true",
        help="Do not emit inverse RETRACTS edges"
    )
    parser.add_argument(
        "--no-inverse-negation",
        action="store_true",
        help="Do not emit inverse NEGATES edges"
    )
    parser.add_argument(
        "--emit-pairwise-conflicts",
        action="store_true",
        help="Emit pairwise CONFLICTS_WITH edges (can be expensive)"
    )
    parser.add_argument(
        "--conflict-pairwise-max-n",
        type=int,
        default=25,
        help="Max group size for pairwise conflict edges (default: 25)"
    )
    parser.add_argument(
        "--id-namespace",
        type=str,
        default="550e8400-e29b-41d4-a716-446655440000",
        help="UUID namespace for ID generation"
    )

    args = parser.parse_args()

    config = Stage5Config(
        output_file_path=args.db,
        id_namespace=args.id_namespace,
        emit_message_nodes=not args.no_message_nodes,
        emit_inverse_retraction_edges=not args.no_inverse_retraction,
        emit_inverse_negation_edges=not args.no_inverse_negation,
        emit_pairwise_conflict_edges=args.emit_pairwise_conflicts,
        conflict_pairwise_max_n=args.conflict_pairwise_max_n,
    )

    stats = run_stage5(config)

    logger.info("\n=== Stage 5 Summary ===")
    for key, value in sorted(stats.items()):
        logger.info(f"  {key}: {value}")
