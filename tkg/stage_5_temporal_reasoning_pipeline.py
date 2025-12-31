"""
Stage 5: Temporal Reasoning Layer

This module implements the temporal reasoning layer for a personal bitemporal
knowledge graph. It processes assertions from Stage 4 to:
1. Assign valid-time intervals based on time mentions and temporal qualifiers
2. Apply correction supersessions from Stage 4
3. Apply retractions
4. Apply negation closures
5. Apply functional invalidation rules
6. Detect and materialize conflicts
"""
import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Tuple, Set
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

class ValidTimeType(StrEnum):
    """Types of valid time intervals."""
    INSTANT = "instant"
    INTERVAL = "interval"
    UNKNOWN = "unknown"


class LifecycleStatus(StrEnum):
    """Assertion lifecycle statuses with precedence order."""
    ACTIVE = "active"
    INELIGIBLE = "ineligible"
    CONFLICTED = "conflicted"
    SUPERSEDED = "superseded"
    NEGATED = "negated"
    RETRACTED = "retracted"


class StatusPrecedence(IntEnum):
    """
    Status precedence (higher = stronger).
    Used to prevent downgrading a status during later passes.
    """
    ACTIVE = 0
    INELIGIBLE = 1
    CONFLICTED = 2
    SUPERSEDED = 3
    NEGATED = 4
    RETRACTED = 5


STATUS_PRECEDENCE_MAP: Dict[str, int] = {
    LifecycleStatus.ACTIVE: StatusPrecedence.ACTIVE,
    LifecycleStatus.INELIGIBLE: StatusPrecedence.INELIGIBLE,
    LifecycleStatus.CONFLICTED: StatusPrecedence.CONFLICTED,
    LifecycleStatus.SUPERSEDED: StatusPrecedence.SUPERSEDED,
    LifecycleStatus.NEGATED: StatusPrecedence.NEGATED,
    LifecycleStatus.RETRACTED: StatusPrecedence.RETRACTED,
}


class TimeSource(StrEnum):
    """Source of valid-time information."""
    QUALIFIER_ID = "QUALIFIER_ID"
    PROXIMITY = "PROXIMITY"
    ASSERTED_AT_FALLBACK = "ASSERTED_AT_FALLBACK"
    NONE = "NONE"


class ConflictType(StrEnum):
    """Types of temporal conflicts."""
    OBJECT_DISAGREEMENT = "OBJECT_DISAGREEMENT"
    NEGATION_AMBIGUOUS = "NEGATION_AMBIGUOUS"
    RETRACTION_TARGET_NOT_UNIQUE = "RETRACTION_TARGET_NOT_UNIQUE"
    OBJ_TRANSITION_CONFLICT = "OBJ_TRANSITION_CONFLICT"
    OBJ_CONFIDENCE_TOO_CLOSE = "OBJ_CONFIDENCE_TOO_CLOSE"
    SAME_TIME_DUPLICATES = "SAME_TIME_DUPLICATES"


class InvalidationPolicy(StrEnum):
    """Invalidation policies for functional predicates."""
    NONE = "none"
    CLOSE_PREVIOUS_ON_NEWER_STATE = "close_previous_on_newer_state"


class TemporalQualifierType(StrEnum):
    """Temporal qualifier types from Stage 4."""
    AT = "at"
    SINCE = "since"
    UNTIL = "until"
    DURING = "during"


# ===| CONFIGURATION |===

@dataclass
class Stage5Config:
    """Configuration for Stage 5 pipeline."""
    output_file_path: Path
    invalidation_rules_path: Optional[Path] = None
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"

    # Time linking parameters
    time_link_proximity_chars: int = 200
    time_link_min_alignment: float = 0.1

    # Fallback behavior
    fallback_valid_from_asserted: bool = True

    # Threshold parameters
    threshold_close: float = 0.7
    confidence_supersession_margin: float = 0.01

    # Detector tier and salience tie-breaking (§5.0 Configuration parameters)
    use_detector_tier_tiebreak: bool = True
    use_salience_conflict_tiebreak: bool = False


# ===| DATA CLASSES |===

@dataclass
class AssertionRecord:
    """Record of an assertion with all fields needed for temporal processing."""
    assertion_id: str
    message_id: str
    conversation_id: str
    order_index: int
    subject_entity_id: str
    subject_entity_type: str
    predicate_id: str
    object_signature: str
    temporal_qualifier_type: Optional[str]
    temporal_qualifier_id: Optional[str]
    modality: str
    polarity: str
    asserted_role: str
    asserted_at_utc: Optional[str]
    confidence_final: float
    has_user_corroboration: int
    superseded_by_assertion_id: Optional[str]
    supersession_type: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    message_timestamp_quality: str
    fact_key: str
    # Detection tiers for audit and tie-breaking (§5.4.0, §5.10.3, §5.10.4)
    subject_detection_tier: Optional[int] = None
    object_detection_tier: Optional[int] = None


@dataclass
class TimeMentionRecord:
    """Record of a time mention."""
    time_mention_id: str
    message_id: str
    char_start: int
    char_end: int
    resolved_type: str
    valid_from_utc: Optional[str]
    valid_to_utc: Optional[str]
    confidence: float


@dataclass
class RetractionRecord:
    """Record of a retraction."""
    retraction_id: str
    retraction_message_id: str
    target_assertion_id: Optional[str]
    target_fact_key: Optional[str]
    retraction_type: str
    confidence: float
    char_start: Optional[int]
    retraction_message_role: str


@dataclass
class InvalidationRule:
    """An invalidation rule from configuration."""
    rule_id: str
    predicate_id: Optional[str]
    subject_entity_type: str
    is_functional: bool
    invalidation_policy: str
    notes: Optional[str] = None


@dataclass
class TemporalizedAssertion:
    """Temporalized assertion record for insert/update."""
    assertion_id: str
    valid_time_type: str
    valid_from_utc: Optional[str]
    valid_to_utc: Optional[str]
    valid_until_hint_utc: Optional[str]
    status: str
    temporal_superseded_by_assertion_id: Optional[str] = None
    retracted_by_retraction_id: Optional[str] = None
    negated_by_assertion_id: Optional[str] = None
    rule_id_applied: Optional[str] = None
    raw_temporalize_json: Optional[str] = None


@dataclass
class ConflictGroup:
    """A conflict group with its members."""
    conflict_group_id: str
    conflict_type: str
    conflict_key: str
    detected_at_utc: str
    raw_conflict_json: Optional[str]
    member_assertion_ids: List[str] = field(default_factory=list)


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
        """Execute a SQL statement."""
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """Execute a SQL statement with multiple parameter sets."""
        return self.connection.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Fetch a single row."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Fetch all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()


class Stage5Database(Database):
    """Stage 5 specific database operations."""

    REQUIRED_TABLES = [
        "conversations", "messages", "entities",
        "time_mentions", "assertions", "predicates"
    ]

    STAGE5_TABLES = [
        "assertion_temporalized",
        "invalidation_rules",
        "conflict_groups",
        "conflict_members"
    ]

    def check_required_tables(self):
        """Verify that required tables from previous stages exist."""
        for table in self.REQUIRED_TABLES:
            result = self.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            if result is None:
                raise RuntimeError(f"Required table '{table}' not found. Run previous stages first.")
        logger.info("All required tables present")

    def drop_stage5_tables(self):
        """Drop existing Stage 5 tables (for re-run)."""
        for table in reversed(self.STAGE5_TABLES):
            self.execute(f"DROP TABLE IF EXISTS {table}")
        logger.info("Dropped existing Stage 5 tables")

    def initialize_stage5_schema(self):
        """Create Stage 5 tables and indices."""
        self.drop_stage5_tables()

        # assertion_temporalized
        self.execute("""
            CREATE TABLE assertion_temporalized (
                assertion_id TEXT PRIMARY KEY,
                valid_time_type TEXT NOT NULL,
                valid_from_utc TEXT,
                valid_to_utc TEXT,
                valid_until_hint_utc TEXT,
                status TEXT NOT NULL,
                temporal_superseded_by_assertion_id TEXT,
                retracted_by_retraction_id TEXT,
                negated_by_assertion_id TEXT,
                rule_id_applied TEXT,
                raw_temporalize_json TEXT,
                FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id),
                FOREIGN KEY (temporal_superseded_by_assertion_id) REFERENCES assertions(assertion_id),
                FOREIGN KEY (negated_by_assertion_id) REFERENCES assertions(assertion_id)
            )
        """)
        self.execute("CREATE INDEX idx_temporalized_status ON assertion_temporalized(status)")
        self.execute("""
            CREATE INDEX idx_temporalized_valid_from ON assertion_temporalized(valid_from_utc) 
            WHERE valid_from_utc IS NOT NULL
        """)
        self.execute("""
            CREATE INDEX idx_temporalized_superseded_by ON assertion_temporalized(temporal_superseded_by_assertion_id) 
            WHERE temporal_superseded_by_assertion_id IS NOT NULL
        """)

        # invalidation_rules
        self.execute("""
            CREATE TABLE invalidation_rules (
                rule_id TEXT PRIMARY KEY,
                predicate_id TEXT,
                subject_entity_type TEXT NOT NULL,
                is_functional INTEGER NOT NULL,
                invalidation_policy TEXT NOT NULL,
                notes TEXT,
                FOREIGN KEY (predicate_id) REFERENCES predicates(predicate_id)
            )
        """)
        self.execute("""
            CREATE INDEX idx_rules_predicate ON invalidation_rules(predicate_id) 
            WHERE predicate_id IS NOT NULL
        """)
        self.execute("CREATE INDEX idx_rules_subject_type ON invalidation_rules(subject_entity_type)")

        # conflict_groups
        self.execute("""
            CREATE TABLE conflict_groups (
                conflict_group_id TEXT PRIMARY KEY,
                conflict_type TEXT NOT NULL,
                conflict_key TEXT NOT NULL,
                detected_at_utc TEXT NOT NULL,
                raw_conflict_json TEXT
            )
        """)
        self.execute("CREATE INDEX idx_conflict_groups_type ON conflict_groups(conflict_type)")
        self.execute("CREATE UNIQUE INDEX idx_conflict_groups_key ON conflict_groups(conflict_key)")

        # conflict_members
        self.execute("""
            CREATE TABLE conflict_members (
                conflict_group_id TEXT NOT NULL,
                assertion_id TEXT NOT NULL,
                FOREIGN KEY (conflict_group_id) REFERENCES conflict_groups(conflict_group_id),
                FOREIGN KEY (assertion_id) REFERENCES assertions(assertion_id)
            )
        """)
        self.execute("""
            CREATE UNIQUE INDEX idx_conflict_members_pk ON conflict_members(conflict_group_id, assertion_id)
        """)
        self.execute("CREATE INDEX idx_conflict_members_assertion ON conflict_members(assertion_id)")
        self.execute("CREATE INDEX idx_conflict_members_group ON conflict_members(conflict_group_id)")

        logger.info("Stage 5 schema initialized")

    def get_predicate_id_by_label(self, label_norm: str) -> Optional[str]:
        """Look up predicate_id by normalized label."""
        row = self.fetchone(
            "SELECT predicate_id FROM predicates WHERE canonical_label_norm = ?",
            (label_norm,)
        )
        return row["predicate_id"] if row else None

    def insert_invalidation_rule(self, rule: InvalidationRule):
        """Insert or replace an invalidation rule."""
        self.execute("""
            INSERT OR REPLACE INTO invalidation_rules 
            (rule_id, predicate_id, subject_entity_type, is_functional, invalidation_policy, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id,
            rule.predicate_id,
            rule.subject_entity_type,
            1 if rule.is_functional else 0,
            rule.invalidation_policy,
            rule.notes
        ))

    def get_invalidation_rule(
        self, predicate_id: Optional[str], subject_entity_type: str
    ) -> Optional[InvalidationRule]:
        """
        Get the best matching invalidation rule using priority:
        1. Exact (predicate_id, subject_entity_type) match
        2. (predicate_id, '*') match
        3. (NULL, subject_entity_type) match
        4. Default rule (NULL, '*')
        """
        # Try exact match
        if predicate_id:
            row = self.fetchone("""
                SELECT * FROM invalidation_rules 
                WHERE predicate_id = ? AND subject_entity_type = ?
            """, (predicate_id, subject_entity_type))
            if row:
                return self._row_to_rule(row)

            # Try predicate with wildcard entity type
            row = self.fetchone("""
                SELECT * FROM invalidation_rules 
                WHERE predicate_id = ? AND subject_entity_type = '*'
            """, (predicate_id,))
            if row:
                return self._row_to_rule(row)

        # Try wildcard predicate with specific entity type
        row = self.fetchone("""
            SELECT * FROM invalidation_rules 
            WHERE predicate_id IS NULL AND subject_entity_type = ?
        """, (subject_entity_type,))
        if row:
            return self._row_to_rule(row)

        # Try default rule
        row = self.fetchone("""
            SELECT * FROM invalidation_rules 
            WHERE predicate_id IS NULL AND subject_entity_type = '*'
        """)
        if row:
            return self._row_to_rule(row)

        return None

    def _row_to_rule(self, row: sqlite3.Row) -> InvalidationRule:
        """Convert a database row to an InvalidationRule."""
        return InvalidationRule(
            rule_id=row["rule_id"],
            predicate_id=row["predicate_id"],
            subject_entity_type=row["subject_entity_type"],
            is_functional=bool(row["is_functional"]),
            invalidation_policy=row["invalidation_policy"],
            notes=row["notes"]
        )

    def iter_assertions_ordered(self) -> Iterator[AssertionRecord]:
        """
        Iterate over all assertions in deterministic order:
        (conversation_id, order_index, message_id, assertion_id)

        Includes subject_detection_tier and object_detection_tier per §5.4.0.
        """
        cursor = self.execute("""
            SELECT 
                a.assertion_id,
                a.message_id,
                m.conversation_id,
                m.order_index,
                a.subject_entity_id,
                e.entity_type as subject_entity_type,
                a.predicate_id,
                a.object_signature,
                a.temporal_qualifier_type,
                a.temporal_qualifier_id,
                a.modality,
                a.polarity,
                a.asserted_role,
                a.asserted_at_utc,
                a.confidence_final,
                a.has_user_corroboration,
                a.superseded_by_assertion_id,
                a.supersession_type,
                a.char_start,
                a.char_end,
                m.timestamp_quality as message_timestamp_quality,
                a.fact_key,
                a.subject_detection_tier,
                a.object_detection_tier
            FROM assertions a
            JOIN messages m ON a.message_id = m.message_id
            JOIN entities e ON a.subject_entity_id = e.entity_id
            ORDER BY m.conversation_id ASC, m.order_index ASC, a.message_id ASC, a.assertion_id ASC
        """)
        for row in cursor:
            yield AssertionRecord(
                assertion_id=row["assertion_id"],
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                order_index=row["order_index"],
                subject_entity_id=row["subject_entity_id"],
                subject_entity_type=row["subject_entity_type"],
                predicate_id=row["predicate_id"],
                object_signature=row["object_signature"],
                temporal_qualifier_type=row["temporal_qualifier_type"],
                temporal_qualifier_id=row["temporal_qualifier_id"],
                modality=row["modality"],
                polarity=row["polarity"],
                asserted_role=row["asserted_role"],
                asserted_at_utc=row["asserted_at_utc"],
                confidence_final=row["confidence_final"],
                has_user_corroboration=row["has_user_corroboration"],
                superseded_by_assertion_id=row["superseded_by_assertion_id"],
                supersession_type=row["supersession_type"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                message_timestamp_quality=row["message_timestamp_quality"],
                fact_key=row["fact_key"],
                subject_detection_tier=row["subject_detection_tier"],
                object_detection_tier=row["object_detection_tier"]
            )

    def get_time_mentions_for_message(self, message_id: str) -> List[TimeMentionRecord]:
        """Get all resolved time mentions for a message."""
        rows = self.fetchall("""
            SELECT 
                time_mention_id, message_id, char_start, char_end,
                resolved_type, valid_from_utc, valid_to_utc, confidence
            FROM time_mentions
            WHERE message_id = ? AND resolved_type IN ('instant', 'interval')
            ORDER BY char_start ASC, time_mention_id ASC
        """, (message_id,))
        return [
            TimeMentionRecord(
                time_mention_id=row["time_mention_id"],
                message_id=row["message_id"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                resolved_type=row["resolved_type"],
                valid_from_utc=row["valid_from_utc"],
                valid_to_utc=row["valid_to_utc"],
                confidence=row["confidence"]
            )
            for row in rows
        ]

    def get_time_mention_by_id(self, time_mention_id: str) -> Optional[TimeMentionRecord]:
        """Get a specific time mention by ID."""
        row = self.fetchone("""
            SELECT 
                time_mention_id, message_id, char_start, char_end,
                resolved_type, valid_from_utc, valid_to_utc, confidence
            FROM time_mentions
            WHERE time_mention_id = ?
        """, (time_mention_id,))
        if row is None:
            return None
        return TimeMentionRecord(
            time_mention_id=row["time_mention_id"],
            message_id=row["message_id"],
            char_start=row["char_start"],
            char_end=row["char_end"],
            resolved_type=row["resolved_type"],
            valid_from_utc=row["valid_from_utc"],
            valid_to_utc=row["valid_to_utc"],
            confidence=row["confidence"]
        )

    def insert_temporalized_assertion(self, ta: TemporalizedAssertion):
        """Insert or replace a temporalized assertion."""
        self.execute("""
            INSERT OR REPLACE INTO assertion_temporalized (
                assertion_id, valid_time_type, valid_from_utc, valid_to_utc,
                valid_until_hint_utc, status, temporal_superseded_by_assertion_id,
                retracted_by_retraction_id, negated_by_assertion_id, 
                rule_id_applied, raw_temporalize_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ta.assertion_id,
            ta.valid_time_type,
            ta.valid_from_utc,
            ta.valid_to_utc,
            ta.valid_until_hint_utc,
            ta.status,
            ta.temporal_superseded_by_assertion_id,
            ta.retracted_by_retraction_id,
            ta.negated_by_assertion_id,
            ta.rule_id_applied,
            ta.raw_temporalize_json
        ))

    def update_temporalized_status(
        self,
        assertion_id: str,
        new_status: str,
        temporal_superseded_by: Optional[str] = None,
        retracted_by: Optional[str] = None,
        negated_by: Optional[str] = None,
        rule_id: Optional[str] = None,
        valid_to_utc: Optional[str] = None,
        append_json: Optional[dict] = None
    ):
        """
        Update temporalized assertion status with precedence check.
        Only updates if new status has higher precedence than current.
        """
        # Get current status
        row = self.fetchone(
            "SELECT status, raw_temporalize_json FROM assertion_temporalized WHERE assertion_id = ?",
            (assertion_id,)
        )
        if row is None:
            logger.warning(f"Assertion {assertion_id} not found in temporalized table")
            return False

        current_status = row["status"]
        current_precedence = STATUS_PRECEDENCE_MAP.get(current_status, 0)
        new_precedence = STATUS_PRECEDENCE_MAP.get(new_status, 0)

        if new_precedence <= current_precedence:
            logger.debug(f"Skipping status update for {assertion_id}: {current_status} -> {new_status}")
            return False

        # Build update query
        updates = ["status = ?"]
        params = [new_status]

        if temporal_superseded_by is not None:
            updates.append("temporal_superseded_by_assertion_id = ?")
            params.append(temporal_superseded_by)
        if retracted_by is not None:
            updates.append("retracted_by_retraction_id = ?")
            params.append(retracted_by)
        if negated_by is not None:
            updates.append("negated_by_assertion_id = ?")
            params.append(negated_by)
        if rule_id is not None:
            updates.append("rule_id_applied = ?")
            params.append(rule_id)
        if valid_to_utc is not None:
            updates.append("valid_to_utc = ?")
            params.append(valid_to_utc)

        # Handle JSON append
        if append_json:
            current_json = json.loads(row["raw_temporalize_json"]) if row["raw_temporalize_json"] else {}
            if "events" not in current_json:
                current_json["events"] = []
            current_json["events"].append(append_json)
            updates.append("raw_temporalize_json = ?")
            params.append(JCS.canonicalize(current_json))

        params.append(assertion_id)
        self.execute(
            f"UPDATE assertion_temporalized SET {', '.join(updates)} WHERE assertion_id = ?",
            tuple(params)
        )
        return True

    def get_temporalized_assertion(self, assertion_id: str) -> Optional[dict]:
        """Get a temporalized assertion record."""
        row = self.fetchone(
            "SELECT * FROM assertion_temporalized WHERE assertion_id = ?",
            (assertion_id,)
        )
        return dict(row) if row else None

    def iter_correction_supersessions(self) -> Iterator[Tuple[str, str]]:
        """
        Iterate over assertions with correction supersessions from Stage 4.
        Returns (assertion_id, superseded_by_assertion_id) pairs.
        """
        cursor = self.execute("""
            SELECT assertion_id, superseded_by_assertion_id
            FROM assertions
            WHERE supersession_type = 'correction' 
              AND superseded_by_assertion_id IS NOT NULL
            ORDER BY assertion_id ASC
        """)
        for row in cursor:
            yield row["assertion_id"], row["superseded_by_assertion_id"]

    def iter_retractions_ordered(self) -> Iterator[RetractionRecord]:
        """Iterate over retractions in deterministic order."""
        cursor = self.execute("""
            SELECT 
                r.retraction_id,
                r.retraction_message_id,
                r.target_assertion_id,
                r.target_fact_key,
                r.retraction_type,
                r.confidence,
                r.char_start,
                m.role as retraction_message_role
            FROM retractions r
            JOIN messages m ON r.retraction_message_id = m.message_id
            ORDER BY r.retraction_message_id ASC, r.char_start ASC, r.retraction_id ASC
        """)
        for row in cursor:
            yield RetractionRecord(
                retraction_id=row["retraction_id"],
                retraction_message_id=row["retraction_message_id"],
                target_assertion_id=row["target_assertion_id"],
                target_fact_key=row["target_fact_key"],
                retraction_type=row["retraction_type"],
                confidence=row["confidence"],
                char_start=row["char_start"],
                retraction_message_role=row["retraction_message_role"]
            )

    def find_assertions_by_fact_key(self, fact_key: str, exclude_retracted: bool = True) -> List[str]:
        """Find assertion IDs by fact key."""
        if exclude_retracted:
            rows = self.fetchall("""
                SELECT a.assertion_id 
                FROM assertions a
                JOIN assertion_temporalized at ON a.assertion_id = at.assertion_id
                WHERE a.fact_key = ? AND at.status != 'retracted'
                ORDER BY a.assertion_id ASC
            """, (fact_key,))
        else:
            rows = self.fetchall(
                "SELECT assertion_id FROM assertions WHERE fact_key = ? ORDER BY assertion_id ASC",
                (fact_key,)
            )
        return [row["assertion_id"] for row in rows]

    def iter_negative_assertions_ordered(self) -> Iterator[AssertionRecord]:
        """Iterate over negative assertions in deterministic order."""
        cursor = self.execute("""
            SELECT 
                a.assertion_id,
                a.message_id,
                m.conversation_id,
                m.order_index,
                a.subject_entity_id,
                e.entity_type as subject_entity_type,
                a.predicate_id,
                a.object_signature,
                a.temporal_qualifier_type,
                a.temporal_qualifier_id,
                a.modality,
                a.polarity,
                a.asserted_role,
                a.asserted_at_utc,
                a.confidence_final,
                a.has_user_corroboration,
                a.superseded_by_assertion_id,
                a.supersession_type,
                a.char_start,
                a.char_end,
                m.timestamp_quality as message_timestamp_quality,
                a.fact_key,
                a.subject_detection_tier,
                a.object_detection_tier
            FROM assertions a
            JOIN messages m ON a.message_id = m.message_id
            JOIN entities e ON a.subject_entity_id = e.entity_id
            WHERE a.polarity = 'negative'
            ORDER BY m.conversation_id ASC, m.order_index ASC, a.message_id ASC, a.assertion_id ASC
        """)
        for row in cursor:
            yield AssertionRecord(
                assertion_id=row["assertion_id"],
                message_id=row["message_id"],
                conversation_id=row["conversation_id"],
                order_index=row["order_index"],
                subject_entity_id=row["subject_entity_id"],
                subject_entity_type=row["subject_entity_type"],
                predicate_id=row["predicate_id"],
                object_signature=row["object_signature"],
                temporal_qualifier_type=row["temporal_qualifier_type"],
                temporal_qualifier_id=row["temporal_qualifier_id"],
                modality=row["modality"],
                polarity=row["polarity"],
                asserted_role=row["asserted_role"],
                asserted_at_utc=row["asserted_at_utc"],
                confidence_final=row["confidence_final"],
                has_user_corroboration=row["has_user_corroboration"],
                superseded_by_assertion_id=row["superseded_by_assertion_id"],
                supersession_type=row["supersession_type"],
                char_start=row["char_start"],
                char_end=row["char_end"],
                message_timestamp_quality=row["message_timestamp_quality"],
                fact_key=row["fact_key"],
                subject_detection_tier=row["subject_detection_tier"],
                object_detection_tier=row["object_detection_tier"]
            )

    def find_negation_targets(
        self,
        subject_entity_id: str,
        predicate_id: str,
        object_signature: str
    ) -> List[str]:
        """
        Find active, eligible assertions that match a negation's target.
        Returns assertion IDs.
        """
        rows = self.fetchall("""
            SELECT a.assertion_id
            FROM assertions a
            JOIN assertion_temporalized at ON a.assertion_id = at.assertion_id
            WHERE a.subject_entity_id = ?
              AND a.predicate_id = ?
              AND a.object_signature = ?
              AND at.status = 'active'
              AND a.polarity = 'positive'
            ORDER BY a.assertion_id ASC
        """, (subject_entity_id, predicate_id, object_signature))
        return [row["assertion_id"] for row in rows]

    def get_functional_groups(self) -> Iterator[Tuple[str, str, List[dict]]]:
        """
        Get assertion groups by (subject_entity_id, predicate_id) for functional invalidation.
        Only includes active assertions with valid_from_utc.
        Includes detection tiers per §5.10.3.
        Returns (subject_entity_id, predicate_id, [assertion_records])
        """
        # First get distinct (subject, predicate) pairs
        pairs = self.fetchall("""
            SELECT DISTINCT a.subject_entity_id, a.predicate_id
            FROM assertions a
            JOIN assertion_temporalized at ON a.assertion_id = at.assertion_id
            WHERE at.status = 'active' AND at.valid_from_utc IS NOT NULL
            ORDER BY a.subject_entity_id ASC, a.predicate_id ASC
        """)

        for pair in pairs:
            subject_id = pair["subject_entity_id"]
            predicate_id = pair["predicate_id"]

            # Get assertions for this group including detection tiers
            rows = self.fetchall("""
                SELECT 
                    a.assertion_id,
                    a.object_signature,
                    a.confidence_final,
                    at.valid_from_utc,
                    at.valid_to_utc,
                    at.status,
                    e.entity_type as subject_entity_type,
                    a.subject_detection_tier,
                    a.object_detection_tier
                FROM assertions a
                JOIN assertion_temporalized at ON a.assertion_id = at.assertion_id
                JOIN entities e ON a.subject_entity_id = e.entity_id
                WHERE a.subject_entity_id = ?
                  AND a.predicate_id = ?
                  AND at.status = 'active'
                  AND at.valid_from_utc IS NOT NULL
                ORDER BY at.valid_from_utc ASC, a.assertion_id ASC
            """, (subject_id, predicate_id))

            assertions = [dict(row) for row in rows]
            if assertions:
                yield subject_id, predicate_id, assertions

    def insert_conflict_group(self, conflict: ConflictGroup):
        """Insert a conflict group with its members."""
        self.execute("""
            INSERT OR IGNORE INTO conflict_groups 
            (conflict_group_id, conflict_type, conflict_key, detected_at_utc, raw_conflict_json)
            VALUES (?, ?, ?, ?, ?)
        """, (
            conflict.conflict_group_id,
            conflict.conflict_type,
            conflict.conflict_key,
            conflict.detected_at_utc,
            conflict.raw_conflict_json
        ))

        for assertion_id in conflict.member_assertion_ids:
            self.execute("""
                INSERT OR IGNORE INTO conflict_members (conflict_group_id, assertion_id)
                VALUES (?, ?)
            """, (conflict.conflict_group_id, assertion_id))

    def get_assertion_count(self) -> int:
        """Get total assertion count."""
        row = self.fetchone("SELECT COUNT(*) as cnt FROM assertions")
        return row["cnt"] if row else 0

    def get_temporalized_count_by_status(self) -> Dict[str, int]:
        """Get counts of temporalized assertions by status."""
        rows = self.fetchall("""
            SELECT status, COUNT(*) as cnt 
            FROM assertion_temporalized 
            GROUP BY status
        """)
        return {row["status"]: row["cnt"] for row in rows}

    def get_conflict_count_by_type(self) -> Dict[str, int]:
        """Get counts of conflicts by type."""
        rows = self.fetchall("""
            SELECT conflict_type, COUNT(*) as cnt 
            FROM conflict_groups 
            GROUP BY conflict_type
        """)
        return {row["conflict_type"]: row["cnt"] for row in rows}

    def build_entity_salience_index(self) -> Dict[str, Optional[float]]:
        """
        Build entity salience index for conflict tie-breaks (§5.3.3).
        Returns dict mapping entity_id to salience_score.
        """
        rows = self.fetchall("""
            SELECT entity_id, salience_score
            FROM entities
        """)
        return {row["entity_id"]: row["salience_score"] for row in rows}


# ===| MAIN PIPELINE |===

class TemporalReasoningPipeline:
    """
    Stage 5: Temporal Reasoning Layer pipeline.

    Phases:
    1. Initialize schema and seed rules
    2. Build entity salience index (if configured)
    3. Populate assertion_temporalized (valid-time assignment)
    4. Apply Stage 4 correction supersessions
    5. Apply retractions
    6. Apply negation closures
    7. Apply functional invalidation
    8. Finalize conflicts and persist
    """

    # Sentinel value for NULL object_detection_tier in min() calculations (§5.10.3)
    DETECTION_TIER_NULL_SENTINEL = 5

    def __init__(self, config: Stage5Config):
        self.config = config
        self.db = Stage5Database(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.stage_started_at_utc = TimestampUtils.now_utc()

        # Cache for time mentions per message
        self._time_mentions_cache: Dict[str, List[TimeMentionRecord]] = {}

        # Entity salience index (§5.3.3)
        self._entity_salience: Dict[str, Optional[float]] = {}

        # Collected conflicts
        self._conflicts: List[ConflictGroup] = []

    def run(self) -> Dict[str, Any]:
        """Execute Stage 5 pipeline. Returns statistics."""
        logger.info("Starting Stage 5: Temporal Reasoning Layer")
        stats = {
            "assertions_total": 0,
            "assertions_temporalized": 0,
            "corrections_applied": 0,
            "retractions_applied": 0,
            "negations_applied": 0,
            "functional_supersessions": 0,
            "conflicts_by_type": {},
        }

        # Check prerequisites
        self.db.check_required_tables()
        stats["assertions_total"] = self.db.get_assertion_count()

        # Initialize schema
        self.db.initialize_stage5_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Phase 1: Seed invalidation rules
            logger.info("Phase 1: Seeding invalidation rules")
            rules_count = self._seed_invalidation_rules()
            logger.info(f"Seeded {rules_count} invalidation rules")

            # Phase 2: Build entity salience index (§5.3.3)
            if self.config.use_salience_conflict_tiebreak:
                logger.info("Phase 2: Building entity salience index")
                self._entity_salience = self.db.build_entity_salience_index()
                logger.info(f"Built salience index with {len(self._entity_salience)} entities")
            else:
                logger.info("Phase 2: Skipping entity salience index (disabled)")

            # Phase 3: Populate assertion_temporalized
            logger.info("Phase 3: Populating assertion_temporalized (valid-time assignment)")
            stats["assertions_temporalized"] = self._populate_temporalized()
            logger.info(f"Temporalized {stats['assertions_temporalized']} assertions")

            # Phase 4: Apply Stage 4 correction supersessions
            logger.info("Phase 4: Applying correction supersessions from Stage 4")
            stats["corrections_applied"] = self._apply_correction_supersessions()
            logger.info(f"Applied {stats['corrections_applied']} correction supersessions")

            # Phase 5: Apply retractions
            logger.info("Phase 5: Applying retractions")
            stats["retractions_applied"] = self._apply_retractions()
            logger.info(f"Applied {stats['retractions_applied']} retractions")

            # Phase 6: Apply negation closures
            logger.info("Phase 6: Applying negation closures")
            stats["negations_applied"] = self._apply_negations()
            logger.info(f"Applied {stats['negations_applied']} negation closures")

            # Phase 7: Apply functional invalidation
            logger.info("Phase 7: Applying functional invalidation")
            stats["functional_supersessions"] = self._apply_functional_invalidation()
            logger.info(f"Applied {stats['functional_supersessions']} functional supersessions")

            # Phase 8: Finalize conflicts
            logger.info("Phase 8: Finalizing conflicts")
            self._finalize_conflicts()
            stats["conflicts_by_type"] = self.db.get_conflict_count_by_type()
            logger.info(f"Finalized {sum(stats['conflicts_by_type'].values())} conflict groups")

            # Commit transaction
            self.db.commit()
            logger.info("Stage 5 completed successfully")

            # Add final status counts
            stats["status_counts"] = self.db.get_temporalized_count_by_status()

        except Exception as e:
            logger.error(f"Stage 5 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return stats

    def _seed_invalidation_rules(self) -> int:
        """Seed invalidation rules from config file and ensure default rule exists."""
        count = 0

        # Insert default rule
        default_rule_id = self.id_generator.generate(["invalidation_rule", "__NULL__", "*"])
        default_rule = InvalidationRule(
            rule_id=default_rule_id,
            predicate_id=None,
            subject_entity_type="*",
            is_functional=False,
            invalidation_policy=InvalidationPolicy.NONE,
            notes="Default rule (auto-generated)"
        )
        self.db.insert_invalidation_rule(default_rule)
        count += 1

        # Load rules from YAML file if provided
        if self.config.invalidation_rules_path and self.config.invalidation_rules_path.exists():
            rules_data = yaml.safe_load(self.config.invalidation_rules_path.read_text())
            rules_list = rules_data.get("rules", [])

            for i, rule_data in enumerate(rules_list):
                predicate_id = None

                # Resolve predicate reference
                if "predicate_id" in rule_data:
                    predicate_id = rule_data["predicate_id"]
                elif "predicate_label" in rule_data:
                    label_norm = rule_data["predicate_label"].lower()
                    predicate_id = self.db.get_predicate_id_by_label(label_norm)
                    if predicate_id is None:
                        logger.debug(f"Predicate '{rule_data['predicate_label']}' not found, rule skipped")
                        continue

                subject_entity_type = rule_data.get("subject_entity_type", "*")

                # Generate rule ID
                rule_id = self.id_generator.generate([
                    "invalidation_rule",
                    predicate_id if predicate_id else "__NULL__",
                    subject_entity_type
                ])

                rule = InvalidationRule(
                    rule_id=rule_id,
                    predicate_id=predicate_id,
                    subject_entity_type=subject_entity_type,
                    is_functional=rule_data.get("is_functional", False),
                    invalidation_policy=rule_data.get("invalidation_policy", InvalidationPolicy.NONE),
                    notes=f"From invalidation_rules.yaml, index {i}"
                )
                self.db.insert_invalidation_rule(rule)
                count += 1

        return count

    def _populate_temporalized(self) -> int:
        """Populate assertion_temporalized with valid-time assignment."""
        count = 0

        for assertion in self.db.iter_assertions_ordered():
            ta = self._compute_temporalized_assertion(assertion)
            self.db.insert_temporalized_assertion(ta)
            count += 1

            if count % 1000 == 0:
                logger.debug(f"Temporalized {count} assertions")

        return count

    def _compute_temporalized_assertion(self, assertion: AssertionRecord) -> TemporalizedAssertion:
        """Compute valid-time and initial status for an assertion."""
        decision_log: Dict[str, Any] = {
            "assertion_id": assertion.assertion_id,
            "time_link": None,
            "candidates": [],
            "qualifier_interpretation": None,
            "fallback": None,
            "eligibility_reasons": [],
            # Detection tiers for audit (§5.4.0)
            "subject_detection_tier": assertion.subject_detection_tier,
            "object_detection_tier": assertion.object_detection_tier,
        }

        # Initialize output fields
        valid_time_type = ValidTimeType.UNKNOWN
        valid_from_utc: Optional[str] = None
        valid_to_utc: Optional[str] = None
        valid_until_hint_utc: Optional[str] = None
        time_source = TimeSource.NONE
        has_explicit_valid_time = False
        fallback_blocked_reason: Optional[str] = None

        # Get time mentions for message
        time_mentions = self._get_time_mentions(assertion.message_id)

        chosen_time: Optional[TimeMentionRecord] = None

        # Step 1: Check explicit temporal qualifier
        if assertion.temporal_qualifier_id:
            tm = self.db.get_time_mention_by_id(assertion.temporal_qualifier_id)
            if tm and tm.resolved_type in ('instant', 'interval'):
                chosen_time = tm
                time_source = TimeSource.QUALIFIER_ID
                has_explicit_valid_time = True
                decision_log["time_link"] = {
                    "method": "QUALIFIER_ID",
                    "time_mention_id": tm.time_mention_id
                }
            else:
                decision_log["time_link"] = {
                    "method": "QUALIFIER_ID_MISSING_OR_UNRESOLVED",
                    "qualifier_id": assertion.temporal_qualifier_id
                }

        # Step 2: Proximity-based selection if no qualifier
        if chosen_time is None and time_mentions:
            candidates_with_scores = []

            for tm in time_mentions:
                alignment = self._compute_alignment(assertion, tm)
                if alignment >= self.config.time_link_min_alignment:
                    score = tm.confidence * alignment
                    candidates_with_scores.append({
                        "time_mention_id": tm.time_mention_id,
                        "alignment": alignment,
                        "confidence": tm.confidence,
                        "score": score,
                        "char_start": tm.char_start,
                    })

            decision_log["candidates"] = candidates_with_scores

            if candidates_with_scores:
                # Sort by score desc, alignment desc, char_start asc, id asc
                candidates_with_scores.sort(
                    key=lambda c: (-c["score"], -c["alignment"], c["char_start"], c["time_mention_id"])
                )
                winner = candidates_with_scores[0]
                chosen_time = next(
                    tm for tm in time_mentions
                    if tm.time_mention_id == winner["time_mention_id"]
                )
                time_source = TimeSource.PROXIMITY
                has_explicit_valid_time = True
                decision_log["time_link"] = {
                    "method": "PROXIMITY",
                    "time_mention_id": chosen_time.time_mention_id,
                    "score": winner["score"],
                    "alignment": winner["alignment"]
                }

        # Step 3: Interpret temporal qualifier
        if chosen_time:
            qualifier = assertion.temporal_qualifier_type
            t_from = chosen_time.valid_from_utc
            t_to = chosen_time.valid_to_utc
            is_instant = chosen_time.resolved_type == "instant"
            is_interval = chosen_time.resolved_type == "interval"

            if qualifier is None or qualifier == TemporalQualifierType.AT:
                if is_instant:
                    valid_time_type = ValidTimeType.INSTANT
                    valid_from_utc = t_from
                else:
                    valid_time_type = ValidTimeType.INTERVAL
                    valid_from_utc = t_from
                    valid_to_utc = t_to
                decision_log["qualifier_interpretation"] = {
                    "qualifier": qualifier or "NULL",
                    "result": "standard"
                }

            elif qualifier == TemporalQualifierType.DURING:
                if is_interval:
                    valid_time_type = ValidTimeType.INTERVAL
                    valid_from_utc = t_from
                    valid_to_utc = t_to
                else:
                    # Instant treated as AT
                    valid_time_type = ValidTimeType.INSTANT
                    valid_from_utc = t_from
                    decision_log["qualifier_interpretation"] = {
                        "qualifier": "during",
                        "result": "DURING_INSTANT_DOWNGRADED"
                    }

            elif qualifier == TemporalQualifierType.SINCE:
                valid_time_type = ValidTimeType.INTERVAL
                valid_from_utc = t_from
                valid_to_utc = None  # Open-ended
                decision_log["qualifier_interpretation"] = {
                    "qualifier": "since",
                    "result": "open_ended_interval"
                }

            elif qualifier == TemporalQualifierType.UNTIL:
                # Conservative: don't invent start time
                valid_time_type = ValidTimeType.UNKNOWN
                valid_from_utc = None
                valid_to_utc = None
                valid_until_hint_utc = t_to if is_interval else t_from
                decision_log["qualifier_interpretation"] = {
                    "qualifier": "until",
                    "result": "UNTIL_QUALIFIER_HINT_STORED"
                }

        # Step 4: Fallback if no time chosen
        if time_source == TimeSource.NONE:
            if self.config.fallback_valid_from_asserted:
                if assertion.message_timestamp_quality != "original":
                    valid_time_type = ValidTimeType.UNKNOWN
                    fallback_blocked_reason = "TIMESTAMP_NOT_ORIGINAL"
                    decision_log["fallback"] = {
                        "blocked": True,
                        "reason": "TIMESTAMP_NOT_ORIGINAL"
                    }
                elif assertion.asserted_at_utc is None:
                    valid_time_type = ValidTimeType.UNKNOWN
                    fallback_blocked_reason = "ASSERTED_AT_NULL"
                    decision_log["fallback"] = {
                        "blocked": True,
                        "reason": "ASSERTED_AT_NULL"
                    }
                else:
                    valid_time_type = ValidTimeType.INSTANT
                    valid_from_utc = assertion.asserted_at_utc
                    time_source = TimeSource.ASSERTED_AT_FALLBACK
                    has_explicit_valid_time = False  # Important!
                    decision_log["fallback"] = {
                        "used": True,
                        "asserted_at_utc": assertion.asserted_at_utc
                    }
            else:
                valid_time_type = ValidTimeType.UNKNOWN
                fallback_blocked_reason = "FALLBACK_DISABLED"
                decision_log["fallback"] = {
                    "blocked": True,
                    "reason": "FALLBACK_DISABLED"
                }

        # Step 5: Integrity check for intervals
        if (valid_time_type == ValidTimeType.INTERVAL and
            valid_to_utc is not None and
            valid_from_utc is not None and
            valid_to_utc <= valid_from_utc):
            valid_time_type = ValidTimeType.UNKNOWN
            valid_from_utc = None
            valid_to_utc = None
            decision_log["integrity_check"] = "VALID_TIME_NONPOSITIVE_INTERVAL"

        # Store decision flags (§5.4.0)
        decision_log["time_source"] = time_source
        decision_log["has_explicit_valid_time"] = has_explicit_valid_time
        if fallback_blocked_reason:
            decision_log["fallback_blocked_reason"] = fallback_blocked_reason

        # Compute eligibility and initial status
        status, eligibility_reasons = self._compute_eligibility(
            assertion, valid_from_utc, has_explicit_valid_time
        )
        decision_log["eligibility_reasons"] = eligibility_reasons

        return TemporalizedAssertion(
            assertion_id=assertion.assertion_id,
            valid_time_type=valid_time_type,
            valid_from_utc=valid_from_utc,
            valid_to_utc=valid_to_utc,
            valid_until_hint_utc=valid_until_hint_utc,
            status=status,
            raw_temporalize_json=JCS.canonicalize(decision_log)
        )

    def _get_time_mentions(self, message_id: str) -> List[TimeMentionRecord]:
        """Get time mentions for a message with caching."""
        if message_id not in self._time_mentions_cache:
            self._time_mentions_cache[message_id] = self.db.get_time_mentions_for_message(message_id)
        return self._time_mentions_cache[message_id]

    def _compute_alignment(self, assertion: AssertionRecord, time_mention: TimeMentionRecord) -> float:
        """Compute alignment score between assertion and time mention."""
        a_start = assertion.char_start
        a_end = assertion.char_end
        t_start = time_mention.char_start
        t_end = time_mention.char_end

        # Spanless assertion
        if a_start is None or a_end is None:
            # If only one time mention, give minimum alignment
            time_mentions = self._get_time_mentions(assertion.message_id)
            if len(time_mentions) == 1:
                return self.config.time_link_min_alignment
            return 0.0

        # Compute gap
        if a_end >= t_start and t_end >= a_start:
            # Overlapping or touching
            gap = 0
        elif a_end < t_start:
            gap = t_start - a_end
        else:
            gap = a_start - t_end

        # Apply proximity threshold
        if gap <= self.config.time_link_proximity_chars:
            return 1.0 / (1.0 + gap)
        return 0.0

    def _compute_eligibility(
        self,
        assertion: AssertionRecord,
        valid_from_utc: Optional[str],
        has_explicit_valid_time: bool
    ) -> Tuple[str, List[str]]:
        """
        Compute eligibility for temporal operations and initial status (§5.5.2).
        Returns (status, list of ineligibility reasons).
        """
        reasons = []

        # Check modality
        if assertion.modality not in ('state', 'fact', 'preference'):
            reasons.append(f"MODALITY_EXCLUDED:{assertion.modality}")

        # Check polarity
        if assertion.polarity != 'positive':
            reasons.append(f"NEGATIVE_POLARITY:{assertion.polarity}")

        # Check confidence
        if assertion.confidence_final < self.config.threshold_close:
            reasons.append(f"CONFIDENCE_BELOW_THRESHOLD:{assertion.confidence_final}")

        # Check user-grounded
        user_grounded = (
            assertion.asserted_role == 'user' or
            assertion.has_user_corroboration == 1
        )
        if not user_grounded:
            reasons.append("NOT_USER_GROUNDED")

        # Check valid_from
        if valid_from_utc is None:
            reasons.append("VALID_FROM_NULL")

        # Check explicit valid time (§5.5.2 - key change from spec)
        if not has_explicit_valid_time:
            reasons.append("NONEXPLICIT_TIME_SOURCE")

        if reasons:
            return LifecycleStatus.INELIGIBLE, reasons
        return LifecycleStatus.ACTIVE, reasons

    def _apply_correction_supersessions(self) -> int:
        """Apply correction supersessions from Stage 4."""
        count = 0

        for assertion_id, superseded_by_id in self.db.iter_correction_supersessions():
            # Validate the superseding assertion exists
            if not self._assertion_exists(superseded_by_id):
                logger.warning(f"Skipping correction supersession: {assertion_id} -> {superseded_by_id} (target assertion does not exist)")
                continue

            updated = self.db.update_temporalized_status(
                assertion_id=assertion_id,
                new_status=LifecycleStatus.SUPERSEDED,
                temporal_superseded_by=superseded_by_id,
                append_json={"event": "CORRECTION_SUPERSESSION_FROM_STAGE4", "superseded_by": superseded_by_id, "timestamp": self.stage_started_at_utc},
            )
            if updated:
                count += 1

        return count

    def _assertion_exists(self, assertion_id: str) -> bool:
        """Check if an assertion exists."""
        row = self.db.fetchone("SELECT 1 FROM assertions WHERE assertion_id = ?", (assertion_id,))
        return row is not None

    def _apply_retractions(self) -> int:
        """Apply retractions from Stage 4."""
        count = 0

        for retraction in self.db.iter_retractions_ordered():
            # Check preconditions
            if retraction.confidence < self.config.threshold_close:
                continue
            if retraction.retraction_message_role != 'user':
                continue

            target_ids: List[str] = []

            # Find target
            if retraction.target_assertion_id:
                target_ids = [retraction.target_assertion_id]
            elif retraction.target_fact_key:
                target_ids = self.db.find_assertions_by_fact_key(
                    retraction.target_fact_key, exclude_retracted=True
                )

            if len(target_ids) == 0:
                continue
            elif len(target_ids) > 1:
                # Create conflict group
                conflict_key = JCS.canonicalize([
                    "retract_ambig",
                    retraction.retraction_id,
                    retraction.target_fact_key
                ])
                conflict_id = self.id_generator.generate(["conflict", conflict_key])

                conflict = ConflictGroup(
                    conflict_group_id=conflict_id,
                    conflict_type=ConflictType.RETRACTION_TARGET_NOT_UNIQUE,
                    conflict_key=conflict_key,
                    detected_at_utc=self.stage_started_at_utc,
                    raw_conflict_json=JCS.canonicalize({
                        "retraction_id": retraction.retraction_id,
                        "target_fact_key": retraction.target_fact_key,
                        "candidate_assertion_ids": target_ids
                    }),
                    member_assertion_ids=target_ids
                )
                self._conflicts.append(conflict)
                continue

            # Apply retraction
            target_id = target_ids[0]
            updated = self.db.update_temporalized_status(
                assertion_id=target_id,
                new_status=LifecycleStatus.RETRACTED,
                retracted_by=retraction.retraction_id,
                append_json={
                    "event": "RETRACTION_APPLIED",
                    "retraction_id": retraction.retraction_id,
                    "timestamp": self.stage_started_at_utc
                }
            )
            if updated:
                count += 1

        return count

    def _apply_negations(self) -> int:
        """Apply negation closures."""
        count = 0

        for negation in self.db.iter_negative_assertions_ordered():
            # Check preconditions
            user_grounded = (
                negation.asserted_role == 'user' or
                negation.has_user_corroboration == 1
            )
            if not user_grounded:
                continue
            if negation.confidence_final < self.config.threshold_close:
                continue

            # Get negation's valid_from
            neg_temporal = self.db.get_temporalized_assertion(negation.assertion_id)
            if not neg_temporal or not neg_temporal.get("valid_from_utc"):
                continue
            neg_valid_from = neg_temporal["valid_from_utc"]

            # Find targets
            target_ids = self.db.find_negation_targets(
                negation.subject_entity_id,
                negation.predicate_id,
                negation.object_signature
            )

            if len(target_ids) == 0:
                continue
            elif len(target_ids) > 1:
                # Create conflict group
                conflict_key = JCS.canonicalize([
                    "neg",
                    negation.subject_entity_id,
                    negation.predicate_id,
                    negation.object_signature,
                    neg_valid_from if neg_valid_from else "__NULL__"
                ])
                conflict_id = self.id_generator.generate(["conflict", conflict_key])

                conflict = ConflictGroup(
                    conflict_group_id=conflict_id,
                    conflict_type=ConflictType.NEGATION_AMBIGUOUS,
                    conflict_key=conflict_key,
                    detected_at_utc=self.stage_started_at_utc,
                    raw_conflict_json=JCS.canonicalize({
                        "negation_assertion_id": negation.assertion_id,
                        "target_assertion_ids": target_ids
                    }),
                    member_assertion_ids=target_ids + [negation.assertion_id]
                )
                self._conflicts.append(conflict)
                continue

            # Apply negation closure
            target_id = target_ids[0]
            target_temporal = self.db.get_temporalized_assertion(target_id)

            # Compute new valid_to
            existing_valid_to = target_temporal.get("valid_to_utc") if target_temporal else None
            new_valid_to = TimestampUtils.min_time(existing_valid_to, neg_valid_from)

            updated = self.db.update_temporalized_status(
                assertion_id=target_id,
                new_status=LifecycleStatus.NEGATED,
                negated_by=negation.assertion_id,
                valid_to_utc=new_valid_to,
                append_json={
                    "event": "NEGATION_CLOSURE_APPLIED",
                    "negation_assertion_id": negation.assertion_id,
                    "valid_to_utc": new_valid_to,
                    "timestamp": self.stage_started_at_utc
                }
            )
            if updated:
                count += 1

        return count

    def _compute_effective_tier(self, assertion: dict) -> int:
        """
        Compute effective detection tier for an assertion (§5.10.3).
        Returns min(subject_detection_tier, object_detection_tier ?? 5).
        """
        subject_tier = assertion.get("subject_detection_tier")
        object_tier = assertion.get("object_detection_tier")

        # Use sentinel for NULL values
        effective_subject = subject_tier if subject_tier is not None else self.DETECTION_TIER_NULL_SENTINEL
        effective_object = object_tier if object_tier is not None else self.DETECTION_TIER_NULL_SENTINEL

        return min(effective_subject, effective_object)

    def _apply_functional_invalidation(self) -> int:
        """Apply functional invalidation rules with detector tier tiebreaking (§5.10)."""
        count = 0

        for subject_id, predicate_id, assertions in self.db.get_functional_groups():
            if not assertions:
                continue

            # Get subject entity type
            subject_entity_type = assertions[0]["subject_entity_type"]

            # Look up applicable rule
            rule = self.db.get_invalidation_rule(predicate_id, subject_entity_type)
            if not rule or not rule.is_functional:
                continue
            if rule.invalidation_policy != InvalidationPolicy.CLOSE_PREVIOUS_ON_NEWER_STATE:
                continue

            # Check eligibility for all assertions (§5.10.2)
            eligible_assertions = []
            for a in assertions:
                # Re-check eligibility (must be active and have explicit time)
                temporal = self.db.get_temporalized_assertion(a["assertion_id"])
                if temporal and temporal["status"] == LifecycleStatus.ACTIVE:
                    raw_json = json.loads(temporal["raw_temporalize_json"]) if temporal.get("raw_temporalize_json") else {}
                    if raw_json.get("has_explicit_valid_time", False):
                        eligible_assertions.append(a)

            if len(eligible_assertions) < 2:
                continue

            # Group by timepoint
            timepoint_groups: Dict[str, List[dict]] = {}
            for a in eligible_assertions:
                t = a["valid_from_utc"]
                if t not in timepoint_groups:
                    timepoint_groups[t] = []
                timepoint_groups[t].append(a)

            # Process each timepoint
            representatives: List[Tuple[str, dict]] = []  # (timepoint, winner)

            for t, group in sorted(timepoint_groups.items()):
                # Partition by object signature
                by_object: Dict[str, List[dict]] = {}
                for a in group:
                    sig = a["object_signature"]
                    if sig not in by_object:
                        by_object[sig] = []
                    by_object[sig].append(a)

                if len(by_object) > 1:
                    # Object disagreement at same timepoint (§5.10.3 step 2)
                    conflict_key = JCS.canonicalize([
                        "obj_disagree", subject_id, predicate_id, t
                    ])
                    conflict_id = self.id_generator.generate(["conflict", conflict_key])

                    all_ids = [a["assertion_id"] for a in group]
                    conflict = ConflictGroup(
                        conflict_group_id=conflict_id,
                        conflict_type=ConflictType.OBJECT_DISAGREEMENT,
                        conflict_key=conflict_key,
                        detected_at_utc=self.stage_started_at_utc,
                        raw_conflict_json=JCS.canonicalize({
                            "subject_entity_id": subject_id,
                            "predicate_id": predicate_id,
                            "timepoint": t,
                            "object_signatures": list(by_object.keys())
                        }),
                        member_assertion_ids=all_ids
                    )
                    self._conflicts.append(conflict)

                    # Mark all as conflicted
                    for a in group:
                        self.db.update_temporalized_status(
                            assertion_id=a["assertion_id"],
                            new_status=LifecycleStatus.CONFLICTED,
                            append_json={
                                "event": "OBJECT_DISAGREEMENT_CONFLICT",
                                "timepoint": t,
                                "timestamp": self.stage_started_at_utc
                            }
                        )
                    continue

                # Single object signature
                same_sig_assertions = list(by_object.values())[0]

                if len(same_sig_assertions) > 1:
                    # Same-time duplicates (informational) (§5.10.3 step 3)
                    conflict_key = JCS.canonicalize([
                        "same_time_dup", subject_id, predicate_id,
                        same_sig_assertions[0]["object_signature"], t
                    ])
                    conflict_id = self.id_generator.generate(["conflict", conflict_key])

                    all_ids = [a["assertion_id"] for a in same_sig_assertions]
                    conflict = ConflictGroup(
                        conflict_group_id=conflict_id,
                        conflict_type=ConflictType.SAME_TIME_DUPLICATES,
                        conflict_key=conflict_key,
                        detected_at_utc=self.stage_started_at_utc,
                        raw_conflict_json=JCS.canonicalize({
                            "subject_entity_id": subject_id,
                            "predicate_id": predicate_id,
                            "timepoint": t,
                            "assertion_ids": all_ids
                        }),
                        member_assertion_ids=all_ids
                    )
                    self._conflicts.append(conflict)

                # Select representative W(t) deterministically (§5.10.3 step 4)
                # Sort by: confidence_final DESC, min_tier ASC, subject_detection_tier ASC, assertion_id ASC
                sorted_assertions = sorted(
                    same_sig_assertions,
                    key=lambda a: (
                        -a["confidence_final"],
                        self._compute_effective_tier(a),
                        a.get("subject_detection_tier") if a.get("subject_detection_tier") is not None else self.DETECTION_TIER_NULL_SENTINEL,
                        a["assertion_id"]
                    )
                )
                winner = sorted_assertions[0]
                representatives.append((t, winner))

            # Walk timepoints to supersede (§5.10.4)
            if len(representatives) < 2:
                continue

            representatives.sort(key=lambda x: (x[0], x[1]["assertion_id"]))
            prev_t, prev_a = representatives[0]
            prev_temporal = self.db.get_temporalized_assertion(prev_a["assertion_id"])
            prev_status = prev_temporal["status"] if prev_temporal else "active"

            for cur_t, cur_a in representatives[1:]:
                cur_temporal = self.db.get_temporalized_assertion(cur_a["assertion_id"])
                cur_status = cur_temporal["status"] if cur_temporal else "active"

                if prev_a["object_signature"] == cur_a["object_signature"]:
                    # Same state continues
                    prev_t, prev_a, prev_status = cur_t, cur_a, cur_status
                    continue

                # Object differs - check if we can supersede
                if prev_status == LifecycleStatus.CONFLICTED or cur_status == LifecycleStatus.CONFLICTED:
                    # Create transition conflict
                    conflict_key = JCS.canonicalize([
                        "trans_conflict", subject_id, predicate_id, prev_t, cur_t
                    ])
                    conflict_id = self.id_generator.generate(["conflict", conflict_key])

                    conflict = ConflictGroup(
                        conflict_group_id=conflict_id,
                        conflict_type=ConflictType.OBJ_TRANSITION_CONFLICT,
                        conflict_key=conflict_key,
                        detected_at_utc=self.stage_started_at_utc,
                        raw_conflict_json=JCS.canonicalize({
                            "subject_entity_id": subject_id,
                            "predicate_id": predicate_id,
                            "prev_timepoint": prev_t,
                            "cur_timepoint": cur_t
                        }),
                        member_assertion_ids=[prev_a["assertion_id"], cur_a["assertion_id"]]
                    )
                    self._conflicts.append(conflict)
                    prev_t, prev_a, prev_status = cur_t, cur_a, cur_status
                    continue

                # Check confidence margin and detector tier tiebreak (§5.10.4)
                confidence_diff = cur_a["confidence_final"] - prev_a["confidence_final"]
                supersession_reason = None
                should_supersede = False
                should_conflict = False

                # Compute detection tiers for tiebreaking
                cur_tier = self._compute_effective_tier(cur_a)
                prev_tier = self._compute_effective_tier(prev_a)

                if confidence_diff >= self.config.confidence_supersession_margin:
                    # Clear confidence winner
                    should_supersede = True
                    supersession_reason = "CONFIDENCE_MARGIN"
                elif self.config.use_detector_tier_tiebreak and abs(confidence_diff) < self.config.confidence_supersession_margin:
                    # Confidence too close, use detector tier tiebreak
                    if cur_tier < prev_tier:
                        # cur has higher reliability (lower tier number)
                        should_supersede = True
                        supersession_reason = "DETECTOR_TIER_TIEBREAK"
                    elif cur_tier > prev_tier:
                        # prev has higher reliability - treat as conflict
                        should_conflict = True
                    else:
                        # Same tier and confidence too close - conflict
                        should_conflict = True
                else:
                    # Confidence too close without tier tiebreak
                    should_conflict = True

                if should_supersede:
                    # Supersede prev at time cur_t
                    existing_valid_to = prev_temporal.get("valid_to_utc") if prev_temporal else None
                    new_valid_to = TimestampUtils.min_time(existing_valid_to, cur_t)

                    updated = self.db.update_temporalized_status(
                        assertion_id=prev_a["assertion_id"],
                        new_status=LifecycleStatus.SUPERSEDED,
                        temporal_superseded_by=cur_a["assertion_id"],
                        rule_id=rule.rule_id,
                        valid_to_utc=new_valid_to,
                        append_json={
                            "event": "FUNCTIONAL_SUPERSESSION",
                            "superseded_by": cur_a["assertion_id"],
                            "rule_id": rule.rule_id,
                            "valid_to_utc": new_valid_to,
                            "supersession_reason": supersession_reason,
                            "confidence_diff": confidence_diff,
                            "prev_tier": prev_tier,
                            "cur_tier": cur_tier,
                            "timestamp": self.stage_started_at_utc
                        }
                    )
                    if updated:
                        count += 1
                elif should_conflict:
                    # Confidence too close - create conflict
                    conflict_key = JCS.canonicalize([
                        "conf_close", subject_id, predicate_id,
                        prev_a["assertion_id"], cur_a["assertion_id"]
                    ])
                    conflict_id = self.id_generator.generate(["conflict", conflict_key])

                    conflict = ConflictGroup(
                        conflict_group_id=conflict_id,
                        conflict_type=ConflictType.OBJ_CONFIDENCE_TOO_CLOSE,
                        conflict_key=conflict_key,
                        detected_at_utc=self.stage_started_at_utc,
                        raw_conflict_json=JCS.canonicalize({
                            "prev_assertion_id": prev_a["assertion_id"],
                            "prev_confidence": prev_a["confidence_final"],
                            "prev_detection_tier": prev_tier,
                            "cur_assertion_id": cur_a["assertion_id"],
                            "cur_confidence": cur_a["confidence_final"],
                            "cur_detection_tier": cur_tier,
                            "margin_required": self.config.confidence_supersession_margin,
                            "confidence_diff": confidence_diff,
                            "subject_detection_tiers": {
                                "prev": prev_a.get("subject_detection_tier"),
                                "cur": cur_a.get("subject_detection_tier")
                            },
                            "object_detection_tiers": {
                                "prev": prev_a.get("object_detection_tier"),
                                "cur": cur_a.get("object_detection_tier")
                            }
                        }),
                        member_assertion_ids=[prev_a["assertion_id"], cur_a["assertion_id"]]
                    )
                    self._conflicts.append(conflict)

                    # Mark both as conflicted
                    for aid in [prev_a["assertion_id"], cur_a["assertion_id"]]:
                        self.db.update_temporalized_status(
                            assertion_id=aid,
                            new_status=LifecycleStatus.CONFLICTED,
                            append_json={
                                "event": "OBJ_CONFIDENCE_TOO_CLOSE",
                                "timestamp": self.stage_started_at_utc
                            }
                        )

                prev_t, prev_a = cur_t, cur_a
                prev_temporal = self.db.get_temporalized_assertion(prev_a["assertion_id"])
                prev_status = prev_temporal["status"] if prev_temporal else "active"

        return count

    def _finalize_conflicts(self):
        """Persist all collected conflict groups."""
        for conflict in self._conflicts:
            self.db.insert_conflict_group(conflict)


def run_stage5(config: Stage5Config) -> Dict[str, Any]:
    """Run Stage 5 pipeline on existing database."""
    pipeline = TemporalReasoningPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 5: Temporal Reasoning Layer")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: ../data/output/kg.db)"
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=Path("../data/metadata/invalidation_rules.yaml"),
        help="Path to invalidation_rules.yaml (optional)"
    )
    parser.add_argument(
        "--time-link-proximity",
        type=int,
        default=200,
        help="Max character distance for time linking (default: 200)"
    )
    parser.add_argument(
        "--threshold-close",
        type=float,
        default=0.7,
        help="Confidence threshold for closing assertions (default: 0.7)"
    )
    parser.add_argument(
        "--confidence-margin",
        type=float,
        default=0.01,
        help="Minimum confidence advantage for supersession (default: 0.01)"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable asserted_at fallback for valid_from"
    )
    parser.add_argument(
        "--no-detector-tier-tiebreak",
        action="store_true",
        help="Disable detector tier as tie-breaker in supersession"
    )
    parser.add_argument(
        "--use-salience-tiebreak",
        action="store_true",
        help="Enable entity salience in conflict resolution tie-breaks"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="550e8400-e29b-41d4-a716-446655440000",
        help="UUID namespace for ID generation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve rules path
    rules_path = args.rules if args.rules.exists() else None

    config = Stage5Config(
        output_file_path=args.db,
        invalidation_rules_path=rules_path,
        id_namespace=args.namespace,
        time_link_proximity_chars=args.time_link_proximity,
        threshold_close=args.threshold_close,
        confidence_supersession_margin=args.confidence_margin,
        fallback_valid_from_asserted=not args.no_fallback,
        use_detector_tier_tiebreak=not args.no_detector_tier_tiebreak,
        use_salience_conflict_tiebreak=args.use_salience_tiebreak
    )

    stats = run_stage5(config)

    logger.info("\n=== Stage 5 Summary ===")
    logger.info(f"Total assertions: {stats['assertions_total']}")
    logger.info(f"Assertions temporalized: {stats['assertions_temporalized']}")
    logger.info(f"Corrections applied: {stats['corrections_applied']}")
    logger.info(f"Retractions applied: {stats['retractions_applied']}")
    logger.info(f"Negations applied: {stats['negations_applied']}")
    logger.info(f"Functional supersessions: {stats['functional_supersessions']}")

    logger.info("\nStatus distribution:")
    for status, count in sorted(stats.get('status_counts', {}).items()):
        logger.info(f"  {status}: {count}")

    logger.info("\nConflicts by type:")
    for ctype, count in sorted(stats.get('conflicts_by_type', {}).items()):
        logger.info(f"  {ctype}: {count}")