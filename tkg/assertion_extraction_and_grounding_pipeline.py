"""
Stage 4: Assertion Extraction & Grounding Layer

Converts Stage 1-3 evidence (messages + refined entities + time mentions) into
auditable, replayable assertions with deterministic IDs, conservative span handling,
and role-aware trust.

Outputs:
- predicates: normalized relation vocabulary
- assertions: grounded semantic claims with entity/literal objects
- retractions: explicit negations or corrections linked to prior assertions
- Optional LLM extraction artifacts (llm_extraction_runs, llm_extraction_calls)
"""

import hashlib
import json
import logging
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Tuple, Set, Callable
import unicodedata

try:
    import pendulum
except ImportError:
    pendulum = None  # Will use stdlib datetime if pendulum not available


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


# =============================================================================
# UTILITIES
# =============================================================================

class JCS:
    """
    JSON Canonicalization Scheme (RFC 8785) implementation.
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
                encoded.append(str(c))

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
    ISO_UTC_MILLIS = "%Y-%m-%dT%H:%M:%S.%f"

    @staticmethod
    def now_utc() -> str:
        """Get current UTC time in canonical format."""
        if pendulum:
            return pendulum.now("UTC").format("YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
        else:
            dt = datetime.utcnow()
            return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

    @staticmethod
    def normalize_to_utc(timestamp: Any, source_tz: str | None = None) -> str | None:
        """Normalize any timestamp format to canonical UTC string."""
        if timestamp is None:
            return None

        try:
            if pendulum:
                if isinstance(timestamp, (int, float)):
                    dt = pendulum.from_timestamp(timestamp, tz="UTC")
                    return dt.format("YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")

                if isinstance(timestamp, datetime):
                    dt = pendulum.instance(timestamp)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=pendulum.timezone(source_tz or "UTC"))
                    return dt.in_timezone("UTC").format("YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")

                if isinstance(timestamp, str):
                    s = timestamp.strip()
                    dt = pendulum.parse(s, tz=source_tz or "UTC", strict=False)
                    return dt.in_timezone("UTC").format("YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
            else:
                # Fallback to stdlib
                if isinstance(timestamp, (int, float)):
                    dt = datetime.utcfromtimestamp(timestamp)
                    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

                if isinstance(timestamp, datetime):
                    return timestamp.strftime("%Y-%m-%dT%H:%M:%S.") + f"{timestamp.microsecond // 1000:03d}Z"

                if isinstance(timestamp, str):
                    # Try to parse ISO format
                    s = timestamp.strip().rstrip('Z')
                    dt = datetime.fromisoformat(s)
                    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"

            return None
        except Exception:
            return None

    @staticmethod
    def parse_iso(iso_string: str) -> datetime | None:
        """Parse ISO string to datetime."""
        try:
            if pendulum:
                return pendulum.parse(iso_string, strict=False)
            else:
                s = iso_string.strip().rstrip('Z')
                return datetime.fromisoformat(s)
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


class TextUtils:
    """Text processing utilities."""

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse whitespace to single spaces and trim."""
        return ' '.join(text.split())

    @staticmethod
    def nfkc_normalize(text: str) -> str:
        """Apply Unicode NFKC normalization."""
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str) -> float:
        """
        Compute Jaro-Winkler similarity between two strings.
        Returns value in [0, 1] where 1 is exact match.
        """
        if s1 == s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Jaro similarity
        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        if match_distance < 0:
            match_distance = 0

        s1_matches = [False] * len1
        s2_matches = [False] * len2
        matches = 0
        transpositions = 0

        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)
            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

        # Winkler modification
        prefix = 0
        for i in range(min(len1, len2, 4)):
            if s1[i] == s2[i]:
                prefix += 1
            else:
                break

        return jaro + prefix * 0.1 * (1 - jaro)


# =============================================================================
# ENUMS
# =============================================================================

class Modality(StrEnum):
    """Assertion modality types."""
    STATE = "state"
    FACT = "fact"
    PREFERENCE = "preference"
    INTENTION = "intention"
    QUESTION = "question"


class Polarity(StrEnum):
    """Assertion polarity."""
    POSITIVE = "positive"
    NEGATIVE = "negative"


class ObjectValueType(StrEnum):
    """Literal object value types."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    JSON = "json"


class ExtractionMethod(StrEnum):
    """How an assertion was extracted."""
    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"


class MaskingStrategy(StrEnum):
    """Text masking strategy for excluded regions."""
    LENGTH_PRESERVING = "length_preserving"
    MARKER = "marker"
    REMOVE = "remove"


class RetractionType(StrEnum):
    """Types of retractions."""
    FULL = "full"
    CORRECTION = "correction"
    TEMPORAL_BOUND = "temporal_bound"


class SupersessionType(StrEnum):
    """How an assertion was superseded."""
    RETRACTION = "retraction"
    CORRECTION = "correction"
    TEMPORAL_END = "temporal_end"


class UpsertPolicy(StrEnum):
    """Assertion upsert policies."""
    KEEP_HIGHEST_CONFIDENCE = "keep_highest_confidence"
    KEEP_FIRST = "keep_first"
    KEEP_ALL = "keep_all"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Stage4Config:
    """Configuration for Stage 4 pipeline."""
    # Database
    output_file_path: Path = field(default_factory=lambda: Path("kg.db"))
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"

    # Exclusion handling
    ignore_markdown_blockquotes: bool = False
    masking_strategy: str = "length_preserving"

    # Extraction settings
    enable_llm_assertion_extraction: bool = False
    llm_model_name: str = "claude-sonnet-4-20250514"
    llm_model_version: str = "1.0"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_seed: Optional[int] = None
    llm_max_retries: int = 3
    llm_multi_run_count: int = 1
    k_context: int = 5
    min_extractable_chars: int = 10

    # Entity linking
    enable_fuzzy_entity_linking: bool = True
    threshold_link_string_sim: float = 0.85
    predicate_similarity_threshold: float = 0.9

    # Trust weights
    trust_weight_user: float = 1.0
    trust_weight_assistant_corroborated: float = 0.9
    trust_weight_assistant_uncorroborated: float = 0.5

    # Corroboration
    coref_window_size: int = 10

    # Persistence
    assertion_upsert_policy: str = "keep_highest_confidence"
    update_entity_assertion_counts: bool = False


@dataclass
class ExclusionRange:
    """A range to exclude from extraction."""
    char_start: int
    char_end: int
    reason: str = ""


@dataclass
class AssertionCandidate:
    """A candidate assertion before grounding."""
    subject: str
    predicate_label: str
    object_entity_ref: Optional[str] = None
    object_literal_type: Optional[str] = None
    object_literal_value: Optional[Any] = None
    modality: str = "state"
    polarity: str = "positive"
    temporal_qualifier_surface: Optional[str] = None
    temporal_qualifier_type: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    quote: Optional[str] = None
    confidence: float = 0.8
    extraction_method: str = "rule_based"
    extraction_model: Optional[str] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundedAssertion:
    """A fully grounded assertion ready for persistence."""
    assertion_id: str
    message_id: str
    assertion_key: str
    fact_key: str
    subject_entity_id: str
    predicate_id: str
    object_entity_id: Optional[str]
    object_value_type: Optional[str]
    object_value: Optional[str]
    object_signature: str
    temporal_qualifier_type: Optional[str]
    temporal_qualifier_id: Optional[str]
    modality: str
    polarity: str
    asserted_role: str
    asserted_at_utc: Optional[str]
    confidence_extraction: float
    confidence_final: float
    has_user_corroboration: int
    superseded_by_assertion_id: Optional[str]
    supersession_type: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    surface_text: Optional[str]
    extraction_method: str
    extraction_model: Optional[str]
    raw_assertion_json: str


@dataclass
class RetractionCandidate:
    """A detected retraction before linking."""
    retraction_type: str
    target_clause: str
    replacement_clause: Optional[str]
    char_start: Optional[int]
    char_end: Optional[int]
    surface_text: Optional[str]
    confidence: float
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageContext:
    """Context for processing a message."""
    message_id: str
    conversation_id: str
    role: str
    text_raw: str
    created_at_utc: Optional[str]
    order_index: int
    exclusion_ranges: List[ExclusionRange]
    context_messages: List[Dict[str, Any]]
    context_entities: Dict[str, Dict[str, Any]]
    context_times: List[Dict[str, Any]]


@dataclass
class ExtractionPattern:
    """A pattern for rule-based extraction."""
    pattern_id: str
    regex: re.Pattern
    handler: Callable
    confidence: float
    priority: int


# =============================================================================
# DATABASE
# =============================================================================

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
        """Execute SQL for multiple parameter sets."""
        return self.connection.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute and fetch one row."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute and fetch all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()


class Stage4Database(Database):
    """Stage 4 specific database operations."""

    REQUIRED_TABLES = ["conversations", "messages", "message_parts", "entities",
                       "entity_mentions", "time_mentions"]

    def check_required_tables(self):
        """Check that required tables from prior stages exist."""
        for table in self.REQUIRED_TABLES:
            cursor = self.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            if cursor.fetchone() is None:
                raise RuntimeError(f"Required table '{table}' not found. Run prior stages first.")
        logger.info("All required tables present")

    def initialize_stage4_schema(self):
        """Create Stage 4 tables."""
        logger.info("Initializing Stage 4 schema")

        # predicates table
        self.execute("""
            CREATE TABLE IF NOT EXISTS predicates (
                predicate_id TEXT PRIMARY KEY,
                canonical_label TEXT NOT NULL,
                canonical_label_norm TEXT NOT NULL,
                inverse_label TEXT,
                category TEXT,
                arity INTEGER NOT NULL DEFAULT 2,
                value_type_constraint TEXT,
                first_seen_at_utc TEXT,
                assertion_count INTEGER NOT NULL DEFAULT 0,
                raw_predicate_json TEXT
            )
        """)
        self.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_predicates_label_norm 
            ON predicates(canonical_label_norm)
        """)

        # assertions table
        self.execute("""
            CREATE TABLE IF NOT EXISTS assertions (
                assertion_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL REFERENCES messages(message_id),
                assertion_key TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                subject_entity_id TEXT NOT NULL REFERENCES entities(entity_id),
                predicate_id TEXT NOT NULL REFERENCES predicates(predicate_id),
                object_entity_id TEXT REFERENCES entities(entity_id),
                object_value_type TEXT,
                object_value TEXT,
                object_signature TEXT NOT NULL,
                temporal_qualifier_type TEXT,
                temporal_qualifier_id TEXT REFERENCES time_mentions(time_mention_id),
                modality TEXT NOT NULL,
                polarity TEXT NOT NULL,
                asserted_role TEXT NOT NULL,
                asserted_at_utc TEXT,
                confidence_extraction REAL NOT NULL,
                confidence_final REAL NOT NULL,
                has_user_corroboration INTEGER NOT NULL DEFAULT 0,
                superseded_by_assertion_id TEXT REFERENCES assertions(assertion_id),
                supersession_type TEXT,
                char_start INTEGER,
                char_end INTEGER,
                surface_text TEXT,
                extraction_method TEXT NOT NULL,
                extraction_model TEXT,
                raw_assertion_json TEXT NOT NULL
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS idx_assertions_message ON assertions(message_id)")
        self.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_assertions_key ON assertions(assertion_key)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_assertions_fact_key ON assertions(fact_key)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_assertions_subject ON assertions(subject_entity_id)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_assertions_predicate ON assertions(predicate_id)")
        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_assertions_object_entity 
            ON assertions(object_entity_id) WHERE object_entity_id IS NOT NULL
        """)
        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_assertions_temporal_qualifier 
            ON assertions(temporal_qualifier_id) WHERE temporal_qualifier_id IS NOT NULL
        """)

        # llm_extraction_runs table
        self.execute("""
            CREATE TABLE IF NOT EXISTS llm_extraction_runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                config_json TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                messages_processed INTEGER DEFAULT 0,
                assertions_extracted INTEGER DEFAULT 0,
                raw_stats_json TEXT
            )
        """)

        # llm_extraction_calls table
        self.execute("""
            CREATE TABLE IF NOT EXISTS llm_extraction_calls (
                call_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES llm_extraction_runs(run_id),
                message_id TEXT NOT NULL REFERENCES messages(message_id),
                request_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                call_timestamp_utc TEXT NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                seed_honored INTEGER,
                parse_success INTEGER NOT NULL,
                raw_io_json TEXT NOT NULL
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_run ON llm_extraction_calls(run_id)")
        self.execute("CREATE INDEX IF NOT EXISTS idx_llm_calls_message ON llm_extraction_calls(message_id)")

        # retractions table
        self.execute("""
            CREATE TABLE IF NOT EXISTS retractions (
                retraction_id TEXT PRIMARY KEY,
                retraction_message_id TEXT NOT NULL REFERENCES messages(message_id),
                target_assertion_id TEXT REFERENCES assertions(assertion_id),
                target_fact_key TEXT,
                retraction_type TEXT NOT NULL,
                replacement_assertion_id TEXT REFERENCES assertions(assertion_id),
                confidence REAL NOT NULL,
                char_start INTEGER,
                char_end INTEGER,
                surface_text TEXT,
                raw_retraction_json TEXT NOT NULL
            )
        """)
        self.execute("CREATE INDEX IF NOT EXISTS idx_retractions_message ON retractions(retraction_message_id)")
        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_retractions_target 
            ON retractions(target_assertion_id) WHERE target_assertion_id IS NOT NULL
        """)
        self.execute("""
            CREATE INDEX IF NOT EXISTS idx_retractions_fact_key 
            ON retractions(target_fact_key) WHERE target_fact_key IS NOT NULL
        """)

        logger.info("Stage 4 schema initialized")

    # --- Message operations ---

    def get_eligible_messages(self, min_chars: int = 10) -> List[sqlite3.Row]:
        """Get messages eligible for extraction, in deterministic order."""
        return self.fetchall("""
            SELECT 
                m.message_id, m.conversation_id, m.role, m.text_raw, 
                m.created_at_utc, m.order_index, m.code_fence_ranges_json,
                m.blockquote_ranges_json
            FROM messages m
            WHERE m.role IN ('user', 'assistant')
              AND m.text_raw IS NOT NULL
              AND LENGTH(m.text_raw) >= ?
            ORDER BY m.conversation_id ASC, m.order_index ASC, m.message_id ASC
        """, (min_chars,))

    def get_context_messages(self, conversation_id: str, order_index: int, k: int) -> List[sqlite3.Row]:
        """Get k prior messages for context."""
        return self.fetchall("""
            SELECT message_id, role, text_raw, created_at_utc, order_index
            FROM messages
            WHERE conversation_id = ?
              AND order_index < ?
              AND text_raw IS NOT NULL
            ORDER BY order_index DESC
            LIMIT ?
        """, (conversation_id, order_index, k))

    # --- Entity operations ---

    def get_self_entity_id(self) -> Optional[str]:
        """Get the SELF entity ID."""
        row = self.fetchone(
            "SELECT entity_id FROM entities WHERE entity_type = 'PERSON' AND entity_key = '__SELF__'"
        )
        return row['entity_id'] if row else None

    def get_entity_by_id(self, entity_id: str) -> Optional[sqlite3.Row]:
        """Get entity by ID."""
        return self.fetchone("SELECT * FROM entities WHERE entity_id = ?", (entity_id,))

    def get_entity_by_canonical_name(self, name: str) -> Optional[sqlite3.Row]:
        """Get entity by canonical name (case-insensitive)."""
        return self.fetchone(
            "SELECT * FROM entities WHERE LOWER(canonical_name) = LOWER(?)",
            (name,)
        )

    def get_entity_by_key(self, entity_type: str, entity_key: str) -> Optional[sqlite3.Row]:
        """Get entity by type and key."""
        return self.fetchone(
            "SELECT * FROM entities WHERE entity_type = ? AND entity_key = ? AND status = 'active'",
            (entity_type, entity_key)
        )

    def get_all_active_entities(self) -> List[sqlite3.Row]:
        """Get all active entities."""
        return self.fetchall("SELECT * FROM entities WHERE status = 'active'")

    def search_entity_by_alias(self, alias: str) -> Optional[sqlite3.Row]:
        """Search for entity by alias in aliases_json."""
        # SQLite JSON search
        rows = self.fetchall("""
            SELECT * FROM entities 
            WHERE status = 'active' 
              AND aliases_json LIKE ?
        """, (f'%"{alias}"%',))

        # Verify exact match in JSON
        for row in rows:
            if row['aliases_json']:
                try:
                    aliases = json.loads(row['aliases_json'])
                    if alias in aliases:
                        return row
                except json.JSONDecodeError:
                    pass
        return None

    def create_entity(self, entity_id: str, entity_type: str, entity_key: str,
                      canonical_name: str) -> None:
        """Create a new entity."""
        now = TimestampUtils.now_utc()
        self.execute("""
            INSERT INTO entities (
                entity_id, entity_type, entity_key, canonical_name, aliases_json,
                status, first_seen_at_utc, last_seen_at_utc, mention_count, conversation_count
            ) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, 0, 0)
        """, (entity_id, entity_type, entity_key, canonical_name,
              JCS.canonicalize([canonical_name]), now, now))

    def get_entity_mentions_for_context(self, message_ids: List[str]) -> List[sqlite3.Row]:
        """Get entity mentions for a set of messages."""
        if not message_ids:
            return []
        placeholders = ','.join(['?'] * len(message_ids))
        return self.fetchall(f"""
            SELECT em.*, e.canonical_name, e.entity_type
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.entity_id
            WHERE em.message_id IN ({placeholders})
        """, tuple(message_ids))

    # --- Time mention operations ---

    def get_time_mentions_for_context(self, message_ids: List[str]) -> List[sqlite3.Row]:
        """Get resolved time mentions for a set of messages."""
        if not message_ids:
            return []
        placeholders = ','.join(['?'] * len(message_ids))
        return self.fetchall(f"""
            SELECT *
            FROM time_mentions
            WHERE message_id IN ({placeholders})
              AND resolved_type IN ('instant', 'interval')
            ORDER BY char_start ASC
        """, tuple(message_ids))

    def get_time_mention_by_surface(self, message_id: str, surface_text: str) -> Optional[sqlite3.Row]:
        """Get time mention by message and surface text."""
        return self.fetchone("""
            SELECT * FROM time_mentions
            WHERE message_id = ? AND surface_text = ?
            ORDER BY confidence DESC, char_start ASC
            LIMIT 1
        """, (message_id, surface_text))

    # --- Predicate operations ---

    def get_predicate_by_label_norm(self, label_norm: str) -> Optional[sqlite3.Row]:
        """Get predicate by normalized label."""
        return self.fetchone(
            "SELECT * FROM predicates WHERE canonical_label_norm = ?",
            (label_norm,)
        )

    def upsert_predicate(self, predicate_id: str, canonical_label: str,
                         canonical_label_norm: str, arity: int,
                         first_seen_at_utc: str) -> None:
        """Insert or update predicate."""
        existing = self.get_predicate_by_label_norm(canonical_label_norm)
        if existing:
            return  # Already exists

        raw_json = JCS.canonicalize({
            "predicate_id": predicate_id,
            "canonical_label": canonical_label,
            "canonical_label_norm": canonical_label_norm,
            "arity": arity
        })

        self.execute("""
            INSERT INTO predicates (
                predicate_id, canonical_label, canonical_label_norm, arity,
                first_seen_at_utc, raw_predicate_json
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (predicate_id, canonical_label, canonical_label_norm, arity,
              first_seen_at_utc, raw_json))

    # --- Assertion operations ---

    def get_assertion_by_key(self, assertion_key: str) -> Optional[sqlite3.Row]:
        """Get assertion by assertion_key."""
        return self.fetchone(
            "SELECT * FROM assertions WHERE assertion_key = ?",
            (assertion_key,)
        )

    def get_assertions_by_fact_key(self, fact_key: str) -> List[sqlite3.Row]:
        """Get assertions by fact_key."""
        return self.fetchall(
            "SELECT * FROM assertions WHERE fact_key = ?",
            (fact_key,)
        )

    def get_prior_user_assertions(self, conversation_id: str, order_index: int,
                                  window_size: int) -> List[sqlite3.Row]:
        """Get prior user assertions within window for corroboration."""
        return self.fetchall("""
            SELECT a.*
            FROM assertions a
            JOIN messages m ON a.message_id = m.message_id
            WHERE m.conversation_id = ?
              AND m.order_index >= ?
              AND m.order_index < ?
              AND a.asserted_role = 'user'
            ORDER BY m.order_index ASC, a.assertion_id ASC
        """, (conversation_id, max(0, order_index - window_size), order_index))

    def insert_assertion(self, assertion: GroundedAssertion) -> None:
        """Insert a new assertion."""
        self.execute("""
            INSERT INTO assertions (
                assertion_id, message_id, assertion_key, fact_key,
                subject_entity_id, predicate_id, object_entity_id,
                object_value_type, object_value, object_signature,
                temporal_qualifier_type, temporal_qualifier_id,
                modality, polarity, asserted_role, asserted_at_utc,
                confidence_extraction, confidence_final, has_user_corroboration,
                superseded_by_assertion_id, supersession_type,
                char_start, char_end, surface_text,
                extraction_method, extraction_model, raw_assertion_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assertion.assertion_id, assertion.message_id, assertion.assertion_key,
            assertion.fact_key, assertion.subject_entity_id, assertion.predicate_id,
            assertion.object_entity_id, assertion.object_value_type, assertion.object_value,
            assertion.object_signature, assertion.temporal_qualifier_type,
            assertion.temporal_qualifier_id, assertion.modality, assertion.polarity,
            assertion.asserted_role, assertion.asserted_at_utc,
            assertion.confidence_extraction, assertion.confidence_final,
            assertion.has_user_corroboration, assertion.superseded_by_assertion_id,
            assertion.supersession_type, assertion.char_start, assertion.char_end,
            assertion.surface_text, assertion.extraction_method, assertion.extraction_model,
            assertion.raw_assertion_json
        ))

    def update_assertion_confidence(self, assertion_id: str, confidence_final: float) -> None:
        """Update assertion confidence."""
        self.execute(
            "UPDATE assertions SET confidence_final = ? WHERE assertion_id = ?",
            (confidence_final, assertion_id)
        )

    def update_assertion_supersession(self, assertion_id: str, superseded_by: str,
                                      supersession_type: str) -> None:
        """Mark assertion as superseded."""
        self.execute("""
            UPDATE assertions 
            SET superseded_by_assertion_id = ?, supersession_type = ?
            WHERE assertion_id = ?
        """, (superseded_by, supersession_type, assertion_id))

    # --- Retraction operations ---

    def insert_retraction(self, retraction_id: str, message_id: str,
                          target_assertion_id: Optional[str], target_fact_key: Optional[str],
                          retraction_type: str, replacement_assertion_id: Optional[str],
                          confidence: float, char_start: Optional[int],
                          char_end: Optional[int], surface_text: Optional[str],
                          raw_json: str) -> None:
        """Insert a retraction."""
        self.execute("""
            INSERT INTO retractions (
                retraction_id, retraction_message_id, target_assertion_id,
                target_fact_key, retraction_type, replacement_assertion_id,
                confidence, char_start, char_end, surface_text, raw_retraction_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (retraction_id, message_id, target_assertion_id, target_fact_key,
              retraction_type, replacement_assertion_id, confidence,
              char_start, char_end, surface_text, raw_json))

    # --- Stats operations ---

    def refresh_predicate_counts(self) -> None:
        """Recompute assertion counts for all predicates."""
        self.execute("""
            UPDATE predicates
            SET assertion_count = (
                SELECT COUNT(*) FROM assertions 
                WHERE assertions.predicate_id = predicates.predicate_id
            )
        """)

    # --- LLM run operations ---

    def insert_llm_run(self, run_id: str, model_name: str, model_version: str,
                       config_json: str, started_at_utc: str) -> None:
        """Insert LLM extraction run record."""
        self.execute("""
            INSERT INTO llm_extraction_runs (
                run_id, model_name, model_version, config_json, started_at_utc
            ) VALUES (?, ?, ?, ?, ?)
        """, (run_id, model_name, model_version, config_json, started_at_utc))

    def update_llm_run(self, run_id: str, completed_at_utc: str,
                       messages_processed: int, assertions_extracted: int,
                       raw_stats_json: str) -> None:
        """Update LLM run with completion info."""
        self.execute("""
            UPDATE llm_extraction_runs
            SET completed_at_utc = ?, messages_processed = ?,
                assertions_extracted = ?, raw_stats_json = ?
            WHERE run_id = ?
        """, (completed_at_utc, messages_processed, assertions_extracted,
              raw_stats_json, run_id))

    def insert_llm_call(self, call_id: str, run_id: str, message_id: str,
                        request_json: str, response_json: str,
                        call_timestamp_utc: str, retry_count: int,
                        seed_honored: Optional[int], parse_success: int,
                        raw_io_json: str) -> None:
        """Insert LLM API call record."""
        self.execute("""
            INSERT INTO llm_extraction_calls (
                call_id, run_id, message_id, request_json, response_json,
                call_timestamp_utc, retry_count, seed_honored, parse_success, raw_io_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (call_id, run_id, message_id, request_json, response_json,
              call_timestamp_utc, retry_count, seed_honored, parse_success, raw_io_json))


# =============================================================================
# EXTRACTION PATTERNS
# =============================================================================

class PatternRegistry:
    """Registry of extraction patterns."""

    def __init__(self, id_generator: IDGenerator):
        self.id_generator = id_generator
        self.patterns: List[ExtractionPattern] = []
        self._register_default_patterns()

    def _register_default_patterns(self):
        """Register default extraction patterns."""

        # Pattern: "I am <attribute>" / "I'm <attribute>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_attribute_am",
            regex=re.compile(
                r"\b(?:I\s+am|I'm)\s+(?:a\s+)?([A-Za-z][A-Za-z\s]{1,30}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_attribute,
            confidence=0.75,
            priority=10
        ))

        # Pattern: "I have <noun>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_has",
            regex=re.compile(
                r"\bI\s+have\s+(?:a\s+)?([A-Za-z][A-Za-z\s]{1,40}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_has,
            confidence=0.7,
            priority=20
        ))

        # Pattern: "I work at/for <org>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_works_at",
            regex=re.compile(
                r"\bI\s+work\s+(?:at|for)\s+([A-Z][A-Za-z\s&.,'-]{1,50}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_works_at,
            confidence=0.85,
            priority=5
        ))

        # Pattern: "I live in <location>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_lives_in",
            regex=re.compile(
                r"\bI\s+live\s+in\s+([A-Z][A-Za-z\s,'-]{1,50}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_lives_in,
            confidence=0.85,
            priority=5
        ))

        # Pattern: "My name is <name>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_name",
            regex=re.compile(
                r"\bMy\s+name\s+is\s+([A-Z][A-Za-z\s'-]{1,40}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_name,
            confidence=0.9,
            priority=1
        ))

        # Pattern: "I like/love/prefer <thing>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_preference",
            regex=re.compile(
                r"\bI\s+(?:like|love|prefer|enjoy)\s+([A-Za-z][A-Za-z\s]{1,40}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_preference,
            confidence=0.7,
            priority=15
        ))

        # Pattern: "I don't like/hate/dislike <thing>"
        self.patterns.append(ExtractionPattern(
            pattern_id="self_dispreference",
            regex=re.compile(
                r"\bI\s+(?:don't\s+like|hate|dislike)\s+([A-Za-z][A-Za-z\s]{1,40}?)\b",
                re.IGNORECASE
            ),
            handler=self._handle_self_dispreference,
            confidence=0.7,
            priority=15
        ))

        # Pattern: "<Person> is my <relation>"
        self.patterns.append(ExtractionPattern(
            pattern_id="person_relation",
            regex=re.compile(
                r"\b([A-Z][A-Za-z]+)\s+is\s+my\s+(mother|father|sister|brother|"
                r"wife|husband|friend|colleague|boss|partner|child|son|daughter)\b",
                re.IGNORECASE
            ),
            handler=self._handle_person_relation,
            confidence=0.85,
            priority=5
        ))

        # Pattern: "My <relation> is <Person>"
        self.patterns.append(ExtractionPattern(
            pattern_id="relation_person",
            regex=re.compile(
                r"\bMy\s+(mother|father|sister|brother|wife|husband|friend|"
                r"colleague|boss|partner|child|son|daughter)\s+is\s+([A-Z][A-Za-z]+)\b",
                re.IGNORECASE
            ),
            handler=self._handle_relation_person,
            confidence=0.85,
            priority=5
        ))

        # Sort by priority
        self.patterns.sort(key=lambda p: p.priority)

    def _handle_self_attribute(self, match: re.Match, context: MessageContext
                               ) -> List[AssertionCandidate]:
        """Handle 'I am <attribute>' pattern."""
        attribute = match.group(1).strip()
        if len(attribute) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="is_a",
            object_literal_type="string",
            object_literal_value=attribute,
            modality="state",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.75,
            extraction_method="rule_based"
        )]

    def _handle_self_has(self, match: re.Match, context: MessageContext
                         ) -> List[AssertionCandidate]:
        """Handle 'I have <noun>' pattern."""
        noun = match.group(1).strip()
        if len(noun) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="has",
            object_literal_type="string",
            object_literal_value=noun,
            modality="state",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.7,
            extraction_method="rule_based"
        )]

    def _handle_self_works_at(self, match: re.Match, context: MessageContext
                              ) -> List[AssertionCandidate]:
        """Handle 'I work at/for <org>' pattern."""
        org = match.group(1).strip()
        if len(org) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="works_at",
            object_entity_ref=org,  # Will be resolved to entity
            modality="state",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.85,
            extraction_method="rule_based"
        )]

    def _handle_self_lives_in(self, match: re.Match, context: MessageContext
                              ) -> List[AssertionCandidate]:
        """Handle 'I live in <location>' pattern."""
        location = match.group(1).strip()
        if len(location) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="lives_in",
            object_entity_ref=location,  # Will be resolved to entity
            modality="state",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.85,
            extraction_method="rule_based"
        )]

    def _handle_self_name(self, match: re.Match, context: MessageContext
                          ) -> List[AssertionCandidate]:
        """Handle 'My name is <name>' pattern."""
        name = match.group(1).strip()
        if len(name) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="has_name",
            object_literal_type="string",
            object_literal_value=name,
            modality="fact",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.9,
            extraction_method="rule_based"
        )]

    def _handle_self_preference(self, match: re.Match, context: MessageContext
                                ) -> List[AssertionCandidate]:
        """Handle 'I like/love/prefer <thing>' pattern."""
        thing = match.group(1).strip()
        if len(thing) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="likes",
            object_literal_type="string",
            object_literal_value=thing,
            modality="preference",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.7,
            extraction_method="rule_based"
        )]

    def _handle_self_dispreference(self, match: re.Match, context: MessageContext
                                   ) -> List[AssertionCandidate]:
        """Handle 'I don't like/hate <thing>' pattern."""
        thing = match.group(1).strip()
        if len(thing) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label="dislikes",
            object_literal_type="string",
            object_literal_value=thing,
            modality="preference",
            polarity="positive",  # "dislikes" is positive polarity for negative sentiment
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.7,
            extraction_method="rule_based"
        )]

    def _handle_person_relation(self, match: re.Match, context: MessageContext
                                ) -> List[AssertionCandidate]:
        """Handle '<Person> is my <relation>' pattern."""
        person = match.group(1).strip()
        relation = match.group(2).strip().lower()
        if len(person) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label=f"has_{relation}",
            object_entity_ref=person,
            modality="fact",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.85,
            extraction_method="rule_based"
        )]

    def _handle_relation_person(self, match: re.Match, context: MessageContext
                                ) -> List[AssertionCandidate]:
        """Handle 'My <relation> is <Person>' pattern."""
        relation = match.group(1).strip().lower()
        person = match.group(2).strip()
        if len(person) < 2:
            return []

        return [AssertionCandidate(
            subject="SELF",
            predicate_label=f"has_{relation}",
            object_entity_ref=person,
            modality="fact",
            polarity="positive",
            char_start=match.start(),
            char_end=match.end(),
            confidence=0.85,
            extraction_method="rule_based"
        )]

    def extract(self, context: MessageContext) -> List[AssertionCandidate]:
        """Extract candidates from message using all patterns."""
        candidates = []
        text = context.text_raw

        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                # Check if match overlaps with excluded regions
                if self._is_excluded(match.start(), match.end(), context.exclusion_ranges):
                    continue

                try:
                    extracted = pattern.handler(match, context)
                    for candidate in extracted:
                        candidate.confidence = pattern.confidence
                        candidates.append(candidate)
                except Exception as e:
                    logger.warning(
                        f"Pattern handler failed for {pattern.pattern_id}: {e}"
                    )

        return candidates

    def _is_excluded(self, start: int, end: int,
                     exclusions: List[ExclusionRange]) -> bool:
        """Check if span overlaps with any exclusion range."""
        for ex in exclusions:
            if start < ex.char_end and end > ex.char_start:
                return True
        return False


class RetractionPatternRegistry:
    """Registry of retraction patterns."""

    def __init__(self):
        self.patterns: List[Tuple[re.Pattern, str, float]] = []
        self._register_default_patterns()

    def _register_default_patterns(self):
        """Register default retraction patterns."""

        # Full retractions
        self.patterns.append((
            re.compile(
                r"\b(?:Actually|Wait|No),?\s+(?:that's\s+)?(?:not\s+true|wrong|incorrect)\b",
                re.IGNORECASE
            ),
            "full",
            0.8
        ))

        self.patterns.append((
            re.compile(
                r"\bI\s+was\s+wrong\s+(?:about\s+)?(.+?)(?:\.|$)",
                re.IGNORECASE
            ),
            "full",
            0.85
        ))

        # Corrections
        self.patterns.append((
            re.compile(
                r"\bActually,?\s+(?:it's|I\s+meant?)\s+(.+?)\s+not\s+(.+?)(?:\.|$)",
                re.IGNORECASE
            ),
            "correction",
            0.85
        ))

        self.patterns.append((
            re.compile(
                r"\bSorry,?\s+I\s+meant\s+(.+?)(?:\.|$)",
                re.IGNORECASE
            ),
            "correction",
            0.8
        ))

        # Temporal bounds
        self.patterns.append((
            re.compile(
                r"\b(?:That's\s+)?no\s+longer\s+(?:true|the\s+case)\b",
                re.IGNORECASE
            ),
            "temporal_bound",
            0.75
        ))

        self.patterns.append((
            re.compile(
                r"\bI\s+(?:stopped|quit|no\s+longer)\s+(.+?)(?:\.|$)",
                re.IGNORECASE
            ),
            "temporal_bound",
            0.8
        ))

    def extract(self, text: str, exclusions: List[ExclusionRange]
                ) -> List[RetractionCandidate]:
        """Extract retraction candidates from text."""
        candidates = []

        for pattern, rtype, confidence in self.patterns:
            for match in pattern.finditer(text):
                # Check exclusions
                if self._is_excluded(match.start(), match.end(), exclusions):
                    continue

                candidate = RetractionCandidate(
                    retraction_type=rtype,
                    target_clause=match.group(0),
                    replacement_clause=None,
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=match.group(0),
                    confidence=confidence
                )

                # Extract target/replacement for corrections
                if rtype == "correction" and len(match.groups()) >= 2:
                    candidate.replacement_clause = match.group(1)
                    candidate.target_clause = match.group(2) if len(match.groups()) > 1 else match.group(0)

                candidates.append(candidate)

        return candidates

    def _is_excluded(self, start: int, end: int,
                     exclusions: List[ExclusionRange]) -> bool:
        """Check if span overlaps with any exclusion range."""
        for ex in exclusions:
            if start < ex.char_end and end > ex.char_start:
                return True
        return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AssertionExtractionAndGroundingPipeline:
    """
    Stage 4: Assertion Extraction & Grounding Layer

    Processes messages from prior stages to extract semantic assertions
    with entity/predicate grounding and role-aware trust scoring.
    """

    def __init__(self, config: Stage4Config):
        self.config = config
        self.db = Stage4Database(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.stage_started_at_utc = TimestampUtils.now_utc()

        # Pattern registries
        self.pattern_registry = PatternRegistry(self.id_generator)
        self.retraction_registry = RetractionPatternRegistry()

        # LLM run tracking
        self.llm_run_id: Optional[str] = None
        self.llm_call_count = 0

        # Caches
        self._entity_cache: Dict[str, sqlite3.Row] = {}
        self._predicate_cache: Dict[str, str] = {}  # label_norm -> predicate_id
        self._self_entity_id: Optional[str] = None

        # Stats
        self.stats = {
            "messages_processed": 0,
            "assertions_inserted": 0,
            "assertions_updated": 0,
            "predicates_created": 0,
            "entities_created": 0,
            "retractions_detected": 0,
            "retractions_linked": 0,
            "candidates_extracted": 0,
            "candidates_validated": 0,
            "candidates_invalid": 0
        }

    def run(self) -> Dict[str, int]:
        """Execute Stage 4 pipeline. Returns statistics."""
        logger.info("Starting Stage 4: Assertion Extraction & Grounding Layer")

        # Check prerequisites
        self.db.check_required_tables()

        # Initialize schema
        self.db.initialize_stage4_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Phase 1: Initialize + prepare
            self._phase1_initialize()

            # Phase 2-5: Process each message
            eligible_messages = self.db.get_eligible_messages(self.config.min_extractable_chars)
            logger.info(f"Processing {len(eligible_messages)} eligible messages")

            for msg_row in eligible_messages:
                self._process_message(msg_row)
                self.stats["messages_processed"] += 1

                if self.stats["messages_processed"] % 100 == 0:
                    logger.info(f"Processed {self.stats['messages_processed']} messages")

            # Phase 5b: Refresh stats
            self.db.refresh_predicate_counts()

            # Phase 6: Finalize LLM run if applicable
            if self.llm_run_id:
                self._finalize_llm_run()

            # Commit transaction
            self.db.commit()
            logger.info("Stage 4 completed successfully")

        except Exception as e:
            logger.error(f"Stage 4 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return self.stats

    # =========================================================================
    # Phase 1: Initialize + Prepare
    # =========================================================================

    def _phase1_initialize(self):
        """Phase 1: Initialize and prepare for extraction."""
        logger.info("Phase 1: Initializing extraction")

        # Get SELF entity
        self._self_entity_id = self.db.get_self_entity_id()
        if not self._self_entity_id:
            logger.warning("SELF entity not found - will create during grounding if needed")

        # Initialize LLM run if enabled
        if self.config.enable_llm_assertion_extraction:
            self.llm_run_id = self.id_generator.generate([
                "llm_run",
                self.stage_started_at_utc,
                self.config.llm_model_name
            ])
            config_json = JCS.canonicalize({
                "model_name": self.config.llm_model_name,
                "model_version": self.config.llm_model_version,
                "temperature": self.config.llm_temperature,
                "top_p": self.config.llm_top_p,
                "seed": self.config.llm_seed,
                "k_context": self.config.k_context,
                "multi_run_count": self.config.llm_multi_run_count
            })
            self.db.insert_llm_run(
                self.llm_run_id,
                self.config.llm_model_name,
                self.config.llm_model_version,
                config_json,
                self.stage_started_at_utc
            )
            logger.info(f"Created LLM extraction run: {self.llm_run_id}")

    # =========================================================================
    # Phase 2-5: Per-Message Processing
    # =========================================================================

    def _process_message(self, msg_row: sqlite3.Row):
        """Process a single message through phases 2-5."""
        message_id = msg_row['message_id']
        conversation_id = msg_row['conversation_id']
        role = msg_row['role']
        text_raw = msg_row['text_raw']
        created_at_utc = msg_row['created_at_utc']
        order_index = msg_row['order_index']

        # Build exclusion ranges
        exclusions = self._build_exclusion_ranges(msg_row)

        # Check non-excluded length
        total_excluded = sum(e.char_end - e.char_start for e in exclusions)
        non_excluded_len = len(text_raw) - total_excluded
        if non_excluded_len < self.config.min_extractable_chars:
            return

        # Build context
        context = self._build_message_context(
            message_id, conversation_id, role, text_raw,
            created_at_utc, order_index, exclusions
        )

        # Phase 2: Extract candidates
        candidates = self._phase2_extract_candidates(context)
        self.stats["candidates_extracted"] += len(candidates)

        if not candidates:
            # Check for retractions in user messages
            if role == "user":
                self._extract_retractions(context)
            return

        # Phase 3: Validate candidates
        valid_candidates = self._phase3_validate_candidates(candidates, context)
        self.stats["candidates_validated"] += len(valid_candidates)
        self.stats["candidates_invalid"] += len(candidates) - len(valid_candidates)

        if not valid_candidates:
            return

        # Phase 4: Ground candidates
        grounded = self._phase4_ground_candidates(valid_candidates, context)

        # Phase 5: Persist assertions
        for assertion in grounded:
            self._phase5_persist_assertion(assertion)

        # Check for retractions in user messages
        if role == "user":
            self._extract_retractions(context)

    def _build_exclusion_ranges(self, msg_row: sqlite3.Row) -> List[ExclusionRange]:
        """Build exclusion ranges from code fences and optionally blockquotes."""
        exclusions = []

        # Code fences
        if msg_row['code_fence_ranges_json']:
            try:
                ranges = json.loads(msg_row['code_fence_ranges_json'])
                for r in ranges:
                    exclusions.append(ExclusionRange(
                        char_start=r['char_start'],
                        char_end=r['char_end'],
                        reason="code_fence"
                    ))
            except json.JSONDecodeError:
                logger.warning(f"Invalid code_fence_ranges_json for message {msg_row['message_id']}")

        # Blockquotes (optional)
        if self.config.ignore_markdown_blockquotes and msg_row['blockquote_ranges_json']:
            try:
                ranges = json.loads(msg_row['blockquote_ranges_json'])
                for r in ranges:
                    exclusions.append(ExclusionRange(
                        char_start=r['char_start'],
                        char_end=r['char_end'],
                        reason="blockquote"
                    ))
            except json.JSONDecodeError:
                logger.warning(f"Invalid blockquote_ranges_json for message {msg_row['message_id']}")

        # Sort and merge overlapping ranges
        return self._merge_exclusion_ranges(exclusions)

    def _merge_exclusion_ranges(self, ranges: List[ExclusionRange]) -> List[ExclusionRange]:
        """Merge overlapping exclusion ranges."""
        if not ranges:
            return []

        sorted_ranges = sorted(ranges, key=lambda r: (r.char_start, r.char_end))
        merged = [sorted_ranges[0]]

        for current in sorted_ranges[1:]:
            last = merged[-1]
            if current.char_start <= last.char_end:
                # Overlapping - merge
                merged[-1] = ExclusionRange(
                    char_start=last.char_start,
                    char_end=max(last.char_end, current.char_end),
                    reason=f"{last.reason}+{current.reason}"
                )
            else:
                merged.append(current)

        return merged

    def _build_message_context(
        self, message_id: str, conversation_id: str, role: str,
        text_raw: str, created_at_utc: Optional[str], order_index: int,
        exclusions: List[ExclusionRange]
    ) -> MessageContext:
        """Build context for message processing."""

        # Get prior messages for context
        prior_rows = self.db.get_context_messages(
            conversation_id, order_index, self.config.k_context
        )
        context_messages = [
            {
                "message_id": r['message_id'],
                "role": r['role'],
                "text": r['text_raw'],
                "created_at": r['created_at_utc'],
                "order_index": r['order_index']
            }
            for r in reversed(prior_rows)  # Chronological order
        ]

        # Collect message IDs for entity/time lookup
        all_message_ids = [message_id] + [m['message_id'] for m in context_messages]

        # Get context entities
        entity_mentions = self.db.get_entity_mentions_for_context(all_message_ids)
        context_entities = {}
        for em in entity_mentions:
            eid = em['entity_id']
            if eid not in context_entities:
                context_entities[eid] = {
                    "entity_id": eid,
                    "canonical_name": em['canonical_name'],
                    "entity_type": em['entity_type'],
                    "mentions": []
                }
            context_entities[eid]["mentions"].append({
                "message_id": em['message_id'],
                "surface_text": em['surface_text']
            })

        # Get context time mentions
        time_rows = self.db.get_time_mentions_for_context(all_message_ids)
        context_times = [
            {
                "time_mention_id": t['time_mention_id'],
                "message_id": t['message_id'],
                "surface_text": t['surface_text'],
                "valid_from_utc": t['valid_from_utc'],
                "valid_to_utc": t['valid_to_utc'],
                "resolution_granularity": t['resolution_granularity']
            }
            for t in time_rows
        ]

        return MessageContext(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            text_raw=text_raw,
            created_at_utc=created_at_utc,
            order_index=order_index,
            exclusion_ranges=exclusions,
            context_messages=context_messages,
            context_entities=context_entities,
            context_times=context_times
        )

    # =========================================================================
    # Phase 2: Candidate Extraction
    # =========================================================================

    def _phase2_extract_candidates(self, context: MessageContext
                                   ) -> List[AssertionCandidate]:
        """Phase 2: Extract assertion candidates from message."""

        # Rule-based extraction
        candidates = self.pattern_registry.extract(context)

        # LLM extraction (if enabled)
        if self.config.enable_llm_assertion_extraction:
            llm_candidates = self._extract_with_llm(context)
            candidates = self._merge_candidates(candidates, llm_candidates)

        return candidates

    def _extract_with_llm(self, context: MessageContext) -> List[AssertionCandidate]:
        """Extract candidates using LLM."""
        # Placeholder for LLM extraction
        # In a real implementation, this would:
        # 1. Build prompt with context
        # 2. Call LLM API with determinism settings
        # 3. Parse response into candidates
        # 4. Log call in llm_extraction_calls
        logger.debug(f"LLM extraction for message {context.message_id} (placeholder)")
        return []

    def _merge_candidates(self, rule_candidates: List[AssertionCandidate],
                          llm_candidates: List[AssertionCandidate]
                          ) -> List[AssertionCandidate]:
        """Merge rule-based and LLM candidates, preferring rule-based."""
        if not llm_candidates:
            return rule_candidates

        merged = list(rule_candidates)
        rule_spans = set()

        for c in rule_candidates:
            if c.char_start is not None and c.char_end is not None:
                rule_spans.add((c.char_start, c.char_end))

        for llm_c in llm_candidates:
            # Check for overlap with rule-based
            is_duplicate = False
            if llm_c.char_start is not None and llm_c.char_end is not None:
                for start, end in rule_spans:
                    if llm_c.char_start < end and llm_c.char_end > start:
                        is_duplicate = True
                        break

            if not is_duplicate:
                llm_c.extraction_method = "llm"
                merged.append(llm_c)

        return merged

    # =========================================================================
    # Phase 3: Candidate Validation
    # =========================================================================

    def _phase3_validate_candidates(self, candidates: List[AssertionCandidate],
                                    context: MessageContext
                                    ) -> List[AssertionCandidate]:
        """Phase 3: Validate assertion candidates."""
        valid = []

        for candidate in candidates:
            try:
                if self._validate_candidate(candidate, context):
                    valid.append(candidate)
            except Exception as e:
                logger.warning(f"Candidate validation error: {e}")
                self.stats["candidates_invalid"] += 1

        return valid

    def _validate_candidate(self, candidate: AssertionCandidate,
                            context: MessageContext) -> bool:
        """Validate a single candidate."""

        # Required field validation
        if not candidate.subject or not candidate.subject.strip():
            logger.debug("Candidate rejected: empty subject")
            return False

        if not candidate.predicate_label or not candidate.predicate_label.strip():
            logger.debug("Candidate rejected: empty predicate_label")
            return False

        if candidate.modality not in [m.value for m in Modality]:
            logger.debug(f"Candidate rejected: invalid modality {candidate.modality}")
            return False

        if candidate.polarity not in [p.value for p in Polarity]:
            logger.debug(f"Candidate rejected: invalid polarity {candidate.polarity}")
            return False

        # Object validation: exactly one must be true
        has_entity_obj = candidate.object_entity_ref is not None
        has_literal_obj = (candidate.object_literal_type is not None and
                          candidate.object_literal_value is not None)
        is_unary = not has_entity_obj and not has_literal_obj

        if has_entity_obj and has_literal_obj:
            logger.debug("Candidate rejected: both entity and literal object")
            return False

        # Span verification
        if candidate.char_start is not None and candidate.char_end is not None:
            text_len = len(context.text_raw)
            if not (0 <= candidate.char_start < candidate.char_end <= text_len):
                logger.debug(f"Candidate rejected: invalid span [{candidate.char_start}, {candidate.char_end})")
                candidate.char_start = None
                candidate.char_end = None

        # Quote-to-offset resolution
        if candidate.quote and candidate.char_start is None:
            self._resolve_quote_to_offset(candidate, context)

        return True

    def _resolve_quote_to_offset(self, candidate: AssertionCandidate,
                                 context: MessageContext):
        """Attempt to find quote in text and set offsets."""
        quote = candidate.quote
        text = context.text_raw

        # Find all occurrences
        start = 0
        matches = []
        while True:
            idx = text.find(quote, start)
            if idx == -1:
                break
            matches.append(idx)
            start = idx + 1

        if len(matches) == 1:
            candidate.char_start = matches[0]
            candidate.char_end = matches[0] + len(quote)
            logger.debug(f"Resolved quote to offset [{candidate.char_start}, {candidate.char_end})")
        elif len(matches) == 0:
            logger.debug(f"Quote not found in text: {quote[:50]}...")
        else:
            logger.debug(f"Quote is ambiguous ({len(matches)} matches): {quote[:50]}...")

    # =========================================================================
    # Phase 4: Grounding
    # =========================================================================

    def _phase4_ground_candidates(self, candidates: List[AssertionCandidate],
                                  context: MessageContext
                                  ) -> List[GroundedAssertion]:
        """Phase 4: Ground candidates to entities and predicates."""
        grounded = []

        for candidate in candidates:
            try:
                assertion = self._ground_candidate(candidate, context)
                if assertion:
                    grounded.append(assertion)
            except Exception as e:
                logger.warning(f"Grounding error for candidate: {e}")

        return grounded

    def _ground_candidate(self, candidate: AssertionCandidate,
                          context: MessageContext) -> Optional[GroundedAssertion]:
        """Ground a single candidate."""

        # Resolve subject
        subject_entity_id = self._resolve_entity(candidate.subject, context)
        if not subject_entity_id:
            logger.warning(f"Could not resolve subject: {candidate.subject}")
            return None

        # Resolve predicate
        predicate_id = self._resolve_predicate(candidate.predicate_label, context)

        # Resolve object
        object_entity_id = None
        object_value_type = None
        object_value = None

        if candidate.object_entity_ref:
            object_entity_id = self._resolve_entity(candidate.object_entity_ref, context)
            if not object_entity_id:
                logger.warning(f"Could not resolve object entity: {candidate.object_entity_ref}")
                # Fall back to literal
                object_value_type = "string"
                object_value = JCS.canonicalize(candidate.object_entity_ref)

        elif candidate.object_literal_type and candidate.object_literal_value is not None:
            object_value_type = candidate.object_literal_type
            object_value = self._normalize_literal_value(
                candidate.object_literal_type,
                candidate.object_literal_value
            )

        # Compute object signature
        object_signature = self._compute_object_signature(
            object_entity_id, object_value_type, object_value
        )

        # Determine arity and update predicate if needed
        arity = 1 if object_signature == "N:__NONE__" else 2

        # Resolve temporal qualifier
        temporal_qualifier_type = None
        temporal_qualifier_id = None
        if candidate.temporal_qualifier_surface:
            tm = self.db.get_time_mention_by_surface(
                context.message_id,
                candidate.temporal_qualifier_surface
            )
            if tm:
                temporal_qualifier_id = tm['time_mention_id']
                temporal_qualifier_type = candidate.temporal_qualifier_type or "at"

        # Compute keys
        assertion_key = self._compute_assertion_key(
            context.message_id, subject_entity_id, predicate_id,
            object_signature, candidate.char_start,
            candidate.modality, candidate.polarity
        )

        fact_key = self._compute_fact_key(
            subject_entity_id, predicate_id, object_signature
        )

        # Check corroboration (for assistant assertions)
        has_user_corroboration = 0
        if context.role == "assistant":
            has_user_corroboration = self._check_corroboration(
                context, subject_entity_id, predicate_id, object_signature
            )

        # Compute confidence
        trust_weight = self._get_trust_weight(context.role, has_user_corroboration)
        confidence_final = min(1.0, max(0.0, candidate.confidence * trust_weight))

        # Build surface text
        surface_text = None
        if candidate.char_start is not None and candidate.char_end is not None:
            surface_text = context.text_raw[candidate.char_start:candidate.char_end]

        # Build raw JSON
        raw_data = {
            "subject_raw": candidate.subject,
            "predicate_raw": candidate.predicate_label,
            "object_entity_ref": candidate.object_entity_ref,
            "object_literal_type": candidate.object_literal_type,
            "object_literal_value": candidate.object_literal_value,
            "extraction_method": candidate.extraction_method,
            "pattern_data": candidate.raw_data
        }
        raw_assertion_json = JCS.canonicalize(raw_data)

        # Generate assertion ID
        assertion_id = self.id_generator.generate([
            "assertion",
            assertion_key,
            HashUtils.sha256_string(raw_assertion_json)
        ])

        return GroundedAssertion(
            assertion_id=assertion_id,
            message_id=context.message_id,
            assertion_key=assertion_key,
            fact_key=fact_key,
            subject_entity_id=subject_entity_id,
            predicate_id=predicate_id,
            object_entity_id=object_entity_id,
            object_value_type=object_value_type,
            object_value=object_value,
            object_signature=object_signature,
            temporal_qualifier_type=temporal_qualifier_type,
            temporal_qualifier_id=temporal_qualifier_id,
            modality=candidate.modality,
            polarity=candidate.polarity,
            asserted_role=context.role,
            asserted_at_utc=context.created_at_utc,
            confidence_extraction=candidate.confidence,
            confidence_final=confidence_final,
            has_user_corroboration=has_user_corroboration,
            superseded_by_assertion_id=None,
            supersession_type=None,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            surface_text=surface_text,
            extraction_method=candidate.extraction_method,
            extraction_model=candidate.extraction_model,
            raw_assertion_json=raw_assertion_json
        )

    def _resolve_entity(self, reference: str, context: MessageContext) -> Optional[str]:
        """Resolve entity reference to entity_id."""
        reference = reference.strip()

        # 1. Reserved SELF reference
        self_refs = {"SELF", "self", "I", "i", "me", "my", "Me", "My"}
        if reference in self_refs:
            if self._self_entity_id:
                return self._self_entity_id
            else:
                # Create SELF entity
                return self._create_self_entity()

        # 2. Check cache
        cache_key = reference.lower()
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]['entity_id']

        # 3. Direct UUID check
        try:
            uuid.UUID(reference)
            entity = self.db.get_entity_by_id(reference)
            if entity:
                self._entity_cache[cache_key] = entity
                return entity['entity_id']
        except ValueError:
            pass

        # 4. Canonical name match (case-insensitive)
        entity = self.db.get_entity_by_canonical_name(reference)
        if entity:
            self._entity_cache[cache_key] = entity
            return entity['entity_id']

        # 5. Alias match
        entity = self.db.search_entity_by_alias(reference)
        if entity:
            self._entity_cache[cache_key] = entity
            return entity['entity_id']

        # 6. Context entities check
        for eid, edata in context.context_entities.items():
            if edata['canonical_name'].lower() == reference.lower():
                return eid
            # Check mention surfaces
            for m in edata.get('mentions', []):
                if m.get('surface_text', '').lower() == reference.lower():
                    return eid

        # 7. Fuzzy match (if enabled)
        if self.config.enable_fuzzy_entity_linking:
            best_match = None
            best_score = 0.0

            all_entities = self.db.get_all_active_entities()
            for entity in all_entities:
                # Check canonical name
                score = TextUtils.jaro_winkler_similarity(
                    reference.lower(),
                    entity['canonical_name'].lower()
                )
                if score > best_score:
                    best_score = score
                    best_match = entity

                # Check aliases
                if entity['aliases_json']:
                    try:
                        aliases = json.loads(entity['aliases_json'])
                        for alias in aliases:
                            score = TextUtils.jaro_winkler_similarity(
                                reference.lower(),
                                alias.lower()
                            )
                            if score > best_score:
                                best_score = score
                                best_match = entity
                    except json.JSONDecodeError:
                        pass

            if best_match and best_score >= self.config.threshold_link_string_sim:
                self._entity_cache[cache_key] = best_match
                return best_match['entity_id']

        # 8. Create new entity (fallback)
        return self._create_entity_fallback(reference)

    def _create_self_entity(self) -> str:
        """Create the SELF entity."""
        entity_id = self.id_generator.generate(["entity", "PERSON", "__SELF__"])
        self.db.create_entity(entity_id, "PERSON", "__SELF__", "SELF")
        self._self_entity_id = entity_id
        self.stats["entities_created"] += 1
        logger.info(f"Created SELF entity: {entity_id}")
        return entity_id

    def _create_entity_fallback(self, reference: str) -> str:
        """Create a new entity as fallback."""
        # Normalize key
        entity_key = TextUtils.nfkc_normalize(reference.strip().lower())
        entity_key = TextUtils.normalize_whitespace(entity_key)

        entity_id = self.id_generator.generate(["entity", "OTHER", entity_key])

        # Check if already exists with this key
        existing = self.db.get_entity_by_key("OTHER", entity_key)
        if existing:
            return existing['entity_id']

        self.db.create_entity(entity_id, "OTHER", entity_key, reference)
        self.stats["entities_created"] += 1
        logger.warning(f"Created entity during grounding: {reference} -> {entity_id}")

        return entity_id

    def _resolve_predicate(self, label: str, context: MessageContext) -> str:
        """Resolve or create predicate."""
        # Normalize label
        canonical_label = TextUtils.nfkc_normalize(label.strip())
        canonical_label = TextUtils.normalize_whitespace(canonical_label)
        canonical_label_norm = canonical_label.lower()

        # Check cache
        if canonical_label_norm in self._predicate_cache:
            return self._predicate_cache[canonical_label_norm]

        # Check database
        existing = self.db.get_predicate_by_label_norm(canonical_label_norm)
        if existing:
            self._predicate_cache[canonical_label_norm] = existing['predicate_id']
            return existing['predicate_id']

        # Create new predicate
        predicate_id = self.id_generator.generate(["pred", canonical_label_norm])
        self.db.upsert_predicate(
            predicate_id,
            canonical_label,
            canonical_label_norm,
            arity=2,  # Default to binary
            first_seen_at_utc=context.created_at_utc or TimestampUtils.now_utc()
        )

        self._predicate_cache[canonical_label_norm] = predicate_id
        self.stats["predicates_created"] += 1

        return predicate_id

    def _normalize_literal_value(self, value_type: str, value: Any) -> str:
        """Normalize and canonicalize literal value."""
        if value_type == "string":
            return JCS.canonicalize(str(value))
        elif value_type == "number":
            if isinstance(value, (int, float)):
                return JCS.canonicalize(value)
            try:
                return JCS.canonicalize(float(value))
            except (ValueError, TypeError):
                return JCS.canonicalize(str(value))
        elif value_type == "boolean":
            if isinstance(value, bool):
                return JCS.canonicalize(value)
            return JCS.canonicalize(str(value).lower() in ('true', '1', 'yes'))
        elif value_type == "date":
            normalized = TimestampUtils.normalize_to_utc(value)
            return JCS.canonicalize(normalized or str(value))
        elif value_type == "json":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            return JCS.canonicalize(value)
        else:
            return JCS.canonicalize(str(value))

    def _compute_object_signature(self, object_entity_id: Optional[str],
                                  object_value_type: Optional[str],
                                  object_value: Optional[str]) -> str:
        """Compute object signature for deduplication."""
        if object_entity_id:
            return f"E:{object_entity_id}"
        elif object_value is not None:
            hash_input = JCS.canonicalize([object_value_type, object_value])
            return f"V:{HashUtils.sha256_string(hash_input)}"
        else:
            return "N:__NONE__"

    def _compute_assertion_key(self, message_id: str, subject_entity_id: str,
                               predicate_id: str, object_signature: str,
                               char_start: Optional[int],
                               modality: str, polarity: str) -> str:
        """Compute assertion key for exact deduplication."""
        return JCS.canonicalize([
            message_id,
            subject_entity_id,
            predicate_id,
            object_signature,
            char_start if char_start is not None else "__NULL__",
            modality,
            polarity
        ])

    def _compute_fact_key(self, subject_entity_id: str, predicate_id: str,
                          object_signature: str) -> str:
        """Compute fact key for semantic deduplication."""
        return JCS.canonicalize([
            subject_entity_id,
            predicate_id,
            object_signature
        ])

    def _check_corroboration(self, context: MessageContext,
                             subject_entity_id: str, predicate_id: str,
                             object_signature: str) -> int:
        """Check if assistant assertion is corroborated by prior user assertion."""
        prior_assertions = self.db.get_prior_user_assertions(
            context.conversation_id,
            context.order_index,
            self.config.coref_window_size
        )

        for prior in prior_assertions:
            # Same subject
            if prior['subject_entity_id'] != subject_entity_id:
                continue

            # Same or similar predicate
            predicate_match = prior['predicate_id'] == predicate_id
            if not predicate_match and self.config.enable_fuzzy_entity_linking:
                # Fuzzy predicate matching would go here
                pass

            if not predicate_match:
                continue

            # Same object
            if prior['object_signature'] == object_signature:
                return 1

        return 0

    def _get_trust_weight(self, role: str, has_corroboration: int) -> float:
        """Get trust weight based on role and corroboration."""
        if role == "user":
            return self.config.trust_weight_user
        elif has_corroboration:
            return self.config.trust_weight_assistant_corroborated
        else:
            return self.config.trust_weight_assistant_uncorroborated

    # =========================================================================
    # Phase 5: Persistence
    # =========================================================================

    def _phase5_persist_assertion(self, assertion: GroundedAssertion):
        """Phase 5: Persist a grounded assertion."""
        policy = self.config.assertion_upsert_policy

        existing = self.db.get_assertion_by_key(assertion.assertion_key)

        if existing:
            if policy == UpsertPolicy.KEEP_HIGHEST_CONFIDENCE.value:
                if assertion.confidence_final > existing['confidence_final'] + 0.001:
                    # Update existing
                    self.db.update_assertion_confidence(
                        existing['assertion_id'],
                        assertion.confidence_final
                    )
                    self.stats["assertions_updated"] += 1
                # else: keep existing, do nothing
            elif policy == UpsertPolicy.KEEP_FIRST.value:
                # Keep existing, do nothing
                pass
            elif policy == UpsertPolicy.KEEP_ALL.value:
                # Insert anyway (requires dropping unique constraint)
                self.db.insert_assertion(assertion)
                self.stats["assertions_inserted"] += 1
        else:
            self.db.insert_assertion(assertion)
            self.stats["assertions_inserted"] += 1

    def _extract_retractions(self, context: MessageContext):
        """Extract and persist retractions from user message."""
        if context.role != "user":
            return

        candidates = self.retraction_registry.extract(
            context.text_raw,
            context.exclusion_ranges
        )

        for candidate in candidates:
            self._persist_retraction(candidate, context)

    def _persist_retraction(self, candidate: RetractionCandidate,
                            context: MessageContext):
        """Persist a retraction."""
        self.stats["retractions_detected"] += 1

        # Attempt to link to prior assertion
        target_assertion_id = None
        target_fact_key = None

        # Try to parse target clause and find matching assertion
        # This is a simplified implementation
        prior_assertions = self.db.get_prior_user_assertions(
            context.conversation_id,
            context.order_index,
            self.config.coref_window_size * 2  # Wider window for retractions
        )

        if prior_assertions:
            # Use most recent as target (simplified)
            target_assertion_id = prior_assertions[-1]['assertion_id']
            target_fact_key = prior_assertions[-1]['fact_key']
            self.stats["retractions_linked"] += 1

            # Mark assertion as superseded
            if candidate.retraction_type == "full":
                self.db.update_assertion_supersession(
                    target_assertion_id,
                    None,  # No replacement
                    SupersessionType.RETRACTION.value
                )

        # Generate retraction ID
        retraction_id = self.id_generator.generate([
            "retraction",
            context.message_id,
            candidate.char_start if candidate.char_start is not None else "__NULL__",
            candidate.char_end if candidate.char_end is not None else "__NULL__"
        ])

        # Build raw JSON
        raw_json = JCS.canonicalize({
            "retraction_type": candidate.retraction_type,
            "target_clause": candidate.target_clause,
            "replacement_clause": candidate.replacement_clause,
            "confidence": candidate.confidence,
            "raw_data": candidate.raw_data
        })

        self.db.insert_retraction(
            retraction_id,
            context.message_id,
            target_assertion_id,
            target_fact_key,
            candidate.retraction_type,
            None,  # replacement_assertion_id - would need full grounding
            candidate.confidence,
            candidate.char_start,
            candidate.char_end,
            candidate.surface_text,
            raw_json
        )

    def _finalize_llm_run(self):
        """Finalize LLM extraction run record."""
        if not self.llm_run_id:
            return

        completed_at = TimestampUtils.now_utc()
        assertions_extracted = self.stats.get("llm_assertions", 0)

        stats_json = JCS.canonicalize({
            "calls_total": self.llm_call_count,
            "messages_processed": self.stats["messages_processed"],
            "assertions_extracted": assertions_extracted
        })

        self.db.update_llm_run(
            self.llm_run_id,
            completed_at,
            self.stats["messages_processed"],
            assertions_extracted,
            stats_json
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def run_stage4(config: Stage4Config) -> Dict[str, int]:
    """Run Stage 4 pipeline on existing database."""
    pipeline = AssertionExtractionAndGroundingPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Stage 4: Assertion Extraction & Grounding Layer")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: kg.db)"
    )
    parser.add_argument(
        "--enable-llm",
        default=False,
        action="store_true",
        help="Enable LLM-based assertion extraction"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="LLM model name for extraction"
    )
    parser.add_argument(
        "--k-context",
        type=int,
        default=5,
        help="Number of prior messages for context"
    )
    parser.add_argument(
        "--trust-user",
        type=float,
        default=1.0,
        help="Trust weight for user assertions"
    )
    parser.add_argument(
        "--trust-assistant-corroborated",
        type=float,
        default=0.9,
        help="Trust weight for corroborated assistant assertions"
    )
    parser.add_argument(
        "--trust-assistant-uncorroborated",
        type=float,
        default=0.5,
        help="Trust weight for uncorroborated assistant assertions"
    )
    parser.add_argument(
        "--ignore-blockquotes",
        action="store_true",
        help="Exclude markdown blockquotes from extraction"
    )
    parser.add_argument(
        "--upsert-policy",
        type=str,
        choices=["keep_highest_confidence", "keep_first", "keep_all"],
        default="keep_highest_confidence",
        help="Assertion upsert policy"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.85,
        help="String similarity threshold for entity linking"
    )
    parser.add_argument(
        "--coref-window",
        type=int,
        default=10,
        help="Window size for corroboration detection"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    config = Stage4Config(
        output_file_path=args.db,
        enable_llm_assertion_extraction=args.enable_llm,
        llm_model_name=args.llm_model,
        k_context=args.k_context,
        trust_weight_user=args.trust_user,
        trust_weight_assistant_corroborated=args.trust_assistant_corroborated,
        trust_weight_assistant_uncorroborated=args.trust_assistant_uncorroborated,
        ignore_markdown_blockquotes=args.ignore_blockquotes,
        assertion_upsert_policy=args.upsert_policy,
        threshold_link_string_sim=args.fuzzy_threshold,
        coref_window_size=args.coref_window
    )

    try:
        stats = run_stage4(config)

        print("\n" + "=" * 60)
        print("Stage 4: Assertion Extraction & Grounding - Complete")
        print("=" * 60)
        print(f"Messages processed:     {stats['messages_processed']:>8}")
        print(f"Candidates extracted:   {stats['candidates_extracted']:>8}")
        print(f"Candidates validated:   {stats['candidates_validated']:>8}")
        print(f"Candidates invalid:     {stats['candidates_invalid']:>8}")
        print(f"Assertions inserted:    {stats['assertions_inserted']:>8}")
        print(f"Assertions updated:     {stats['assertions_updated']:>8}")
        print(f"Predicates created:     {stats['predicates_created']:>8}")
        print(f"Entities created:       {stats['entities_created']:>8}")
        print(f"Retractions detected:   {stats['retractions_detected']:>8}")
        print(f"Retractions linked:     {stats['retractions_linked']:>8}")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")
        raise