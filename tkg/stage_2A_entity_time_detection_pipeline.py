"""
Stage 2A: Entity & Time Detection Layer

Builds a lossless detection layer over Stage 1 text by producing:
1. Entity mention candidates with offset-correct spans
2. Time mentions with conservative resolution anchored to message time
3. Deterministic overlap resolution to emit winning mentions
"""
import hashlib
import json
import logging
import re
import sqlite3
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Iterator, List, Optional, Dict, Tuple, Set, Callable
import pendulum


# ============================================================================
# LOGGING
# ============================================================================

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


# ============================================================================
# UTILITIES
# ============================================================================

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

    @staticmethod
    def compute_surface_hash(surface_text: Optional[str]) -> str:
        """Compute hash for surface text, handling NULL case."""
        if surface_text is None:
            return HashUtils.sha256_string("__NO_SURFACE__")
        return HashUtils.sha256_string(surface_text)


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


# ============================================================================
# ENUMS
# ============================================================================

class DetectorType(StrEnum):
    """Entity detector types in execution order."""
    EMAIL = "EMAIL"
    URL = "URL"
    DOI = "DOI"
    UUID = "UUID"
    HASH_HEX = "HASH_HEX"
    IP_ADDRESS = "IP_ADDRESS"
    PHONE = "PHONE"
    FILEPATH = "FILEPATH"
    BARE_DOMAIN = "BARE_DOMAIN"
    ARXIV = "ARXIV"
    CVE = "CVE"
    ORCID = "ORCID"
    HANDLE = "HANDLE"
    HASHTAG = "HASHTAG"
    NER = "NER"
    LEXICON = "LEXICON"


class EntityTypeHint(StrEnum):
    """Entity type hints from detectors."""
    EMAIL = "EMAIL"
    URL = "URL"
    DOI = "DOI"
    UUID = "UUID"
    HASH_HEX = "HASH_HEX"
    IP_ADDRESS = "IP_ADDRESS"
    PHONE = "PHONE"
    FILEPATH = "FILEPATH"
    BARE_DOMAIN = "BARE_DOMAIN"
    ARXIV = "ARXIV"
    CVE = "CVE"
    ORCID = "ORCID"
    HANDLE = "HANDLE"
    HASHTAG = "HASHTAG"
    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    CUSTOM_TERM = "CUSTOM_TERM"
    OTHER = "OTHER"


class SuppressionReason(StrEnum):
    """Reasons for suppressing a candidate."""
    OVERLAP_HIGHER_SCORE = "OVERLAP_HIGHER_SCORE"
    INTERSECTS_CODE_FENCE = "INTERSECTS_CODE_FENCE"
    INTERSECTS_BLOCKQUOTE = "INTERSECTS_BLOCKQUOTE"
    NO_OFFSETS_UNRELIABLE = "NO_OFFSETS_UNRELIABLE"
    CODE_LIKE_TOKEN = "CODE_LIKE_TOKEN"
    DENYLIST = "DENYLIST"
    OFFSET_MISMATCH = "OFFSET_MISMATCH"


class TimeResolvedType(StrEnum):
    """Time resolution outcome types."""
    INSTANT = "instant"
    INTERVAL = "interval"
    UNRESOLVED = "unresolved"


class TimeGranularity(StrEnum):
    """Time resolution granularity."""
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Stage2AConfig:
    """Configuration for Stage 2A pipeline."""
    # Database path
    output_file_path: Path = field(default_factory=lambda: Path("kg.db"))

    # ID generation
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"

    # Time handling
    anchor_timezone: str = "UTC"

    # Exclusion options
    ignore_markdown_blockquotes: bool = False

    # Domain validation
    domain_tld_allowlist_enabled: bool = False
    domain_tld_allowlist: Set[str] = field(default_factory=lambda: {
        "com", "org", "net", "edu", "gov", "io", "co", "ai", "dev", "app",
        "uk", "de", "fr", "jp", "cn", "au", "ca", "nl", "ru", "br", "in"
    })

    # NER configuration
    enable_ner: bool = True
    ner_model_name: str = "en_core_web_sm"
    ner_max_chars: int = 10000
    ner_stride: int = 1000
    ner_label_allowlist: Set[str] = field(default_factory=lambda: {"PERSON", "ORG", "GPE", "LOC"})
    emit_spanless_ner: bool = False

    # URL handling
    url_sort_query_params: bool = False

    # Optional detectors
    enable_arxiv: bool = True
    enable_cve: bool = True
    enable_orcid: bool = True
    enable_handle: bool = False
    enable_hashtag: bool = False

    # Noise suppression
    code_like_denylist: Set[str] = field(default_factory=lambda: {
        "null", "true", "false", "undefined", "none", "nil",
        "int", "str", "bool", "float", "dict", "list", "tuple",
        "void", "const", "var", "let", "function", "class", "def",
        "return", "if", "else", "for", "while", "try", "except",
        "import", "from", "as", "with", "lambda", "yield", "async", "await"
    })

    # Detector versions (for auditability)
    detector_versions: Dict[str, str] = field(default_factory=lambda: {
        "EMAIL": "1.0.0",
        "URL": "1.0.0",
        "DOI": "1.0.0",
        "UUID": "1.0.0",
        "HASH_HEX": "1.0.0",
        "IP_ADDRESS": "1.0.0",
        "PHONE": "1.0.0",
        "FILEPATH": "1.0.0",
        "BARE_DOMAIN": "1.0.0",
        "ARXIV": "1.0.0",
        "CVE": "1.0.0",
        "ORCID": "1.0.0",
        "HANDLE": "1.0.0",
        "HASHTAG": "1.0.0",
    })


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TextSpan:
    """Represents a span of text with character offsets."""
    char_start: int
    char_end: int

    def overlaps(self, other: "TextSpan") -> bool:
        """Check if this span overlaps with another."""
        return self.char_start < other.char_end and other.char_start < self.char_end

    def contains(self, other: "TextSpan") -> bool:
        """Check if this span fully contains another."""
        return self.char_start <= other.char_start and self.char_end >= other.char_end

    def length(self) -> int:
        """Return the length of the span."""
        return self.char_end - self.char_start


@dataclass
class EntityCandidate:
    """A detected entity candidate."""
    detector: str
    detector_version: str
    entity_type_hint: str
    char_start: Optional[int]
    char_end: Optional[int]
    surface_text: Optional[str]
    confidence: float
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeCandidate:
    """A detected time expression candidate."""
    char_start: int
    char_end: int
    surface_text: str
    pattern_id: str
    pattern_precedence: int
    parsed_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageContext:
    """Context for processing a single message."""
    message_id: str
    conversation_id: str
    text_raw: str
    created_at_utc: Optional[str]
    timestamp_quality: Optional[str]
    role: str
    order_index: int
    code_fence_ranges: List[TextSpan]
    blockquote_ranges: List[TextSpan]
    excluded_ranges: List[TextSpan]


# ============================================================================
# DATABASE
# ============================================================================

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


class Stage2ADatabase(Database):
    """Stage 2A specific database operations."""

    STAGE2A_SCHEMA = """
    -- Entity mention candidates (all detector outputs for auditability)
    CREATE TABLE IF NOT EXISTS entity_mention_candidates (
        candidate_id TEXT PRIMARY KEY,
        message_id TEXT NOT NULL,
        detector TEXT NOT NULL,
        detector_version TEXT NOT NULL,
        entity_type_hint TEXT NOT NULL,
        char_start INTEGER,
        char_end INTEGER,
        surface_text TEXT,
        surface_hash TEXT NOT NULL,
        confidence REAL NOT NULL,
        is_eligible INTEGER NOT NULL DEFAULT 1,
        suppressed_by_candidate_id TEXT,
        suppression_reason TEXT,
        raw_candidate_json TEXT,
        FOREIGN KEY (message_id) REFERENCES messages(message_id)
    );

    CREATE INDEX IF NOT EXISTS idx_candidates_message 
        ON entity_mention_candidates(message_id);
    CREATE INDEX IF NOT EXISTS idx_candidates_eligible 
        ON entity_mention_candidates(message_id, is_eligible);

    -- Entity mentions (emitted winners only)
    CREATE TABLE IF NOT EXISTS entity_mentions (
        mention_id TEXT PRIMARY KEY,
        message_id TEXT NOT NULL,
        entity_id TEXT,
        candidate_id TEXT NOT NULL,
        detector TEXT NOT NULL,
        detector_version TEXT NOT NULL,
        entity_type_hint TEXT NOT NULL,
        char_start INTEGER,
        char_end INTEGER,
        surface_text TEXT,
        surface_hash TEXT NOT NULL,
        confidence REAL NOT NULL,
        raw_mention_json TEXT,
        FOREIGN KEY (message_id) REFERENCES messages(message_id),
        FOREIGN KEY (candidate_id) REFERENCES entity_mention_candidates(candidate_id)
    );

    CREATE INDEX IF NOT EXISTS idx_mentions_message 
        ON entity_mentions(message_id);
    CREATE INDEX IF NOT EXISTS idx_mentions_entity 
        ON entity_mentions(entity_id);

    -- Time mentions
    CREATE TABLE IF NOT EXISTS time_mentions (
        time_mention_id TEXT PRIMARY KEY,
        message_id TEXT NOT NULL,
        char_start INTEGER NOT NULL,
        char_end INTEGER NOT NULL,
        surface_text TEXT NOT NULL,
        surface_hash TEXT NOT NULL,
        pattern_id TEXT NOT NULL,
        pattern_precedence INTEGER NOT NULL,
        anchor_time_utc TEXT,
        resolved_type TEXT NOT NULL,
        valid_from_utc TEXT,
        valid_to_utc TEXT,
        resolution_granularity TEXT,
        timezone_assumed TEXT NOT NULL,
        confidence REAL NOT NULL,
        raw_parse_json TEXT NOT NULL,
        FOREIGN KEY (message_id) REFERENCES messages(message_id)
    );

    CREATE INDEX IF NOT EXISTS idx_time_mentions_message 
        ON time_mentions(message_id);

    -- NER model runs (if enabled)
    CREATE TABLE IF NOT EXISTS ner_model_runs (
        run_id TEXT PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL,
        config_json TEXT NOT NULL,
        started_at_utc TEXT NOT NULL,
        completed_at_utc TEXT,
        raw_io_json TEXT
    );
    """

    def check_required_tables(self):
        """Verify that Stage 1 tables exist."""
        required = ["conversations", "messages", "message_parts"]
        cursor = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing = {row[0] for row in cursor.fetchall()}

        missing = set(required) - existing
        if missing:
            raise RuntimeError(
                f"Stage 1 tables missing: {missing}. Run Stage 1 first."
            )
        logger.info("Stage 1 tables verified")

    def initialize_stage2A_schema(self):
        """Create Stage 2A tables."""
        self.connection.executescript(self.STAGE2A_SCHEMA)
        logger.info("Stage 2A schema initialized")

    def get_messages_for_processing(self) -> Iterator[sqlite3.Row]:
        """
        Iterate through messages in deterministic order.
        Order: (conversation_id ASC, order_index ASC, message_id ASC)
        """
        query = """
            SELECT 
                m.message_id,
                m.conversation_id,
                m.role,
                m.text_raw,
                m.created_at_utc,
                m.timestamp_quality,
                m.order_index,
                m.code_fence_ranges_json,
                m.blockquote_ranges_json
            FROM messages m
            WHERE m.text_raw IS NOT NULL
            ORDER BY m.conversation_id ASC, m.order_index ASC, m.message_id ASC
        """
        cursor = self.connection.execute(query)
        for row in cursor:
            yield row

    def get_message_count(self) -> int:
        """Get count of messages with text_raw."""
        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE text_raw IS NOT NULL"
        )
        return cursor.fetchone()[0]

    def insert_entity_candidate(
        self,
        candidate_id: str,
        message_id: str,
        detector: str,
        detector_version: str,
        entity_type_hint: str,
        char_start: Optional[int],
        char_end: Optional[int],
        surface_text: Optional[str],
        surface_hash: str,
        confidence: float,
        is_eligible: int,
        suppressed_by: Optional[str],
        suppression_reason: Optional[str],
        raw_json: Optional[str]
    ):
        """Insert an entity mention candidate."""
        self.connection.execute(
            """
            INSERT INTO entity_mention_candidates (
                candidate_id, message_id, detector, detector_version,
                entity_type_hint, char_start, char_end, surface_text,
                surface_hash, confidence, is_eligible, suppressed_by_candidate_id,
                suppression_reason, raw_candidate_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id, message_id, detector, detector_version,
                entity_type_hint, char_start, char_end, surface_text,
                surface_hash, confidence, is_eligible, suppressed_by,
                suppression_reason, raw_json
            )
        )

    def insert_entity_mention(
        self,
        mention_id: str,
        message_id: str,
        candidate_id: str,
        detector: str,
        detector_version: str,
        entity_type_hint: str,
        char_start: Optional[int],
        char_end: Optional[int],
        surface_text: Optional[str],
        surface_hash: str,
        confidence: float,
        raw_json: Optional[str]
    ):
        """Insert an emitted entity mention."""
        self.connection.execute(
            """
            INSERT INTO entity_mentions (
                mention_id, message_id, entity_id, candidate_id,
                detector, detector_version, entity_type_hint,
                char_start, char_end, surface_text, surface_hash,
                confidence, raw_mention_json
            ) VALUES (?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mention_id, message_id, candidate_id,
                detector, detector_version, entity_type_hint,
                char_start, char_end, surface_text, surface_hash,
                confidence, raw_json
            )
        )

    def update_candidate_suppression(
        self,
        candidate_id: str,
        suppressed_by: str,
        reason: str
    ):
        """Mark a candidate as suppressed."""
        self.connection.execute(
            """
            UPDATE entity_mention_candidates
            SET is_eligible = 0,
                suppressed_by_candidate_id = ?,
                suppression_reason = ?
            WHERE candidate_id = ?
            """,
            (suppressed_by, reason, candidate_id)
        )

    def insert_time_mention(
        self,
        time_mention_id: str,
        message_id: str,
        char_start: int,
        char_end: int,
        surface_text: str,
        surface_hash: str,
        pattern_id: str,
        pattern_precedence: int,
        anchor_time_utc: Optional[str],
        resolved_type: str,
        valid_from_utc: Optional[str],
        valid_to_utc: Optional[str],
        resolution_granularity: Optional[str],
        timezone_assumed: str,
        confidence: float,
        raw_parse_json: str
    ):
        """Insert a time mention."""
        self.connection.execute(
            """
            INSERT INTO time_mentions (
                time_mention_id, message_id, char_start, char_end,
                surface_text, surface_hash, pattern_id, pattern_precedence,
                anchor_time_utc, resolved_type, valid_from_utc, valid_to_utc,
                resolution_granularity, timezone_assumed, confidence, raw_parse_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                time_mention_id, message_id, char_start, char_end,
                surface_text, surface_hash, pattern_id, pattern_precedence,
                anchor_time_utc, resolved_type, valid_from_utc, valid_to_utc,
                resolution_granularity, timezone_assumed, confidence, raw_parse_json
            )
        )

    def insert_ner_model_run(
        self,
        run_id: str,
        model_name: str,
        model_version: str,
        config_json: str,
        started_at_utc: str,
        completed_at_utc: Optional[str],
        raw_io_json: Optional[str]
    ):
        """Insert a NER model run record."""
        self.connection.execute(
            """
            INSERT INTO ner_model_runs (
                run_id, model_name, model_version, config_json,
                started_at_utc, completed_at_utc, raw_io_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, model_name, model_version, config_json,
                started_at_utc, completed_at_utc, raw_io_json
            )
        )

    def update_ner_model_run(self, run_id: str, completed_at_utc: str, raw_io_json: str):
        """Update NER model run with completion data."""
        self.connection.execute(
            """
            UPDATE ner_model_runs
            SET completed_at_utc = ?, raw_io_json = ?
            WHERE run_id = ?
            """,
            (completed_at_utc, raw_io_json, run_id)
        )

    def get_candidate_stats(self) -> Dict[str, int]:
        """Get candidate statistics."""
        stats = {}

        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM entity_mention_candidates"
        )
        stats["total_candidates"] = cursor.fetchone()[0]

        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM entity_mention_candidates WHERE is_eligible = 1"
        )
        stats["eligible_candidates"] = cursor.fetchone()[0]

        cursor = self.connection.execute(
            """
            SELECT suppression_reason, COUNT(*) 
            FROM entity_mention_candidates 
            WHERE is_eligible = 0 AND suppression_reason IS NOT NULL
            GROUP BY suppression_reason
            """
        )
        for row in cursor:
            stats[f"suppressed_{row[0].lower()}"] = row[1]

        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM entity_mentions"
        )
        stats["emitted_mentions"] = cursor.fetchone()[0]

        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM time_mentions"
        )
        stats["time_mentions"] = cursor.fetchone()[0]

        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM time_mentions WHERE resolved_type != 'unresolved'"
        )
        stats["time_mentions_resolved"] = cursor.fetchone()[0]

        return stats


# ============================================================================
# DETECTORS
# ============================================================================

class EntityDetector(ABC):
    """Base class for entity detectors."""

    def __init__(self, config: Stage2AConfig):
        self.config = config

    @property
    @abstractmethod
    def detector_name(self) -> str:
        """Return the detector name."""
        pass

    @property
    def detector_version(self) -> str:
        """Return the detector version."""
        return self.config.detector_versions.get(self.detector_name, "1.0.0")

    @abstractmethod
    def detect(self, text: str) -> List[EntityCandidate]:
        """Detect entities in text. Returns list of candidates."""
        pass


class EmailDetector(EntityDetector):
    """Detect email addresses."""

    # RFC 5322 simplified pattern
    PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.EMAIL

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.EMAIL,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group(),
                confidence=0.95,
                raw_data={"pattern": "rfc5322_simplified"}
            ))
        return candidates


class URLDetector(EntityDetector):
    """Detect HTTP/HTTPS URLs."""

    # Match http:// or https:// URLs
    PATTERN = re.compile(
        r'\bhttps?://[^\s<>\[\]{}|\\^`"\']+',
        re.IGNORECASE
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.URL

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            url = match.group()
            # Clean trailing punctuation
            while url and url[-1] in '.,;:!?)>]':
                url = url[:-1]

            if len(url) > 10:  # Minimum viable URL length
                candidates.append(EntityCandidate(
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint=EntityTypeHint.URL,
                    char_start=match.start(),
                    char_end=match.start() + len(url),
                    surface_text=url,
                    confidence=0.95,
                    raw_data={"original_match": match.group()}
                ))
        return candidates


class DOIDetector(EntityDetector):
    """Detect Digital Object Identifiers."""

    # DOI pattern: 10.XXXX/suffix
    PATTERN = re.compile(
        r'\b(?:doi:?\s*)?10\.\d{4,}/[^\s<>\[\]{}|\\^`"\']+',
        re.IGNORECASE
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.DOI

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            doi = match.group()
            # Clean trailing punctuation
            while doi and doi[-1] in '.,;:!?)>]':
                doi = doi[:-1]

            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.DOI,
                char_start=match.start(),
                char_end=match.start() + len(doi),
                surface_text=doi,
                confidence=0.90,
                raw_data={}
            ))
        return candidates


class UUIDDetector(EntityDetector):
    """Detect UUIDs (v1-v5 format)."""

    PATTERN = re.compile(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.UUID

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.UUID,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group(),
                confidence=0.95,
                raw_data={"version": match.group()[14]}  # UUID version digit
            ))
        return candidates


class HashHexDetector(EntityDetector):
    """Detect hexadecimal hashes (SHA-256, SHA-1, MD5, etc.)."""

    # Match hex strings of common hash lengths
    # MD5: 32, SHA-1: 40, SHA-256: 64, SHA-512: 128
    PATTERN = re.compile(
        r'\b[0-9a-fA-F]{32}\b|\b[0-9a-fA-F]{40}\b|\b[0-9a-fA-F]{64}\b|\b[0-9a-fA-F]{128}\b'
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.HASH_HEX

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            hex_str = match.group()
            # Determine hash type by length
            length = len(hex_str)
            hash_type = {32: "MD5", 40: "SHA-1", 64: "SHA-256", 128: "SHA-512"}.get(length, "UNKNOWN")

            # Lower confidence for shorter hashes (could be other hex data)
            confidence = 0.85 if length >= 64 else 0.70

            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.HASH_HEX,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=hex_str,
                confidence=confidence,
                raw_data={"hash_type": hash_type, "length": length}
            ))
        return candidates


class IPAddressDetector(EntityDetector):
    """Detect IPv4 and IPv6 addresses."""

    # IPv4 pattern
    IPV4_PATTERN = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )

    # Simplified IPv6 pattern
    IPV6_PATTERN = re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b|'
        r'\b(?:[0-9a-fA-F]{1,4}:){1,7}:\b|'
        r'\b::(?:[0-9a-fA-F]{1,4}:){0,6}[0-9a-fA-F]{1,4}\b'
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.IP_ADDRESS

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        # IPv4
        for match in self.IPV4_PATTERN.finditer(text):
            ip = match.group()
            # Validate octets
            octets = ip.split('.')
            if all(0 <= int(o) <= 255 for o in octets):
                candidates.append(EntityCandidate(
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint=EntityTypeHint.IP_ADDRESS,
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=ip,
                    confidence=0.90,
                    raw_data={"version": "IPv4"}
                ))

        # IPv6
        for match in self.IPV6_PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.IP_ADDRESS,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group(),
                confidence=0.85,
                raw_data={"version": "IPv6"}
            ))

        return candidates


class PhoneDetector(EntityDetector):
    """Detect phone numbers in various formats."""

    # International and US phone patterns
    PATTERNS = [
        # International: +1-234-567-8901 or +1 234 567 8901
        re.compile(r'\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
        # US: (123) 456-7890 or 123-456-7890
        re.compile(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
    ]

    @property
    def detector_name(self) -> str:
        return DetectorType.PHONE

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        seen_spans = set()

        for pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if span not in seen_spans:
                    seen_spans.add(span)
                    phone = match.group()
                    # Basic validation: must have at least 10 digits
                    digits = re.sub(r'\D', '', phone)
                    if len(digits) >= 10:
                        candidates.append(EntityCandidate(
                            detector=self.detector_name,
                            detector_version=self.detector_version,
                            entity_type_hint=EntityTypeHint.PHONE,
                            char_start=match.start(),
                            char_end=match.end(),
                            surface_text=phone,
                            confidence=0.80,
                            raw_data={"digits": digits}
                        ))

        return candidates


class FilePathDetector(EntityDetector):
    """Detect file paths (Unix and Windows)."""

    # Unix paths
    UNIX_PATTERN = re.compile(
        r'(?:^|[\s"\'`])(/(?:[^/\s"\'`\x00-\x1f]+/)*[^/\s"\'`\x00-\x1f]+)',
        re.MULTILINE
    )

    # Windows paths
    WINDOWS_PATTERN = re.compile(
        r'\b[A-Za-z]:\\(?:[^\\/:*?"<>|\s]+\\)*[^\\/:*?"<>|\s]+',
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.FILEPATH

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        # Unix paths
        for match in self.UNIX_PATTERN.finditer(text):
            path = match.group(1)
            # Filter out common false positives
            if path.startswith('/usr/') or path.startswith('/home/') or \
               path.startswith('/var/') or path.startswith('/etc/') or \
               '.' in path.split('/')[-1]:  # Has file extension
                candidates.append(EntityCandidate(
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint=EntityTypeHint.FILEPATH,
                    char_start=match.start(1),
                    char_end=match.end(1),
                    surface_text=path,
                    confidence=0.75,
                    raw_data={"type": "unix"}
                ))

        # Windows paths
        for match in self.WINDOWS_PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.FILEPATH,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group(),
                confidence=0.80,
                raw_data={"type": "windows"}
            ))

        return candidates


class BareDomainDetector(EntityDetector):
    """Detect bare domain names (without protocol)."""

    # Domain pattern without protocol
    PATTERN = re.compile(
        r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.BARE_DOMAIN

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        for match in self.PATTERN.finditer(text):
            domain = match.group()

            # Skip if it looks like an email (has @ before it)
            if match.start() > 0 and text[match.start() - 1] == '@':
                continue

            # Extract TLD
            tld = domain.split('.')[-1].lower()

            # Validate TLD if allowlist enabled
            if self.config.domain_tld_allowlist_enabled:
                if tld not in self.config.domain_tld_allowlist:
                    continue

            # Skip common false positives (file extensions, version numbers)
            if domain.lower() in {'i.e.', 'e.g.', 'etc.'}:
                continue

            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.BARE_DOMAIN,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=domain,
                confidence=0.70,
                raw_data={"tld": tld}
            ))

        return candidates


class ArxivDetector(EntityDetector):
    """Detect arXiv identifiers."""

    # arXiv patterns: old (hep-th/9901001) and new (1234.56789)
    PATTERNS = [
        re.compile(r'\barXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)\b', re.IGNORECASE),
        re.compile(r'\barXiv:\s*([a-z-]+/\d{7}(?:v\d+)?)\b', re.IGNORECASE),
        re.compile(r'\b(\d{4}\.\d{4,5}(?:v\d+)?)\b'),  # Bare new format
    ]

    @property
    def detector_name(self) -> str:
        return DetectorType.ARXIV

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []
        seen_ids = set()

        for pattern in self.PATTERNS:
            for match in pattern.finditer(text):
                arxiv_id = match.group(1) if match.lastindex else match.group()

                if arxiv_id in seen_ids:
                    continue
                seen_ids.add(arxiv_id)

                candidates.append(EntityCandidate(
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint=EntityTypeHint.ARXIV,
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=match.group(),
                    confidence=0.90,
                    raw_data={"arxiv_id": arxiv_id}
                ))

        return candidates


class CVEDetector(EntityDetector):
    """Detect CVE (Common Vulnerabilities and Exposures) identifiers."""

    PATTERN = re.compile(r'\bCVE-\d{4}-\d{4,}\b', re.IGNORECASE)

    @property
    def detector_name(self) -> str:
        return DetectorType.CVE

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        for match in self.PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.CVE,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group().upper(),
                confidence=0.95,
                raw_data={}
            ))

        return candidates


class ORCIDDetector(EntityDetector):
    """Detect ORCID identifiers."""

    # ORCID pattern: 0000-0002-1825-0097
    PATTERN = re.compile(
        r'\b(?:https?://orcid\.org/)?(\d{4}-\d{4}-\d{4}-\d{3}[\dX])\b',
        re.IGNORECASE
    )

    @property
    def detector_name(self) -> str:
        return DetectorType.ORCID

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        for match in self.PATTERN.finditer(text):
            orcid = match.group(1)

            # Validate checksum (last digit)
            digits = orcid.replace('-', '')
            if self._validate_orcid_checksum(digits):
                candidates.append(EntityCandidate(
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint=EntityTypeHint.ORCID,
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=match.group(),
                    confidence=0.95,
                    raw_data={"orcid": orcid}
                ))

        return candidates

    @staticmethod
    def _validate_orcid_checksum(digits: str) -> bool:
        """Validate ORCID checksum using ISO 7064 Mod 11-2."""
        total = 0
        for char in digits[:-1]:
            total = (total + int(char)) * 2
        remainder = total % 11
        check = (12 - remainder) % 11
        expected = 'X' if check == 10 else str(check)
        return digits[-1].upper() == expected


class HandleDetector(EntityDetector):
    """Detect @handles (Twitter/X style)."""

    PATTERN = re.compile(r'(?<![a-zA-Z0-9_])@([a-zA-Z_][a-zA-Z0-9_]{2,})\b')

    @property
    def detector_name(self) -> str:
        return DetectorType.HANDLE

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        for match in self.PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.HANDLE,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group(),
                confidence=0.75,
                raw_data={"handle": match.group(1)}
            ))

        return candidates


class HashtagDetector(EntityDetector):
    """Detect #hashtags."""

    PATTERN = re.compile(r'(?<![a-zA-Z0-9_])#([a-zA-Z][a-zA-Z0-9_]{2,})\b')

    @property
    def detector_name(self) -> str:
        return DetectorType.HASHTAG

    def detect(self, text: str) -> List[EntityCandidate]:
        candidates = []

        for match in self.PATTERN.finditer(text):
            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=EntityTypeHint.HASHTAG,
                char_start=match.start(),
                char_end=match.end(),
                surface_text=match.group(),
                confidence=0.80,
                raw_data={"tag": match.group(1)}
            ))

        return candidates


class NERDetector(EntityDetector):
    """Named Entity Recognition detector using spaCy."""

    def __init__(self, config: Stage2AConfig):
        super().__init__(config)
        self._nlp = None
        self._model_version: Optional[str] = None

    @property
    def detector_name(self) -> str:
        return f"NER:{self.config.ner_model_name}"

    @property
    def detector_version(self) -> str:
        return self._model_version or "unknown"

    def _load_model(self):
        """Lazy load spaCy model."""
        if self._nlp is not None:
            return

        try:
            import spacy
            self._nlp = spacy.load(self.config.ner_model_name)
            self._model_version = self._nlp.meta.get("version", "unknown")
            logger.info(f"Loaded spaCy model {self.config.ner_model_name} v{self._model_version}")
        except ImportError:
            logger.warning("spaCy not installed. NER detection disabled.")
            self._nlp = None
        except OSError:
            logger.warning(f"spaCy model {self.config.ner_model_name} not found. NER detection disabled.")
            self._nlp = None

    def detect(self, text: str) -> List[EntityCandidate]:
        """Detect named entities using spaCy NER."""
        if not self.config.enable_ner:
            return []

        self._load_model()
        if self._nlp is None:
            return []

        candidates = []

        # Handle long texts with chunking
        if len(text) <= self.config.ner_max_chars:
            candidates.extend(self._process_chunk(text, offset=0))
        else:
            # Chunk with stride for overlap
            chunk_size = self.config.ner_max_chars
            stride = self.config.ner_stride

            for start in range(0, len(text), chunk_size - stride):
                end = min(start + chunk_size, len(text))
                chunk = text[start:end]
                chunk_candidates = self._process_chunk(chunk, offset=start)
                candidates.extend(chunk_candidates)

                if end >= len(text):
                    break

            # Deduplicate overlapping detections
            candidates = self._deduplicate_candidates(candidates)

        return candidates

    def _process_chunk(self, text: str, offset: int) -> List[EntityCandidate]:
        """Process a text chunk and return candidates with adjusted offsets."""
        candidates = []
        doc = self._nlp(text)

        for ent in doc.ents:
            # Filter by allowlist
            if ent.label_ not in self.config.ner_label_allowlist:
                continue

            # Map spaCy labels to our type hints
            entity_type = self._map_label(ent.label_)

            candidates.append(EntityCandidate(
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=entity_type,
                char_start=ent.start_char + offset,
                char_end=ent.end_char + offset,
                surface_text=ent.text,
                confidence=0.80,  # spaCy doesn't provide confidence
                raw_data={
                    "label": ent.label_,
                    "model": self.config.ner_model_name
                }
            ))

        return candidates

    def _map_label(self, label: str) -> str:
        """Map spaCy label to our entity type hint."""
        mapping = {
            "PERSON": EntityTypeHint.PERSON,
            "PER": EntityTypeHint.PERSON,
            "ORG": EntityTypeHint.ORG,
            "GPE": EntityTypeHint.LOCATION,
            "LOC": EntityTypeHint.LOCATION,
            "LOCATION": EntityTypeHint.LOCATION,
        }
        return mapping.get(label, EntityTypeHint.OTHER)

    def _deduplicate_candidates(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Remove duplicate candidates from overlapping chunks."""
        if not candidates:
            return candidates

        # Group by unique key
        unique = {}
        for c in candidates:
            key = (c.char_start, c.char_end, c.entity_type_hint, c.surface_text)
            if key not in unique or c.confidence > unique[key].confidence:
                unique[key] = c

        return list(unique.values())

    def get_run_info(self) -> Dict[str, Any]:
        """Get model run information for logging."""
        return {
            "model_name": self.config.ner_model_name,
            "model_version": self._model_version or "unknown",
            "max_chars": self.config.ner_max_chars,
            "stride": self.config.ner_stride,
            "label_allowlist": list(self.config.ner_label_allowlist)
        }


# ============================================================================
# TIME DETECTION
# ============================================================================

@dataclass
class TimePattern:
    """A time expression pattern with resolution logic."""
    pattern_id: str
    precedence: int  # Lower = stronger
    regex: re.Pattern
    resolver: Callable[[re.Match, Optional[pendulum.DateTime]], Dict[str, Any]]
    confidence: float


class TimeDetector:
    """
    Detect and resolve time expressions with conservative anchoring.

    Resolution is conservative: relative expressions only resolve when
    anchor time has 'original' timestamp quality.
    """

    def __init__(self, config: Stage2AConfig):
        self.config = config
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> List[TimePattern]:
        """Build ordered list of time patterns."""
        patterns = []

        # ISO 8601 full datetime (highest precedence)
        patterns.append(TimePattern(
            pattern_id="ISO_DATETIME",
            precedence=10,
            regex=re.compile(
                r'\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\b'
            ),
            resolver=self._resolve_iso_datetime,
            confidence=0.95
        ))

        # ISO 8601 date only
        patterns.append(TimePattern(
            pattern_id="ISO_DATE",
            precedence=20,
            regex=re.compile(r'\b(\d{4}-\d{2}-\d{2})\b'),
            resolver=self._resolve_iso_date,
            confidence=0.90
        ))

        # Written dates: January 15, 2024 or 15 January 2024
        patterns.append(TimePattern(
            pattern_id="WRITTEN_DATE",
            precedence=30,
            regex=re.compile(
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b|'
                r'\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
                re.IGNORECASE
            ),
            resolver=self._resolve_written_date,
            confidence=0.90
        ))

        # Month and year: January 2024
        patterns.append(TimePattern(
            pattern_id="MONTH_YEAR",
            precedence=40,
            regex=re.compile(
                r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
                re.IGNORECASE
            ),
            resolver=self._resolve_month_year,
            confidence=0.85
        ))

        # Year only (4 digits, reasonable range)
        patterns.append(TimePattern(
            pattern_id="YEAR_ONLY",
            precedence=50,
            regex=re.compile(r'\b(19\d{2}|20[0-4]\d|2050)\b'),
            resolver=self._resolve_year,
            confidence=0.60
        ))

        # Relative: yesterday, today, tomorrow
        patterns.append(TimePattern(
            pattern_id="RELATIVE_DAY",
            precedence=60,
            regex=re.compile(r'\b(yesterday|today|tomorrow)\b', re.IGNORECASE),
            resolver=self._resolve_relative_day,
            confidence=0.85
        ))

        # Relative: last/next week/month/year
        patterns.append(TimePattern(
            pattern_id="RELATIVE_PERIOD",
            precedence=70,
            regex=re.compile(
                r'\b(last|next)\s+(week|month|year)\b',
                re.IGNORECASE
            ),
            resolver=self._resolve_relative_period,
            confidence=0.80
        ))

        # Relative: N days/weeks/months ago
        patterns.append(TimePattern(
            pattern_id="RELATIVE_AGO",
            precedence=80,
            regex=re.compile(
                r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b',
                re.IGNORECASE
            ),
            resolver=self._resolve_relative_ago,
            confidence=0.80
        ))

        # Relative: in N days/weeks/months
        patterns.append(TimePattern(
            pattern_id="RELATIVE_IN",
            precedence=80,
            regex=re.compile(
                r'\bin\s+(\d+)\s+(days?|weeks?|months?|years?)\b',
                re.IGNORECASE
            ),
            resolver=self._resolve_relative_in,
            confidence=0.75
        ))

        # Day of week: Monday, Tuesday, etc.
        patterns.append(TimePattern(
            pattern_id="DAY_OF_WEEK",
            precedence=90,
            regex=re.compile(
                r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
                re.IGNORECASE
            ),
            resolver=self._resolve_day_of_week,
            confidence=0.60
        ))

        return patterns

    def detect(
        self,
        text: str,
        anchor_time_utc: Optional[str],
        timestamp_quality: Optional[str]
    ) -> List[TimeCandidate]:
        """Detect all time expressions in text."""
        candidates = []

        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                surface = match.group()
                char_start = match.start()
                char_end = match.end()

                candidates.append(TimeCandidate(
                    char_start=char_start,
                    char_end=char_end,
                    surface_text=surface,
                    pattern_id=pattern.pattern_id,
                    pattern_precedence=pattern.precedence,
                    parsed_data={
                        "match_groups": match.groups(),
                        "anchor_time_utc": anchor_time_utc,
                        "timestamp_quality": timestamp_quality,
                        "confidence": pattern.confidence
                    }
                ))

        return candidates

    def resolve(
        self,
        candidate: TimeCandidate,
        anchor_time_utc: Optional[str],
        timestamp_quality: Optional[str]
    ) -> Dict[str, Any]:
        """
        Resolve a time candidate to a concrete time.

        Conservative resolution: relative expressions only resolve when
        timestamp_quality is 'original'.
        """
        # Find matching pattern
        pattern = next(
            (p for p in self.patterns if p.pattern_id == candidate.pattern_id),
            None
        )
        if pattern is None:
            return self._unresolved_result(candidate, "pattern_not_found")

        # Parse anchor time
        anchor_dt = None
        if anchor_time_utc:
            anchor_dt = TimestampUtils.parse_iso(anchor_time_utc)

        # Check if pattern requires anchor
        is_relative = candidate.pattern_id.startswith("RELATIVE_") or \
                      candidate.pattern_id == "DAY_OF_WEEK"

        if is_relative:
            # Conservative: only resolve if original timestamp
            if timestamp_quality != "original":
                return self._unresolved_result(
                    candidate,
                    f"relative_with_{timestamp_quality}_timestamp"
                )
            if anchor_dt is None:
                return self._unresolved_result(candidate, "no_anchor_time")

        # Run resolver
        try:
            # Create a pseudo-match for the resolver
            match = pattern.regex.search(candidate.surface_text)
            if match is None:
                return self._unresolved_result(candidate, "match_failed")

            result = pattern.resolver(match, anchor_dt)
            result["confidence"] = pattern.confidence
            return result
        except Exception as e:
            return self._unresolved_result(candidate, f"resolver_error: {e}")

    def _unresolved_result(self, candidate: TimeCandidate, reason: str) -> Dict[str, Any]:
        """Create an unresolved result."""
        return {
            "resolved_type": TimeResolvedType.UNRESOLVED,
            "valid_from_utc": None,
            "valid_to_utc": None,
            "resolution_granularity": None,
            "confidence": 0.0,
            "reason": reason
        }

    def _resolve_iso_datetime(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve ISO 8601 datetime."""
        try:
            dt = pendulum.parse(match.group(1), strict=False)
            dt_utc = dt.in_timezone("UTC")
            return {
                "resolved_type": TimeResolvedType.INSTANT,
                "valid_from_utc": dt_utc.format(TimestampUtils.ISO_UTC_MILLIS),
                "valid_to_utc": None,
                "resolution_granularity": TimeGranularity.SECOND
            }
        except Exception:
            return self._unresolved_result(None, "parse_failed")

    def _resolve_iso_date(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve ISO 8601 date to day interval."""
        try:
            dt = pendulum.parse(match.group(1), strict=False)
            start = dt.start_of("day").in_timezone("UTC")
            end = dt.end_of("day").in_timezone("UTC")
            return {
                "resolved_type": TimeResolvedType.INTERVAL,
                "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
                "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
                "resolution_granularity": TimeGranularity.DAY
            }
        except Exception:
            return self._unresolved_result(None, "parse_failed")

    def _resolve_written_date(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve written date like 'January 15, 2024'."""
        try:
            # Get the matched group (either format)
            date_str = match.group(1) or match.group(2)
            dt = pendulum.parse(date_str, strict=False)
            start = dt.start_of("day").in_timezone("UTC")
            end = dt.end_of("day").in_timezone("UTC")
            return {
                "resolved_type": TimeResolvedType.INTERVAL,
                "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
                "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
                "resolution_granularity": TimeGranularity.DAY
            }
        except Exception:
            return self._unresolved_result(None, "parse_failed")

    def _resolve_month_year(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve 'January 2024' to month interval."""
        try:
            dt = pendulum.parse(match.group(1), strict=False)
            start = dt.start_of("month").in_timezone("UTC")
            end = dt.end_of("month").in_timezone("UTC")
            return {
                "resolved_type": TimeResolvedType.INTERVAL,
                "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
                "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
                "resolution_granularity": TimeGranularity.MONTH
            }
        except Exception:
            return self._unresolved_result(None, "parse_failed")

    def _resolve_year(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve year to year interval."""
        try:
            year = int(match.group(1))
            start = pendulum.datetime(year, 1, 1, tz="UTC")
            end = pendulum.datetime(year, 12, 31, 23, 59, 59, tz="UTC")
            return {
                "resolved_type": TimeResolvedType.INTERVAL,
                "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
                "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
                "resolution_granularity": TimeGranularity.YEAR
            }
        except Exception:
            return self._unresolved_result(None, "parse_failed")

    def _resolve_relative_day(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve yesterday/today/tomorrow."""
        if anchor is None:
            return self._unresolved_result(None, "no_anchor")

        word = match.group(1).lower()
        if word == "yesterday":
            dt = anchor.subtract(days=1)
        elif word == "tomorrow":
            dt = anchor.add(days=1)
        else:  # today
            dt = anchor

        start = dt.start_of("day").in_timezone("UTC")
        end = dt.end_of("day").in_timezone("UTC")
        return {
            "resolved_type": TimeResolvedType.INTERVAL,
            "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
            "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
            "resolution_granularity": TimeGranularity.DAY
        }

    def _resolve_relative_period(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve last/next week/month/year."""
        if anchor is None:
            return self._unresolved_result(None, "no_anchor")

        direction = match.group(1).lower()
        unit = match.group(2).lower()

        if direction == "last":
            if unit == "week":
                dt = anchor.subtract(weeks=1)
                start = dt.start_of("week").in_timezone("UTC")
                end = dt.end_of("week").in_timezone("UTC")
                granularity = TimeGranularity.DAY
            elif unit == "month":
                dt = anchor.subtract(months=1)
                start = dt.start_of("month").in_timezone("UTC")
                end = dt.end_of("month").in_timezone("UTC")
                granularity = TimeGranularity.MONTH
            else:  # year
                dt = anchor.subtract(years=1)
                start = dt.start_of("year").in_timezone("UTC")
                end = dt.end_of("year").in_timezone("UTC")
                granularity = TimeGranularity.YEAR
        else:  # next
            if unit == "week":
                dt = anchor.add(weeks=1)
                start = dt.start_of("week").in_timezone("UTC")
                end = dt.end_of("week").in_timezone("UTC")
                granularity = TimeGranularity.DAY
            elif unit == "month":
                dt = anchor.add(months=1)
                start = dt.start_of("month").in_timezone("UTC")
                end = dt.end_of("month").in_timezone("UTC")
                granularity = TimeGranularity.MONTH
            else:  # year
                dt = anchor.add(years=1)
                start = dt.start_of("year").in_timezone("UTC")
                end = dt.end_of("year").in_timezone("UTC")
                granularity = TimeGranularity.YEAR

        return {
            "resolved_type": TimeResolvedType.INTERVAL,
            "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
            "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
            "resolution_granularity": granularity
        }

    def _resolve_relative_ago(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve 'N days/weeks/months ago'."""
        if anchor is None:
            return self._unresolved_result(None, "no_anchor")

        amount = int(match.group(1))
        unit = match.group(2).lower().rstrip('s')  # Normalize plural

        if unit == "day":
            dt = anchor.subtract(days=amount)
            granularity = TimeGranularity.DAY
        elif unit == "week":
            dt = anchor.subtract(weeks=amount)
            granularity = TimeGranularity.DAY
        elif unit == "month":
            dt = anchor.subtract(months=amount)
            granularity = TimeGranularity.MONTH
        else:  # year
            dt = anchor.subtract(years=amount)
            granularity = TimeGranularity.YEAR

        start = dt.start_of("day").in_timezone("UTC")
        end = dt.end_of("day").in_timezone("UTC")

        return {
            "resolved_type": TimeResolvedType.INTERVAL,
            "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
            "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
            "resolution_granularity": granularity
        }

    def _resolve_relative_in(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve 'in N days/weeks/months'."""
        if anchor is None:
            return self._unresolved_result(None, "no_anchor")

        amount = int(match.group(1))
        unit = match.group(2).lower().rstrip('s')

        if unit == "day":
            dt = anchor.add(days=amount)
            granularity = TimeGranularity.DAY
        elif unit == "week":
            dt = anchor.add(weeks=amount)
            granularity = TimeGranularity.DAY
        elif unit == "month":
            dt = anchor.add(months=amount)
            granularity = TimeGranularity.MONTH
        else:  # year
            dt = anchor.add(years=amount)
            granularity = TimeGranularity.YEAR

        start = dt.start_of("day").in_timezone("UTC")
        end = dt.end_of("day").in_timezone("UTC")

        return {
            "resolved_type": TimeResolvedType.INTERVAL,
            "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
            "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
            "resolution_granularity": granularity
        }

    def _resolve_day_of_week(
        self, match: re.Match, anchor: Optional[pendulum.DateTime]
    ) -> Dict[str, Any]:
        """Resolve day of week (assume most recent past occurrence)."""
        if anchor is None:
            return self._unresolved_result(None, "no_anchor")

        day_name = match.group(1).capitalize()
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6
        }

        target_day = day_map.get(day_name)
        if target_day is None:
            return self._unresolved_result(None, "invalid_day")

        current_day = anchor.day_of_week
        days_back = (current_day - target_day) % 7
        if days_back == 0:
            days_back = 7  # Assume last week if same day

        dt = anchor.subtract(days=days_back)
        start = dt.start_of("day").in_timezone("UTC")
        end = dt.end_of("day").in_timezone("UTC")

        return {
            "resolved_type": TimeResolvedType.INTERVAL,
            "valid_from_utc": start.format(TimestampUtils.ISO_UTC_MILLIS),
            "valid_to_utc": end.format(TimestampUtils.ISO_UTC_MILLIS),
            "resolution_granularity": TimeGranularity.DAY
        }


# ============================================================================
# OVERLAP RESOLUTION
# ============================================================================

class OverlapResolver:
    """
    Resolve overlapping entity candidates using deterministic greedy algorithm.

    Sorting criteria (in order):
    1. confidence DESC
    2. span_length DESC
    3. detector_order ASC
    4. char_start ASC
    5. char_end DESC
    6. surface_hash ASC
    """

    # Fixed detector order for tie-breaking
    DETECTOR_ORDER = {
        DetectorType.EMAIL: 1,
        DetectorType.URL: 2,
        DetectorType.DOI: 3,
        DetectorType.UUID: 4,
        DetectorType.HASH_HEX: 5,
        DetectorType.IP_ADDRESS: 6,
        DetectorType.PHONE: 7,
        DetectorType.FILEPATH: 8,
        DetectorType.BARE_DOMAIN: 9,
        DetectorType.ARXIV: 10,
        DetectorType.CVE: 11,
        DetectorType.ORCID: 12,
        DetectorType.HANDLE: 13,
        DetectorType.HASHTAG: 14,
        DetectorType.NER: 100,  # NER has lowest priority
        DetectorType.LEXICON: 50,
    }

    @staticmethod
    def get_detector_priority(detector: str) -> int:
        """Get priority for a detector (lower = higher priority)."""
        # Handle NER:model_name format
        if detector.startswith("NER:"):
            return OverlapResolver.DETECTOR_ORDER.get(DetectorType.NER, 100)
        if detector.startswith("LEXICON:"):
            return OverlapResolver.DETECTOR_ORDER.get(DetectorType.LEXICON, 50)
        return OverlapResolver.DETECTOR_ORDER.get(detector, 99)

    @staticmethod
    def sort_key(candidate: Dict[str, Any]) -> Tuple:
        """Generate sort key for a candidate."""
        char_start = candidate.get("char_start")
        char_end = candidate.get("char_end")
        span_length = (char_end - char_start) if (char_start is not None and char_end is not None) else 0

        return (
            -candidate.get("confidence", 0),  # DESC
            -span_length,  # DESC
            OverlapResolver.get_detector_priority(candidate.get("detector", "")),  # ASC
            char_start if char_start is not None else float('inf'),  # ASC
            -(char_end if char_end is not None else 0),  # DESC
            candidate.get("surface_hash", ""),  # ASC
        )

    @staticmethod
    def resolve(candidates: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Tuple[Dict, str, str]]]:
        """
        Resolve overlapping candidates.

        Returns:
            Tuple of (winners, suppressions) where suppressions is
            list of (candidate, winner_id, reason)
        """
        if not candidates:
            return [], []

        # Filter to eligible candidates with valid spans
        eligible = [
            c for c in candidates
            if c.get("is_eligible", True) and
               c.get("char_start") is not None and
               c.get("char_end") is not None
        ]

        # Sort by criteria
        sorted_candidates = sorted(eligible, key=OverlapResolver.sort_key)

        winners = []
        winner_spans: List[TextSpan] = []
        suppressions = []

        for candidate in sorted_candidates:
            span = TextSpan(candidate["char_start"], candidate["char_end"])

            # Check for overlap with existing winners
            overlaps = False
            overlapping_winner = None
            for i, winner_span in enumerate(winner_spans):
                if span.overlaps(winner_span):
                    overlaps = True
                    overlapping_winner = winners[i]
                    break

            if overlaps:
                suppressions.append((
                    candidate,
                    overlapping_winner["candidate_id"],
                    SuppressionReason.OVERLAP_HIGHER_SCORE
                ))
            else:
                winners.append(candidate)
                winner_spans.append(span)

        return winners, suppressions

    @staticmethod
    def resolve_time_candidates(
        candidates: List[TimeCandidate]
    ) -> Tuple[List[TimeCandidate], List[TimeCandidate]]:
        """
        Resolve overlapping time candidates.

        Sorting:
        1. span_length DESC
        2. pattern_precedence ASC
        3. char_start ASC
        4. char_end DESC
        5. surface_hash ASC
        """
        if not candidates:
            return [], []

        def sort_key(c: TimeCandidate) -> Tuple:
            span_length = c.char_end - c.char_start
            surface_hash = HashUtils.sha256_string(c.surface_text)
            return (
                -span_length,
                c.pattern_precedence,
                c.char_start,
                -c.char_end,
                surface_hash
            )

        sorted_candidates = sorted(candidates, key=sort_key)

        winners = []
        winner_spans: List[TextSpan] = []
        suppressed = []

        for candidate in sorted_candidates:
            span = TextSpan(candidate.char_start, candidate.char_end)

            overlaps = any(span.overlaps(ws) for ws in winner_spans)

            if overlaps:
                suppressed.append(candidate)
            else:
                winners.append(candidate)
                winner_spans.append(span)

        return winners, suppressed


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class Stage2APipeline:
    """
    Stage 2A: Entity & Time Detection Layer

    Phases:
    1. Initialize - begin transaction, capture build metadata
    2. Build exclusion ranges - load code fence and blockquote ranges per message
    3. Detect candidates - run detectors in fixed order
    4. Eligibility + overlap resolution - filter and select winners
    5. Time mentions - detect, resolve, and store time expressions
    """

    def __init__(self, config: Stage2AConfig):
        self.config = config
        self.db = Stage2ADatabase(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.stage_started_at_utc = TimestampUtils.now_utc()

        # Initialize detectors
        self.detectors = self._build_detector_chain()
        self.time_detector = TimeDetector(config)

        # Statistics
        self.stats: Dict[str, int] = {
            "messages_processed": 0,
            "total_candidates": 0,
            "eligible_candidates": 0,
            "emitted_mentions": 0,
            "time_mentions": 0,
            "time_mentions_resolved": 0,
        }

    def _build_detector_chain(self) -> List[EntityDetector]:
        """Build ordered chain of entity detectors."""
        detectors = [
            EmailDetector(self.config),
            URLDetector(self.config),
            DOIDetector(self.config),
            UUIDDetector(self.config),
            HashHexDetector(self.config),
            IPAddressDetector(self.config),
            PhoneDetector(self.config),
            FilePathDetector(self.config),
            BareDomainDetector(self.config),
        ]

        # Optional detectors
        if self.config.enable_arxiv:
            detectors.append(ArxivDetector(self.config))
        if self.config.enable_cve:
            detectors.append(CVEDetector(self.config))
        if self.config.enable_orcid:
            detectors.append(ORCIDDetector(self.config))
        if self.config.enable_handle:
            detectors.append(HandleDetector(self.config))
        if self.config.enable_hashtag:
            detectors.append(HashtagDetector(self.config))

        # NER last
        if self.config.enable_ner:
            detectors.append(NERDetector(self.config))

        return detectors

    def _parse_range_json(self, json_str: Optional[str]) -> List[TextSpan]:
        """Parse code fence or blockquote ranges from JSON."""
        if not json_str:
            return []

        try:
            ranges = json.loads(json_str)
            return [
                TextSpan(r["char_start"], r["char_end"])
                for r in ranges
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _build_exclusion_ranges(self, ctx: MessageContext) -> List[TextSpan]:
        """Build combined exclusion ranges for a message."""
        excluded = list(ctx.code_fence_ranges)

        if self.config.ignore_markdown_blockquotes:
            excluded.extend(ctx.blockquote_ranges)

        # Sort and merge overlapping ranges
        if not excluded:
            return []

        excluded.sort(key=lambda s: (s.char_start, s.char_end))
        merged = [excluded[0]]

        for span in excluded[1:]:
            if span.char_start <= merged[-1].char_end:
                # Overlapping or adjacent - merge
                merged[-1] = TextSpan(
                    merged[-1].char_start,
                    max(merged[-1].char_end, span.char_end)
                )
            else:
                merged.append(span)

        return merged

    def _span_intersects_excluded(
        self, char_start: int, char_end: int, excluded: List[TextSpan]
    ) -> Optional[str]:
        """Check if span intersects any excluded range. Returns reason or None."""
        span = TextSpan(char_start, char_end)

        for exc in excluded:
            if span.overlaps(exc):
                # Determine reason based on range type
                # (We don't track type separately, but code fences are primary)
                return SuppressionReason.INTERSECTS_CODE_FENCE

        return None

    def _is_code_like_token(self, surface: str) -> bool:
        """Check if surface text looks like code."""
        if not surface:
            return False

        normalized = surface.lower().strip()
        return normalized in self.config.code_like_denylist

    def _verify_offset(
        self, text_raw: str, char_start: Optional[int], char_end: Optional[int], surface: Optional[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify that offsets correctly extract the surface text.

        Returns (is_valid, actual_substring).
        """
        if char_start is None or char_end is None:
            return False, None

        if char_start < 0 or char_end > len(text_raw) or char_start >= char_end:
            return False, None

        actual = text_raw[char_start:char_end]

        if surface is None:
            return True, actual

        return actual == surface, actual

    def run(self) -> Dict[str, int]:
        """Execute Stage 2A pipeline. Returns statistics."""
        logger.info("Starting Stage 2A: Entity & Time Detection Layer...")

        # Check prerequisites
        self.db.check_required_tables()

        # Initialize schema
        self.db.initialize_stage2A_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Phase 1: Initialize
            logger.info("Phase 1: Initializing...")
            ner_run_id = None
            if self.config.enable_ner:
                ner_run_id = self._initialize_ner_run()

            # Get message count for progress
            total_messages = self.db.get_message_count()
            logger.info(f"Processing {total_messages} messages...")

            # Process messages
            for i, row in enumerate(self.db.get_messages_for_processing()):
                if (i + 1) % 100 == 0:
                    logger.info(f"  Processed {i + 1}/{total_messages} messages...")

                ctx = self._build_message_context(row)
                self._process_message(ctx)
                self.stats["messages_processed"] += 1

            # Finalize NER run
            if ner_run_id:
                self._finalize_ner_run(ner_run_id)

            # Get final stats from DB
            db_stats = self.db.get_candidate_stats()
            self.stats.update(db_stats)

            # Commit transaction
            self.db.commit()
            logger.info("Stage 2A completed successfully")

        except Exception as e:
            logger.error(f"Stage 2A failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return self.stats

    def _initialize_ner_run(self) -> str:
        """Initialize NER model run record."""
        run_id = self.id_generator.generate([
            "ner_run",
            self.stage_started_at_utc
        ])

        config_json = JCS.canonicalize({
            "model_name": self.config.ner_model_name,
            "max_chars": self.config.ner_max_chars,
            "stride": self.config.ner_stride,
            "label_allowlist": sorted(self.config.ner_label_allowlist)
        })

        self.db.insert_ner_model_run(
            run_id=run_id,
            model_name=self.config.ner_model_name,
            model_version="pending",  # Updated on completion
            config_json=config_json,
            started_at_utc=self.stage_started_at_utc,
            completed_at_utc=None,
            raw_io_json=None
        )

        return run_id

    def _finalize_ner_run(self, run_id: str):
        """Finalize NER model run with completion data."""
        completed_at = TimestampUtils.now_utc()

        # Get NER detector for version info
        ner_detector = next(
            (d for d in self.detectors if isinstance(d, NERDetector)),
            None
        )

        raw_io = {}
        if ner_detector:
            raw_io = ner_detector.get_run_info()

        self.db.update_ner_model_run(
            run_id=run_id,
            completed_at_utc=completed_at,
            raw_io_json=JCS.canonicalize(raw_io)
        )

    def _build_message_context(self, row: sqlite3.Row) -> MessageContext:
        """Build message context from database row."""
        code_fences = self._parse_range_json(row["code_fence_ranges_json"])
        blockquotes = self._parse_range_json(row["blockquote_ranges_json"])

        ctx = MessageContext(
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            text_raw=row["text_raw"],
            created_at_utc=row["created_at_utc"],
            timestamp_quality=row["timestamp_quality"],
            role=row["role"],
            order_index=row["order_index"],
            code_fence_ranges=code_fences,
            blockquote_ranges=blockquotes,
            excluded_ranges=[]
        )

        ctx.excluded_ranges = self._build_exclusion_ranges(ctx)
        return ctx

    def _process_message(self, ctx: MessageContext):
        """Process a single message: detect entities and times."""
        # Phase 3: Detect entity candidates
        all_candidates: List[Dict[str, Any]] = []

        for detector in self.detectors:
            try:
                candidates = detector.detect(ctx.text_raw)
                for c in candidates:
                    candidate_dict = self._process_candidate(ctx, c, detector)
                    if candidate_dict:
                        all_candidates.append(candidate_dict)
            except Exception as e:
                logger.warning(
                    f"Detector {detector.detector_name} failed on message "
                    f"{ctx.message_id}: {e}"
                )

        # Phase 4: Overlap resolution
        winners, suppressions = OverlapResolver.resolve(all_candidates)

        # Update suppressed candidates
        for candidate, winner_id, reason in suppressions:
            self.db.update_candidate_suppression(
                candidate["candidate_id"],
                winner_id,
                reason
            )

        # Emit winning mentions
        for winner in winners:
            self._emit_mention(winner)

        # Phase 5: Time detection
        self._process_time_mentions(ctx)

    def _process_candidate(
        self, ctx: MessageContext, candidate: EntityCandidate, detector: EntityDetector
    ) -> Optional[Dict[str, Any]]:
        """Process a single candidate and insert into database."""
        char_start = candidate.char_start
        char_end = candidate.char_end
        surface = candidate.surface_text

        # Verify offsets
        is_valid, actual = self._verify_offset(
            ctx.text_raw, char_start, char_end, surface
        )

        suppression_reason = None
        is_eligible = True

        if not is_valid:
            if char_start is not None and char_end is not None:
                # Offsets exist but don't match surface
                suppression_reason = SuppressionReason.OFFSET_MISMATCH
                is_eligible = False
                # Clear invalid offsets
                char_start = None
                char_end = None
                logger.warning(
                    f"OFFSET_UNRELIABLE: {detector.detector_name} in {ctx.message_id}"
                )
            elif not self.config.emit_spanless_ner:
                # No offsets and not allowed to emit spanless
                suppression_reason = SuppressionReason.NO_OFFSETS_UNRELIABLE
                is_eligible = False

        # Check exclusion ranges (only if we have valid offsets)
        if is_eligible and char_start is not None and char_end is not None:
            exclusion_reason = self._span_intersects_excluded(
                char_start, char_end, ctx.excluded_ranges
            )
            if exclusion_reason:
                suppression_reason = exclusion_reason
                is_eligible = False

        # Check code-like tokens
        if is_eligible and surface and self._is_code_like_token(surface):
            suppression_reason = SuppressionReason.CODE_LIKE_TOKEN
            is_eligible = False

        # Use actual substring as surface if we verified offsets
        if is_valid and actual:
            surface = actual

        # Compute surface hash
        surface_hash = HashUtils.compute_surface_hash(surface)

        # Generate candidate ID
        candidate_id = self.id_generator.generate([
            "candidate",
            ctx.message_id,
            detector.detector_name,
            char_start,
            char_end,
            surface_hash
        ])

        # Build raw candidate JSON
        raw_data = {
            **candidate.raw_data,
            "entity_type_hint": candidate.entity_type_hint,
            "original_surface": candidate.surface_text,
        }
        raw_json = JCS.canonicalize(raw_data)

        # Insert candidate
        self.db.insert_entity_candidate(
            candidate_id=candidate_id,
            message_id=ctx.message_id,
            detector=detector.detector_name,
            detector_version=detector.detector_version,
            entity_type_hint=candidate.entity_type_hint,
            char_start=char_start,
            char_end=char_end,
            surface_text=surface,
            surface_hash=surface_hash,
            confidence=candidate.confidence,
            is_eligible=1 if is_eligible else 0,
            suppressed_by=None,
            suppression_reason=suppression_reason,
            raw_json=raw_json
        )

        self.stats["total_candidates"] += 1
        if is_eligible:
            self.stats["eligible_candidates"] += 1

        # Return dict for overlap resolution
        return {
            "candidate_id": candidate_id,
            "message_id": ctx.message_id,
            "detector": detector.detector_name,
            "detector_version": detector.detector_version,
            "entity_type_hint": candidate.entity_type_hint,
            "char_start": char_start,
            "char_end": char_end,
            "surface_text": surface,
            "surface_hash": surface_hash,
            "confidence": candidate.confidence,
            "is_eligible": is_eligible,
            "raw_data": raw_data
        }

    def _emit_mention(self, winner: Dict[str, Any]):
        """Emit a winning candidate as an entity mention."""
        mention_id = self.id_generator.generate([
            "mention",
            winner["message_id"],
            winner["candidate_id"]
        ])

        raw_json = JCS.canonicalize(winner.get("raw_data", {}))

        self.db.insert_entity_mention(
            mention_id=mention_id,
            message_id=winner["message_id"],
            candidate_id=winner["candidate_id"],
            detector=winner["detector"],
            detector_version=winner["detector_version"],
            entity_type_hint=winner["entity_type_hint"],
            char_start=winner["char_start"],
            char_end=winner["char_end"],
            surface_text=winner["surface_text"],
            surface_hash=winner["surface_hash"],
            confidence=winner["confidence"],
            raw_json=raw_json
        )

        self.stats["emitted_mentions"] += 1

    def _process_time_mentions(self, ctx: MessageContext):
        """Detect and resolve time mentions for a message."""
        # Detect all time candidates
        candidates = self.time_detector.detect(
            ctx.text_raw,
            ctx.created_at_utc,
            ctx.timestamp_quality
        )

        if not candidates:
            return

        # Filter by exclusion ranges
        eligible_candidates = []
        for c in candidates:
            exclusion = self._span_intersects_excluded(
                c.char_start, c.char_end, ctx.excluded_ranges
            )
            if not exclusion:
                eligible_candidates.append(c)

        # Resolve overlaps
        winners, _ = OverlapResolver.resolve_time_candidates(eligible_candidates)

        # Resolve and store each winner
        for candidate in winners:
            resolution = self.time_detector.resolve(
                candidate,
                ctx.created_at_utc,
                ctx.timestamp_quality
            )

            surface_hash = HashUtils.sha256_string(candidate.surface_text)

            time_mention_id = self.id_generator.generate([
                "time",
                ctx.message_id,
                candidate.char_start,
                surface_hash
            ])

            # Build raw parse JSON
            raw_parse = {
                **candidate.parsed_data,
                **resolution
            }

            self.db.insert_time_mention(
                time_mention_id=time_mention_id,
                message_id=ctx.message_id,
                char_start=candidate.char_start,
                char_end=candidate.char_end,
                surface_text=candidate.surface_text,
                surface_hash=surface_hash,
                pattern_id=candidate.pattern_id,
                pattern_precedence=candidate.pattern_precedence,
                anchor_time_utc=ctx.created_at_utc,
                resolved_type=resolution.get("resolved_type", TimeResolvedType.UNRESOLVED),
                valid_from_utc=resolution.get("valid_from_utc"),
                valid_to_utc=resolution.get("valid_to_utc"),
                resolution_granularity=resolution.get("resolution_granularity"),
                timezone_assumed=self.config.anchor_timezone,
                confidence=resolution.get("confidence", 0.0),
                raw_parse_json=JCS.canonicalize(raw_parse)
            )

            self.stats["time_mentions"] += 1
            if resolution.get("resolved_type") != TimeResolvedType.UNRESOLVED:
                self.stats["time_mentions_resolved"] += 1


def run_stage2A(config: Stage2AConfig) -> Dict[str, int]:
    """Run Stage 2A pipeline on existing database."""
    pipeline = Stage2APipeline(config)
    return pipeline.run()


# ============================================================================
# CLI ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 2A: Entity & Time Detection Layer")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: kg.db)"
    )
    parser.add_argument(
        "--id-namespace",
        type=str,
        default="550e8400-e29b-41d4-a716-446655440000",
        help="UUID namespace for ID generation"
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="Default timezone for time resolution (default: UTC)"
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable NER detection"
    )
    parser.add_argument(
        "--ner-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy NER model name (default: en_core_web_sm)"
    )
    parser.add_argument(
        "--ignore-blockquotes",
        action="store_true",
        help="Exclude blockquotes from entity detection"
    )
    parser.add_argument(
        "--enable-handles",
        action="store_true",
        help="Enable @handle detection"
    )
    parser.add_argument(
        "--enable-hashtags",
        action="store_true",
        help="Enable #hashtag detection"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = Stage2AConfig(
        output_file_path=args.db,
        id_namespace=args.id_namespace,
        anchor_timezone=args.timezone,
        enable_ner=not args.no_ner,
        ner_model_name=args.ner_model,
        ignore_markdown_blockquotes=args.ignore_blockquotes,
        enable_handle=args.enable_handles,
        enable_hashtag=args.enable_hashtags,
    )

    stats = run_stage2A(config)

    logger.info("\n" + "=" * 50)
    logger.info("Stage 2A Summary")
    logger.info("=" * 50)
    logger.info(f"  Messages processed:      {stats.get('messages_processed', 0):,}")
    logger.info(f"  Total candidates:        {stats.get('total_candidates', 0):,}")
    logger.info(f"  Eligible candidates:     {stats.get('eligible_candidates', 0):,}")
    logger.info(f"  Emitted mentions:        {stats.get('emitted_mentions', 0):,}")
    logger.info(f"  Time mentions:           {stats.get('time_mentions', 0):,}")
    logger.info(f"  Time mentions resolved:  {stats.get('time_mentions_resolved', 0):,}")

    # Suppression breakdown
    suppression_keys = [k for k in stats.keys() if k.startswith("suppressed_")]
    if suppression_keys:
        logger.info("\nSuppression breakdown:")
        for key in sorted(suppression_keys):
            reason = key.replace("suppressed_", "").upper()
            logger.info(f"    {reason}: {stats[key]:,}")