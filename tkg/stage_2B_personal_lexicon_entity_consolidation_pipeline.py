"""
Stage 2B: Personal Lexicon & Entity Consolidation Layer

Objectives:
1. Induce a personal term lexicon from corpus patterns (project codenames, acronyms, nicknames)
2. Run LexiconMatch detector to emit learned terms with exact offsets
3. Consolidate all emitted mentions into canonical entities

Non-negotiables:
- Lossless + replayable: lexicon induction fully logged; all candidates stored
- Deterministic: stable candidate generators, scoring, and selection
- Role-aware trust: user-weighted counts prioritized over assistant
- Transactional: Stage 2B is one transaction
"""
import hashlib
import json
import logging
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

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

class EntityType(StrEnum):
    EMAIL = "EMAIL"
    URL = "URL"
    DOI = "DOI"
    UUID = "UUID"
    HASH_HEX = "HASH_HEX"
    IP_ADDRESS = "IP_ADDRESS"
    PHONE = "PHONE"
    FILEPATH = "FILEPATH"
    BARE_DOMAIN = "BARE_DOMAIN"
    PERSON = "PERSON"
    ORG = "ORG"
    LOCATION = "LOCATION"
    CUSTOM_TERM = "CUSTOM_TERM"
    OTHER = "OTHER"


class EntityStatus(StrEnum):
    ACTIVE = "active"
    MERGED = "merged"
    DELETED = "deleted"


class LexiconGenerator(StrEnum):
    TITLE_CASE = "TITLE_CASE"
    ALLCAPS = "ALLCAPS"
    CAMEL_CASE = "CAMEL_CASE"
    HASHTAG = "HASHTAG"
    HANDLE = "HANDLE"
    QUOTED = "QUOTED"
    NOUN_CHUNK = "NOUN_CHUNK"


class RejectionReason(StrEnum):
    BELOW_MIN_COUNT = "BELOW_MIN_COUNT"
    BELOW_MIN_CONV = "BELOW_MIN_CONV"
    DENYLIST = "DENYLIST"
    CODE_HEAVY = "CODE_HEAVY"
    LOW_DIVERSITY = "LOW_DIVERSITY"
    CAP_EXCEEDED = "CAP_EXCEEDED"


# ===| CONFIGURATION |===

@dataclass
class Stage2BConfig:
    """Configuration for Stage 2B pipeline."""

    output_file_path: Path
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"
    anchor_timezone: str = "UTC"

    # Lexicon Induction
    enable_lexicon_induction: bool = True
    enable_noun_chunk_induction: bool = False
    lexicon_min_user_mentions: int = 3
    lexicon_min_conversations: int = 2
    lexicon_max_code_ratio: float = 0.5
    lexicon_min_diversity: float = 0.1
    lexicon_max_terms: int = 1000
    lexicon_denylist_path: Optional[Path] = None

    # Lexicon Scoring Weights
    lexicon_weight_mentions: float = 1.0
    lexicon_weight_conversations: float = 2.0
    lexicon_weight_diversity: float = 0.5
    lexicon_penalty_code: float = 1.0

    # Salience Scoring
    salience_weight_mentions: float = 0.3
    salience_weight_conversations: float = 0.4
    salience_weight_user_ratio: float = 0.2
    salience_weight_recency: float = 0.1
    salience_recency_halflife_days: int = 90

    # Role weights for user-weighted counts
    role_weight_user: float = 1.0
    role_weight_assistant: float = 0.5
    role_weight_other: float = 0.3


# ===| DEFAULT DENYLIST |===

DEFAULT_DENYLIST = {
    # Common abbreviations
    "usa", "uk", "eu", "un", "us", "am", "pm", "bc", "ad", "ce", "bce",
    # Days
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    # Months
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Common words that match patterns
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "has", "his", "how", "its", "let",
    "new", "now", "old", "see", "two", "way", "who", "did", "get",
    "come", "made", "find", "here", "just", "know", "take", "want", "also",
    "back", "been", "call", "down", "even", "from", "good", "have", "into",
    "make", "much", "only", "over", "such", "than", "them", "then", "they",
    "very", "what", "when", "your", "more", "some", "time", "will", "with",
    # Tech terms that are too common
    "api", "url", "html", "css", "sql", "xml", "json", "http", "https",
    "fyi", "asap", "etc", "aka", "todo", "tbd", "wip", "eof", "eol",
    # Common proper nouns (too generic)
    "google", "apple", "amazon", "microsoft", "facebook", "twitter",
}


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
        """Execute SQL with parameters."""
        return self.connection.execute(sql, params)

    def executemany(self, sql: str, params_list: List[tuple]) -> sqlite3.Cursor:
        """Execute SQL with multiple parameter sets."""
        return self.connection.executemany(sql, params_list)

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute SQL and fetch one row."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute SQL and fetch all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()


class Stage2BDatabase(Database):
    """Stage 2B specific database operations."""

    def check_required_tables(self):
        """Verify Stage 2A tables exist."""
        required_tables = [
            "conversations", "messages", "message_parts",
            "entity_mention_candidates", "entity_mentions", "time_mentions"
        ]
        for table in required_tables:
            result = self.fetchone(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            )
            if not result:
                raise RuntimeError(f"Required table '{table}' not found. Run Stage 2A first.")

    def initialize_stage2B_schema(self):
        """Create Stage 2B tables if they don't exist."""
        # entities table
        self.execute("""
                     CREATE TABLE IF NOT EXISTS entities (
                                                             entity_id TEXT PRIMARY KEY,
                                                             entity_type TEXT NOT NULL,
                                                             entity_key TEXT NOT NULL,
                                                             canonical_name TEXT,
                                                             aliases_json TEXT,
                                                             status TEXT NOT NULL DEFAULT 'active',
                                                             first_seen_at_utc TEXT,
                                                             last_seen_at_utc TEXT,
                                                             mention_count INTEGER DEFAULT 0,
                                                             conversation_count INTEGER DEFAULT 0,
                                                             salience_score REAL,
                                                             raw_stats_json TEXT
                     )
                     """)

        # Unique constraint on active entities
        self.execute("""
                     CREATE UNIQUE INDEX IF NOT EXISTS entities_active_uniq
                         ON entities(entity_type, entity_key)
                         WHERE status='active'
                     """)

        # lexicon_builds table
        self.execute("""
                     CREATE TABLE IF NOT EXISTS lexicon_builds (
                                                                   build_id TEXT PRIMARY KEY,
                                                                   build_version INTEGER NOT NULL,
                                                                   config_json TEXT NOT NULL,
                                                                   started_at_utc TEXT NOT NULL,
                                                                   completed_at_utc TEXT,
                                                                   candidates_total INTEGER DEFAULT 0,
                                                                   terms_selected INTEGER DEFAULT 0,
                                                                   raw_stats_json TEXT
                     )
                     """)

        # lexicon_term_candidates table
        self.execute("""
                     CREATE TABLE IF NOT EXISTS lexicon_term_candidates (
                                                                            candidate_id TEXT PRIMARY KEY,
                                                                            build_id TEXT NOT NULL REFERENCES lexicon_builds(build_id),
                         generator TEXT NOT NULL,
                         term_key TEXT NOT NULL,
                         canonical_surface TEXT NOT NULL,
                         aliases_json TEXT NOT NULL,
                         total_count INTEGER NOT NULL,
                         user_weighted_count REAL NOT NULL,
                         conversation_count INTEGER NOT NULL,
                         code_likeness_ratio REAL NOT NULL,
                         context_diversity REAL NOT NULL,
                         score REAL NOT NULL,
                         is_selected INTEGER NOT NULL DEFAULT 0,
                         rejection_reason TEXT,
                         evidence_json TEXT NOT NULL
                         )
                     """)

        self.execute("""
                     CREATE INDEX IF NOT EXISTS idx_lex_cand_build
                         ON lexicon_term_candidates(build_id)
                     """)

        self.execute("""
                     CREATE INDEX IF NOT EXISTS idx_lex_cand_term_key
                         ON lexicon_term_candidates(term_key)
                     """)

        # lexicon_terms table
        self.execute("""
                     CREATE TABLE IF NOT EXISTS lexicon_terms (
                                                                  term_id TEXT PRIMARY KEY,
                                                                  build_id TEXT NOT NULL REFERENCES lexicon_builds(build_id),
                         candidate_id TEXT NOT NULL REFERENCES lexicon_term_candidates(candidate_id),
                         term_key TEXT NOT NULL,
                         canonical_surface TEXT NOT NULL,
                         aliases_json TEXT NOT NULL,
                         score REAL NOT NULL,
                         entity_type_hint TEXT NOT NULL DEFAULT 'CUSTOM_TERM'
                         )
                     """)

        self.execute("""
                     CREATE INDEX IF NOT EXISTS idx_lexicon_terms_build
                         ON lexicon_terms(build_id)
                     """)

        self.execute("""
                     CREATE INDEX IF NOT EXISTS idx_lexicon_terms_key
                         ON lexicon_terms(term_key)
                     """)

        logger.info("Stage 2B schema initialized")

    def get_next_build_version(self) -> int:
        """Get next lexicon build version number."""
        result = self.fetchone("SELECT MAX(build_version) FROM lexicon_builds")
        if result and result[0] is not None:
            return result[0] + 1
        return 1

    def get_self_entity_exists(self) -> bool:
        """Check if SELF entity already exists."""
        result = self.fetchone(
            "SELECT entity_id FROM entities WHERE entity_key = '__SELF__' AND entity_type = 'PERSON'"
        )
        return result is not None

    def get_messages_with_text(self) -> List[sqlite3.Row]:
        """Get all messages with text_raw, ordered deterministically."""
        return self.fetchall("""
                             SELECT
                                 m.message_id, m.conversation_id, m.role, m.text_raw,
                                 m.created_at_utc, m.code_fence_ranges_json, m.blockquote_ranges_json,
                                 m.order_index
                             FROM messages m
                             WHERE m.text_raw IS NOT NULL
                             ORDER BY m.conversation_id ASC, m.order_index ASC, m.message_id ASC
                             """)

    def get_emitted_mentions(self) -> List[sqlite3.Row]:
        """Get all emitted entity mentions from Stage 2A."""
        return self.fetchall("""
                             SELECT
                                 em.mention_id, em.message_id, em.candidate_id, em.detector,
                                 em.detector_version, em.entity_type_hint, em.char_start, em.char_end,
                                 em.surface_text, em.surface_hash, em.confidence, em.raw_mention_json,
                                 m.conversation_id, m.role, m.created_at_utc
                             FROM entity_mentions em
                                      JOIN messages m ON em.message_id = m.message_id
                             ORDER BY m.conversation_id ASC, m.order_index ASC, em.char_start ASC
                             """)

    def insert_entity(self, entity: Dict[str, Any]):
        """Insert or update an entity."""
        self.execute("""
                     INSERT INTO entities (
                         entity_id, entity_type, entity_key, canonical_name, aliases_json,
                         status, first_seen_at_utc, last_seen_at_utc, mention_count,
                         conversation_count, salience_score, raw_stats_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                         ON CONFLICT(entity_id) DO UPDATE SET
                         canonical_name = excluded.canonical_name,
                                                       aliases_json = excluded.aliases_json,
                                                       first_seen_at_utc = excluded.first_seen_at_utc,
                                                       last_seen_at_utc = excluded.last_seen_at_utc,
                                                       mention_count = excluded.mention_count,
                                                       conversation_count = excluded.conversation_count,
                                                       salience_score = excluded.salience_score,
                                                       raw_stats_json = excluded.raw_stats_json
                     """, (
                         entity["entity_id"], entity["entity_type"], entity["entity_key"],
                         entity["canonical_name"], entity["aliases_json"], entity["status"],
                         entity["first_seen_at_utc"], entity["last_seen_at_utc"],
                         entity["mention_count"], entity["conversation_count"],
                         entity.get("salience_score"), entity.get("raw_stats_json")
                     ))

    def update_mention_entity_id(self, mention_id: str, entity_id: str):
        """Link a mention to its entity."""
        self.execute(
            "UPDATE entity_mentions SET entity_id = ? WHERE mention_id = ?",
            (entity_id, mention_id)
        )

    def insert_lexicon_build(self, build: Dict[str, Any]):
        """Insert a lexicon build record."""
        self.execute("""
                     INSERT INTO lexicon_builds (
                         build_id, build_version, config_json, started_at_utc,
                         completed_at_utc, candidates_total, terms_selected, raw_stats_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         build["build_id"], build["build_version"], build["config_json"],
                         build["started_at_utc"], build.get("completed_at_utc"),
                         build.get("candidates_total", 0), build.get("terms_selected", 0),
                         build.get("raw_stats_json")
                     ))

    def update_lexicon_build(self, build_id: str, updates: Dict[str, Any]):
        """Update lexicon build with completion data."""
        self.execute("""
                     UPDATE lexicon_builds SET
                                               completed_at_utc = ?,
                                               candidates_total = ?,
                                               terms_selected = ?,
                                               raw_stats_json = ?
                     WHERE build_id = ?
                     """, (
                         updates.get("completed_at_utc"),
                         updates.get("candidates_total", 0),
                         updates.get("terms_selected", 0),
                         updates.get("raw_stats_json"),
                         build_id
                     ))

    def insert_lexicon_candidate(self, candidate: Dict[str, Any]):
        """Insert a lexicon term candidate."""
        self.execute("""
                     INSERT INTO lexicon_term_candidates (
                         candidate_id, build_id, generator, term_key, canonical_surface,
                         aliases_json, total_count, user_weighted_count, conversation_count,
                         code_likeness_ratio, context_diversity, score, is_selected,
                         rejection_reason, evidence_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         candidate["candidate_id"], candidate["build_id"], candidate["generator"],
                         candidate["term_key"], candidate["canonical_surface"], candidate["aliases_json"],
                         candidate["total_count"], candidate["user_weighted_count"],
                         candidate["conversation_count"], candidate["code_likeness_ratio"],
                         candidate["context_diversity"], candidate["score"], candidate["is_selected"],
                         candidate.get("rejection_reason"), candidate["evidence_json"]
                     ))

    def insert_lexicon_term(self, term: Dict[str, Any]):
        """Insert a selected lexicon term."""
        self.execute("""
                     INSERT INTO lexicon_terms (
                         term_id, build_id, candidate_id, term_key, canonical_surface,
                         aliases_json, score, entity_type_hint
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         term["term_id"], term["build_id"], term["candidate_id"],
                         term["term_key"], term["canonical_surface"], term["aliases_json"],
                         term["score"], term["entity_type_hint"]
                     ))

    def get_lexicon_terms(self, build_id: str) -> List[sqlite3.Row]:
        """Get all selected terms for a build."""
        return self.fetchall("""
                             SELECT term_id, term_key, canonical_surface, aliases_json, entity_type_hint
                             FROM lexicon_terms WHERE build_id = ?
                             """, (build_id,))

    def insert_entity_mention_candidate(self, candidate: Dict[str, Any]):
        """Insert an entity mention candidate (for lexicon matches)."""
        self.execute("""
                     INSERT INTO entity_mention_candidates (
                         candidate_id, message_id, detector, detector_version,
                         entity_type_hint, char_start, char_end, surface_text,
                         surface_hash, confidence, is_eligible, suppressed_by_candidate_id,
                         suppression_reason, raw_candidate_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         candidate["candidate_id"], candidate["message_id"],
                         candidate["detector"], candidate["detector_version"],
                         candidate["entity_type_hint"], candidate.get("char_start"),
                         candidate.get("char_end"), candidate.get("surface_text"),
                         candidate["surface_hash"], candidate["confidence"],
                         candidate["is_eligible"], candidate.get("suppressed_by_candidate_id"),
                         candidate.get("suppression_reason"), candidate.get("raw_candidate_json")
                     ))

    def insert_entity_mention(self, mention: Dict[str, Any]):
        """Insert an entity mention."""
        self.execute("""
                     INSERT INTO entity_mentions (
                         mention_id, message_id, entity_id, candidate_id,
                         detector, detector_version, entity_type_hint,
                         char_start, char_end, surface_text, surface_hash,
                         confidence, raw_mention_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         mention["mention_id"], mention["message_id"], mention.get("entity_id"),
                         mention["candidate_id"], mention["detector"], mention["detector_version"],
                         mention["entity_type_hint"], mention.get("char_start"), mention.get("char_end"),
                         mention.get("surface_text"), mention["surface_hash"], mention["confidence"],
                         mention.get("raw_mention_json")
                     ))

    def get_existing_mentions_for_message(self, message_id: str) -> List[sqlite3.Row]:
        """Get existing emitted mentions for overlap checking."""
        return self.fetchall("""
                             SELECT mention_id, char_start, char_end, detector, confidence
                             FROM entity_mentions
                             WHERE message_id = ? AND char_start IS NOT NULL AND char_end IS NOT NULL
                             ORDER BY char_start ASC
                             """, (message_id,))


# ===| LEXICON CANDIDATE GENERATORS |===

@dataclass
class LexiconOccurrence:
    """A single occurrence of a lexicon candidate."""

    message_id: str
    conversation_id: str
    role: str
    char_start: int
    char_end: int
    surface: str
    in_code_fence: bool
    context_before: str
    context_after: str


@dataclass
class LexiconCandidate:
    """Aggregated lexicon candidate."""

    generator: str
    term_key: str
    occurrences: List[LexiconOccurrence] = field(default_factory=list)
    surfaces: Dict[str, int] = field(default_factory=dict)  # surface -> count
    user_surfaces: Dict[str, int] = field(default_factory=dict)  # surface -> user count
    conversation_ids: Set[str] = field(default_factory=set)

    def add_occurrence(self, occ: LexiconOccurrence, role_weight: float):
        """Add an occurrence to this candidate."""
        self.occurrences.append(occ)
        self.surfaces[occ.surface] = self.surfaces.get(occ.surface, 0) + 1
        if occ.role == "user":
            self.user_surfaces[occ.surface] = self.user_surfaces.get(occ.surface, 0) + 1
        self.conversation_ids.add(occ.conversation_id)


class LexiconCandidateGenerator:
    """Base class for lexicon candidate generators."""

    def __init__(self, generator_name: str):
        self.generator_name = generator_name

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        """Generate (start, end, surface) tuples. Override in subclass."""
        raise NotImplementedError

    def is_in_excluded(self, start: int, end: int, excluded_ranges: List[Tuple[int, int]]) -> bool:
        """Check if span overlaps any excluded range."""
        for ex_start, ex_end in excluded_ranges:
            if start < ex_end and end > ex_start:
                return True
        return False


class TitleCaseGenerator(LexiconCandidateGenerator):
    """Detect sequences of 2+ TitleCase words."""

    PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")

    def __init__(self):
        super().__init__(LexiconGenerator.TITLE_CASE)

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        for match in self.PATTERN.finditer(text):
            start, end = match.start(), match.end()
            if not self.is_in_excluded(start, end, excluded_ranges):
                yield start, end, match.group(1)


class AllCapsGenerator(LexiconCandidateGenerator):
    """Detect tokens of 2-6 uppercase letters."""

    PATTERN = re.compile(r"\b([A-Z]{2,6})\b")

    def __init__(self):
        super().__init__(LexiconGenerator.ALLCAPS)

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        for match in self.PATTERN.finditer(text):
            start, end = match.start(), match.end()
            if not self.is_in_excluded(start, end, excluded_ranges):
                yield start, end, match.group(1)


class CamelCaseGenerator(LexiconCandidateGenerator):
    """Detect camelCase or PascalCase tokens."""

    PATTERN = re.compile(r"\b([a-z]+[A-Z][a-zA-Z]*|[A-Z][a-z]+[A-Z][a-zA-Z]*)\b")

    def __init__(self):
        super().__init__(LexiconGenerator.CAMEL_CASE)

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        for match in self.PATTERN.finditer(text):
            start, end = match.start(), match.end()
            if not self.is_in_excluded(start, end, excluded_ranges):
                yield start, end, match.group(1)


class HashtagGenerator(LexiconCandidateGenerator):
    """Detect #tag patterns."""

    PATTERN = re.compile(r"#([a-zA-Z][a-zA-Z0-9_]{2,})\b")

    def __init__(self):
        super().__init__(LexiconGenerator.HASHTAG)

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        for match in self.PATTERN.finditer(text):
            start, end = match.start(), match.end()
            if not self.is_in_excluded(start, end, excluded_ranges):
                # Include the # in the surface
                yield start, end, match.group(0)


class HandleGenerator(LexiconCandidateGenerator):
    """Detect @handle patterns."""

    PATTERN = re.compile(r"@([a-zA-Z][a-zA-Z0-9_]{2,})\b")

    def __init__(self):
        super().__init__(LexiconGenerator.HANDLE)

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        for match in self.PATTERN.finditer(text):
            start, end = match.start(), match.end()
            if not self.is_in_excluded(start, end, excluded_ranges):
                yield start, end, match.group(0)


class QuotedGenerator(LexiconCandidateGenerator):
    """Detect double-quoted phrases 2-5 words with initial capital."""

    PATTERN = re.compile(r'"([A-Z][^"]{3,50})"')

    def __init__(self):
        super().__init__(LexiconGenerator.QUOTED)

    def generate(self, text: str, excluded_ranges: List[Tuple[int, int]]) -> Iterator[Tuple[int, int, str]]:
        for match in self.PATTERN.finditer(text):
            content = match.group(1)
            # Check word count (2-5 words)
            words = content.split()
            if 2 <= len(words) <= 5:
                start, end = match.start(), match.end()
                if not self.is_in_excluded(start, end, excluded_ranges):
                    yield start, end, content


# ===| ENTITY KEY NORMALIZATION |===

class EntityKeyNormalizer:
    """Type-specific entity key normalization."""

    @staticmethod
    def normalize(entity_type: str, surface: str) -> str:
        """Normalize surface text to entity key based on type."""
        if entity_type == EntityType.EMAIL:
            return surface.lower().strip()

        elif entity_type == EntityType.URL:
            # Lowercase, keep scheme and host + path
            url = surface.lower().strip()
            # Remove query string and fragment for basic normalization
            if "?" in url:
                url = url.split("?")[0]
            if "#" in url:
                url = url.split("#")[0]
            return url

        elif entity_type == EntityType.DOI:
            key = surface.lower().strip()
            if key.startswith("doi:"):
                key = key[4:]
            return key

        elif entity_type == EntityType.UUID:
            # Lowercase, ensure hyphenated format
            return surface.lower().strip()

        elif entity_type == EntityType.HASH_HEX:
            return surface.lower().strip()

        elif entity_type == EntityType.IP_ADDRESS:
            # Could add canonical form normalization here
            return surface.strip()

        elif entity_type == EntityType.PHONE:
            # Basic normalization - could add E.164 parsing
            return re.sub(r"[\s\-\(\)]", "", surface)

        elif entity_type == EntityType.FILEPATH:
            return surface.strip()

        elif entity_type == EntityType.BARE_DOMAIN:
            return surface.lower().strip()

        elif entity_type in (EntityType.PERSON, EntityType.ORG, EntityType.LOCATION, EntityType.CUSTOM_TERM):
            # Lowercase, whitespace-collapsed
            return " ".join(surface.lower().split())

        else:
            return surface.lower().strip()


# ===| AHO-CORASICK MATCHER |===

class AhoCorasickMatcher:
    """Simple Aho-Corasick-like matcher for lexicon terms."""

    def __init__(self):
        self.patterns: Dict[str, List[Tuple[str, str]]] = {}  # surface -> [(term_key, entity_type_hint)]

    def add_pattern(self, surface: str, term_key: str, entity_type_hint: str):
        key = surface.lower()
        if key not in self.patterns:
            self.patterns[key] = []
        entry = (term_key, entity_type_hint)
        if entry not in self.patterns[key]:  # Prevent duplicates
            self.patterns[key].append(entry)

    def find_all(self, text: str) -> Iterator[Tuple[int, int, str, str, str]]:
        """Find all matches in text. Yields (start, end, surface, term_key, entity_type_hint)."""
        text_lower = text.lower()

        # Sort patterns by length descending for longest match first
        sorted_patterns = sorted(self.patterns.keys(), key=len, reverse=True)

        matched_ranges: List[Tuple[int, int]] = []

        for pattern_lower in sorted_patterns:
            pattern_len = len(pattern_lower)
            start = 0
            while True:
                pos = text_lower.find(pattern_lower, start)
                if pos == -1:
                    break

                end = pos + pattern_len

                # Check word boundaries
                if pos > 0 and text_lower[pos-1].isalnum():
                    start = pos + 1
                    continue
                if end < len(text_lower) and text_lower[end].isalnum():
                    start = pos + 1
                    continue

                # Check if this overlaps with already matched ranges
                overlaps = False
                for m_start, m_end in matched_ranges:
                    if pos < m_end and end > m_start:
                        overlaps = True
                        break

                if not overlaps:
                    matched_ranges.append((pos, end))
                    surface = text[pos:end]
                    for term_key, entity_type_hint in self.patterns[pattern_lower]:
                        yield pos, end, surface, term_key, entity_type_hint

                start = pos + 1


# ===| MAIN PIPELINE |===

class Stage2BPipeline:
    """
    Stage 2B: Personal Lexicon & Entity Consolidation Layer

    Phases:
    1. Initialize & seed
    2. Lexicon induction (candidate generation)
    3. Lexicon selection (threshold filtering)
    4. LexiconMatch detection
    5. Entity upsert + canonicalization
    6. Salience scoring
    7. Commit + stats
    """

    def __init__(self, config: Stage2BConfig):
        self.config = config
        self.db = Stage2BDatabase(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.stage_started_at_utc = TimestampUtils.now_utc()
        self.denylist = self._load_denylist()
        self.build_id: Optional[str] = None
        self.stats: Dict[str, int] = {}

    def _load_denylist(self) -> Set[str]:
        """Load denylist from file or use default."""
        denylist = DEFAULT_DENYLIST.copy()
        if self.config.lexicon_denylist_path and self.config.lexicon_denylist_path.exists():
            with open(self.config.lexicon_denylist_path, "r") as f:
                for line in f:
                    term = line.strip().lower()
                    if term and not term.startswith("#"):
                        denylist.add(term)
            logger.info(f"Loaded {len(denylist)} denylist terms")
        return denylist

    def run(self) -> Dict[str, int]:
        """Execute Stage 2B pipeline. Returns statistics."""
        logger.info("Starting Stage 2B...")

        # Check prerequisites
        self.db.check_required_tables()

        # Initialize schema
        self.db.initialize_stage2B_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Phase 1: Initialize & seed
            self._phase1_initialize_and_seed()

            # Phase 2: Lexicon induction
            candidates = {}
            if self.config.enable_lexicon_induction:
                candidates = self._phase2_lexicon_induction()

            # Phase 3: Lexicon selection
            selected_terms = []
            if self.config.enable_lexicon_induction:
                selected_terms = self._phase3_lexicon_selection(candidates)

            # Phase 4: LexiconMatch detection
            if selected_terms:
                self._phase4_lexicon_match_detection(selected_terms)

            # Phase 5: Entity upsert + canonicalization
            self._phase5_entity_consolidation()

            # Phase 6: Salience scoring
            self._phase6_salience_scoring()

            # Phase 7: Commit + stats
            self._phase7_commit_and_stats()

            logger.info("Stage 2B completed successfully")

        except Exception as e:
            logger.error(f"Stage 2B failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

        return self.stats

    def _phase1_initialize_and_seed(self):
        """Phase 1: Initialize & seed SELF entity."""
        logger.info("Phase 1: Initialize & seed")

        # Seed SELF entity if not exists
        if not self.db.get_self_entity_exists():
            self_entity_id = self.id_generator.generate(["entity", "PERSON", "__SELF__"])
            self.db.insert_entity({
                "entity_id": self_entity_id,
                "entity_type": EntityType.PERSON,
                "entity_key": "__SELF__",
                "canonical_name": "SELF",
                "aliases_json": JCS.canonicalize([]),
                "status": EntityStatus.ACTIVE,
                "first_seen_at_utc": None,
                "last_seen_at_utc": None,
                "mention_count": 0,
                "conversation_count": 0,
                "salience_score": None,
                "raw_stats_json": None
            })
            logger.info("Seeded SELF entity")

        # Generate build_id for lexicon induction
        self.build_id = self.id_generator.generate(["lexicon_build", self.stage_started_at_utc])
        build_version = self.db.get_next_build_version()

        # Insert lexicon build record
        self.db.insert_lexicon_build({
            "build_id": self.build_id,
            "build_version": build_version,
            "config_json": JCS.canonicalize({
                "enable_lexicon_induction": self.config.enable_lexicon_induction,
                "lexicon_min_user_mentions": self.config.lexicon_min_user_mentions,
                "lexicon_min_conversations": self.config.lexicon_min_conversations,
                "lexicon_max_code_ratio": self.config.lexicon_max_code_ratio,
                "lexicon_min_diversity": self.config.lexicon_min_diversity,
                "lexicon_max_terms": self.config.lexicon_max_terms
            }),
            "started_at_utc": self.stage_started_at_utc
        })

        logger.info(f"Created lexicon build {self.build_id} (version {build_version})")

    def _parse_ranges_json(self, ranges_json: Optional[str]) -> List[Tuple[int, int]]:
        """Parse code fence or blockquote ranges JSON."""
        if not ranges_json:
            return []
        try:
            ranges = json.loads(ranges_json)
            return [(r["char_start"], r["char_end"]) for r in ranges]
        except (json.JSONDecodeError, KeyError):
            return []

    def _get_context(self, text: str, start: int, end: int, window: int = 30) -> Tuple[str, str]:
        """Extract context before and after a span."""
        context_before = text[max(0, start - window):start]
        context_after = text[end:min(len(text), end + window)]
        return context_before, context_after

    def _get_role_weight(self, role: str) -> float:
        """Get weight for a role."""
        if role == "user":
            return self.config.role_weight_user
        elif role == "assistant":
            return self.config.role_weight_assistant
        return self.config.role_weight_other

    def _phase2_lexicon_induction(self) -> Dict[str, LexiconCandidate]:
        """Phase 2: Lexicon induction - candidate generation."""
        logger.info("Phase 2: Lexicon induction (candidate generation)")

        # Initialize generators
        generators = [
            TitleCaseGenerator(),
            AllCapsGenerator(),
            CamelCaseGenerator(),
            HashtagGenerator(),
            HandleGenerator(),
            QuotedGenerator(),
        ]

        # Collect all candidates by term_key
        candidates: Dict[str, LexiconCandidate] = {}

        messages = self.db.get_messages_with_text()
        logger.info(f"Processing {len(messages)} messages for lexicon candidates")

        for msg in messages:
            text = msg["text_raw"]
            message_id = msg["message_id"]
            conversation_id = msg["conversation_id"]
            role = msg["role"] or "unknown"

            # Build excluded ranges (code fences + blockquotes if configured)
            excluded_ranges = self._parse_ranges_json(msg["code_fence_ranges_json"])
            # Note: blockquotes not excluded by default in Stage 2B

            role_weight = self._get_role_weight(role)

            for generator in generators:
                for start, end, surface in generator.generate(text, excluded_ranges):
                    # Check if in code fence
                    in_code = any(
                        start >= cf_start and end <= cf_end
                        for cf_start, cf_end in self._parse_ranges_json(msg["code_fence_ranges_json"])
                    )

                    # Get context for diversity calculation
                    ctx_before, ctx_after = self._get_context(text, start, end)

                    # Normalize term key
                    term_key = " ".join(surface.lower().split())

                    # Create occurrence
                    occ = LexiconOccurrence(
                        message_id=message_id,
                        conversation_id=conversation_id,
                        role=role,
                        char_start=start,
                        char_end=end,
                        surface=surface,
                        in_code_fence=in_code,
                        context_before=ctx_before,
                        context_after=ctx_after
                    )

                    # Add to or create candidate
                    candidate_key = f"{generator.generator_name}:{term_key}"
                    if candidate_key not in candidates:
                        candidates[candidate_key] = LexiconCandidate(
                            generator=generator.generator_name,
                            term_key=term_key
                        )
                    candidates[candidate_key].add_occurrence(occ, role_weight)

        logger.info(f"Generated {len(candidates)} unique lexicon candidates")
        self.stats["lexicon_candidates_generated"] = len(candidates)

        return candidates

    def _compute_candidate_stats(self, candidate: LexiconCandidate) -> Dict[str, Any]:
        """Compute statistics for a lexicon candidate."""
        total_count = len(candidate.occurrences)

        # User-weighted count
        user_weighted_count = sum(
            self._get_role_weight(occ.role) for occ in candidate.occurrences
        )

        # Conversation count
        conversation_count = len(candidate.conversation_ids)

        # Code-likeness ratio
        in_code_count = sum(1 for occ in candidate.occurrences if occ.in_code_fence)
        code_likeness_ratio = in_code_count / total_count if total_count > 0 else 0.0

        # Context diversity: distinct bigrams / total occurrences
        contexts = set()
        for occ in candidate.occurrences:
            # Extract simple context tokens
            before_tokens = occ.context_before.split()[-2:] if occ.context_before else []
            after_tokens = occ.context_after.split()[:2] if occ.context_after else []
            context_sig = tuple(before_tokens + after_tokens)
            contexts.add(context_sig)
        context_diversity = len(contexts) / total_count if total_count > 0 else 0.0

        # Canonical surface: most frequent, tie-break by user count, then lexicographic
        surfaces_sorted = sorted(
            candidate.surfaces.keys(),
            key=lambda s: (
                -candidate.surfaces[s],
                -candidate.user_surfaces.get(s, 0),
                s
            )
        )
        canonical_surface = surfaces_sorted[0] if surfaces_sorted else candidate.term_key

        # All aliases sorted
        aliases = sorted(candidate.surfaces.keys())

        return {
            "total_count": total_count,
            "user_weighted_count": user_weighted_count,
            "conversation_count": conversation_count,
            "code_likeness_ratio": code_likeness_ratio,
            "context_diversity": context_diversity,
            "canonical_surface": canonical_surface,
            "aliases": aliases
        }

    def _phase3_lexicon_selection(self, candidates: Dict[str, LexiconCandidate]) -> List[Dict[str, Any]]:
        """Phase 3: Lexicon selection - threshold filtering."""
        logger.info("Phase 3: Lexicon selection (threshold filtering)")

        scored_candidates = []
        generator_stats: Dict[str, Dict[str, int]] = {}
        rejection_stats: Dict[str, int] = {}

        for candidate_key, candidate in candidates.items():
            stats = self._compute_candidate_stats(candidate)

            # Generate candidate_id
            candidate_id = self.id_generator.generate([
                "lex_cand", self.build_id, candidate.generator, candidate.term_key
            ])

            # Initialize candidate record
            candidate_record = {
                "candidate_id": candidate_id,
                "build_id": self.build_id,
                "generator": candidate.generator,
                "term_key": candidate.term_key,
                "canonical_surface": stats["canonical_surface"],
                "aliases_json": JCS.canonicalize(stats["aliases"]),
                "total_count": stats["total_count"],
                "user_weighted_count": stats["user_weighted_count"],
                "conversation_count": stats["conversation_count"],
                "code_likeness_ratio": stats["code_likeness_ratio"],
                "context_diversity": stats["context_diversity"],
                "score": 0.0,
                "is_selected": 0,
                "rejection_reason": None,
                "evidence_json": JCS.canonicalize({
                    "occurrence_count": stats["total_count"],
                    "sample_surfaces": stats["aliases"][:5]
                })
            }

            # Track generator stats
            if candidate.generator not in generator_stats:
                generator_stats[candidate.generator] = {"total": 0, "selected": 0}
            generator_stats[candidate.generator]["total"] += 1

            # Apply selection criteria in order
            rejection = None

            # 1. Denylist check
            if candidate.term_key in self.denylist:
                rejection = RejectionReason.DENYLIST

            # 2. Minimum count
            elif stats["user_weighted_count"] < self.config.lexicon_min_user_mentions:
                rejection = RejectionReason.BELOW_MIN_COUNT

            # 3. Minimum conversations
            elif stats["conversation_count"] < self.config.lexicon_min_conversations:
                rejection = RejectionReason.BELOW_MIN_CONV

            # 4. Code-likeness threshold
            elif stats["code_likeness_ratio"] > self.config.lexicon_max_code_ratio:
                rejection = RejectionReason.CODE_HEAVY

            # 5. Context diversity threshold
            elif stats["context_diversity"] < self.config.lexicon_min_diversity:
                rejection = RejectionReason.LOW_DIVERSITY

            if rejection:
                candidate_record["rejection_reason"] = rejection
                rejection_stats[rejection] = rejection_stats.get(rejection, 0) + 1
            else:
                # Compute score
                score = (
                        stats["user_weighted_count"] * self.config.lexicon_weight_mentions +
                        stats["conversation_count"] * self.config.lexicon_weight_conversations +
                        stats["context_diversity"] * self.config.lexicon_weight_diversity -
                        stats["code_likeness_ratio"] * self.config.lexicon_penalty_code
                )
                candidate_record["score"] = score
                scored_candidates.append(candidate_record)

            # Insert candidate record
            self.db.insert_lexicon_candidate(candidate_record)

        # Sort by score descending, then term_key ascending for determinism
        scored_candidates.sort(key=lambda c: (-c["score"], c["term_key"]))

        seen_term_keys: Set[str] = set()
        deduplicated_candidates = []
        for candidate_record in scored_candidates:
            if candidate_record["term_key"] not in seen_term_keys:
                seen_term_keys.add(candidate_record["term_key"])
                deduplicated_candidates.append(candidate_record)

        # Apply top-K cap
        selected_terms = []
        for i, candidate_record in enumerate(deduplicated_candidates):
            if i < self.config.lexicon_max_terms:
                candidate_record["is_selected"] = 1

                # Infer entity type hint
                entity_type_hint = EntityType.CUSTOM_TERM
                surface = candidate_record["canonical_surface"]
                if any(surface.endswith(suf) for suf in [" Inc.", " Corp.", " LLC", " Ltd."]):
                    entity_type_hint = EntityType.ORG
                elif surface.startswith("@"):
                    entity_type_hint = EntityType.PERSON

                # Generate term_id
                term_id = self.id_generator.generate([
                    "lex_term", self.build_id, candidate_record["term_key"]
                ])

                term_record = {
                    "term_id": term_id,
                    "build_id": self.build_id,
                    "candidate_id": candidate_record["candidate_id"],
                    "term_key": candidate_record["term_key"],
                    "canonical_surface": candidate_record["canonical_surface"],
                    "aliases_json": candidate_record["aliases_json"],
                    "score": candidate_record["score"],
                    "entity_type_hint": entity_type_hint
                }

                self.db.insert_lexicon_term(term_record)
                selected_terms.append(term_record)

                generator_stats[candidate_record["generator"]]["selected"] = \
                    generator_stats.get(candidate_record["generator"], {}).get("selected", 0) + 1

                # Update candidate in database
                self.db.execute(
                    "UPDATE lexicon_term_candidates SET is_selected = 1 WHERE candidate_id = ?",
                    (candidate_record["candidate_id"],)
                )
            else:
                # Mark as cap exceeded
                self.db.execute(
                    "UPDATE lexicon_term_candidates SET rejection_reason = ? WHERE candidate_id = ?",
                    (RejectionReason.CAP_EXCEEDED, candidate_record["candidate_id"])
                )
                rejection_stats[RejectionReason.CAP_EXCEEDED] = \
                    rejection_stats.get(RejectionReason.CAP_EXCEEDED, 0) + 1

        logger.info(f"Selected {len(selected_terms)} lexicon terms from {len(candidates)} candidates")
        logger.info(f"Rejection stats: {rejection_stats}")

        self.stats["lexicon_terms_selected"] = len(selected_terms)
        self.stats.update({f"rejection_{k}": v for k, v in rejection_stats.items()})

        return selected_terms

    def _phase4_lexicon_match_detection(self, selected_terms: List[Dict[str, Any]]):
        """Phase 4: LexiconMatch detection."""
        logger.info("Phase 4: LexiconMatch detection")

        # Build matcher from selected terms
        matcher = AhoCorasickMatcher()
        for term in selected_terms:
            # Add canonical surface
            matcher.add_pattern(
                term["canonical_surface"],
                term["term_key"],
                term["entity_type_hint"]
            )
            # Add aliases
            try:
                aliases = json.loads(term["aliases_json"])
                for alias in aliases:
                    matcher.add_pattern(alias, term["term_key"], term["entity_type_hint"])
            except json.JSONDecodeError:
                pass

        detector = f"LEXICON:{self.build_id}"
        detector_version = "1.0"

        messages = self.db.get_messages_with_text()
        lexicon_matches = 0
        lexicon_emitted = 0

        for msg in messages:
            text = msg["text_raw"]
            message_id = msg["message_id"]

            # Get existing mentions for overlap checking
            existing_mentions = self.db.get_existing_mentions_for_message(message_id)
            existing_ranges = [
                (m["char_start"], m["char_end"], m["detector"], m["confidence"])
                for m in existing_mentions
            ]

            # Build excluded ranges
            excluded_ranges = self._parse_ranges_json(msg["code_fence_ranges_json"])

            # Find all lexicon matches
            for start, end, surface, term_key, entity_type_hint in matcher.find_all(text):
                lexicon_matches += 1

                # Verify offset
                if text[start:end].lower() != surface.lower():
                    logger.warning(f"Offset mismatch for lexicon term at {start}:{end}")
                    continue

                # Check exclusion
                in_excluded = any(
                    start < ex_end and end > ex_start
                    for ex_start, ex_end in excluded_ranges
                )

                if in_excluded:
                    continue

                # Check overlap with existing mentions
                # Structured detectors (EMAIL, URL, etc.) win over LEXICON
                # LEXICON wins over NER
                overlaps_structured = False
                overlaps_ner = False
                for ex_start, ex_end, ex_detector, ex_conf in existing_ranges:
                    if start < ex_end and end > ex_start:
                        if ex_detector.startswith("NER:"):
                            overlaps_ner = True
                        else:
                            overlaps_structured = True

                if overlaps_structured:
                    continue

                # Generate candidate and mention IDs
                surface_hash = HashUtils.sha256_string(surface) if surface else \
                    HashUtils.sha256_string("**NO_SURFACE**")

                candidate_id = self.id_generator.generate([
                    "lex_mention_cand", message_id, start, surface_hash
                ])

                # Insert candidate
                self.db.insert_entity_mention_candidate({
                    "candidate_id": candidate_id,
                    "message_id": message_id,
                    "detector": detector,
                    "detector_version": detector_version,
                    "entity_type_hint": entity_type_hint,
                    "char_start": start,
                    "char_end": end,
                    "surface_text": surface,
                    "surface_hash": surface_hash,
                    "confidence": 0.8,  # High confidence for lexicon matches
                    "is_eligible": 1,
                    "suppressed_by_candidate_id": None,
                    "suppression_reason": None,
                    "raw_candidate_json": JCS.canonicalize({
                        "term_key": term_key,
                        "build_id": self.build_id
                    })
                })

                # Emit mention
                mention_id = self.id_generator.generate(["mention", message_id, candidate_id])

                self.db.insert_entity_mention({
                    "mention_id": mention_id,
                    "message_id": message_id,
                    "entity_id": None,  # Will be set in Phase 5
                    "candidate_id": candidate_id,
                    "detector": detector,
                    "detector_version": detector_version,
                    "entity_type_hint": entity_type_hint,
                    "char_start": start,
                    "char_end": end,
                    "surface_text": surface,
                    "surface_hash": surface_hash,
                    "confidence": 0.8,
                    "raw_mention_json": JCS.canonicalize({
                        "term_key": term_key,
                        "build_id": self.build_id
                    })
                })

                lexicon_emitted += 1

        logger.info(f"LexiconMatch: {lexicon_matches} matches, {lexicon_emitted} emitted")
        self.stats["lexicon_matches_found"] = lexicon_matches
        self.stats["lexicon_mentions_emitted"] = lexicon_emitted

    def _phase5_entity_consolidation(self):
        """Phase 5: Entity upsert + canonicalization."""
        logger.info("Phase 5: Entity upsert + canonicalization")

        # Get all emitted mentions
        mentions = self.db.get_emitted_mentions()
        logger.info(f"Consolidating {len(mentions)} mentions into entities")

        # Group mentions by (entity_type, entity_key)
        entity_groups: Dict[Tuple[str, str], List[sqlite3.Row]] = {}

        for mention in mentions:
            entity_type = mention["entity_type_hint"] or EntityType.OTHER
            surface = mention["surface_text"]

            if not surface:
                continue

            entity_key = EntityKeyNormalizer.normalize(entity_type, surface)
            key = (entity_type, entity_key)

            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(mention)

        entities_created = 0
        entities_updated = 0
        mentions_linked = 0

        for (entity_type, entity_key), group_mentions in entity_groups.items():
            # Generate entity_id
            entity_id = self.id_generator.generate(["entity", entity_type, entity_key])

            # Collect statistics
            surfaces: Dict[str, int] = {}
            user_surfaces: Dict[str, int] = {}
            conversation_ids: Set[str] = set()
            timestamps: List[str] = []

            for m in group_mentions:
                surface = m["surface_text"]
                if surface:
                    surfaces[surface] = surfaces.get(surface, 0) + 1
                    if m["role"] == "user":
                        user_surfaces[surface] = user_surfaces.get(surface, 0) + 1

                conversation_ids.add(m["conversation_id"])

                if m["created_at_utc"]:
                    timestamps.append(m["created_at_utc"])

            # Select canonical name (most frequent, prefer user mentions)
            if surfaces:
                canonical_name = max(
                    surfaces.keys(),
                    key=lambda s: (surfaces[s], user_surfaces.get(s, 0), s)
                )
            else:
                canonical_name = entity_key

            # Compute temporal bounds
            first_seen = min(timestamps) if timestamps else None
            last_seen = max(timestamps) if timestamps else None

            # Build aliases
            aliases = sorted(set(surfaces.keys()))

            # Check if entity exists
            existing = self.db.fetchone(
                "SELECT entity_id FROM entities WHERE entity_id = ?",
                (entity_id,)
            )

            entity_record = {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "entity_key": entity_key,
                "canonical_name": canonical_name,
                "aliases_json": JCS.canonicalize(aliases),
                "status": EntityStatus.ACTIVE,
                "first_seen_at_utc": first_seen,
                "last_seen_at_utc": last_seen,
                "mention_count": len(group_mentions),
                "conversation_count": len(conversation_ids),
                "salience_score": None,  # Computed in Phase 6
                "raw_stats_json": JCS.canonicalize({
                    "surface_counts": surfaces,
                    "user_surface_counts": user_surfaces
                })
            }

            self.db.insert_entity(entity_record)

            if existing:
                entities_updated += 1
            else:
                entities_created += 1

            # Link mentions to entity
            for m in group_mentions:
                self.db.update_mention_entity_id(m["mention_id"], entity_id)
                mentions_linked += 1

        logger.info(f"Entities: {entities_created} created, {entities_updated} updated")
        logger.info(f"Mentions linked: {mentions_linked}")

        self.stats["entities_created"] = entities_created
        self.stats["entities_updated"] = entities_updated
        self.stats["mentions_linked"] = mentions_linked

    def _phase6_salience_scoring(self):
        """Phase 6: Salience scoring."""
        logger.info("Phase 6: Salience scoring")

        # Get all active entities
        entities = self.db.fetchall(
            "SELECT entity_id, mention_count, conversation_count, last_seen_at_utc FROM entities WHERE status = 'active'"
        )

        if not entities:
            logger.info("No entities to score")
            return

        # Compute salience for each entity
        now = pendulum.parse(self.stage_started_at_utc)
        halflife_days = self.config.salience_recency_halflife_days

        for entity in entities:
            mention_count = entity["mention_count"] or 0
            conversation_count = entity["conversation_count"] or 0

            # Get user mention ratio
            entity_id = entity["entity_id"]
            user_mentions = self.db.fetchone("""
                                             SELECT COUNT(*) as cnt FROM entity_mentions em
                                                                             JOIN messages m ON em.message_id = m.message_id
                                             WHERE em.entity_id = ? AND m.role = 'user'
                                             """, (entity_id,))
            user_count = user_mentions["cnt"] if user_mentions else 0
            user_ratio = user_count / mention_count if mention_count > 0 else 0.0

            # Compute recency factor (exponential decay)
            last_seen = entity["last_seen_at_utc"]
            if last_seen:
                try:
                    last_dt = pendulum.parse(last_seen)
                    days_ago = (now - last_dt).days
                    recency_factor = math.exp(-math.log(2) * days_ago / halflife_days)
                except Exception:
                    recency_factor = 0.5
            else:
                recency_factor = 0.5

            # Compute salience score
            salience = (
                    mention_count * self.config.salience_weight_mentions +
                    conversation_count * self.config.salience_weight_conversations +
                    user_ratio * self.config.salience_weight_user_ratio +
                    recency_factor * self.config.salience_weight_recency
            )

            # Update entity
            self.db.execute(
                "UPDATE entities SET salience_score = ? WHERE entity_id = ?",
                (salience, entity_id)
            )

        logger.info(f"Computed salience for {len(entities)} entities")
        self.stats["entities_scored"] = len(entities)

    def _phase7_commit_and_stats(self):
        """Phase 7: Commit transaction and finalize stats."""
        logger.info("Phase 7: Commit + stats")

        completed_at = TimestampUtils.now_utc()

        # Update lexicon build with completion data
        self.db.update_lexicon_build(self.build_id, {
            "completed_at_utc": completed_at,
            "candidates_total": self.stats.get("lexicon_candidates_generated", 0),
            "terms_selected": self.stats.get("lexicon_terms_selected", 0),
            "raw_stats_json": JCS.canonicalize(self.stats)
        })

        # Commit transaction
        self.db.commit()

        logger.info("Transaction committed")


def run_stage2B(config: Stage2BConfig) -> Dict[str, int]:
    """Run Stage 2B pipeline on existing database."""
    pipeline = Stage2BPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 2B: Personal Lexicon & Entity Consolidation")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: ../data/output/kg.db)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="550e8400-e29b-41d4-a716-446655440000",
        help="UUID namespace for ID generation"
    )
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=3,
        help="Minimum user-weighted mentions for lexicon term selection (default: 3)"
    )
    parser.add_argument(
        "--min-conversations",
        type=int,
        default=2,
        help="Minimum conversation count for lexicon term selection (default: 2)"
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=1000,
        help="Maximum lexicon terms to select (default: 1000)"
    )
    parser.add_argument(
        "--denylist",
        type=Path,
        default=None,
        help="Path to additional denylist file"
    )
    parser.add_argument(
        "--disable-lexicon",
        action="store_true",
        help="Disable lexicon induction (only run entity consolidation)"
    )

    args = parser.parse_args()

    config = Stage2BConfig(
        output_file_path=args.db,
        id_namespace=args.namespace,
        enable_lexicon_induction=not args.disable_lexicon,
        lexicon_min_user_mentions=args.min_mentions,
        lexicon_min_conversations=args.min_conversations,
        lexicon_max_terms=args.max_terms,
        lexicon_denylist_path=args.denylist
    )

    stats = run_stage2B(config)

    logger.info("\n=== Stage 2B Summary ===")
    for key, value in sorted(stats.items()):
        logger.info(f"  {key}: {value}")
