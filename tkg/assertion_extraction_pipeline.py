"""
Stage 3: Assertion Extraction Layer

Extracts assertions from messages using rule-based patterns and optional LLM methods,
grounds them to entities and predicates, handles retractions and corrections,
with full auditability, determinism, and role-aware trust scoring.

Assumes that the database "kg.db" exists with Stage 1 and Stage 2 tables populated.

Key Outputs:
- assertions: semantic claims extracted from messages
- predicates: normalized predicate labels
- retractions: explicit user retractions/corrections
- Full audit trail via llm_extraction_* and entity_canonicalization_* tables
"""

import json
import logging
import re
import sqlite3
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterator

from tkg.database_base import Database
# ============================================================================
# LOGGING SETUP
# ============================================================================

# Import shared utilities from extraction pipeline
from tkg.extraction_pipeline import (
    PipelineConfig,
    JCS,
    IDGenerator,
    TimestampUtils,
    DatabaseStage1 as BaseDatabase,
)
from tkg.hash_utils import HashUtils

# ===| LOGGING |===

from tkg.logger import get_logger
logger = get_logger(__name__)



# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Stage3Config:
    """Configuration for Stage 3 pipeline."""

    # Database
    output_file_path: Path = Path("kg.db")
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"

    # Timezone
    anchor_timezone: str = "UTC"

    # Extraction settings
    enable_llm_assertion_extraction: bool = False
    llm_model_name: str = "claude-3-sonnet"
    llm_model_version: str = "20240229"
    llm_temperature: float = 0.0
    llm_top_p: float = 1.0
    llm_seed: int | None = 42
    max_retries_llm: int = 3
    llm_multi_run_count: int = 1

    # Context settings
    k_context: int = 5
    code_fence_masking: str = "whitespace_preserve_length"

    # Linking thresholds
    threshold_link_string_sim: float = 0.85

    # Upsert policy
    assertion_upsert_policy: str = "keep_highest_confidence"  # keep_first, keep_all

    # Trust weights
    trust_weight_user: float = 1.0
    trust_weight_assistant_corroborated: float = 0.8
    trust_weight_assistant_uncorroborated: float = 0.4

    # Corroboration
    coref_window_size: int = 5

    # Minimum text length after exclusions
    min_text_length: int = 10


# ============================================================================
# ENUMS
# ============================================================================

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


class ExtractionMethod(StrEnum):
    """Extraction method types."""

    RULE_BASED = "rule_based"
    LLM = "llm"
    HYBRID = "hybrid"


class ObjectValueType(StrEnum):
    """Literal object value types."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    JSON = "json"


class TemporalQualifierType(StrEnum):
    """Temporal qualifier types."""

    AT = "at"
    SINCE = "since"
    UNTIL = "until"
    DURING = "during"


class RetractionType(StrEnum):
    """Retraction types."""

    FULL = "full"
    CORRECTION = "correction"
    TEMPORAL_BOUND = "temporal_bound"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AssertionCandidate:
    """
    Raw assertion candidate from extractors.

    This is the output of Pass B before grounding.
    """

    message_id: str
    subject_ref: str  # "SELF" or entity canonical name/key/id
    predicate_label: str

    # Object (exactly one type or none for unary)
    object_entity_ref: str | None = None  # entity canonical name/key/id
    object_literal_type: str | None = None  # ObjectValueType
    object_literal_value: Any = None  # raw value

    # Semantics
    modality: str = Modality.STATE
    polarity: str = Polarity.POSITIVE

    # Evidence
    char_start: int | None = None
    char_end: int | None = None
    quote: str | None = None  # alternative to offsets

    # Temporal
    temporal_qualifier_type: str | None = None
    temporal_qualifier_surface: str | None = None  # to match time_mentions
    temporal_qualifier_value: str | None = None  # explicit ISO datetime

    # Provenance
    extraction_method: str = ExtractionMethod.RULE_BASED
    extraction_model: str | None = None
    confidence: float = 1.0
    pattern_id: str | None = None

    # Raw data for audit
    raw_data: dict = field(default_factory=dict)


@dataclass
class GroundedAssertion:
    """
    Fully grounded assertion ready for persistence.

    This is the output of Pass C.
    """

    message_id: str
    assertion_key: str
    fact_key: str

    subject_entity_id: str
    predicate_id: str

    object_entity_id: str | None = None
    object_value_type: str | None = None
    object_value: str | None = None  # JCS-canonical JSON
    object_signature: str = "N:__NONE__"

    temporal_qualifier_type: str | None = None
    temporal_qualifier_id: str | None = None

    modality: str = Modality.STATE
    polarity: str = Polarity.POSITIVE
    asserted_role: str = "user"
    asserted_at_utc: str | None = None

    confidence_extraction: float = 1.0
    confidence_final: float = 1.0
    has_user_corroboration: int = 0

    char_start: int | None = None
    char_end: int | None = None
    surface_text: str | None = None

    extraction_method: str = ExtractionMethod.RULE_BASED
    extraction_model: str | None = None

    raw_assertion_json: str = "{}"

    # Corroboration details (stored in raw_assertion_json)
    corroboration_details: dict = field(default_factory=dict)


@dataclass
class RetractionCandidate:
    """Retraction/correction detected from user messages."""

    message_id: str
    retraction_type: str

    # Target specification
    target_subject_ref: str | None = None
    target_predicate_label: str | None = None
    target_object_ref: str | None = None

    # Replacement (for corrections)
    replacement_candidate: AssertionCandidate | None = None

    # Evidence
    char_start: int | None = None
    char_end: int | None = None
    surface_text: str | None = None
    confidence: float = 1.0

    # Raw data
    raw_data: dict = field(default_factory=dict)


@dataclass
class MessageContext:
    """Context package for extraction."""

    target_message: dict
    context_messages: list[dict]
    context_entities: list[dict]
    context_times: list[dict]
    masked_text: str
    excluded_ranges: list[tuple[int, int]]


# ============================================================================
# STRING SIMILARITY
# ============================================================================

class StringSimilarity:
    """String similarity utilities for fuzzy matching."""

    @staticmethod
    def normalize_for_comparison(s: str) -> str:
        """Normalize string for comparison."""
        # NFKC normalization
        s = unicodedata.normalize("NFKC", s)
        # Lowercase
        s = s.lower()
        # Collapse whitespace
        s = " ".join(s.split())
        return s.strip()

    @staticmethod
    def levenshtein_ratio(s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein similarity ratio.

        Returns value in [0, 1] where 1 means identical.
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        # Normalize
        s1 = StringSimilarity.normalize_for_comparison(s1)
        s2 = StringSimilarity.normalize_for_comparison(s2)

        if s1 == s2:
            return 1.0

        # Levenshtein distance
        m, n = len(s1), len(s2)
        if m > n:
            s1, s2, m, n = s2, s1, n, m

        # Use two rows for space efficiency
        prev = list(range(n + 1))
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    curr[j] = prev[j-1]
                else:
                    curr[j] = 1 + min(prev[j], curr[j-1], prev[j-1])
            prev, curr = curr, prev

        distance = prev[n]
        max_len = max(m, n)
        return 1.0 - (distance / max_len) if max_len > 0 else 1.0


# ============================================================================
# PREDICATE NORMALIZATION
# ============================================================================

class PredicateNormalizer:
    """Normalize predicate labels for canonical storage."""

    @staticmethod
    def normalize(label: str) -> tuple[str, str]:
        """
        Normalize predicate label.

        Returns (canonical_label, canonical_label_norm).
        """
        # Trim and NFKC normalize
        label = unicodedata.normalize("NFKC", label.strip())
        # Collapse whitespace
        canonical = " ".join(label.split())
        # Lowercase for dedup matching
        canonical_norm = canonical.lower()
        return canonical, canonical_norm


# ============================================================================
# DATABASE EXTENSION
# ============================================================================

class Stage3Database(Database):
    """Database operations for Stage 3."""

    STAGE3_TABLES = {
        "predicates": [
            "predicate_id",
            "canonical_label",
            "canonical_label_norm",
            "inverse_label",
            "category",
            "arity",
            "value_type_constraint",
            "first_seen_at_utc",
            "assertion_count",
            "raw_predicate_json",
        ],
        "assertions": [
            "assertion_id",
            "message_id",
            "assertion_key",
            "fact_key",
            "subject_entity_id",
            "predicate_id",
            "object_entity_id",
            "object_value_type",
            "object_value",
            "object_signature",
            "temporal_qualifier_type",
            "temporal_qualifier_id",
            "modality",
            "polarity",
            "asserted_role",
            "asserted_at_utc",
            "confidence_extraction",
            "confidence_final",
            "has_user_corroboration",
            "superseded_by_assertion_id",
            "supersession_type",
            "char_start",
            "char_end",
            "surface_text",
            "extraction_method",
            "extraction_model",
            "raw_assertion_json",
        ],
        "retractions": [
            "retraction_id",
            "retraction_message_id",
            "target_assertion_id",
            "target_fact_key",
            "retraction_type",
            "replacement_assertion_id",
            "confidence",
            "char_start",
            "char_end",
            "surface_text",
            "raw_retraction_json",
        ],
        "llm_extraction_runs": ["run_id", "model_name", "model_version", "config_json", "started_at_utc", "completed_at_utc", "messages_processed", "assertions_extracted", "raw_stats_json"],
        "llm_extraction_calls": ["call_id", "run_id", "message_id", "request_json", "response_json", "call_timestamp_utc", "retry_count", "seed_honored", "parse_success", "raw_io_json"],
        "entity_canonicalization_runs": [
            "run_id",
            "method",
            "model_name",
            "model_version",
            "config_json",
            "started_at_utc",
            "completed_at_utc",
            "entities_processed",
            "names_changed",
            "raw_stats_json",
        ],
        "entity_canonical_name_history": ["history_id", "entity_id", "run_id", "previous_name", "canonical_name", "selection_method", "confidence", "selected_at_utc", "raw_selection_json"],
    }

    def __init__(self, database_path: Path):
        super().__init__(database_path)

    def initialize_stage3_schema(self, overwrite: bool = True):
        """Create Stage 3 tables and indices."""
        cursor = self.connection.cursor()

        if overwrite:
            cursor.executescript("""
                                 DROP TABLE IF EXISTS retractions;
                                 DROP TABLE IF EXISTS assertions;
                                 DROP TABLE IF EXISTS predicates;
                                 DROP TABLE IF EXISTS llm_extraction_calls;
                                 DROP TABLE IF EXISTS llm_extraction_runs;
                                 DROP TABLE IF EXISTS entity_canonical_name_history;
                                 DROP TABLE IF EXISTS entity_canonicalization_runs;
                                 """)

        # predicates table
        cursor.execute("""
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

        cursor.execute("""
                       CREATE UNIQUE INDEX IF NOT EXISTS idx_predicates_label_norm
                           ON predicates(canonical_label_norm)
                       """)

        # assertions table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS assertions (
                                                                 assertion_id TEXT PRIMARY KEY,
                                                                 message_id TEXT NOT NULL,
                                                                 assertion_key TEXT NOT NULL,
                                                                 fact_key TEXT NOT NULL,
                                                                 subject_entity_id TEXT NOT NULL,
                                                                 predicate_id TEXT NOT NULL,
                                                                 object_entity_id TEXT,
                                                                 object_value_type TEXT,
                                                                 object_value TEXT,
                                                                 object_signature TEXT NOT NULL,
                                                                 temporal_qualifier_type TEXT,
                                                                 temporal_qualifier_id TEXT,
                                                                 modality TEXT NOT NULL,
                                                                 polarity TEXT NOT NULL,
                                                                 asserted_role TEXT NOT NULL,
                                                                 asserted_at_utc TEXT,
                                                                 confidence_extraction REAL NOT NULL,
                                                                 confidence_final REAL NOT NULL,
                                                                 has_user_corroboration INTEGER NOT NULL DEFAULT 0,
                                                                 superseded_by_assertion_id TEXT,
                                                                 supersession_type TEXT,
                                                                 char_start INTEGER,
                                                                 char_end INTEGER,
                                                                 surface_text TEXT,
                                                                 extraction_method TEXT NOT NULL,
                                                                 extraction_model TEXT,
                                                                 raw_assertion_json TEXT NOT NULL,
                                                                 FOREIGN KEY (message_id) REFERENCES messages(message_id),
                           FOREIGN KEY (subject_entity_id) REFERENCES entities(entity_id),
                           FOREIGN KEY (predicate_id) REFERENCES predicates(predicate_id),
                           FOREIGN KEY (object_entity_id) REFERENCES entities(entity_id),
                           FOREIGN KEY (temporal_qualifier_id) REFERENCES time_mentions(time_mention_id)
                           )
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_assertions_message
                           ON assertions(message_id)
                       """)
        cursor.execute("""
                       CREATE UNIQUE INDEX IF NOT EXISTS idx_assertions_key
                           ON assertions(assertion_key)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_assertions_fact_key
                           ON assertions(fact_key)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_assertions_subject
                           ON assertions(subject_entity_id)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_assertions_predicate
                           ON assertions(predicate_id)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_assertions_object_entity
                           ON assertions(object_entity_id) WHERE object_entity_id IS NOT NULL
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_assertions_temporal_qualifier
                           ON assertions(temporal_qualifier_id) WHERE temporal_qualifier_id IS NOT NULL
                       """)

        # llm_extraction_runs table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS llm_extraction_runs (
                                                                          run_id TEXT PRIMARY KEY,
                                                                          model_name TEXT NOT NULL,
                                                                          model_version TEXT NOT NULL,
                                                                          config_json TEXT NOT NULL,
                                                                          started_at_utc TEXT NOT NULL,
                                                                          completed_at_utc TEXT,
                                                                          messages_processed INTEGER NOT NULL DEFAULT 0,
                                                                          assertions_extracted INTEGER NOT NULL DEFAULT 0,
                                                                          raw_stats_json TEXT
                       )
                       """)

        # llm_extraction_calls table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS llm_extraction_calls (
                                                                           call_id TEXT PRIMARY KEY,
                                                                           run_id TEXT NOT NULL,
                                                                           message_id TEXT NOT NULL,
                                                                           request_json TEXT NOT NULL,
                                                                           response_json TEXT,
                                                                           call_timestamp_utc TEXT NOT NULL,
                                                                           retry_count INTEGER NOT NULL DEFAULT 0,
                                                                           seed_honored INTEGER,
                                                                           parse_success INTEGER NOT NULL DEFAULT 0,
                                                                           raw_io_json TEXT NOT NULL,
                                                                           FOREIGN KEY (run_id) REFERENCES llm_extraction_runs(run_id),
                           FOREIGN KEY (message_id) REFERENCES messages(message_id)
                           )
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_llm_calls_run
                           ON llm_extraction_calls(run_id)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_llm_calls_message
                           ON llm_extraction_calls(message_id)
                       """)

        # entity_canonicalization_runs table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS entity_canonicalization_runs (
                                                                                   run_id TEXT PRIMARY KEY,
                                                                                   method TEXT NOT NULL,
                                                                                   model_name TEXT,
                                                                                   model_version TEXT,
                                                                                   config_json TEXT NOT NULL,
                                                                                   started_at_utc TEXT NOT NULL,
                                                                                   completed_at_utc TEXT,
                                                                                   entities_processed INTEGER NOT NULL DEFAULT 0,
                                                                                   names_changed INTEGER NOT NULL DEFAULT 0,
                                                                                   raw_stats_json TEXT
                       )
                       """)

        # entity_canonical_name_history table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS entity_canonical_name_history (
                                                                                    history_id TEXT PRIMARY KEY,
                                                                                    entity_id TEXT NOT NULL,
                                                                                    run_id TEXT NOT NULL,
                                                                                    previous_name TEXT,
                                                                                    canonical_name TEXT NOT NULL,
                                                                                    selection_method TEXT NOT NULL,
                                                                                    confidence REAL NOT NULL,
                                                                                    selected_at_utc TEXT NOT NULL,
                                                                                    raw_selection_json TEXT NOT NULL,
                                                                                    FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
                           FOREIGN KEY (run_id) REFERENCES entity_canonicalization_runs(run_id)
                           )
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_canonical_history_entity
                           ON entity_canonical_name_history(entity_id)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_canonical_history_run
                           ON entity_canonical_name_history(run_id)
                       """)

        # retractions table
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS retractions (
                                                                  retraction_id TEXT PRIMARY KEY,
                                                                  retraction_message_id TEXT NOT NULL,
                                                                  target_assertion_id TEXT,
                                                                  target_fact_key TEXT,
                                                                  retraction_type TEXT NOT NULL,
                                                                  replacement_assertion_id TEXT,
                                                                  confidence REAL NOT NULL,
                                                                  char_start INTEGER,
                                                                  char_end INTEGER,
                                                                  surface_text TEXT,
                                                                  raw_retraction_json TEXT NOT NULL,
                                                                  FOREIGN KEY (retraction_message_id) REFERENCES messages(message_id),
                           FOREIGN KEY (target_assertion_id) REFERENCES assertions(assertion_id),
                           FOREIGN KEY (replacement_assertion_id) REFERENCES assertions(assertion_id)
                           )
                       """)

        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_retractions_message
                           ON retractions(retraction_message_id)
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_retractions_target
                           ON retractions(target_assertion_id) WHERE target_assertion_id IS NOT NULL
                       """)
        cursor.execute("""
                       CREATE INDEX IF NOT EXISTS idx_retractions_fact_key
                           ON retractions(target_fact_key) WHERE target_fact_key IS NOT NULL
                       """)

        self.connection.commit()

    def iter_messages_for_stage3(self) -> Iterator[dict]:
        """
        Iterate over messages eligible for Stage 3 processing.

        Order: (conversation_id ASC, order_index ASC, message_id ASC)
        Filter: role IN ('user', 'assistant') AND text_raw IS NOT NULL
        """
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT
                           m.message_id,
                           m.conversation_id,
                           m.role,
                           m.order_index,
                           m.created_at_utc,
                           m.timestamp_quality,
                           m.text_raw,
                           m.code_fence_ranges_json,
                           m.blockquote_ranges_json
                       FROM messages m
                       WHERE m.role IN ('user', 'assistant')
                         AND m.text_raw IS NOT NULL
                       ORDER BY m.conversation_id ASC, m.order_index ASC, m.message_id ASC
                       """)

        for row in cursor:
            yield dict(row)

    def get_context_messages(
            self,
            conversation_id: str,
            target_order_index: int,
            k: int
    ) -> list[dict]:
        """Get k prior messages for context."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT
                           m.message_id,
                           m.role,
                           m.order_index,
                           m.text_raw,
                           m.created_at_utc
                       FROM messages m
                       WHERE m.conversation_id = ?
                         AND m.order_index < ?
                         AND m.text_raw IS NOT NULL
                       ORDER BY m.order_index DESC, m.message_id DESC
                           LIMIT ?
                       """, (conversation_id, target_order_index, k))

        # Reverse to get chronological order
        rows = list(cursor)
        rows.reverse()
        return [dict(r) for r in rows]

    def get_context_entities(self, message_ids: list[str]) -> list[dict]:
        """Get entities mentioned in context messages."""
        if not message_ids:
            return []

        placeholders = ",".join("?" * len(message_ids))
        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT DISTINCT
                e.entity_id,
                e.entity_type,
                e.canonical_name
            FROM entity_mentions em
            JOIN entities e ON em.entity_id = e.entity_id
            WHERE em.message_id IN ({placeholders})
            ORDER BY e.entity_type, e.canonical_name
        """, message_ids)

        return [dict(r) for r in cursor]

    def get_context_times(self, message_ids: list[str]) -> list[dict]:
        """Get resolved time mentions from context messages."""
        if not message_ids:
            return []

        placeholders = ",".join("?" * len(message_ids))
        cursor = self.connection.cursor()
        cursor.execute(f"""
            SELECT 
                tm.time_mention_id,
                tm.message_id,
                tm.surface_text,
                tm.resolved_type,
                tm.valid_from_utc,
                tm.valid_to_utc
            FROM time_mentions tm
            WHERE tm.message_id IN ({placeholders})
              AND tm.resolved_type IN ('instant', 'interval')
            ORDER BY tm.message_id, tm.char_start
        """, message_ids)

        return [dict(r) for r in cursor]

    def get_time_mentions_for_message(self, message_id: str) -> list[dict]:
        """Get all time mentions for a specific message."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT
                           time_mention_id,
                           char_start,
                           char_end,
                           surface_text,
                           resolved_type,
                           valid_from_utc,
                           valid_to_utc,
                           confidence
                       FROM time_mentions
                       WHERE message_id = ?
                         AND resolved_type IN ('instant', 'interval')
                       ORDER BY confidence DESC, char_start ASC
                       """, (message_id,))

        return [dict(r) for r in cursor]

    def get_all_active_entities(self) -> list[dict]:
        """Get all active entities for resolution."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT
                           entity_id,
                           entity_type,
                           entity_key,
                           canonical_name,
                           aliases_json
                       FROM entities
                       WHERE status = 'active'
                       ORDER BY entity_type, canonical_name
                       """)

        return [dict(r) for r in cursor]

    def get_self_entity_id(self) -> str | None:
        """Get the reserved SELF entity ID."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT entity_id FROM entities
                       WHERE entity_type = 'PERSON' AND entity_key = '__SELF__'
                       """)
        row = cursor.fetchone()
        return row["entity_id"] if row else None

    def get_entity_by_id(self, entity_id: str) -> dict | None:
        """Get entity by ID."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT * FROM entities WHERE entity_id = ?
                       """, (entity_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def create_entity(
            self,
            id_generator: IDGenerator,
            entity_type: str,
            entity_key: str,
            canonical_name: str
    ) -> str:
        """Create a new entity and return its ID."""
        entity_id = id_generator.generate(["entity", entity_type, entity_key])
        now = TimestampUtils.now_utc()

        self.connection.execute("""
                                INSERT INTO entities (
                                    entity_id, entity_type, entity_key, canonical_name,
                                    aliases_json, status, first_seen_at_utc, mention_count, conversation_count
                                ) VALUES (?, ?, ?, ?, ?, 'active', ?, 0, 0)
                                """, (
                                    entity_id, entity_type, entity_key, canonical_name,
                                    JCS.canonicalize([canonical_name]), now
                                ))

        return entity_id

    def get_or_create_predicate(
            self,
            id_generator: IDGenerator,
            canonical_label: str,
            canonical_label_norm: str,
            arity: int = 2,
            category: str | None = None
    ) -> str:
        """Get or create predicate, return predicate_id."""
        cursor = self.connection.cursor()

        # Check if exists
        cursor.execute("""
                       SELECT predicate_id FROM predicates WHERE canonical_label_norm = ?
                       """, (canonical_label_norm,))
        row = cursor.fetchone()

        if row:
            return row["predicate_id"]

        # Create new
        predicate_id = id_generator.generate(["pred", canonical_label_norm])
        now = TimestampUtils.now_utc()

        cursor.execute("""
                       INSERT INTO predicates (
                           predicate_id, canonical_label, canonical_label_norm,
                           arity, category, first_seen_at_utc, assertion_count
                       ) VALUES (?, ?, ?, ?, ?, ?, 0)
                       """, (
                           predicate_id, canonical_label, canonical_label_norm,
                           arity, category, now
                       ))

        return predicate_id

    def insert_assertion(self, assertion: GroundedAssertion, assertion_id: str):
        """Insert a new assertion."""
        self.connection.execute("""
                                INSERT INTO assertions (
                                    assertion_id, message_id, assertion_key, fact_key,
                                    subject_entity_id, predicate_id,
                                    object_entity_id, object_value_type, object_value, object_signature,
                                    temporal_qualifier_type, temporal_qualifier_id,
                                    modality, polarity, asserted_role, asserted_at_utc,
                                    confidence_extraction, confidence_final, has_user_corroboration,
                                    char_start, char_end, surface_text,
                                    extraction_method, extraction_model, raw_assertion_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    assertion_id, assertion.message_id, assertion.assertion_key, assertion.fact_key,
                                    assertion.subject_entity_id, assertion.predicate_id,
                                    assertion.object_entity_id, assertion.object_value_type,
                                    assertion.object_value, assertion.object_signature,
                                    assertion.temporal_qualifier_type, assertion.temporal_qualifier_id,
                                    assertion.modality, assertion.polarity, assertion.asserted_role, assertion.asserted_at_utc,
                                    assertion.confidence_extraction, assertion.confidence_final, assertion.has_user_corroboration,
                                    assertion.char_start, assertion.char_end, assertion.surface_text,
                                    assertion.extraction_method, assertion.extraction_model, assertion.raw_assertion_json
                                ))

    def get_assertion_by_key(self, assertion_key: str) -> dict | None:
        """Get assertion by assertion_key."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT * FROM assertions WHERE assertion_key = ?
                       """, (assertion_key,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_assertion_confidence(
            self,
            assertion_key: str,
            new_confidence: float,
            new_raw_json: str
    ):
        """Update assertion confidence if higher."""
        self.connection.execute("""
                                UPDATE assertions
                                SET confidence_final = ?, raw_assertion_json = ?
                                WHERE assertion_key = ?
                                """, (new_confidence, new_raw_json, assertion_key))

    def get_assertions_for_corroboration(
            self,
            conversation_id: str,
            target_order_index: int,
            window_size: int,
            subject_entity_id: str
    ) -> list[dict]:
        """
        Get user assertions within window for corroboration check.

        Only looks backward (prior assertions can corroborate later assistant claims).
        """
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT
                           a.assertion_id,
                           a.predicate_id,
                           a.object_signature,
                           a.modality,
                           a.polarity,
                           p.canonical_label_norm
                       FROM assertions a
                                JOIN messages m ON a.message_id = m.message_id
                                JOIN predicates p ON a.predicate_id = p.predicate_id
                       WHERE m.conversation_id = ?
                         AND m.order_index >= ? AND m.order_index < ?
                         AND a.asserted_role = 'user'
                         AND a.subject_entity_id = ?
                       ORDER BY m.order_index ASC, a.assertion_id ASC
                       """, (
                           conversation_id,
                           max(0, target_order_index - window_size),
                           target_order_index,
                           subject_entity_id
                       ))

        return [dict(r) for r in cursor]

    def get_assertions_by_fact_key(self, fact_key: str) -> list[dict]:
        """Get assertions matching a fact key."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT * FROM assertions
                       WHERE fact_key = ?
                       ORDER BY asserted_at_utc DESC, assertion_id ASC
                       """, (fact_key,))
        return [dict(r) for r in cursor]

    def insert_retraction(
            self,
            retraction_id: str,
            retraction_message_id: str,
            target_assertion_id: str | None,
            target_fact_key: str | None,
            retraction_type: str,
            replacement_assertion_id: str | None,
            confidence: float,
            char_start: int | None,
            char_end: int | None,
            surface_text: str | None,
            raw_retraction_json: str
    ):
        """Insert a retraction record."""
        self.connection.execute("""
                                INSERT INTO retractions (
                                    retraction_id, retraction_message_id,
                                    target_assertion_id, target_fact_key, retraction_type,
                                    replacement_assertion_id, confidence,
                                    char_start, char_end, surface_text, raw_retraction_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    retraction_id, retraction_message_id,
                                    target_assertion_id, target_fact_key, retraction_type,
                                    replacement_assertion_id, confidence,
                                    char_start, char_end, surface_text, raw_retraction_json
                                ))

    def update_assertion_supersession(
            self,
            assertion_id: str,
            superseded_by: str,
            supersession_type: str
    ):
        """Mark an assertion as superseded."""
        self.connection.execute("""
                                UPDATE assertions
                                SET superseded_by_assertion_id = ?, supersession_type = ?
                                WHERE assertion_id = ?
                                """, (superseded_by, supersession_type, assertion_id))

    def refresh_predicate_stats(self):
        """Recompute assertion counts for all predicates."""
        self.connection.execute("""
                                UPDATE predicates
                                SET assertion_count = (
                                    SELECT COUNT(*) FROM assertions
                                    WHERE assertions.predicate_id = predicates.predicate_id
                                )
                                """)

    def insert_canonicalization_run(
            self,
            run_id: str,
            method: str,
            config_json: str,
            started_at_utc: str
    ):
        """Insert canonicalization run record."""
        self.connection.execute("""
                                INSERT INTO entity_canonicalization_runs (
                                    run_id, method, config_json, started_at_utc,
                                    entities_processed, names_changed
                                ) VALUES (?, ?, ?, ?, 0, 0)
                                """, (run_id, method, config_json, started_at_utc))

    def update_canonicalization_run(
            self,
            run_id: str,
            completed_at_utc: str,
            entities_processed: int,
            names_changed: int,
            raw_stats_json: str | None
    ):
        """Update canonicalization run with completion stats."""
        self.connection.execute("""
                                UPDATE entity_canonicalization_runs
                                SET completed_at_utc = ?, entities_processed = ?,
                                    names_changed = ?, raw_stats_json = ?
                                WHERE run_id = ?
                                """, (completed_at_utc, entities_processed, names_changed, raw_stats_json, run_id))

    def insert_canonical_name_history(
            self,
            history_id: str,
            entity_id: str,
            run_id: str,
            previous_name: str | None,
            canonical_name: str,
            selection_method: str,
            confidence: float,
            selected_at_utc: str,
            raw_selection_json: str
    ):
        """Insert canonical name history record."""
        self.connection.execute("""
                                INSERT INTO entity_canonical_name_history (
                                    history_id, entity_id, run_id, previous_name, canonical_name,
                                    selection_method, confidence, selected_at_utc, raw_selection_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    history_id, entity_id, run_id, previous_name, canonical_name,
                                    selection_method, confidence, selected_at_utc, raw_selection_json
                                ))

    def update_entity_canonical_name(self, entity_id: str, new_name: str):
        """Update entity's canonical name."""
        self.connection.execute("""
                                UPDATE entities SET canonical_name = ? WHERE entity_id = ?
                                """, (new_name, entity_id))

    def get_entity_mentions_with_roles(self, entity_id: str) -> list[dict]:
        """Get all mentions of an entity with message roles."""
        cursor = self.connection.cursor()
        cursor.execute("""
                       SELECT
                           em.surface_text,
                           m.role,
                           m.created_at_utc,
                           m.conversation_id,
                           m.order_index,
                           m.message_id,
                           em.mention_id
                       FROM entity_mentions em
                                JOIN messages m ON em.message_id = m.message_id
                       WHERE em.entity_id = ?
                         AND em.surface_text IS NOT NULL
                       ORDER BY m.created_at_utc NULLS LAST, m.conversation_id, m.order_index, m.message_id, em.mention_id
                       """, (entity_id,))
        return [dict(r) for r in cursor]

    def insert_llm_extraction_run(
            self,
            run_id: str,
            model_name: str,
            model_version: str,
            config_json: str,
            started_at_utc: str
    ):
        """Insert LLM extraction run record."""
        self.connection.execute("""
                                INSERT INTO llm_extraction_runs (
                                    run_id, model_name, model_version, config_json, started_at_utc,
                                    messages_processed, assertions_extracted
                                ) VALUES (?, ?, ?, ?, ?, 0, 0)
                                """, (run_id, model_name, model_version, config_json, started_at_utc))

    def update_llm_extraction_run(
            self,
            run_id: str,
            completed_at_utc: str,
            messages_processed: int,
            assertions_extracted: int,
            raw_stats_json: str | None
    ):
        """Update LLM extraction run with completion stats."""
        self.connection.execute("""
                                UPDATE llm_extraction_runs
                                SET completed_at_utc = ?, messages_processed = ?,
                                    assertions_extracted = ?, raw_stats_json = ?
                                WHERE run_id = ?
                                """, (completed_at_utc, messages_processed, assertions_extracted, raw_stats_json, run_id))

    def insert_llm_extraction_call(
            self,
            call_id: str,
            run_id: str,
            message_id: str,
            request_json: str,
            response_json: str | None,
            call_timestamp_utc: str,
            retry_count: int,
            seed_honored: int | None,
            parse_success: int,
            raw_io_json: str
    ):
        """Insert LLM extraction call record."""
        self.connection.execute("""
                                INSERT INTO llm_extraction_calls (
                                    call_id, run_id, message_id, request_json, response_json,
                                    call_timestamp_utc, retry_count, seed_honored, parse_success, raw_io_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    call_id, run_id, message_id, request_json, response_json,
                                    call_timestamp_utc, retry_count, seed_honored, parse_success, raw_io_json
                                ))


# ============================================================================
# RULE-BASED EXTRACTOR
# ============================================================================

@dataclass
class ExtractionPattern:
    """A single extraction pattern definition."""

    pattern_id: str
    regex: re.Pattern
    predicate_label: str
    modality: str = Modality.STATE
    polarity: str = Polarity.POSITIVE
    arity: int = 2  # 1=unary, 2=binary
    subject_group: int = 0  # 0 means SELF
    object_group: int | None = None  # capture group for object
    object_type: str | None = None  # entity type hint or literal type
    confidence: float = 0.9
    category: str | None = None


class RuleBasedExtractor:
    """
    Rule-based assertion extractor using high-precision patterns.

    Patterns are conservative and designed to minimize false positives
    while capturing common personal knowledge assertions.
    """

    def __init__(self):
        self.patterns: list[ExtractionPattern] = self._build_patterns()

    def _build_patterns(self) -> list[ExtractionPattern]:
        """Build the pinned registry of extraction patterns."""
        patterns = []

        # === IDENTITY PATTERNS ===

        # "My name is [Name]" / "I'm [Name]" / "I am [Name]"
        patterns.append(ExtractionPattern(
            pattern_id="identity_name_1",
            regex=re.compile(
                r"\b(?:my\s+name\s+is|i'?m|i\s+am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                re.IGNORECASE
            ),
            predicate_label="has_name",
            modality=Modality.FACT,
            object_group=1,
            object_type="PERSON",
            confidence=0.95,
            category="identity"
        ))

        # "Call me [Name]"
        patterns.append(ExtractionPattern(
            pattern_id="identity_name_2",
            regex=re.compile(
                r"\bcall\s+me\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
                re.IGNORECASE
            ),
            predicate_label="prefers_name",
            modality=Modality.PREFERENCE,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="identity"
        ))

        # === LOCATION PATTERNS ===

        # "I live in [Location]" / "I'm from [Location]" / "I'm based in [Location]"
        patterns.append(ExtractionPattern(
            pattern_id="location_live_1",
            regex=re.compile(
                r"\bi\s+(?:live|reside|am\s+(?:based|located))\s+in\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$|\s+(?:and|but|since|for))",
                re.IGNORECASE
            ),
            predicate_label="lives_in",
            modality=Modality.STATE,
            object_group=1,
            object_type="LOCATION",
            confidence=0.9,
            category="location"
        ))

        # "I'm from [Location]"
        patterns.append(ExtractionPattern(
            pattern_id="location_from_1",
            regex=re.compile(
                r"\bi(?:'m|\s+am)\s+from\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$|\s+(?:and|but|originally))",
                re.IGNORECASE
            ),
            predicate_label="from_location",
            modality=Modality.FACT,
            object_group=1,
            object_type="LOCATION",
            confidence=0.85,
            category="location"
        ))

        # "I moved to [Location]"
        patterns.append(ExtractionPattern(
            pattern_id="location_moved_1",
            regex=re.compile(
                r"\bi\s+(?:moved|relocated)\s+to\s+([A-Z][a-zA-Z\s,]+?)(?:\.|,|$|\s+(?:and|but|in|last))",
                re.IGNORECASE
            ),
            predicate_label="lives_in",
            modality=Modality.STATE,
            object_group=1,
            object_type="LOCATION",
            confidence=0.85,
            category="location"
        ))

        # === OCCUPATION PATTERNS ===

        # "I am a/an [Occupation]" / "I'm a/an [Occupation]"
        patterns.append(ExtractionPattern(
            pattern_id="occupation_1",
            regex=re.compile(
                r"\bi(?:'m|\s+am)\s+(?:a|an)\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|at|for|who|with|working))",
                re.IGNORECASE
            ),
            predicate_label="has_occupation",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.85,
            category="occupation"
        ))

        # "I work as a/an [Occupation]"
        patterns.append(ExtractionPattern(
            pattern_id="occupation_2",
            regex=re.compile(
                r"\bi\s+work\s+as\s+(?:a|an)\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|at|for))",
                re.IGNORECASE
            ),
            predicate_label="has_occupation",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="occupation"
        ))

        # "My job/role/title is [Title]"
        patterns.append(ExtractionPattern(
            pattern_id="occupation_3",
            regex=re.compile(
                r"\bmy\s+(?:job|role|title|position)\s+is\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|at|for))",
                re.IGNORECASE
            ),
            predicate_label="job_title",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="occupation"
        ))

        # === EMPLOYER PATTERNS ===

        # "I work at/for [Company]"
        patterns.append(ExtractionPattern(
            pattern_id="employer_1",
            regex=re.compile(
                r"\bi\s+work\s+(?:at|for)\s+([A-Z][a-zA-Z\s&.]+?)(?:\.|,|$|\s+(?:and|as|in|since))",
                re.IGNORECASE
            ),
            predicate_label="works_at",
            modality=Modality.STATE,
            object_group=1,
            object_type="ORG",
            confidence=0.9,
            category="employment"
        ))

        # "I'm employed by/at [Company]"
        patterns.append(ExtractionPattern(
            pattern_id="employer_2",
            regex=re.compile(
                r"\bi(?:'m|\s+am)\s+employed\s+(?:by|at)\s+([A-Z][a-zA-Z\s&.]+?)(?:\.|,|$|\s+(?:and|as|in))",
                re.IGNORECASE
            ),
            predicate_label="works_at",
            modality=Modality.STATE,
            object_group=1,
            object_type="ORG",
            confidence=0.9,
            category="employment"
        ))

        # === PREFERENCE PATTERNS ===

        # "I like/love/prefer [Thing]"
        patterns.append(ExtractionPattern(
            pattern_id="preference_like_1",
            regex=re.compile(
                r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer)\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|but|because|so|very))",
                re.IGNORECASE
            ),
            predicate_label="likes",
            modality=Modality.PREFERENCE,
            object_group=1,
            object_type="string",
            confidence=0.8,
            category="preference"
        ))

        # "I don't like/hate [Thing]"
        patterns.append(ExtractionPattern(
            pattern_id="preference_dislike_1",
            regex=re.compile(
                r"\bi\s+(?:don'?t\s+(?:really\s+)?like|hate|dislike|can'?t\s+stand)\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|but|because))",
                re.IGNORECASE
            ),
            predicate_label="dislikes",
            modality=Modality.PREFERENCE,
            object_group=1,
            object_type="string",
            confidence=0.8,
            category="preference"
        ))

        # "My favorite [Category] is [Thing]"
        patterns.append(ExtractionPattern(
            pattern_id="preference_favorite_1",
            regex=re.compile(
                r"\bmy\s+(?:favorite|favourite)\s+([a-zA-Z]+)\s+is\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|but|because))",
                re.IGNORECASE
            ),
            predicate_label="favorite",
            modality=Modality.PREFERENCE,
            object_group=2,  # The thing, not the category
            object_type="string",
            confidence=0.85,
            category="preference"
        ))

        # === AGE/BIRTH PATTERNS ===

        # "I am [Age] years old"
        patterns.append(ExtractionPattern(
            pattern_id="age_1",
            regex=re.compile(
                r"\bi(?:'m|\s+am)\s+(\d{1,3})\s+(?:years?\s+old|y/?o)\b",
                re.IGNORECASE
            ),
            predicate_label="has_age",
            modality=Modality.STATE,
            object_group=1,
            object_type="number",
            confidence=0.9,
            category="demographics"
        ))

        # "I was born in [Year/Location]"
        patterns.append(ExtractionPattern(
            pattern_id="birth_year_1",
            regex=re.compile(
                r"\bi\s+was\s+born\s+in\s+(\d{4})\b",
                re.IGNORECASE
            ),
            predicate_label="birth_year",
            modality=Modality.FACT,
            object_group=1,
            object_type="date",
            confidence=0.9,
            category="demographics"
        ))

        # === LANGUAGE PATTERNS ===

        # "I speak [Language]"
        patterns.append(ExtractionPattern(
            pattern_id="language_1",
            regex=re.compile(
                r"\bi\s+speak\s+([A-Z][a-z]+(?:\s+and\s+[A-Z][a-z]+)*)\b",
                re.IGNORECASE
            ),
            predicate_label="speaks_language",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="language"
        ))

        # "My native language is [Language]"
        patterns.append(ExtractionPattern(
            pattern_id="language_2",
            regex=re.compile(
                r"\bmy\s+(?:native|first|primary)\s+language\s+is\s+([A-Z][a-z]+)\b",
                re.IGNORECASE
            ),
            predicate_label="native_language",
            modality=Modality.FACT,
            object_group=1,
            object_type="string",
            confidence=0.95,
            category="language"
        ))

        # === EDUCATION PATTERNS ===

        # "I studied [Subject] at [School]"
        patterns.append(ExtractionPattern(
            pattern_id="education_1",
            regex=re.compile(
                r"\bi\s+(?:studied|majored\s+in)\s+([a-zA-Z\s]+?)(?:\s+at\s+([A-Z][a-zA-Z\s]+))?(?:\.|,|$)",
                re.IGNORECASE
            ),
            predicate_label="studied",
            modality=Modality.FACT,
            object_group=1,
            object_type="string",
            confidence=0.85,
            category="education"
        ))

        # "I have a [Degree] in [Subject]"
        patterns.append(ExtractionPattern(
            pattern_id="education_2",
            regex=re.compile(
                r"\bi\s+have\s+(?:a|an)\s+(bachelor'?s?|master'?s?|phd|doctorate|degree)\s+in\s+([a-zA-Z\s]+?)(?:\.|,|$)",
                re.IGNORECASE
            ),
            predicate_label="has_degree",
            modality=Modality.FACT,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="education"
        ))

        # === FAMILY PATTERNS ===

        # "I have [Number] [Relation]" (kids, children, siblings, etc.)
        patterns.append(ExtractionPattern(
            pattern_id="family_1",
            regex=re.compile(
                r"\bi\s+have\s+(\d+|one|two|three|four|five|no)\s+(kids?|children|siblings?|brothers?|sisters?|sons?|daughters?)\b",
                re.IGNORECASE
            ),
            predicate_label="has_family_members",
            modality=Modality.STATE,
            object_group=0,  # Will handle specially
            object_type="string",
            confidence=0.85,
            category="family"
        ))

        # "I'm married" / "I'm single"
        patterns.append(ExtractionPattern(
            pattern_id="family_2",
            regex=re.compile(
                r"\bi(?:'m|\s+am)\s+(married|single|divorced|widowed|engaged)\b",
                re.IGNORECASE
            ),
            predicate_label="marital_status",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="family"
        ))

        # === CONTACT PATTERNS ===

        # "My email is [Email]"
        patterns.append(ExtractionPattern(
            pattern_id="contact_email_1",
            regex=re.compile(
                r"\bmy\s+email(?:\s+address)?\s+is\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
                re.IGNORECASE
            ),
            predicate_label="has_email",
            modality=Modality.STATE,
            object_group=1,
            object_type="EMAIL",
            confidence=0.95,
            category="contact"
        ))

        # "My timezone is [TZ]"
        patterns.append(ExtractionPattern(
            pattern_id="contact_tz_1",
            regex=re.compile(
                r"\bmy\s+timezone\s+is\s+([A-Z]{2,4}(?:[+-]\d{1,2})?|[A-Z][a-z]+/[A-Z][a-z_]+)\b",
                re.IGNORECASE
            ),
            predicate_label="timezone",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.9,
            category="contact"
        ))

        # === HOBBY/INTEREST PATTERNS ===

        # "I'm interested in [Topic]"
        patterns.append(ExtractionPattern(
            pattern_id="interest_1",
            regex=re.compile(
                r"\bi(?:'m|\s+am)\s+interested\s+in\s+([a-zA-Z\s]+?)(?:\.|,|$|\s+(?:and|but|because))",
                re.IGNORECASE
            ),
            predicate_label="interested_in",
            modality=Modality.PREFERENCE,
            object_group=1,
            object_type="string",
            confidence=0.8,
            category="interest"
        ))

        # "My hobby is [Hobby]"
        patterns.append(ExtractionPattern(
            pattern_id="hobby_1",
            regex=re.compile(
                r"\bmy\s+hobb(?:y|ies)\s+(?:is|are|include)\s+([a-zA-Z\s,]+?)(?:\.|$|\s+(?:and|but))",
                re.IGNORECASE
            ),
            predicate_label="has_hobby",
            modality=Modality.STATE,
            object_group=1,
            object_type="string",
            confidence=0.85,
            category="interest"
        ))

        return patterns

    def extract(
            self,
            text: str,
            message_id: str,
            excluded_ranges: list[tuple[int, int]]
    ) -> list[AssertionCandidate]:
        """
        Extract assertion candidates from text using patterns.

        Args:
            text: The message text to analyze
            message_id: ID of the source message
            excluded_ranges: Character ranges to skip (code fences, etc.)

        Returns:
            List of AssertionCandidate objects, sorted deterministically

        """
        candidates = []

        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                char_start = match.start()
                char_end = match.end()

                # Skip if overlaps with excluded ranges
                if self._overlaps_excluded(char_start, char_end, excluded_ranges):
                    continue

                # Extract object value
                object_value = None
                object_literal_type = None
                object_entity_ref = None

                if pattern.object_group is not None and pattern.object_group > 0:
                    try:
                        object_value = match.group(pattern.object_group)
                        if object_value:
                            object_value = object_value.strip()

                            # Determine if entity or literal
                            if pattern.object_type in ("PERSON", "ORG", "LOCATION", "EMAIL"):
                                object_entity_ref = object_value
                            elif pattern.object_type == "number":
                                object_literal_type = ObjectValueType.NUMBER
                                # Convert word numbers
                                object_value = self._word_to_number(object_value)
                            elif pattern.object_type == "date":
                                object_literal_type = ObjectValueType.DATE
                            else:
                                object_literal_type = ObjectValueType.STRING
                    except IndexError:
                        continue

                candidate = AssertionCandidate(
                    message_id=message_id,
                    subject_ref="SELF",
                    predicate_label=pattern.predicate_label,
                    object_entity_ref=object_entity_ref,
                    object_literal_type=object_literal_type,
                    object_literal_value=object_value,
                    modality=pattern.modality,
                    polarity=pattern.polarity,
                    char_start=char_start,
                    char_end=char_end,
                    extraction_method=ExtractionMethod.RULE_BASED,
                    confidence=pattern.confidence,
                    pattern_id=pattern.pattern_id,
                    raw_data={
                        "pattern_id": pattern.pattern_id,
                        "full_match": match.group(0),
                        "groups": match.groups(),
                        "category": pattern.category
                    }
                )
                candidates.append(candidate)

        # Sort deterministically
        candidates.sort(key=lambda c: (
            c.char_start if c.char_start is not None else float("inf"),
            c.char_end if c.char_end is not None else float("inf"),
            PredicateNormalizer.normalize(c.predicate_label)[1],
            -c.confidence,
            HashUtils.sha256_string(str(c.raw_data))[:16]
        ))

        return candidates

    def _overlaps_excluded(
            self,
            start: int,
            end: int,
            excluded_ranges: list[tuple[int, int]]
    ) -> bool:
        """Check if span overlaps any excluded range."""
        for ex_start, ex_end in excluded_ranges:
            if start < ex_end and end > ex_start:
                return True
        return False

    def _word_to_number(self, word: str) -> Any:
        """Convert word numbers to integers."""
        word_map = {
            "zero": 0, "no": 0, "one": 1, "two": 2, "three": 3,
            "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
            "nine": 9, "ten": 10
        }
        lower = word.lower().strip()
        if lower in word_map:
            return word_map[lower]
        try:
            return int(word)
        except ValueError:
            return word


# ============================================================================
# LLM EXTRACTOR (SCAFFOLDING)
# ============================================================================

class LLMExtractor:
    """
    LLM-based assertion extractor.

    This is scaffolding for future implementation. Currently provides
    structure for LLM extraction with full audit logging.

    To implement:
    1. Create prompt with context, masked text, and entity/time info
    2. Send to LLM API with deterministic settings (temp=0, seed)
    3. Parse structured JSON response
    4. Log all I/O to llm_extraction_calls
    5. Handle retries and errors

    Output JSON schema expected from LLM:
    {
        "assertions": [
            {
                "subject": "SELF" | "<entity_name>",
                "predicate": "<predicate_label>",
                "object": {
                    "type": "entity" | "literal",
                    "value": "<value>",
                    "literal_type": "string" | "number" | "boolean" | "date" | null
                },
                "modality": "state" | "fact" | "preference" | "intention",
                "polarity": "positive" | "negative",
                "evidence": {
                    "quote": "<exact quote from text>",
                    "char_start": <int> | null,
                    "char_end": <int> | null
                },
                "temporal": {
                    "qualifier_type": "at" | "since" | "until" | "during" | null,
                    "surface": "<time expression>" | null,
                    "value": "<ISO datetime>" | null
                },
                "confidence": <float 0-1>
            }
        ]
    }
    """

    def __init__(self, config: Stage3Config):
        self.config = config
        self.model_name = config.llm_model_name
        self.model_version = config.llm_model_version

    def extract(
            self,
            context: MessageContext,
            db: Stage3Database,
            id_generator: IDGenerator,
            run_id: str
    ) -> list[AssertionCandidate]:
        """
        Extract assertions using LLM.

        This is a placeholder implementation. To fully implement:

        1. Build the prompt:
           - Include masked target text
           - Include k context messages
           - Include entity list with canonical names
           - Include resolved time mentions
           - Specify output JSON schema

        2. Call LLM API:
           - Use config.llm_temperature, llm_top_p, llm_seed
           - Implement retry logic with config.max_retries_llm

        3. Parse response:
           - Validate against schema
           - Convert to AssertionCandidate objects

        4. Log to database:
           - Insert llm_extraction_calls record for each call

        5. If llm_multi_run_count > 1:
           - Run multiple times
           - Aggregate by semantic equivalence
           - Keep only majority occurrences

        Args:
            context: MessageContext with target message and context
            db: Database connection for logging
            id_generator: For generating call IDs
            run_id: LLM extraction run ID

        Returns:
            List of AssertionCandidate objects

        """
        # Placeholder - return empty list
        # Full implementation would call LLM API here

        logger.debug(
            f"LLM extraction not implemented. "
            f"Would process message {context.target_message['message_id']}"
        )

        return []

    def _build_prompt(self, context: MessageContext) -> str:
        """
        Build the LLM prompt.

        Structure:
        - System instructions for extraction
        - Context messages (role-labeled)
        - Entity reference list
        - Time mention list
        - Target message (masked)
        - Output format specification
        """
        # Placeholder
        return ""

    def _parse_response(
            self,
            response_json: dict,
            message_id: str
    ) -> list[AssertionCandidate]:
        """
        Parse LLM response into AssertionCandidate objects.

        Validates response structure and converts to internal format.
        """
        candidates = []
        # Placeholder implementation
        return candidates


# ============================================================================
# RETRACTION DETECTOR
# ============================================================================

class RetractionDetector:
    """
    Detects retractions and corrections in user messages.

    Uses conservative patterns to identify explicit retractions,
    corrections, and negations of previous assertions.
    """

    def __init__(self):
        self.patterns = self._build_patterns()

    def _build_patterns(self) -> list[tuple[str, re.Pattern, str, float]]:
        """
        Build retraction/correction patterns.

        Returns list of (pattern_id, regex, retraction_type, confidence)
        """
        patterns = []

        # Full retractions
        patterns.append((
            "retract_actually_not",
            re.compile(
                r"\b(?:actually|wait),?\s+(?:i\s+)?(?:don'?t|do\s+not)\s+(.+?)(?:\.|,|$)",
                re.IGNORECASE
            ),
            RetractionType.FULL,
            0.85
        ))

        patterns.append((
            "retract_no_longer",
            re.compile(
                r"\bi\s+(?:no\s+longer|don'?t\s+(?:anymore|any\s+more))\s+(.+?)(?:\.|,|$)",
                re.IGNORECASE
            ),
            RetractionType.TEMPORAL_BOUND,
            0.85
        ))

        patterns.append((
            "retract_thats_wrong",
            re.compile(
                r"\b(?:that'?s\s+(?:not\s+(?:right|correct|true)|wrong)|i\s+was\s+wrong)\b",
                re.IGNORECASE
            ),
            RetractionType.FULL,
            0.75
        ))

        # Corrections
        patterns.append((
            "correct_actually",
            re.compile(
                r"\b(?:actually|correction|wait),?\s+(?:i\s+)?(.+?)(?:\s+(?:not|instead|rather))\s+(.+?)(?:\.|,|$)",
                re.IGNORECASE
            ),
            RetractionType.CORRECTION,
            0.8
        ))

        patterns.append((
            "correct_meant_to_say",
            re.compile(
                r"\bi\s+meant\s+(?:to\s+say\s+)?(.+?)(?:\.|,|$)",
                re.IGNORECASE
            ),
            RetractionType.CORRECTION,
            0.85
        ))

        patterns.append((
            "correct_should_be",
            re.compile(
                r"\b(?:it\s+)?should\s+(?:be|have\s+been)\s+(.+?)(?:\s+not\s+(.+?))?(?:\.|,|$)",
                re.IGNORECASE
            ),
            RetractionType.CORRECTION,
            0.8
        ))

        return patterns

    def detect(
            self,
            text: str,
            message_id: str,
            message_role: str,
            excluded_ranges: list[tuple[int, int]]
    ) -> list[RetractionCandidate]:
        """
        Detect retractions in message text.

        Only processes user messages (retractions must come from user).
        """
        if message_role != "user":
            return []

        candidates = []

        for pattern_id, regex, retraction_type, confidence in self.patterns:
            for match in regex.finditer(text):
                char_start = match.start()
                char_end = match.end()

                # Skip if overlaps excluded ranges
                if self._overlaps_excluded(char_start, char_end, excluded_ranges):
                    continue

                candidate = RetractionCandidate(
                    message_id=message_id,
                    retraction_type=retraction_type,
                    char_start=char_start,
                    char_end=char_end,
                    surface_text=match.group(0),
                    confidence=confidence,
                    raw_data={
                        "pattern_id": pattern_id,
                        "groups": match.groups()
                    }
                )
                candidates.append(candidate)

        return candidates

    def _overlaps_excluded(
            self,
            start: int,
            end: int,
            excluded_ranges: list[tuple[int, int]]
    ) -> bool:
        """Check if span overlaps any excluded range."""
        for ex_start, ex_end in excluded_ranges:
            if start < ex_end and end > ex_start:
                return True
        return False


# ============================================================================
# ENTITY RESOLVER
# ============================================================================

class EntityResolver:
    """
    Resolves entity references to entity IDs.

    Resolution order (per spec §3.5.1):
    1. Direct entity_id (if valid and exists)
    2. Exact canonical_name match among active entities
    3. Exact alias match (string in aliases_json)
    4. entity_key match using type-specific normalization
    5. Optional fuzzy match if enabled
    """

    def __init__(
            self,
            db: Stage3Database,
            id_generator: IDGenerator,
            config: Stage3Config
    ):
        self.db = db
        self.id_generator = id_generator
        self.config = config

        # Cache entities for efficient lookup
        self._entity_cache: dict[str, dict] = {}
        self._canonical_name_map: dict[str, str] = {}
        self._alias_map: dict[str, str] = {}
        self._key_map: dict[tuple[str, str], str] = {}

        self._self_entity_id: str | None = None

    def load_entities(self):
        """Load all active entities into cache."""
        self._self_entity_id = self.db.get_self_entity_id()

        entities = self.db.get_all_active_entities()
        for e in entities:
            entity_id = e["entity_id"]
            self._entity_cache[entity_id] = e

            # Index by canonical name (lowercase)
            canonical_lower = e["canonical_name"].lower()
            self._canonical_name_map[canonical_lower] = entity_id

            # Index by entity key
            self._key_map[(e["entity_type"], e["entity_key"])] = entity_id

            # Index aliases
            if e["aliases_json"]:
                try:
                    aliases = json.loads(e["aliases_json"])
                    for alias in aliases:
                        if alias:
                            self._alias_map[alias.lower()] = entity_id
                except json.JSONDecodeError:
                    pass

    def resolve(
            self,
            ref: str,
            type_hint: str | None = None
    ) -> str | None:
        """
        Resolve an entity reference to entity_id.

        Args:
            ref: Entity reference (name, key, or ID)
            type_hint: Optional entity type hint

        Returns:
            entity_id or None if not found

        """
        if not ref:
            return None

        # Handle SELF reference
        if ref.upper() == "SELF":
            return self._self_entity_id

        # 1. Direct entity_id
        if ref in self._entity_cache:
            return ref

        ref_lower = ref.lower().strip()

        # 2. Exact canonical_name match
        if ref_lower in self._canonical_name_map:
            return self._canonical_name_map[ref_lower]

        # 3. Exact alias match
        if ref_lower in self._alias_map:
            return self._alias_map[ref_lower]

        # 4. Entity key match (need to normalize)
        if type_hint:
            normalized_key = self._normalize_key(type_hint, ref)
            if (type_hint, normalized_key) in self._key_map:
                return self._key_map[(type_hint, normalized_key)]

        # 5. Fuzzy match (if threshold > 0)
        if self.config.threshold_link_string_sim > 0:
            best_match = None
            best_score = 0.0

            for name, entity_id in self._canonical_name_map.items():
                score = StringSimilarity.levenshtein_ratio(ref_lower, name)
                if score >= self.config.threshold_link_string_sim and score > best_score:
                    best_score = score
                    best_match = entity_id

            if best_match:
                return best_match

        return None

    def resolve_or_create(
            self,
            ref: str,
            type_hint: str | None = None
    ) -> str:
        """
        Resolve entity reference, creating if not found.

        Creates with type OTHER unless type_hint provided.
        """
        entity_id = self.resolve(ref, type_hint)
        if entity_id:
            return entity_id

        # Create new entity
        entity_type = type_hint if type_hint else "OTHER"
        entity_key = self._normalize_key(entity_type, ref)
        canonical_name = ref.strip()

        logger.warning(
            f"ENTITY_CREATED_FROM_REF: Creating entity for unresolved ref '{ref}' "
            f"(type={entity_type}, key={entity_key})"
        )

        entity_id = self.db.create_entity(
            self.id_generator,
            entity_type,
            entity_key,
            canonical_name
        )

        # Update cache
        self._entity_cache[entity_id] = {
            "entity_id": entity_id,
            "entity_type": entity_type,
            "entity_key": entity_key,
            "canonical_name": canonical_name
        }
        self._canonical_name_map[canonical_name.lower()] = entity_id
        self._key_map[(entity_type, entity_key)] = entity_id

        return entity_id

    def _normalize_key(self, entity_type: str, value: str) -> str:
        """Normalize value to entity key format."""
        # Basic normalization - can be extended per entity type
        return value.strip().lower()


# ============================================================================
# GROUNDING ENGINE
# ============================================================================

class GroundingEngine:
    """
    Grounds assertion candidates to entities, predicates, and times.

    Converts raw AssertionCandidate objects into fully-specified
    GroundedAssertion objects ready for persistence.
    """

    def __init__(
            self,
            db: Stage3Database,
            id_generator: IDGenerator,
            config: Stage3Config,
            entity_resolver: EntityResolver
    ):
        self.db = db
        self.id_generator = id_generator
        self.config = config
        self.entity_resolver = entity_resolver

    def ground(
            self,
            candidate: AssertionCandidate,
            message: dict,
            text_raw: str
    ) -> GroundedAssertion | None:
        """
        Ground a single assertion candidate.

        Returns GroundedAssertion or None if grounding fails.
        """
        # Resolve subject
        subject_entity_id = self.entity_resolver.resolve_or_create(
            candidate.subject_ref,
            "PERSON" if candidate.subject_ref.upper() == "SELF" else None
        )

        if not subject_entity_id:
            logger.warning(f"GROUNDING_FAILED: Could not resolve subject '{candidate.subject_ref}'")
            return None

        # Normalize and get predicate
        canonical_label, canonical_label_norm = PredicateNormalizer.normalize(
            candidate.predicate_label
        )

        # Determine arity
        has_object = (
                candidate.object_entity_ref is not None or
                candidate.object_literal_value is not None
        )
        arity = 2 if has_object else 1

        predicate_id = self.db.get_or_create_predicate(
            self.id_generator,
            canonical_label,
            canonical_label_norm,
            arity=arity
        )

        # Resolve object
        object_entity_id = None
        object_value_type = None
        object_value = None
        object_signature = "N:__NONE__"

        if candidate.object_entity_ref:
            # Entity object
            object_entity_id = self.entity_resolver.resolve_or_create(
                candidate.object_entity_ref,
                None  # Type hint from extraction if available
            )
            object_signature = f"E:{object_entity_id}"

        elif candidate.object_literal_value is not None:
            # Literal object
            object_value_type = candidate.object_literal_type or ObjectValueType.STRING

            # Canonicalize value
            if object_value_type == ObjectValueType.NUMBER:
                try:
                    parsed = float(candidate.object_literal_value) if "." in str(candidate.object_literal_value) else int(candidate.object_literal_value)
                except (ValueError, TypeError):
                    parsed = candidate.object_literal_value
                object_value = JCS.canonicalize(parsed)
            elif object_value_type == ObjectValueType.BOOLEAN:
                parsed = str(candidate.object_literal_value).lower() in ("true", "1", "yes")
                object_value = JCS.canonicalize(parsed)
            else:
                object_value = JCS.canonicalize(str(candidate.object_literal_value))

            # Compute signature
            sig_input = [object_value_type, json.loads(object_value)]
            object_signature = f"V:{HashUtils.sha256_string(JCS.canonicalize(sig_input))}"

        # Resolve temporal qualifier
        temporal_qualifier_type = candidate.temporal_qualifier_type
        temporal_qualifier_id = None

        if candidate.temporal_qualifier_surface:
            # Try to match to time_mentions
            time_mentions = self.db.get_time_mentions_for_message(candidate.message_id)

            for tm in time_mentions:
                if tm["surface_text"].lower() == candidate.temporal_qualifier_surface.lower():
                    temporal_qualifier_id = tm["time_mention_id"]
                    break

        # Get surface text from offsets if available
        surface_text = None
        if candidate.char_start is not None and candidate.char_end is not None:
            surface_text = text_raw[candidate.char_start:candidate.char_end]
        elif candidate.quote:
            # Try to find quote in text for offset resolution
            idx = text_raw.find(candidate.quote)
            if idx >= 0:
                candidate.char_start = idx
                candidate.char_end = idx + len(candidate.quote)
                surface_text = candidate.quote

        # Build raw assertion JSON
        raw_data = {
            "extraction": {
                "method": candidate.extraction_method,
                "model": candidate.extraction_model,
                "pattern_id": candidate.pattern_id,
                "confidence": candidate.confidence
            },
            "candidate": candidate.raw_data,
            "grounding": {
                "subject_ref": candidate.subject_ref,
                "subject_resolved": subject_entity_id,
                "object_ref": candidate.object_entity_ref,
                "object_resolved": object_entity_id,
                "predicate_normalized": canonical_label
            }
        }

        if candidate.temporal_qualifier_value:
            raw_data["temporal_qualifier"] = {
                "type": temporal_qualifier_type,
                "surface": candidate.temporal_qualifier_surface,
                "value": candidate.temporal_qualifier_value
            }

        # Compute keys
        char_start_key = candidate.char_start if candidate.char_start is not None else "__NULL__"

        assertion_key = JCS.canonicalize([
            candidate.message_id,
            subject_entity_id,
            predicate_id,
            object_signature,
            char_start_key,
            candidate.modality,
            candidate.polarity
        ])

        fact_key = JCS.canonicalize([
            subject_entity_id,
            predicate_id,
            object_signature
        ])

        # Build grounded assertion
        return GroundedAssertion(
            message_id=candidate.message_id,
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
            asserted_role=message["role"],
            asserted_at_utc=message.get("created_at_utc"),
            confidence_extraction=candidate.confidence,
            confidence_final=candidate.confidence,  # Updated after corroboration
            has_user_corroboration=0,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            surface_text=surface_text,
            extraction_method=candidate.extraction_method,
            extraction_model=candidate.extraction_model,
            raw_assertion_json=JCS.canonicalize(raw_data)
        )


# ============================================================================
# CORROBORATION DETECTOR
# ============================================================================

class CorroborationDetector:
    """
    Detects user corroboration for assistant assertions.

    An assistant assertion is corroborated if a prior user assertion
    exists with matching subject, similar predicate, and compatible object.
    """

    def __init__(self, config: Stage3Config):
        self.config = config

    def check_corroboration(
            self,
            assertion: GroundedAssertion,
            db: Stage3Database,
            conversation_id: str,
            order_index: int
    ) -> tuple[bool, dict]:
        """
        Check if assertion has user corroboration.

        Only applies to assistant assertions.

        Returns:
            (is_corroborated, details_dict)

        """
        if assertion.asserted_role != "assistant":
            return False, {}

        # Get candidate corroborating assertions
        candidates = db.get_assertions_for_corroboration(
            conversation_id=conversation_id,
            target_order_index=order_index,
            window_size=self.config.coref_window_size,
            subject_entity_id=assertion.subject_entity_id
        )

        for cand in candidates:
            # Check predicate match
            pred_match = self._predicate_matches(
                assertion.predicate_id,
                cand["predicate_id"],
                cand["canonical_label_norm"]
            )

            if not pred_match:
                continue

            # Check object compatibility
            if assertion.object_signature == cand["object_signature"]:
                return True, {
                    "corroborating_assertion_id": cand["assertion_id"],
                    "match_type": "exact",
                    "predicate_match": pred_match
                }

        return False, {}

    def _predicate_matches(
            self,
            pred_id1: str,
            pred_id2: str,
            pred_label_norm: str
    ) -> str | None:
        """
        Check if predicates match.

        Returns match type or None.
        """
        if pred_id1 == pred_id2:
            return "exact_id"

        # Could add fuzzy predicate matching here
        return None


# ============================================================================
# CANONICALIZATION REFINER
# ============================================================================

class CanonicalizationRefiner:
    """
    Refines entity canonical names using role-weighted mention evidence.

    This implements Pass A of Stage 3.
    """

    def __init__(
            self,
            db: Stage3Database,
            id_generator: IDGenerator,
            config: Stage3Config
    ):
        self.db = db
        self.id_generator = id_generator
        self.config = config

    def run(self) -> tuple[str, int, int]:
        """
        Run canonicalization refinement.

        Returns:
            (run_id, entities_processed, names_changed)

        """
        started_at = TimestampUtils.now_utc()
        method = "role_weighted"

        run_id = self.id_generator.generate([
            "canon_run",
            started_at,
            method
        ])

        config_json = JCS.canonicalize({
            "method": method,
            "trust_weight_user": self.config.trust_weight_user,
            "trust_weight_assistant": self.config.trust_weight_assistant_uncorroborated
        })

        self.db.insert_canonicalization_run(
            run_id=run_id,
            method=method,
            config_json=config_json,
            started_at_utc=started_at
        )

        entities_processed = 0
        names_changed = 0

        # Get all active entities
        entities = self.db.get_all_active_entities()

        for entity in entities:
            entity_id = entity["entity_id"]
            current_name = entity["canonical_name"]

            # Skip SELF entity
            if entity["entity_key"] == "__SELF__":
                entities_processed += 1
                continue

            # Get mentions with roles
            mentions = self.db.get_entity_mentions_with_roles(entity_id)

            if not mentions:
                entities_processed += 1
                continue

            # Compute weighted surface frequencies
            surface_weights: dict[str, float] = {}
            surface_counts: dict[str, int] = {}
            surface_first_seen: dict[str, tuple] = {}

            for m in mentions:
                surface = m["surface_text"]
                if not surface:
                    continue

                # Weight by role
                role = m["role"]
                if role == "user":
                    weight = self.config.trust_weight_user
                elif role == "assistant":
                    weight = self.config.trust_weight_assistant_uncorroborated
                else:
                    weight = 0.5

                surface_weights[surface] = surface_weights.get(surface, 0.0) + weight
                surface_counts[surface] = surface_counts.get(surface, 0) + 1

                # Track first occurrence
                if surface not in surface_first_seen:
                    surface_first_seen[surface] = (
                        m["created_at_utc"] or "",
                        m["conversation_id"],
                        m["order_index"],
                        m["message_id"],
                        m["mention_id"],
                        surface
                    )

            if not surface_weights:
                entities_processed += 1
                continue

            # Select best name
            best_surface = max(
                surface_weights.keys(),
                key=lambda s: (
                    surface_weights[s],
                    surface_counts[s],
                    surface_first_seen[s]
                )
            )

            if best_surface != current_name:
                # Record history
                history_id = self.id_generator.generate([
                    "canon_hist",
                    entity_id,
                    run_id
                ])

                selection_json = JCS.canonicalize({
                    "previous": current_name,
                    "selected": best_surface,
                    "weighted_score": surface_weights[best_surface],
                    "unweighted_count": surface_counts[best_surface],
                    "all_surfaces": list(surface_weights.keys())
                })

                self.db.insert_canonical_name_history(
                    history_id=history_id,
                    entity_id=entity_id,
                    run_id=run_id,
                    previous_name=current_name,
                    canonical_name=best_surface,
                    selection_method=method,
                    confidence=min(surface_weights[best_surface] / sum(surface_weights.values()), 1.0),
                    selected_at_utc=TimestampUtils.now_utc(),
                    raw_selection_json=selection_json
                )

                # Update entity
                self.db.update_entity_canonical_name(entity_id, best_surface)
                names_changed += 1

            entities_processed += 1

        # Update run record
        completed_at = TimestampUtils.now_utc()
        self.db.update_canonicalization_run(
            run_id=run_id,
            completed_at_utc=completed_at,
            entities_processed=entities_processed,
            names_changed=names_changed,
            raw_stats_json=None
        )

        return run_id, entities_processed, names_changed


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class AssertionExtractionPipeline:
    """
    Stage 3: Assertion Extraction Layer pipeline.

    Implements four passes:
    - Pass A: Canonicalization Prep
    - Pass B: Per-message Extraction
    - Pass C: Candidate Grounding
    - Pass D: Persistence
    """

    def __init__(self, config: Stage3Config):
        self.config = config
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.db = Stage3Database(config.output_file_path)

        # Components
        self.rule_extractor = RuleBasedExtractor()
        self.llm_extractor = LLMExtractor(config) if config.enable_llm_assertion_extraction else None
        self.retraction_detector = RetractionDetector()
        self.corroboration_detector = CorroborationDetector(config)

        # Statistics
        self.stats = {
            "messages_processed": 0,
            "candidates_extracted": 0,
            "assertions_persisted": 0,
            "assertions_deduplicated": 0,
            "retractions_detected": 0,
            "entities_created": 0
        }

    def run(self):
        """Execute Stage 3 pipeline."""
        logger.info("Starting Stage 3: Assertion Extraction Layer")
        started_at = TimestampUtils.now_utc()

        # Initialize schema
        self.db.initialize_stage3_schema()

        # Begin transaction
        self.db.begin()

        try:
            # === PASS A: Canonicalization Prep ===
            logger.info("Pass A: Canonicalization refinement")
            refiner = CanonicalizationRefiner(self.db, self.id_generator, self.config)
            canon_run_id, entities_processed, names_changed = refiner.run()
            logger.info(f"  Processed {entities_processed} entities, changed {names_changed} names")

            # Initialize entity resolver (after canonicalization)
            entity_resolver = EntityResolver(self.db, self.id_generator, self.config)
            entity_resolver.load_entities()

            # Initialize grounding engine
            grounding_engine = GroundingEngine(
                self.db, self.id_generator, self.config, entity_resolver
            )

            # Track conversation for context
            current_conversation_id = None
            context_messages: list[dict] = []

            # LLM run tracking
            llm_run_id = None
            if self.config.enable_llm_assertion_extraction:
                llm_run_id = self.id_generator.generate([
                    "llm_run",
                    started_at,
                    self.config.llm_model_name
                ])

                llm_config = {
                    "model_name": self.config.llm_model_name,
                    "model_version": self.config.llm_model_version,
                    "temperature": self.config.llm_temperature,
                    "top_p": self.config.llm_top_p,
                    "seed": self.config.llm_seed,
                    "k_context": self.config.k_context,
                    "multi_run_count": self.config.llm_multi_run_count
                }

                self.db.insert_llm_extraction_run(
                    run_id=llm_run_id,
                    model_name=self.config.llm_model_name,
                    model_version=self.config.llm_model_version,
                    config_json=JCS.canonicalize(llm_config),
                    started_at_utc=started_at
                )

            # === PASS B, C, D: Message Processing ===
            logger.info("Pass B/C/D: Message extraction, grounding, and persistence")

            for msg in self.db.iter_messages_for_stage3():
                message_id = msg["message_id"]
                conversation_id = msg["conversation_id"]
                order_index = msg["order_index"]
                text_raw = msg["text_raw"]
                role = msg["role"]

                # Update context tracking
                if conversation_id != current_conversation_id:
                    current_conversation_id = conversation_id
                    context_messages = []

                # Parse exclusion ranges
                excluded_ranges = []
                if msg.get("code_fence_ranges_json"):
                    for r in json.loads(msg["code_fence_ranges_json"]):
                        excluded_ranges.append((r["char_start"], r["char_end"]))
                if msg.get("blockquote_ranges_json"):
                    for r in json.loads(msg["blockquote_ranges_json"]):
                        excluded_ranges.append((r["char_start"], r["char_end"]))

                # Check minimum text length after exclusions
                effective_length = len(text_raw)
                for start, end in excluded_ranges:
                    effective_length -= (end - start)

                if effective_length < self.config.min_text_length:
                    context_messages.append(msg)
                    context_messages = context_messages[-self.config.k_context:]
                    continue

                # === PASS B: Extraction ===
                candidates: list[AssertionCandidate] = []

                # Rule-based extraction
                rule_candidates = self.rule_extractor.extract(
                    text_raw, message_id, excluded_ranges
                )
                candidates.extend(rule_candidates)

                # LLM extraction (if enabled)
                if self.llm_extractor and llm_run_id:
                    context = MessageContext(
                        target_message=msg,
                        context_messages=context_messages.copy(),
                        context_entities=self.db.get_context_entities(
                            [m["message_id"] for m in context_messages] + [message_id]
                        ),
                        context_times=self.db.get_context_times(
                            [m["message_id"] for m in context_messages] + [message_id]
                        ),
                        masked_text=self._mask_code_fences(text_raw, excluded_ranges),
                        excluded_ranges=excluded_ranges
                    )

                    llm_candidates = self.llm_extractor.extract(
                        context, self.db, self.id_generator, llm_run_id
                    )

                    # Hybrid merge if both present
                    if rule_candidates and llm_candidates:
                        candidates = self._merge_candidates(rule_candidates, llm_candidates)
                    else:
                        candidates.extend(llm_candidates)

                self.stats["candidates_extracted"] += len(candidates)

                # === PASS C: Grounding ===
                grounded_assertions: list[GroundedAssertion] = []

                for candidate in candidates:
                    grounded = grounding_engine.ground(candidate, msg, text_raw)
                    if grounded:
                        # Check corroboration for assistant assertions
                        if grounded.asserted_role == "assistant":
                            is_corroborated, details = self.corroboration_detector.check_corroboration(
                                grounded, self.db, conversation_id, order_index
                            )

                            if is_corroborated:
                                grounded.has_user_corroboration = 1
                                grounded.corroboration_details = details
                                trust_weight = self.config.trust_weight_assistant_corroborated
                            else:
                                trust_weight = self.config.trust_weight_assistant_uncorroborated
                        else:
                            trust_weight = self.config.trust_weight_user

                        # Compute final confidence
                        grounded.confidence_final = max(0.0, min(1.0,
                                                                 grounded.confidence_extraction * trust_weight
                                                                 ))

                        # Update raw JSON with corroboration details
                        if grounded.corroboration_details:
                            raw = json.loads(grounded.raw_assertion_json)
                            raw["corroboration"] = grounded.corroboration_details
                            grounded.raw_assertion_json = JCS.canonicalize(raw)

                        grounded_assertions.append(grounded)

                # === PASS D: Persistence ===
                for assertion in grounded_assertions:
                    # Generate assertion ID
                    assertion_id = self.id_generator.generate([
                        "assertion",
                        assertion.assertion_key,
                        HashUtils.sha256_string(assertion.raw_assertion_json)
                    ])

                    # Check for existing assertion with same key
                    existing = self.db.get_assertion_by_key(assertion.assertion_key)

                    if existing:
                        # Upsert logic
                        if self.config.assertion_upsert_policy == "keep_highest_confidence":
                            if assertion.confidence_final > existing["confidence_final"] + 0.001:
                                self.db.update_assertion_confidence(
                                    assertion.assertion_key,
                                    assertion.confidence_final,
                                    assertion.raw_assertion_json
                                )
                                self.stats["assertions_persisted"] += 1
                            else:
                                self.stats["assertions_deduplicated"] += 1
                        elif self.config.assertion_upsert_policy == "keep_first":
                            self.stats["assertions_deduplicated"] += 1
                        # keep_all would require different schema
                    else:
                        self.db.insert_assertion(assertion, assertion_id)
                        self.stats["assertions_persisted"] += 1

                # Detect retractions (user messages only)
                if role == "user":
                    retraction_candidates = self.retraction_detector.detect(
                        text_raw, message_id, role, excluded_ranges
                    )

                    for rc in retraction_candidates:
                        self._process_retraction(rc, grounded_assertions)

                # Update context
                context_messages.append(msg)
                context_messages = context_messages[-self.config.k_context:]

                self.stats["messages_processed"] += 1

                if self.stats["messages_processed"] % 500 == 0:
                    logger.info(f"  Processed {self.stats['messages_processed']} messages...")

            # Refresh predicate stats
            logger.info("Refreshing predicate statistics")
            self.db.refresh_predicate_stats()

            # Update LLM run if used
            if llm_run_id:
                self.db.update_llm_extraction_run(
                    run_id=llm_run_id,
                    completed_at_utc=TimestampUtils.now_utc(),
                    messages_processed=self.stats["messages_processed"],
                    assertions_extracted=self.stats["assertions_persisted"],
                    raw_stats_json=JCS.canonicalize(self.stats)
                )

            # Commit transaction
            self.db.commit()

            completed_at = TimestampUtils.now_utc()
            logger.info("Stage 3 completed successfully")
            logger.info(f"  Messages processed: {self.stats['messages_processed']}")
            logger.info(f"  Candidates extracted: {self.stats['candidates_extracted']}")
            logger.info(f"  Assertions persisted: {self.stats['assertions_persisted']}")
            logger.info(f"  Assertions deduplicated: {self.stats['assertions_deduplicated']}")
            logger.info(f"  Retractions detected: {self.stats['retractions_detected']}")
            logger.info(f"  Duration: {started_at} -> {completed_at}")

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

    def _mask_code_fences(
            self,
            text: str,
            excluded_ranges: list[tuple[int, int]]
    ) -> str:
        """
        Mask code fences according to config strategy.

        whitespace_preserve_length: Replace with spaces, same length
        """
        if self.config.code_fence_masking != "whitespace_preserve_length":
            return text

        result = list(text)
        for start, end in excluded_ranges:
            for i in range(start, min(end, len(result))):
                if result[i] != "\n":
                    result[i] = " "

        return "".join(result)

    def _merge_candidates(
            self,
            rule_candidates: list[AssertionCandidate],
            llm_candidates: list[AssertionCandidate]
    ) -> list[AssertionCandidate]:
        """
        Merge rule-based and LLM candidates.

        Prefer rule-based for semantic duplicates with overlapping spans.
        """
        merged = []
        used_llm = set()

        for rc in rule_candidates:
            merged.append(rc)

            # Find overlapping LLM candidates
            for i, lc in enumerate(llm_candidates):
                if i in used_llm:
                    continue

                if self._candidates_overlap(rc, lc):
                    # Check semantic similarity
                    if self._candidates_semantically_similar(rc, lc):
                        # Keep rule-based, mark as hybrid
                        rc.extraction_method = ExtractionMethod.HYBRID
                        used_llm.add(i)

        # Add remaining LLM candidates
        for i, lc in enumerate(llm_candidates):
            if i not in used_llm:
                merged.append(lc)

        return merged

    def _candidates_overlap(
            self,
            c1: AssertionCandidate,
            c2: AssertionCandidate
    ) -> bool:
        """Check if candidates have overlapping spans."""
        if (c1.char_start is None or c1.char_end is None or
                c2.char_start is None or c2.char_end is None):
            return False

        return c1.char_start < c2.char_end and c2.char_start < c1.char_end

    def _candidates_semantically_similar(
            self,
            c1: AssertionCandidate,
            c2: AssertionCandidate
    ) -> bool:
        """Check if candidates are semantically equivalent."""
        # Same subject
        if c1.subject_ref.upper() != c2.subject_ref.upper():
            return False

        # Similar predicate
        p1 = PredicateNormalizer.normalize(c1.predicate_label)[1]
        p2 = PredicateNormalizer.normalize(c2.predicate_label)[1]

        pred_sim = StringSimilarity.levenshtein_ratio(p1, p2)
        if pred_sim < self.config.threshold_link_string_sim:
            return False

        # Same modality and polarity
        if c1.modality != c2.modality or c1.polarity != c2.polarity:
            return False

        return True

    def _process_retraction(
            self,
            rc: RetractionCandidate,
            recent_assertions: list[GroundedAssertion]
    ):
        """Process a retraction candidate."""
        # Generate retraction ID
        retraction_id = self.id_generator.generate([
            "retraction",
            rc.message_id,
            rc.char_start,
            rc.char_end
        ])

        # Try to find target
        target_assertion_id = None
        target_fact_key = None
        replacement_assertion_id = None

        # For corrections, try to link to replacement
        if rc.retraction_type == RetractionType.CORRECTION and rc.replacement_candidate:
            # The replacement would need to be grounded and persisted
            # This is a simplified implementation
            pass

        # Try to find target in recent assertions
        # This is a simplified heuristic - full implementation would
        # parse the retraction content more carefully
        if recent_assertions:
            # Just link to most recent for now
            # A full implementation would analyze the retraction text
            target = recent_assertions[-1] if recent_assertions else None
            if target:
                target_fact_key = target.fact_key

        self.db.insert_retraction(
            retraction_id=retraction_id,
            retraction_message_id=rc.message_id,
            target_assertion_id=target_assertion_id,
            target_fact_key=target_fact_key,
            retraction_type=rc.retraction_type,
            replacement_assertion_id=replacement_assertion_id,
            confidence=rc.confidence,
            char_start=rc.char_start,
            char_end=rc.char_end,
            surface_text=rc.surface_text,
            raw_retraction_json=JCS.canonicalize(rc.raw_data)
        )

        self.stats["retractions_detected"] += 1

# ============================================================================
# VALIDATION PILELINE
# ============================================================================

class ValidationSeverity(StrEnum):
    """Severity levels for validation issues."""
    ERROR = "ERROR"      # Blocks Stage 4, must be fixed
    WARNING = "WARNING"  # May cause issues, should investigate
    INFO = "INFO"        # Informational, no action needed


class ValidationCategory(StrEnum):
    """Categories of validation checks."""
    SCHEMA = "schema"
    DATA_INTEGRITY = "data_integrity"
    REFERENTIAL = "referential"
    STATISTICAL = "statistical"
    STAGE4_READINESS = "stage4_readiness"


# Statistical boundaries (configurable)
DEFAULT_BOUNDARIES = {
    # Minimum counts
    "min_conversations": 1,
    "min_messages": 1,
    "min_entities": 1,  # At least SELF
    "min_predicates": 0,  # May have no assertions
    "min_assertions": 0,  # May have no extractable assertions

    # Ratios (as fractions)
    "max_orphaned_mentions_ratio": 0.01,  # <1% orphaned mentions
    "max_invalid_fk_ratio": 0.0,  # 0% invalid foreign keys (strict)
    "min_self_entity_present": 1.0,  # SELF must exist

    # Coverage expectations
    "min_message_coverage_ratio": 0.0,  # Assertions may cover 0% of messages
    "max_message_coverage_ratio": 1.0,  # Up to 100%

    # Assertion quality
    "min_avg_confidence": 0.3,  # Average confidence >= 0.3
    "max_ungrounded_ratio": 0.05,  # <5% with NULL subject/predicate

    # Stage 4 readiness
    "min_assertions_with_timestamp": 0.0,  # May have no timestamps
    "max_assertions_without_modality": 0.0,  # All must have modality
}

@dataclass
class ValidationIssue:
    """A single validation issue."""
    severity: ValidationSeverity
    category: ValidationCategory
    code: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


@dataclass
class TableStats:
    """Statistics for a single table."""
    name: str
    row_count: int
    column_count: int
    null_counts: dict[str, int] = field(default_factory=dict)
    sample_values: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "null_counts": self.null_counts
        }


@dataclass
class ValidationReport:
    """Complete validation report."""
    database_path: str
    validation_timestamp: str
    is_valid: bool
    is_stage4_ready: bool

    # Counts by severity
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0

    # Issues
    issues: list[ValidationIssue] = field(default_factory=list)

    # Statistics
    table_stats: dict[str, TableStats] = field(default_factory=dict)
    summary_stats: dict[str, Any] = field(default_factory=dict)

    # Boundaries used
    boundaries: dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue):
        """Add an issue and update counts."""
        self.issues.append(issue)
        if issue.severity == ValidationSeverity.ERROR:
            self.error_count += 1
            self.is_valid = False
            if issue.category == ValidationCategory.STAGE4_READINESS:
                self.is_stage4_ready = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warning_count += 1
        else:
            self.info_count += 1

    def to_dict(self) -> dict:
        return {
            "database_path": self.database_path,
            "validation_timestamp": self.validation_timestamp,
            "is_valid": self.is_valid,
            "is_stage4_ready": self.is_stage4_ready,
            "summary": {
                "errors": self.error_count,
                "warnings": self.warning_count,
                "info": self.info_count,
                "total_issues": len(self.issues)
            },
            "issues": [i.to_dict() for i in self.issues],
            "table_stats": {k: v.to_dict() for k, v in self.table_stats.items()},
            "summary_stats": self.summary_stats,
            "boundaries": self.boundaries
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "=" * 70)
        print("STAGE 3 VALIDATION REPORT")
        print("=" * 70)
        print(f"Database: {self.database_path}")
        print(f"Timestamp: {self.validation_timestamp}")
        print("-" * 70)

        # Overall status
        if self.is_valid and self.is_stage4_ready:
            status = "✅ PASSED - Ready for Stage 4"
        elif self.is_valid:
            status = "⚠️  VALID but NOT Stage 4 Ready"
        else:
            status = "❌ FAILED - Validation errors found"

        print(f"Status: {status}")
        print(f"Errors: {self.error_count} | Warnings: {self.warning_count} | Info: {self.info_count}")
        print("-" * 70)

        # Key statistics
        if self.summary_stats:
            print("\nKEY STATISTICS:")
            for key, value in self.summary_stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        # Issues by category
        if self.issues:
            print("\nISSUES BY CATEGORY:")
            by_category: dict[str, list] = {}
            for issue in self.issues:
                cat = issue.category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(issue)

            for category, issues in sorted(by_category.items()):
                print(f"\n  [{category.upper()}]")
                for issue in issues[:5]:  # Show first 5
                    icon = "❌" if issue.severity == ValidationSeverity.ERROR else "⚠️" if issue.severity == ValidationSeverity.WARNING else "ℹ️"
                    print(f"    {icon} {issue.code}: {issue.message}")
                if len(issues) > 5:
                    print(f"    ... and {len(issues) - 5} more")

        print("\n" + "=" * 70)


class Stage3Validator:
    """
    Validates Stage 3 output and checks readiness for Stage 4.

    Performs comprehensive checks on:
    - Schema integrity (all tables and columns exist)
    - Data integrity (no invalid NULLs, valid JSON)
    - Referential integrity (foreign keys valid)
    - Statistical boundaries (counts within expected ranges)
    - Stage 4 readiness (required fields for temporal reasoning)
    """

    def __init__(
        self,
        database_path: Path,
        boundaries: dict[str, Any] | None = None,
        strict: bool = False
    ):
        """
        Initialize validator.

        Args:
            database_path: Path to the SQLite database
            boundaries: Custom statistical boundaries (merged with defaults)
            strict: If True, treat warnings as errors
        """
        self.database_path = database_path
        self.strict = strict
        self.connection: sqlite3.Connection | None = None

        # Merge custom boundaries with defaults
        self.boundaries = DEFAULT_BOUNDARIES.copy()
        if boundaries:
            self.boundaries.update(boundaries)

        # Initialize report
        self.report = ValidationReport(
            database_path=str(database_path),
            validation_timestamp=datetime.utcnow().isoformat() + "Z",
            is_valid=True,
            is_stage4_ready=True,
            boundaries=self.boundaries
        )

    def connect(self):
        """Open database connection."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")

        self.connection = sqlite3.connect(str(self.database_path))
        self.connection.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def validate(self) -> ValidationReport:
        """
        Run all validation checks.

        Returns:
            ValidationReport with all issues and statistics
        """
        logger.info(f"Starting validation of {self.database_path}")

        try:
            self.connect()

            # Run checks in order
            # self._check_schema_integrity()
            self._check_data_integrity()
            self._check_referential_integrity()
            self._check_statistical_boundaries()
            self._check_stage4_readiness()

            # Collect statistics
            # self._collect_table_stats()
            self._compute_summary_stats()

        except Exception as e:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SCHEMA,
                code="VALIDATION_FAILED",
                message=f"Validation failed with exception: {str(e)}",
                details={"exception_type": type(e).__name__}
            ))

        finally:
            self.close()

        logger.info(f"Validation complete: {self.report.error_count} errors, {self.report.warning_count} warnings")
        return self.report

    # ========================================================================
    # SCHEMA INTEGRITY CHECKS
    # ========================================================================

    def _check_schema_integrity(self):
        """Check that all required tables and columns exist."""
        logger.info("Checking schema integrity...")
        raise NotImplementedError("Schema integrity check is not implemented.")

    def _check_table_schema(
        self,
        table: str,
        expected_columns: list[str],
        existing_tables: set[str],
        stage: str
    ):
        """Check a single table's schema."""
        if table not in existing_tables:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SCHEMA,
                code="MISSING_TABLE",
                message=f"{stage} table '{table}' does not exist",
                details={"table": table, "stage": stage}
            ))
            return

        # Get actual columns
        cursor = self.connection.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        actual_columns = {row['name'] for row in cursor}

        # Check for missing columns
        missing = set(expected_columns) - actual_columns
        if missing:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.SCHEMA,
                code="MISSING_COLUMNS",
                message=f"Table '{table}' missing columns: {missing}",
                details={"table": table, "missing_columns": list(missing)}
            ))

    # ========================================================================
    # DATA INTEGRITY CHECKS
    # ========================================================================

    def _check_data_integrity(self):
        """Check data integrity within tables."""
        logger.info("Checking data integrity...")

        cursor = self.connection.cursor()

        # Check assertions table
        self._check_assertions_integrity(cursor)

        # Check predicates table
        self._check_predicates_integrity(cursor)

        # Check retractions table
        self._check_retractions_integrity(cursor)

        # Check JSON validity
        self._check_json_validity(cursor)

    def _check_assertions_integrity(self, cursor):
        """Check assertions table integrity."""
        # Check for NULL required fields
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions
            WHERE subject_entity_id IS NULL
        """)
        null_subjects = cursor.fetchone()['cnt']
        if null_subjects > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                code="NULL_SUBJECT",
                message=f"{null_subjects} assertions have NULL subject_entity_id",
                details={"count": null_subjects}
            ))

        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions
            WHERE predicate_id IS NULL
        """)
        null_predicates = cursor.fetchone()['cnt']
        if null_predicates > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                code="NULL_PREDICATE",
                message=f"{null_predicates} assertions have NULL predicate_id",
                details={"count": null_predicates}
            ))

        # Check modality values
        cursor.execute("""
            SELECT DISTINCT modality FROM assertions
        """)
        modalities = [row['modality'] for row in cursor]
        valid_modalities = {'state', 'fact', 'preference', 'intention', 'question'}
        invalid = set(modalities) - valid_modalities
        if invalid:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATA_INTEGRITY,
                code="INVALID_MODALITY",
                message=f"Found invalid modality values: {invalid}",
                details={"invalid_values": list(invalid)}
            ))

        # Check polarity values
        cursor.execute("""
            SELECT DISTINCT polarity FROM assertions
        """)
        polarities = [row['polarity'] for row in cursor]
        valid_polarities = {'positive', 'negative'}
        invalid = set(polarities) - valid_polarities
        if invalid:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATA_INTEGRITY,
                code="INVALID_POLARITY",
                message=f"Found invalid polarity values: {invalid}",
                details={"invalid_values": list(invalid)}
            ))

        # Check confidence ranges
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions
            WHERE confidence_final < 0 OR confidence_final > 1
        """)
        invalid_conf = cursor.fetchone()['cnt']
        if invalid_conf > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                code="INVALID_CONFIDENCE",
                message=f"{invalid_conf} assertions have confidence outside [0,1]",
                details={"count": invalid_conf}
            ))

        # Check object model constraints
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions
            WHERE object_entity_id IS NOT NULL 
              AND object_value IS NOT NULL
        """)
        both_object = cursor.fetchone()['cnt']
        if both_object > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                code="DUAL_OBJECT",
                message=f"{both_object} assertions have both entity and literal objects",
                details={"count": both_object}
            ))

    def _check_predicates_integrity(self, cursor):
        """Check predicates table integrity."""
        # Check for duplicate normalized labels
        cursor.execute("""
            SELECT canonical_label_norm, COUNT(*) as cnt
            FROM predicates
            GROUP BY canonical_label_norm
            HAVING cnt > 1
        """)
        duplicates = cursor.fetchall()
        if duplicates:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.DATA_INTEGRITY,
                code="DUPLICATE_PREDICATE",
                message=f"Found {len(duplicates)} duplicate predicate labels",
                details={"duplicates": [dict(d) for d in duplicates]}
            ))

        # Check arity values
        cursor.execute("""
            SELECT DISTINCT arity FROM predicates
        """)
        arities = [row['arity'] for row in cursor]
        invalid = [a for a in arities if a not in (1, 2, 3)]
        if invalid:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATA_INTEGRITY,
                code="INVALID_ARITY",
                message=f"Found invalid arity values: {invalid}",
                details={"invalid_values": invalid}
            ))

    def _check_retractions_integrity(self, cursor):
        """Check retractions table integrity."""
        # Check retraction types
        cursor.execute("""
            SELECT DISTINCT retraction_type FROM retractions
        """)
        types = [row['retraction_type'] for row in cursor]
        valid_types = {'full', 'correction', 'temporal_bound'}
        invalid = set(types) - valid_types
        if invalid:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.DATA_INTEGRITY,
                code="INVALID_RETRACTION_TYPE",
                message=f"Found invalid retraction types: {invalid}",
                details={"invalid_values": list(invalid)}
            ))

    def _check_json_validity(self, cursor):
        """Spot-check JSON field validity."""
        json_fields = [
            ("assertions", "raw_assertion_json"),
            ("predicates", "raw_predicate_json"),
            ("retractions", "raw_retraction_json"),
        ]

        for table, column in json_fields:
            cursor.execute(f"""
                SELECT {column} FROM {table}
                WHERE {column} IS NOT NULL
                LIMIT 100
            """)

            invalid_count = 0
            for row in cursor:
                try:
                    if row[column]:
                        json.loads(row[column])
                except json.JSONDecodeError:
                    invalid_count += 1

            if invalid_count > 0:
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    category=ValidationCategory.DATA_INTEGRITY,
                    code="INVALID_JSON",
                    message=f"Found {invalid_count} invalid JSON values in {table}.{column}",
                    details={"table": table, "column": column, "count": invalid_count}
                ))

    # ========================================================================
    # REFERENTIAL INTEGRITY CHECKS
    # ========================================================================

    def _check_referential_integrity(self):
        """Check foreign key relationships."""
        logger.info("Checking referential integrity...")

        cursor = self.connection.cursor()

        # assertions.message_id -> messages.message_id
        self._check_fk(cursor, "assertions", "message_id", "messages", "message_id")

        # assertions.subject_entity_id -> entities.entity_id
        self._check_fk(cursor, "assertions", "subject_entity_id", "entities", "entity_id")

        # assertions.predicate_id -> predicates.predicate_id
        self._check_fk(cursor, "assertions", "predicate_id", "predicates", "predicate_id")

        # assertions.object_entity_id -> entities.entity_id (nullable)
        self._check_fk(cursor, "assertions", "object_entity_id", "entities", "entity_id", nullable=True)

        # assertions.temporal_qualifier_id -> time_mentions.time_mention_id (nullable)
        self._check_fk(cursor, "assertions", "temporal_qualifier_id", "time_mentions", "time_mention_id", nullable=True)

        # retractions.retraction_message_id -> messages.message_id
        self._check_fk(cursor, "retractions", "retraction_message_id", "messages", "message_id")

        # retractions.target_assertion_id -> assertions.assertion_id (nullable)
        self._check_fk(cursor, "retractions", "target_assertion_id", "assertions", "assertion_id", nullable=True)

        # entity_canonical_name_history.entity_id -> entities.entity_id
        self._check_fk(cursor, "entity_canonical_name_history", "entity_id", "entities", "entity_id")

        # entity_canonical_name_history.run_id -> entity_canonicalization_runs.run_id
        self._check_fk(cursor, "entity_canonical_name_history", "run_id", "entity_canonicalization_runs", "run_id")

    def _check_fk(
        self,
        cursor,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
        nullable: bool = False
    ):
        """Check a single foreign key relationship."""
        # ALWAYS exclude NULL source values - NULLs don't violate FK constraints
        # The 'nullable' flag only affects severity reporting, not the count logic
        null_clause = f"AND {source_table}.{source_column} IS NOT NULL"

        query = f"""
            SELECT COUNT(*) as cnt FROM {source_table}
            LEFT JOIN {target_table} ON {source_table}.{source_column} = {target_table}.{target_column}
            WHERE {target_table}.{target_column} IS NULL
            {null_clause}
        """

        try:
            cursor.execute(query)
            orphaned = cursor.fetchone()['cnt']

            if orphaned > 0:
                # Get total for ratio
                cursor.execute(f"SELECT COUNT(*) as cnt FROM {source_table}")
                total = cursor.fetchone()['cnt']
                ratio = orphaned / total if total > 0 else 0

                severity = ValidationSeverity.ERROR if ratio > self.boundaries["max_invalid_fk_ratio"] else ValidationSeverity.WARNING

                self.report.add_issue(ValidationIssue(
                    severity=severity,
                    category=ValidationCategory.REFERENTIAL,
                    code="ORPHANED_FK",
                    message=f"{orphaned} rows in {source_table}.{source_column} reference non-existent {target_table}.{target_column}",
                    details={
                        "source_table": source_table,
                        "source_column": source_column,
                        "target_table": target_table,
                        "target_column": target_column,
                        "orphaned_count": orphaned,
                        "total_count": total,
                        "ratio": ratio
                    }
                ))
        except sqlite3.OperationalError as e:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.REFERENTIAL,
                code="FK_CHECK_FAILED",
                message=f"Could not check FK {source_table}.{source_column}: {str(e)}",
                details={"error": str(e)}
            ))

    # ========================================================================
    # STATISTICAL BOUNDARY CHECKS
    # ========================================================================

    def _check_statistical_boundaries(self):
        """Check that counts and ratios are within expected boundaries."""
        logger.info("Checking statistical boundaries...")

        cursor = self.connection.cursor()

        # Check minimum counts
        self._check_min_count(cursor, "conversations", self.boundaries["min_conversations"])
        self._check_min_count(cursor, "messages", self.boundaries["min_messages"])
        self._check_min_count(cursor, "entities", self.boundaries["min_entities"])
        self._check_min_count(cursor, "predicates", self.boundaries["min_predicates"])
        self._check_min_count(cursor, "assertions", self.boundaries["min_assertions"])

        # Check SELF entity exists
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM entities
            WHERE entity_type = 'PERSON' AND entity_key = '__SELF__'
        """)
        self_count = cursor.fetchone()['cnt']
        if self_count < 1:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STATISTICAL,
                code="MISSING_SELF_ENTITY",
                message="Reserved SELF entity does not exist",
                details={}
            ))

        # Check average confidence
        cursor.execute("""
            SELECT AVG(confidence_final) as avg_conf FROM assertions
        """)
        row = cursor.fetchone()
        avg_conf = row['avg_conf'] if row['avg_conf'] is not None else 0
        if avg_conf < self.boundaries["min_avg_confidence"]:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.STATISTICAL,
                code="LOW_AVG_CONFIDENCE",
                message=f"Average assertion confidence ({avg_conf:.4f}) below minimum ({self.boundaries['min_avg_confidence']})",
                details={"average": avg_conf, "minimum": self.boundaries["min_avg_confidence"]}
            ))

        # Check predicate assertion counts match
        cursor.execute("""
            SELECT p.predicate_id, p.canonical_label, p.assertion_count as stored,
                   COUNT(a.assertion_id) as actual
            FROM predicates p
            LEFT JOIN assertions a ON p.predicate_id = a.predicate_id
            GROUP BY p.predicate_id
            HAVING stored != actual
        """)
        mismatches = cursor.fetchall()
        if mismatches:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category=ValidationCategory.STATISTICAL,
                code="PREDICATE_COUNT_MISMATCH",
                message=f"{len(mismatches)} predicates have incorrect assertion_count",
                details={"mismatches": [dict(m) for m in mismatches[:10]]}
            ))

    def _check_min_count(self, cursor, table: str, minimum: int):
        """Check minimum row count for a table."""
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        count = cursor.fetchone()['cnt']

        if count < minimum:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR if minimum > 0 else ValidationSeverity.WARNING,
                category=ValidationCategory.STATISTICAL,
                code="BELOW_MINIMUM_COUNT",
                message=f"Table '{table}' has {count} rows, minimum expected: {minimum}",
                details={"table": table, "count": count, "minimum": minimum}
            ))

    # ========================================================================
    # STAGE 4 READINESS CHECKS
    # ========================================================================

    def _check_stage4_readiness(self):
        """Check that data is ready for Stage 4 temporal reasoning."""
        logger.info("Checking Stage 4 readiness...")

        cursor = self.connection.cursor()

        # Check assertions have modality
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions WHERE modality IS NULL
        """)
        null_modality = cursor.fetchone()['cnt']
        if null_modality > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STAGE4_READINESS,
                code="NULL_MODALITY",
                message=f"{null_modality} assertions have NULL modality (required for Stage 4)",
                details={"count": null_modality}
            ))

        # Check assertions have polarity
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions WHERE polarity IS NULL
        """)
        null_polarity = cursor.fetchone()['cnt']
        if null_polarity > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STAGE4_READINESS,
                code="NULL_POLARITY",
                message=f"{null_polarity} assertions have NULL polarity (required for Stage 4)",
                details={"count": null_polarity}
            ))

        # Check assertions have object_signature
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions WHERE object_signature IS NULL
        """)
        null_sig = cursor.fetchone()['cnt']
        if null_sig > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STAGE4_READINESS,
                code="NULL_OBJECT_SIGNATURE",
                message=f"{null_sig} assertions have NULL object_signature (required for Stage 4 conflict detection)",
                details={"count": null_sig}
            ))

        # Check eligible assertions have timestamps (for valid time assignment)
        cursor.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN asserted_at_utc IS NOT NULL THEN 1 ELSE 0 END) as with_time
            FROM assertions
            WHERE modality IN ('state', 'fact', 'preference')
              AND polarity = 'positive'
        """)
        row = cursor.fetchone()
        total_eligible = row['total']
        with_time = row['with_time']

        if total_eligible > 0:
            time_ratio = with_time / total_eligible
            if time_ratio < 0.5:  # Less than 50% have timestamps
                self.report.add_issue(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    category=ValidationCategory.STAGE4_READINESS,
                    code="LOW_TIMESTAMP_COVERAGE",
                    message=f"Only {time_ratio*100:.1f}% of eligible assertions have timestamps",
                    details={
                        "total_eligible": total_eligible,
                        "with_timestamp": with_time,
                        "ratio": time_ratio
                    }
                ))

        # Check fact_key is populated
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions WHERE fact_key IS NULL OR fact_key = ''
        """)
        null_fact_key = cursor.fetchone()['cnt']
        if null_fact_key > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STAGE4_READINESS,
                code="NULL_FACT_KEY",
                message=f"{null_fact_key} assertions have NULL/empty fact_key (required for Stage 4 conflict detection)",
                details={"count": null_fact_key}
            ))

        # Check time_mentions are properly linked (if referenced)
        cursor.execute("""
            SELECT COUNT(*) as cnt FROM assertions
            WHERE temporal_qualifier_id IS NOT NULL
              AND temporal_qualifier_id NOT IN (SELECT time_mention_id FROM time_mentions)
        """)
        orphaned_time = cursor.fetchone()['cnt']
        if orphaned_time > 0:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STAGE4_READINESS,
                code="ORPHANED_TEMPORAL_QUALIFIER",
                message=f"{orphaned_time} assertions reference non-existent time_mentions",
                details={"count": orphaned_time}
            ))

        # Check assertion_key uniqueness
        cursor.execute("""
            SELECT assertion_key, COUNT(*) as cnt
            FROM assertions
            GROUP BY assertion_key
            HAVING cnt > 1
        """)
        duplicate_keys = cursor.fetchall()
        if duplicate_keys:
            self.report.add_issue(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category=ValidationCategory.STAGE4_READINESS,
                code="DUPLICATE_ASSERTION_KEY",
                message=f"Found {len(duplicate_keys)} duplicate assertion_keys (must be unique)",
                details={"count": len(duplicate_keys)}
            ))

    # ========================================================================
    # STATISTICS COLLECTION
    # ========================================================================

    def _compute_summary_stats(self):
        """Compute summary statistics across tables."""
        cursor = self.connection.cursor()

        stats = {}

        # Conversation/message counts
        cursor.execute("SELECT COUNT(*) FROM conversations")
        stats["total_conversations"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages")
        stats["total_messages"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages WHERE role = 'user'")
        stats["user_messages"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM messages WHERE role = 'assistant'")
        stats["assistant_messages"] = cursor.fetchone()[0]

        # Entity counts
        cursor.execute("SELECT COUNT(*) FROM entities")
        stats["total_entities"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM entities WHERE status = 'active'")
        stats["active_entities"] = cursor.fetchone()[0]

        cursor.execute("SELECT entity_type, COUNT(*) FROM entities GROUP BY entity_type")
        stats["entities_by_type"] = {row[0]: row[1] for row in cursor}

        # Mention counts
        cursor.execute("SELECT COUNT(*) FROM entity_mentions")
        stats["total_entity_mentions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM time_mentions")
        stats["total_time_mentions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM time_mentions WHERE resolved_type = 'instant'")
        stats["resolved_time_instants"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM time_mentions WHERE resolved_type = 'interval'")
        stats["resolved_time_intervals"] = cursor.fetchone()[0]

        # Assertion counts
        cursor.execute("SELECT COUNT(*) FROM assertions")
        stats["total_assertions"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM predicates")
        stats["total_predicates"] = cursor.fetchone()[0]

        cursor.execute("SELECT modality, COUNT(*) FROM assertions GROUP BY modality")
        stats["assertions_by_modality"] = {row[0]: row[1] for row in cursor}

        cursor.execute("SELECT polarity, COUNT(*) FROM assertions GROUP BY polarity")
        stats["assertions_by_polarity"] = {row[0]: row[1] for row in cursor}

        cursor.execute("SELECT asserted_role, COUNT(*) FROM assertions GROUP BY asserted_role")
        stats["assertions_by_role"] = {row[0]: row[1] for row in cursor}

        cursor.execute("SELECT extraction_method, COUNT(*) FROM assertions GROUP BY extraction_method")
        stats["assertions_by_method"] = {row[0]: row[1] for row in cursor}

        # Confidence stats
        cursor.execute("SELECT AVG(confidence_final), MIN(confidence_final), MAX(confidence_final) FROM assertions")
        row = cursor.fetchone()
        stats["confidence_avg"] = row[0]
        stats["confidence_min"] = row[1]
        stats["confidence_max"] = row[2]

        # Corroboration stats
        cursor.execute("SELECT SUM(has_user_corroboration), COUNT(*) FROM assertions WHERE asserted_role = 'assistant'")
        row = cursor.fetchone()
        corroborated = row[0] or 0
        total_assistant = row[1]
        stats["assistant_assertions_corroborated"] = corroborated
        stats["assistant_assertions_total"] = total_assistant
        stats["corroboration_ratio"] = corroborated / total_assistant if total_assistant > 0 else 0

        # Retraction counts
        cursor.execute("SELECT COUNT(*) FROM retractions")
        stats["total_retractions"] = cursor.fetchone()[0]

        cursor.execute("SELECT retraction_type, COUNT(*) FROM retractions GROUP BY retraction_type")
        stats["retractions_by_type"] = {row[0]: row[1] for row in cursor}

        # Coverage metrics
        cursor.execute("""
            SELECT COUNT(DISTINCT message_id) FROM assertions
        """)
        messages_with_assertions = cursor.fetchone()[0]
        stats["messages_with_assertions"] = messages_with_assertions
        stats["message_coverage_ratio"] = messages_with_assertions / stats["total_messages"] if stats["total_messages"] > 0 else 0

        # Top predicates
        cursor.execute("""
            SELECT p.canonical_label, COUNT(*) as cnt
            FROM assertions a
            JOIN predicates p ON a.predicate_id = p.predicate_id
            GROUP BY p.predicate_id
            ORDER BY cnt DESC
            LIMIT 10
        """)
        stats["top_predicates"] = [(row[0], row[1]) for row in cursor]

        self.report.summary_stats = stats

def validate_stage3(
    database_path: Path,
    boundaries: dict | None = None,
    strict: bool = False
) -> ValidationReport:
    """
    Validate Stage 3 output and check Stage 4 readiness.

    Args:
        database_path: Path to the SQLite database
        boundaries: Custom statistical boundaries
        strict: If True, treat warnings as errors

    Returns:
        ValidationReport with all issues and statistics
    """
    validator = Stage3Validator(database_path, boundaries, strict)
    return validator.validate()


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_stage3(config: Stage3Config) -> None:
    """Run Stage 3 pipeline on existing database."""
    pipeline = AssertionExtractionPipeline(config)
    # pipeline.run()

    report = validate_stage3(config.output_file_path, strict=True)
    report.print_summary()




if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 3: Assertion Extraction Layer")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: kg.db)"
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        default=False,
        help="Enable LLM-based assertion extraction (default: False)"
    )
    parser.add_argument(
        "--k-context",
        type=int,
        default=5,
        help="Number of context messages (default: 5)"
    )
    parser.add_argument(
        "--trust-user",
        type=float,
        default=1.0,
        help="Trust weight for user assertions (default: 1.0)"
    )
    parser.add_argument(
        "--trust-assistant-corroborated",
        type=float,
        default=0.8,
        help="Trust weight for corroborated assistant assertions (default: 0.8)"
    )
    parser.add_argument(
        "--trust-assistant-uncorroborated",
        type=float,
        default=0.4,
        help="Trust weight for uncorroborated assistant assertions (default: 0.4)"
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="String similarity threshold for fuzzy matching (default: 0.85)"
    )
    parser.add_argument(
        "--upsert-policy",
        type=str,
        choices=["keep_highest_confidence", "keep_first"],
        default="keep_highest_confidence",
        help="Assertion upsert policy (default: keep_highest_confidence)"
    )

    args = parser.parse_args()

    config = Stage3Config(
        output_file_path=args.db,
        enable_llm_assertion_extraction=args.enable_llm,
        k_context=args.k_context,
        trust_weight_user=args.trust_user,
        trust_weight_assistant_corroborated=args.trust_assistant_corroborated,
        trust_weight_assistant_uncorroborated=args.trust_assistant_uncorroborated,
        threshold_link_string_sim=args.similarity_threshold,
        assertion_upsert_policy=args.upsert_policy
    )

    run_stage3(config)
