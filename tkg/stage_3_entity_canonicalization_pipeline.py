"""
Stage 3: Entity Canonicalization Layer

Refines entity canonical names using detector-weighted, role-weighted mention evidence
from Stage 2, producing a fully auditable refinement trail without altering entity identity.
"""
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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


# ============================================================================
# DETECTOR RELIABILITY TIERS (§3.4.2)
# ============================================================================

# Tier 1 — Structured/Validated (highest reliability)
TIER1_DETECTORS: Set[str] = {
    "EMAIL", "URL", "DOI", "UUID", "IP_ADDRESS", "PHONE"
}

# Tier 2 — Pattern-based
TIER2_DETECTORS: Set[str] = {
    "HASH_HEX", "FILEPATH", "BARE_DOMAIN", "arXiv", "CVE", "ORCID", "HANDLE", "HASHTAG"
}

# Tier 3 — Learned/Lexicon: LEXICON:* (detected by prefix)
# Tier 4 — Statistical/NER: NER:* (detected by prefix)


def get_detector_tier(detector_name: str) -> int:
    """
    Determine the reliability tier for a detector.

    Tier 1: Structured/Validated (highest reliability)
    Tier 2: Pattern-based
    Tier 3: Learned/Lexicon (LEXICON:*)
    Tier 4: Statistical/NER (NER:* or unknown)

    Returns:
        Tier number (1-4), lower is more reliable
    """
    if detector_name in TIER1_DETECTORS:
        return 1
    elif detector_name in TIER2_DETECTORS:
        return 2
    elif detector_name.startswith("LEXICON:"):
        return 3
    elif detector_name.startswith("NER:"):
        return 4
    else:
        return 4  # unknown detectors treated as lowest tier


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Stage3Config:
    """Configuration for Stage 3 pipeline."""

    # Database path
    output_file_path: Path = field(default_factory=lambda: Path("kg.db"))

    # ID generation namespace
    id_namespace: str = "550e8400-e29b-41d4-a716-446655440000"

    # Trust weights for role-based canonicalization
    trust_weight_user: float = 1.0
    trust_weight_assistant_corroborated: float = 0.8
    trust_weight_assistant_uncorroborated: float = 0.5

    # Detector reliability weights (§3.7)
    detector_weight_tier1: float = 1.0   # Structured/Validated
    detector_weight_tier2: float = 0.9   # Pattern-based
    detector_weight_tier3: float = 0.8   # Lexicon
    detector_weight_tier4: float = 0.6   # NER/Statistical

    # Detector tier confidence bonus (§3.4.4)
    detector_tier_confidence_bonus: float = 0.05

    # Lexicon integration (§3.4.6)
    lexicon_canonical_bonus: float = 0.2

    # Processing options (§3.4.5)
    enable_salience_prioritization: bool = False

    # Canonicalization method
    canonicalization_method: str = "detector_role_weighted"

    # Optional LLM settings (for future hybrid methods)
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    def get_detector_weight(self, tier: int) -> float:
        """Get the weight multiplier for a detector tier."""
        if tier == 1:
            return self.detector_weight_tier1
        elif tier == 2:
            return self.detector_weight_tier2
        elif tier == 3:
            return self.detector_weight_tier3
        else:
            return self.detector_weight_tier4


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
        if self._in_transaction:
            raise RuntimeError("Already in transaction")
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
        """Execute and fetch one row."""
        cursor = self.execute(sql, params)
        return cursor.fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute and fetch all rows."""
        cursor = self.execute(sql, params)
        return cursor.fetchall()


class Stage3Database(Database):
    """Stage 3 specific database operations."""

    # Required tables from Stage 1 and Stage 2
    REQUIRED_TABLES = [
        "conversations",
        "messages",
        "message_parts",
        "entities",
        "entity_mentions",
    ]

    # Stage 3 schema
    STAGE3_SCHEMA = """
                    -- Entity canonicalization runs: one row per canonicalization execution
                    CREATE TABLE IF NOT EXISTS entity_canonicalization_runs (
                                                                                run_id TEXT PRIMARY KEY,
                                                                                method TEXT NOT NULL,
                                                                                model_name TEXT,
                                                                                model_version TEXT,
                                                                                config_json TEXT NOT NULL,
                                                                                started_at_utc TEXT NOT NULL,
                                                                                completed_at_utc TEXT,
                                                                                entities_processed INTEGER DEFAULT 0,
                                                                                names_changed INTEGER DEFAULT 0,
                                                                                raw_stats_json TEXT
                    );

                    -- Entity canonical name history: one row per canonical name change event
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
                        );

                    -- Indices for Stage 3 tables
                    CREATE INDEX IF NOT EXISTS idx_canonical_history_entity
                        ON entity_canonical_name_history(entity_id);
                    CREATE INDEX IF NOT EXISTS idx_canonical_history_run
                        ON entity_canonical_name_history(run_id);
                    """

    def check_required_tables(self) -> None:
        """Verify that required tables from previous stages exist."""
        cursor = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        existing_tables = {row[0] for row in cursor.fetchall()}

        missing = set(self.REQUIRED_TABLES) - existing_tables
        if missing:
            raise RuntimeError(
                f"Missing required tables from previous stages: {missing}. "
                "Ensure Stage 1 and Stage 2 have been run."
            )
        logger.info("All required tables present from previous stages")

    def initialize_stage3_schema(self) -> None:
        """Create Stage 3 tables if they don't exist."""
        self.connection.executescript(self.STAGE3_SCHEMA)
        logger.info("Stage 3 schema initialized")

    def get_active_entities_ordered(self, salience_prioritized: bool = False) -> List[sqlite3.Row]:
        """
        Get all active entities in deterministic order.

        Default order: (entity_type ASC, entity_key ASC, entity_id ASC)

        If salience_prioritized=True (§3.4.5):
            Order: (salience_score DESC NULLS LAST, entity_type ASC, entity_key ASC)
        """
        if salience_prioritized:
            return self.fetchall("""
                                 SELECT entity_id, entity_type, entity_key, canonical_name,
                                        aliases_json, status, first_seen_at_utc, last_seen_at_utc,
                                        mention_count, conversation_count,
                                        salience_score
                                 FROM entities
                                 WHERE status = 'active'
                                 ORDER BY salience_score DESC NULLS LAST,
                                          entity_type ASC,
                                          entity_key ASC,
                                          entity_id ASC
                                 """)
        else:
            return self.fetchall("""
                                 SELECT entity_id, entity_type, entity_key, canonical_name,
                                        aliases_json, status, first_seen_at_utc, last_seen_at_utc,
                                        mention_count, conversation_count
                                 FROM entities
                                 WHERE status = 'active'
                                 ORDER BY entity_type ASC, entity_key ASC, entity_id ASC
                                 """)

    def get_mentions_for_entity(self, entity_id: str) -> List[sqlite3.Row]:
        """
        Get all emitted mentions for an entity with message role and detector info.
        Returns mentions joined with message role, timestamp, and detector info.

        Updated to include detector field per §3.4.3.
        """
        return self.fetchall("""
                             SELECT
                                 em.mention_id,
                                 em.message_id,
                                 em.surface_text,
                                 em.confidence,
                                 em.detector,
                                 m.role,
                                 m.created_at_utc,
                                 m.conversation_id,
                                 m.order_index
                             FROM entity_mentions em
                                      JOIN messages m ON em.message_id = m.message_id
                             WHERE em.entity_id = ?
                             ORDER BY m.created_at_utc NULLS LAST,
                                      m.conversation_id ASC,
                                      m.order_index ASC,
                                      m.message_id ASC,
                                      em.mention_id ASC
                             """, (entity_id,))

    def get_lexicon_term(self, term_key: str) -> Optional[sqlite3.Row]:
        """
        Get lexicon term information for CUSTOM_TERM entities (§3.4.6).

        Returns lexicon term row if found, None otherwise.
        """
        # Check if lexicon_terms table exists first
        cursor = self.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='lexicon_terms'"
        )
        if not cursor.fetchone():
            return None

        return self.fetchone("""
                             SELECT term_id, term_key, canonical_surface, score
                             FROM lexicon_terms
                             WHERE term_key = ?
                             """, (term_key,))

    def insert_canonicalization_run(
            self,
            run_id: str,
            method: str,
            model_name: Optional[str],
            model_version: Optional[str],
            config_json: str,
            started_at_utc: str,
            completed_at_utc: Optional[str],
            entities_processed: int,
            names_changed: int,
            raw_stats_json: Optional[str]
    ) -> None:
        """Insert a canonicalization run record."""
        self.execute("""
                     INSERT INTO entity_canonicalization_runs (
                         run_id, method, model_name, model_version, config_json,
                         started_at_utc, completed_at_utc, entities_processed,
                         names_changed, raw_stats_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         run_id, method, model_name, model_version, config_json,
                         started_at_utc, completed_at_utc, entities_processed,
                         names_changed, raw_stats_json
                     ))

    def update_canonicalization_run(
            self,
            run_id: str,
            completed_at_utc: str,
            entities_processed: int,
            names_changed: int,
            raw_stats_json: str
    ) -> None:
        """Update a canonicalization run record with final stats."""
        self.execute("""
                     UPDATE entity_canonicalization_runs
                     SET completed_at_utc = ?,
                         entities_processed = ?,
                         names_changed = ?,
                         raw_stats_json = ?
                     WHERE run_id = ?
                     """, (
                         completed_at_utc, entities_processed, names_changed,
                         raw_stats_json, run_id
                     ))

    def insert_canonical_name_history(
            self,
            history_id: str,
            entity_id: str,
            run_id: str,
            previous_name: Optional[str],
            canonical_name: str,
            selection_method: str,
            confidence: float,
            selected_at_utc: str,
            raw_selection_json: str
    ) -> None:
        """Insert a canonical name history record."""
        self.execute("""
                     INSERT INTO entity_canonical_name_history (
                         history_id, entity_id, run_id, previous_name, canonical_name,
                         selection_method, confidence, selected_at_utc, raw_selection_json
                     ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                     """, (
                         history_id, entity_id, run_id, previous_name, canonical_name,
                         selection_method, confidence, selected_at_utc, raw_selection_json
                     ))

    def update_entity_canonical_name(
            self,
            entity_id: str,
            canonical_name: str
    ) -> None:
        """Update the canonical name for an entity."""
        self.execute("""
                     UPDATE entities
                     SET canonical_name = ?
                     WHERE entity_id = ?
                     """, (canonical_name, entity_id))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SurfaceStats:
    """Statistics for a single surface form."""

    surface_text: str
    weighted_score: float
    unweighted_count: int
    best_detector_tier: int  # Added per §3.4.3: lowest (best) tier seen for this surface
    first_occurrence: Tuple[Optional[str], str, int, str, str]  # (created_at, conv_id, order_idx, msg_id, mention_id)
    detector_breakdown: Dict[str, float] = field(default_factory=dict)  # detector -> weight contribution

    def sort_key(self) -> Tuple:
        """
        Generate deterministic sort key for canonical name selection.

        Per §3.4.4:
        1. Primary sort: weighted_score DESC
        2. Tie-break 1: best_detector_tier ASC (prefer higher-reliability detector)
        3. Tie-break 2: unweighted_count DESC
        4. Tie-break 3: first occurrence tuple ASC (earliest wins)
        5. Tie-break 4: surface_text ASC (lexicographic)
        """
        created_at = self.first_occurrence[0] or "9999-99-99"  # NULLS LAST
        return (
            -self.weighted_score,           # DESC
            self.best_detector_tier,        # ASC (lower tier = more reliable)
            -self.unweighted_count,         # DESC
            created_at,
            self.first_occurrence[1],       # conversation_id
            self.first_occurrence[2],       # order_index
            self.first_occurrence[3],       # message_id
            self.first_occurrence[4],       # mention_id
            self.surface_text               # ASC
        )


@dataclass
class CanonicalNameSelection:
    """Result of canonical name selection for an entity."""

    entity_id: str
    previous_name: Optional[str]
    selected_name: str
    confidence: float
    best_detector_tier: int
    selection_details: Dict[str, Any]
    changed: bool
    lexicon_bonus_applied: bool = False


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class EntityCanonicalizationPipeline:
    """
    Stage 3: Entity Canonicalization Layer

    Refines entity canonical names using detector-weighted, role-weighted
    mention evidence from Stage 2, producing a fully auditable refinement trail.

    Phases:
    1. Initialize run - Begin transaction, generate run_id, record metadata
    2. Detector-role-weighted canonicalization - Process each entity deterministically
    3. Persist + commit - Finalize run record, compute stats, commit
    """

    def __init__(self, config: Stage3Config):
        self.config = config
        self.db = Stage3Database(config.output_file_path)
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))

        # Run state
        self.run_id: Optional[str] = None
        self.started_at_utc: Optional[str] = None
        self.entities_processed: int = 0
        self.names_changed: int = 0

        # Statistics tracking
        self.entities_by_type: Dict[str, int] = {}
        self.changes_by_type: Dict[str, int] = {}
        self.confidence_sum: float = 0.0
        self.confidence_count: int = 0
        self.surface_counts: List[int] = []

        # New statistics per §3.5.1
        self.detector_tier_wins: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        self.lexicon_bonus_applied_count: int = 0

    def run(self) -> Dict[str, Any]:
        """
        Execute Stage 3 pipeline.

        Returns:
            Dictionary containing run statistics
        """
        logger.info("Starting Stage 3: Entity Canonicalization Layer")

        # Check prerequisites
        self.db.check_required_tables()

        # Initialize schema
        self.db.initialize_stage3_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Phase 1: Initialize run
            self._phase1_initialize_run()

            # Phase 2: Detector-role-weighted canonicalization
            self._phase2_detector_role_weighted_canonicalization()

            # Phase 3: Persist and commit
            stats = self._phase3_persist_and_commit()

            logger.info("Stage 3 completed successfully")
            return stats

        except Exception as e:
            logger.error(f"Stage 3 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

    def _phase1_initialize_run(self) -> None:
        """
        Phase 1: Initialize run.

        - Generate run_id deterministically
        - Record started_at_utc, method, config snapshot
        - Initialize counters
        - Insert initial run record (required for foreign key constraints)
        """
        logger.info("Phase 1: Initializing canonicalization run")

        self.started_at_utc = TimestampUtils.now_utc()

        # Generate deterministic run_id
        # Formula: uuid5(KG_NS_UUID, JCS(["canon_run", started_at_utc, method]))
        self.run_id = self.id_generator.generate([
            "canon_run",
            self.started_at_utc,
            self.config.canonicalization_method
        ])

        # Initialize counters
        self.entities_processed = 0
        self.names_changed = 0

        # Build config snapshot with all parameters per §3.7
        config_snapshot = {
            "id_namespace": self.config.id_namespace,
            "trust_weight_user": self.config.trust_weight_user,
            "trust_weight_assistant_corroborated": self.config.trust_weight_assistant_corroborated,
            "trust_weight_assistant_uncorroborated": self.config.trust_weight_assistant_uncorroborated,
            "detector_weight_tier1": self.config.detector_weight_tier1,
            "detector_weight_tier2": self.config.detector_weight_tier2,
            "detector_weight_tier3": self.config.detector_weight_tier3,
            "detector_weight_tier4": self.config.detector_weight_tier4,
            "detector_tier_confidence_bonus": self.config.detector_tier_confidence_bonus,
            "lexicon_canonical_bonus": self.config.lexicon_canonical_bonus,
            "enable_salience_prioritization": self.config.enable_salience_prioritization,
            "canonicalization_method": self.config.canonicalization_method
        }

        # Insert initial run record (needed for foreign key constraints on history table)
        self.db.insert_canonicalization_run(
            run_id=self.run_id,
            method=self.config.canonicalization_method,
            model_name=self.config.model_name,
            model_version=self.config.model_version,
            config_json=JCS.canonicalize(config_snapshot),
            started_at_utc=self.started_at_utc,
            completed_at_utc=None,  # Will be updated in Phase 3
            entities_processed=0,    # Will be updated in Phase 3
            names_changed=0,         # Will be updated in Phase 3
            raw_stats_json=None      # Will be updated in Phase 3
        )

        logger.info(f"Initialized run {self.run_id}")

    def _phase2_detector_role_weighted_canonicalization(self) -> None:
        """
        Phase 2: Detector-role-weighted canonicalization.

        Process entities in deterministic order per §3.4.1:
        (entity_type ASC, entity_key ASC, entity_id ASC)

        Or if salience prioritization enabled (§3.4.5):
        (salience_score DESC NULLS LAST, entity_type ASC, entity_key ASC)

        For each entity:
        1. Collect emitted mentions from entity_mentions
        2. Group by surface_text
        3. Compute weighted score per surface based on role AND detector tier
        4. Track first occurrence and best detector tier per surface
        5. Select canonical name using deterministic algorithm
        6. Apply lexicon enhancement for CUSTOM_TERM entities
        7. Record changes
        """
        logger.info("Phase 2: Processing entities for detector-role-weighted canonicalization")

        # Get all active entities in appropriate order
        entities = self.db.get_active_entities_ordered(
            salience_prioritized=self.config.enable_salience_prioritization
        )
        total_entities = len(entities)
        logger.info(f"Found {total_entities} active entities to process")

        if self.config.enable_salience_prioritization:
            logger.info("Salience-adjusted processing enabled (§3.4.5)")

        for idx, entity_row in enumerate(entities):
            entity_id = entity_row["entity_id"]
            entity_type = entity_row["entity_type"]
            entity_key = entity_row["entity_key"]
            current_canonical = entity_row["canonical_name"]

            # Track by type
            self.entities_by_type[entity_type] = \
                self.entities_by_type.get(entity_type, 0) + 1

            # Process this entity
            selection = self._process_entity(
                entity_id=entity_id,
                entity_type=entity_type,
                entity_key=entity_key,
                current_canonical=current_canonical
            )

            self.entities_processed += 1

            # Record change if name changed
            if selection.changed:
                self._record_canonical_name_change(selection)
                self.names_changed += 1
                self.changes_by_type[entity_type] = \
                    self.changes_by_type.get(entity_type, 0) + 1

                # Track confidence for changed entities
                self.confidence_sum += selection.confidence
                self.confidence_count += 1

                # Track winning detector tier
                self.detector_tier_wins[selection.best_detector_tier] = \
                    self.detector_tier_wins.get(selection.best_detector_tier, 0) + 1

                # Track lexicon bonus usage
                if selection.lexicon_bonus_applied:
                    self.lexicon_bonus_applied_count += 1

            # Log progress periodically
            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{total_entities} entities")

        logger.info(
            f"Phase 2 complete: processed {self.entities_processed} entities, "
            f"{self.names_changed} names changed"
        )

    def _process_entity(
            self,
            entity_id: str,
            entity_type: str,
            entity_key: str,
            current_canonical: Optional[str]
    ) -> CanonicalNameSelection:
        """
        Process a single entity for canonical name selection.

        Args:
            entity_id: The entity's ID
            entity_type: The entity type (PERSON, ORG, etc.)
            entity_key: The entity's key (used for lexicon lookup)
            current_canonical: Current canonical name

        Returns:
            CanonicalNameSelection with the result
        """
        # Get all mentions for this entity
        mentions = self.db.get_mentions_for_entity(entity_id)

        if not mentions:
            # No mentions - keep current canonical name
            return CanonicalNameSelection(
                entity_id=entity_id,
                previous_name=current_canonical,
                selected_name=current_canonical or "",
                confidence=1.0,
                best_detector_tier=4,
                selection_details={
                    "reason": "no_mentions",
                    "surfaces": {}
                },
                changed=False
            )

        # Group mentions by surface_text and compute stats
        surface_stats = self._compute_surface_stats(mentions)

        # Track surface count distribution
        self.surface_counts.append(len(surface_stats))

        if not surface_stats:
            # No valid surfaces - keep current
            return CanonicalNameSelection(
                entity_id=entity_id,
                previous_name=current_canonical,
                selected_name=current_canonical or "",
                confidence=1.0,
                best_detector_tier=4,
                selection_details={
                    "reason": "no_valid_surfaces",
                    "surfaces": {}
                },
                changed=False
            )

        # Apply lexicon enhancement for CUSTOM_TERM entities (§3.4.6)
        lexicon_info = None
        lexicon_bonus_applied = False
        if entity_type == "CUSTOM_TERM":
            lexicon_info = self._apply_lexicon_enhancement(entity_key, surface_stats)
            if lexicon_info and lexicon_info.get("bonus_applied"):
                lexicon_bonus_applied = True

        # Select canonical name using deterministic algorithm
        sorted_surfaces = sorted(surface_stats.values(), key=lambda s: s.sort_key())
        winner = sorted_surfaces[0]

        # Compute confidence with detector tier bonus (§3.4.4)
        total_weighted = sum(s.weighted_score for s in surface_stats.values())
        base_confidence = winner.weighted_score / total_weighted if total_weighted > 0 else 1.0

        # Apply detector reliability bonus
        tier_bonus = (4 - winner.best_detector_tier) * self.config.detector_tier_confidence_bonus
        confidence = max(0.0, min(1.0, base_confidence + tier_bonus))  # clamp to [0, 1]

        # Build selection details for audit
        selection_details = {
            "surfaces": {
                s.surface_text: {
                    "weighted_score": s.weighted_score,
                    "unweighted_count": s.unweighted_count,
                    "best_detector_tier": s.best_detector_tier,
                    "detector_breakdown": s.detector_breakdown,
                    "first_occurrence": {
                        "created_at_utc": s.first_occurrence[0],
                        "conversation_id": s.first_occurrence[1],
                        "order_index": s.first_occurrence[2],
                        "message_id": s.first_occurrence[3],
                        "mention_id": s.first_occurrence[4]
                    }
                }
                for s in sorted_surfaces
            },
            "total_weighted_score": total_weighted,
            "winner_surface": winner.surface_text,
            "winner_detector_tier": winner.best_detector_tier,
            "base_confidence": base_confidence,
            "tier_bonus": tier_bonus,
            "final_confidence": confidence,
            "config": {
                "trust_weight_user": self.config.trust_weight_user,
                "trust_weight_assistant_uncorroborated": self.config.trust_weight_assistant_uncorroborated,
                "detector_weight_tier1": self.config.detector_weight_tier1,
                "detector_weight_tier2": self.config.detector_weight_tier2,
                "detector_weight_tier3": self.config.detector_weight_tier3,
                "detector_weight_tier4": self.config.detector_weight_tier4,
                "detector_tier_confidence_bonus": self.config.detector_tier_confidence_bonus
            }
        }

        # Add lexicon info to audit log if applicable (§3.4.6)
        if lexicon_info:
            selection_details["lexicon_enhancement"] = lexicon_info

        # Determine if changed
        new_canonical = winner.surface_text
        changed = new_canonical != current_canonical

        return CanonicalNameSelection(
            entity_id=entity_id,
            previous_name=current_canonical,
            selected_name=new_canonical,
            confidence=confidence,
            best_detector_tier=winner.best_detector_tier,
            selection_details=selection_details,
            changed=changed,
            lexicon_bonus_applied=lexicon_bonus_applied
        )

    def _compute_surface_stats(
            self,
            mentions: List[sqlite3.Row]
    ) -> Dict[str, SurfaceStats]:
        """
        Compute weighted statistics for each surface form.

        Implements §3.4.3 with detector-weighted, role-weighted scoring:
        - role_weight based on message role
        - detector_weight based on detector tier
        - confidence_factor = 0.5 + (0.5 * mention.confidence)
        - mention_weight = role_weight * detector_weight * confidence_factor

        Args:
            mentions: List of mention rows with role and detector info

        Returns:
            Dictionary mapping surface_text to SurfaceStats
        """
        stats: Dict[str, SurfaceStats] = {}

        for mention in mentions:
            surface_text = mention["surface_text"]

            # Skip NULL surfaces
            if surface_text is None:
                continue

            # Determine role weight based on message role
            role = mention["role"]
            if role == "user":
                role_weight = self.config.trust_weight_user
            elif role == "assistant":
                role_weight = self.config.trust_weight_assistant_uncorroborated
            else:
                # system, tool, unknown - use middle weight
                role_weight = 0.5

            # Determine detector weight based on detector tier (§3.4.2)
            detector = mention["detector"] or "UNKNOWN"
            detector_tier = get_detector_tier(detector)
            detector_weight = self.config.get_detector_weight(detector_tier)

            # Compute confidence factor from mention confidence (§3.4.3)
            mention_confidence = mention["confidence"] if mention["confidence"] is not None else 0.5
            confidence_factor = 0.5 + (0.5 * mention_confidence)

            # Compose weights (§3.4.3)
            mention_weight = role_weight * detector_weight * confidence_factor

            # Build first occurrence tuple for this mention
            occurrence = (
                mention["created_at_utc"],
                mention["conversation_id"],
                mention["order_index"],
                mention["message_id"],
                mention["mention_id"]
            )

            if surface_text not in stats:
                # Initialize stats for this surface
                stats[surface_text] = SurfaceStats(
                    surface_text=surface_text,
                    weighted_score=mention_weight,
                    unweighted_count=1,
                    best_detector_tier=detector_tier,
                    first_occurrence=occurrence,
                    detector_breakdown={detector: mention_weight}
                )
            else:
                # Update existing stats
                existing = stats[surface_text]
                existing.weighted_score += mention_weight
                existing.unweighted_count += 1

                # Track best (lowest) detector tier for this surface
                existing.best_detector_tier = min(existing.best_detector_tier, detector_tier)

                # Update detector breakdown
                existing.detector_breakdown[detector] = \
                    existing.detector_breakdown.get(detector, 0) + mention_weight

                # Update first occurrence if this is earlier
                if self._occurrence_is_earlier(occurrence, existing.first_occurrence):
                    existing.first_occurrence = occurrence

        return stats

    def _apply_lexicon_enhancement(
            self,
            entity_key: str,
            surface_stats: Dict[str, SurfaceStats]
    ) -> Optional[Dict[str, Any]]:
        """
        Apply lexicon term enhancement for CUSTOM_TERM entities (§3.4.6).

        1. Retrieve linked lexicon term by term_key = entity_key
        2. If lexicon_terms.canonical_surface matches a candidate surface,
           add lexicon_canonical_bonus to that surface's weighted_score

        Args:
            entity_key: The entity's key (used as term_key lookup)
            surface_stats: Current surface statistics to potentially modify

        Returns:
            Dictionary with lexicon enhancement info for audit log, or None
        """
        lexicon_term = self.db.get_lexicon_term(entity_key)

        if lexicon_term is None:
            return None

        lexicon_info = {
            "lexicon_term_id": lexicon_term["term_id"],
            "lexicon_score": lexicon_term["score"],
            "canonical_surface": lexicon_term["canonical_surface"],
            "bonus_applied": False,
            "bonus_amount": 0.0
        }

        canonical_surface = lexicon_term["canonical_surface"]

        # Check if canonical_surface matches any candidate
        if canonical_surface and canonical_surface in surface_stats:
            bonus = self.config.lexicon_canonical_bonus
            surface_stats[canonical_surface].weighted_score += bonus
            lexicon_info["bonus_applied"] = True
            lexicon_info["bonus_amount"] = bonus
            logger.debug(
                f"Applied lexicon bonus {bonus} to surface '{canonical_surface}' "
                f"for entity_key '{entity_key}'"
            )

        return lexicon_info

    def _occurrence_is_earlier(
            self,
            new: Tuple[Optional[str], str, int, str, str],
            existing: Tuple[Optional[str], str, int, str, str]
    ) -> bool:
        """
        Compare two occurrence tuples to determine if new is earlier.

        Tuple format: (created_at_utc, conversation_id, order_index, message_id, mention_id)

        NULL timestamps sort last.
        """
        new_ts, new_conv, new_order, new_msg, new_mention = new
        exist_ts, exist_conv, exist_order, exist_msg, exist_mention = existing

        # Handle NULL timestamps (NULLS LAST)
        if new_ts is None and exist_ts is not None:
            return False
        if new_ts is not None and exist_ts is None:
            return True

        # Compare timestamps
        if new_ts is not None and exist_ts is not None:
            if new_ts < exist_ts:
                return True
            if new_ts > exist_ts:
                return False

        # Timestamps equal (or both NULL), compare rest
        if new_conv < exist_conv:
            return True
        if new_conv > exist_conv:
            return False

        if new_order < exist_order:
            return True
        if new_order > exist_order:
            return False

        if new_msg < exist_msg:
            return True
        if new_msg > exist_msg:
            return False

        return new_mention < exist_mention

    def _record_canonical_name_change(self, selection: CanonicalNameSelection) -> None:
        """
        Record a canonical name change in the database.

        - Insert history record
        - Update entity's canonical_name
        """
        # Generate deterministic history_id
        # Formula: uuid5(KG_NS_UUID, JCS(["canon_hist", entity_id, run_id]))
        history_id = self.id_generator.generate([
            "canon_hist",
            selection.entity_id,
            self.run_id
        ])

        selected_at_utc = TimestampUtils.now_utc()

        # Insert history record
        self.db.insert_canonical_name_history(
            history_id=history_id,
            entity_id=selection.entity_id,
            run_id=self.run_id,
            previous_name=selection.previous_name,
            canonical_name=selection.selected_name,
            selection_method=self.config.canonicalization_method,
            confidence=selection.confidence,
            selected_at_utc=selected_at_utc,
            raw_selection_json=JCS.canonicalize(selection.selection_details)
        )

        # Update entity's canonical name
        self.db.update_entity_canonical_name(
            entity_id=selection.entity_id,
            canonical_name=selection.selected_name
        )

    def _phase3_persist_and_commit(self) -> Dict[str, Any]:
        """
        Phase 3: Persist and commit.

        - Finalize run record with completed_at_utc
        - Compute detailed statistics per §3.5.1
        - Update run record
        - Commit transaction

        Returns:
            Dictionary containing run statistics
        """
        logger.info("Phase 3: Persisting run record and committing")

        completed_at_utc = TimestampUtils.now_utc()

        # Compute statistics
        avg_confidence = (
            self.confidence_sum / self.confidence_count
            if self.confidence_count > 0
            else None
        )

        # Compute surface distribution histogram
        surface_distribution = {}
        for count in self.surface_counts:
            bucket = str(count)
            surface_distribution[bucket] = surface_distribution.get(bucket, 0) + 1

        # Build raw_stats per §3.5.1
        raw_stats = {
            "entities_by_type": self.entities_by_type,
            "changes_by_type": self.changes_by_type,
            "avg_confidence": avg_confidence,
            "surface_distribution": surface_distribution,
            "detector_tier_distribution": {
                str(tier): count for tier, count in self.detector_tier_wins.items()
            },
            "lexicon_bonus_applied_count": self.lexicon_bonus_applied_count
        }

        # Update run record with final stats
        self.db.update_canonicalization_run(
            run_id=self.run_id,
            completed_at_utc=completed_at_utc,
            entities_processed=self.entities_processed,
            names_changed=self.names_changed,
            raw_stats_json=JCS.canonicalize(raw_stats)
        )

        # Commit transaction
        self.db.commit()

        # Log summary
        logger.info(f"Run {self.run_id} committed successfully")
        logger.info(f"  Entities processed: {self.entities_processed}")
        logger.info(f"  Names changed: {self.names_changed}")
        if avg_confidence is not None:
            logger.info(f"  Average confidence: {avg_confidence:.4f}")
        logger.info(f"  Detector tier wins: {self.detector_tier_wins}")
        if self.lexicon_bonus_applied_count > 0:
            logger.info(f"  Lexicon bonuses applied: {self.lexicon_bonus_applied_count}")

        # Return stats
        return {
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": completed_at_utc,
            "entities_processed": self.entities_processed,
            "names_changed": self.names_changed,
            "entities_by_type": self.entities_by_type,
            "changes_by_type": self.changes_by_type,
            "avg_confidence": avg_confidence,
            "detector_tier_distribution": self.detector_tier_wins,
            "lexicon_bonus_applied_count": self.lexicon_bonus_applied_count
        }


# ============================================================================
# ENTRY POINTS
# ============================================================================

def run_stage3(config: Stage3Config) -> Dict[str, Any]:
    """
    Run Stage 3 pipeline on existing database.

    Args:
        config: Stage 3 configuration

    Returns:
        Dictionary containing run statistics
    """
    pipeline = EntityCanonicalizationPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 3: Entity Canonicalization Layer")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: kg.db)"
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default="550e8400-e29b-41d4-a716-446655440000",
        help="UUID namespace for ID generation"
    )
    parser.add_argument(
        "--trust-user",
        type=float,
        default=1.0,
        help="Trust weight for user role mentions (default: 1.0)"
    )
    parser.add_argument(
        "--trust-assistant",
        type=float,
        default=0.5,
        help="Trust weight for assistant role mentions (default: 0.5)"
    )
    parser.add_argument(
        "--detector-tier1",
        type=float,
        default=1.0,
        help="Weight for Tier 1 (Structured) detectors (default: 1.0)"
    )
    parser.add_argument(
        "--detector-tier2",
        type=float,
        default=0.9,
        help="Weight for Tier 2 (Pattern-based) detectors (default: 0.9)"
    )
    parser.add_argument(
        "--detector-tier3",
        type=float,
        default=0.8,
        help="Weight for Tier 3 (Lexicon) detectors (default: 0.8)"
    )
    parser.add_argument(
        "--detector-tier4",
        type=float,
        default=0.6,
        help="Weight for Tier 4 (NER) detectors (default: 0.6)"
    )
    parser.add_argument(
        "--tier-confidence-bonus",
        type=float,
        default=0.05,
        help="Per-tier confidence bonus (default: 0.05)"
    )
    parser.add_argument(
        "--lexicon-bonus",
        type=float,
        default=0.2,
        help="Bonus for lexicon-preferred surface (default: 0.2)"
    )
    parser.add_argument(
        "--salience-priority",
        action="store_true",
        help="Enable salience-adjusted processing order"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="detector_role_weighted",
        choices=["detector_role_weighted", "assertion_informed", "llm_assisted", "hybrid"],
        help="Canonicalization method (default: detector_role_weighted)"
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

    config = Stage3Config(
        output_file_path=args.db,
        id_namespace=args.namespace,
        trust_weight_user=args.trust_user,
        trust_weight_assistant_uncorroborated=args.trust_assistant,
        detector_weight_tier1=args.detector_tier1,
        detector_weight_tier2=args.detector_tier2,
        detector_weight_tier3=args.detector_tier3,
        detector_weight_tier4=args.detector_tier4,
        detector_tier_confidence_bonus=args.tier_confidence_bonus,
        lexicon_canonical_bonus=args.lexicon_bonus,
        enable_salience_prioritization=args.salience_priority,
        canonicalization_method=args.method
    )

    stats = run_stage3(config)

    print("\n" + "=" * 60)
    print("STAGE 3 SUMMARY: Entity Canonicalization Layer")
    print("=" * 60)
    print(f"Run ID:             {stats['run_id']}")
    print(f"Started:            {stats['started_at_utc']}")
    print(f"Completed:          {stats['completed_at_utc']}")
    print(f"Entities processed: {stats['entities_processed']}")
    print(f"Names changed:      {stats['names_changed']}")

    if stats["entities_by_type"]:
        print("\nEntities by type:")
        for etype, count in sorted(stats["entities_by_type"].items()):
            changed = stats["changes_by_type"].get(etype, 0)
            print(f"  {etype}: {count} ({changed} changed)")

    if stats["avg_confidence"] is not None:
        print(f"\nAverage confidence: {stats['avg_confidence']:.4f}")

    if stats["detector_tier_distribution"]:
        print("\nWinning surfaces by detector tier:")
        for tier, count in sorted(stats["detector_tier_distribution"].items()):
            if count > 0:
                print(f"  Tier {tier}: {count}")

    if stats["lexicon_bonus_applied_count"] > 0:
        print(f"\nLexicon bonuses applied: {stats['lexicon_bonus_applied_count']}")

    print("=" * 60)