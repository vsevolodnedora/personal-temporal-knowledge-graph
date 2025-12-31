"""
Builds a semantic-lossless, bitemporal personal knowledge graph from ChatGPT
conversations.json with conservative temporal anchoring and deterministic,
auditable transformations.
"""

from argparse import ArgumentParser
import hashlib
import json
import os.path
import re
import sqlite3
import uuid
from collections import defaultdict
from enum import StrEnum
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Any, TypedDict, Iterator, Optional

from jsonpointer import resolve_pointer, JsonPointerException
import pendulum
from datetime import datetime
import yaml

from tkg.logger import get_logger
logger = get_logger(__name__)

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

# ===| CONFIG |===

@dataclass
class PipelineConfig:
    """Configuration for pipeline."""
    input_file_path: Path | None = None
    output_file_path: Path | None = None
    export_mapping_path: Path | None = None
    id_namespace: str | None = "550e8400-e29b-41d4-a716-446655440000"

# ===| UTILS |===

@dataclass(slots=True)
class ExportMapping:
    """Defines JSON pointers and rules for parsing various ChatGPT export formats."""
    format_version: str = "1.0"

    conversation_id_path: str | None = None
    conversation_title_path: str | None = None
    conversation_created_path: str | None = None
    conversation_updated_path: str | None = None

    messages_path: str = "/mapping"
    messages_is_mapping: bool = True

    message_id_path: str | None = None
    message_role_path: str | None = None
    message_parent_path: str | None = None
    message_created_path: str | None = None
    message_content_path: str | None = None

    role_mapping: dict[str, str] = field(default_factory=dict)
    content_part_rules: list[Any] = field(default_factory=list)
    known_attachment_paths: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExportMapping":
        """Load and validate export mapping from YAML file."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(f"Expected YAML top-level mapping (dict), got {type(data).__name__}")

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportMapping":
        """Create from dictionary."""
        return cls(
            format_version=data.get("format_version", "1.0"),
            conversation_id_path=data.get("conversation_id_path"),
            conversation_title_path=data.get("conversation_title_path"),
            conversation_created_path=data.get("conversation_created_path"),
            conversation_updated_path=data.get("conversation_updated_path"),
            messages_path=data.get("messages_path") or "/mapping",
            messages_is_mapping=bool(data.get("messages_is_mapping", True)),
            message_id_path=data.get("message_id_path"),
            message_role_path=data.get("message_role_path"),
            message_parent_path=data.get("message_parent_path"),
            message_created_path=data.get("message_created_path"),
            message_content_path=data.get("message_content_path"),
            role_mapping=data.get("role_mapping") or {},
            content_part_rules=data.get("content_part_rules") or [],
            known_attachment_paths=data.get("known_attachment_paths") or [],
        )

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
            # Handle special float values
            if obj != obj:  # NaN
                raise ValueError("NaN is not allowed in JCS")
            if obj == float('inf') or obj == float('-inf'):
                raise ValueError("Infinity is not allowed in JCS")
            # Use repr for precise float representation, then clean up
            s = repr(obj)
            # Normalize scientific notation
            if 'e' in s or 'E' in s:
                s = s.lower()
            return s
        elif isinstance(obj, str):
            return JCS._escape_string(obj)
        elif isinstance(obj, (list, tuple)):
            items = ','.join(JCS._serialize(item) for item in obj)
            return f'[{items}]'
        elif isinstance(obj, dict):
            # Sort keys by UTF-16 code units (for ASCII, this is same as lexicographic)
            sorted_keys = sorted(obj.keys(), key=lambda k: k.encode('utf-16-be'))
            items = ','.join(
                f'{JCS._escape_string(k)}:{JCS._serialize(obj[k])}'
                for k in sorted_keys
            )
            return '{' + items + '}'
        else:
            # Try to convert to a basic type
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
                # Control characters
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
            # Unix timestamp (seconds since epoch) - int or float
            if isinstance(timestamp, (int, float)):
                dt = pendulum.from_timestamp(timestamp, tz="UTC")
                return dt.format(TimestampUtils.ISO_UTC_MILLIS)

            # Python datetime
            if isinstance(timestamp, datetime):
                dt = pendulum.instance(timestamp)
                # If naive datetime, assume source_tz (or UTC)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=pendulum.timezone(source_tz or "UTC"))
                return dt.in_timezone("UTC").format(TimestampUtils.ISO_UTC_MILLIS)

            # String timestamps (ISO / RFC3339 / some common formats)
            if isinstance(timestamp, str):
                s = timestamp.strip()
                # tz=... is used as the default for naive strings
                dt = pendulum.parse(s, tz=source_tz or "UTC", strict=False)
                return dt.in_timezone("UTC").format(TimestampUtils.ISO_UTC_MILLIS)

            return None
        except Exception:
            return None

    @staticmethod
    def parse_iso(iso_string: str) -> datetime | None:
        try:
            # Returns a pendulum.DateTime (subclass of datetime)
            return pendulum.parse(iso_string, strict=False)
        except Exception:
            return None

    @staticmethod
    def compare(ts1: str | None, ts2: str | None) -> int:
        # If you store ONLY canonical strings (YYYY...SSS Z), lexicographic compare is valid
        if ts1 is None and ts2 is None:
            return 0
        if ts1 is None:
            return 1
        if ts2 is None:
            return -1
        return -1 if ts1 < ts2 else (1 if ts1 > ts2 else 0)

class TimestampQuality(StrEnum):
    """Quality indicator for message timestamps."""
    ORIGINAL = "original"
    IMPUTED_PARENT = "imputed_parent"
    IMPUTED_PRIOR = "imputed_prior"

class PartType(StrEnum):
    """Message part type classification per export_mapping rules."""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OTHER = "other"

class SpanDict(TypedDict):
    """Half-open character span [char_start, char_end) in Unicode codepoints."""
    char_start: int
    char_end: int

class CodeFenceRange(TypedDict):
    """Detected code fence range with optional language."""
    char_start: int
    char_end: int
    language: str|None

class ContentType(StrEnum):
    """Message content type classification."""
    TEXT = "text"
    MIXED = "mixed"
    EMPTY = "empty"
    UNKNOWN = "unknown"

class JSONPointer:
    """RFC 6901 JSON Pointer implementation (wrapper around python-json-pointer)."""

    @staticmethod
    def resolve(pointer: str, document: Any) -> Any:
        # Preserve your existing rules for "" and leading "/"
        if not pointer:
            return document
        if not pointer.startswith("/"):
            raise ValueError(f"Invalid JSON pointer: {pointer}")

        try:
            return resolve_pointer(document, pointer)
        except JsonPointerException as e:
            # Preserve your "not found" contract as KeyError
            raise KeyError(str(e)) from e

    @staticmethod
    def resolve_safe(pointer: str, document: Any) -> Any:
        # Return None on any failure (including bad pointer)
        if not pointer:
            return document
        if not pointer.startswith("/"):
            return None

        # resolve_pointer supports a default value when missing
        return resolve_pointer(document, pointer, None)

    @staticmethod
    def validate(pointer: str) -> bool:
        # If you want *pure syntactic* validation without adding imports,
        # keep your current validate() implementation (it’s already small and fast).
        if pointer == "":
            return True
        if not pointer.startswith("/"):
            return False

        # Minimal syntactic check for "~" escapes (same logic you have today)
        i = 0
        while i < len(pointer):
            if pointer[i] == "~":
                if i + 1 >= len(pointer) or pointer[i + 1] not in ("0", "1"):
                    return False
                i += 2
            else:
                i += 1
        return True

# ===| DATABASE |===

class DatabaseStage1(Database):
    """Interface for the SQLite database."""

    STAGE1_TABLES = {
        "conversations": ["conversation_id", "export_conversation_id", "title", "created_at_utc", "updated_at_utc", "message_count", "raw_conversation_json"],
        "messages": [
            "message_id",
            "conversation_id",
            "role",
            "parent_id",
            "tree_path",
            "order_index",
            "created_at_utc",
            "timestamp_quality",
            "content_type",
            "text_raw",
            "text_part_map_json",
            "code_fence_ranges_json",
            "blockquote_ranges_json",
            "attachment_count",
            "raw_message_json",
        ],
        "message_parts": ["part_id", "message_id", "part_index", "part_type", "text_content", "mime_type", "file_path", "metadata_json", "raw_part_json"],
    }

    def __init__(self, database_path: Path):
        super().__init__(database_path)

    def initialize_schema_stage1(self):
        """Create all tables and indices."""
        cursor = self.connection.cursor()

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS conversations (
                                                                    conversation_id TEXT PRIMARY KEY,
                                                                    export_conversation_id TEXT,
                                                                    title TEXT,
                                                                    created_at_utc TEXT,
                                                                    updated_at_utc TEXT,
                                                                    message_count INTEGER NOT NULL DEFAULT 0,
                                                                    raw_conversation_json TEXT NOT NULL
                       )
                       """)

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS messages (
                                                               message_id TEXT PRIMARY KEY,
                                                               conversation_id TEXT NOT NULL,
                                                               role TEXT NOT NULL,
                                                               parent_id TEXT,
                                                               tree_path TEXT NOT NULL,
                                                               order_index INTEGER NOT NULL,
                                                               created_at_utc TEXT,
                                                               timestamp_quality TEXT NOT NULL,
                                                               content_type TEXT NOT NULL,
                                                               text_raw TEXT,
                                                               text_part_map_json TEXT,
                                                               code_fence_ranges_json TEXT,
                                                               blockquote_ranges_json TEXT,
                                                               attachment_count INTEGER NOT NULL DEFAULT 0,
                                                               raw_message_json TEXT NOT NULL,
                                                               FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                           )
                       """)

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS message_parts (
                                                                    part_id TEXT PRIMARY KEY,
                                                                    message_id TEXT NOT NULL,
                                                                    part_index INTEGER NOT NULL,
                                                                    part_type TEXT NOT NULL,
                                                                    text_content TEXT,
                                                                    mime_type TEXT,
                                                                    file_path TEXT,
                                                                    metadata_json TEXT,
                                                                    raw_part_json TEXT NOT NULL,
                                                                    FOREIGN KEY (message_id) REFERENCES messages(message_id)
                           )
                       """)

        # Create indices for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_order ON messages(conversation_id, order_index)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_parts_message ON message_parts(message_id)")

        self.connection.commit()

    # --- Conversations ---

    def insert_conversation(
            self,
            conversation_id: str,
            export_conversation_id: str|None,
            title: str|None,
            created_at_utc: str|None,
            updated_at_utc: str|None,
            message_count: int,
            raw_conversation_json: str,
    ):
        """Insert conversation record."""
        self.connection.execute("""
                                INSERT INTO conversations (
                                    conversation_id, export_conversation_id, title,
                                    created_at_utc, updated_at_utc, message_count, raw_conversation_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    conversation_id, export_conversation_id, title,
                                    created_at_utc, updated_at_utc, message_count, raw_conversation_json
                                ))

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        """Retrieve conversation by ID."""
        cursor = self.connection.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def iter_conversations(self) -> Iterator[dict[str, Any]]:
        """Iterate conversations in deterministic order."""
        cursor = self.connection.execute("SELECT * FROM conversations ORDER BY conversation_id")
        for row in cursor:
            yield dict(row)

    def update_conversation_message_count(self, conversation_id: str, count: int):
        """Update message count for conversation."""
        self.connection.execute(
                "UPDATE conversations SET message_count = ? WHERE conversation_id = ?",
                (count, conversation_id),
        )

    def get_conversations_count(self) -> int:
        cursor = self.connection.execute("SELECT COUNT(*) FROM conversations")
        return int(cursor.fetchone()[0])

    # --- Messages ---

    def insert_message(
            self,
            message_id: str,
            conversation_id: str,
            role: str,
            parent_id: str|None,
            tree_path: str,
            order_index: int,
            created_at_utc: str|None,
            timestamp_quality: str,
            content_type: str,
            text_raw: str|None,
            text_part_map_json: str|None,
            code_fence_ranges_json: str|None,
            blockquote_ranges_json: str|None,
            attachment_count: int,
            raw_message_json: str,
    ):
        """Insert message record."""
        self.connection.execute("""
                                INSERT INTO messages (
                                    message_id, conversation_id, role, parent_id, tree_path, order_index,
                                    created_at_utc, timestamp_quality, content_type, text_raw,
                                    text_part_map_json, code_fence_ranges_json, blockquote_ranges_json,
                                    attachment_count, raw_message_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    message_id, conversation_id, role, parent_id, tree_path, order_index,
                                    created_at_utc, timestamp_quality, content_type, text_raw,
                                    text_part_map_json, code_fence_ranges_json, blockquote_ranges_json,
                                    attachment_count, raw_message_json
                                ))

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        """Retrieve message by ID."""
        cursor = self.connection.execute(
            "SELECT * FROM messages WHERE message_id = ?", (message_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def iter_messages(self, conversation_id: str|None = None) -> Iterator[dict[str, Any]]:
        """Iterate messages in deterministic order."""
        if conversation_id:
            cursor = self.connection.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY order_index, message_id",
                (conversation_id,)
            )
        else:
            cursor = self.connection.execute(
                "SELECT * FROM messages ORDER BY conversation_id, order_index, message_id"
            )
        for row in cursor:
            yield dict(row)

    def iter_messages_for_tree_computation(self, conversation_id: str) -> Iterator[dict[str, Any]]:
        """Iterate messages for tree path computation."""
        cursor = self.connection.execute(
            "SELECT message_id, parent_id FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        for row in cursor:
            yield dict(row)

    def update_message_tree_path(self, message_id: str, tree_path: str, order_index: int):
        """Update tree path and order index."""
        self.connection.execute(
            "UPDATE messages SET tree_path = ?, order_index = ? WHERE message_id = ?",
            (tree_path, order_index, message_id)
        )

    def update_message_text_fields(
            self,
            message_id: str,
            text_raw: str|None,
            text_part_map_json: str|None,
            code_fence_ranges_json: str|None,
            blockquote_ranges_json: str|None,
    ):
        """Update text extraction fields."""
        self.connection.execute("""
                                UPDATE messages SET
                                                    text_raw = ?,
                                                    text_part_map_json = ?,
                                                    code_fence_ranges_json = ?,
                                                    blockquote_ranges_json = ?
                                WHERE message_id = ?
                                """, (text_raw, text_part_map_json, code_fence_ranges_json, blockquote_ranges_json, message_id))

    def compute_order_indices(self, conversation_id: str):
        """
        Assign deterministic order_index to messages.

        Sort by: (tree_path ASC, message_id ASC)
        Assign order_index = 0, 1, 2, ...
        """
        # Get messages sorted by tree_path, then message_id
        cursor = self.connection.execute("""
                                            SELECT message_id FROM messages
                                            WHERE conversation_id = ?
                                            ORDER BY tree_path, message_id
                                            """, (conversation_id,))

        for order_index, row in enumerate(cursor):
            self.connection.execute(
                "UPDATE messages SET order_index = ? WHERE message_id = ?",
                (order_index, row['message_id'])
            )

    def get_messages_count(self, conversation_id: str) -> int:
        """Get count of messages in conversation."""
        cursor = self.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,)
        )
        return cursor.fetchone()[0]

    def get_total_messages_count(self) -> int:
        cursor = self.connection.execute("SELECT COUNT(*) FROM messages")
        return int(cursor.fetchone()[0])

    # --- Message Parts ---

    def insert_message_part(
            self,
            part_id: str,
            message_id: str,
            part_index: int,
            part_type: str,
            text_content: str|None,
            mime_type: str|None,
            file_path: str|None,
            metadata_json: str|None,
            raw_part_json: str,
    ):
        """Insert message part record."""
        self.connection.execute("""
                                INSERT INTO message_parts (
                                    part_id, message_id, part_index, part_type, text_content,
                                    mime_type, file_path, metadata_json, raw_part_json
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    part_id, message_id, part_index, part_type, text_content,
                                    mime_type, file_path, metadata_json, raw_part_json
                                ))

    def iter_message_parts(self, message_id: str) -> Iterator[dict[str, Any]]:
        """Iterate parts in part_index order."""
        cursor = self.connection.execute(
            "SELECT * FROM message_parts WHERE message_id = ? ORDER BY part_index",
            (message_id,)
        )
        for row in cursor:
            yield dict(row)

    def get_total_text_chars(self) -> int:
        """
        Total number of characters across:
        - messages.text_raw
        - message_parts.text_content
        """
        cursor = self.connection.execute("""
            SELECT
                COALESCE((
                    SELECT SUM(LENGTH(text_raw)) FROM messages
                ), 0)
                +
                COALESCE((
                    SELECT SUM(LENGTH(text_content)) FROM message_parts
                ), 0)
        """)
        return int(cursor.fetchone()[0])

# ===| PIPELINES |===

class RawIngestStage:
    """Ingesting raw conversations.json into the database."""
    def __init__(self, database:DatabaseStage1, id_generator:IDGenerator, export_mapping:ExportMapping):
        self.db = database
        self.id_gen = id_generator
        self.export_mapping = export_mapping

    def _classify_part(self, part_data: dict[str, Any]) -> tuple[str, str|None, str|None, str|None, dict|None]:
        """Classify a content part of a message in conversations.json using rules."""

        # Special case: Handle 'thoughts' content type
        if isinstance(part_data, dict) and part_data.get("content_type") == "thoughts":
            thoughts_list = part_data.get("thoughts", [])
            text_parts = []
            for thought in thoughts_list:
                if isinstance(thought, dict):
                    if "summary" in thought:
                        text_parts.append(f"[{thought['summary']}]")
                    if "content" in thought:
                        text_parts.append(thought["content"])

            if text_parts:
                return (
                    str(PartType.TEXT.value),
                    str("\n\n".join(text_parts)),
                    None,
                    None,
                    {"thought_count": len(thoughts_list)},
                )


        # Try each rule in order
        for rule in self.export_mapping.content_part_rules:
            match_value = JSONPointer.resolve_safe(rule["match_path"], part_data)

            if match_value is not None:
                expected = rule["match_value"]

                # Check if value matches
                matches = False
                if expected == "*":
                    matches = True
                elif isinstance(expected, list):
                    matches = match_value in expected
                else:
                    matches = match_value == expected

                if matches:
                    # Extract fields according to rule
                    text_content = None
                    if rule.get("text_extract_path"):
                        extracted = JSONPointer.resolve_safe(
                            rule["text_extract_path"], part_data
                        )
                        if extracted is not None:
                            # Handle arrays (like ChatGPT's parts) by joining
                            if isinstance(extracted, list):
                                text_parts = []
                                for item in extracted:
                                    if isinstance(item, str):
                                        text_parts.append(item)
                                    elif isinstance(item, dict):
                                        # For structured items, try to extract text
                                        if "content" in item:
                                            text_parts.append(str(item["content"]))
                                        elif "text" in item:
                                            text_parts.append(str(item["text"]))
                                        else:
                                            # Use the first string value found
                                            for v in item.values():
                                                if isinstance(v, str):
                                                    text_parts.append(v)
                                                    break
                                    else:
                                        text_parts.append(str(item))
                                text_content = (
                                    "".join(text_parts) if text_parts else None
                                )
                            else:
                                text_content = str(extracted)

                    mime_type = None
                    if rule.get("mime_type_path"):
                        mime_type = JSONPointer.resolve_safe(
                            rule["mime_type_path"], part_data
                        )
                        if mime_type is not None:
                            mime_type = str(mime_type)

                    file_path = None
                    if rule.get("file_path_path"):
                        file_path = JSONPointer.resolve_safe(
                            rule["file_path_path"], part_data
                        )
                        if file_path is not None:
                            file_path = str(file_path)

                    # Collect metadata
                    metadata = {}
                    for meta_path in rule.get("metadata_paths", []):
                        meta_value = JSONPointer.resolve_safe(meta_path, part_data)
                        if meta_value is not None:
                            # Use last segment of path as key
                            key = meta_path.split("/")[-1]
                            metadata[key] = meta_value

                    return (
                        rule["part_type"],
                        text_content,
                        mime_type,
                        file_path,
                        metadata if metadata else None,
                    )

        # Default: try to extract text from common fields
        text_content = None
        for field in ["text", "content", "value", "parts"]:
            if field in part_data:
                value = part_data[field]
                if isinstance(value, str):
                    text_content = value
                    break
                elif isinstance(value, list):
                    # Join list items
                    text_parts = [str(item) for item in value if item]
                    text_content = "".join(text_parts) if text_parts else None
                    break

        if text_content is not None:
            return (str(PartType.TEXT.value), text_content, None, None, None)

        # Fallback to 'other'
        return (str(PartType.OTHER.value), None, None, None, None)

    def _normalize_role(self, role_raw: Any) -> str:
        """Normalize role using role_mapping."""
        if role_raw is None:
            return "unknown"

        role_str = str(role_raw).lower()

        # Check role_mapping
        if role_str in self.export_mapping.role_mapping:
            return str(self.export_mapping.role_mapping[role_str])

        logger.warning(f"Unknown role {role_raw}")
        return "unknown"

    def _process_message_parts(self, message_id: str, msg_data: dict[str, Any]):
        """Process message content parts."""
        # Get content container
        content = JSONPointer.resolve_safe(
            self.export_mapping.message_content_path, msg_data
        ) if self.export_mapping.message_content_path else None

        if content is None:
            logger.warning(f"No content found for message {message_id}")
            return

        # Handle different content structures
        parts: List[dict[str, Any]] = []

        if isinstance(content, str):
            # Simple text content
            parts = [{"type": "text", "text": content}]
        elif isinstance(content, dict):
            # Single part object or content_type/parts structure
            if "parts" in content:
                # ChatGPT format: {"content_type": "text", "parts": ["..."]}
                raw_parts = content.get("parts", [])
                content_type = content.get("content_type", "text")
                for part in raw_parts:
                    if isinstance(part, str):
                        parts.append({"type": content_type, "text": part})
                    elif isinstance(part, dict):
                        parts.append(part)
            else:
                # Single part
                parts = [content]
        elif isinstance(content, list):
            # Array of parts
            for item in content:
                if isinstance(item, str):
                    parts.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    parts.append(item)

        # Process each part
        attachment_count = 0
        for part_index, part_data in enumerate(parts):
            part_type, text_content, mime_type, file_path, metadata = self._classify_part(part_data)

            if file_path or mime_type:
                attachment_count += 1

            # Generate part ID
            part_id = self.id_gen.generate(["part", message_id, part_index])

            # Canonicalize raw JSON
            raw_json = JCS.canonicalize(part_data)

            self.db.insert_message_part(
                part_id=part_id,
                message_id=message_id,
                part_index=part_index,
                part_type=part_type,
                text_content=text_content,
                mime_type=mime_type,
                file_path=file_path,
                metadata_json=JCS.canonicalize(metadata) if metadata else None,
                raw_part_json=raw_json,
            )

        # Update attachment count
        if attachment_count > 0:
            self.db.connection.execute(
                "UPDATE messages SET attachment_count = ? WHERE message_id = ?",
                (attachment_count, message_id)
            )

    def _process_message(self, conversation_id: str, msg_data: dict[str, Any], mapping_key: str | None = None):
        """Process a single message."""
        message_content = (
            JSONPointer.resolve_safe(self.export_mapping.message_content_path, msg_data)
            if self.export_mapping.message_content_path
            else None
        )
        # Skip nodes with null message content - these are tree artifacts, not messages
        if message_content is None:
            logger.debug(f"Message content not found for conv_id: {conversation_id} msg_data.keys(): {list(msg_data.keys())}. Skipping...")
            return

        # Extract message fields using export mapping
        export_msg_id = JSONPointer.resolve_safe(
            self.export_mapping.message_id_path, msg_data
        ) if self.export_mapping.message_id_path else None

        # Use mapping key as ID if available and no explicit ID
        if export_msg_id is None and mapping_key is not None:
            export_msg_id = mapping_key

        role_raw = JSONPointer.resolve_safe(
            self.export_mapping.message_role_path, msg_data
        ) if self.export_mapping.message_role_path else None

        created_raw = JSONPointer.resolve_safe(
            self.export_mapping.message_created_path, msg_data
        ) if self.export_mapping.message_created_path else None

        parent_export_id = (
            JSONPointer.resolve_safe(self.export_mapping.message_parent_path, msg_data)
            if self.export_mapping.message_parent_path
            else None
        )

        # Normalize role
        role = self._normalize_role(role_raw)

        # Normalize timestamp
        created_at_utc = TimestampUtils.normalize_to_utc(created_raw)
        timestamp_quality = TimestampQuality.ORIGINAL.value if created_at_utc else TimestampQuality.IMPUTED_PRIOR.value

        # Canonicalize raw JSON
        raw_json = JCS.canonicalize(msg_data)

        # Generate message ID - ALWAYS include conversation_id to ensure uniqueness
        if export_msg_id:
            message_id = self.id_gen.generate(["message", conversation_id, str(export_msg_id)])
        else:
            message_id = self.id_gen.generate(["message", conversation_id, HashUtils.sha256_string(raw_json)])

        # Convert parent export ID to UUID using same generation logic
        if parent_export_id:
            parent_message_id = self.id_gen.generate(["message", conversation_id, str(parent_export_id)])
        else:
            parent_message_id = None

        self.db.insert_message(
            message_id=message_id,
            conversation_id=conversation_id,
            role=role,
            parent_id=parent_message_id,
            tree_path="",  # Will be computed later
            order_index=0,  # Will be computed later
            created_at_utc=created_at_utc,
            timestamp_quality=str(timestamp_quality),
            content_type=str(ContentType.UNKNOWN.value), # inferred later
            text_raw=None,
            text_part_map_json=None,
            code_fence_ranges_json=None,
            blockquote_ranges_json=None,
            attachment_count=0,
            raw_message_json=raw_json,
        )

        # Process message parts
        self._process_message_parts(message_id, msg_data)

    def _process_conversation(self, conv_data:dict[str, Any]):
        """Process a single conversation."""

        export_conv_id = (
            JSONPointer.resolve_safe(self.export_mapping.conversation_id_path, conv_data)
            if self.export_mapping.conversation_id_path
            else None
        )

        title = (
            JSONPointer.resolve_safe(self.export_mapping.conversation_title_path, conv_data)
            if self.export_mapping.conversation_title_path
            else None
        )

        created_raw = (
            JSONPointer.resolve_safe(self.export_mapping.conversation_created_path, conv_data)
            if self.export_mapping.conversation_created_path
            else None
        )

        updated_raw = (
            JSONPointer.resolve_safe(self.export_mapping.conversation_updated_path, conv_data)
            if self.export_mapping.conversation_updated_path
            else None
        )

        raw_json = JCS.canonicalize(conv_data)

        # Normalize timestamps
        created_at_utc = TimestampUtils.normalize_to_utc(created_raw, source_tz="Europe/Berlin")
        updated_at_utc = TimestampUtils.normalize_to_utc(updated_raw, source_tz="Europe/Berlin")

        if export_conv_id:
            conversation_id = str(export_conv_id)
        else:
            conversation_id = self.id_gen.generate(["conversation", HashUtils.sha256_string(raw_json)])

        # Insert conversation
        self.db.insert_conversation(
            conversation_id=conversation_id,
            export_conversation_id=str(export_conv_id) if export_conv_id else None,
            title=str(title) if title else None,
            created_at_utc=created_at_utc,
            updated_at_utc=updated_at_utc,
            message_count=0,  # Will be updated after messages are processed
            raw_conversation_json=raw_json,
        )
        # Get messages container
        messages_container = JSONPointer.resolve_safe(
            self.export_mapping.messages_path, conv_data
        )
        if messages_container is None:
            logger.warning(f"No messages found for conversation. Conv_ID: {conversation_id}")

        # Process messages
        if self.export_mapping.messages_is_mapping:
            # Messages are in a mapping (dict) keyed by ID
            if isinstance(messages_container, dict):
                for i_msg, (msg_key, msg_data) in enumerate(messages_container.items()):
                    if msg_data is not None:
                        self._process_message(conversation_id, msg_data, msg_key)
                logger.info(f"Finished processing {len(messages_container.items())} DICT messages for conversation: {conversation_id}")
        else:
            # Messages are in an array
            if isinstance(messages_container, list):
                for i_msg, msg_data in enumerate(messages_container):
                    if msg_data is not None:
                        self._process_message(conversation_id, msg_data, None)
                logger.info(f"Finished processing {len(messages_container)} LIST messages for conversation: {conversation_id}")

    def _compute_tree_paths(self, conversation_id: str, tree_path_padding: int = 6):
        """
        Compute tree_path for all messages in conversation.

        Algorithm:
        1. Build parent→children adjacency
        2. Identify root messages (parent_id NULL)
        3. DFS from each root, assign tree_path
        """
        # Load all messages for this conversation
        messages = list(self.db.iter_messages_for_tree_computation(conversation_id))

        # Build adjacency
        children: dict[str|None, List[str]] = defaultdict(list)
        message_ids = set()

        for msg in messages:
            msg_id = msg['message_id']
            parent_id = msg['parent_id']
            message_ids.add(msg_id)
            children[parent_id].append(msg_id)

        # Sort children for determinism
        for parent in children:
            children[parent].sort()

        # Track visited for cycle detection
        visited = set()
        tree_paths: dict[str, str] = {}

        padding = tree_path_padding

        def dfs(msg_id: str, path: str):
            if msg_id in visited:
                logger.warning(f"Cycle detected at message {msg_id} | conversation_id: {conversation_id}")
                return

            visited.add(msg_id)
            tree_paths[msg_id] = path

            # Process children
            child_list = children.get(msg_id, [])
            for i, child_id in enumerate(child_list):
                child_segment = str(i + 1).zfill(padding)
                child_path = f"{path}.{child_segment}"
                dfs(child_id, child_path)

        # Find roots (parent_id is None or parent not in message_ids)
        roots = []
        for msg in messages:
            parent_id = msg['parent_id']
            if parent_id is None or parent_id not in message_ids:
                roots.append(msg['message_id'])

        roots.sort()  # Deterministic order

        # DFS from each root
        for i, root_id in enumerate(roots):
            root_path = str(i + 1).zfill(padding)
            dfs(root_id, root_path)

        # Handle orphans (messages whose parent is not in the set)
        orphan_index = len(roots)
        for msg in messages:
            msg_id = msg['message_id']
            if msg_id not in tree_paths:
                orphan_index += 1
                tree_paths[msg_id] = str(orphan_index).zfill(padding)
                logger.warning(f"Message {msg_id} has no valid parent chain: conversation_id = {conversation_id}")

        # Update database
        for msg_id, path in tree_paths.items():
            self.db.update_message_tree_path(msg_id, path, 0)  # order_index updated in next step

    def _compute_tree_paths_and_indices(self):
        """Compute tree paths and order indices for all conversations."""
        for conv in self.db.iter_conversations():
            self._compute_tree_paths(conv["conversation_id"])
            self.db.compute_order_indices(conv["conversation_id"])

            # Update message count
            count = self.db.get_messages_count(conv["conversation_id"])
            self.db.update_conversation_message_count(conv["conversation_id"], count)

    def _detect_code_fences(self, text: str) -> List[CodeFenceRange]:
        """
        Detect Markdown code fences (triple backticks).

        Returns list of ranges with optional language tag.
        """
        ranges = []

        # Pattern for code fence: ``` optionally followed by language
        pattern = re.compile(r'^(`{3,})(\w*)\s*$', re.MULTILINE)

        matches = list(pattern.finditer(text))

        i = 0
        while i < len(matches):
            open_match = matches[i]
            open_ticks = open_match.group(1)
            language = open_match.group(2) or None

            # Find matching close
            close_match = None
            for j in range(i + 1, len(matches)):
                candidate = matches[j]
                # Close must have same or more backticks and no language
                if len(candidate.group(1)) >= len(open_ticks) and not candidate.group(2):
                    close_match = candidate
                    i = j + 1
                    break

            if close_match:
                ranges.append({
                    "char_start": open_match.start(),
                    "char_end": close_match.end(),
                    "language": language
                })
            else:
                # Unclosed fence - extend to end of text
                ranges.append({
                    "char_start": open_match.start(),
                    "char_end": len(text),
                    "language": language
                })
                break

            if close_match is None:
                i += 1

        return ranges

    def _detect_blockquotes(self, text: str) -> List[SpanDict]:
        """
        Detect Markdown blockquote lines.

        A line is blockquote if it matches: ^\\s*>
        Returns full line ranges [line_start, line_end).
        """
        ranges = []

        pattern = re.compile(r'^\s*>.*$', re.MULTILINE)

        for match in pattern.finditer(text):
            ranges.append({
                "char_start": match.start(),
                "char_end": match.end()
            })

        return ranges

    def _extract_text(self, message_id: str):
        """
        Build text_raw from message parts.

        Rules:
        - Single text part: text_raw = part.text_content
        - Multiple parts: join with "\\n\\n", build text_part_map_json
        - No text parts: text_raw = NULL
        """
        parts = list(self.db.iter_message_parts(message_id))

        # Collect text parts
        text_parts: list[tuple[int, str]] = []
        for part in parts:
            if part['text_content']:
                text_parts.append((part['part_index'], part['text_content']))

        if not text_parts:
            # No text content
            self.db.connection.execute(
                "UPDATE messages SET content_type = ? WHERE message_id = ?",
                (ContentType.EMPTY.value if not parts else ContentType.UNKNOWN.value, message_id)
            )
            return

        if len(text_parts) == 1:
            # Single text part
            text_raw = text_parts[0][1]
            content_type = ContentType.TEXT.value
            text_part_map_json = None
        else:
            # Multiple parts - join with double newline
            segments = []
            part_map = []
            current_pos = 0

            for i, (part_index, text) in enumerate(text_parts):
                if i > 0:
                    segments.append("\n\n")
                    current_pos += 2

                start = current_pos
                segments.append(text)
                current_pos += len(text)

                part_map.append({
                    "part_index": part_index,
                    "char_start": start,
                    "char_end": current_pos
                })

            text_raw = "".join(segments)
            content_type = ContentType.MIXED.value
            text_part_map_json = JCS.canonicalize(part_map)

        # Detect code fences
        code_fence_ranges = self._detect_code_fences(text_raw)
        code_fence_ranges_json = JCS.canonicalize(code_fence_ranges) if code_fence_ranges else None

        # Detect blockquotes
        blockquote_ranges = self._detect_blockquotes(text_raw)
        blockquote_ranges_json = JCS.canonicalize(blockquote_ranges) if blockquote_ranges else None

        # Update message
        self.db.update_message_text_fields(
            message_id,
            text_raw,
            text_part_map_json,
            code_fence_ranges_json,
            blockquote_ranges_json,
        )

        self.db.connection.execute(
            "UPDATE messages SET content_type = ? WHERE message_id = ?",
            (content_type, message_id)
        )

    def run(self, input_file_path: Path):
        """Load the .json with conversations and ingest the data into the database."""
        logger.info(f"Loading {input_file_path}")
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both array and single conversation
        if isinstance(data, dict):
            data = [data]

        self.db.begin() # Opening transaction

        # Process each conversation
        for i_data, conv_data in enumerate(data):
            logger.info(f"Processing conversation {i_data}/{len(data)}")
            self._process_conversation(conv_data)

        conversation_count = self.db.get_conversations_count()
        total_messages = self.db.get_total_messages_count()
        avg_messages = (total_messages / conversation_count) if conversation_count else 0.0

        logger.info(f"Finished processing raw file with {conversation_count} conversations")
        logger.info(f"With average number of messages {avg_messages:.2f} per conversation")

        # Compute tree paths and order indices for all conversations
        self._compute_tree_paths_and_indices()

        # Extract text and detect code fences for all messages
        for msg in self.db.iter_messages():
            self._extract_text(msg["message_id"])

        total_text = self.db.get_total_text_chars()
        logger.info(f"Total chars {total_text} ")

        self.db.commit() # Commiting all transactions
        self.db.close()

class Pipeline:
    """Main pipeline orchestrator."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))

        if os.path.isfile(self.config.output_file_path): os.remove(self.config.output_file_path)
        self.db = DatabaseStage1(self.config.output_file_path)
        self.db.initialize_schema_stage1()

        self.export_mapping = ExportMapping.from_yaml(config.export_mapping_path)

        self.ingest_pipeline = RawIngestStage(self.db, self.id_generator, self.export_mapping)

    def run(self):
        logger.info(f"Running pipeline: {self.config.input_file_path}")

        # Perform ingestion
        self.ingest_pipeline.run(self.config.input_file_path)

        self.db = DatabaseStage1(self.config.output_file_path)
        total_text = self.db.get_total_text_chars()
        logger.info(f"Total chars {total_text} ")

def run_pipeline(input_file_path: Path, output_file_path: Path, export_mapping_path:Path) -> None:
    """Main pipeline orchestrator."""
    config = PipelineConfig(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        export_mapping_path=export_mapping_path,
    )

    pipeline = Pipeline(config)

    pipeline.run()

if __name__ == "__main__":
    parser = ArgumentParser(description="Run the knowledge graph pipeline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../data/raw/conversations.json"),
        help="Path to the input conversations JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the output SQLite DB file.",
    )
    parser.add_argument(
        "--export_mapping",
        type=Path,
        default=Path("../data/metadata/export_mapping.yaml"),
        help="Path to the export mapping YAML file.",
    )

    args = parser.parse_args()

    run_pipeline(
        input_file_path=args.input,
        output_file_path=args.output,
        export_mapping_path=args.export_mapping,
    )