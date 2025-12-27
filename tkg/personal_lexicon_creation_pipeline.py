"""
Stage 2: Personal Lexicon Layer

Detects entity mentions and time mentions with reliable spans,
resolves entity mentions deterministically into entities,
and keeps everything lossless, auditable, replayable, and offset-correct.

Assumes that the database "kg.db" exists with Stage 1 tables populated.
"""

import ipaddress
import json
import re
import sqlite3
import unicodedata
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum, IntEnum
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse, urlunparse

import pendulum

# Import shared utilities from extraction pipeline
from tkg.extraction_pipeline import (
    PipelineConfig,
    JCS,
    IDGenerator,
    TimestampUtils,
)
from tkg.database_base import Database
from tkg.hash_utils import HashUtils

# ===| LOGGING |===

from tkg.logger import get_logger
logger = get_logger(__name__)

# ===| CONFIGURATION |===

@dataclass
class Stage2Config(PipelineConfig):
    """Configuration for Stage 2 pipeline."""
    anchor_timezone: str = "UTC"

    # Entity detection
    ignore_markdown_blockquotes: bool = True  # Exclude blockquotes from entity detection
    domain_tld_allowlist_enabled: bool = True  # Require valid TLDs to prevent code pattern matches

    # NER Entity detection parameters
    enable_ner: bool = True  # Enable NER-based entity detection (only spacy model is supported for now)
    ner_max_chars: int = 10000
    ner_stride: int = 1000
    ner_label_allowlist: list = field(default_factory=lambda: ["PERSON", "ORG", "GPE", "LOC"])
    emit_spanless_ner: bool = False
    url_sort_query_params: bool = False

    # NER model config (when enabled)
    ner_model_name: str = "en_core_web_sm"
    ner_model_version: str = "3.7.0"


# ===| ENUMS |===

class SuppressionReason(StrEnum):
    """Reasons for suppressing a candidate."""
    OVERLAP_HIGHER_SCORE = "OVERLAP_HIGHER_SCORE"
    INTERSECTS_CODE_FENCE = "INTERSECTS_CODE_FENCE"
    INTERSECTS_BLOCKQUOTE = "INTERSECTS_BLOCKQUOTE"
    NO_OFFSETS_UNRELIABLE = "NO_OFFSETS_UNRELIABLE"
    INVALID_ENTITY_PATTERN = "INVALID_ENTITY_PATTERN"  # Code artifact or blocklisted pattern


class TimeResolvedType(StrEnum):
    """Time mention resolution type."""
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


# ===| DATA CLASSES |===

@dataclass
class TimeCandidate:
    """Raw time mention candidate."""
    message_id: str
    char_start: int
    char_end: int
    surface_text: str
    pattern_id: str
    pattern_precedence: int
    confidence: float
    raw_data: dict = field(default_factory=dict)

    @property
    def surface_hash(self) -> str:
        return HashUtils.sha256_string(self.surface_text)

    @property
    def span_length(self) -> int:
        return self.char_end - self.char_start


@dataclass
class ResolvedTime:
    """Resolved time mention."""
    resolved_type: str
    valid_from_utc: str | None
    valid_to_utc: str | None
    resolution_granularity: str | None
    timezone_assumed: str
    raw_parse: dict

# ===| DATA CLASSES |===

@dataclass
class EntityCandidate:
    """Raw entity detection candidate."""
    message_id: str
    detector: str
    detector_version: str
    entity_type_hint: str
    char_start: int | None
    char_end: int | None
    surface_text: str | None
    confidence: float
    raw_data: dict = field(default_factory=dict)
    rejected_as_code: bool = False
    code_rejection_reason: str | None = None

    @property
    def surface_hash(self) -> str:
        """Compute SHA-256 hash of surface text."""
        if self.surface_text is None:
            return HashUtils.sha256_string("__NO_SURFACE__")
        return HashUtils.sha256_string(self.surface_text)

    @property
    def span_length(self) -> int:
        """Compute span length, -1 if offsets are NULL."""
        if self.char_start is None or self.char_end is None:
            return -1
        return self.char_end - self.char_start


# ===| ENUMS |===

class EntityType(StrEnum):
    """Entity type taxonomy."""
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
    OTHER = "OTHER"


class DetectorOrder(IntEnum):
    """Order for detectors."""
    EMAIL = 1
    URL = 2
    DOI = 3
    UUID = 4
    HASH_HEX = 5
    IP_ADDRESS = 6
    PHONE = 7
    FILEPATH = 8
    BARE_DOMAIN = 9
    NER = 10

# ===| CODE POLLUTION FILTER |===

class CodeLanguageHint(StrEnum):
    """Detected code language hint."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    SHELL = "shell"
    LATEX = "latex"
    GENERIC = "generic"
    UNKNOWN = "unknown"


@dataclass
class CodeDetectionResult:
    """Result of code pollution detection."""

    is_code: bool
    confidence: float  # 0.0 to 1.0
    language_hint: CodeLanguageHint
    matched_pattern: str  # Which pattern triggered detection
    reason: str  # Human-readable explanation


class CodePollutionFilter:
    """
    Comprehensive filter for detecting and rejecting code artifacts
    that should not become entity mentions.
    """

    SQL_PHRASE_PATTERNS = [
        (re.compile(r"\b(create|drop|alter)\s+(table|index|view|database|dataframe)", re.I), "sql_ddl_phrase"),
        (re.compile(r"\b(select|insert|update|delete)\s+\w+", re.I), "sql_dml_phrase"),
        (re.compile(r"\bon\s+(delete|update)\s+(cascade|restrict|set null)", re.I), "sql_constraint_phrase"),
        (re.compile(r"\b(inner|outer|left|right|cross)\s+join\b", re.I), "sql_join_phrase"),
    ]

    # === EXPANDED EXACT BLOCKLIST ===
    EXACT_BLOCKLIST = {
        # Path/comment operators
        "//",
        "///",
        "..",
        "./",
        "../",
        "...",
        ".....",
        # Python/JS/C keywords
        "self",
        "cls",
        "this",
        "super",
        "class",
        "def",
        "async",
        "await",
        "function",
        "const",
        "let",
        "var",
        "return",
        "yield",
        "import",
        "export",
        "from",
        "as",
        "try",
        "catch",
        "finally",
        "throw",
        "new",
        "delete",
        "typeof",
        "instanceof",
        "void",
        "extends",
        "implements",
        "interface",
        "public",
        "private",
        "protected",
        "static",
        "readonly",
        "abstract",
        "enum",
        "namespace",
        "module",
        "declare",
        "type",
        "keyof",
        "infer",
        # Boolean/null literals
        "true",
        "false",
        "null",
        "none",
        "undefined",
        "nil",
        "nan",
        "inf",
        # HTML/XML fragments
        "div",
        "span",
        "href",
        "src",
        "onclick",
        "onload",
        "xmlns",
        # Common short tokens
        "id",
        "pk",
        "fk",
        "idx",
        "len",
        "str",
        "int",
        "num",
        "val",
        "obj",
        "arr",
        "ptr",
        "ref",
        "err",
        "msg",
        "ctx",
        "cfg",
        "env",
        "tmp",
        "buf",
        # React hooks
        "usestate",
        "useeffect",
        "useref",
        "usecallback",
        "usememo",
        "usecontext",
        "usereducer",
        "uselayouteffect",
        "useimperativehandle",
        "usedebugvalue",
        "usetransition",
        "useid",
        "usedeferredvalue",
        "useinsertioneffect",
        "usesyncexternalstore",
        # === NEW v1.3.0: Data formats ===
        "json",
        "xml",
        "csv",
        "html",
        "yaml",
        "toml",
        "ini",
        "pdf",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "svg",
        "mp4",
        "mp3",
        "wav",
        "zip",
        "tar",
        "gz",
        # === NEW v1.3.0: Common function/method names ===
        "convert",
        "parse",
        "format",
        "encode",
        "decode",
        "compress",
        "extract",
        "load",
        "save",
        "read",
        "write",
        "get",
        "set",
        "put",
        "post",
        "delete",
        "add",
        "remove",
        "insert",
        "update",
        "create",
        "destroy",
        "init",
        "run",
        "start",
        "stop",
        "open",
        "close",
        "send",
        "receive",
        "fetch",
        "push",
        "pop",
        "shift",
        "sort",
        "filter",
        "map",
        "reduce",
        "find",
        "search",
        "match",
        "replace",
        "split",
        "join",
        "merge",
        "clone",
        "copy",
        "move",
        "resize",
        "scale",
        "rotate",
        "transform",
        "render",
        "draw",
        "paint",
        "print",
        "log",
        "debug",
        "trace",
        "warn",
        "error",
        "info",
        "assert",
        "validate",
        "verify",
        "check",
        "test",
        "mock",
        "stub",
        "spy",
        # === NEW v1.3.0: Common variable names ===
        "data",
        "result",
        "output",
        "input",
        "value",
        "item",
        "element",
        "node",
        "list",
        "array",
        "dict",
        "map",
        "set",
        "queue",
        "stack",
        "tree",
        "graph",
        "blob",
        "buffer",
        "stream",
        "chunk",
        "block",
        "frame",
        "packet",
        "batch",
        "config",
        "options",
        "settings",
        "params",
        "args",
        "kwargs",
        "props",
        "state",
        "context",
        "store",
        "cache",
        "index",
        "count",
        "size",
        "length",
        "width",
        "height",
        "depth",
        "offset",
        "position",
        "location",
        "path",
        "name",
        "label",
        "title",
        "text",
        "content",
        "body",
        "header",
        "footer",
        "row",
        "col",
        "column",
        "field",
        "key",
        "value",
        "pair",
        "tuple",
        "min",
        "max",
        "avg",
        "sum",
        "mean",
        "median",
        "mode",
        "std",
        "var",
        "todo",
        "fixme",
        "hack",
        "note",
        "bug",
        "issue",
        "task",
        # === NEW v1.3.0: Technology terms ===
        "api",
        "sdk",
        "cli",
        "gui",
        "url",
        "uri",
        "sql",
        "css",
        "dom",
        "jwt",
        "oauth",
        "http",
        "https",
        "tcp",
        "udp",
        "ftp",
        "ssh",
        "ssl",
        "tls",
        "rest",
        "soap",
        "grpc",
        "graphql",
        "websocket",
        "ml",
        "ai",
        "nlp",
        "cv",
        "dl",
        "xgboost",
        "mlops",
        "tso",
        # === Timezones ===
        "utc",
        "gmt",
        "est",
        "pst",
        "cst",
        "cet",
        "edt",
        "pdt",
        "cdt",
    }

    # === Unit abbreviations ===
    UNIT_ABBREVIATIONS = {
        # Power/Energy
        "mw",
        "kw",
        "gw",
        "tw",
        "kwh",
        "mwh",
        "gwh",
        "twh",
        "wh",
        # Electrical
        "kv",
        "mv",
        "v",
        "ma",
        "ka",
        "a",
        "ohm",
        # Data
        "kb",
        "mb",
        "gb",
        "tb",
        "pb",
        "kbps",
        "mbps",
        "gbps",
        # Time
        "ms",
        "ns",
        "us",
        "μs",
        # Other measurements
        "kg",
        "mg",
        "km",
        "mm",
        "cm",
        "nm",
        "hz",
        "khz",
        "mhz",
        "ghz",
        "rpm",
        "fps",
        "dpi",
        "ppi",
        # Currency
        "eur",
        "usd",
        "gbp",
        "jpy",
        "cny",
        "chf",
    }

    # Code module prefixes
    CODE_MODULE_PREFIXES = {
        "pd",
        "np",
        "tf",
        "plt",
        "sns",
        "sk",
        "cv",
        "cv2",
        "scipy",
        "torch",
        "keras",
        "sklearn",
        "xgb",
        "lgb",
        "optuna",
        "ray",
        "dask",
        "os",
        "re",
        "sys",
        "io",
        "json",
        "time",
        "math",
        "random",
        "copy",
        "collections",
        "itertools",
        "functools",
        "operator",
        "typing",
        "pathlib",
        "datetime",
        "logging",
        "threading",
        "asyncio",
        "socket",
        "http",
        "urllib",
        "hashlib",
        "base64",
        "struct",
        "pickle",
        "sqlite3",
        "flask",
        "django",
        "fastapi",
        "starlette",
        "aiohttp",
        "requests",
        "httpx",
        "uvicorn",
        "gunicorn",
        "celery",
        "redis",
        "sqlalchemy",
        "console",
        "window",
        "document",
        "process",
        "require",
        "module",
        "exports",
        "Promise",
        "Object",
        "Array",
        "String",
        "Number",
        "Math",
        "JSON",
        "Date",
        "RegExp",
        "Error",
        "Buffer",
        "fs",
        "path",
        "util",
        "crypto",
        "stream",
        "events",
        "https",
        "net",
        "child_process",
        "React",
        "useState",
        "useEffect",
        "useRef",
        "useCallback",
        "useMemo",
        "Vue",
        "ref",
        "reactive",
        "computed",
        "watch",
        "Angular",
        "Component",
        "self",
        "cls",
        "this",
        "super",
        "base",
        "parent",
    }

    # Python patterns
    PYTHON_PATTERNS = [
        (re.compile(r"^__\w+__$"), "dunder_method"),
        (re.compile(r"^@\w+"), "decorator"),
        (re.compile(r"\{[^}]+\}"), "fstring_interpolation"),
        (re.compile(r"^(?:List|Dict|Set|Tuple|Optional|Union|Callable|Any|Type)\["), "type_annotation"),
        (re.compile(r"\blambda\s+\w+\s*:"), "lambda_expression"),
        (re.compile(r"\bfor\s+\w+\s+in\s+"), "comprehension"),
        (re.compile(r"^from\s+\w+\s+import\s+"), "import_statement"),
        (re.compile(r"^import\s+\w+"), "import_statement"),
        (re.compile(r"\braise\s+\w+"), "raise_statement"),
        (re.compile(r"\bexcept\s+\w+"), "except_clause"),
    ]

    # JavaScript patterns
    JS_PATTERNS = [
        (re.compile(r"=>"), "arrow_function"),
        (re.compile(r"\$\{[^}]+\}"), "template_literal"),
        (re.compile(r"\$\{[^}]*"), "template_literal"),  # Catches ${id without }
        (re.compile(r"`[^`]*\$\{"), "template_literal_start"),  # Catches `...${
        (re.compile(r"^\s*(?:const|let|var)\s*[\[{]"), "destructuring"),
        (re.compile(r"<[A-Z]\w+[\s/>]"), "jsx_component"),
        (re.compile(r"</[A-Z]\w+>"), "jsx_closing_tag"),
        (re.compile(r"\.\.\.\w+"), "spread_operator"),
        (re.compile(r"\w+\?\.\w+"), "optional_chaining"),
        (re.compile(r"\?\?"), "nullish_coalescing"),
        (re.compile(r"as\s+(?:string|number|boolean|any|unknown|never)\b"), "type_assertion"),
    ]

    # HTML patterns
    HTML_PATTERNS = [
        (re.compile(r"^<[a-z][a-z0-9]*(?:\s|>|/>)", re.IGNORECASE), "html_opening_tag"),
        (re.compile(r"^</[a-z][a-z0-9]*>", re.IGNORECASE), "html_closing_tag"),
        (re.compile(r"/\s*>$"), "self_closing_tag"),
        (re.compile(r'\w+\s*=\s*["\']'), "html_attribute"),
        (re.compile(r"^<[!?]", re.IGNORECASE), "declaration"),
        (re.compile(r"<!\[CDATA\["), "cdata"),
        (re.compile(r"&[a-z]+;|&#\d+;|&#x[0-9a-f]+;", re.IGNORECASE), "html_entity"),
    ]

    # Generic code patterns
    GENERIC_CODE_PATTERNS = [
        (re.compile(r"^[a-z][a-z0-9_]*\.[A-Z][a-zA-Z0-9_]*$"), "module_class_access"),
        (re.compile(r"\.\w+\([^)]*\)\.\w+"), "method_chain"),
        (re.compile(r"^\w+\s*\([^)]*\)$"), "function_call"),
        (re.compile(r"[+\-*/|&^]="), "compound_assignment"),
        (re.compile(r"[<>=!]{2,}"), "comparison_operator"),
        (re.compile(r"[|&^~](?!\w)"), "bitwise_operator"),
        (re.compile(r'\w+\[\s*["\']?\w+["\']?\s*\]'), "indexing"),
        (re.compile(r"\+\+|--"), "increment_decrement"),
        (re.compile(r"\?\s*[^:]+\s*:"), "ternary_operator"),
        (re.compile(r"[\[\]{}]\s*[\[\]{}]"), "empty_brackets"),
        (re.compile(r"\.\w+\("), "method_call"),
        (re.compile(r"^/[^/]+/[gimsuvy]*$"), "regex_literal"),
        (re.compile(r'^["\'][^"\']*["\']$'), "string_literal"),
    ]

    # Valid TLDs
    VALID_TLDS = {
        "com",
        "org",
        "net",
        "edu",
        "gov",
        "io",
        "co",
        "ai",
        "dev",
        "app",
        "me",
        "info",
        "biz",
        "xyz",
        "tech",
        "cloud",
        "site",
        "web",
        "us",
        "uk",
        "de",
        "fr",
        "jp",
        "cn",
        "ru",
        "br",
        "in",
        "au",
        "ca",
    }

    # Character density thresholds
    CODE_CHAR_THRESHOLDS = {
        "brackets": (r"[\[\]{}()]", 0.15),
        "operators": (r"[+\-*/=<>!&|^~%]", 0.20),
        "punctuation": (r"[;:,.]", 0.25),
    }

    # === Hex color pattern ===
    HEX_COLOR_PATTERN = re.compile(r"^[0-9a-fA-F]{3}$|^[0-9a-fA-F]{6}$")

    # === Case-insensitive snake_case ===
    SNAKE_CASE_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+$")

    # === Single-letter variable pattern ===
    SINGLE_LETTER_VAR_PATTERN = re.compile(r"^[a-zA-Z]_[a-zA-Z0-9_]+$")

    # Shell variable detection
    SHELL_PATTERNS = [
        (re.compile(r"\$\d+\b"), "shell_positional"),
        (re.compile(r"\$\{[^}]+\}"), "shell_variable_braced"),
        (re.compile(r"\$[A-Za-z_]\w*"), "shell_variable"),
    ]

    # LaTeX detection
    LATEX_PATTERNS = [
        (re.compile(r"\\(?:in|deg|frac|sum|prod|int|sqrt)\b"), "latex_command"),
        (re.compile(r"\{[^}]*\\[a-z]+[^}]*\}"), "latex_braced"),
    ]

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._density_patterns = {name: (re.compile(pattern), threshold) for name, (pattern, threshold) in self.CODE_CHAR_THRESHOLDS.items()}

    def is_code_pollution(
        self,
        surface_text: str,
        entity_type: Optional[str] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> CodeDetectionResult:
        """
        Check if surface text is likely code pollution.
        """
        if not surface_text:
            return CodeDetectionResult(is_code=False, confidence=0.0, language_hint=CodeLanguageHint.UNKNOWN, matched_pattern="", reason="Empty surface text")

        text = surface_text.strip()
        text_lower = text.lower()

        # === PHASE 0: Hex color detection ===
        if self.HEX_COLOR_PATTERN.match(text):
            if any(c.isdigit() for c in text):
                return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.CSS, matched_pattern="hex_color", reason=f"Hex color code: '{text}'")
            if len(text) == 6 and all(c in "0123456789abcdefABCDEF" for c in text):
                return CodeDetectionResult(is_code=True, confidence=0.85, language_hint=CodeLanguageHint.CSS, matched_pattern="hex_color", reason=f"Hex color code: '{text}'")
            # 3-char hex shorthand (fff, eee, aaa)
            if len(text) == 3 and len(set(text.lower())) == 1 and text.lower()[0] in "abcdef":
                return CodeDetectionResult(is_code=True, confidence=0.80, language_hint=CodeLanguageHint.CSS, matched_pattern="hex_color_shorthand", reason=f"Hex color shorthand: '{text}'")

        # === PHASE 1: Exact blocklist ===
        if text_lower in self.EXACT_BLOCKLIST:
            return CodeDetectionResult(is_code=True, confidence=0.95, language_hint=CodeLanguageHint.GENERIC, matched_pattern="exact_blocklist", reason=f"Exact match in blocklist: '{text}'")

        # === PHASE 1.2: SQL phrase patterns ===
        for pattern, pattern_name in self.SQL_PHRASE_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(
                    is_code=True,
                    confidence=0.92,
                    language_hint=CodeLanguageHint.SQL,
                    matched_pattern=pattern_name,
                    reason=f"SQL phrase pattern: {pattern_name}"
                )

        # === PHASE 1.5: Unit abbreviations ===
        if text_lower in self.UNIT_ABBREVIATIONS:
            return CodeDetectionResult(is_code=True, confidence=0.85, language_hint=CodeLanguageHint.GENERIC, matched_pattern="unit_abbreviation", reason=f"Unit abbreviation: '{text}'")

        # === PHASE 2: Module prefix check ===
        if "." in text:
            prefix = text.split(".")[0]
            if prefix in self.CODE_MODULE_PREFIXES or prefix.lower() in self.CODE_MODULE_PREFIXES:
                return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.PYTHON, matched_pattern="module_prefix", reason=f"Code module prefix detected: '{prefix}'")

        # === PHASE 2.5: Snake_case detection ===
        if self.SNAKE_CASE_PATTERN.match(text):
            return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.PYTHON, matched_pattern="snake_case", reason=f"Snake_case identifier: '{text}'")

        # === PHASE 2.6: Single letter variable pattern ===
        if self.SINGLE_LETTER_VAR_PATTERN.match(text):
            return CodeDetectionResult(is_code=True, confidence=0.92, language_hint=CodeLanguageHint.PYTHON, matched_pattern="single_letter_var", reason=f"Single-letter variable pattern: '{text}'")

        # === PHASE 3: Pattern matching by language ===
        for pattern, pattern_name in self.PYTHON_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(is_code=True, confidence=0.85, language_hint=CodeLanguageHint.PYTHON, matched_pattern=pattern_name, reason=f"Python pattern matched: {pattern_name}")

        for pattern, pattern_name in self.JS_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(is_code=True, confidence=0.85, language_hint=CodeLanguageHint.JAVASCRIPT, matched_pattern=pattern_name, reason=f"JavaScript pattern matched: {pattern_name}")

        for pattern, pattern_name in self.HTML_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.HTML, matched_pattern=pattern_name, reason=f"HTML pattern matched: {pattern_name}")

        for pattern, pattern_name in self.SHELL_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.SHELL, matched_pattern=pattern_name, reason=f"SHELL pattern matched: {pattern_name}")

        for pattern, pattern_name in self.LATEX_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.LATEX, matched_pattern=pattern_name, reason=f"LATEX pattern matched: {pattern_name}")


        for pattern, pattern_name in self.GENERIC_CODE_PATTERNS:
            if pattern.search(text):
                return CodeDetectionResult(is_code=True, confidence=0.80, language_hint=CodeLanguageHint.GENERIC, matched_pattern=pattern_name, reason=f"Generic code pattern matched: {pattern_name}")

        # === PHASE 4: Skip density for certain types ===
        skip_density_types = {"PHONE", "IP_ADDRESS", "IP:v4", "IP:v6", "UUID", "HASH_HEX", "DOI", "URL", "EMAIL"}

        short_honorifics = {"dr.", "mr.", "ms.", "mrs.", "jr.", "sr.", "ph.d.", "m.d.", "d.o."}
        if text_lower in short_honorifics:
            return CodeDetectionResult(is_code=False, confidence=0.0, language_hint=CodeLanguageHint.UNKNOWN, matched_pattern="", reason="Recognized honorific/title")

        if "." in text and entity_type in {"BARE_DOMAIN", None}:
            parts = text.lower().split(".")
            if len(parts) >= 2 and parts[-1] in self.VALID_TLDS:
                return CodeDetectionResult(is_code=False, confidence=0.0, language_hint=CodeLanguageHint.UNKNOWN, matched_pattern="", reason="Valid domain with recognized TLD")

        if len(text) >= 3 and entity_type not in skip_density_types:
            for name, (pattern, threshold) in self._density_patterns.items():
                matches = len(pattern.findall(text))
                density = matches / len(text)
                actual_threshold = threshold * 0.8 if self.strict_mode else threshold
                if density > actual_threshold:
                    return CodeDetectionResult(
                        is_code=True, confidence=0.70, language_hint=CodeLanguageHint.GENERIC, matched_pattern=f"density_{name}", reason=f"High {name} density: {density:.2%} > {actual_threshold:.0%}"
                    )

        # === PHASE 5: Structural heuristics ===
        if re.match(r"^[A-Z][A-Z0-9_]+$", text) and "_" in text:
            return CodeDetectionResult(
                is_code=True, confidence=0.65, language_hint=CodeLanguageHint.GENERIC, matched_pattern="constant_name", reason="Constant naming convention (ALL_CAPS_UNDERSCORE)"
            )

        capitals_in_middle = sum(1 for i, c in enumerate(text[1:-1], 1) if c.isupper())
        if capitals_in_middle >= 3 and text[0].islower():
            return CodeDetectionResult(is_code=True, confidence=0.60, language_hint=CodeLanguageHint.JAVASCRIPT, matched_pattern="camel_case_multi", reason="Multi-segment camelCase identifier")

        # === PHASE 6: Context-aware filtering ===
        if context_before or context_after:
            context_result = self._check_context(text, context_before, context_after)
            if context_result:
                return context_result

        return CodeDetectionResult(is_code=False, confidence=0.0, language_hint=CodeLanguageHint.UNKNOWN, matched_pattern="", reason="No code patterns detected")

    def _check_context(self, text: str, context_before: Optional[str], context_after: Optional[str]) -> Optional[CodeDetectionResult]:
        """Check surrounding context for code indicators."""
        if context_before:
            before = context_before[-50:]
            if "```" in before or "~~~" in before:
                return CodeDetectionResult(is_code=True, confidence=0.95, language_hint=CodeLanguageHint.GENERIC, matched_pattern="code_fence_context", reason="Inside code fence")
            if before.rstrip().endswith("`"):
                return CodeDetectionResult(is_code=True, confidence=0.90, language_hint=CodeLanguageHint.GENERIC, matched_pattern="inline_code_context", reason="After inline code marker")

        if context_after:
            after = context_after[:50]
            if "```" in after or "~~~" in after:
                return CodeDetectionResult(is_code=True, confidence=0.95, language_hint=CodeLanguageHint.GENERIC, matched_pattern="code_fence_context", reason="Inside code fence")

        return None

    def is_valid_entity_text(self, surface_text: str, entity_type: Optional[str] = None) -> bool:
        """Returns True if text is NOT code pollution."""
        return not self.is_code_pollution(surface_text, entity_type).is_code

# === SINGLETON INSTANCE ===

_default_filter: Optional[CodePollutionFilter] = None

def get_filter(strict_mode: bool = False) -> CodePollutionFilter:
    """Get the default CodePollutionFilter instance."""
    global _default_filter
    if _default_filter is None or _default_filter.strict_mode != strict_mode:
        _default_filter = CodePollutionFilter(strict_mode=strict_mode)
    return _default_filter


def is_code_pollution(surface_text: str, entity_type: Optional[str] = None) -> bool:
    """Convenience function to check if text is code pollution."""
    return get_filter().is_code_pollution(surface_text, entity_type).is_code


def is_valid_entity(surface_text: str, entity_type: Optional[str] = None) -> bool:
    """Convenience function to check if text is valid (not code)."""
    return get_filter().is_valid_entity_text(surface_text, entity_type)

# ===| PINNED CONFIDENCES |===

PINNED_CONFIDENCES = {
    EntityType.EMAIL: 0.95,
    EntityType.URL: 0.90,
    EntityType.DOI: 0.93,
    EntityType.UUID: 0.92,
    EntityType.HASH_HEX: 0.88,
    EntityType.IP_ADDRESS: 0.90,
    EntityType.PHONE: 0.85,
    EntityType.FILEPATH: 0.80,
    EntityType.BARE_DOMAIN: 0.82,
    EntityType.PERSON: 0.70,
    EntityType.ORG: 0.70,
    EntityType.LOCATION: 0.70,
    EntityType.OTHER: 0.70,
}


class BaseDetector(ABC):
    """
    Base class for entity detectors.
    """

    detector_name: str
    detector_version: str = "1.0.0"
    detector_order: int
    entity_type: EntityType

    # Shared code pollution filter instance
    _code_filter: Optional[CodePollutionFilter] = None

    @classmethod
    def get_code_filter(cls) -> CodePollutionFilter:
        """Get or create the shared code pollution filter."""
        if cls._code_filter is None:
            cls._code_filter = CodePollutionFilter(strict_mode=False)
        return cls._code_filter

    @abstractmethod
    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        """Detect entities in text, return candidates sorted deterministically."""
        pass

    def _sort_candidates(self, candidates: list[EntityCandidate]) -> list[EntityCandidate]:
        """Sort candidates deterministically: (char_start, char_end, surface_hash)."""
        return sorted(
            candidates,
            key=lambda c: (
                c.char_start if c.char_start is not None else float("inf"),
                c.char_end if c.char_end is not None else float("inf"),
                c.surface_hash
            )
        )

    def _is_code_pollution(
        self,
        surface_text: str,
        entity_type: Optional[str] = None,
        context_before: Optional[str] = None,
        context_after: Optional[str] = None,
    ) -> CodeDetectionResult:
        """
        Check if surface text is code pollution using shared filter.

        Args:
            surface_text: Text to check
            entity_type: Optional entity type for context-aware filtering
            context_before: Optional preceding text for context
            context_after: Optional following text for context

        Returns:
            CodeDetectionResult with detection status
        """
        return self.get_code_filter().is_code_pollution(
            surface_text,
            entity_type,
            context_before,
            context_after
        )

    def _filter_code_pollution(
        self,
        candidates: list[EntityCandidate],
        text: Optional[str] = None,
    ) -> list[EntityCandidate]:
        """
        Filter out code pollution from candidates.

        Args:
            candidates: List of entity candidates
            text: Full text for context extraction (optional)

        Returns:
            Filtered list with valid candidates only
        """
        filtered = []
        for c in candidates:
            if c.surface_text is None:
                filtered.append(c)
                continue

            # Get context if available
            context_before = None
            context_after = None
            if text and c.char_start is not None and c.char_end is not None:
                context_before = text[max(0, c.char_start - 50):c.char_start]
                context_after = text[c.char_end:min(len(text), c.char_end + 50)]

            result = self._is_code_pollution(
                c.surface_text,
                c.entity_type_hint,
                context_before,
                context_after
            )

            if result.is_code:
                logger.debug(
                    f"CODE_POLLUTION_DETECTED[{self.detector_name}]: "
                    f"'{c.surface_text}' - {result.reason}"
                )
                # Mark as rejected but don't add to filtered list
                c.rejected_as_code = True
                c.code_rejection_reason = result.reason
            else:
                filtered.append(c)

        return filtered


class EmailDetector(BaseDetector):
    """Detect email addresses."""

    detector_name = "EMAIL"
    detector_version = "1.0.0"
    detector_order = DetectorOrder.EMAIL
    entity_type = EntityType.EMAIL

    # RFC 5322-ish pattern, conservative
    PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)

            # Basic validation: at least one dot in domain
            if surface.count("@") != 1:
                continue

            local, domain = surface.rsplit("@", 1)
            if "." not in domain or not local:
                continue

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=str(self.entity_type.value),
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface}
            ))

        return self._sort_candidates(candidates)


class URLDetector(BaseDetector):
    """Detect URLs with http/https scheme."""

    detector_name = "URL"
    detector_version = "1.0.0"
    detector_order = DetectorOrder.URL
    entity_type = EntityType.URL

    # Require scheme, at least one dot in host
    PATTERN = re.compile(
        r'https?://[^\s<>\[\]{}|\\^`"\']+',
        re.IGNORECASE
    )

    # Trailing punctuation to trim
    TRAILING_PUNCT = set(".,;:!?)>]}'\"")

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)
            char_start = match.start()
            char_end = match.end()

            # Trim trailing punctuation deterministically
            trimmed = []
            while surface and surface[-1] in self.TRAILING_PUNCT:
                trimmed.append(surface[-1])
                surface = surface[:-1]
                char_end -= 1

            if not surface:
                continue

            # Validate: hostname must have at least one dot
            try:
                parsed = urlparse(surface)
                if "." not in parsed.netloc:
                    continue
            except Exception:
                continue

            raw_data = {"match": match.group(0), "trimmed": surface}
            if trimmed:
                raw_data["trimmed_chars"] = "".join(reversed(trimmed))
                logger.info(f"TRAILING_PUNCT_TRIMMED: URL detector trimmed '{raw_data['trimmed_chars']}'")

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=str(self.entity_type.value),
                char_start=char_start,
                char_end=char_end,
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data=raw_data
            ))

        return self._sort_candidates(candidates)


class DOIDetector(BaseDetector):
    """Detect Digital Object Identifiers."""

    detector_name = "DOI"
    detector_version = "1.0.0"
    detector_order = DetectorOrder.DOI
    entity_type = EntityType.DOI

    # DOI pattern: 10.<4-9 digits>/<suffix>
    PATTERN = re.compile(
        r'\b(?:doi:?\s*)?10\.\d{4,9}/[^\s<>\[\]{}|\\^`"\']+',
        re.IGNORECASE
    )

    TRAILING_PUNCT = set(".,;:!?)>]}'\"")

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)
            char_start = match.start()
            char_end = match.end()

            # Trim trailing punctuation
            trimmed = []
            while surface and surface[-1] in self.TRAILING_PUNCT:
                trimmed.append(surface[-1])
                surface = surface[:-1]
                char_end -= 1

            if not surface or "10." not in surface:
                continue

            raw_data = {"match": match.group(0)}
            if trimmed:
                raw_data["trimmed_chars"] = "".join(reversed(trimmed))

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=str(self.entity_type.value),
                char_start=char_start,
                char_end=char_end,
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data=raw_data
            ))

        return self._sort_candidates(candidates)


class UUIDDetector(BaseDetector):
    """Detect UUIDs (versions 1-5)."""

    detector_name = "UUID"
    detector_version = "1.0.0"
    detector_order = DetectorOrder.UUID
    entity_type = EntityType.UUID

    # Strict UUID pattern
    PATTERN = re.compile(
        r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b'
    )

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=str(self.entity_type.value),
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface}
            ))

        return self._sort_candidates(candidates)


class HashHexDetector(BaseDetector):
    """Detect hex hashes (MD5, SHA1, SHA256, SHA512)."""

    detector_name = "HASH_HEX"
    detector_version = "1.0.0"
    detector_order = DetectorOrder.HASH_HEX
    entity_type = EntityType.HASH_HEX

    # Hash lengths: MD5=32, SHA1=40, SHA256=64, SHA512=128
    HASH_LENGTHS = {32: "MD5", 40: "SHA1", 64: "SHA256", 128: "SHA512"}

    # Boundary-guarded hex pattern
    PATTERN = re.compile(r'(?<![0-9a-fA-F])[0-9a-fA-F]{32,128}(?![0-9a-fA-F])')

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)
            length = len(surface)

            # Only accept known hash lengths
            if length not in self.HASH_LENGTHS:
                continue

            hash_type = self.HASH_LENGTHS[length]

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=f"HASH:{hash_type}",
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface, "hash_type": hash_type, "length": length}
            ))

        return self._sort_candidates(candidates)


class IPAddressDetector(BaseDetector):
    """Detect IPv4 and IPv6 addresses with validation."""

    detector_name = "IP_ADDRESS"
    detector_version = "1.0.0"
    detector_order = DetectorOrder.IP_ADDRESS
    entity_type = EntityType.IP_ADDRESS

    # IPv4 pattern
    IPV4_PATTERN = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )

    # IPv6 pattern (simplified, relies on validation)
    IPV6_PATTERN = re.compile(
        r'\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b|\b::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b'
    )

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []

        # IPv4
        for match in self.IPV4_PATTERN.finditer(text):
            surface = match.group(0)
            try:
                # Validate
                ip = ipaddress.ip_address(surface)
                candidates.append(EntityCandidate(
                    message_id=message_id,
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint="IP:v4",
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=surface,
                    confidence=PINNED_CONFIDENCES[self.entity_type],
                    raw_data={"match": surface, "version": 4, "canonical": str(ip)}
                ))
            except ValueError:
                continue

        # IPv6
        for match in self.IPV6_PATTERN.finditer(text):
            surface = match.group(0)
            try:
                ip = ipaddress.ip_address(surface)
                candidates.append(EntityCandidate(
                    message_id=message_id,
                    detector=self.detector_name,
                    detector_version=self.detector_version,
                    entity_type_hint="IP:v6",
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=surface,
                    confidence=PINNED_CONFIDENCES[self.entity_type],
                    raw_data={"match": surface, "version": 6, "canonical": str(ip)}
                ))
            except ValueError:
                continue

        return self._sort_candidates(candidates)


class PhoneDetector(BaseDetector):
    """Detect phone numbers with digit count validation."""

    detector_name = "PHONE"
    detector_version = "1.2.0"  # Updated for code pollution filtering
    detector_order = DetectorOrder.PHONE
    entity_type = EntityType.PHONE

    TIMESTAMP_PATTERN_YYYYMMDDHH = re.compile(r"^20[12]\d{7}$")  # 2024082119
    UNIX_TIMESTAMP_PATTERN = re.compile(r"^\d{10,13}$")  # Epoch seconds or milliseconds

    # Phone pattern: optional + or 00, digits with separators
    # Require at least one common phone separator to distinguish from version numbers
    PATTERN = re.compile(
        r'(?:\+|00)?[\d][\d\s.\-()]{5,}[\d]'
    )

    # Patterns to reject (version numbers, timestamps, etc.)
    VERSION_PATTERN = re.compile(r'^\d+\.\d+\.\d+')  # x.y.z version
    TIMESTAMP_PATTERN = re.compile(r'^\d{1,2}:\d{2}:\d{2}')  # HH:MM:SS
    DATE_PATTERN = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}$')  # YYYY-MM-DD
    IP_LIKE_PATTERN = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')  # IP address
    DECIMAL_PATTERN = re.compile(r'^\d+\.\d+$')  # Pure decimal number

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)
            digits_only = ''.join(c for c in surface if c.isdigit())
            # Reject timestamp-like patterns
            if self.TIMESTAMP_PATTERN_YYYYMMDDHH.match(digits_only):
                continue

            # Reject Unix timestamps
            if self.UNIX_TIMESTAMP_PATTERN.match(digits_only):
                if 1_000_000_000 <= int(digits_only) <= 9_999_999_999_999:
                    continue  # Likely timestamp (2001-2286 range)

            # Skip if it looks like a date
            if self.DATE_PATTERN.match(surface):
                continue

            # Skip if it looks like a version number (x.y.z)
            if self.VERSION_PATTERN.match(surface):
                continue

            # Skip if it looks like a timestamp (HH:MM:SS)
            if self.TIMESTAMP_PATTERN.match(surface):
                continue

            # Skip if it looks like an IP address
            if self.IP_LIKE_PATTERN.match(surface):
                continue

            # Skip if it's just a decimal number (no phone separators)
            if self.DECIMAL_PATTERN.match(surface):
                continue

            # Count digits
            digits = sum(1 for c in surface if c.isdigit())

            # Validate: 7-15 digits
            if digits < 7 or digits > 15:
                continue

            # Check for phone-like separators (at least one of: space, dash, parentheses)
            # to distinguish from random digit sequences
            has_phone_separator = any(c in surface for c in ' -()') or surface.startswith('+') or surface.startswith('00')

            # If no phone separators and no international prefix, require more digits
            # (helps filter out IDs and serial numbers)
            if not has_phone_separator and digits < 10:
                continue

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=str(self.entity_type.value),
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface, "digit_count": digits}
            ))

        return self._sort_candidates(candidates)


class FilepathDetector(BaseDetector):
    """
    Detect Unix and Windows file paths.

    UPDATED: Enhanced code pollution filtering to reject:
    - Comment delimiters (// ///)
    - Import statements
    - Relative path operators
    - Code-like path fragments
    """

    detector_name = "FILEPATH"
    detector_version = "1.2.0"  # Updated for enhanced filtering
    detector_order = DetectorOrder.FILEPATH
    entity_type = EntityType.FILEPATH

    # Unix absolute or home path - require at least one alphanumeric after initial slash(es)
    UNIX_PATTERN = re.compile(r'(?:^|(?<=\s))(?:~)?/(?:[a-zA-Z0-9_][^\s<>|:*?"]*)')

    # Windows drive path
    WINDOWS_PATTERN = re.compile(r'\b[A-Za-z]:\\[^\s<>|*?"]+')

    # Patterns to reject (comment syntax, operators, etc.)
    REJECT_PATTERNS = {
        '//',       # Comment delimiter
        '///',      # Documentation comment
        '/./',      # Current directory reference
        '/../',     # Parent directory reference
        '/.',       # Hidden file in root (often code artifact)
    }

    # Additional code-like patterns to reject
    CODE_PATH_PATTERNS = [
        re.compile(r'^/[a-z]+\.[A-Z]'),      # /module.Class
        re.compile(r'^/\$'),                  # Shell variable in path
        re.compile(r'^/\{'),                  # Template literal
        re.compile(r'^/\*'),                  # Comment start
        re.compile(r'^\s*//.+$'),             # Full line comment
        re.compile(r'^/[a-z]{1,3}$'),         # Single short segment (likely code)
    ]

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []

        # Unix paths
        for match in self.UNIX_PATTERN.finditer(text):
            surface = match.group(0)

            # Skip if it's a rejected pattern
            if surface in self.REJECT_PATTERNS or surface.rstrip('/') in self.REJECT_PATTERNS:
                continue

            # Skip pure slash sequences (e.g., //, ///, etc.)
            stripped = surface.lstrip('~').rstrip('/')
            if stripped == '' or all(c == '/' for c in stripped):
                continue

            # Skip code-like path patterns
            is_code_path = False
            for pattern in self.CODE_PATH_PATTERNS:
                if pattern.match(surface):
                    is_code_path = True
                    logger.debug(f"FILEPATH_CODE_PATTERN: Rejected '{surface}'")
                    break
            if is_code_path:
                continue

            # Must have at least one path component with alphanumeric characters
            # Split by / and check for meaningful components
            components = [c for c in surface.split('/') if c and c != '~']
            if not components:
                continue

            # At least one component must have alphanumeric content
            has_meaningful_component = any(
                any(c.isalnum() for c in comp) for comp in components
            )
            if not has_meaningful_component:
                continue

            # Require minimum length for meaningful paths
            if len(surface) < 3:
                continue

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint="FILEPATH:unix",
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface, "style": "unix"}
            ))

        # Windows paths
        for match in self.WINDOWS_PATTERN.finditer(text):
            surface = match.group(0)

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint="FILEPATH:windows",
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface, "style": "windows"}
            ))

        # Apply code pollution filter as final pass
        filtered_candidates = self._filter_code_pollution(candidates, text)

        return self._sort_candidates(filtered_candidates)


class BareDomainDetector(BaseDetector):
    """
    Detect domain names without scheme.

    UPDATED: Enhanced code pollution filtering to reject:
    - Python module patterns (pd.DataFrame, np.array)
    - JavaScript patterns (console.log, document.getElementById)
    - Generic code patterns (object.method)
    """

    detector_name = "BARE_DOMAIN"
    detector_version = "1.2.0"
    detector_order = DetectorOrder.BARE_DOMAIN
    entity_type = EntityType.BARE_DOMAIN

    # Common TLDs for basic validation (expanded list)
    COMMON_TLDS = {
        # Generic TLDs
        "com", "org", "net", "edu", "gov", "mil", "int",
        # Popular new TLDs
        "io", "co", "ai", "dev", "app", "me", "info", "biz", "pro", "xyz",
        "tech", "online", "site", "web", "cloud", "digital", "media", "design",
        "studio", "agency", "blog", "news", "shop", "store", "market",
        # Country code TLDs (common ones)
        "us", "uk", "de", "fr", "jp", "cn", "ru", "br", "in", "au", "ca",
        "nl", "it", "es", "ch", "se", "no", "dk", "fi", "pl", "be", "at",
        "ie", "nz", "sg", "hk", "kr", "tw", "mx", "ar", "za", "il", "ae",
        "pt", "cz", "gr", "hu", "ro", "ua", "tr", "id", "my", "th", "vn",
        "ph", "pk", "ng", "eg", "ke", "gh",
        # Other popular TLDs
        "gg", "tv", "fm", "im", "to", "ly", "cc", "ws", "la", "vc", "sc",
        "gl", "is", "sh", "so", "st", "sx", "yt", "ac", "mobi", "tel",
    }

    # File extensions to reject
    FILE_EXTENSIONS = {
        "py", "js", "ts", "tsx", "jsx", "java", "cpp", "hpp", "h", "c", "cc",
        "txt", "md", "json", "yaml", "yml", "xml", "toml", "ini", "cfg", "conf",
        "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv", "html", "css",
        "htm", "xhtml", "scss", "sass", "less", "vue", "svelte",
        "jpg", "jpeg", "png", "gif", "svg", "webp", "ico", "bmp", "tiff",
        "mp3", "mp4", "wav", "avi", "mov", "mkv", "flv", "wmv", "webm",
        "zip", "tar", "gz", "bz2", "xz", "rar", "7z",
        "sh", "bash", "zsh", "fish", "ps1", "bat", "cmd",
        "rb", "go", "rs", "kt", "kts", "scala", "clj", "ex", "exs",
        "php", "sql", "log", "lock", "sum", "mod",
        "wasm", "whl", "egg", "gem", "jar", "war", "dll", "so", "dylib",
    }

    # Common code module prefixes
    CODE_MODULE_PREFIXES = {
        # Python data science
        "pd", "np", "tf", "plt", "sns", "sk", "cv", "cv2", "scipy", "torch",
        "keras", "sklearn", "xgb", "lgb", "optuna", "ray", "dask", "jax",
        # Python standard lib
        "os", "re", "sys", "io", "json", "time", "math", "random", "copy",
        "collections", "itertools", "functools", "operator", "typing",
        "pathlib", "datetime", "logging", "threading", "asyncio", "socket",
        "http", "urllib", "hashlib", "base64", "struct", "pickle", "sqlite3",
        "subprocess", "shutil", "glob", "tempfile", "contextlib", "abc",
        # Python web/frameworks
        "flask", "django", "fastapi", "starlette", "aiohttp", "requests",
        "httpx", "uvicorn", "gunicorn", "celery", "redis", "sqlalchemy",
        "pydantic", "marshmallow", "pytest", "unittest", "mock",
        # JavaScript/Node
        "console", "window", "document", "process", "require", "module",
        "exports", "Promise", "Object", "Array", "String", "Number", "Math",
        "JSON", "Date", "RegExp", "Error", "Buffer", "fs", "path", "util",
        "crypto", "stream", "events", "http", "https", "net", "child_process",
        "axios", "fetch", "lodash", "moment", "dayjs",
        # OOP common
        "self", "cls", "this", "super", "base", "parent",
        # Error/exception
        "err", "error", "exception", "ex",
    }

    # Common code method/property names that shouldn't be domain segments
    CODE_METHOD_NAMES = {
        'append', 'extend', 'pop', 'push', 'shift', 'unshift',
        'map', 'filter', 'reduce', 'forEach', 'find', 'some', 'every',
        'slice', 'splice', 'concat', 'join', 'split', 'replace',
        'keys', 'values', 'items', 'entries', 'get', 'set', 'has',
        'add', 'delete', 'clear', 'size', 'length', 'count',
        'read', 'write', 'open', 'close', 'flush', 'seek',
        'encode', 'decode', 'parse', 'stringify', 'format',
        'upper', 'lower', 'strip', 'trim', 'startswith', 'endswith',
        'log', 'error', 'warn', 'info', 'debug', 'trace',
        'then', 'catch', 'finally', 'resolve', 'reject',
        'call', 'apply', 'bind', 'create', 'assign', 'freeze',
        'createElement', 'getElementById', 'querySelector',
        'addEventListener', 'removeEventListener', 'dispatch',
    }

    # Pattern to detect lowercase.Uppercase (module.Class pattern common in code)
    CODE_CLASS_PATTERN = re.compile(r'^[a-z]{1,4}\.[A-Z]')

    # Pattern for generic code method access (x.methodName)
    CODE_METHOD_PATTERN = re.compile(r'^[a-z_][a-zA-Z0-9_]*\.[a-z_][a-zA-Z0-9_]*(?:\(|$)')

    # Pattern for JavaScript-style chains
    JS_CHAIN_PATTERN = re.compile(r'^[a-z]+(?:\.[a-z]+){2,}', re.IGNORECASE)

    # Domain pattern: word.word(.word)+ with TLD
    PATTERN = re.compile(
        r'\b(?!https?://)([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b'
    )

    def __init__(self, domain_tld_allowlist_enabled: bool):
        self.domain_tld_allowlist_enabled = domain_tld_allowlist_enabled

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        candidates = []
        for match in self.PATTERN.finditer(text):
            surface = match.group(0)

            # Check for TLD
            parts = surface.lower().split(".")
            if len(parts) < 2:
                continue

            tld = parts[-1]
            prefix = parts[0]

            # Reject file extensions
            if tld in self.FILE_EXTENSIONS:
                continue

            # If TLD allowlist enabled, validate
            if self.domain_tld_allowlist_enabled:
                if tld not in self.COMMON_TLDS:
                    continue

            # Reject code module patterns (pd.DataFrame, np.array, etc.)
            if prefix in self.CODE_MODULE_PREFIXES:
                logger.debug(f"BARE_DOMAIN_CODE_PREFIX: Rejected '{surface}' (prefix: {prefix})")
                continue

            # Reject lowercase.Uppercase patterns (module.Class syntax)
            if self.CODE_CLASS_PATTERN.match(surface):
                logger.debug(f"BARE_DOMAIN_CODE_CLASS: Rejected '{surface}'")
                continue

            # Reject code method patterns
            if self.CODE_METHOD_PATTERN.match(surface):
                logger.debug(f"BARE_DOMAIN_CODE_METHOD: Rejected '{surface}'")
                continue

            # Reject JavaScript-style chained calls with many segments
            if len(parts) >= 3 and self.JS_CHAIN_PATTERN.match(surface):
                # But allow actual subdomains like mail.google.com
                if tld not in self.COMMON_TLDS:
                    logger.debug(f"BARE_DOMAIN_JS_CHAIN: Rejected '{surface}'")
                    continue

            # Reject single character before dot (likely code: x.value, a.b)
            if len(prefix) == 1:
                continue

            # Check for code-like patterns in the second segment
            if len(parts) >= 2:
                second = parts[1]
                if second.lower() in self.CODE_METHOD_NAMES:
                    logger.debug(f"BARE_DOMAIN_CODE_METHOD_NAME: Rejected '{surface}'")
                    continue

            candidates.append(EntityCandidate(
                message_id=message_id,
                detector=self.detector_name,
                detector_version=self.detector_version,
                entity_type_hint=str(self.entity_type.value),
                char_start=match.start(),
                char_end=match.end(),
                surface_text=surface,
                confidence=PINNED_CONFIDENCES[self.entity_type],
                raw_data={"match": surface, "tld": tld}
            ))

        # Apply code pollution filter as final pass
        filtered_candidates = self._filter_code_pollution(candidates, text)

        return self._sort_candidates(filtered_candidates)


class NERDetector(BaseDetector):
    """
    NER-based entity detection using spaCy.

    - Code identifiers matched by NER
    - Programming language keywords
    - Variable and function names
    """

    detector_name = "NER"
    detector_order = DetectorOrder.NER
    entity_type = EntityType.PERSON  # Default, actual type varies

    # Map spaCy labels to our entity types
    LABEL_MAP = {
        "PERSON": EntityType.PERSON,
        "ORG": EntityType.ORG,
        "GPE": EntityType.LOCATION,
        "LOC": EntityType.LOCATION,
        "FAC": EntityType.LOCATION,
    }

    # Additional NER-specific code patterns
    NER_CODE_PATTERNS = [
        re.compile(r'^[A-Z][a-z]+[A-Z]'),        # CamelCase
        re.compile(r'^[a-z]+_[a-z]+'),           # snake_case
        re.compile(r'^\$[a-zA-Z]'),              # Variable ($var)
        re.compile(r'^[a-z]{1,2}[A-Z]'),         # camelCase starting lowercase
        re.compile(r'Error$|Exception$'),        # Error/Exception classes
        re.compile(r'^I[A-Z][a-z]'),             # Interface names (IFoo)
        re.compile(r'^Abstract[A-Z]'),           # Abstract classes
        re.compile(r'^Base[A-Z]'),               # Base classes
    ]

    # NER-specific blocklist (programming terms NER might match)
    NER_BLOCKLIST = {
        # --- Your existing items ---
        # Python builtins that might match PERSON/ORG
        "None",
        "True",
        "False",
        "self",
        "cls",
        # JavaScript
        "null",
        "undefined",
        "NaN",
        "Infinity",
        # Common class names that aren't entities
        "Object",
        "Array",
        "String",
        "Number",
        "Boolean",
        "Function",
        "Error",
        "TypeError",
        "ValueError",
        "KeyError",
        "IndexError",
        "Promise",
        "Date",
        "RegExp",
        "Map",
        "Set",
        "Symbol",
        # Common code names
        "Main",
        "App",
        "Index",
        "Config",
        "Utils",
        "Helper",
        "Handler",
        "Manager",
        "Service",
        "Controller",
        "Model",
        "View",
        "Component",
        "Factory",
        "Builder",
        "Adapter",
        "Wrapper",
        "Provider",
        "Context",
        # ML Libraries
        "Prophet",
        "prophet",
        "ARIMA",
        "arima",
        "SHAP",
        "shap",
        "Matplotlib",
        "matplotlib",
        "Pandas",
        "pandas",
        "NumPy",
        "numpy",
        "Sklearn",
        "sklearn",
        "TensorFlow",
        "tensorflow",
        "PyTorch",
        "pytorch",
        "XGBoost",
        "xgboost",
        "LightGBM",
        "lightgbm",
        "Keras",
        "keras",
        # Data formats
        "Parquet",
        "parquet",
        "JSON",
        "CSV",
        "Feather",
        # Common code terms misidentified
        "Linear",
        "linear",
        "Generate",
        "generate",
        "Order",
        "order",
        "Python",
        "python",
        # Metrics
        "MAE",
        "RMSE",
        "MSE",
        "MAPE",
        "R2",
        # Common single-word code/tech terms
        "logger",
        "logging",
        "debug",
        "trace",
        "error",
        "warning",
        "dynamic",
        "static",
        "async",
        "sync",
        "callback",
        "optuna",
        "hydra",
        "mlflow",
        "wandb",
        "neptune",
        "git",
        "svn",
        "mercurial",
        "examples",
        "example",
        "demo",
        "sample",
        "test",
        "tests",
        "track",
        "tracker",
        "meter",
        "counter",
        "gauge",
        "buy",
        "sell",
        "spot",
        "bid",
        "ask",  # Trading terms often misclassified
        "ci",
        "cd",
        "pca",
        "pde",
        "ode",
        "svm",
        "knn",
        "lstm",
        "gru",
        # --- Extensions ---
        # Python keywords / common identifiers that can be mis-tagged
        "def",
        "class",
        "return",
        "yield",
        "import",
        "from",
        "as",
        "pass",
        "break",
        "continue",
        "try",
        "except",
        "finally",
        "raise",
        "with",
        "lambda",
        "global",
        "nonlocal",
        "assert",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "in",
        "is",
        "and",
        "or",
        "not",
        "__init__",
        "__main__",
        "__name__",
        "__repr__",
        "__str__",
        "kwargs",
        "args",
        "stdin",
        "stdout",
        "stderr",
        # Python builtins / common types
        "int",
        "float",
        "bool",
        "str",
        "bytes",
        "bytearray",
        "list",
        "dict",
        "set",
        "tuple",
        "frozenset",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "sum",
        "min",
        "max",
        "any",
        "all",
        "open",
        "print",
        "Exception",
        "RuntimeError",
        "NotImplementedError",
        "ImportError",
        "StopIteration",
        "AttributeError",
        "OSError",
        "IOError",
        "ZeroDivisionError",
        # NumPy / pandas dtypes & common objects
        "dtype",
        "ndarray",
        "DataFrame",
        "Series",
        "Index",
        "MultiIndex",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "datetime64",
        "timedelta64",
        "categorical",
        "category",
        "object",
        "bool_",
        "int_",
        "float_",
        # Common filenames / entrypoints frequently tagged as ORG/PRODUCT
        "README",
        "LICENSE",
        "CHANGELOG",
        "Makefile",
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        ".env",
        ".gitignore",
        "config.yaml",
        "config.yml",
        # JavaScript / TypeScript keywords & common globals
        "var",
        "let",
        "const",
        "function",
        "async",
        "await",
        "export",
        "default",
        "require",
        "module",
        "exports",
        "this",
        "new",
        "typeof",
        "instanceof",
        "console",
        "document",
        "window",
        "global",
        "process",
        "React",
        "react",
        "Node",
        "node",
        "Express",
        "express",
        "TypeScript",
        "typescript",
        "JavaScript",
        "javascript",
        # Web / API terms often misclassified as ORG/LOC
        "HTTP",
        "HTTPS",
        "REST",
        "GraphQL",
        "gRPC",
        "WebSocket",
        "API",
        "SDK",
        "URL",
        "URI",
        "JWT",
        "OAuth",
        "OIDC",
        "CORS",
        "CSRF",
        "XSS",
        "TLS",
        "SSL",
        "GET",
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
        "request",
        "response",
        "headers",
        "payload",
        "endpoint",
        "timeout",
        "localhost",
        "127.0.0.1",
        # Containers / DevOps / CI tooling
        "Docker",
        "docker",
        "Kubernetes",
        "kubernetes",
        "k8s",
        "Helm",
        "helm",
        "Terraform",
        "terraform",
        "Ansible",
        "ansible",
        "GitHub",
        "github",
        "GitLab",
        "gitlab",
        "Jenkins",
        "jenkins",
        "Airflow",
        "airflow",
        "Dagster",
        "dagster",
        "Prefect",
        "prefect",
        "cron",
        "systemd",
        "nginx",
        "apache",
        # Data engineering / storage / db terms (non-entity in your KG context)
        "SQL",
        "NoSQL",
        "Postgres",
        "postgres",
        "SQLite",
        "sqlite",
        "MySQL",
        "mysql",
        "MongoDB",
        "mongodb",
        "Redis",
        "redis",
        "Kafka",
        "kafka",
        "Spark",
        "spark",
        "Hadoop",
        "hadoop",
        "Delta",
        "delta",
        "Iceberg",
        "iceberg",
        "Hudi",
        "hudi",
        "S3",
        "GCS",
        "ADLS",
        "ETL",
        "ELT",
        "CDC",
        "schema",
        "table",
        "view",
        "index",
        "primary",
        "foreign",
        "join",
        "partition",
        "shard",
        "replica",
        "replication",
        # Common ML/DS method names that get tagged as entities
        "Regression",
        "regression",
        "Classification",
        "classification",
        "Clustering",
        "clustering",
        "Forecast",
        "forecast",
        "Baseline",
        "baseline",
        "Pipeline",
        "pipeline",
        "CrossValidation",
        "crossvalidation",
        "CV",
        "cv",
        "GridSearch",
        "gridsearch",
        "RandomSearch",
        "randomsearch",
        "Bayesian",
        "bayesian",
        "Bayes",
        "bayes",
        "Bootstrap",
        "bootstrap",
        "Bagging",
        "bagging",
        "Boosting",
        "boosting",
        "Ensemble",
        "ensemble",
        "Regularization",
        "regularization",
        "L1",
        "L2",
        "ElasticNet",
        "elasticnet",
        "Ridge",
        "ridge",
        "Lasso",
        "lasso",
        "Scaler",
        "scaler",
        "StandardScaler",
        "MinMaxScaler",
        "Normalizer",
        "Imputer",
        "imputer",
        "OneHot",
        "onehot",
        "LabelEncoder",
        "labelencoder",
        "Tokenizer",
        "tokenizer",
        "Embedding",
        "embedding",
        "PCA",
        "ICA",
        "tSNE",
        "TSNE",
        "UMAP",
        "umap",
        # Common model names (often tagged as ORG/PRODUCT)
        "RandomForest",
        "randomforest",
        "DecisionTree",
        "decisiontree",
        "GradientBoosting",
        "gradientboosting",
        "AdaBoost",
        "adaboost",
        "CatBoost",
        "catboost",
        "Logistic",
        "logistic",
        "NaiveBayes",
        "naivebayes",
        "KMeans",
        "kmeans",
        "DBSCAN",
        "dbscan",
        "HMM",
        "hmm",
        "Transformer",
        "transformer",
        "Attention",
        "attention",
        "CNN",
        "RNN",
        "LSTM",
        "GRU",
        "MLP",
        # Deep learning components / terms
        "ReLU",
        "GELU",
        "Sigmoid",
        "Tanh",
        "Softmax",
        "Dropout",
        "BatchNorm",
        "LayerNorm",
        "Loss",
        "loss",
        "MSELoss",
        "CrossEntropy",
        "crossentropy",
        "Optimizer",
        "optimizer",
        "Adam",
        "adam",
        "AdamW",
        "SGD",
        "RMSProp",
        "learning_rate",
        "lr",
        "epochs",
        "batch",
        "batch_size",
        "backprop",
        "backpropagation",
        "gradient",
        "gradients",
        # Stats / probability terms commonly mis-tagged
        "Mean",
        "mean",
        "Median",
        "median",
        "Mode",
        "mode",
        "Variance",
        "variance",
        "Std",
        "std",
        "Covariance",
        "covariance",
        "Correlation",
        "correlation",
        "Normal",
        "normal",
        "Gaussian",
        "gaussian",
        "Poisson",
        "poisson",
        "Bernoulli",
        "bernoulli",
        "Binomial",
        "binomial",
        "pvalue",
        "p-value",
        "t-test",
        "anova",
        "chi-square",
        "AIC",
        "BIC",
        # Metrics & evaluation extensions
        "Accuracy",
        "accuracy",
        "Precision",
        "precision",
        "Recall",
        "recall",
        "F1",
        "F1Score",
        "ROC",
        "AUC",
        "PR",
        "LogLoss",
        "logloss",
        "Confusion",
        "confusion",
        "ConfusionMatrix",
        "confusionmatrix",
        # Common plotting / viz tools (often falsely ORG/PRODUCT in NER)
        "Seaborn",
        "seaborn",
        "Plotly",
        "plotly",
        "Altair",
        "altair",
        "Bokeh",
        "bokeh",
        "ggplot",
        "ggplot2",
        # Notebooks / tooling
        "Jupyter",
        "jupyter",
        "Notebook",
        "notebook",
        "Colab",
        "colab",
        "VSCode",
        "vscode",
        "PyCharm",
        "pycharm",
        # Generic tech tokens often tagged as ORG
        "Linux",
        "linux",
        "Windows",
        "windows",
        "MacOS",
        "macos",
        "CPU",
        "GPU",
        "TPU",
        "CUDA",
        "cuda",
        "ROCm",
        "rocm",
        "RAM",
        "SSD",
        "HDD",
    }

    def __init__(
        self,
        ner_model_name: str,
        ner_model_version: str,
        ner_max_chars: int,
        ner_stride: int,
        emit_spanless_ner: bool,
        ner_label_allowlist: list
    ) -> None:
        self.ner_model_name = ner_model_name
        self.ner_model_version = ner_model_version
        self.ner_max_chars = ner_max_chars
        self.ner_stride = ner_stride
        self.emit_spanless_ner = emit_spanless_ner
        self.ner_label_allowlist = ner_label_allowlist

        self.detector_version = f"{self.ner_model_name}:{self.ner_model_version}"
        self.nlp = None
        self._load_model()
        # Error tracking
        self.failed_chunks = 0
        self.total_chunks = 0
        self.processing_errors: list[str] = []

    def _load_model(self):
        """Load spaCy model."""
        try:
            import spacy
            self.nlp = spacy.load(self.ner_model_name)
            logger.info(f"Loaded spaCy model: {self.ner_model_name}")
        except ImportError:
            logger.warning("spaCy not installed, NER detection disabled")
            self.nlp = None
        except OSError as e:
            model_name = self.ner_model_name
            install_cmds = [
                f"python -m spacy download {model_name}",
                f"python -m pip install -U {model_name}",
                f"python -m pip install -U spacy",
            ]
            logger.warning(
                f"spaCy model '{model_name}' not found, NER detection disabled. "
                f"(Error: {e})\n"
                "To install the required model, run ONE of the following:\n"
                + "\n".join(f"  {cmd}" for cmd in install_cmds)
            )
            self.nlp = None
            return

    def _is_ner_code_artifact(self, text: str) -> bool:
        """
        Check if NER-detected text is likely a code artifact.

        Args:
            text: The detected entity text

        Returns:
            True if likely code, False if likely valid entity
        """
        # Check blocklist
        if text in self.NER_BLOCKLIST:
            return True

        # Check NER-specific code patterns
        for pattern in self.NER_CODE_PATTERNS:
            if pattern.match(text):
                return True

        # Use general code pollution filter
        result = self._is_code_pollution(text)
        return result.is_code

    def detect(self, text: str, message_id: str) -> list[EntityCandidate]:
        if self.nlp is None:
            return []

        candidates = []

        # Handle long texts with chunking
        if len(text) > self.ner_max_chars:
            chunks = self._chunk_text(text)
        else:
            chunks = [(0, text)]

        seen = set()  # For deduplication: (char_start, char_end, entity_type)

        for offset, chunk in chunks:
            self.total_chunks += 1
            try:
                doc = self.nlp(chunk)

                for ent in doc.ents:
                    # Filter by allowed labels
                    if ent.label_ not in self.ner_label_allowlist:
                        continue

                    entity_type = self.LABEL_MAP.get(ent.label_, EntityType.OTHER)

                    # Adjust offsets for chunk position
                    char_start = offset + ent.start_char
                    char_end = offset + ent.end_char
                    surface_text = text[char_start:char_end]

                    # Verify offset correctness
                    expected = ent.text
                    if surface_text != expected:
                        logger.warning(
                            f"OFFSET_UNRELIABLE: NER offset mismatch for '{expected}' vs '{surface_text}'"
                        )
                        if not self.emit_spanless_ner:
                            continue
                        char_start = None
                        char_end = None
                        surface_text = expected

                    # === NEW: Code pollution filtering ===
                    if self._is_ner_code_artifact(surface_text):
                        logger.debug(f"NER_CODE_ARTIFACT: Rejected '{surface_text}'")
                        continue

                    # Deduplication key
                    dedup_key = (char_start, char_end, entity_type.value)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    # Compute confidence (use pinned baseline)
                    confidence = min(PINNED_CONFIDENCES.get(entity_type, 0.70), 0.70)

                    candidates.append(EntityCandidate(
                        message_id=message_id,
                        detector=f"NER:{self.ner_model_name}",
                        detector_version=self.detector_version,
                        entity_type_hint=str(entity_type.value),
                        char_start=char_start,
                        char_end=char_end,
                        surface_text=surface_text,
                        confidence=confidence,
                        raw_data={
                            "label": ent.label_,
                            "text": ent.text,
                            "chunk_offset": offset
                        }
                    ))
            except Exception as e:
                self.failed_chunks += 1
                error_msg = f"NER processing error in chunk at offset {offset}: {e}"
                self.processing_errors.append(error_msg)
                logger.error(error_msg)

        # Apply general code pollution filter as final pass
        filtered_candidates = self._filter_code_pollution(candidates, text)

        return self._sort_candidates(filtered_candidates)

    def get_run_stats(self) -> dict:
        """Get statistics about NER processing for model run logging."""
        return {
            "total_chunks": self.total_chunks,
            "failed_chunks": self.failed_chunks,
            "error_rate": self.failed_chunks / max(self.total_chunks, 1),
            "sample_errors": self.processing_errors[:10],
        }

    def reset_stats(self):
        """Reset error counters for a new run."""
        self.failed_chunks = 0
        self.total_chunks = 0
        self.processing_errors = []

    def _chunk_text(self, text: str) -> list[tuple[int, str]]:
        """Split text into overlapping chunks."""
        chunks = []
        max_chars = self.ner_max_chars
        stride = self.ner_stride

        pos = 0
        while pos < len(text):
            end = min(pos + max_chars, len(text))
            chunks.append((pos, text[pos:end]))
            pos += max_chars - stride
            if end == len(text):
                break

        return chunks

# ===| DATABASE EXTENSION |===

class Stage2Database(Database):
    """Extended database with Stage 2 tables."""

    STAGE2_TABLES = {
        "entities": [
            "entity_id",
            "entity_type",
            "entity_key",
            "canonical_name",
            "aliases_json",
            "status",
            "first_seen_at_utc",
            "last_seen_at_utc",
            "mention_count",
            "conversation_count",
            "raw_stats_json",
        ],
        "entity_mention_candidates": [
            "candidate_id",
            "message_id",
            "detector",
            "detector_version",
            "entity_type_hint",
            "char_start",
            "char_end",
            "surface_text",
            "surface_hash",
            "confidence",
            "is_eligible",
            "suppressed_by_candidate_id",
            "suppression_reason",
            "overlap_group_id",
            "raw_candidate_json",
        ],
        "entity_mentions": [
            "mention_id",
            "message_id",
            "entity_id",
            "candidate_id",
            "detector",
            "detector_version",
            "entity_type_hint",
            "char_start",
            "char_end",
            "surface_text",
            "surface_hash",
            "confidence",
            "raw_mention_json",
        ],
        "time_mentions": [
            "time_mention_id",
            "message_id",
            "char_start",
            "char_end",
            "surface_text",
            "surface_hash",
            "pattern_id",
            "pattern_precedence",
            "anchor_time_utc",
            "resolved_type",
            "valid_from_utc",
            "valid_to_utc",
            "resolution_granularity",
            "timezone_assumed",
            "confidence",
            "raw_parse_json",
        ],
        "ner_model_runs": ["run_id", "model_name", "model_version", "config_json", "started_at_utc", "completed_at_utc", "raw_io_json"],
    }

    def __init__(self, database_path: Path, overwrite: bool = True) -> None:
        super().__init__(database_path)
        self.overwrite = overwrite

    def initialize_stage2_schema(self):
        """Create Stage 2 tables and indices."""
        cursor = self.connection.cursor()

        if self.overwrite:
            cursor.executescript("""
                DROP TABLE IF EXISTS entity_mentions;
                DROP TABLE IF EXISTS time_mentions;
                DROP TABLE IF EXISTS entity_mention_candidates;
                DROP TABLE IF EXISTS entities;
                DROP TABLE IF EXISTS ner_model_runs;
            """)

        # entities table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_key TEXT NOT NULL,
                canonical_name TEXT NOT NULL,
                aliases_json TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                first_seen_at_utc TEXT,
                last_seen_at_utc TEXT,
                mention_count INTEGER NOT NULL DEFAULT 0,
                conversation_count INTEGER NOT NULL DEFAULT 0,
                raw_stats_json TEXT
            )
        """)

        # Unique index on (entity_type, entity_key) for active entities
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS entities_active_uniq
            ON entities(entity_type, entity_key) 
            WHERE status='active'
        """)

        # entity_mention_candidates table
        cursor.execute("""
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
                overlap_group_id TEXT,
                raw_candidate_json TEXT,
                FOREIGN KEY (message_id) REFERENCES messages(message_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidates_message 
            ON entity_mention_candidates(message_id)
        """)

        # entity_mentions table (emitted winners only)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entity_mentions (
                mention_id TEXT PRIMARY KEY,
                message_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
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
                FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (candidate_id) REFERENCES entity_mention_candidates(candidate_id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mentions_message 
            ON entity_mentions(message_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_mentions_entity 
            ON entity_mentions(entity_id)
        """)

        # time_mentions table
        cursor.execute("""
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
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_time_mentions_message 
            ON time_mentions(message_id)
        """)

        # ner_model_runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ner_model_runs (
                run_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                config_json TEXT NOT NULL,
                started_at_utc TEXT NOT NULL,
                completed_at_utc TEXT,
                raw_io_json TEXT
            )
        """)

        self.connection.commit()

    def seed_self_entity(self, id_generator: IDGenerator):
        """Seed the reserved SELF entity."""
        entity_id = id_generator.generate(["entity", "PERSON", "__SELF__"])

        cursor = self.connection.execute(
            "SELECT entity_id FROM entities WHERE entity_id = ?",
            (entity_id,)
        )
        if cursor.fetchone():
            return  # Already exists

        self.connection.execute("""
            INSERT INTO entities (
                entity_id, entity_type, entity_key, canonical_name,
                aliases_json, status, mention_count, conversation_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity_id, "PERSON", "__SELF__", "SELF",
            JCS.canonicalize(["SELF", "I", "me", "my", "myself"]),
            "active", 0, 0
        ))

    def iter_messages_for_stage2(self) -> Iterator[dict[str, Any]]:
        """Iterate messages eligible for Stage 2 processing."""
        cursor = self.connection.execute("""
            SELECT message_id, conversation_id, role, created_at_utc, 
                   timestamp_quality, text_raw, code_fence_ranges_json, 
                   blockquote_ranges_json
            FROM messages
            WHERE text_raw IS NOT NULL
            ORDER BY conversation_id, order_index, message_id
        """)
        for row in cursor:
            yield dict(row)

    def insert_entity_candidate(
        self,
        candidate_id: str,
        message_id: str,
        detector: str,
        detector_version: str,
        entity_type_hint: str,
        char_start: int | None,
        char_end: int | None,
        surface_text: str | None,
        surface_hash: str,
        confidence: float,
        is_eligible: int,
        suppressed_by_candidate_id: str | None,
        suppression_reason: str | None,
        overlap_group_id: str | None,
        raw_candidate_json: str | None,
    ):
        """Insert entity mention candidate."""
        self.connection.execute("""
            INSERT INTO entity_mention_candidates (
                candidate_id, message_id, detector, detector_version,
                entity_type_hint, char_start, char_end, surface_text,
                surface_hash, confidence, is_eligible,
                suppressed_by_candidate_id, suppression_reason, overlap_group_id, raw_candidate_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            candidate_id, message_id, detector, detector_version,
            entity_type_hint, char_start, char_end, surface_text,
            surface_hash, confidence, is_eligible,
            suppressed_by_candidate_id, suppression_reason, overlap_group_id, raw_candidate_json
        ))

    def update_candidate_suppression(
        self,
        candidate_id: str,
        suppressed_by_candidate_id: str,
        suppression_reason: str,
    ):
        """Update candidate suppression info."""
        self.connection.execute("""
            UPDATE entity_mention_candidates
            SET suppressed_by_candidate_id = ?, suppression_reason = ?, is_eligible = 0
            WHERE candidate_id = ?
        """, (suppressed_by_candidate_id, suppression_reason, candidate_id))

    def get_or_create_entity(
        self,
        id_generator: IDGenerator,
        entity_type: str,
        entity_key: str,
    ) -> str:
        """Get existing entity or create new one, returns entity_id."""
        cursor = self.connection.execute("""
            SELECT entity_id FROM entities
            WHERE entity_type = ? AND entity_key = ? AND status = 'active'
        """, (entity_type, entity_key))

        row = cursor.fetchone()
        if row:
            return row[0]

        # Create new entity
        entity_id = id_generator.generate(["entity", entity_type, entity_key])

        self.connection.execute("""
            INSERT INTO entities (
                entity_id, entity_type, entity_key, canonical_name,
                status, mention_count, conversation_count
            ) VALUES (?, ?, ?, ?, 'active', 0, 0)
        """, (entity_id, entity_type, entity_key, entity_key))

        return entity_id

    def insert_entity_mention(
        self,
        mention_id: str,
        message_id: str,
        entity_id: str,
        candidate_id: str,
        detector: str,
        detector_version: str,
        entity_type_hint: str,
        char_start: int | None,
        char_end: int | None,
        surface_text: str | None,
        surface_hash: str,
        confidence: float,
        raw_mention_json: str | None,
    ):
        """Insert emitted entity mention."""
        self.connection.execute("""
            INSERT INTO entity_mentions (
                mention_id, message_id, entity_id, candidate_id,
                detector, detector_version, entity_type_hint,
                char_start, char_end, surface_text, surface_hash,
                confidence, raw_mention_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mention_id, message_id, entity_id, candidate_id,
            detector, detector_version, entity_type_hint,
            char_start, char_end, surface_text, surface_hash,
            confidence, raw_mention_json
        ))

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
        anchor_time_utc: str | None,
        resolved_type: str,
        valid_from_utc: str | None,
        valid_to_utc: str | None,
        resolution_granularity: str | None,
        timezone_assumed: str,
        confidence: float,
        raw_parse_json: str,
    ):
        """Insert time mention."""
        self.connection.execute("""
            INSERT INTO time_mentions (
                time_mention_id, message_id, char_start, char_end,
                surface_text, surface_hash, pattern_id, pattern_precedence,
                anchor_time_utc, resolved_type, valid_from_utc, valid_to_utc,
                resolution_granularity, timezone_assumed, confidence, raw_parse_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            time_mention_id, message_id, char_start, char_end,
            surface_text, surface_hash, pattern_id, pattern_precedence,
            anchor_time_utc, resolved_type, valid_from_utc, valid_to_utc,
            resolution_granularity, timezone_assumed, confidence, raw_parse_json
        ))

    def insert_ner_model_run(
        self,
        run_id: str,
        model_name: str,
        model_version: str,
        config_json: str,
        started_at_utc: str,
        completed_at_utc: str | None,
        raw_io_json: str | None,
    ):
        """Insert NER model run record."""
        self.connection.execute("""
            INSERT INTO ner_model_runs (
                run_id, model_name, model_version, config_json,
                started_at_utc, completed_at_utc, raw_io_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, model_name, model_version, config_json,
            started_at_utc, completed_at_utc, raw_io_json
        ))

    def update_entity_stats(self, entity_id: str):
        """Recompute and update entity statistics including role breakdown."""
        # Get mention stats
        cursor = self.connection.execute("""
            SELECT 
                COUNT(*) as mention_count,
                COUNT(DISTINCT m.conversation_id) as conversation_count,
                MIN(CASE WHEN m.created_at_utc IS NOT NULL THEN m.created_at_utc END) as first_seen,
                MAX(CASE WHEN m.created_at_utc IS NOT NULL THEN m.created_at_utc END) as last_seen
            FROM entity_mentions em
            JOIN messages m ON em.message_id = m.message_id
            WHERE em.entity_id = ?
        """, (entity_id,))

        row = cursor.fetchone()
        if not row:
            return

        mention_count, conversation_count, first_seen, last_seen = row

        # Get role-specific statistics
        role_cursor = self.connection.execute("""
            SELECT 
                m.role,
                COUNT(*) as count,
                MIN(m.created_at_utc) as first_seen_role
            FROM entity_mentions em
            JOIN messages m ON em.message_id = m.message_id
            WHERE em.entity_id = ?
            GROUP BY m.role
        """, (entity_id,))

        role_stats = {}
        first_seen_role = None
        first_seen_role_ts = None

        for role_row in role_cursor:
            role, count, role_first = role_row
            role_stats[f"mention_count_{role}"] = count
            # Track which role saw this entity first
            if role_first and (first_seen_role_ts is None or role_first < first_seen_role_ts):
                first_seen_role = role
                first_seen_role_ts = role_first

        # Build raw_stats_json
        raw_stats = {
            **role_stats,
            "first_seen_role": first_seen_role,
        }

        self.connection.execute("""
            UPDATE entities SET
                mention_count = ?,
                conversation_count = ?,
                first_seen_at_utc = ?,
                last_seen_at_utc = ?,
                raw_stats_json = ?
            WHERE entity_id = ?
        """, (mention_count, conversation_count, first_seen, last_seen,
              JCS.canonicalize(raw_stats), entity_id))

    def update_entity_canonical_name(self, entity_id: str):
        """
        Update canonical name using role-aware selection:
        - First aggregate total frequency per surface_text
        - Apply role preference: user mentions get priority bonus
        - Then by frequency, then by first occurrence
        """
        # Get all surface texts with weighted scores prioritizing user mentions
        # We use a weighted scoring approach:
        # - Count user mentions with 2x weight
        # - This ensures "John" with 5 user mentions beats "John" with 10 assistant mentions
        cursor = self.connection.execute("""
            WITH surface_stats AS (
                SELECT 
                    em.surface_text,
                    SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_count,
                    SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_count,
                    COUNT(*) as total_count,
                    MIN(m.created_at_utc) as first_ts,
                    MIN(m.conversation_id) as first_conv,
                    MIN(m.order_index) as first_order,
                    MIN(m.message_id) as first_msg,
                    MIN(em.mention_id) as first_mention
                FROM entity_mentions em
                JOIN messages m ON em.message_id = m.message_id
                WHERE em.entity_id = ? AND em.surface_text IS NOT NULL
                GROUP BY em.surface_text
            )
            SELECT 
                surface_text,
                user_count,
                assistant_count,
                total_count,
                -- Weighted score: user mentions count double
                (user_count * 2 + assistant_count) as weighted_score,
                -- Has any user mention (for secondary sort)
                CASE WHEN user_count > 0 THEN 0 ELSE 1 END as no_user_mentions,
                first_ts,
                first_conv,
                first_order,
                first_msg,
                first_mention
            FROM surface_stats
            ORDER BY 
                no_user_mentions,      -- Prefer surfaces with user mentions
                weighted_score DESC,   -- Then by weighted frequency
                total_count DESC,      -- Tie-break by raw count
                CASE WHEN first_ts IS NULL THEN 1 ELSE 0 END,
                first_ts,
                first_conv,
                first_order,
                first_msg,
                first_mention,
                surface_text
        """, (entity_id,))

        rows = cursor.fetchall()
        if not rows:
            return

        # First row is the winner
        canonical_name = rows[0][0]

        # Collect all unique surface texts for aliases
        aliases = sorted(set(row[0] for row in rows if row[0]))

        self.connection.execute("""
            UPDATE entities SET
                canonical_name = ?,
                aliases_json = ?
            WHERE entity_id = ?
        """, (canonical_name, JCS.canonicalize(aliases), entity_id))

# ===| NORMALIZER |===

class EntityNormalizer:
    """
    Normalize entity keys for consistent deduplication.
    """

    # Shared code pollution filter instance
    _code_filter: Optional[CodePollutionFilter] = None

    @classmethod
    def get_code_filter(cls) -> CodePollutionFilter:
        """Get or create the shared code pollution filter."""
        if cls._code_filter is None:
            cls._code_filter = CodePollutionFilter(strict_mode=False)
        return cls._code_filter

    # Legacy blocklist (kept for backward compatibility, but CodePollutionFilter is more comprehensive)
    ENTITY_BLOCKLIST = {
        "//", "///", "..", "./", "../", "...",
        "self", "cls", "this", "super", "base",
        "true", "false", "null", "none", "undefined", "nil", "nan",
    }

    # Legacy patterns (kept for backward compatibility)
    CODE_ARTIFACT_PATTERNS = [
        re.compile(r"^[a-z]{1,3}\.[A-Z]"),  # module.Class (pd.DataFrame)
        re.compile(r"^__\w+__$"),           # Dunder methods (__init__)
        re.compile(r"^\.\w+$"),             # Dot notation (.value)
        re.compile(r"^/+$"),                # Pure slash sequences
    ]

    @classmethod
    def is_valid_entity(cls, entity_type: str, surface_text: str) -> bool:
        """
        Check if surface text should be allowed to create an entity.
        Returns False for code artifacts and other invalid patterns.

        UPDATED: Now uses comprehensive CodePollutionFilter for detection.

        Args:
            entity_type: The entity type (e.g., "PERSON", "URL", "FILEPATH")
            surface_text: The raw text of the entity candidate

        Returns:
            True if valid entity, False if code pollution

        """
        if not surface_text:
            return False

        normalized = surface_text.strip().lower()

        # Quick check: legacy blocklist (fast path)
        if normalized in cls.ENTITY_BLOCKLIST:
            logger.debug(f"ENTITY_BLOCKLIST: Rejected '{surface_text}'")
            return False

        # Quick check: legacy patterns (fast path)
        for pattern in cls.CODE_ARTIFACT_PATTERNS:
            if pattern.match(surface_text):
                logger.debug(f"LEGACY_PATTERN: Rejected '{surface_text}'")
                return False

        # Comprehensive check: CodePollutionFilter
        result = cls.get_code_filter().is_code_pollution(surface_text, entity_type)

        if result.is_code:
            logger.debug(
                f"CODE_POLLUTION_FILTER: Rejected '{surface_text}' "
                f"({result.language_hint.value}: {result.reason})"
            )
            return False

        # Additional entity-type-specific validations
        if not cls._validate_by_type(entity_type, surface_text):
            return False

        return True

    @classmethod
    def _validate_by_type(cls, entity_type: str, surface_text: str) -> bool:
        """
        Additional type-specific validation rules.

        Args:
            entity_type: The entity type
            surface_text: The surface text

        Returns:
            True if valid for this type, False otherwise

        """
        # FILEPATH-specific validations
        if entity_type == "FILEPATH":
            # Reject very short "paths" that are likely operators
            if len(surface_text.strip()) < 3:
                return False
            # Reject pure operator sequences
            if re.match(r"^[/.\\]+$", surface_text):
                return False

        # BARE_DOMAIN-specific validations
        if entity_type == "BARE_DOMAIN":
            # Must have at least one dot
            if "." not in surface_text:
                return False
            # First segment shouldn't be a single character
            parts = surface_text.split(".")
            if len(parts[0]) == 1:
                return False

        # PERSON/ORG/LOCATION (NER) specific validations
        if entity_type in ("PERSON", "ORG", "LOCATION", "OTHER"):
            # Reject if looks like a code identifier
            if re.match(r"^[a-z]+_[a-z]+", surface_text):  # snake_case
                return False
            if re.match(r"^[a-z]+[A-Z][a-z]+[A-Z]", surface_text):  # camelCase
                return False
            # Reject common code class names
            code_class_names = {
                "Object", "Array", "String", "Number", "Boolean", "Function",
                "Promise", "Error", "TypeError", "Exception", "Handler",
                "Manager", "Service", "Controller", "Component", "Module",
            }
            if surface_text in code_class_names:
                return False

        return True

    @classmethod
    def get_rejection_reason(cls, entity_type: str, surface_text: str) -> Optional[str]:
        """
        Get detailed rejection reason if entity would be rejected.

        Args:
            entity_type: The entity type
            surface_text: The surface text

        Returns:
            Rejection reason string, or None if valid

        """
        if not surface_text:
            return "Empty surface text"

        normalized = surface_text.strip().lower()

        if normalized in cls.ENTITY_BLOCKLIST:
            return f"Blocklisted: '{normalized}'"

        for i, pattern in enumerate(cls.CODE_ARTIFACT_PATTERNS):
            if pattern.match(surface_text):
                return f"Legacy pattern #{i} matched"

        result = cls.get_code_filter().is_code_pollution(surface_text, entity_type)
        if result.is_code:
            return result.reason

        if not cls._validate_by_type(entity_type, surface_text):
            return f"Type-specific validation failed for {entity_type}"

        return None

    @staticmethod
    def normalize(entity_type: str, surface_text: str) -> str:
        """Normalize surface text to entity key based on type."""
        if not surface_text:
            return ""

        normalizer = {
            str(EntityType.EMAIL.value): EntityNormalizer._normalize_email,
            str(EntityType.URL.value): EntityNormalizer._normalize_url,
            str(EntityType.DOI.value): EntityNormalizer._normalize_doi,
            str(EntityType.UUID.value): EntityNormalizer._normalize_uuid,
            str(EntityType.HASH_HEX.value): EntityNormalizer._normalize_hash,
            str(EntityType.IP_ADDRESS.value): EntityNormalizer._normalize_ip,
            str(EntityType.PHONE.value): EntityNormalizer._normalize_phone,
            str(EntityType.FILEPATH.value): EntityNormalizer._normalize_filepath,
            str(EntityType.BARE_DOMAIN.value): EntityNormalizer._normalize_domain,
            str(EntityType.PERSON.value): EntityNormalizer._normalize_ner,
            str(EntityType.ORG.value): EntityNormalizer._normalize_ner,
            str(EntityType.LOCATION.value): EntityNormalizer._normalize_ner,
            str(EntityType.OTHER.value): EntityNormalizer._normalize_ner,
        }.get(entity_type, EntityNormalizer._normalize_default)

        return normalizer(surface_text)

    @staticmethod
    def _normalize_email(s: str) -> str:
        """Lowercase entire email, trim whitespace."""
        return s.strip().lower()

    @staticmethod
    def _normalize_url(s: str) -> str:
        """
        Parse URL, lowercase scheme and host, remove default ports,
        preserve path/query/fragment.
        """
        s = s.strip()
        try:
            parsed = urlparse(s)
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc.lower()

            # Remove default ports
            if scheme == "http" and netloc.endswith(":80"):
                netloc = netloc[:-3]
            elif scheme == "https" and netloc.endswith(":443"):
                netloc = netloc[:-4]

            return urlunparse((
                scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment
            ))
        except Exception:
            return s.lower()

    @staticmethod
    def _normalize_doi(s: str) -> str:
        """Lowercase, strip whitespace, remove 'doi:' prefix."""
        s = s.strip().lower()
        if s.startswith("doi:"):
            s = s[4:]
        return s

    @staticmethod
    def _normalize_uuid(s: str) -> str:
        """Lowercase UUID."""
        return s.strip().lower()

    @staticmethod
    def _normalize_hash(s: str) -> str:
        """Lowercase hex hash."""
        return s.strip().lower()

    @staticmethod
    def _normalize_ip(s: str) -> str:
        """Use canonical IP representation."""
        s = s.strip()
        try:
            return str(ipaddress.ip_address(s))
        except ValueError:
            return s

    @staticmethod
    def _normalize_phone(s: str) -> str:
        """
        Preserve explicit international prefix, remove non-digits.
        """
        raw = s.strip()
        has_plus = False

        if raw.startswith("+"):
            has_plus = True
            raw = raw[1:]
        elif raw.startswith("00"):
            has_plus = True
            raw = raw[2:]

        # Extract digits only
        digits = "".join(c for c in raw if c.isdigit())

        if has_plus:
            return "+" + digits
        return digits

    @staticmethod
    def _normalize_filepath(s: str) -> str:
        """
        Minimal normalization: uppercase Windows drive letter,
        preserve everything else.
        """
        s = s.strip()

        # Windows drive path
        if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
            return s[0].upper() + s[1:]

        return s

    @staticmethod
    def _normalize_domain(s: str) -> str:
        """Lowercase domain, strip trailing dot."""
        s = s.strip().lower()
        if s.endswith("."):
            s = s[:-1]
        return s

    @staticmethod
    def _normalize_ner(s: str) -> str:
        """NER entities: NFKC normalize, lowercase for blocking."""
        s = s.strip()
        s = unicodedata.normalize("NFKC", s)
        return s.lower()

    @staticmethod
    def _normalize_default(s: str) -> str:
        """Default: strip and lowercase."""
        return s.strip().lower()

# ===| TIME PATTERNS |===

class TimePatternMatcher:
    """Match and resolve time expressions."""

    # Month names
    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
    }

    # Pattern definitions with precedence
    PATTERNS = [
        # Precedence 1: ISO 8601 datetime with timezone
        (1, "iso_datetime_tz", re.compile(
            r'\b(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:Z|([+-]\d{2}:\d{2}))\b'
        ), 0.95),

        # Precedence 2: ISO 8601 date
        (2, "iso_date", re.compile(
            r'\b(\d{4})-(\d{2})-(\d{2})\b'
        ), 0.95),

        # Precedence 3: Explicit absolute dates
        (3, "explicit_date_mdy", re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
            re.IGNORECASE
        ), 0.90),

        (3, "explicit_date_dmy", re.compile(
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?,?\s+(\d{4})\b',
            re.IGNORECASE
        ), 0.90),

        # Precedence 4: Date ranges (e.g., "March 5-10, 2024" or "Jan 1 - Feb 28, 2024")
        (4, "date_range_same_month", re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?'
            r'\s*[-–—to]+\s*(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
            re.IGNORECASE
        ), 0.88),

        (4, "date_range_cross_month", re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?'
            r'\s*[-–—to]+\s*'
            r'(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
            re.IGNORECASE
        ), 0.88),

        # Precedence 5: Month-year mentions
        (5, "month_year", re.compile(
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December|'
            r'Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+(\d{4})\b',
            re.IGNORECASE
        ), 0.75),

        # Precedence 6: Year-only mentions (guarded)
        (6, "year_only", re.compile(
            r'\b((?:19|20)\d{2})\b'
        ), 0.70),

        # Precedence 7: Relative numeric durations
        (7, "relative_ago", re.compile(
            r'\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b',
            re.IGNORECASE
        ), 0.85),

        (7, "relative_in", re.compile(
            r'\bin\s+(\d+)\s+(days?|weeks?|months?|years?)\b',
            re.IGNORECASE
        ), 0.85),

        # Precedence 8: Relative common
        (8, "yesterday", re.compile(r'\byesterday\b', re.IGNORECASE), 0.80),
        (8, "today", re.compile(r'\btoday\b', re.IGNORECASE), 0.80),
        (8, "tomorrow", re.compile(r'\btomorrow\b', re.IGNORECASE), 0.80),
        (8, "last_week", re.compile(r'\blast\s+week\b', re.IGNORECASE), 0.80),
        (8, "next_week", re.compile(r'\bnext\s+week\b', re.IGNORECASE), 0.80),
        (8, "last_month", re.compile(r'\blast\s+month\b', re.IGNORECASE), 0.80),
        (8, "next_month", re.compile(r'\bnext\s+month\b', re.IGNORECASE), 0.80),
        (8, "last_year", re.compile(r'\blast\s+year\b', re.IGNORECASE), 0.80),
        (8, "next_year", re.compile(r'\bnext\s+year\b', re.IGNORECASE), 0.80),

        # Precedence 9: Timezone tokens (standalone timezone mentions)
        (9, "timezone_abbrev", re.compile(
            r'\b(UTC|GMT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|'
            r'CET|CEST|EET|EEST|WET|WEST|'
            r'JST|KST|IST|AEST|AEDT|NZST|NZDT)\b'
        ), 0.60),

        (9, "timezone_offset", re.compile(
            r'\b(UTC|GMT)?[+-]\d{1,2}(?::\d{2})?\b'
        ), 0.55),
    ]

    def __init__(self, config: Stage2Config):
        self.config = config

    def detect(self, text: str, message_id: str) -> list[TimeCandidate]:
        """Detect all time expressions in text."""
        candidates = []

        for precedence, pattern_id, pattern, confidence in self.PATTERNS:
            for match in pattern.finditer(text):
                surface = match.group(0)
                candidates.append(TimeCandidate(
                    message_id=message_id,
                    char_start=match.start(),
                    char_end=match.end(),
                    surface_text=surface,
                    pattern_id=pattern_id,
                    pattern_precedence=precedence,
                    confidence=confidence,
                    raw_data={
                        "groups": match.groups(),
                        "pattern": pattern_id
                    }
                ))

        return candidates

    def resolve(
        self,
        candidate: TimeCandidate,
        anchor_time_utc: str | None,
        timestamp_quality: str | None,
    ) -> ResolvedTime:
        """Resolve a time candidate to UTC timestamps."""
        tz = self.config.anchor_timezone
        pattern_id = candidate.pattern_id
        groups = candidate.raw_data.get("groups", ())

        try:
            if pattern_id == "iso_datetime_tz":
                return self._resolve_iso_datetime_tz(groups, tz)

            elif pattern_id == "iso_date":
                return self._resolve_iso_date(groups, tz)

            elif pattern_id in ("explicit_date_mdy", "explicit_date_dmy"):
                return self._resolve_explicit_date(pattern_id, groups, tz)

            elif pattern_id == "date_range_same_month":
                return self._resolve_date_range_same_month(groups, tz)

            elif pattern_id == "date_range_cross_month":
                return self._resolve_date_range_cross_month(groups, tz)

            elif pattern_id == "month_year":
                return self._resolve_month_year(groups, tz)

            elif pattern_id == "year_only":
                return self._resolve_year_only(groups, tz)

            elif pattern_id in ("relative_ago", "relative_in"):
                return self._resolve_relative_duration(
                    pattern_id, groups, anchor_time_utc, timestamp_quality, tz
                )

            elif pattern_id in ("yesterday", "today", "tomorrow"):
                return self._resolve_simple_relative(
                    pattern_id, anchor_time_utc, timestamp_quality, tz
                )

            elif pattern_id.startswith("last_") or pattern_id.startswith("next_"):
                return self._resolve_period_relative(
                    pattern_id, anchor_time_utc, timestamp_quality, tz
                )

            elif pattern_id in ("timezone_abbrev", "timezone_offset"):
                return self._resolve_timezone_token(candidate.surface_text, tz)

            else:
                return ResolvedTime(
                    resolved_type=TimeResolvedType.UNRESOLVED,
                    valid_from_utc=None,
                    valid_to_utc=None,
                    resolution_granularity=None,
                    timezone_assumed=tz,
                    raw_parse={"reason": "unknown_pattern", "pattern_id": pattern_id}
                )

        except Exception as e:
            logger.warning(f"TIME_PARSE_FAILED: {pattern_id} '{candidate.surface_text}': {e}")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={"reason": "parse_error", "error": str(e)}
            )

    def _resolve_iso_datetime_tz(self, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve ISO datetime with timezone to instant."""
        year, month, day, hour, minute = int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3]), int(groups[4])
        second = int(groups[5]) if groups[5] else 0
        tz_offset = groups[6]  # None for Z, else +HH:MM or -HH:MM

        if tz_offset:
            dt = pendulum.datetime(year, month, day, hour, minute, second, tz=tz_offset)
        else:
            dt = pendulum.datetime(year, month, day, hour, minute, second, tz="UTC")

        valid_from = dt.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INSTANT,
            valid_from_utc=valid_from,
            valid_to_utc=None,
            resolution_granularity=TimeGranularity.MINUTE if not groups[5] else TimeGranularity.SECOND,
            timezone_assumed=tz,
            raw_parse={"method": "iso_datetime_tz", "groups": groups}
        )

    def _resolve_iso_date(self, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve ISO date to day interval."""
        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])

        start = pendulum.datetime(year, month, day, 0, 0, 0, tz=tz)
        end = start.add(days=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.DAY,
            timezone_assumed=tz,
            raw_parse={"method": "iso_date", "groups": groups}
        )

    def _resolve_explicit_date(self, pattern_id: str, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve explicit date formats."""
        if pattern_id == "explicit_date_mdy":
            month_str, day_str, year_str = groups[0], groups[1], groups[2]
        else:  # dmy
            day_str, month_str, year_str = groups[0], groups[1], groups[2]

        month = self.MONTHS.get(month_str.lower())
        if not month:
            raise ValueError(f"Unknown month: {month_str}")

        day = int(day_str)
        year = int(year_str)

        start = pendulum.datetime(year, month, day, 0, 0, 0, tz=tz)
        end = start.add(days=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.DAY,
            timezone_assumed=tz,
            raw_parse={"method": pattern_id, "groups": groups}
        )

    def _resolve_month_year(self, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve month-year to month interval."""
        month_str, year_str = groups[0], groups[1]

        month = self.MONTHS.get(month_str.lower())
        if not month:
            raise ValueError(f"Unknown month: {month_str}")

        year = int(year_str)

        start = pendulum.datetime(year, month, 1, 0, 0, 0, tz=tz)
        end = start.add(months=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.MONTH,
            timezone_assumed=tz,
            raw_parse={"method": "month_year", "groups": groups}
        )

    def _resolve_year_only(self, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve year-only to year interval."""
        year = int(groups[0])

        start = pendulum.datetime(year, 1, 1, 0, 0, 0, tz=tz)
        end = pendulum.datetime(year + 1, 1, 1, 0, 0, 0, tz=tz)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.YEAR,
            timezone_assumed=tz,
            raw_parse={"method": "year_only", "groups": groups}
        )

    def _resolve_relative_duration(
        self,
        pattern_id: str,
        groups: tuple,
        anchor_time_utc: str | None,
        timestamp_quality: str | None,
        tz: str
    ) -> ResolvedTime:
        """Resolve relative durations like '3 days ago'."""
        if not anchor_time_utc:
            logger.warning("TIME_RELATIVE_WITHOUT_ANCHOR")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={"reason": "no_anchor", "pattern_id": pattern_id}
            )

        # Check timestamp quality - if anchor is imputed, mark as unresolved
        if timestamp_quality and timestamp_quality in ("imputed", "estimated", "unreliable"):
            logger.debug(f"TIME_RELATIVE_IMPUTED_ANCHOR: {pattern_id} with quality={timestamp_quality}")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={
                    "reason": "imputed_anchor",
                    "pattern_id": pattern_id,
                    "anchor_quality": timestamp_quality,
                    "groups": groups
                }
            )

        # Parse anchor
        anchor = pendulum.parse(anchor_time_utc)

        amount = int(groups[0])
        unit = groups[1].lower().rstrip("s")  # Normalize: days -> day

        # Compute offset
        if pattern_id == "relative_ago":
            if unit == "day":
                target = anchor.subtract(days=amount)
            elif unit == "week":
                target = anchor.subtract(weeks=amount)
            elif unit == "month":
                target = anchor.subtract(months=amount)
            elif unit == "year":
                target = anchor.subtract(years=amount)
            else:
                raise ValueError(f"Unknown unit: {unit}")
        else:  # relative_in
            if unit == "day":
                target = anchor.add(days=amount)
            elif unit == "week":
                target = anchor.add(weeks=amount)
            elif unit == "month":
                target = anchor.add(months=amount)
            elif unit == "year":
                target = anchor.add(years=amount)
            else:
                raise ValueError(f"Unknown unit: {unit}")

        # Determine granularity
        if unit == "day":
            granularity = TimeGranularity.DAY
            start = target.start_of("day")
            end = target.end_of("day").add(microseconds=1)
        elif unit == "week":
            granularity = TimeGranularity.DAY
            start = target.start_of("week")
            end = target.end_of("week").add(microseconds=1)
        elif unit == "month":
            granularity = TimeGranularity.MONTH
            start = target.start_of("month")
            end = target.end_of("month").add(microseconds=1)
        else:  # year
            granularity = TimeGranularity.YEAR
            start = target.start_of("year")
            end = target.end_of("year").add(microseconds=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=granularity,
            timezone_assumed=tz,
            raw_parse={"method": pattern_id, "groups": groups, "anchor": anchor_time_utc}
        )

    def _resolve_simple_relative(
        self,
        pattern_id: str,
        anchor_time_utc: str | None,
        timestamp_quality: str | None,
        tz: str
    ) -> ResolvedTime:
        """Resolve yesterday/today/tomorrow."""
        if not anchor_time_utc:
            logger.warning("TIME_RELATIVE_WITHOUT_ANCHOR")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={"reason": "no_anchor", "pattern_id": pattern_id}
            )

        # Check timestamp quality - if anchor is imputed, mark as unresolved
        if timestamp_quality and timestamp_quality in ("imputed", "estimated", "unreliable"):
            logger.debug(f"TIME_RELATIVE_IMPUTED_ANCHOR: {pattern_id} with quality={timestamp_quality}")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={
                    "reason": "imputed_anchor",
                    "pattern_id": pattern_id,
                    "anchor_quality": timestamp_quality
                }
            )

        anchor = pendulum.parse(anchor_time_utc).in_tz(tz)

        if pattern_id == "yesterday":
            target = anchor.subtract(days=1)
        elif pattern_id == "tomorrow":
            target = anchor.add(days=1)
        else:  # today
            target = anchor

        start = target.start_of("day")
        end = target.end_of("day").add(microseconds=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.DAY,
            timezone_assumed=tz,
            raw_parse={"method": pattern_id, "anchor": anchor_time_utc}
        )

    def _resolve_period_relative(
        self,
        pattern_id: str,
        anchor_time_utc: str | None,
        timestamp_quality: str | None,
        tz: str
    ) -> ResolvedTime:
        """Resolve last/next week/month/year."""
        if not anchor_time_utc:
            logger.warning("TIME_RELATIVE_WITHOUT_ANCHOR")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={"reason": "no_anchor", "pattern_id": pattern_id}
            )

        # Check timestamp quality - if anchor is imputed, mark as unresolved
        if timestamp_quality and timestamp_quality in ("imputed", "estimated", "unreliable"):
            logger.debug(f"TIME_RELATIVE_IMPUTED_ANCHOR: {pattern_id} with quality={timestamp_quality}")
            return ResolvedTime(
                resolved_type=TimeResolvedType.UNRESOLVED,
                valid_from_utc=None,
                valid_to_utc=None,
                resolution_granularity=None,
                timezone_assumed=tz,
                raw_parse={
                    "reason": "imputed_anchor",
                    "pattern_id": pattern_id,
                    "anchor_quality": timestamp_quality
                }
            )

        anchor = pendulum.parse(anchor_time_utc).in_tz(tz)

        direction, period = pattern_id.split("_")

        if period == "week":
            if direction == "last":
                target = anchor.subtract(weeks=1)
            else:
                target = anchor.add(weeks=1)
            start = target.start_of("week")
            end = target.end_of("week").add(microseconds=1)
            granularity = TimeGranularity.DAY

        elif period == "month":
            if direction == "last":
                target = anchor.subtract(months=1)
            else:
                target = anchor.add(months=1)
            start = target.start_of("month")
            end = target.end_of("month").add(microseconds=1)
            granularity = TimeGranularity.MONTH

        else:  # year
            if direction == "last":
                target = anchor.subtract(years=1)
            else:
                target = anchor.add(years=1)
            start = target.start_of("year")
            end = target.end_of("year").add(microseconds=1)
            granularity = TimeGranularity.YEAR

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=granularity,
            timezone_assumed=tz,
            raw_parse={"method": pattern_id, "anchor": anchor_time_utc}
        )

    def _resolve_date_range_same_month(self, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve date ranges within the same month (e.g., 'March 5-10, 2024')."""
        month_str, day_start_str, day_end_str, year_str = groups[0], groups[1], groups[2], groups[3]

        month = self.MONTHS.get(month_str.lower())
        if not month:
            raise ValueError(f"Unknown month: {month_str}")

        day_start = int(day_start_str)
        day_end = int(day_end_str)
        year = int(year_str)

        start = pendulum.datetime(year, month, day_start, 0, 0, 0, tz=tz)
        end = pendulum.datetime(year, month, day_end, 0, 0, 0, tz=tz).add(days=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.DAY,
            timezone_assumed=tz,
            raw_parse={"method": "date_range_same_month", "groups": groups}
        )

    def _resolve_date_range_cross_month(self, groups: tuple, tz: str) -> ResolvedTime:
        """Resolve date ranges across months (e.g., 'Jan 1 - Feb 28, 2024')."""
        month_start_str, day_start_str, month_end_str, day_end_str, year_str = (
            groups[0], groups[1], groups[2], groups[3], groups[4]
        )

        month_start = self.MONTHS.get(month_start_str.lower())
        month_end = self.MONTHS.get(month_end_str.lower())
        if not month_start or not month_end:
            raise ValueError(f"Unknown month: {month_start_str} or {month_end_str}")

        day_start = int(day_start_str)
        day_end = int(day_end_str)
        year = int(year_str)

        start = pendulum.datetime(year, month_start, day_start, 0, 0, 0, tz=tz)
        end = pendulum.datetime(year, month_end, day_end, 0, 0, 0, tz=tz).add(days=1)

        return ResolvedTime(
            resolved_type=TimeResolvedType.INTERVAL,
            valid_from_utc=start.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            valid_to_utc=end.in_tz("UTC").format(TimestampUtils.ISO_UTC_MILLIS),
            resolution_granularity=TimeGranularity.DAY,
            timezone_assumed=tz,
            raw_parse={"method": "date_range_cross_month", "groups": groups}
        )

    def _resolve_timezone_token(self, surface_text: str, default_tz: str) -> ResolvedTime:
        """
        Resolve standalone timezone tokens.

        These don't represent a specific time, but provide timezone context.
        They're marked as unresolved but with the timezone information captured.
        """
        # Map common timezone abbreviations to their offsets
        TZ_ABBREV_MAP = {
            "UTC": "UTC", "GMT": "UTC",
            "EST": "America/New_York", "EDT": "America/New_York",
            "CST": "America/Chicago", "CDT": "America/Chicago",
            "MST": "America/Denver", "MDT": "America/Denver",
            "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
            "CET": "Europe/Paris", "CEST": "Europe/Paris",
            "EET": "Europe/Helsinki", "EEST": "Europe/Helsinki",
            "WET": "Europe/Lisbon", "WEST": "Europe/Lisbon",
            "JST": "Asia/Tokyo", "KST": "Asia/Seoul",
            "IST": "Asia/Kolkata",
            "AEST": "Australia/Sydney", "AEDT": "Australia/Sydney",
            "NZST": "Pacific/Auckland", "NZDT": "Pacific/Auckland",
        }

        normalized_tz = surface_text.upper().strip()
        resolved_tz = TZ_ABBREV_MAP.get(normalized_tz, normalized_tz)

        return ResolvedTime(
            resolved_type=TimeResolvedType.UNRESOLVED,
            valid_from_utc=None,
            valid_to_utc=None,
            resolution_granularity=None,
            timezone_assumed=default_tz,
            raw_parse={
                "method": "timezone_token",
                "token": surface_text,
                "resolved_timezone": resolved_tz,
                "note": "Standalone timezone reference, provides context but not a specific time"
            }
        )

# ===| OVERLAP RESOLUTION |===

class OverlapResolver:
    """Resolve overlapping candidates deterministically."""

    @staticmethod
    def spans_overlap(a_start: int | None, a_end: int | None, b_start: int | None, b_end: int | None) -> bool:
        """Check if two spans overlap."""
        if a_start is None or a_end is None or b_start is None or b_end is None:
            return False
        return a_start < b_end and b_start < a_end

    @staticmethod
    def is_in_excluded_range(
        char_start: int | None,
        char_end: int | None,
        excluded_ranges: list[dict]
    ) -> tuple[bool, str | None]:
        """Check if span intersects any excluded range."""
        if char_start is None or char_end is None:
            return False, None

        for r in excluded_ranges:
            r_start = r.get("char_start")
            r_end = r.get("char_end")
            if r_start is not None and r_end is not None:
                if char_start < r_end and r_start < char_end:
                    if "language" in r:  # Code fence
                        return True, SuppressionReason.INTERSECTS_CODE_FENCE
                    else:  # Blockquote
                        return True, SuppressionReason.INTERSECTS_BLOCKQUOTE

        return False, None

    @staticmethod
    def entity_scoring_key(c: EntityCandidate) -> tuple:
        """
        Deterministic scoring key for entity candidates.
        Sort order: higher is better for confidence and span_length.
        """
        detector_order = {
            "EMAIL": 1, "URL": 2, "DOI": 3, "UUID": 4, "HASH_HEX": 5,
            "IP_ADDRESS": 6, "PHONE": 7, "FILEPATH": 8, "BARE_DOMAIN": 9
        }

        # NER detectors
        det_name = c.detector.split(":")[0] if ":" in c.detector else c.detector
        det_order = detector_order.get(det_name, 10)

        return (
            -c.confidence,  # Higher confidence first (negated for ascending sort)
            -c.span_length,  # Longer spans first
            det_order,  # Earlier detector order
            c.char_start if c.char_start is not None else float("inf"),
            -(c.char_end if c.char_end is not None else float("-inf")),
            c.surface_hash
        )

    @staticmethod
    def time_scoring_key(c: TimeCandidate) -> tuple:
        """
        Deterministic scoring key for time candidates.
        """
        return (
            -c.span_length,
            c.pattern_precedence,
            -c.confidence,
            c.char_start,
            -c.char_end,
            c.surface_hash
        )

    @staticmethod
    def select_non_overlapping_entities(
        candidates: list[EntityCandidate],
        excluded_ranges: list[dict],
        emit_spanless_ner: bool = False
    ) -> tuple[list[EntityCandidate], list[tuple[EntityCandidate, str, str | None]]]:
        """
        Select non-overlapping entity candidates.

        Returns:
            (emitted, suppressed) where suppressed is list of (candidate, reason, suppressor_id)
        """
        emitted: list[EntityCandidate] = []
        suppressed: list[tuple[EntityCandidate, str, str | None]] = []

        # Sort by scoring key
        sorted_candidates = sorted(candidates, key=OverlapResolver.entity_scoring_key)

        for c in sorted_candidates:
            # Check exclusion ranges first
            is_excluded, reason = OverlapResolver.is_in_excluded_range(
                c.char_start, c.char_end, excluded_ranges
            )
            if is_excluded:
                suppressed.append((c, reason, None))
                continue

            # Check for NULL spans (NER only)
            if c.char_start is None or c.char_end is None:
                if not emit_spanless_ner:
                    suppressed.append((c, SuppressionReason.NO_OFFSETS_UNRELIABLE, None))
                    logger.warning(f"OFFSET_UNRELIABLE: {c.detector} '{c.surface_text}'")
                    continue

            # Check overlap with emitted
            overlaps = False
            suppressor = None
            for e in emitted:
                if OverlapResolver.spans_overlap(c.char_start, c.char_end, e.char_start, e.char_end):
                    overlaps = True
                    suppressor = e
                    break

            if overlaps:
                suppressed.append((c, SuppressionReason.OVERLAP_HIGHER_SCORE, None))
            else:
                emitted.append(c)

        return emitted, suppressed

    @staticmethod
    def select_non_overlapping_times(
        candidates: list[TimeCandidate],
        excluded_ranges: list[dict]
    ) -> tuple[list[TimeCandidate], list[tuple[TimeCandidate, str]]]:
        """Select non-overlapping time candidates."""
        emitted: list[TimeCandidate] = []
        suppressed: list[tuple[TimeCandidate, str]] = []

        # Sort by scoring key
        sorted_candidates = sorted(candidates, key=OverlapResolver.time_scoring_key)

        for c in sorted_candidates:
            # Check exclusion ranges
            is_excluded, reason = OverlapResolver.is_in_excluded_range(
                c.char_start, c.char_end, excluded_ranges
            )
            if is_excluded:
                suppressed.append((c, reason))
                continue

            # Check overlap with emitted
            overlaps = False
            for e in emitted:
                if OverlapResolver.spans_overlap(c.char_start, c.char_end, e.char_start, e.char_end):
                    overlaps = True
                    break

            if overlaps:
                suppressed.append((c, SuppressionReason.OVERLAP_HIGHER_SCORE))
            else:
                emitted.append(c)

        return emitted, suppressed

# ===| MAIN PIPELINE |===

class PersonalLexiconPipeline:
    """Stage 2: Personal Lexicon Layer pipeline."""

    def __init__(self, config: Stage2Config):
        self.config = config
        self.id_generator = IDGenerator(uuid.UUID(config.id_namespace))
        self.db = Stage2Database(config.output_file_path)

        # Initialize detectors
        self.detectors: list[BaseDetector] = [
            EmailDetector(),
            URLDetector(),
            DOIDetector(),
            UUIDDetector(),
            HashHexDetector(),
            IPAddressDetector(),
            PhoneDetector(),
            FilepathDetector(),
            BareDomainDetector(
                domain_tld_allowlist_enabled=config.domain_tld_allowlist_enabled
            ),
        ]

        # Add a NER-based entity detection using spaCy (enabled by default)
        if config.enable_ner:
            self.detectors.append(
                NERDetector(
                    ner_model_name=config.ner_model_name,
                    ner_model_version=config.ner_model_version,
                    ner_max_chars=config.ner_max_chars,
                    ner_stride=config.ner_stride,
                    emit_spanless_ner=config.emit_spanless_ner,
                    ner_label_allowlist=config.ner_label_allowlist
                )
            )

        # Time pattern matcher (match and resolve time expressions)
        self.time_matcher = TimePatternMatcher(config)

        # Statistics
        self.stats = {
            "messages_processed": 0,
            "entity_candidates": 0,
            "entity_mentions_emitted": 0,
            "entities_created": 0,
            "time_mentions_emitted": 0,
        }

    def run(self):
        """Execute Stage 2 pipeline."""
        logger.info("Starting Stage 2: Personal Lexicon Layer")
        started_at = TimestampUtils.now_utc()

        # Initialize schema
        self.db.initialize_stage2_schema()

        # Begin transaction
        self.db.begin()

        try:
            # Seed SELF entity
            self.db.seed_self_entity(self.id_generator)

            # Log NER model run if NER is enabled
            ner_run_id = None
            ner_detector = None
            if self.config.enable_ner:
                for detector in self.detectors:
                    if isinstance(detector, NERDetector):
                        ner_detector = detector
                        break

                if ner_detector and ner_detector.nlp is not None:
                    ner_run_id = self.id_generator.generate([
                        "ner_run",
                        self.config.ner_model_name,
                        started_at
                    ])

                    ner_config = {
                        "model_name": self.config.ner_model_name,
                        "model_version": self.config.ner_model_version,
                        "max_chars": self.config.ner_max_chars,
                        "stride": self.config.ner_stride,
                        "label_allowlist": self.config.ner_label_allowlist,
                        "emit_spanless_ner": self.config.emit_spanless_ner,
                    }

                    self.db.insert_ner_model_run(
                        run_id=ner_run_id,
                        model_name=self.config.ner_model_name,
                        model_version=self.config.ner_model_version,
                        config_json=JCS.canonicalize(ner_config),
                        started_at_utc=started_at,
                        completed_at_utc=None,
                        raw_io_json=None
                    )
                    ner_detector.reset_stats()

            # Track entities that need stats update
            entities_to_update = set()

            # Process each message
            for msg in self.db.iter_messages_for_stage2():
                self._process_message(msg, entities_to_update)
                self.stats["messages_processed"] += 1

                if self.stats["messages_processed"] % 1000 == 0:
                    logger.info(f"Processed {self.stats['messages_processed']} messages...")

            # Update entity statistics and canonical names
            logger.info(f"Updating stats for {len(entities_to_update)} entities...")
            for entity_id in entities_to_update:
                self.db.update_entity_stats(entity_id)
                self.db.update_entity_canonical_name(entity_id)

            # Update NER model run with completion stats
            completed_at = TimestampUtils.now_utc()
            if ner_run_id and ner_detector:
                ner_stats = ner_detector.get_run_stats()
                ner_stats["pipeline_stats"] = self.stats.copy()

                self.db.connection.execute("""
                    UPDATE ner_model_runs 
                    SET completed_at_utc = ?, raw_io_json = ?
                    WHERE run_id = ?
                """, (completed_at, JCS.canonicalize(ner_stats), ner_run_id)) # Convert Python object to JCS-canonical JSON string.

                # Log summary of NER processing issues
                if ner_detector.failed_chunks > 0:
                    logger.warning(
                        f"NER_PROCESSING_SUMMARY: {ner_detector.failed_chunks}/{ner_detector.total_chunks} "
                        f"chunks failed ({ner_stats['error_rate']*100:.2f}% error rate)"
                    )

            # Commit transaction
            self.db.commit()

            logger.info(f"Stage 2 completed successfully")
            logger.info(f"  Messages processed: {self.stats['messages_processed']}")
            logger.info(f"  Entity candidates: {self.stats['entity_candidates']}")
            logger.info(f"  Entity mentions emitted: {self.stats['entity_mentions_emitted']}")
            logger.info(f"  Unique entities: {len(entities_to_update)}")
            logger.info(f"  Time mentions emitted: {self.stats['time_mentions_emitted']}")
            logger.info(f"  Duration: {started_at} -> {completed_at}")

        except Exception as e:
            logger.error(f"Stage 2 failed: {e}")
            self.db.rollback()
            raise

        finally:
            self.db.close()

    def _process_message(self, msg: dict, entities_to_update: set):
        """Process a single message for entity and time mentions."""
        message_id = msg["message_id"]
        text_raw = msg["text_raw"]

        if not text_raw:
            logger.debug(f"Skipping message {message_id} without text raw")
            return

        # Parse exclusion ranges
        excluded_ranges = []

        if msg.get("code_fence_ranges_json"):
            excluded_ranges.extend(json.loads(msg["code_fence_ranges_json"]))

        if self.config.ignore_markdown_blockquotes and msg.get("blockquote_ranges_json"):
            excluded_ranges.extend(json.loads(msg["blockquote_ranges_json"]))

        # Collect all entity candidates from all detectors
        all_entity_candidates = []
        for detector in self.detectors:
            candidates = detector.detect(text_raw, message_id)
            all_entity_candidates.extend(candidates)

        self.stats["entity_candidates"] += len(all_entity_candidates)

        # Compute overlap groups for all candidates
        # Group overlapping candidates together with a deterministic ID
        overlap_groups = self._compute_overlap_groups(all_entity_candidates, message_id)

        # Store all candidates
        candidate_id_map = {}
        for c in all_entity_candidates:
            candidate_id = self.id_generator.generate([
                "candidate",
                message_id,
                c.detector,
                c.entity_type_hint,
                c.char_start,
                c.char_end,
                c.surface_hash
            ])
            candidate_id_map[id(c)] = candidate_id

            # Get overlap group ID for this candidate
            overlap_group_id = overlap_groups.get(id(c))

            self.db.insert_entity_candidate(
                candidate_id=candidate_id,
                message_id=message_id,
                detector=c.detector,
                detector_version=c.detector_version,
                entity_type_hint=c.entity_type_hint,
                char_start=c.char_start,
                char_end=c.char_end,
                surface_text=c.surface_text,
                surface_hash=c.surface_hash,
                confidence=c.confidence,
                is_eligible=1,  # Will be updated if suppressed
                suppressed_by_candidate_id=None,
                suppression_reason=None,
                overlap_group_id=overlap_group_id,
                raw_candidate_json=JCS.canonicalize(c.raw_data) if c.raw_data else None
            )

        # Resolve overlaps
        emitted, suppressed = OverlapResolver.select_non_overlapping_entities(
            all_entity_candidates,
            excluded_ranges,
            self.config.emit_spanless_ner
        )

        # Update suppressed candidates
        for c, reason, _ in suppressed:
            candidate_id = candidate_id_map[id(c)]
            self.db.update_candidate_suppression(candidate_id, "", reason)

        # Create mentions for emitted candidates
        for c in emitted:
            candidate_id = candidate_id_map[id(c)]

            # Map entity type hint to normalized type
            entity_type = c.entity_type_hint.split(":")[0]  # Handle HASH:SHA256 -> HASH_HEX
            if entity_type.startswith("HASH"):
                entity_type = EntityType.HASH_HEX.value
            elif entity_type.startswith("IP"):
                entity_type = EntityType.IP_ADDRESS.value
            elif entity_type.startswith("FILEPATH"):
                entity_type = EntityType.FILEPATH.value

            # Validate entity before creation (filter code artifacts)
            if not EntityNormalizer.is_valid_entity(entity_type, c.surface_text or ""):
                logger.debug(f"ENTITY_REJECTED: '{c.surface_text}' failed validation")
                self.db.update_candidate_suppression(
                    candidate_id, "", "INVALID_ENTITY_PATTERN"
                )
                continue

            # Normalize entity key (Normalize entity keys for consistent deduplication, e.g., s.strip().lower()).
            entity_key = EntityNormalizer.normalize(entity_type, c.surface_text or "")

            # Get or create entity
            entity_id = self.db.get_or_create_entity(
                self.id_generator,
                entity_type,
                entity_key
            )
            entities_to_update.add(entity_id)

            # Generate mention ID
            mention_id = self.id_generator.generate([
                "mention",
                message_id,
                candidate_id
            ])

            # Insert mention
            self.db.insert_entity_mention(
                mention_id=mention_id,
                message_id=message_id,
                entity_id=entity_id,
                candidate_id=candidate_id,
                detector=c.detector,
                detector_version=c.detector_version,
                entity_type_hint=c.entity_type_hint,
                char_start=c.char_start,
                char_end=c.char_end,
                surface_text=c.surface_text,
                surface_hash=c.surface_hash,
                confidence=c.confidence,
                raw_mention_json=JCS.canonicalize(c.raw_data) if c.raw_data else None
            )

            self.stats["entity_mentions_emitted"] += 1

        # Time mention detection
        time_candidates = self.time_matcher.detect(text_raw, message_id)

        # Resolve overlaps
        emitted_times, _ = OverlapResolver.select_non_overlapping_times(
            time_candidates,
            excluded_ranges
        )

        # Resolve and store time mentions
        anchor_time_utc = msg.get("created_at_utc")
        timestamp_quality = msg.get("timestamp_quality")

        for tc in emitted_times:
            resolved = self.time_matcher.resolve(tc, anchor_time_utc, timestamp_quality)

            time_mention_id = self.id_generator.generate([
                "time_mention",
                message_id,
                tc.char_start,
                tc.char_end,
                tc.surface_hash
            ])

            self.db.insert_time_mention(
                time_mention_id=time_mention_id,
                message_id=message_id,
                char_start=tc.char_start,
                char_end=tc.char_end,
                surface_text=tc.surface_text,
                surface_hash=tc.surface_hash,
                pattern_id=tc.pattern_id,
                pattern_precedence=tc.pattern_precedence,
                anchor_time_utc=anchor_time_utc,
                resolved_type=resolved.resolved_type,
                valid_from_utc=resolved.valid_from_utc,
                valid_to_utc=resolved.valid_to_utc,
                resolution_granularity=resolved.resolution_granularity,
                timezone_assumed=resolved.timezone_assumed,
                confidence=tc.confidence,
                raw_parse_json=JCS.canonicalize(resolved.raw_parse)
            )

            self.stats["time_mentions_emitted"] += 1

    def _compute_overlap_groups(
        self,
        candidates: list[EntityCandidate],
        message_id: str
    ) -> dict[int, str]:
        """
        Compute deterministic overlap group IDs for candidates.

        Candidates that overlap are assigned the same group ID.
        This allows future reprocessing to reconsider alternatives.

        Returns:
            Dict mapping candidate id(obj) -> overlap_group_id
        """
        if not candidates:
            return {}

        # Sort candidates by position for consistent processing
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (
                c.char_start if c.char_start is not None else float("inf"),
                c.char_end if c.char_end is not None else float("inf"),
                c.surface_hash
            )
        )

        # Union-Find to group overlapping candidates
        parent = {id(c): id(c) for c in sorted_candidates}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Group overlapping candidates
        for i, c1 in enumerate(sorted_candidates):
            if c1.char_start is None or c1.char_end is None:
                continue
            for c2 in sorted_candidates[i+1:]:
                if c2.char_start is None or c2.char_end is None:
                    continue
                # Early exit: if c2 starts after c1 ends, no more overlaps possible
                if c2.char_start >= c1.char_end:
                    break
                # Check overlap
                if OverlapResolver.spans_overlap(
                    c1.char_start, c1.char_end,
                    c2.char_start, c2.char_end
                ):
                    union(id(c1), id(c2))

        # Collect groups
        groups: dict[int, list[EntityCandidate]] = {}
        for c in sorted_candidates:
            root = find(id(c))
            if root not in groups:
                groups[root] = []
            groups[root].append(c)

        # Generate deterministic group IDs
        result: dict[int, str] = {}
        for root, group_candidates in groups.items():
            if len(group_candidates) <= 1:
                # No overlap, no group ID needed
                continue

            # Compute group bounds for deterministic ID
            min_start = min(
                c.char_start for c in group_candidates
                if c.char_start is not None
            )
            max_end = max(
                c.char_end for c in group_candidates
                if c.char_end is not None
            )

            # Generate deterministic group ID
            group_id = self.id_generator.generate([
                "overlap",
                message_id,
                min_start,
                max_end
            ])

            for c in group_candidates:
                result[id(c)] = group_id

        return result

# ===| DATA CLEANUP UTILITIES |===

def cleanup_polluted_entities(db_path: Path, dry_run: bool = True) -> dict:
    """
    Clean up polluted entities from Stage 2 processing.

    Removes entities that match known code artifact patterns.

    NOTE: This utility exists primarily for cleaning up legacy databases
    that were processed before the inline filtering was added via
    EntityNormalizer.is_valid_entity(). For new processing runs, the
    inline filtering should catch most code artifacts during entity creation.

    Use cases for this utility:
    1. Cleaning up databases processed with older versions of the pipeline
    2. Applying stricter filtering rules retroactively
    3. Running periodic maintenance on production databases
    4. Testing new code artifact patterns before adding to inline filtering

    Args:
        db_path: Path to the SQLite database
        dry_run: If True, only report what would be deleted without making changes

    Returns:
        Dictionary with cleanup statistics
    """
    # Known polluted entity patterns
    POLLUTED_PATTERNS = [
        '//',           # Comment delimiter matched as FILEPATH
        '///',          # Documentation comment
        '../',          # Relative path operator
        './',           # Current directory
    ]

    # SQL patterns for module.Class code patterns
    CODE_PATTERN_SQL = """
        canonical_name GLOB '[a-z][a-z].[A-Z]*' OR
        canonical_name GLOB '[a-z][a-z][a-z].[A-Z]*' OR
        canonical_name LIKE 'pd.%' OR
        canonical_name LIKE 'np.%' OR
        canonical_name LIKE 'tf.%' OR
        canonical_name LIKE 'os.%' OR
        canonical_name LIKE 'sys.%' OR
        canonical_name LIKE 'self.%'
    """

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    stats = {
        "entities_to_delete": 0,
        "mentions_to_delete": 0,
        "polluted_by_type": {},
        "dry_run": dry_run,
    }

    # Find polluted entities by exact match
    placeholders = ",".join("?" for _ in POLLUTED_PATTERNS)
    cursor.execute(f"""
        SELECT entity_id, entity_type, canonical_name, mention_count
        FROM entities
        WHERE canonical_name IN ({placeholders})
           OR {CODE_PATTERN_SQL}
    """, POLLUTED_PATTERNS)

    polluted_entities = cursor.fetchall()

    for row in polluted_entities:
        entity_type = row["entity_type"]
        stats["entities_to_delete"] += 1
        stats["mentions_to_delete"] += row["mention_count"]
        stats["polluted_by_type"][entity_type] = stats["polluted_by_type"].get(entity_type, 0) + 1

        logger.info(f"POLLUTED_ENTITY: {row['entity_type']}/{row['canonical_name']} ({row['mention_count']} mentions)")

    if not dry_run and polluted_entities:
        entity_ids = [row["entity_id"] for row in polluted_entities]
        placeholders = ",".join("?" for _ in entity_ids)

        # Delete mentions first (foreign key constraint)
        cursor.execute(f"""
            DELETE FROM entity_mentions WHERE entity_id IN ({placeholders})
        """, entity_ids)

        # Delete entities
        cursor.execute(f"""
            DELETE FROM entities WHERE entity_id IN ({placeholders})
        """, entity_ids)

        conn.commit()
        logger.info(f"CLEANUP_COMPLETE: Deleted {stats['entities_to_delete']} entities and {stats['mentions_to_delete']} mentions")

    conn.close()
    return stats

# ===| VALIDATION |===

@dataclass
class ValidationResult:
    """Result of validation check."""

    is_valid: bool
    error_count: int = 0
    warning_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str):
        """Add an error to the result."""
        self.errors.append(message)
        self.error_count += 1
        self.is_valid = False
        logger.error(f"VALIDATION_ERROR: {message}")

    def add_warning(self, message: str):
        """Add a warning to the result."""
        self.warnings.append(message)
        self.warning_count += 1
        logger.warning(f"VALIDATION_WARNING: {message}")

    def add_stat(self, key: str, value: Any):
        """Add a statistic to the result."""
        self.stats[key] = value

    def summary(self) -> str:
        """Generate a summary report."""
        lines = [
            "=" * 80,
            "VALIDATION SUMMARY",
            "=" * 80,
            f"Status: {'✓ PASSED' if self.is_valid else '✗ FAILED'}",
            f"Errors: {self.error_count}",
            f"Warnings: {self.warning_count}",
            "",
            "Statistics:",
        ]

        for key, value in sorted(self.stats.items()):
            lines.append(f"  {key}: {value}")

        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for i, error in enumerate(self.errors[:20], 1):  # Limit to first 20
                lines.append(f"  {i}. {error}")
            if len(self.errors) > 20:
                lines.append(f"  ... and {len(self.errors) - 20} more errors")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for i, warning in enumerate(self.warnings[:20], 1):  # Limit to first 20
                lines.append(f"  {i}. {warning}")
            if len(self.warnings) > 20:
                lines.append(f"  ... and {len(self.warnings) - 20} more warnings")

        lines.append("=" * 80)
        return "\n".join(lines)

class Stage2Validator:
    """
    Validator for Stage 2 Personal Lexicon Pipeline output.

    Performs comprehensive validation including:
    - Code pollution detection in entities
    - Blocklist checking
    - Data integrity verification
    - Statistics consistency
    - Foreign key constraints
    """

    def __init__(self, db_path: Path, strict_mode: bool = False):
        """
        Initialize validator.

        Args:
            db_path: Path to the SQLite database
            strict_mode: If True, apply stricter validation rules
        """
        self.db_path = db_path
        self.strict_mode = strict_mode
        self.conn: Optional[sqlite3.Connection] = None
        self.code_filter = CodePollutionFilter(strict_mode=strict_mode)
        self.result = ValidationResult(is_valid=True)

    def __enter__(self):
        """Context manager entry."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.conn:
            self.conn.close()

    def validate_all(self) -> ValidationResult:
        """
        Run all validation checks.

        Returns:
            ValidationResult with all findings
        """
        logger.info(f"Starting validation of {self.db_path}")
        logger.info(f"Strict mode: {self.strict_mode}")

        # Reset result
        self.result = ValidationResult(is_valid=True)

        # Run all checks
        self._check_database_structure()
        self._check_entity_code_pollution()
        self._check_entity_blocklist()
        self._check_entity_mention_consistency()
        self._check_candidate_suppression()
        self._check_time_mention_resolution()
        self._check_foreign_key_integrity()
        self._check_statistics_consistency()
        self._check_overlap_groups()
        self._collect_summary_statistics()

        logger.info(f"Validation complete: {self.result.error_count} errors, {self.result.warning_count} warnings")
        return self.result

    def _check_database_structure(self):
        """Verify that all required tables and columns exist."""
        logger.info("Checking database structure...")

        required_tables = [
            "entities",
            "entity_mention_candidates",
            "entity_mentions",
            "time_mentions",
            "messages",
        ]

        cursor = self.conn.cursor()

        for table in required_tables:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if not cursor.fetchone():
                self.result.add_error(f"Missing required table: {table}")

    def _check_entity_code_pollution(self):
        """Check entities table for code pollution."""
        logger.info("Checking entities for code pollution...")

        cursor = self.conn.cursor()
        cursor.execute("""
                       SELECT entity_id, entity_type, canonical_name, entity_key, mention_count
                       FROM entities
                       WHERE status = 'active'
                       """)

        polluted_count = 0
        for row in cursor:
            entity_type = row["entity_type"]
            canonical_name = row["canonical_name"]
            entity_key = row["entity_key"]

            # Check canonical name
            result = self.code_filter.is_code_pollution(canonical_name, entity_type)
            if result.is_code:
                polluted_count += 1
                self.result.add_error(f"Code pollution in entity {row['entity_id']}: {entity_type}/{canonical_name} - {result.reason}")

            # Check entity key
            result = self.code_filter.is_code_pollution(entity_key, entity_type)
            if result.is_code:
                polluted_count += 1
                self.result.add_error(f"Code pollution in entity key {row['entity_id']}: {entity_type}/{entity_key} - {result.reason}")

        self.result.add_stat("entities_with_code_pollution", polluted_count)

    def _check_entity_blocklist(self):
        """Check entities against known blocklists."""
        logger.info("Checking entities against blocklists...")

        cursor = self.conn.cursor()

        # Check exact blocklist matches
        exact_blocklist = list(CodePollutionFilter.EXACT_BLOCKLIST)
        placeholders = ",".join("?" * len(exact_blocklist))

        cursor.execute(
            f"""
            SELECT entity_id, entity_type, canonical_name, mention_count
            FROM entities
            WHERE LOWER(canonical_name) IN ({placeholders})
              AND status = 'active'
        """,
            exact_blocklist,
        )

        blocklisted_count = 0
        for row in cursor:
            blocklisted_count += 1
            self.result.add_error(f"Blocklisted entity {row['entity_id']}: {row['entity_type']}/{row['canonical_name']} ({row['mention_count']} mentions)")

        self.result.add_stat("blocklisted_entities", blocklisted_count)

        ner_blocklist = list(NERDetector.NER_BLOCKLIST)
        placeholders = ",".join("?" * len(ner_blocklist))

        cursor.execute(
            f"""
            SELECT entity_id, entity_type, canonical_name, mention_count
            FROM entities
            WHERE entity_type IN ('PERSON', 'ORG', 'LOCATION', 'OTHER')
              AND canonical_name IN ({placeholders})
              AND status = 'active'
        """,
            ner_blocklist,
        )

        ner_blocklisted_count = 0
        for row in cursor:
            ner_blocklisted_count += 1
            self.result.add_error(f"NER-blocklisted entity {row['entity_id']}: {row['entity_type']}/{row['canonical_name']} ({row['mention_count']} mentions)")

        self.result.add_stat("ner_blocklisted_entities", ner_blocklisted_count)

    def _check_entity_mention_consistency(self):
        """Check consistency between entities, candidates, and mentions."""
        logger.info("Checking entity mention consistency...")

        cursor = self.conn.cursor()

        # Check that all mentions reference valid entities
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mentions em
                           LEFT JOIN entities e ON em.entity_id = e.entity_id
                       WHERE e.entity_id IS NULL
                       """)
        orphan_mentions = cursor.fetchone()["count"]
        if orphan_mentions > 0:
            self.result.add_error(f"Found {orphan_mentions} entity mentions with invalid entity_id")

        # Check that all mentions reference valid candidates
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mentions em
                           LEFT JOIN entity_mention_candidates c ON em.candidate_id = c.candidate_id
                       WHERE c.candidate_id IS NULL
                       """)
        orphan_mention_candidates = cursor.fetchone()["count"]
        if orphan_mention_candidates > 0:
            self.result.add_error(f"Found {orphan_mention_candidates} entity mentions with invalid candidate_id")

        # Check that emitted mentions have is_eligible=1 in candidates
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mentions em
                           JOIN entity_mention_candidates c ON em.candidate_id = c.candidate_id
                       WHERE c.is_eligible = 0
                       """)
        ineligible_mentions = cursor.fetchone()["count"]
        if ineligible_mentions > 0:
            self.result.add_error(f"Found {ineligible_mentions} entity mentions with is_eligible=0 candidate")

        # Check that suppressed candidates are not emitted
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mention_candidates c
                       WHERE c.is_eligible = 0
                         AND EXISTS (
                           SELECT 1 FROM entity_mentions em
                           WHERE em.candidate_id = c.candidate_id
                           )
                       """)
        suppressed_but_emitted = cursor.fetchone()["count"]
        if suppressed_but_emitted > 0:
            self.result.add_error(f"Found {suppressed_but_emitted} suppressed candidates that were emitted")

        # Check surface_text consistency
        cursor.execute("""
                       SELECT em.mention_id, em.surface_text as mention_text,
                              c.surface_text as candidate_text
                       FROM entity_mentions em
                                JOIN entity_mention_candidates c ON em.candidate_id = c.candidate_id
                       WHERE em.surface_text != c.surface_text
               OR (em.surface_text IS NULL AND c.surface_text IS NOT NULL)
               OR (em.surface_text IS NOT NULL AND c.surface_text IS NULL)
            LIMIT 100
                       """)
        inconsistent_surface = cursor.fetchall()
        if inconsistent_surface:
            self.result.add_warning(f"Found {len(inconsistent_surface)} mentions with inconsistent surface_text")

    def _check_candidate_suppression(self):
        """Check that candidate suppression is properly recorded."""
        logger.info("Checking candidate suppression...")

        cursor = self.conn.cursor()

        # Check that suppressed candidates have suppression_reason
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mention_candidates
                       WHERE is_eligible = 0
                         AND suppression_reason IS NULL
                       """)
        missing_reason = cursor.fetchone()["count"]
        if missing_reason > 0:
            self.result.add_error(f"Found {missing_reason} suppressed candidates without suppression_reason")

        valid_reasons = {r.value for r in SuppressionReason}

        cursor.execute("""
                       SELECT DISTINCT suppression_reason
                       FROM entity_mention_candidates
                       WHERE is_eligible = 0
                         AND suppression_reason IS NOT NULL
                       """)
        for row in cursor:
            reason = row["suppression_reason"]
            if reason not in valid_reasons:
                self.result.add_warning(f"Invalid suppression reason: '{reason}'")

        # Count candidates by suppression reason
        cursor.execute("""
                       SELECT suppression_reason, COUNT(*) as count
                       FROM entity_mention_candidates
                       WHERE is_eligible = 0
                       GROUP BY suppression_reason
                       """)
        for row in cursor:
            self.result.add_stat(f"suppressed_{row['suppression_reason']}", row["count"])

    def _check_time_mention_resolution(self):
        """Check time mention resolution quality."""
        logger.info("Checking time mention resolution...")

        cursor = self.conn.cursor()

        # Check that all time mentions have required fields
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM time_mentions
                       WHERE surface_text IS NULL
                          OR pattern_id IS NULL
                          OR resolved_type IS NULL
                       """)
        incomplete = cursor.fetchone()["count"]
        if incomplete > 0:
            self.result.add_error(f"Found {incomplete} time mentions with missing required fields")

        # Check resolved vs unresolved counts
        cursor.execute("""
                       SELECT resolved_type, COUNT(*) as count
                       FROM time_mentions
                       GROUP BY resolved_type
                       """)
        for row in cursor:
            self.result.add_stat(f"time_mentions_{row['resolved_type']}", row["count"])

        # Check that resolved times have valid timestamps
        cursor.execute("""
                       SELECT time_mention_id, resolved_type, valid_from_utc, valid_to_utc
                       FROM time_mentions
                       WHERE resolved_type IN ('instant', 'interval')
                         AND valid_from_utc IS NULL
                           LIMIT 100
                       """)
        missing_timestamps = cursor.fetchall()
        if missing_timestamps:
            self.result.add_error(f"Found {len(missing_timestamps)} resolved time mentions without valid_from_utc")

        # Check interval consistency (valid_from < valid_to)
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM time_mentions
                       WHERE resolved_type = 'interval'
                         AND valid_from_utc IS NOT NULL
                         AND valid_to_utc IS NOT NULL
                         AND valid_from_utc >= valid_to_utc
                       """)
        invalid_intervals = cursor.fetchone()["count"]
        if invalid_intervals > 0:
            self.result.add_error(f"Found {invalid_intervals} time mentions with invalid intervals (from >= to)")

    def _check_foreign_key_integrity(self):
        """Check foreign key constraints."""
        logger.info("Checking foreign key integrity...")

        cursor = self.conn.cursor()

        # Check entity_mentions -> messages
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mentions em
                           LEFT JOIN messages m ON em.message_id = m.message_id
                       WHERE m.message_id IS NULL
                       """)
        orphan = cursor.fetchone()["count"]
        if orphan > 0:
            self.result.add_error(f"Found {orphan} entity mentions with invalid message_id")

        # Check time_mentions -> messages
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM time_mentions tm
                           LEFT JOIN messages m ON tm.message_id = m.message_id
                       WHERE m.message_id IS NULL
                       """)
        orphan = cursor.fetchone()["count"]
        if orphan > 0:
            self.result.add_error(f"Found {orphan} time mentions with invalid message_id")

        # Check entity_mention_candidates -> messages
        cursor.execute("""
                       SELECT COUNT(*) as count
                       FROM entity_mention_candidates c
                           LEFT JOIN messages m ON c.message_id = m.message_id
                       WHERE m.message_id IS NULL
                       """)
        orphan = cursor.fetchone()["count"]
        if orphan > 0:
            self.result.add_error(f"Found {orphan} entity candidates with invalid message_id")

    def _check_statistics_consistency(self):
        """Check that entity statistics are consistent with actual data."""
        logger.info("Checking statistics consistency...")

        cursor = self.conn.cursor()

        # Check mention_count consistency
        cursor.execute("""
                       SELECT e.entity_id, e.canonical_name, e.mention_count,
                              COUNT(em.mention_id) as actual_count
                       FROM entities e
                                LEFT JOIN entity_mentions em ON e.entity_id = em.entity_id
                       WHERE e.status = 'active'
                       GROUP BY e.entity_id, e.canonical_name, e.mention_count
                       HAVING e.mention_count != actual_count
            LIMIT 100
                       """)
        inconsistent = cursor.fetchall()
        if inconsistent:
            self.result.add_error(f"Found {len(inconsistent)} entities with inconsistent mention_count")
            for row in inconsistent[:10]:  # Show first 10
                self.result.add_error(f"  Entity {row['canonical_name']}: stored={row['mention_count']}, actual={row['actual_count']}")

        # Check conversation_count consistency
        cursor.execute("""
                       SELECT e.entity_id, e.canonical_name, e.conversation_count,
                              COUNT(DISTINCT m.conversation_id) as actual_count
                       FROM entities e
                                LEFT JOIN entity_mentions em ON e.entity_id = em.entity_id
                                LEFT JOIN messages m ON em.message_id = m.message_id
                       WHERE e.status = 'active'
                       GROUP BY e.entity_id, e.canonical_name, e.conversation_count
                       HAVING e.conversation_count != actual_count
            LIMIT 100
                       """)
        inconsistent = cursor.fetchall()
        if inconsistent:
            self.result.add_error(f"Found {len(inconsistent)} entities with inconsistent conversation_count")

    def _check_overlap_groups(self):
        """Check overlap group consistency."""
        logger.info("Checking overlap groups...")

        cursor = self.conn.cursor()

        # Check that overlap groups contain multiple candidates
        cursor.execute("""
                       SELECT overlap_group_id, COUNT(*) as count
                       FROM entity_mention_candidates
                       WHERE overlap_group_id IS NOT NULL
                       GROUP BY overlap_group_id
                       HAVING count = 1
                       """)
        singleton_groups = cursor.fetchall()
        if singleton_groups and self.strict_mode:
            self.result.add_warning(f"Found {len(singleton_groups)} overlap groups with single candidate")

        # Check that overlapping candidates in same group actually overlap
        cursor.execute("""
                       SELECT overlap_group_id, message_id,
                              MIN(char_start) as min_start,
                              MAX(char_end) as max_end,
                              COUNT(*) as count
                       FROM entity_mention_candidates
                       WHERE overlap_group_id IS NOT NULL
                         AND char_start IS NOT NULL
                         AND char_end IS NOT NULL
                       GROUP BY overlap_group_id, message_id
                       HAVING count > 1
                       """)
        groups = cursor.fetchall()

        for group in groups:
            group_id = group["overlap_group_id"]
            message_id = group["message_id"]

            # Get all candidates in this group
            cursor.execute(
                """
                           SELECT candidate_id, char_start, char_end, surface_text
                           FROM entity_mention_candidates
                           WHERE overlap_group_id = ?
                             AND message_id = ?
                             AND char_start IS NOT NULL
                             AND char_end IS NOT NULL
                           ORDER BY char_start, char_end
                           """,
                (group_id, message_id),
            )

            candidates = cursor.fetchall()
            if len(candidates) < 2:
                continue

            # Check pairwise overlaps
            has_overlap = False
            for i, c1 in enumerate(candidates):
                for c2 in candidates[i + 1 :]:
                    if c1["char_start"] < c2["char_end"] and c2["char_start"] < c1["char_end"]:
                        has_overlap = True
                        break
                if has_overlap:
                    break

            if not has_overlap and self.strict_mode:
                self.result.add_warning(f"Overlap group {group_id} contains non-overlapping candidates")

    def _collect_summary_statistics(self):
        """Collect summary statistics for the report."""
        logger.info("Collecting summary statistics...")

        cursor = self.conn.cursor()

        # Entity counts by type
        cursor.execute("""
                       SELECT entity_type, COUNT(*) as count
                       FROM entities
                       WHERE status = 'active'
                       GROUP BY entity_type
                       """)
        for row in cursor:
            self.result.add_stat(f"entities_{row['entity_type']}", row["count"])

        # Total counts
        cursor.execute("SELECT COUNT(*) as count FROM entities WHERE status = 'active'")
        self.result.add_stat("total_entities", cursor.fetchone()["count"])

        cursor.execute("SELECT COUNT(*) as count FROM entity_mentions")
        self.result.add_stat("total_entity_mentions", cursor.fetchone()["count"])

        cursor.execute("SELECT COUNT(*) as count FROM entity_mention_candidates")
        self.result.add_stat("total_entity_candidates", cursor.fetchone()["count"])

        cursor.execute("SELECT COUNT(*) as count FROM time_mentions")
        self.result.add_stat("total_time_mentions", cursor.fetchone()["count"])

        # Candidate eligibility
        cursor.execute("""
                       SELECT is_eligible, COUNT(*) as count
                       FROM entity_mention_candidates
                       GROUP BY is_eligible
                       """)
        for row in cursor:
            eligible = "eligible" if row["is_eligible"] else "suppressed"
            self.result.add_stat(f"candidates_{eligible}", row["count"])

        # Top entities by mention count
        cursor.execute("""
                       SELECT entity_type, canonical_name, mention_count
                       FROM entities
                       WHERE status = 'active'
                       ORDER BY mention_count DESC
                           LIMIT 10
                       """)
        top_entities = []
        for row in cursor:
            top_entities.append(f"{row['entity_type']}/{row['canonical_name']} ({row['mention_count']})")
        self.result.add_stat("top_10_entities", top_entities)

def validate_stage2_output(db_path: Path, strict_mode: bool = False, print_summary: bool = True) -> ValidationResult:
    """
    Validate Stage 2 pipeline output.

    Args:
        db_path: Path to the SQLite database
        strict_mode: If True, apply stricter validation rules
        print_summary: If True, print validation summary

    Returns:
        ValidationResult with all findings
    """
    with Stage2Validator(db_path, strict_mode=strict_mode) as validator:
        result = validator.validate_all()

    if print_summary:
        print(result.summary())

    return result

# ===| ENTRY POINT |===

def run_stage2(config: Stage2Config) -> None:
    """Run Stage 2 pipeline on existing database."""
    pipeline = PersonalLexiconPipeline(config)
    pipeline.run()

    result = validate_stage2_output(config.output_file_path, strict_mode=True, print_summary=True)
    if not result.is_valid:
        logger.warning(f"Validation failed with {result.error_count} errors")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Run Stage 2: Personal Lexicon Layer")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("../data/output/kg.db"),
        help="Path to the SQLite database file (default: kg.db)",
    )
    parser.add_argument(
        "--enable-ner",
        default=True,
        action="store_true",
        help="Enable NER-based entity detection (requires spacy) (default: True)",
    )
    parser.add_argument(
        "--disable-ner",
        dest="enable_ner",
        action="store_false",
        help="Disable NER-based entity detection",
    )
    parser.add_argument(
        "--ignore-blockquotes",
        default=True,
        action="store_true",
        help="Exclude blockquotes from entity detection (default: True)",
    )
    parser.add_argument(
        "--include-blockquotes",
        dest="ignore_blockquotes",
        action="store_false",
        help="Include blockquotes in entity detection",
    )
    parser.add_argument(
        "--timezone",
        type=str,
        default="UTC",
        help="Default timezone for time resolution (default: UTC)",
    )

    args = parser.parse_args()

    config = Stage2Config(
        output_file_path=args.db,
        enable_ner=args.enable_ner,
        ignore_markdown_blockquotes=args.ignore_blockquotes,
        anchor_timezone=args.timezone,
        domain_tld_allowlist_enabled=True,  # Always enable for production
    )

    run_stage2(config)