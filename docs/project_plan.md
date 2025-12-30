# Personal Bitemporal Knowledge Graph from ChatGPT `conversations.json`

Build a **semantic-lossless, bitemporal** personal knowledge graph from a ChatGPT export with **conservative temporal anchoring**, reasonably deterministic, but employing LLMs/NER, fully logged, replayable.

### Key principles (deterministic, auditable)

1. **Lossless raw preservation first**: always store JCS-canonical raw conversation/message/part JSON strings.
2. **Offsets are sacred**: all spans refer to **stored** `messages.text_raw`.
3. **Determinism everywhere**:

   * stable ordering rules
   * store full raw I/O for any external model call (e.g., NER, LLM)
4. **Role-aware trust**:

   * user text is primary evidence for personal facts
   * assistant claims about the user are low-trust unless directly corroborated by user evidence
5. **Every stage transactional**: a stage either commits fully or rolls back fully.


### UTC storage policy

* Store timestamps as UTC ISO-8601 strings: `YYYY-MM-DDTHH:MM:SS.sssZ`
* Normalize incoming timestamps to UTC before storage
* Missing timestamps: store `NULL`; track quality in `timestamp_quality`


### Text offsets

* `char_start`, `char_end` are 0-based Unicode codepoint indices into `messages.text_raw`
* `char_end` is exclusive
* If offsets are not reliable: store both `NULL` and log WARN `OFFSET_UNRELIABLE` with detector/model + reason


### Interval semantics

* Use half-open intervals `[start, end)` when `end` non-null
* Open-ended: `[start, ∞)` when `end` null
* Instants: `valid_time_type='instant'` implies `valid_to_utc IS NULL`


### Canonical JSON (JCS, RFC 8785)

* JCS-canonical JSON string = RFC 8785 output
* All JSON hashes are computed on UTF-8 bytes of the JCS-canonical string
* Pin the exact canonicalization library in the build (store in `build_meta.notes` or config fields)

* Identifier policy (deterministic IDs) -- Namespace: All uuid5 IDs use `KG_NS_UUID` from config.
* UUIDv5 input encoding: uuid5 name input is a **JCS-canonical JSON array** to avoid delimiter ambiguity.
* Canonical NULL encoding in ID components
  * Missing fields included in ID components use the literal JSON string `"__NULL__"`.
  * Logically empty but present uses `"__EMPTY__"`.


### Deterministic ID formulas (examples)

* `conversation_id = uuid5(KG_NS_UUID, JCS(["conversation", sha256_hex(raw_conversation_json_jcs)]))` (when export conversation_id is NULL)
* `part_id = uuid5(KG_NS_UUID, JCS(["part", message_id, part_index]))`

---


## Stage 0: Preprocessing

### 0.1 Goals

* Every run produces a **new SQLite snapshot** (`build_id` unique per run).

### 0.2 Required checks

Assume that `conversations.json` is verified and `export_mapping.yaml` is available

1. **Load export mapping** - Reads and validates `export_mapping.yaml` 
2. **Load invalidation rules** - Reads `invalidation_rules.yaml` if present (for Stage 4)
3. **Initialize database** - Creates SQLite file and schema (3 tables only for Stage 1)
4. **Set up ID generation** - Configures UUIDv5 namespace
5. **Enable foreign keys** - Sets `PRAGMA foreign_keys=ON`

### 0.3 Export mapping schema (`export_mapping.yaml`)

All JSON pointers are RFC 6901.

Required fields:

* `format_version` (TEXT, default="1.0")
* `conversation_id_path` (TEXT; JSON pointer; nullable)
* `conversation_title_path` (TEXT; JSON pointer; nullable)
* `conversation_created_path` (TEXT; JSON pointer; nullable)
* `conversation_updated_path` (TEXT; JSON pointer; nullable)
* `messages_path` (TEXT; JSON pointer, default="/mapping")
* `messages_is_mapping` (BOOLEAN, default=True)
* `message_id_path` (TEXT; JSON pointer relative to message object; nullable)
* `message_role_path` (TEXT; JSON pointer relative to message object; nullable)
* `message_parent_path` (TEXT; JSON pointer relative to message object; nullable)
* `message_created_path` (TEXT; JSON pointer relative to message object; nullable)
* `message_content_path` (TEXT; JSON pointer relative to message object; nullable)
* `role_mapping` (DICT mapping export roles → normalized roles)
* `content_part_rules` (LIST; ordered rules; see §0.4)
* `known_attachment_paths` (LIST of TEXT; not actively used in current implementation)

**Validation:**
* Basic YAML loading and dictionary conversion
* No strict validation of JSON pointer syntax
* No enforcement of required fields (uses defaults)

### 0.4 Content Part Rules Schema

Each rule object in `content_part_rules`:

* `match_path`: JSON pointer relative to the part object
* `match_value`: TEXT or LIST of TEXT; `"*"` means "any non-null"
* `part_type`: `text|image|file|tool_call|tool_result|other`
* `text_extract_path`: JSON pointer relative to part object (optional)
* `mime_type_path`: JSON pointer relative to part object (optional)
* `file_path_path`: JSON pointer relative to part object (optional)
* `metadata_paths`: LIST of JSON pointers relative to part object (optional)

**Rule evaluation:**
* Rules evaluated in order; first matching rule assigns `part_type`
* If no rule matches, fallback logic tries to extract text from common fields (`text`, `content`, `value`, `parts`)
* If still no match, defaults to `part_type='other'`

**Special handling:**
* **Thoughts content** - Special case for `content_type="thoughts"` with custom extraction
* **Array content** - Lists are joined together for text extraction
* **Structured content** - Dict items are inspected for text fields

### 0.5 Pipeline Configuration Schema

The following configuration parameters are used across stages:

**ID Generation (Stage 0+):**
* `id_namespace` (TEXT): UUIDv5 namespace UUID (default: `"550e8400-e29b-41d4-a716-446655440000"`)

**Time Handling (Stage 1+):**
* `anchor_timezone` (TEXT): Default timezone for time resolution (default: `"UTC"`)

**Entity Detection (Stage 2):**
* `ignore_markdown_blockquotes` (BOOLEAN): Exclude blockquotes from entity detection (default: `false`)
* `domain_tld_allowlist_enabled` (BOOLEAN): Validate domains against TLD allowlist (default: `false`)
* `enable_ner` (BOOLEAN): Enable NER-based entity detection (default: `true`)
* `ner_max_chars` (INTEGER): Max characters per NER chunk (default: `10000`)
* `ner_stride` (INTEGER): Overlap stride for NER chunking (default: `1000`)
* `ner_label_allowlist` (LIST): Allowed NER label types (default: `["PERSON", "ORG", "LOCATION"]`)
* `emit_spanless_ner` (BOOLEAN): Emit NER candidates without offsets (default: `false`)
* `url_sort_query_params` (BOOLEAN): Sort URL query parameters (default: `false`)

**Assertion Extraction (Stage 3):**
* `enable_llm_assertion_extraction` (BOOLEAN): Enable LLM-based extraction (default: `false`)
* `llm_temperature` (REAL): LLM temperature parameter (default: `0`)
* `llm_top_p` (REAL): LLM top_p parameter (default: `1`)
* `llm_seed` (INTEGER): LLM seed for reproducibility (optional)
* `max_retries_llm` (INTEGER): Max LLM call retries (default: `3`)
* `llm_multi_run_count` (INTEGER): Number of LLM runs per message (default: `1`)
* `k_context` (INTEGER): Number of preceding messages for context (default: `5`)
* `code_fence_masking` (TEXT): Code fence masking strategy (default: `"whitespace_preserve_length"`)
* `threshold_link_string_sim` (REAL): String similarity threshold for predicate matching (default: `0.85`)
* `assertion_upsert_policy` (TEXT): Deduplication policy (default: `"keep_highest_confidence"`)

**Trust Weights (Stage 3):**
* `trust_weight_user` (REAL): Trust weight for user assertions (default: `1.0`)
* `trust_weight_assistant_corroborated` (REAL): Trust weight for corroborated assistant assertions (default: `0.8`)
* `trust_weight_assistant_uncorroborated` (REAL): Trust weight for uncorroborated assistant assertions (default: `0.4`)

**Corroboration (Stage 3):**
* `coref_window_size` (INTEGER): Window size for corroboration detection (default: `5`)

**Temporal Reasoning (Stage 4):**
* `time_link_proximity_chars` (INTEGER): Max character distance for time linking (default: `200`)
* `time_link_min_alignment` (REAL): Minimum alignment score for time linking (default: `0.1`)
* `fallback_valid_from_asserted` (BOOLEAN): Use asserted_at as fallback valid_from (default: `true`)
* `threshold_close` (REAL): Confidence threshold for closing assertions (default: `0.7`)

---


## Stage 1: Raw Ingest (Import)

### 1.1 Goals

1. Parse `conversations.json` with **no semantic loss**
2. Flatten hierarchy → 3 core tables (conversations, messages, message_parts)
3. Compute tree paths for message threading
4. Extract text from parts into `messages.text_raw` for downstream processing
5. Detect code fences and blockquotes in text
6. Store everything with deterministic IDs and canonical JSON

### 1.2 Database schema (Stage 1 tables)

#### **conversations**

| column                      | type     | notes                                    |
| --------------------------- | -------- | ---------------------------------------- |
| `conversation_id` TEXT PK   |          | UUIDv5 from §1.3                         |
| `export_conversation_id`    | TEXT     | Original ID from export (if available)   |
| `title`                     | TEXT     | Conversation title (nullable)            |
| `created_at_utc`            | TEXT     | ISO-8601 UTC timestamp (nullable)        |
| `updated_at_utc`            | TEXT     | ISO-8601 UTC timestamp (nullable)        |
| `message_count`             | INTEGER  | Number of messages (default 0)           |
| `raw_conversation_json`     | TEXT     | JCS-canonical JSON of entire object      |

**Indices:**
* Primary key on `conversation_id`

#### **messages**

| column                      | type     | notes                                    |
| --------------------------- | -------- | ---------------------------------------- |
| `message_id` TEXT PK        |          | From export or UUIDv5 from §1.3          |
| `conversation_id` TEXT FK   |          | → conversations                          |
| `role`                      | TEXT     | Normalized role (user/assistant/system/tool/unknown) |
| `parent_id`                 | TEXT     | Parent message ID (nullable)             |
| `tree_path`                 | TEXT     | Slash-separated path (e.g., "0/2/5")     |
| `order_index`               | INTEGER  | Deterministic ordering within conversation |
| `created_at_utc`            | TEXT     | ISO-8601 UTC timestamp (nullable)        |
| `timestamp_quality`         | TEXT     | original/imputed_parent/imputed_prior    |
| `content_type`              | TEXT     | text/mixed/empty/unknown                 |
| `text_raw`                  | TEXT     | Extracted text for analysis (nullable)   |
| `text_part_map_json`        | TEXT     | JCS-canonical mapping if multi-part (nullable) |
| `code_fence_ranges_json`    | TEXT     | JCS-canonical detected code blocks (nullable) |
| `blockquote_ranges_json`    | TEXT     | JCS-canonical detected blockquotes (nullable) |
| `attachment_count`          | INTEGER  | Number of file/image parts (default 0)   |
| `raw_message_json`          | TEXT     | JCS-canonical JSON of entire message     |

**Indices:**
* Primary key on `message_id`
* `idx_messages_conversation` on `conversation_id`
* `idx_messages_order` on `(conversation_id, order_index)`

#### **message_parts**

| column                      | type     | notes                                    |
| --------------------------- | -------- | ---------------------------------------- |
| `part_id` TEXT PK           |          | UUIDv5 from §1.3                         |
| `message_id` TEXT FK        |          | → messages                               |
| `part_index`                | INTEGER  | 0-based position in message              |
| `part_type`                 | TEXT     | text/image/file/tool_call/tool_result/other |
| `text_content`              | TEXT     | Extracted text (nullable)                |
| `mime_type`                 | TEXT     | MIME type if available (nullable)        |
| `file_path`                 | TEXT     | File reference if available (nullable)   |
| `metadata_json`             | TEXT     | JCS-canonical extra metadata (nullable)  |
| `raw_part_json`             | TEXT     | JCS-canonical JSON of part object        |

**Indices:**
* Primary key on `part_id`
* `idx_parts_message` on `message_id`

### 1.3 Deterministic ID Generation

**Namespace:**
* Default: `"550e8400-e29b-41d4-a716-446655440000"` (configurable via `PipelineConfig.id_namespace`)

**NULL encoding:**
* `None` → `"__NULL__"`
* `""` (empty string) → `"__EMPTY__"`

**ID formulas:**
* `conversation_id = uuid5(KG_NS_UUID, JCS(["conversation", sha256_hex(raw_conversation_json_jcs)]))` (when export conversation_id is NULL)
* `message_id = uuid5(KG_NS_UUID, JCS(["message", conversation_id, message_index_or_hash]))` (when export message_id is NULL)
* `part_id = uuid5(KG_NS_UUID, JCS(["part", message_id, part_index]))`

### 1.4 Canonical JSON (JCS) 

**Implementation:** Custom `JCS` class implementing RFC 8785 subset

**Rules:**
* Sort object keys by UTF-16 BE encoding (for ASCII: lexicographic)
* No whitespace
* Precise float representation (no Infinity/NaN)
* Control characters escaped as `\uXXXX`
* All stored JSON passes through `JCS.canonicalize(obj) -> str`

### 1.5 Timestamp Handling

**Storage format:** `YYYY-MM-DDTHH:MM:SS.sssZ` (ISO-8601 UTC with milliseconds)

**Key methods:**
* `now_utc()` - Current time in canonical format
* `normalize_to_utc(timestamp, source_tz=None)` - Convert any timestamp to canonical UTC
  * Handles: Unix timestamps (int/float), Python datetime, ISO strings
  * Returns `None` on parse failure (no exceptions)

**Timestamp quality tracking:**
* `original` - Direct from export
* `imputed_parent` - Inherited from parent message
* `imputed_prior` - Estimated from prior message in sequence

**Imputation logic:**
* If message has no timestamp but has parent → copy parent's timestamp, set quality to `imputed_parent`
* If no parent, find previous message in tree order → copy timestamp, set quality to `imputed_prior`
* All imputation happens during tree path computation phase

### 1.6 Tree path computation

**Purpose:** Establish deterministic message ordering for threaded conversations

**Tree path format:** Slash-separated depth-first indices (e.g., `"0/2/5"`)

**Algorithm (`_compute_tree_paths_and_indices` method):**

1. **Build parent-child mapping** for all messages in conversation
2. **Identify roots** - messages with `parent_id IS NULL`
3. **Sort roots** deterministically by `message_id ASC`
4. **Depth-first traversal:**
   * For each root, recursively traverse children
   * Children sorted by `message_id ASC` at each level
   * Assign tree path: `parent_path + "/" + child_index`
   * Root messages get simple indices: `"0"`, `"1"`, `"2"`
5. **Handle orphaned messages** - Messages with non-existent parent_id are treated as roots
6. **Timestamp imputation** during traversal when timestamps are missing
7. **Compute order indices** - Sort all messages by `(tree_path ASC, message_id ASC)`, assign sequential indices

**Update message records** with computed `tree_path` and `order_index`

### 1.7 Text Extraction

**Goal:** Build `messages.text_raw` from message parts for downstream analysis

**Rules (`_extract_text` method):**

1. **Collect text parts** - Find all parts with `text_content` not NULL
2. **No text parts:**
   * If no parts at all → `content_type='empty'`
   * If parts exist but no text → `content_type='unknown'`
   * `text_raw=NULL`
3. **Single text part:**
   * `text_raw = part.text_content` (direct copy)
   * `content_type='text'`
   * `text_part_map_json=NULL`
4. **Multiple text parts:**
   * Join with `"\n\n"` separator
   * `content_type='mixed'`
   * Build `text_part_map_json` as JCS-canonical array:
     [
       {"part_index": 0, "char_start": 0, "char_end": 150},
       {"part_index": 2, "char_start": 152, "char_end": 300}
     ]
   * Character positions are 0-based Unicode codepoints

### 1.8 Code Fence Detection

**Purpose:** Identify Markdown code blocks for later exclusion from entity extraction

**Algorithm (`_detect_code_fences` method):**

1. **Pattern:** Match opening fence: `^(\`\`\`+)(\w+)?\s*$` (3+ backticks, optional language)
2. **For each opening fence:**
   * Find matching close: same or more backticks, no language specifier
   * If found: record `[fence_start, fence_end)` range
   * If not found: extend to end of text (unclosed fence)
3. **Store as `code_fence_ranges_json`** - JCS-canonical array:
   [
     {"char_start": 120, "char_end": 450, "language": "python"},
     {"char_start": 600, "char_end": 890, "language": null}
   ]


**Character offsets:** 0-based Unicode codepoints into `text_raw`

### 1.9 Blockquote Detection

**Purpose:** Identify quoted content for potential special handling

**Algorithm (`_detect_blockquotes` method):**

1. **Pattern:** `^\s*>.*$` (line starting with `>`, possibly with leading whitespace)
2. **Match all lines** in text using `re.MULTILINE`
3. **Store as `blockquote_ranges_json`** - JCS-canonical array:
   [
     {"char_start": 45, "char_end": 89},
     {"char_start": 90, "char_end": 134}
   ]

**Character offsets:** 0-based Unicode codepoints into `text_raw`, covering entire line including newline

### 1.10 Content Part Classification

**Method:** `_classify_part(part_data: dict) -> tuple[part_type, text, mime, file, metadata]`

**Special case - Thoughts content:**

**Rule-based classification:**
1. Evaluate `content_part_rules` in order
2. For each rule:
   * Extract value at `match_path`
   * Check if it matches `match_value` (supports `"*"`, single value, or list)
   * If match: extract text, mime, file, metadata per rule paths
   * Return with specified `part_type`

**Fallback logic (if no rule matches):**
1. Try to extract from common fields: `text`, `content`, `value`, `parts`
2. Handle both string and list values
3. If text found → return as `text` part
4. Otherwise → return as `other` part with no extracted content

### 1.11 Role Normalization

**Method:** `_normalize_role(role_raw: Any) -> str`

**Process:**
1. Convert to lowercase string
2. Look up in `export_mapping.role_mapping`
3. If found → return mapped value
4. If not found → log warning, return `"unknown"`

**Standard roles:** `user`, `assistant`, `system`, `tool`, `unknown`

### 1.12 Message Processing Flow

**For each conversation:**

1. **Extract conversation-level fields:**
   * `export_conversation_id` (from export mapping path)
   * `title` (from export mapping path)
   * `created_at_utc` (normalize to UTC)
   * `updated_at_utc` (normalize to UTC)

2. **Generate `conversation_id`:**
   * If export provides ID → use directly
   * Otherwise → UUIDv5 from canonical JSON hash

3. **Canonicalize and store:**
   * `raw_conversation_json = JCS.canonicalize(conv_data)`
   * Insert conversation row (message_count starts at 0)

4. **Process messages:**
   * Extract from `messages_path` (handles both dict mapping and arrays)
   * For each message:
     * Extract message-level fields (id, role, parent, timestamp)
     * Normalize role using role_mapping
     * Generate message_id if not in export
     * Initialize tree_path as empty string
     * Canonicalize `raw_message_json`
     * Insert message row
     * Process message parts (see §1.13)
     * Update attachment_count

5. **Update conversation message_count**

**For all conversations after loading:**

6. **Compute tree paths and order indices** (one pass over all conversations)
7. **Extract text and detect structures** (for each message)
8. **Commit transaction**

### 1.13 Message Parts Processing

**Method:** `_process_message_parts(message_id, msg_data)`

**Content extraction:**
1. Resolve `message_content_path` on message object
2. Handle multiple content structures:
   * **String** → wrap as single text part
   * **Dict with "parts" key** → ChatGPT format, extract parts array
   * **Dict without "parts"** → treat as single part
   * **Array** → treat as multiple parts

**For each part:**
1. **Classify part** using `_classify_part` → get type, text, mime, file, metadata
2. **Count attachments** if file_path or mime_type present
3. **Generate part_id** using UUIDv5
4. **Canonicalize** `raw_part_json`
5. **Insert** into message_parts table

### 1.14 Stage 1 Completion

**Final statistics logged:**
* Total conversations processed
* Total messages processed
* Average messages per conversation
* Total character count (messages.text_raw + message_parts.text_content)

**Transaction handling:**
* Single transaction wraps entire Stage 1
* Commit at end (or rollback on error)
* Database connection closed after commit

### 1.15 Stage 1 Outputs

**Database file** contains:
* Complete conversation history with threading
* Deterministic ordering (tree_path + order_index)
* Semantic-lossless raw JSON storage
* Extracted text ready for NER/analysis
* Code fence and blockquote detection
* All content parts classified and preserved

---


## Stage 2A: Entity & Time Detection Layer

### 2A.0 Objectives

Stage 2A builds a **lossless detection layer** over Stage 1 text by producing:

1. **Entity mention candidates** (emails, URLs, NER names, etc.) with **offset-correct spans** when available.
2. **Time mentions** with **conservative resolution** anchored to message time.
3. Deterministic overlap resolution to emit winning mentions.

**Non-negotiables**

* **Lossless + replayable:** every detector output is stored (even if suppressed).
* **Offsets are sacred:** spans refer to `messages.text_raw` exactly; if not provably correct → offsets become `NULL`.
* **Deterministic ordering:** fixed detector order, fixed scoring/tie-breaks, pinned versions for external models.
* **Transactional:** Stage 2A is one transaction; commit or rollback.

---

### 2A.1 Database schema (Stage 2A tables)

#### entity_mention_candidates

Stores **all** detector outputs for auditability.

| column                       | type         | notes                                                                                  |
| ---------------------------- | ------------ | -------------------------------------------------------------------------------------- |
| `candidate_id`               | TEXT PK      | uuid5 per §1.3                                                                         |
| `message_id`                 | TEXT FK      | → messages                                                                             |
| `detector`                   | TEXT         | EMAIL, URL, DOI, UUID, HASH_HEX, IP_ADDRESS, PHONE, FILEPATH, BARE_DOMAIN, NER:\<model\>, LEXICON:\<build_id\> |
| `detector_version`           | TEXT         | pinned                                                                                 |
| `entity_type_hint`           | TEXT         | detector taxonomy                                                                      |
| `char_start`                 | INTEGER NULL | nullable for NER if unavoidable                                                        |
| `char_end`                   | INTEGER NULL | nullable                                                                               |
| `surface_text`               | TEXT NULL    | exact substring if offsets reliable; else detector-provided surface                    |
| `surface_hash`               | TEXT         | sha256(UTF-8(surface_text)) or sha256("**NO_SURFACE**") if NULL                        |
| `confidence`                 | REAL         | [0,1]                                                                                  |
| `is_eligible`                | INTEGER      | 0/1 after exclusion filtering                                                          |
| `suppressed_by_candidate_id` | TEXT NULL    | winner that suppressed it (if overlap)                                                 |
| `suppression_reason`         | TEXT NULL    | OVERLAP_HIGHER_SCORE, INTERSECTS_CODE_FENCE, NO_OFFSETS_UNRELIABLE, CODE_LIKE_TOKEN    |
| `raw_candidate_json`         | TEXT NULL    | canonical JSON                                                                         |

---

#### entity_mentions

Contains only **emitted** (winner) mentions.

| column             | type         | notes                                                                                  |
| ------------------ | ------------ | -------------------------------------------------------------------------------------- |
| `mention_id`       | TEXT PK      | uuid5 per §1.3                                                                         |
| `message_id`       | TEXT FK      | → messages.message_id                                                                  |
| `entity_id`        | TEXT FK NULL | → entities.entity_id (NULL until Stage 2B consolidation)                               |
| `candidate_id`     | TEXT FK      | → entity_mention_candidates                                                            |
| `detector`         | TEXT         | detector name                                                                          |
| `detector_version` | TEXT         | pinned                                                                                 |
| `entity_type_hint` | TEXT         | detector taxonomy                                                                      |
| `char_start`       | INTEGER NULL | 0-based codepoint index                                                                |
| `char_end`         | INTEGER NULL | exclusive                                                                              |
| `surface_text`     | TEXT NULL    | exact substring if offsets non-null                                                    |
| `surface_hash`     | TEXT         | sha256 of UTF-8 `surface_text` or marker                                               |
| `confidence`       | REAL         | [0,1]                                                                                  |
| `raw_mention_json` | TEXT NULL    | canonical JSON (model label, features, etc.)                                           |

---

#### time_mentions

One row per detected time expression.

| column                   | type      | notes                                 |
| ------------------------ | --------- | ------------------------------------- |
| `time_mention_id`        | TEXT PK   | uuid5 per §1.3                        |
| `message_id`             | TEXT FK   | → messages                            |
| `char_start`             | INTEGER   | required                              |
| `char_end`               | INTEGER   | required                              |
| `surface_text`           | TEXT      | exact substring                       |
| `surface_hash`           | TEXT      | sha256 UTF-8                          |
| `pattern_id`             | TEXT      | identifier of matched pattern         |
| `pattern_precedence`     | INTEGER   | lower = stronger                      |
| `anchor_time_utc`        | TEXT NULL | message created_at_utc                |
| `resolved_type`          | TEXT      | instant, interval, unresolved         |
| `valid_from_utc`         | TEXT NULL | if resolved                           |
| `valid_to_utc`           | TEXT NULL | if interval                           |
| `resolution_granularity` | TEXT NULL | year, month, day, minute              |
| `timezone_assumed`       | TEXT      | config timezone                       |
| `confidence`             | REAL      | pinned scale                          |
| `raw_parse_json`         | TEXT      | canonical JSON                        |

---

#### ner_model_runs

If `enable_ner=true` (default), log the run.

| column             | type    |
| ------------------ | ------- |
| `run_id`           | TEXT PK |
| `model_name`       | TEXT    |
| `model_version`    | TEXT    |
| `config_json`      | TEXT    |
| `started_at_utc`   | TEXT    |
| `completed_at_utc` | TEXT    |
| `raw_io_json`      | TEXT    |

---

### 2A.2 Stage 2A processing flow

Stage 2A executes in **five deterministic phases**:

1. **Initialize**
2. **Build exclusion ranges**
3. **Detect candidates (entities + times)**
4. **Eligibility + overlap resolution (emit winners)**
5. **Commit + stats**

Each phase operates in stable message order:
`(conversation_id ASC, messages.order_index ASC, message_id ASC)`.

---

### 2A.3 Phase 1 — Initialize

* Begin transaction.
* Capture pinned build metadata (detector versions, NER model versions).

---

### 2A.4 Phase 2 — Build exclusion ranges (per message)

For each eligible message (`messages.text_raw IS NOT NULL`):

* Load `messages.code_fence_ranges_json` → **excluded**
* If `config.ignore_markdown_blockquotes=true`, also load `messages.blockquote_ranges_json` → **excluded**
* Exclusion test: simple interval intersection on `[start, end)`.

---

### 2A.5 Phase 3 — Detect candidates

#### 2A.5.1 Detector execution order

Run in fixed order:

1. EMAIL
2. URL (http/https only)
3. DOI
4. UUID
5. HASH_HEX
6. IP_ADDRESS
7. PHONE
8. FILEPATH
9. BARE_DOMAIN
10. arXiv (optional)
11. CVE (optional, high precision)
12. ORCID (optional, high precision)
13. HASH_HEX (but consider GIT_SHA as a separate detector before this, if you want semantics)
14. HANDLE (optional)
15. HASHTAG (optional)
16. NER (if enabled)

*Note: LEXICON detectors added by Stage 2B run after BARE_DOMAIN, before NER.*

#### 2A.5.2 Candidate record contract

When inserting `entity_mention_candidates`:

* For regex/validator detectors: `char_start`/`char_end` **must be non-NULL**.
* `surface_text`:
  * If spans present: **must equal** `text_raw[char_start:char_end]`.
  * If spans absent (NER-only): store detector surface; treat as offset-unreliable.
* Always compute `surface_hash` deterministically.
* Put detector-specific details into `raw_candidate_json`.

**Offset verification (mandatory):**
If spans non-NULL but `substring != surface_text`:
* Set `char_start=NULL`, `char_end=NULL`
* Set `is_eligible=0`
* Record mismatch in `raw_candidate_json`
* Log WARN `OFFSET_UNRELIABLE`

#### 2A.5.3 Noise suppression

Apply pinned allowlist/denylist **after** candidate creation:
* If suppressed: set `is_eligible=0`, `suppression_reason=CODE_LIKE_TOKEN`
* Details in `raw_candidate_json`

#### 2A.5.4 NER runs (if enabled)

If `config.enable_ner=true`:

* Create `ner_model_runs` row with pinned config and raw I/O.
* Run on exact `messages.text_raw`.
* For long texts: chunk by `ner_max_chars` with `ner_stride`, adjust offsets.
* Deduplicate within message on `(char_start, char_end, entity_type_hint, surface_hash)`, keep highest confidence.
* Restrict labels to `config.ner_label_allowlist`.

---

### 2A.6 Phase 4 — Eligibility + overlap resolution

#### 2A.6.1 Eligibility filtering

Set `is_eligible=1` only if:
* Spans non-NULL and verified, AND
* Span does not intersect excluded ranges

Spanless NER candidates: suppressed unless `config.emit_spanless_ner=true`.

#### 2A.6.2 Winner selection (greedy non-overlapping)

From eligible candidates per message, sort by:

1. `confidence DESC`
2. `span_length DESC`
3. `detector_order ASC`
4. `char_start ASC`
5. `char_end DESC`
6. `surface_hash ASC`

Iterate in order:
* If overlaps already-emitted winner → suppress with `OVERLAP_HIGHER_SCORE`
* Else emit to `entity_mentions` (with `entity_id=NULL` pending Stage 2B)

---

### 2A.7 Phase 5 — Time mentions

#### 2A.7.1 Detect + exclude + resolve overlaps

* Generate all time candidates from pinned patterns.
* Exclude spans intersecting excluded ranges.
* Resolve overlaps by sorting:
  1. `span_length DESC`
  2. `pattern_precedence ASC`
  3. `confidence DESC`
  4. `char_start ASC`
  5. `char_end DESC`
  6. `surface_hash ASC`

Greedily select non-overlapping matches.

#### 2A.7.2 Conservative resolution

* `anchor_time_utc = messages.created_at_utc`
* If `timestamp_quality != 'original'`: relative expressions → `resolved_type='unresolved'`
* Resolution outcomes: instant, interval (date/month/year), or unresolved
* All decisions stored in `raw_parse_json`

---

### 2A.8 Stage 2A completion

On commit:
* Log counts: candidates, suppressed-by-reason, emitted mentions, time mentions resolved/unresolved
* `entity_mentions.entity_id` remains NULL (populated in Stage 2B)

---


## Stage 2B: Personal Lexicon & Entity Consolidation Layer

### 2B.0 Objectives

Stage 2B completes the lexicon layer by:

1. **Inducing a personal term lexicon** from corpus patterns (project codenames, acronyms, nicknames).
2. **Running LexiconMatch detector** to emit learned terms with exact offsets.
3. **Consolidating all emitted mentions** into canonical **entities**.

**Non-negotiables**

* **Lossless + replayable:** lexicon induction fully logged; all candidates stored.
* **Deterministic:** stable candidate generators, scoring, and selection.
* **Role-aware trust:** user-weighted counts prioritized over assistant.
* **Transactional:** Stage 2B is one transaction.

---

### 2B.1 Database schema (Stage 2B tables)

#### entities

One row per canonical entity.

| column               | type      | notes                                                                                   |
| -------------------- | --------- | --------------------------------------------------------------------------------------- |
| `entity_id`          | TEXT PK   | uuid5 per §1.3                                                                          |
| `entity_type`        | TEXT      | EMAIL, URL, DOI, UUID, HASH_HEX, IP_ADDRESS, PHONE, FILEPATH, BARE_DOMAIN, PERSON, ORG, LOCATION, CUSTOM_TERM, OTHER |
| `entity_key`         | TEXT      | normalized blocking key (see §2B.6)                                                     |
| `canonical_name`     | TEXT      | deterministic selection                                                                 |
| `aliases_json`       | TEXT      | canonical JSON array of strings                                                         |
| `status`             | TEXT      | `active`; reserved for future merges                                                    |
| `first_seen_at_utc`  | TEXT NULL | from message timestamps                                                                 |
| `last_seen_at_utc`   | TEXT NULL | from message timestamps                                                                 |
| `mention_count`      | INTEGER   | deterministic recompute                                                                 |
| `conversation_count` | INTEGER   | distinct conversations                                                                  |
| `salience_score`     | REAL NULL | computed importance (see §2B.7)                                                         |
| `raw_stats_json`     | TEXT NULL | canonical JSON                                                                          |

**Index requirement:**
```sql
CREATE UNIQUE INDEX entities_active_uniq 
ON entities(entity_type, entity_key) 
WHERE status='active';
```

**Reserved SELF entity (seed at Stage 2B start):**
* `entity_type = 'PERSON'`
* `entity_key = '__SELF__'`
* `canonical_name = 'SELF'`
* `entity_id = uuid5(KG_NS_UUID, JCS(["entity", "PERSON", "__SELF__"]))`

---

#### lexicon_builds

One row per lexicon induction run.

| column             | type      | notes                              |
| ------------------ | --------- | ---------------------------------- |
| `build_id`         | TEXT PK   | uuid5 per §1.3                     |
| `build_version`    | INTEGER   | monotonic version number           |
| `config_json`      | TEXT      | all induction parameters           |
| `started_at_utc`   | TEXT      |                                    |
| `completed_at_utc` | TEXT      |                                    |
| `candidates_total` | INTEGER   | total candidates generated         |
| `terms_selected`   | INTEGER   | terms passing thresholds           |
| `raw_stats_json`   | TEXT NULL | detailed statistics                |

---

#### lexicon_term_candidates

Stores **all** lexicon induction candidates for auditability.

| column                | type      | notes                                           |
| --------------------- | --------- | ----------------------------------------------- |
| `candidate_id`        | TEXT PK   | uuid5 per §1.3                                  |
| `build_id`            | TEXT FK   | → lexicon_builds                                |
| `generator`           | TEXT      | TITLE_CASE, ALLCAPS, CAMEL_CASE, HASHTAG, QUOTED, NOUN_CHUNK |
| `term_key`            | TEXT      | normalized key (lowercase, stripped)            |
| `canonical_surface`   | TEXT      | most frequent surface form                      |
| `aliases_json`        | TEXT      | all observed surfaces (canonical JSON array)    |
| `total_count`         | INTEGER   | raw mention count                               |
| `user_weighted_count` | REAL      | role-weighted count                             |
| `conversation_count`  | INTEGER   | distinct conversations                          |
| `code_likeness_ratio` | REAL      | fraction in code fences (0-1)                   |
| `context_diversity`   | REAL      | distinct surrounding tokens ratio               |
| `score`               | REAL      | composite selection score                       |
| `is_selected`         | INTEGER   | 0/1 after threshold filtering                   |
| `rejection_reason`    | TEXT NULL | if not selected: BELOW_MIN_COUNT, BELOW_MIN_CONV, DENYLIST, CODE_HEAVY, LOW_DIVERSITY, CAP_EXCEEDED |
| `evidence_json`       | TEXT      | canonical JSON (occurrences, contexts)          |

---

#### lexicon_terms

Selected terms (subset of candidates with `is_selected=1`).

| column            | type    | notes                              |
| ----------------- | ------- | ---------------------------------- |
| `term_id`         | TEXT PK | uuid5 per §1.3                     |
| `build_id`        | TEXT FK | → lexicon_builds                   |
| `candidate_id`    | TEXT FK | → lexicon_term_candidates          |
| `term_key`        | TEXT    | normalized key                     |
| `canonical_surface` | TEXT  | most frequent surface              |
| `aliases_json`    | TEXT    | all surfaces (canonical JSON)      |
| `score`           | REAL    | selection score                    |
| `entity_type_hint`| TEXT    | CUSTOM_TERM (default) or inferred  |

**Index:**
```sql
CREATE INDEX idx_lexicon_terms_build ON lexicon_terms(build_id);
CREATE INDEX idx_lexicon_terms_key ON lexicon_terms(term_key);
```

---

### 2B.2 Stage 2B processing flow

Stage 2B executes in **seven deterministic phases**:

1. **Initialize & seed**
2. **Lexicon induction (candidate generation)**
3. **Lexicon selection (threshold filtering)**
4. **LexiconMatch detection**
5. **Entity upsert + canonicalization**
6. **Salience scoring**
7. **Commit + stats**

---

### 2B.3 Phase 1 — Initialize & seed

* Begin transaction.
* Insert (or ensure) reserved **SELF** entity row.
* Generate `build_id` for lexicon induction.

---

### 2B.4 Phase 2 — Lexicon induction (candidate generation)

#### 2B.4.1 Candidate generators (deterministic, offset-bearing)

Run in fixed order on each message's `text_raw`, excluding code fences and blockquotes:

1. **TITLE_CASE**: Sequences of 2+ TitleCase words (e.g., `My Project Phoenix`)
   * Pattern: `\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b`
   
2. **ALLCAPS**: Tokens of 2-6 uppercase letters (e.g., `OKR`, `SRE`)
   * Pattern: `\b([A-Z]{2,6})\b`
   * Apply pinned stoplist (common abbreviations: USA, UK, etc.)
   
3. **CAMEL_CASE**: camelCase or PascalCase tokens (e.g., `myCoolTool`, `ProjectAtlas`)
   * Pattern: `\b([a-z]+[A-Z][a-zA-Z]*|[A-Z][a-z]+[A-Z][a-zA-Z]*)\b`
   
4. **HASHTAG**: `#tag` patterns (e.g., `#zettelkasten`)
   * Pattern: `#([a-zA-Z][a-zA-Z0-9_]{2,})\b`
   
5. **HANDLE**: `@handle` patterns (e.g., `@alice`)
   * Pattern: `@([a-zA-Z][a-zA-Z0-9_]{2,})\b`
   
6. **QUOTED**: Double-quoted phrases 2-5 words (e.g., `"Blue Garden"`)
   * Pattern: `"([A-Z][^"]{3,50})"`
   * Conservative: require initial capital

7. **NOUN_CHUNK** (optional, if `enable_noun_chunk_induction=true`):
   * Run spaCy noun chunker with pinned model version
   * Store raw I/O in `lexicon_builds.raw_stats_json`

#### 2B.4.2 Candidate aggregation

For each unique `term_key` (normalized: lowercase, whitespace-collapsed):

1. **Collect all occurrences** across messages
2. **Compute statistics:**
   * `total_count`: raw occurrence count
   * `user_weighted_count`: sum of weights (user=1.0, assistant=0.5)
   * `conversation_count`: distinct conversation_ids
   * `code_likeness_ratio`: fraction of occurrences inside code fences
   * `context_diversity`: distinct bigrams surrounding term / total occurrences
3. **Select canonical_surface**: most frequent surface, tie-break by user-weighted count, then lexicographic
4. **Build aliases_json**: unique surfaces, sorted lexicographically

#### 2B.4.3 Insert candidates

Insert all candidates into `lexicon_term_candidates` with computed statistics.

---

### 2B.5 Phase 3 — Lexicon selection (threshold filtering)

#### 2B.5.1 Selection criteria (configurable)

Apply in order:

1. **Denylist check**: reject if `term_key` in pinned denylist (common words, month names, days)
   * Rejection: `DENYLIST`
   
2. **Minimum count**: `user_weighted_count >= config.lexicon_min_user_mentions` (default: 3)
   * Rejection: `BELOW_MIN_COUNT`
   
3. **Minimum conversations**: `conversation_count >= config.lexicon_min_conversations` (default: 2)
   * Rejection: `BELOW_MIN_CONV`
   
4. **Code-likeness threshold**: `code_likeness_ratio <= config.lexicon_max_code_ratio` (default: 0.5)
   * Rejection: `CODE_HEAVY`
   
5. **Context diversity threshold**: `context_diversity >= config.lexicon_min_diversity` (default: 0.1)
   * Rejection: `LOW_DIVERSITY`

#### 2B.5.2 Scoring function

For passing candidates, compute:
```
score = (user_weighted_count * config.lexicon_weight_mentions) 
      + (conversation_count * config.lexicon_weight_conversations)
      + (context_diversity * config.lexicon_weight_diversity)
      - (code_likeness_ratio * config.lexicon_penalty_code)
```

Default weights: mentions=1.0, conversations=2.0, diversity=0.5, code_penalty=1.0

#### 2B.5.3 Top-K cap

Sort by `score DESC`, `term_key ASC`. Select top `config.lexicon_max_terms` (default: 1000).
Remaining candidates: `rejection_reason = CAP_EXCEEDED`.

#### 2B.5.4 Persist selected terms

For selected candidates (`is_selected=1`):
* Insert into `lexicon_terms` with `entity_type_hint = 'CUSTOM_TERM'`
* Apply optional type inference rules:
  * Ends with "Inc.", "Corp.", "LLC" → `ORG`
  * Starts with "@" → `PERSON` (tentative)

---

### 2B.6 Phase 4 — LexiconMatch detection

#### 2B.6.1 Build matching structure

Construct Aho-Corasick automaton (or sorted longest-first matcher) from:
* All `canonical_surface` values from `lexicon_terms`
* All surfaces in `aliases_json`

#### 2B.6.2 Scan messages

For each message with `text_raw`:

1. Run matcher to find all occurrences
2. For each match:
   * Verify offset: `text_raw[start:end] == surface`
   * If verified: emit candidate with `detector = 'LEXICON:<build_id>'`
   * Insert into `entity_mention_candidates`
   
3. Apply same exclusion ranges (code fences, blockquotes)
4. Run overlap resolution against Stage 2A mentions:
   * Structured detectors (EMAIL, URL, etc.) win over LEXICON
   * LEXICON wins over NER

#### 2B.6.3 Emit winners

For each winning lexicon match:
* Insert into `entity_mentions`
* Link to term via `raw_mention_json`

---

### 2B.7 Phase 5 — Entity upsert + canonicalization

#### 2B.7.1 entity_key normalization (type-specific)

| entity_type   | normalization                                    |
| ------------- | ------------------------------------------------ |
| EMAIL         | lowercase                                        |
| URL           | scheme + lowercase host + path (no query/fragment unless config) |
| DOI           | lowercase, strip "doi:" prefix                   |
| UUID          | lowercase, hyphenated                            |
| HASH_HEX      | lowercase                                        |
| IP_ADDRESS    | canonical form (no leading zeros)                |
| PHONE         | E.164 format if parseable                        |
| FILEPATH      | as-is                                            |
| BARE_DOMAIN   | lowercase                                        |
| PERSON/ORG/LOCATION | lowercase, whitespace-collapsed            |
| CUSTOM_TERM   | lowercase, whitespace-collapsed                  |

#### 2B.7.2 Upsert entity rows

For each unique `(entity_type, entity_key)` from emitted mentions:

* **Insert** if new; **update** if exists
* Update `entity_mentions.entity_id` to link mentions to entities

#### 2B.7.3 Canonical name selection (role-aware)

1. Most frequent `surface_text` (non-NULL)
2. Tie-break: prefer surfaces from user-role messages
3. Final tie-break: first occurrence by `(created_at_utc, conversation_id, order_index, message_id, mention_id, surface_text)`

Update `aliases_json`: unique non-null surfaces, lexicographically sorted.

---

### 2B.8 Phase 6 — Salience scoring

#### 2B.8.1 Compute salience per entity

For each active entity:
```
salience_score = (mention_count * config.salience_weight_mentions)
               + (conversation_count * config.salience_weight_conversations)
               + (user_mention_ratio * config.salience_weight_user_ratio)
               + (recency_factor * config.salience_weight_recency)
```

Where:
* `user_mention_ratio`: mentions in user messages / total mentions
* `recency_factor`: exponential decay from `last_seen_at_utc` to build time

Default weights: mentions=0.3, conversations=0.4, user_ratio=0.2, recency=0.1

#### 2B.8.2 Update entity rows

Set `entities.salience_score` for all active entities.

---

### 2B.9 Phase 7 — Commit + stats

Finalize `lexicon_builds` row:
* `completed_at_utc`
* `candidates_total`, `terms_selected`
* `raw_stats_json`: generator breakdown, rejection reasons, score distribution

Commit transaction.

Log counts:
* Lexicon candidates generated/selected
* Entity mentions linked
* Entities created/updated
* Salience score distribution

---

### 2B.10 Deterministic ID formulas (Stage 2B)

| Entity               | Formula                                                                |
| -------------------- | ---------------------------------------------------------------------- |
| `entity_id`          | `uuid5(KG_NS_UUID, JCS(["entity", entity_type, entity_key]))`          |
| `mention_id`         | `uuid5(KG_NS_UUID, JCS(["mention", message_id, candidate_id]))`        |
| `time_mention_id`    | `uuid5(KG_NS_UUID, JCS(["time", message_id, char_start, surface_hash]))` |
| `build_id`           | `uuid5(KG_NS_UUID, JCS(["lexicon_build", started_at_utc]))`            |
| `term_candidate_id`  | `uuid5(KG_NS_UUID, JCS(["lex_cand", build_id, generator, term_key]))` |
| `term_id`            | `uuid5(KG_NS_UUID, JCS(["lex_term", build_id, term_key]))`             |

---

### 2B.11 Pipeline Configuration (Stage 2B additions)

**Lexicon Induction:**
* `enable_lexicon_induction` (BOOLEAN): Enable personal lexicon learning (default: `true`)
* `enable_noun_chunk_induction` (BOOLEAN): Include spaCy noun chunks (default: `false`)
* `lexicon_min_user_mentions` (INTEGER): Minimum user-weighted count (default: `3`)
* `lexicon_min_conversations` (INTEGER): Minimum conversation spread (default: `2`)
* `lexicon_max_code_ratio` (REAL): Maximum code-likeness ratio (default: `0.5`)
* `lexicon_min_diversity` (REAL): Minimum context diversity (default: `0.1`)
* `lexicon_max_terms` (INTEGER): Maximum terms to select (default: `1000`)
* `lexicon_denylist_path` (TEXT): Path to pinned denylist file (optional)

**Lexicon Scoring Weights:**
* `lexicon_weight_mentions` (REAL): default `1.0`
* `lexicon_weight_conversations` (REAL): default `2.0`
* `lexicon_weight_diversity` (REAL): default `0.5`
* `lexicon_penalty_code` (REAL): default `1.0`

**Salience Scoring:**
* `salience_weight_mentions` (REAL): default `0.3`
* `salience_weight_conversations` (REAL): default `0.4`
* `salience_weight_user_ratio` (REAL): default `0.2`
* `salience_weight_recency` (REAL): default `0.1`
* `salience_recency_halflife_days` (INTEGER): Decay half-life (default: `90`)

---

### 2B.12 Stage 2B outputs and completion criteria

On successful commit, Stage 2B guarantees:

* **Lexicon build** with full audit trail of candidates and selections
* **LexiconMatch mentions** emitted with verified offsets
* **All entity mentions** linked to canonical entities
* **SELF** entity exists and is stable
* **Salience scores** computed for all active entities
* Entity `canonical_name` reflects role-weighted evidence

**Rollback conditions:**
* Any database error during processing
* Constraint violation on entity or mention insert


## Stage 3: Entity Canonicalization Layer

### 3.0 Objectives

Stage 3 refines entity canonical names using **detector-weighted, role-aware mention evidence** from Stage 2, producing a fully auditable refinement trail without altering entity identity.

**Outputs:**

* Refined `entities.canonical_name` values based on detector-weighted, role-weighted surface frequency
* Full audit trail (`entity_canonicalization_runs`, `entity_canonical_name_history`)

**Invariants:**

* `entity_id` and `entity_key` remain **immutable** (identity preserved)
* `aliases_json` unchanged (lossless surface accumulation from Stage 2)
* Deterministic selection with stable tie-breaks
* Stage is **single transaction**: commit all or rollback all

---

### 3.1 Database schema (Stage 3 tables)

#### entity_canonicalization_runs

One row per canonicalization execution.

| column               | type      | notes                                                   |
| -------------------- | --------- | ------------------------------------------------------- |
| `run_id`             | TEXT PK   | uuid5 per §1.3                                          |
| `method`             | TEXT      | detector_role_weighted, assertion_informed, llm_assisted, hybrid |
| `model_name`         | TEXT NULL | if LLM method used                                      |
| `model_version`      | TEXT NULL |                                                         |
| `config_json`        | TEXT      | all config parameters (canonical JSON)                  |
| `started_at_utc`     | TEXT      |                                                         |
| `completed_at_utc`   | TEXT      |                                                         |
| `entities_processed` | INTEGER   |                                                         |
| `names_changed`      | INTEGER   | count of entities with updated canonical name           |
| `raw_stats_json`     | TEXT NULL | detailed statistics (canonical JSON)                    |

**Indices:**

* Primary key on `run_id`

---

#### entity_canonical_name_history

One row per canonical name change event.

| column               | type      | notes                              |
| -------------------- | --------- | ---------------------------------- |
| `history_id`         | TEXT PK   | uuid5 per §1.3                     |
| `entity_id`          | TEXT FK   | → entities                         |
| `run_id`             | TEXT FK   | → entity_canonicalization_runs     |
| `previous_name`      | TEXT NULL | prior canonical name               |
| `canonical_name`     | TEXT      | selected name                      |
| `selection_method`   | TEXT      | method used                        |
| `confidence`         | REAL      | [0,1]                              |
| `selected_at_utc`    | TEXT      |                                    |
| `raw_selection_json` | TEXT      | full decision log (canonical JSON) |

**Indices:**

* Primary key on `history_id`
* `idx_canonical_history_entity` on `entity_id`
* `idx_canonical_history_run` on `run_id`

---

### 3.2 Stage 3 processing flow

Stage 3 executes in **three deterministic phases**:

1. **Phase 1 — Initialize run**
2. **Phase 2 — Detector-role-weighted canonicalization**
3. **Phase 3 — Persist + commit**

---

### 3.3 Phase 1 — Initialize run

* Begin transaction
* Generate `run_id` deterministically
* Record `started_at_utc`, method, config snapshot
* Initialize counters: `entities_processed=0`, `names_changed=0`

---

### 3.4 Phase 2 — Detector-role-weighted canonicalization

#### 3.4.1 Entity iteration order

Process active entities in deterministic order:
`(entity_type ASC, entity_key ASC, entity_id ASC)`

#### 3.4.2 Detector reliability tiers (configurable)

Detectors are assigned to reliability tiers that determine their weight multiplier:

**Tier 1 — Structured/Validated (highest reliability):**
* EMAIL, URL, DOI, UUID, IP_ADDRESS, PHONE
* Default weight multiplier: `config.detector_weight_tier1` (default 1.0)

**Tier 2 — Pattern-based:**
* HASH_HEX, FILEPATH, BARE_DOMAIN, arXiv, CVE, ORCID, HANDLE, HASHTAG
* Default weight multiplier: `config.detector_weight_tier2` (default 0.9)

**Tier 3 — Learned/Lexicon:**
* LEXICON:* (any lexicon build)
* Default weight multiplier: `config.detector_weight_tier3` (default 0.8)

**Tier 4 — Statistical/NER (lowest reliability):**
* NER:* (any NER model)
* Default weight multiplier: `config.detector_weight_tier4` (default 0.6)

**Tier lookup:**
```
detector_tier(detector_name):
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
```

#### 3.4.3 Mention aggregation

For each active entity:

1. **Collect emitted mentions** from `entity_mentions` where `entity_id` matches
2. **Group by `surface_text`** (exclude NULL surfaces)
3. **Compute composite weighted score per surface:**

   For each mention:
   * Retrieve `messages.role` via `entity_mentions.message_id`
   * Retrieve `entity_mentions.detector` for detector tier
   * Compute role weight:
     * `role = 'user'` → `config.trust_weight_user` (default 1.0)
     * `role = 'assistant'` → `config.trust_weight_assistant_uncorroborated` (default 0.5)
     * other roles → 0.5
   * Compute detector weight:
     * `detector_weight = config.detector_weight_tier{N}` based on tier
   * Compute mention confidence boost (from `entity_mentions.confidence`):
     * `confidence_factor = 0.5 + (0.5 * mention.confidence)`
   * Compose weights:
     * `mention_weight = role_weight * detector_weight * confidence_factor`
   * Accumulate: `weighted_score[surface] += mention_weight`
   * Track: `unweighted_count[surface] += 1`
   * Track: `best_detector_tier[surface] = min(best_detector_tier[surface], tier)`

4. **Track first occurrence per surface:**
   * For each surface, record the earliest mention by:
     `(message.created_at_utc NULLS LAST, conversation_id, order_index, message_id, mention_id)`

#### 3.4.4 Canonical name selection (deterministic)

**Selection algorithm:**

1. **Primary sort:** `weighted_score DESC`
2. **Tie-break 1:** `best_detector_tier ASC` (prefer higher-reliability detector)
3. **Tie-break 2:** `unweighted_count DESC`
4. **Tie-break 3:** first occurrence tuple ASC (earliest wins)
5. **Tie-break 4:** `surface_text ASC` (lexicographic)

Select the top-ranked surface as `new_canonical_name`.

**Confidence calculation:**
```
total_weighted = sum(weighted_score.values())
base_confidence = weighted_score[new_canonical_name] / total_weighted if total_weighted > 0 else 1.0

# Apply detector reliability bonus
tier_bonus = (4 - best_detector_tier[new_canonical_name]) * config.detector_tier_confidence_bonus
# default detector_tier_confidence_bonus = 0.05

confidence = clamp(base_confidence + tier_bonus, 0.0, 1.0)
```

#### 3.4.5 Salience-adjusted processing (optional)

If `config.enable_salience_prioritization = true`:

* Sort entities for processing by `(salience_score DESC NULLS LAST, entity_type ASC, entity_key ASC)`
* High-salience entities processed first (allows potential early termination in resource-constrained environments)
* Does not affect selection algorithm, only processing order

#### 3.4.6 Lexicon term enhancement (for CUSTOM_TERM entities)

For entities with `entity_type = 'CUSTOM_TERM'`:

1. **Retrieve linked lexicon term:**
   * Query `lexicon_terms` where `term_key = entities.entity_key`
   * If found, retrieve `lexicon_terms.score` and `canonical_surface`

2. **Apply lexicon bonus:**
   * If lexicon term exists and `lexicon_terms.canonical_surface` matches a candidate surface:
     * Add `config.lexicon_canonical_bonus` (default 0.2) to that surface's weighted_score
   * Rationale: Lexicon induction already performed corpus-wide frequency analysis

3. **Record in audit log:**
   * Include `lexicon_term_id`, `lexicon_score`, `bonus_applied` in `raw_selection_json`

#### 3.4.7 Change detection and recording

For each entity:

* Increment `entities_processed`
* If `new_canonical_name != entities.canonical_name`:
  * Increment `names_changed`
  * Insert `entity_canonical_name_history` row with:
    * `previous_name` = old value
    * `canonical_name` = new value
    * `selection_method` = `'detector_role_weighted'`
    * `confidence` = computed confidence
    * `raw_selection_json` = canonical JSON containing:
      * all surfaces with scores and detector breakdowns
      * weight config used
      * detector tier per surface
      * first occurrence details
      * lexicon bonus (if applicable)
      * winning surface rationale
  * Update `entities.canonical_name` in place

---

### 3.5 Phase 3 — Persist + commit

#### 3.5.1 Finalize run record

* Set `completed_at_utc`
* Compute `raw_stats_json`:
  * `entities_by_type`: count per entity_type processed
  * `changes_by_type`: count per entity_type changed
  * `avg_confidence`: mean confidence across changed entities
  * `surface_distribution`: histogram of unique surfaces per entity
  * `detector_tier_distribution`: count of winning surfaces by detector tier
  * `lexicon_bonus_applied_count`: entities where lexicon bonus affected outcome
* Insert `entity_canonicalization_runs` row

#### 3.5.2 Commit

* Commit transaction
* Log summary: entities processed, names changed, duration, detector tier breakdown

---

### 3.6 Deterministic ID formulas (Stage 3)

| Entity       | Formula                                                               |
| ------------ | --------------------------------------------------------------------- |
| `run_id`     | `uuid5(KG_NS_UUID, JCS(["canon_run", started_at_utc, method]))`       |
| `history_id` | `uuid5(KG_NS_UUID, JCS(["canon_hist", entity_id, run_id]))`           |

---

### 3.7 Pipeline Configuration (Stage 3)

**Detector Reliability Weights:**
* `detector_weight_tier1` (REAL): Structured detector weight (default: `1.0`)
* `detector_weight_tier2` (REAL): Pattern-based detector weight (default: `0.9`)
* `detector_weight_tier3` (REAL): Lexicon detector weight (default: `0.8`)
* `detector_weight_tier4` (REAL): NER detector weight (default: `0.6`)
* `detector_tier_confidence_bonus` (REAL): Per-tier confidence bonus (default: `0.05`)

**Lexicon Integration:**
* `lexicon_canonical_bonus` (REAL): Bonus for lexicon-preferred surface (default: `0.2`)

**Processing Options:**
* `enable_salience_prioritization` (BOOLEAN): Process high-salience entities first (default: `false`)

**Trust Weights (inherited from Stage 2/4):**
* `trust_weight_user` (REAL): Weight for user-role mentions (default: `1.0`)
* `trust_weight_assistant_uncorroborated` (REAL): Weight for assistant mentions (default: `0.5`)

---

### 3.8 Stage 3 outputs and completion criteria

On successful commit, Stage 3 guarantees:

* Every active entity evaluated with detector-weighted, role-weighted evidence
* `canonical_name` reflects highest-confidence surface per composite weighting
* Detector reliability properly prioritizes structured detectors over NER
* LexiconMatch surfaces properly weighted between structured and NER
* Full audit trail in history table for any changed names
* Run metadata captured for reproducibility

**Rollback conditions:**

* Any database error during processing
* Constraint violation on history insert

---


## Stage 4: Assertion Extraction & Grounding Layer

### 4.0 Objectives

Stage 4 converts Stage 1–3 evidence (messages + refined entities + time mentions) into **auditable, replayable assertions** with deterministic IDs, conservative span handling, detector-aware entity reliability, and role-aware trust.

**Outputs:**

* `predicates` — normalized relation vocabulary
* `assertions` — grounded semantic claims with entity/literal objects
* `retractions` — explicit negations or corrections linked to prior assertions
* Optional LLM extraction artifacts (`llm_extraction_runs`, `llm_extraction_calls`)

**Non-goals (Stage 5 responsibilities):**

* Valid-time assignment and bitemporal closing/superseding
* Cross-message conflict resolution beyond basic dedup + corroboration flags

**Determinism requirements:**

* Stable processing order: `(conversation_id ASC, order_index ASC, message_id ASC)`
* Offsets anchored to `messages.text_raw`
* Canonical JSON for all model I/O and decision logs
* Stage is **single transaction**: commit all or rollback all

---

### 4.1 Database schema (Stage 4 tables)

#### predicates

One row per normalized predicate label.

| column                  | type      | notes                                                |
| ----------------------- | --------- | ---------------------------------------------------- |
| `predicate_id`          | TEXT PK   | uuid5 per §1.3                                       |
| `canonical_label`       | TEXT      | normalized display form                              |
| `canonical_label_norm`  | TEXT      | lowercase for dedup                                  |
| `inverse_label`         | TEXT NULL | if known (e.g., "works_for" ↔ "employs")             |
| `category`              | TEXT NULL | domain-specific taxonomy                             |
| `arity`                 | INTEGER   | 1 (unary), 2 (binary), or 3 (ternary with qualifier) |
| `value_type_constraint` | TEXT NULL | expected object type if applicable                   |
| `first_seen_at_utc`     | TEXT NULL |                                                      |
| `assertion_count`       | INTEGER   | default 0                                            |
| `raw_predicate_json`    | TEXT NULL | canonical JSON                                       |

**Indices:**

* Primary key on `predicate_id`
* `idx_predicates_label_norm` UNIQUE on `canonical_label_norm`

---

#### assertions

One row per grounded assertion.

| column                       | type         | notes                                            |
| ---------------------------- | ------------ | ------------------------------------------------ |
| `assertion_id`               | TEXT PK      | uuid5 per §1.3                                   |
| `message_id`                 | TEXT FK      | → messages                                       |
| `assertion_key`              | TEXT         | dedup key (see §4.7)                             |
| `fact_key`                   | TEXT         | semantic dedup key (see §4.7)                    |
| `subject_entity_id`          | TEXT FK      | → entities                                       |
| `subject_detection_tier`     | INTEGER NULL | detector tier of subject entity mention (1-4)   |
| `predicate_id`               | TEXT FK      | → predicates                                     |
| `object_entity_id`           | TEXT NULL FK | → entities (if object is entity)                 |
| `object_detection_tier`      | INTEGER NULL | detector tier of object entity mention (1-4)    |
| `object_value_type`          | TEXT NULL    | string, number, boolean, date, json (if literal) |
| `object_value`               | TEXT NULL    | canonical JSON value                             |
| `object_signature`           | TEXT         | for conflict detection (see §4.6)                |
| `temporal_qualifier_type`    | TEXT NULL    | "at", "since", "until", "during"                 |
| `temporal_qualifier_id`      | TEXT NULL FK | → time_mentions.time_mention_id                  |
| `modality`                   | TEXT         | state, fact, preference, intention, question     |
| `polarity`                   | TEXT         | positive, negative                               |
| `asserted_role`              | TEXT         | user, assistant                                  |
| `asserted_at_utc`            | TEXT NULL    | message timestamp                                |
| `confidence_extraction`      | REAL         | [0,1] from extractor                             |
| `confidence_grounding`       | REAL         | [0,1] from entity resolution quality             |
| `confidence_final`           | REAL         | composed (see §4.6)                              |
| `has_user_corroboration`     | INTEGER      | 0/1 (see §4.5)                                   |
| `superseded_by_assertion_id` | TEXT NULL FK | → assertions (if retracted or corrected)         |
| `supersession_type`          | TEXT NULL    | "retraction", "correction", "temporal_end"       |
| `char_start`                 | INTEGER NULL |                                                  |
| `char_end`                   | INTEGER NULL |                                                  |
| `surface_text`               | TEXT NULL    | if span available                                |
| `extraction_method`          | TEXT         | rule_based, llm, hybrid                          |
| `extraction_model`           | TEXT NULL    | model name if LLM                                |
| `raw_assertion_json`         | TEXT         | canonical JSON                                   |

**Object model constraints:**
An assertion must have exactly one of:

* `object_entity_id` IS NOT NULL, OR
* `(object_value_type, object_value)` both NOT NULL, OR
* neither (unary assertion):
  * `object_signature = "N:__NONE__"`
  * all object fields NULL

**Indices:**

* Primary key on `assertion_id`
* `idx_assertions_message` on `message_id`
* `idx_assertions_key` UNIQUE on `assertion_key`
* `idx_assertions_fact_key` on `fact_key`
* `idx_assertions_subject` on `subject_entity_id`
* `idx_assertions_predicate` on `predicate_id`
* `idx_assertions_object_entity` on `object_entity_id` WHERE `object_entity_id IS NOT NULL`
* `idx_assertions_temporal_qualifier` on `temporal_qualifier_id` WHERE `temporal_qualifier_id IS NOT NULL`

---

#### llm_extraction_runs

One row per LLM extraction stage execution.

| column                 | type      | notes               |
| ---------------------- | --------- | ------------------- |
| `run_id`               | TEXT PK   | uuid5 per §1.3      |
| `model_name`           | TEXT      |                     |
| `model_version`        | TEXT      |                     |
| `config_json`          | TEXT      | canonical JSON      |
| `started_at_utc`       | TEXT      |                     |
| `completed_at_utc`     | TEXT      |                     |
| `messages_processed`   | INTEGER   |                     |
| `assertions_extracted` | INTEGER   |                     |
| `raw_stats_json`       | TEXT NULL | detailed statistics |

**Indices:**

* Primary key on `run_id`

---

#### llm_extraction_calls

One row per LLM API call.

| column               | type         | notes                               |
| -------------------- | ------------ | ----------------------------------- |
| `call_id`            | TEXT PK      | uuid5 per §1.3                      |
| `run_id`             | TEXT FK      | → llm_extraction_runs               |
| `message_id`         | TEXT FK      | → messages                          |
| `request_json`       | TEXT         | canonical JSON (full prompt)        |
| `response_json`      | TEXT         | canonical JSON (full response)      |
| `call_timestamp_utc` | TEXT         |                                     |
| `retry_count`        | INTEGER      | 0-based                             |
| `seed_honored`       | INTEGER NULL | 0/1 if determinable, NULL otherwise |
| `parse_success`      | INTEGER      | 0/1                                 |
| `raw_io_json`        | TEXT         | canonical JSON (includes timing)    |

**Indices:**

* Primary key on `call_id`
* `idx_llm_calls_run` on `run_id`
* `idx_llm_calls_message` on `message_id`

---

#### retractions

One row per detected retraction or correction.

| column                     | type         | notes                                          |
| -------------------------- | ------------ | ---------------------------------------------- |
| `retraction_id`            | TEXT PK      | uuid5 per §1.3                                 |
| `retraction_message_id`    | TEXT FK      | → messages                                     |
| `target_assertion_id`      | TEXT NULL FK | → assertions (specific target)                 |
| `target_fact_key`          | TEXT NULL    | semantic target                                |
| `retraction_type`          | TEXT         | "full", "correction", "temporal_bound"         |
| `replacement_assertion_id` | TEXT NULL FK | → assertions (if correction provides new fact) |
| `confidence`               | REAL         | [0,1]                                          |
| `char_start`               | INTEGER NULL |                                                |
| `char_end`                 | INTEGER NULL |                                                |
| `surface_text`             | TEXT NULL    |                                                |
| `raw_retraction_json`      | TEXT         | canonical JSON                                 |

**Indices:**

* Primary key on `retraction_id`
* `idx_retractions_message` on `retraction_message_id`
* `idx_retractions_target` on `target_assertion_id` WHERE `target_assertion_id IS NOT NULL`
* `idx_retractions_fact_key` on `target_fact_key` WHERE `target_fact_key IS NOT NULL`

---

### 4.2 Stage 4 processing flow

Stage 4 executes in **six deterministic phases**:

1. **Phase 1 — Initialize + prepare exclusions + build entity index**
2. **Phase 2 — Candidate extraction (per-message)**
3. **Phase 3 — Candidate validation**
4. **Phase 4 — Grounding (entity/predicate linking + scoring)**
5. **Phase 5 — Persistence (assertions + retractions + stats)**
6. **Phase 6 — Commit**

---

### 4.3 Phase 1 — Initialize + prepare exclusions + build entity index

#### 4.3.1 Transaction setup

* Begin transaction
* Generate `run_id` for LLM extraction (if enabled)
* Record `started_at_utc`

#### 4.3.2 Build exclusion index

For efficient exclusion checking, build an in-memory index:
```
exclusion_index: Dict[message_id, List[Tuple[char_start, char_end]]]
```

**Population algorithm:**

1. For each message with `text_raw IS NOT NULL`:
   * Load `code_fence_ranges_json` → add all `[char_start, char_end)` intervals
   * If `config.ignore_markdown_blockquotes=true`:
     * Load `blockquote_ranges_json` → add all intervals
   * Sort intervals by `char_start ASC`
   * Merge overlapping intervals for efficiency

#### 4.3.3 Build entity resolution index

Construct in-memory indices for efficient entity lookup during grounding:

**Entity lookup structures:**
```
entity_by_id: Dict[entity_id, EntityRecord]
entity_by_key: Dict[(entity_type, entity_key), entity_id]
entity_by_canonical_name: Dict[lowercase(canonical_name), List[entity_id]]
entity_by_alias: Dict[lowercase(alias), List[entity_id]]
```

**EntityRecord structure:**
```
EntityRecord:
    entity_id: str
    entity_type: str
    entity_key: str
    canonical_name: str
    aliases: List[str]
    salience_score: float | None
    mention_count: int
    conversation_count: int
    first_seen_at_utc: str | None
    last_seen_at_utc: str | None
    best_detector_tier: int  # computed from mentions
```

**Best detector tier computation:**
```
For each active entity:
    Query entity_mentions where entity_id matches
    best_tier = min(detector_tier(m.detector) for m in mentions) or 4
    Store in EntityRecord.best_detector_tier
```

#### 4.3.4 Build salience-ordered entity list

For efficient "did you mean" suggestions and ambiguity resolution:
```
entities_by_salience: List[entity_id]  # sorted by salience_score DESC NULLS LAST
```

#### 4.3.5 Determine eligible messages

A message is eligible if:

* `messages.role IN ('user', 'assistant')`
* `messages.text_raw IS NOT NULL`
* Non-excluded text length ≥ `config.min_extractable_chars` (default 10)

**Non-excluded length calculation:**
```
total_excluded = sum(end - start for (start, end) in exclusion_index[message_id])
non_excluded_length = len(text_raw) - total_excluded
```

Store eligible message IDs in deterministic order:
`(conversation_id ASC, order_index ASC, message_id ASC)`

---

### 4.4 Phase 2 — Candidate extraction (per-message)

#### 4.4.1 Context window construction

For each eligible message (the **target**):

1. **Retrieve prior messages** in same conversation:
   * Up to `config.k_context` messages (default 5)
   * Filter: `order_index < target.order_index`
   * Order: `order_index DESC` (most recent first), take top k, then reverse to chronological

2. **Build context entity set:**
   * Collect distinct `entity_id` from `entity_mentions` across context window + target
   * For each entity: retrieve from `entity_by_id`:
     * `canonical_name`, `entity_type`
     * `salience_score` (for ranking hints to LLM)
     * `best_detector_tier` (for reliability hints)

3. **Build context time set:**
   * Collect `time_mentions` from context window + target
   * Filter: `resolved_type IN ('instant', 'interval')`
   * Include: `surface_text`, `valid_from_utc`, `valid_to_utc`, `resolution_granularity`

4. **Build entity mention map for target message:**
   * For each entity mention in target message:
     * Store: `(entity_id, char_start, char_end, detector, detector_tier, confidence)`
   * Used for span-to-entity resolution during grounding

#### 4.4.2 Target text masking

Apply masking to excluded regions for extraction input:

**Masking strategy** (configurable, default `length_preserving`):

* `length_preserving`: Replace excluded chars with space characters (preserves offsets)
* `marker`: Replace each excluded region with `[CODE]` or `[QUOTE]` marker (offset adjustment required)
* `remove`: Delete excluded regions (offset adjustment required)

Store chosen strategy and any offset mapping in extraction context.

#### 4.4.3 Rule-based extraction

Execute pinned pattern registry against (masked) target text:

**Pattern registry structure:**
```
patterns: List[ExtractionPattern]
  - pattern_id: str
  - regex: compiled Pattern
  - handler: Callable → List[AssertionCandidate]
  - confidence: float (pinned per pattern)
  - priority: int (lower = higher priority for overlap)
```

**Execution:**
1. For each pattern in priority order:
   * Find all non-overlapping matches in target text
   * For each match, invoke handler to produce candidates
   * Each candidate includes:
     * `subject` (entity ref or "SELF")
     * `predicate_label`
     * `object` (entity ref, literal spec, or None for unary)
     * `modality`, `polarity`
     * `char_start`, `char_end` (from match span)
     * `confidence` (from pattern)
     * `extraction_method = 'rule_based'`

2. **Offset adjustment** (if masking strategy != `length_preserving`):
   * Map extracted offsets back to original `text_raw` coordinates
   * If mapping fails, set offsets to NULL and log WARN

#### 4.4.4 LLM extraction (optional)

If `config.enable_llm_assertion_extraction=true`:

**Request construction:**
```
{
  "messages": [context_messages...],
  "target": {
    "role": target.role,
    "text": masked_target_text,
    "entities": [
      {
        "id": entity_id,
        "name": canonical_name,
        "type": entity_type,
        "salience": salience_score,  # helps LLM prioritize
        "reliability": detector_tier_label  #  "high"/"medium"/"low"
      }
      ...
    ],
    "times": [context_time_set...]
  },
  "schema": assertion_output_schema,
  "instructions": extraction_prompt_template
}
```

**Determinism settings:**
* `temperature = config.llm_temperature` (default 0.0)
* `top_p = config.llm_top_p` (default 1.0)
* `seed = config.llm_seed` (pinned)

**Execution:**
1. Call LLM API with retry logic (max `config.llm_max_retries`, default 3)
2. Log each attempt in `llm_extraction_calls`:
   * `request_json`: full prompt (canonical)
   * `response_json`: full response (canonical)
   * `retry_count`: 0-based attempt number
   * `parse_success`: 1 if valid JSON extracted, else 0

3. Parse response into `List[AssertionCandidate]`

**Multi-run aggregation** (if `config.llm_multi_run_count > 1`):
* Execute N independent runs with same input
* Group candidates by equivalence key:
  `(subject_norm, predicate_label_norm, object_signature_intent, modality, polarity)`
* Keep only candidates appearing in majority (> N/2) of runs
* Average confidence across contributing runs

#### 4.4.5 Hybrid merge

If both rule-based and LLM extraction enabled:

1. Union all candidates from both sources
2. Detect semantic duplicates:
   * Same subject (normalized)
   * Same predicate (normalized label)
   * Same object (signature intent)
   * Overlapping spans (if both have spans)

3. For duplicates:
   * Keep rule-based candidate (higher trust)
   * Set `extraction_method = 'hybrid'`
   * Record LLM candidate in `raw_assertion_json` for audit

---

### 4.5 Phase 3 — Candidate validation

For each candidate, apply validation contract:

#### 4.5.1 Required field validation

* `subject` must be non-empty
* `predicate_label` must be non-empty
* `modality` must be in: `state`, `fact`, `preference`, `intention`, `question`
* `polarity` must be in: `positive`, `negative`

#### 4.5.2 Object validation

Exactly one must be true:

* `object.entity_ref` is provided (entity object)
* `object.literal` is provided with valid `type` and `value` (literal object)
* `object` is None/null (unary assertion)

#### 4.5.3 Span verification

If candidate has `char_start` and `char_end`:

1. Verify `0 <= char_start < char_end <= len(text_raw)`
2. Extract `surface_text = text_raw[char_start:char_end]`
3. If candidate includes `quote` field:
   * Verify `quote` matches `surface_text` (exact or after whitespace normalization)
   * If mismatch: set spans to NULL, log WARN with details

#### 4.5.4 Quote-to-offset resolution

If candidate has `quote` but no spans:

1. Search for exact match of `quote` in `text_raw`
2. If exactly one match found:
   * Set `char_start`, `char_end` from match
   * Extract `surface_text`
3. If zero or multiple matches:
   * Keep spans as NULL
   * Store `quote` in `raw_assertion_json`
   * Log WARN `QUOTE_OFFSET_UNRESOLVED`

#### 4.5.5 Invalid candidate handling

* Drop invalid candidates (do not persist)
* Log WARN with:
  * `message_id`
  * validation failure reason
  * original candidate data (for debugging)

---

### 4.6 Phase 4 — Grounding (entity/predicate linking + scoring)

#### 4.6.1 Subject resolution

Resolve `candidate.subject` to `subject_entity_id` and capture detection quality:

**Resolution algorithm (returns tuple: entity_id, detection_tier, resolution_confidence):**

1. **Reserved SELF reference:**
   * If `subject` is "SELF", "self", "I", "me", "my" → return (`SELF_entity_id`, 1, 1.0)

2. **Direct entity_id:**
   * If `subject` is a valid UUID in `entity_by_id`:
     * Retrieve `best_detector_tier` from EntityRecord
     * Return (`entity_id`, `best_detector_tier`, 1.0)

3. **Span-based resolution (preferred for assertions with offsets):**
   * If candidate has `char_start`/`char_end`:
     * Search entity mention map for overlapping mentions
     * If exactly one overlap: return (`entity_id`, `detector_tier`, `mention.confidence`)
     * If multiple overlaps: select by `(detector_tier ASC, confidence DESC, entity_id ASC)`

4. **Canonical name match:**
   * Search `entity_by_canonical_name[lowercase(subject)]`
   * If exactly one match:
     * tier = EntityRecord.best_detector_tier
     * Return (`entity_id`, tier, 0.95)
   * If multiple matches:
     * Select by `(best_detector_tier ASC, salience_score DESC NULLS LAST, entity_id ASC)`
     * Return (selected_id, tier, 0.8)  # lower confidence due to ambiguity

5. **Alias match:**
   * Search `entity_by_alias[lowercase(subject)]`
   * If exactly one match → return (`entity_id`, tier, 0.9)
   * If multiple matches → select as above, return with confidence 0.75

6. **Entity key match:**
   * Normalize `subject` using type-appropriate rules from Stage 2
   * Search by `entity_by_key`
   * If match → return (`entity_id`, tier, 0.9)

7. **Fuzzy match** (if `config.enable_fuzzy_entity_linking=true`):
   * Compute similarity against all canonical names + aliases
   * If best match ≥ `config.threshold_link_string_sim` (default 0.85):
     * Confidence = similarity_score * 0.8
     * Return (`entity_id`, tier, confidence)
   * Use deterministic similarity: Jaro-Winkler or normalized Levenshtein
   * Tie-break by `(salience_score DESC, best_detector_tier ASC, entity_id ASC)`

8. **Create new entity** (fallback):
   * Create entity with `entity_type = 'OTHER'`
   * `entity_key = normalize(subject)`
   * `canonical_name = subject`
   * Log WARN `ENTITY_CREATED_DURING_GROUNDING`
   * Return (`new_entity_id`, 4, 0.5)  # lowest tier, low confidence

**Store resolution result:**
* `subject_entity_id` = resolved entity_id
* `subject_detection_tier` = detector tier
* `subject_resolution_confidence` (internal, used in confidence composition)

#### 4.6.2 Object resolution

If candidate has entity object reference:
* Apply same resolution algorithm as subject
* Store:
  * `object_entity_id`
  * `object_detection_tier`
  * `object_resolution_confidence` (internal)

If candidate has literal object:
* Validate and normalize value:
  * `string`: store as JSON string
  * `number`: parse and store as JSON number
  * `boolean`: store as JSON `true`/`false`
  * `date`: normalize to ISO-8601, store as JSON string
  * `json`: validate and store canonical JSON
* Set `object_entity_id = NULL`, `object_detection_tier = NULL`

If unary (no object):
* Set all object fields to NULL

#### 4.6.3 Object signature computation
```
if object_entity_id is not NULL:
    object_signature = "E:" + object_entity_id
elif object_value is not NULL:
    object_signature = "V:" + sha256_hex(JCS([object_value_type, object_value]))
else:
    object_signature = "N:__NONE__"
```

#### 4.6.4 Predicate canonicalization

For each unique `predicate_label` in candidates:

1. **Normalize:**
   * Apply Unicode NFKC normalization
   * Trim leading/trailing whitespace
   * Collapse internal whitespace to single space
   * Result: `canonical_label`

2. **Compute lookup key:**
   * `canonical_label_norm = lowercase(canonical_label)`

3. **Generate ID:**
   * `predicate_id = uuid5(KG_NS_UUID, JCS(["pred", canonical_label_norm]))`

4. **Upsert:**
   * If exists by `canonical_label_norm` → retrieve existing row
   * Else insert new row with:
     * `canonical_label`, `canonical_label_norm`
     * `arity` inferred from candidate (1 if unary, 2 if binary)
     * `first_seen_at_utc` = current message timestamp
     * `assertion_count = 0` (updated in stats phase)

#### 4.6.5 Temporal qualifier linking

If candidate includes temporal qualifier:

1. **Direct time_mention_id reference:**
   * If provided and exists in `time_mentions` → use it

2. **Surface match in same message:**
   * Search `time_mentions` where `message_id` matches and `surface_text` equals qualifier surface
   * If multiple matches: select by `(confidence DESC, char_start ASC)`
   * If match found → link `temporal_qualifier_id`

3. **ISO value provided:**
   * Store `temporal_qualifier_type` from candidate
   * Store ISO value in `raw_assertion_json` under `temporal_qualifier_value`
   * Leave `temporal_qualifier_id = NULL`

4. **No linkable time:**
   * Set `temporal_qualifier_type = NULL`, `temporal_qualifier_id = NULL`
   * Log INFO `TEMPORAL_QUALIFIER_UNLINKED`

#### 4.6.6 Temporal bounds validation

If candidate references entities with temporal bounds:

1. **Check subject temporal bounds:**
   * If `subject_entity.first_seen_at_utc` is set AND assertion has temporal qualifier:
     * If `temporal_qualifier.valid_from_utc < subject_entity.first_seen_at_utc`:
       * Log WARN `ASSERTION_PRECEDES_ENTITY_FIRST_SEEN`
       * Apply confidence penalty: `temporal_penalty = config.temporal_bounds_violation_penalty` (default 0.1)

2. **Check object temporal bounds (if entity object):**
   * Apply same validation as subject

3. **Store validation results:**
   * Include `temporal_bounds_check` in `raw_assertion_json`:
     * `subject_bounds_valid`: true/false
     * `object_bounds_valid`: true/false (if applicable)
     * `penalty_applied`: float

#### 4.6.7 Corroboration detection (assistant assertions only)

If `asserted_role = 'assistant'`:

1. **Define corroboration window:**
   * Prior assertions in same conversation
   * Within `±config.coref_window_size` messages (default 10)
   * From `user` role only

2. **Search for corroborating assertions:**
   * Same `subject_entity_id`
   * Predicate match:
     * Exact: `predicate_id` matches
     * Or fuzzy: `canonical_label` similarity ≥ `config.predicate_similarity_threshold` (default 0.9)
   * Object compatible:
     * Entity objects: same `object_entity_id`
     * Literal objects: same `object_signature` OR normalized string equality

3. **If corroboration found:**
   * Set `has_user_corroboration = 1`
   * Store corroboration details in `raw_assertion_json`:
     * `corroborating_assertion_ids`
     * `match_type` (exact_predicate, fuzzy_predicate)
     * `similarity_score` (if fuzzy)

4. **If no corroboration:**
   * Set `has_user_corroboration = 0`

#### 4.6.8 Confidence composition (enhanced with detector awareness)

**Grounding confidence:**
```
# Base grounding confidence from entity resolution
subject_grounding = subject_resolution_confidence
object_grounding = object_resolution_confidence if object_entity_id else 1.0

# Detector tier bonus (higher-reliability entities boost confidence)
subject_tier_factor = 1.0 + (4 - subject_detection_tier) * config.detector_grounding_bonus
# default detector_grounding_bonus = 0.05
object_tier_factor = 1.0 + (4 - object_detection_tier) * config.detector_grounding_bonus if object_detection_tier else 1.0

# Compose grounding confidence
confidence_grounding = clamp(
    (subject_grounding * subject_tier_factor + object_grounding * object_tier_factor) / 2,
    0.0, 1.0
)
```

**Trust weight selection:**
```
if asserted_role == 'user':
    trust_weight = config.trust_weight_user  # default 1.0
elif has_user_corroboration == 1:
    trust_weight = config.trust_weight_assistant_corroborated  # default 0.9
else:
    trust_weight = config.trust_weight_assistant_uncorroborated  # default 0.5
```

**Salience factor (optional, default=true):**
```
if config.enable_salience_confidence_boost:
    subject_salience = entity_by_id[subject_entity_id].salience_score or 0.0
    # Normalize salience to [0, 1] range using corpus max
    salience_factor = 1.0 + (subject_salience / max_corpus_salience) * config.salience_confidence_bonus
    # default salience_confidence_bonus = 0.1
else:
    salience_factor = 1.0
```

**Temporal penalty (from §4.6.6):**
```
temporal_factor = 1.0 - temporal_penalty  # 0.0 if no violation
```

**Final confidence:**
```
confidence_final = clamp(
    confidence_extraction * confidence_grounding * trust_weight * salience_factor * temporal_factor,
    0.0, 1.0
)
```

#### 4.6.9 Key computation

**Assertion key** (for exact deduplication):
```
assertion_key = JCS([
    message_id,
    subject_entity_id,
    predicate_id,
    object_signature,
    char_start if char_start is not None else "__NULL__",
    modality,
    polarity
])
```

**Fact key** (for semantic deduplication):
```
fact_key = JCS([
    subject_entity_id,
    predicate_id,
    object_signature
])
```

---

### 4.7 Phase 5 — Persistence (assertions + retractions + stats)

#### 4.7.1 Assertion ID generation
```
assertion_id = uuid5(KG_NS_UUID, JCS([
    "assertion",
    assertion_key,
    sha256_hex(raw_assertion_json)
]))
```

#### 4.7.2 Assertion upsert

**Upsert policy** (`config.assertion_upsert_policy`):

* `keep_highest_confidence` (default):
  * Query existing by `assertion_key`
  * If exists and new `confidence_final` > existing + 0.001 → UPDATE
  * If exists and not higher → skip (keep existing)
  * If not exists → INSERT

* `keep_first`:
  * Query existing by `assertion_key`
  * If exists → skip
  * If not exists → INSERT

* `keep_all`:
  * Requires dropping unique constraint on `assertion_key`
  * Always INSERT (audit mode)

#### 4.7.3 Retraction extraction

Process **user** messages for retraction patterns:

**Pattern categories:**

1. **Full retraction:** "Actually, that's not true", "I was wrong about X"
2. **Correction:** "Actually, it's Y not X", "I meant Y"
3. **Temporal bound:** "That's no longer true", "I stopped X"

**Execution:**
1. Apply pinned retraction pattern registry to user message text
2. For each match:
   * Extract retraction type, target clause, replacement clause (if correction)
   * Compute span in `text_raw`

#### 4.7.4 Retraction linking

For each detected retraction:

1. **Attempt target resolution:**
   * Parse target clause using same grounding logic as assertions
   * Compute `target_fact_key` from parsed subject + predicate + object

2. **Link to prior assertion:**
   * Search `assertions` where `fact_key = target_fact_key`
   * Order by: `(asserted_at_utc DESC, message.order_index DESC, assertion_id DESC)`
   * If match found: set `target_assertion_id`
   * If ambiguous or no match: keep `target_assertion_id = NULL`, retain `target_fact_key`

3. **For corrections:**
   * Create replacement assertion from correction clause
   * Link: `retractions.replacement_assertion_id = new_assertion_id`
   * Update superseded assertion:
     * `superseded_by_assertion_id = new_assertion_id`
     * `supersession_type = 'correction'`

4. **Generate retraction ID:**
```
   retraction_id = uuid5(KG_NS_UUID, JCS([
       "retraction",
       retraction_message_id,
       char_start if char_start is not None else "__NULL__",
       char_end if char_end is not None else "__NULL__"
   ]))
```

5. **Insert retraction row**

#### 4.7.5 Stats refresh (deterministic recompute)

Recompute counts from persisted data:
```sql
UPDATE predicates
SET assertion_count = (
    SELECT COUNT(*) FROM assertions 
    WHERE assertions.predicate_id = predicates.predicate_id
)
```

Update `entities` counts only if configured (default: preserve Stage 2 values):
```sql
-- Only if config.update_entity_assertion_counts = true
-- Add assertion_count column to entities if using this
```

---

### 4.8 Phase 6 — Commit

#### 4.8.1 Finalize LLM run record

If LLM extraction enabled:
* Set `completed_at_utc`
* Compute `assertions_extracted` = count of assertions with `extraction_method IN ('llm', 'hybrid')`
* Compute `raw_stats_json`:
  * `calls_total`, `calls_succeeded`, `calls_failed`
  * `avg_latency_ms`
  * `retry_distribution`
* Insert `llm_extraction_runs` row

#### 4.8.2 Commit transaction

* Commit all changes
* Log summary:
  * Messages processed
  * Assertions inserted/updated
  * Retractions detected/linked
  * Predicates created
  * Entities created during grounding (should be low)
  * Duration
  * **Grounding quality stats:**
    * Assertions by subject_detection_tier
    * Assertions by object_detection_tier
    * Average confidence_grounding
    * Temporal bounds violations count

---

### 4.9 Deterministic processing order

**Global message iteration:**
`(conversation_id ASC, order_index ASC, message_id ASC)`

**Within-message candidate ordering:**
```
(
    char_start NULLS LAST,
    char_start ASC,
    char_end DESC,
    predicate_label_norm ASC,
    confidence DESC,
    sha256(JCS(candidate)) ASC  -- stable hash tie-break
)
```

**Corroboration search ordering:**
```
(
    order_index ASC,
    message_id ASC,
    assertion_id ASC
)
```

**Entity resolution tie-breaking:**
```
(
    best_detector_tier ASC,
    salience_score DESC NULLS LAST,
    entity_id ASC
)
```

---

### 4.10 Deterministic ID formulas (Stage 4)

| Entity         | Formula                                                                                |
| -------------- | -------------------------------------------------------------------------------------- |
| `assertion_id` | `uuid5(KG_NS_UUID, JCS(["assertion", assertion_key, sha256_hex(raw_assertion_json)]))` |
| `predicate_id` | `uuid5(KG_NS_UUID, JCS(["pred", canonical_label_norm]))`                               |
| `retraction_id`| `uuid5(KG_NS_UUID, JCS(["retraction", retraction_message_id, char_start, char_end]))`  |
| `run_id`       | `uuid5(KG_NS_UUID, JCS(["llm_run", started_at_utc, model_name]))`                      |
| `call_id`      | `uuid5(KG_NS_UUID, JCS(["llm_call", run_id, message_id, retry_count]))`                |

---

### 4.11 Pipeline Configuration (Stage 4)

**Detector-Aware Grounding:**
* `detector_grounding_bonus` (REAL): Per-tier grounding confidence bonus (default: `0.05`)

**Salience Integration:**
* `enable_salience_confidence_boost` (BOOLEAN): Use salience in confidence (default: `false`)
* `salience_confidence_bonus` (REAL): Max salience boost factor (default: `0.1`)

**Temporal Validation:**
* `enable_temporal_bounds_validation` (BOOLEAN): Validate against entity temporal bounds (default: `true`)
* `temporal_bounds_violation_penalty` (REAL): Confidence penalty for violations (default: `0.1`)

**Entity Resolution:**
* `enable_fuzzy_entity_linking` (BOOLEAN): Enable fuzzy matching (default: `true`)
* `threshold_link_string_sim` (REAL): Minimum similarity for fuzzy match (default: `0.85`)

**Trust Weights:**
* `trust_weight_user` (REAL): Trust weight for user assertions (default: `1.0`)
* `trust_weight_assistant_corroborated` (REAL): Trust weight for corroborated assistant assertions (default: `0.9`)
* `trust_weight_assistant_uncorroborated` (REAL): Trust weight for uncorroborated assistant assertions (default: `0.5`)

**Corroboration:**
* `coref_window_size` (INTEGER): Window size for corroboration detection (default: `10`)
* `predicate_similarity_threshold` (REAL): Minimum predicate similarity (default: `0.9`)

**Extraction:**
* `enable_llm_assertion_extraction` (BOOLEAN): Enable LLM-based extraction (default: `false`)
* `llm_temperature` (REAL): LLM temperature parameter (default: `0.0`)
* `llm_top_p` (REAL): LLM top_p parameter (default: `1.0`)
* `llm_seed` (INTEGER): LLM seed for reproducibility (optional)
* `llm_max_retries` (INTEGER): Max LLM call retries (default: `3`)
* `llm_multi_run_count` (INTEGER): Number of LLM runs per message (default: `1`)
* `k_context` (INTEGER): Number of preceding messages for context (default: `5`)
* `min_extractable_chars` (INTEGER): Minimum non-excluded chars (default: `10`)

**Persistence:**
* `assertion_upsert_policy` (TEXT): Deduplication policy (default: `"keep_highest_confidence"`)
* `update_entity_assertion_counts` (BOOLEAN): Update entity stats (default: `false`)

---

### 4.12 Stage 4 outputs and completion criteria

On successful commit, Stage 4 guarantees:

* All eligible messages processed in deterministic order
* Assertions inserted/upserted per configured policy
* All subjects and objects grounded to entities (existing or newly created)
* **Detector tier captured for entity-based assertions**
* **Grounding confidence reflects entity resolution quality and detector reliability**
* **Salience optionally incorporated into confidence scoring**
* **Temporal bounds validated against entity first/last seen timestamps**
* Predicates normalized and deduplicated
* Temporal qualifiers linked when unambiguous
* Corroboration flags set for assistant assertions
* Retractions detected and linked when unambiguous
* All LLM interactions fully logged for replayability
* Predicate stats refreshed

**Rollback conditions:**

* Any database constraint violation
* LLM API failure after max retries (if LLM required)
* Transaction deadlock

**Warning conditions (non-fatal):**

* Entity created during grounding (resolution failure)
* Span/quote mismatch
* Temporal qualifier unlinked
* Retraction target ambiguous
* **Temporal bounds violation detected**
* **Ambiguous entity resolution (multiple candidates)**


## Stage 5: Temporal Reasoning Layer

### 5.0 Purpose and invariants

#### Stage 5 responsibilities

1. Assign **valid-time** (`valid_from_utc`, `valid_to_utc`, `valid_time_type`) to each assertion conservatively.
2. Compute assertion lifecycle **status** (active/superseded/retracted/negated/conflicted/ineligible).
3. Apply **functional invalidation** (one value at a time) using `invalidation_rules` (must be provided).
4. Create **conflict groups** with deterministic IDs and membership.

#### Hard invariants

* **Deterministic iteration order**: `(conversation_id ASC, messages.order_index ASC, message_id ASC, assertion_id ASC)`.
* **Transactional**: Stage 5 is one transaction (commit or rollback).
* **Auditable**: every decision recorded as JCS-canonical JSON in `raw_temporalize_json` / `raw_conflict_json`.

#### Configuration parameters (Stage 5)

* `time_link_proximity_chars` (INTEGER): Max character distance for time linking (default: `200`)
* `time_link_min_alignment` (REAL): Minimum alignment score for time linking (default: `0.1`)
* `fallback_valid_from_asserted` (BOOLEAN): Use asserted_at as fallback valid_from (default: `true`)
* `threshold_close` (REAL): Confidence threshold for closing assertions (default: `0.7`)
* `confidence_supersession_margin` (REAL): Minimum confidence advantage required to supersede a prior assertion (default: `0.01`)
* `use_detector_tier_tiebreak` (BOOLEAN): Use detector tier as tie-breaker in supersession (default: `true`)
* `use_salience_conflict_tiebreak` (BOOLEAN): Use entity salience in conflict resolution tie-breaks (default: `false`)

**Adjustment notes (no schema change):**
* `asserted_at` fallback MUST NOT create *eligibility* for temporal operations.
* `asserted_at` fallback MUST be skipped if `messages.timestamp_quality != 'original'` (prevents anchoring to imputed timestamps).

---

### 5.1 Inputs and outputs

#### Inputs (must already exist from Stages 1–4)

* `messages(message_id, conversation_id, order_index, created_at_utc, timestamp_quality, role, …)`
* `entities(entity_id, entity_type, salience_score, first_seen_at_utc, last_seen_at_utc, …)`
* `entity_mentions(mention_id, message_id, entity_id, detector, detector_version, confidence, …)`
* `time_mentions(time_mention_id, message_id, char_start, char_end, resolved_type, valid_from_utc, valid_to_utc, confidence, …)`
* `assertions(assertion_id, message_id, subject_entity_id, subject_detection_tier, predicate_id, object_entity_id, object_detection_tier, object_signature, temporal_qualifier_type, temporal_qualifier_id, modality, polarity, asserted_role, asserted_at_utc, confidence_extraction, confidence_grounding, confidence_final, has_user_corroboration, superseded_by_assertion_id, supersession_type, …)`
* `retractions(retraction_id, retraction_message_id, target_assertion_id, target_fact_key, retraction_type, replacement_assertion_id, confidence, …)`

#### Outputs (Stage 5 tables)

##### assertion_temporalized

| column                                 | type      | notes                                                                    |
| -------------------------------------- | --------- | ------------------------------------------------------------------------ |
| `assertion_id`                         | TEXT PK   | FK → assertions                                                          |
| `valid_time_type`                      | TEXT      | instant, interval, unknown                                               |
| `valid_from_utc`                       | TEXT NULL | start of validity                                                        |
| `valid_to_utc`                         | TEXT NULL | end of validity (NULL = open)                                            |
| `valid_until_hint_utc`                 | TEXT NULL | hint from "until" qualifier when start unknown (see §5.4.4)              |
| `status`                               | TEXT      | active, superseded, retracted, negated, conflicted, ineligible           |
| `temporal_superseded_by_assertion_id`  | TEXT NULL | FK → assertions (temporal supersession, distinct from Stage 4 correction)|
| `retracted_by_retraction_id`           | TEXT NULL | FK → retractions                                                         |
| `negated_by_assertion_id`              | TEXT NULL | FK → assertions                                                          |
| `rule_id_applied`                      | TEXT NULL | FK → invalidation_rules                                                  |
| `raw_temporalize_json`                 | TEXT NULL | decision log (canonical JSON)                                            |

**Indices:**

* Primary key on `assertion_id`
* `idx_temporalized_status` on `status`
* `idx_temporalized_valid_from` on `valid_from_utc` WHERE `valid_from_utc IS NOT NULL`
* `idx_temporalized_superseded_by` on `temporal_superseded_by_assertion_id` WHERE `temporal_superseded_by_assertion_id IS NOT NULL`

---

##### invalidation_rules

| column                | type      | notes                                       |
| --------------------- | --------- | ------------------------------------------- |
| `rule_id`             | TEXT PK   | uuid5                                       |
| `predicate_id`        | TEXT NULL | FK → predicates (NULL for wildcard)         |
| `subject_entity_type` | TEXT      | entity type or '*' for wildcard             |
| `is_functional`       | INTEGER   | 0/1 (if 1, only one object allowed at a time) |
| `invalidation_policy` | TEXT      | none, close_previous_on_newer_state         |
| `notes`               | TEXT NULL |                                             |

**Indices:**

* Primary key on `rule_id`
* `idx_rules_predicate` on `predicate_id` WHERE `predicate_id IS NOT NULL`
* `idx_rules_subject_type` on `subject_entity_type`

---

##### conflict_groups

| column              | type      | notes                                                                                                                                         |
| ------------------- | --------- |-----------------------------------------------------------------------------------------------------------------------------------------------|
| `conflict_group_id` | TEXT PK   | uuid5                                                                                                                                         |
| `conflict_type`     | TEXT      | OBJECT_DISAGREEMENT, NEGATION_AMBIGUOUS, RETRACTION_TARGET_NOT_UNIQUE, OBJ_TRANSITION_CONFLICT, OBJ_CONFIDENCE_TOO_CLOSE, SAME_TIME_DUPLICATES |
| `conflict_key`      | TEXT      | deterministic grouping key                                                                                                                    |
| `detected_at_utc`   | TEXT      | use stage_started_at_utc                                                                                                                      |
| `raw_conflict_json` | TEXT NULL | canonical JSON                                                                                                                                |

**Indices:**

* Primary key on `conflict_group_id`
* `idx_conflict_groups_type` on `conflict_type`
* `idx_conflict_groups_key` UNIQUE on `conflict_key`

---

##### conflict_members

| column              | type    | notes             |
| ------------------- | ------- | ----------------- |
| `conflict_group_id` | TEXT FK | → conflict_groups |
| `assertion_id`      | TEXT FK | → assertions      |

**Indices:**

* `idx_conflict_members_pk` UNIQUE on `(conflict_group_id, assertion_id)`
* `idx_conflict_members_assertion` on `assertion_id`
* `idx_conflict_members_group` on `conflict_group_id`

---

### 5.2 Stage start: Database check & schema/migration + seeding

* Check that required tables from previous stages exist. Raise error if not and exit.

#### 5.2.1 Ensure Stage 5 tables exist

**Database:**

1. Begin transaction.
2. Check presence of Stage 4 tables; delete if exist (overwrite=True by default).
3. Create tables with indices and FK constraints.

#### 5.2.2 Seed invalidation rules deterministically

**Algorithm (rule seed):**

1. Ensure the **default rule** exists exactly as specified:

   * `predicate_id=NULL, subject_entity_type='*', is_functional=0, invalidation_policy='none'`.
   * `rule_id = uuid5(KG_NS_UUID, JCS(["invalidation_rule", "__NULL__", "*"]))`

2. Load `invalidation_rules.yaml` (already validated/loaded in Stage 0), and upsert rules deterministically:

   * For each rule, resolve predicate reference:
     * If `predicate_label` is provided: look up `predicate_id` via `predicates.canonical_label_norm`
     * If `predicate_id` is provided directly: use it
     * If neither (wildcard): use `NULL`
   * Compute `rule_id`:
     * `rule_id = uuid5(KG_NS_UUID, JCS(["invalidation_rule", predicate_id ?? "__NULL__", subject_entity_type]))`
   * Insert or update rule row.

3. For each inserted/updated rule, store provenance (file hash, version, load order) in `notes`.

---

### 5.3 Build the deterministic working stream (compartment)

#### 5.3.1 Assertion stream (logical view)

Create a conceptual stream (actual SQL view optional) joining:

* `assertions` → `messages` (order & timestamps)
* `assertions.subject_entity_id` → `entities` (to obtain `subject_entity_type` for rule selection, `salience_score` for tie-breaks)

**Stable sort key for all per-assertion processing:**
`(messages.conversation_id, messages.order_index, assertions.message_id, assertions.assertion_id)`

#### 5.3.2 Stage timestamp

Capture exactly one `stage_started_at_utc` at Stage 5 start and reuse it for all `conflict_groups.detected_at_utc`.

#### 5.3.3 Build entity salience index (for conflict tie-breaks)

If `config.use_salience_conflict_tiebreak=true`:
```
entity_salience: Dict[entity_id, float | None]
```

Populate from `entities.salience_score`. Used in §5.10 for deterministic tie-breaking.

---

### 5.4 Valid-time assignment

#### 5.4.0 Additional decision flags (stored in `raw_temporalize_json`)

For each assertion `A`, compute and log:

* `time_source` ∈ {`QUALIFIER_ID`, `PROXIMITY`, `ASSERTED_AT_FALLBACK`, `NONE`}
* `has_explicit_valid_time` (BOOLEAN):
  * `true` iff `time_source ∈ {QUALIFIER_ID, PROXIMITY}`
  * `false` otherwise
* `fallback_blocked_reason` ∈ {`TIMESTAMP_NOT_ORIGINAL`, `ASSERTED_AT_NULL`, `FALLBACK_DISABLED`} (nullable)
* `subject_detection_tier` (INTEGER): copied from `assertions.subject_detection_tier` for audit
* `object_detection_tier` (INTEGER NULL): copied from `assertions.object_detection_tier` for audit

These are *not* schema fields; they are appended into `raw_temporalize_json` under a pinned key, e.g.:
`{"time_link": {...}, "time_source": "...", "has_explicit_valid_time": true, "subject_detection_tier": 2, ...}`.

#### 5.4.1 Candidate time set per assertion

**Inputs:** assertion `A`, message `M`, time mentions in same message.

**Algorithm:**

1. `T0 = { time_mentions in message M where resolved_type ∈ {instant, interval} }`
2. If `A.temporal_qualifier_id` is non-NULL:

   * If that referenced time mention exists and is resolved → choose it; set:
     * `time_source = QUALIFIER_ID`
     * `has_explicit_valid_time = true`
     * skip proximity scoring
   * Else record `QUALIFIER_ID_MISSING_OR_UNRESOLVED` in log and proceed to proximity.

3. Else proceed to proximity-based selection.

#### 5.4.2 Proximity alignment scoring (span-aware, conservative)

**Goal:** pick a single time mention only when the evidence is strong enough.

**Definitions:**

* Let assertion span be `[a_start, a_end)` if present; else spanless.
* Let time span be `[t_start, t_end)` (time spans are required in Stage 2A).
* Define `gap(A,T)`:

  * `0` if spans overlap or touch (`a_end >= t_start` and `t_end >= a_start`)
  * else the positive distance between them:

    * if `a_end < t_start` then `t_start - a_end`
    * else `a_start - t_end`
* Alignment:

  * If A has span:

    * if `gap <= config.time_link_proximity_chars`: `alignment = 1 / (1 + gap)` else `0`
  * If A is spanless:

    * if `|T0| == 1`: `alignment = config.time_link_min_alignment`
    * else `alignment = 0` (avoid guessing)

#### 5.4.3 Selecting the best time mention

**Algorithm:**

1. For each candidate `T` in `T0`, compute `alignment(A,T)`.
2. Keep `Tcand = {T | alignment >= config.time_link_min_alignment}`.
3. If empty → no time mention chosen.
4. Else choose deterministic winner by max of:

   * primary score: `(T.confidence × alignment)` descending
   * tie-breakers (descending unless stated):

     1. `alignment`
     2. smaller `T.char_start` (earlier mention) **ASC**
     3. `T.time_mention_id` **ASC**

* If a proximity winner is selected, set:
  * `time_source = PROXIMITY`
  * `has_explicit_valid_time = true`

* If no time mention chosen, proceed to fallback rules (§5.4.5) and set:
  * `has_explicit_valid_time = false`

Record *all* candidates and scores in `raw_temporalize_json`.

#### 5.4.4 Interpret temporal qualifier type into valid-time

Stage 4 provides `temporal_qualifier_type ∈ {NULL, at, since, until, during}`.

Let chosen time mention resolve to either:

* `instant` with `t_from`
* `interval` with `[t_from, t_to)`

**Algorithm (qualifier semantics):**

1. If qualifier is `NULL` or `at`:

   * instant → `valid_time_type='instant'`, `valid_from=t_from`, `valid_to=NULL`
   * interval → `valid_time_type='interval'`, `valid_from=t_from`, `valid_to=t_to`

2. If qualifier is `during`:

   * require interval; if instant, treat as `at` (record `DURING_INSTANT_DOWNGRADED`)

3. If qualifier is `since`:

   * require `t_from` (instant or interval start)
   * set open-ended interval: `valid_time_type='interval'`, `valid_from=t_from`, `valid_to=NULL`

4. If qualifier is `until`:

   * This implies an end boundary without a known start. To stay conservative:
     * set `valid_time_type='unknown'`
     * set `valid_from_utc=NULL`, `valid_to_utc=NULL`
     * set `valid_until_hint_utc = t_from` (for instants) or `t_to` (for intervals)
     * record `UNTIL_QUALIFIER_HINT_STORED` in `raw_temporalize_json`
   * The `valid_until_hint_utc` field preserves this information for downstream consumers without inventing "-infinity" semantics.

#### 5.4.5 Fallback when no time mention is linkable (Adjusted)

Fallback is allowed only as a **non-world-time hint** unless explicit time evidence exists.
It may populate `valid_from_utc` for usability, but MUST NOT drive eligibility for temporal operations (see §5.5).

**Algorithm:**

If no chosen time mention:

1. If `config.fallback_valid_from_asserted=true`:

   * If `M.timestamp_quality != 'original'`:
     * set `valid_time_type='unknown'`, `valid_from=NULL`, `valid_to=NULL`
     * set `time_source = NONE`
     * set `fallback_blocked_reason = TIMESTAMP_NOT_ORIGINAL`
     * record `FALLBACK_BLOCKED_TIMESTAMP_NOT_ORIGINAL`
   * Else if `A.asserted_at_utc IS NULL`:
     * set `valid_time_type='unknown'`, null both
     * set `time_source = NONE`
     * set `fallback_blocked_reason = ASSERTED_AT_NULL`
     * record `FALLBACK_BLOCKED_ASSERTED_AT_NULL`
   * Else:
     * set `valid_time_type='instant'`, `valid_from=A.asserted_at_utc`, `valid_to=NULL`
     * set `time_source = ASSERTED_AT_FALLBACK`
     * set `has_explicit_valid_time = false`
     * record `FALLBACK_ASSERTED_AT_USED_NONEXPLICIT`

2. Else (fallback disabled):
   * set `valid_time_type='unknown'`, null both
   * set `time_source = NONE`
   * set `fallback_blocked_reason = FALLBACK_DISABLED`

**Important invariant:**
`time_source = ASSERTED_AT_FALLBACK` never implies `has_explicit_valid_time=true`.

#### 5.4.6 Integrity checks

If `valid_time_type='interval'` and `valid_to` is non-NULL and `valid_to <= valid_from`:

* force `valid_time_type='unknown'`, null both times
* record `VALID_TIME_NONPOSITIVE_INTERVAL`

---

### 5.5 Initial lifecycle status assignment

Stage 5 must decide which assertions can participate in closing/superseding logic.

#### 5.5.1 "User-grounded" test

`user_grounded(A) = (A.asserted_role='user') OR (A.has_user_corroboration=1)`

#### 5.5.2 "Eligible for temporal operations" test (Adjusted)

Eligibility determines whether an assertion can participate in:
* functional invalidation (closing/superseding)
* negation closure targeting (as a closeable active prior)
* any rule-driven state transition logic

**Eligible iff all:**

* `A.modality ∈ {state, fact, preference}`
* `A.polarity='positive'`
* `A.confidence_final >= config.threshold_close`
* `user_grounded(A)=true`
* `valid_from_utc IS NOT NULL`
* **`has_explicit_valid_time=true`** (from §5.4.0)

If not eligible → set initial `status='ineligible'` (unless later overridden by higher-precedence statuses).

**Required logging:**
Append `eligibility_reasons` list to `raw_temporalize_json`. Possible reasons:
* `MODALITY_EXCLUDED` when modality not in allowed set
* `NEGATIVE_POLARITY` when polarity is negative
* `CONFIDENCE_BELOW_THRESHOLD` when confidence_final < threshold
* `NOT_USER_GROUNDED` when neither user-asserted nor corroborated
* `VALID_FROM_NULL` when valid_from_utc is NULL
* `NONEXPLICIT_TIME_SOURCE` when `has_explicit_valid_time=false` (includes asserted_at fallback cases)

**Rationale:**
Prevents inventing world-time transitions from transaction-time timestamps (`asserted_at_utc`).

---

### 5.6 Populate / refresh `assertion_temporalized`

**Algorithm (temporalized upsert):**

For each assertion in deterministic order:

1. Compute valid-time (§5.4), producing `valid_*` plus flags (`time_source`, `has_explicit_valid_time`, etc.).
2. Compute eligibility/status (§5.5). Initial:
   * `status='active'` iff eligible
   * else `status='ineligible'`

Then `INSERT OR REPLACE` into `assertion_temporalized` with:
* computed valid fields
* `status`
* linkage fields cleared to NULL
* `raw_temporalize_json` includes:
  * time-link candidates & selection
  * qualifier interpretation
  * fallback usage and any fallback blocks
  * `time_source`, `has_explicit_valid_time`
  * eligibility evaluation
  * `subject_detection_tier`, `object_detection_tier` (from assertion)

---

### 5.7 Apply Stage 4 "correction supersessions" first

Stage 4 may already encode corrections via:

* `assertions.superseded_by_assertion_id`
* `assertions.supersession_type='correction'`

**Algorithm (correction projection):**
For each assertion `A` where `supersession_type='correction'` and `superseded_by_assertion_id` not NULL:

1. In `assertion_temporalized` for `A`, set:

   * `status='superseded'` (status precedence applies)
   * `temporal_superseded_by_assertion_id = A.superseded_by_assertion_id`

2. **Do not** set/force `valid_to_utc` (a correction is epistemic; time-of-world-change is unknown unless explicitly stated elsewhere).

3. Log `CORRECTION_SUPERSESSION_FROM_STAGE4` in `raw_temporalize_json`.

This prevents Stage 5 from later "re-closing" the corrected assertion as if it were a normal state transition.

---

### 5.8 Apply retractions

Stage 4 produces `retractions` with confidence and optional targeting.

**Algorithm (retraction application):**
For each retraction `R` in deterministic order (by `retraction_message_id`, `char_start`, `retraction_id`):

1. Require:

   * `R.confidence >= config.threshold_close`
   * retraction message is user-role: join `messages` on `R.retraction_message_id` and verify `role='user'`

2. Determine target:

   * If `R.target_assertion_id` exists → target that single assertion.
   * Else if `R.target_fact_key` exists:

     * find non-retracted assertions with that `fact_key`
     * if exactly one → target it
     * else → create conflict group `RETRACTION_TARGET_NOT_UNIQUE` and stop (no action)

3. Apply:

   * `assertion_temporalized.status='retracted'`
   * `retracted_by_retraction_id = R.retraction_id`

4. Do **not** set `valid_to_utc` (epistemic change).

5. Log details into both:

   * target assertion's `raw_temporalize_json` (append event)
   * `raw_conflict_json` when ambiguous

---

### 5.9 Apply negation closures

Negation logic remains structurally unchanged; however, eligibility rules already prevent closing/superseding based on asserted_at fallback for *positive* assertions. For negations:

* Negation assertions (`polarity='negative'`) still require `N.valid_from_utc` known and `user_grounded(N)=true` and `N.confidence_final >= threshold_close`.
* If `N.valid_from_utc` came from `ASSERTED_AT_FALLBACK`, it will only happen when timestamp quality is original; this is still *transaction-time* anchored. Stage 5 remains conservative by allowing negation closure only when the target is unambiguous; ambiguity yields conflict groups.

**Algorithm (negation closure):**

For each negative assertion `N` in deterministic order:

1. Preconditions:

   * `user_grounded(N)=true`
   * `N.confidence_final >= config.threshold_close`
   * `N.valid_from_utc` is known

2. Candidate targets `P`:

   * `P.status='active'` (not retracted/superseded/negated already)
   * `P` eligible (§5.5)
   * same `subject_entity_id` and `predicate_id`
   * object match:

     * binary: `P.object_signature == N.object_signature`
     * unary: both are `"N:__NONE__"`

3. If exactly one `P`:

   * set `P.valid_to_utc = min(existing_valid_to, N.valid_from_utc)` if existing non-null, else `N.valid_from_utc`
   * set `P.status='negated'`
   * set `P.negated_by_assertion_id = N.assertion_id`

4. Else:

   * create conflict group `NEGATION_AMBIGUOUS`
   * do not close anything

**Conflict key (deterministic):**
`JCS(["neg", subject_entity_id, predicate_id, N.object_signature, N.valid_from_utc ?? "__NULL__"])`

---

### 5.10 Functional invalidation

Functional invalidation uses the same rule selection and walk, but the candidate set is now protected from asserted-at anchoring artifacts because eligibility requires `has_explicit_valid_time=true`.

#### 5.10.1 Rule selection (deterministic)

For each assertion `A` needing rule context, determine selected rule by priority:

1. exact `(predicate_id, subject_entity_type)` match
2. `(predicate_id, '*')` match
3. `(NULL, subject_entity_type)` match
4. default rule `(predicate_id=NULL, subject_entity_type='*')`

#### 5.10.2 Candidate set for invalidation

For each unique `(subject_entity_id, predicate_id)` pair, consider only assertions where:

* `assertion_temporalized.status='active'`
* eligible (§5.5.2) **(now implies explicit time source)**
* not retracted
* `valid_from_utc IS NOT NULL`

If selected rule is not functional (`is_functional=0` or policy `none`) → skip group.

#### 5.10.3 Timepoint grouping + winner selection (enhanced with detector tier)

Group candidates by exact `valid_from_utc`:

* `G(t) = {a | a.valid_from_utc = t}`

Within each `G(t)`:

1. **Partition by `object_signature`.**

2. **If multiple object signatures at same timepoint:**

   * Create `OBJECT_DISAGREEMENT` conflict group for timepoint `t`
   * Mark all members in `G(t)` as `conflicted` (unless higher-precedence status already)
   * Do not select a representative for this timepoint
   * Continue to next timepoint

3. **If single object signature but multiple assertions (same-time duplicates):**

   * Create `SAME_TIME_DUPLICATES` conflict group for timepoint `t` (informational, not an error)
   * All assertions remain `active` but are recorded in the conflict group for visibility
   * Select one deterministic representative `W(t)` for temporal walk (step 4)

4. **Choose timepoint representative `W(t)` deterministically:**

   Sort candidates at timepoint `t` by:
   1. `confidence_final DESC`
   2. `min(subject_detection_tier, object_detection_tier ?? 5) ASC` (prefer higher-reliability detectors; use 5 as sentinel for NULL)
   3. `subject_detection_tier ASC`
   4. `assertion_id ASC`
   
   Select first as `W(t)`.

#### 5.10.4 Walk timepoints to supersede earlier states (enhanced)

Sort timepoints by:

1. `t = valid_from_utc ASC`
2. tie-break `W(t).assertion_id ASC`

Walk in order with `prev = None`:

* If `prev is None`: set `prev = W(t)` and continue.

* If `prev.object_signature == cur.object_signature`:

  * no closing needed (same state continues); set `prev = cur`

* Else (object differs):

  * If either `prev` or `cur` is `conflicted`:

    * do not close; ensure `OBJ_TRANSITION_CONFLICT` group exists
    * set `prev = cur`

  * Else (both non-conflicted):

    * **Compute effective confidence advantage:**
```
      conf_diff = cur.confidence_final - prev.confidence_final
```

    * **If `config.use_detector_tier_tiebreak=true` and `abs(conf_diff) < config.confidence_supersession_margin`:**
      * Compare detection tiers:
        * `cur_tier = min(cur.subject_detection_tier, cur.object_detection_tier ?? 5)`
        * `prev_tier = min(prev.subject_detection_tier, prev.object_detection_tier ?? 5)`
      * If `cur_tier < prev_tier`: treat as if `cur` wins (higher reliability)
      * Else if `cur_tier > prev_tier`: treat as if confidence is too close (conflict)
      * Else: fall through to confidence-only logic

    * If `cur.confidence_final >= prev.confidence_final + config.confidence_supersession_margin` (or won via tier):

      * supersede `prev` at time `t`:

        * `prev.valid_to_utc = min(prev.valid_to_utc, t)` if prev.valid_to_utc non-null, else `t`
        * `prev.status='superseded'`
        * `prev.temporal_superseded_by_assertion_id = cur.assertion_id`
        * `prev.rule_id_applied = selected_rule_id`
      * Log supersession reason in `raw_temporalize_json`:
        * `supersession_reason`: `CONFIDENCE_MARGIN` or `DETECTOR_TIER_TIEBREAK`
        * `confidence_diff`, `prev_tier`, `cur_tier`

    * Else:

      * mark both `conflicted`
      * create conflict group `OBJ_CONFIDENCE_TOO_CLOSE`
      * Include in `raw_conflict_json`:
        * `confidence_diff`, `subject_detection_tiers`, `object_detection_tiers`

    * set `prev = cur`

All decisions per `(subject, predicate)` should be summarized into each involved assertion's `raw_temporalize_json` (append-only structure).

---

### 5.11 Conflict finalization

**Algorithm (group materialization):**
Whenever a conflict is detected:

1. Compute `conflict_key` (JCS array; stable components per conflict type).

2. `conflict_group_id = uuid5(KG_NS_UUID, JCS(["conflict", conflict_key]))`

3. `INSERT OR IGNORE conflict_groups(conflict_group_id, conflict_type, conflict_key, detected_at_utc=stage_started_at_utc, raw_conflict_json)`

4. For each involved assertion:

   * `INSERT OR IGNORE conflict_members(conflict_group_id, assertion_id)`

This yields deduped conflict groups even if detected multiple times via different passes.

**Conflict key formulas by type:**

| conflict_type                  | conflict_key formula                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------- |
| `OBJECT_DISAGREEMENT`          | `JCS(["obj_disagree", subject_entity_id, predicate_id, valid_from_utc])`                          |
| `NEGATION_AMBIGUOUS`           | `JCS(["neg", subject_entity_id, predicate_id, object_signature, valid_from_utc ?? "__NULL__"])`   |
| `RETRACTION_TARGET_NOT_UNIQUE` | `JCS(["retract_ambig", retraction_id, target_fact_key])`                                          |
| `OBJ_TRANSITION_CONFLICT`      | `JCS(["trans_conflict", subject_entity_id, predicate_id, prev_valid_from, cur_valid_from])`       |
| `OBJ_CONFIDENCE_TOO_CLOSE`     | `JCS(["conf_close", subject_entity_id, predicate_id, prev_assertion_id, cur_assertion_id])`       |
| `SAME_TIME_DUPLICATES`         | `JCS(["same_time_dup", subject_entity_id, predicate_id, object_signature, valid_from_utc])`       |

---

### 5.12 Status precedence and update rule

Status precedence (highest to lowest):
`retracted > negated > superseded > conflicted > active > ineligible`

**Algorithm (safe status update):**
When a pass wants to set a new status:

* update only if `precedence(new) > precedence(current)`

This prevents a later pass (e.g., functional invalidation) from downgrading a retracted assertion.

---

### 5.13 Commit and Stage 5 handoff contract

On commit, Stage 5 guarantees:

* Every `assertions.assertion_id` has exactly one `assertion_temporalized` row.
* `assertion_temporalized` is the **authoritative** place for:

  * `valid_from_utc`, `valid_to_utc`, `valid_time_type`, `valid_until_hint_utc`
  * lifecycle `status`
  * temporal supersession link (`temporal_superseded_by_assertion_id`)
  * negation/retraction links

**Consumption notes:**

* Read lifecycle edges from `assertion_temporalized`, not from Stage 4's `assertions.superseded_by_assertion_id` (which is correction-focused).
* The `valid_until_hint_utc` field can be used to generate a `VALID_UNTIL_HINT` edge type if desired.
* `SAME_TIME_DUPLICATES` conflict groups are informational; member assertions remain `active` and should all appear in the graph.
* Detection tier information is preserved in `raw_temporalize_json` for audit and can inform downstream analysis.

---

### 5.14 Deterministic ID formulas

| Entity             | Formula                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| `rule_id`          | `uuid5(KG_NS_UUID, JCS(["invalidation_rule", predicate_id ?? "__NULL__", subject_entity_type]))`     |
| `conflict_group_id`| `uuid5(KG_NS_UUID, JCS(["conflict", conflict_key]))`                                                 |

**NULL encoding (consistent with §1.3):**

* `None` / missing → `"__NULL__"`
* `""` (empty string) → `"__EMPTY__"`

---

### 5.15 Transaction boundary and completion

Stage 5 runs inside a **single DB transaction**:

* Begin transaction at Stage 5 start
* Execute phases: seed rules → populate temporalized → apply corrections → apply retractions → apply negations → functional invalidation → finalize conflicts
* Commit only if all steps succeed; otherwise rollback fully

**Stage 5 completion criteria:**

* All assertions have `assertion_temporalized` rows with computed valid-time and status
* Correction supersessions from Stage 4 projected
* Retractions applied where unambiguous
* Negation closures applied where unambiguous
* Functional invalidation applied per rules (with detector-tier-aware tie-breaking)
* Conflict groups materialized with complete membership
* All decisions recorded in `raw_temporalize_json` / `raw_conflict_json`
* Detection tier information preserved in audit logs

---


## Stage 6: Graph Materialization

### 6.0 Goal

Materialize **deterministic, read-only graph tables** (`graph_nodes`, `graph_edges`) for visualization/export.

**Authoritative source:** Stages 1–5 tables provide facts, provenance, lifecycle, and valid-time; Stage 6 only *projects* this into a stable graph shape (no upstream mutation).

**Primary visualization target (supported explicitly):** a **SELF-centered temporal profile** suitable for “monthly clusters of what mattered”:

`SELF(Entity) → TemporalProfile(month) → TemporalWindow(YYYY-MM) → SemanticCluster(category) → Members(Entity/Value)`
with optional shortcut edges for “top items” and optional evidence/time qualifier drill-down.

**Hard invariants (carried forward):**

* No mutation of Stages 1–5 tables.
* Deterministic iteration + stable tie-breaks.
* `metadata_json` stored as **JCS-canonical JSON**.
* Single transaction: commit or rollback fully.

---

### 6.0.1 Stage 6 available input tables (read-only)

Stage 6 may read from any prior-stage tables, notably:

**Stage 1:** `conversations`, `messages`, `message_parts`
**Stage 2A:** `entity_mention_candidates`, `entity_mentions`, `time_mentions`
**Stage 2B:** `entities`, `lexicon_builds`, `lexicon_term_candidates`, `lexicon_terms`
**Stage 3:** `entity_canonicalization_runs`, `entity_canonical_name_history`
**Stage 4:** `predicates`, `assertions`, `llm_extraction_runs`, `llm_extraction_calls`, `retractions`
**Stage 5:** `assertion_temporalized`, `invalidation_rules`, `conflict_groups`, `conflict_members`

Stage 6 creates only: `graph_nodes`, `graph_edges`.

---

### 6.0.2 Data Quality Check

Run preliminary analysis on required tables; collect and log:

* counts: entities, predicates, assertions, temporalized assertions
* `assertion_temporalized` coverage (% of assertions with temporal row)
* status distribution (`active/superseded/retracted/negated/conflicted/ineligible`)
* valid-time availability (`valid_from_utc` non-null rate) and date span
* literal object usage rate (`object_value_type/object_value` present) — important for tasks/plans/desires
* SELF presence: `entities.entity_key='__SELF__' AND entity_type='PERSON' AND status='active'`
* optional availability: `time_mentions` and `assertions.temporal_qualifier_id` linkage rate

If insufficient, log expected impact (e.g., “few valid_from_utc → sparse windows”, “few literal objects → fewer task-like items”).

---

### 6.1 Graph model

#### 6.1.1 `graph_nodes`

| column          | type    | notes                                                      |
| --------------- | ------- | ---------------------------------------------------------- |
| `node_id`       | TEXT PK | uuid5 per §6.2                                             |
| `node_type`     | TEXT    | see inventory below                                        |
| `source_id`     | TEXT    | ID from source table (or deterministic hash for synthetic) |
| `label`         | TEXT    | human-friendly label                                       |
| `metadata_json` | TEXT    | JCS-canonical JSON (typed payload; §6.4)                   |

**Node types**

* Core: `Entity`, `Predicate`, `Assertion`, `Value`, `TimeInterval`
* Optional provenance: `Message`, `Retraction`, `ConflictGroup`, `LexiconTerm`
* Optional temporal profile: `TemporalProfile`, `TemporalWindow`, `SemanticCluster`
* Optional category/time UI anchors: `SemanticCategory` (stable category nodes), `TimeMention` (from Stage 2A)

**Recommended indices**

* `idx_graph_nodes_type` on `(node_type, node_id)`
* `idx_graph_nodes_source` on `(node_type, source_id)`

#### 6.1.2 `graph_edges`

| column          | type      | notes              |
| --------------- | --------- | ------------------ |
| `edge_id`       | TEXT PK   | uuid5 per §6.3     |
| `edge_type`     | TEXT      | see §6.5           |
| `src_node_id`   | TEXT FK   | → graph_nodes      |
| `dst_node_id`   | TEXT FK   | → graph_nodes      |
| `metadata_json` | TEXT NULL | JCS-canonical JSON |

**Recommended indices**

* `idx_graph_edges_src` on `(src_node_id)`
* `idx_graph_edges_dst` on `(dst_node_id)`
* `idx_graph_edges_type` on `(edge_type, edge_id)`

---

### 6.2 Node ID construction (deterministic)

**General rule**

* `node_id = uuid5(KG_NS_UUID, JCS(["node", node_type, source_id]))`

**Synthetic node `source_id` rules**

* `TimeInterval.source_id = sha256_hex(JCS([valid_from_utc_or_null, valid_to_utc_or_null]))`

  * includes hint-only nodes using `[NULL, valid_until_hint_utc]`
* `Value.source_id = sha256_hex(JCS([object_value_type, object_value]))`
* `TemporalWindow.source_id = sha256_hex(JCS([entity_id, window_start_utc, window_end_utc, granularity]))`
* `SemanticCluster.source_id = sha256_hex(JCS([window_source_id, category_key, classification_tier]))`
* `TemporalProfile.source_id = sha256_hex(JCS([entity_id, granularity, profile_version]))`
* `SemanticCategory.source_id = category_key` (canonical string key, e.g. `"project"`)
* `TimeMention.source_id = time_mention_id`

**NULL encoding**

* missing/None → `"__NULL__"`
* empty string → `"__EMPTY__"`

---

### 6.3 Edge ID construction (deterministic, collision-safe)

* `edge_id = uuid5(KG_NS_UUID, JCS(["edge", edge_type, src_node_id, dst_node_id]))`

---

### 6.4 Node generation rules

Nodes are generated in **type order** and inserted with stable sorting:
`(node_type ASC, node_id ASC)`.

#### 6.4.1 Deterministic inputs (authoritative read streams)

Read upstream tables as stable, ordered streams (IDs ascending; message order stable).

**Required join stream**

* Assertions stream joins:

  * `assertions A`
  * `assertion_temporalized T` (authoritative lifecycle + valid-time)
  * `messages M` (for ordering; required if message nodes enabled)

**Other streams**

* active entities
* predicates
* optional: messages, retractions, conflict groups + members, lexicon terms
* optional: time_mentions (for TimeMention nodes)

#### 6.4.2 Node inventory (set-accumulate, then sorted insert)

Accumulate unique `(node_type, source_id)` then insert.

Create:

* **Entity**: one per active entity.
* **Predicate**: one per predicate.
* **Message** (optional): one per message if emitting `ASSERTED_IN`.
* **Assertion**: one per assertion; temporal fields sourced from `assertion_temporalized`.
* **Value**: one per literal `(object_value_type, object_value)` present in any assertion.
* **TimeInterval**:

  * `(valid_from_utc, valid_to_utc)` when `valid_from_utc` present
  * hint-only `(NULL, valid_until_hint_utc)` when `valid_until_hint_utc` present
  * never for `(NULL, NULL)`
* **Retraction**: one per retraction.
* **ConflictGroup**: one per conflict group.
* **LexiconTerm** (optional): one per lexicon term.
* **TimeMention** (optional): one per time mention (either all, or only those referenced by `assertions.temporal_qualifier_id` when configured).
* **SemanticCategory** (optional): one per pinned category key used by temporal profiles.
* **TemporalProfile / TemporalWindow / SemanticCluster** (optional): created by §6.6.

#### 6.4.3 Labels + metadata_json (pinned payload shapes)

All `metadata_json` is JCS-canonical and includes `schema_version` (default `"1.0"`).

(Existing node payloads remain as originally specified: Entity, Predicate, Message, Assertion, Value, TimeInterval, Retraction, ConflictGroup, LexiconTerm.)

**SemanticCategory (optional)**

* `label`: display label (e.g., `"Project"`)
* `metadata_json`: `{category_key, display_label}`

**TimeMention (optional)**

* `label`: `surface_text` (truncated)
* `metadata_json`: `{time_mention_id, message_id, surface_text, resolved_type, valid_from_utc, valid_to_utc, resolution_granularity, timezone_assumed, confidence, anchor_time_utc, pattern_id}`

---

### 6.5 Edge generation rules

Edges are generated after nodes exist, in stable order:
`(edge_type ASC, edge_id ASC)`.

#### 6.5.1 Assertion semantic + temporal edges

For each joined `(A, T)`:

**Semantic**

* `HAS_SUBJECT`: `Assertion → Entity(subject_entity_id)`
* `HAS_PREDICATE`: `Assertion → Predicate(predicate_id)`
* `HAS_OBJECT` (exclusive):

  * entity object: `Assertion → Entity(object_entity_id)`
  * literal object: `Assertion → Value(value_source)`
  * unary: none

**Temporal (from Stage 5)**

* `VALID_IN`: `Assertion → TimeInterval([valid_from_utc, valid_to_utc])` when `valid_from_utc` present
* `VALID_UNTIL_HINT`: `Assertion → TimeInterval([NULL, valid_until_hint_utc])` when hint present

**Optional semantic-edge metadata**

* include detection tier on subject/object edges when enabled

**Optional time qualifier provenance (if TimeMention enabled)**

* `QUALIFIED_BY_TIME`: `Assertion → TimeMention(temporal_qualifier_id)` when `assertions.temporal_qualifier_id` present

  * edge metadata: `{temporal_qualifier_type}`

#### 6.5.2 Message anchoring (optional)

If message nodes are enabled:

* `ASSERTED_IN`: `Assertion → Message(message_id)`
* Log deterministic warning if link cannot be created.

#### 6.5.3 Lifecycle edges (from Stage 5 pointers)

* `SUPERSEDES`: `Assertion(newer) → Assertion(older)` when `temporal_superseded_by_assertion_id` present

  * edge metadata may include applied rule id
* `RETRACTED_BY`: `Assertion → Retraction` when `retracted_by_retraction_id` present

  * optional inverse `RETRACTS`
* `NEGATED_BY`: `Assertion → Assertion(negator)` when `negated_by_assertion_id` present

  * optional inverse `NEGATES`

#### 6.5.4 Conflicts (group-preserving; optional pairwise fan-out)

* `HAS_CONFLICT_MEMBER`: `ConflictGroup → Assertion` for each member (optional metadata: `{conflict_type}`)
* Optional `CONFLICTS_WITH` among members if group size ≤ cap, using deterministic direction rule.

#### 6.5.5 Lexicon provenance (optional, default=true)

If lexicon nodes enabled:

* `DERIVED_FROM_LEXICON`: `Entity → LexiconTerm` on deterministic key match (metadata includes build_id/score)

#### 6.5.6 Temporal-profile navigation + “what mattered” shortcuts (optional additions)

If temporal profiles enabled:

**Required anchor**

* `HAS_PROFILE`: `Entity(focal) → TemporalProfile` (metadata: `{granularity, resolution_mode}`)

**Window ordering (optional but recommended, default=true)**

* `WINDOW_PRECEDES`: `TemporalWindow(i) → TemporalWindow(i+1)` (metadata: `{granularity}`)

**Category anchoring (optional; choose one deterministic pattern)**

* Pattern A (preferred): `CLUSTER_OF_CATEGORY`: `SemanticCluster → SemanticCategory`
* Pattern B: `WINDOW_HAS_CATEGORY`: `TemporalWindow → SemanticCategory` (clusters still carry category metadata)

**Top shortcuts (optional but recommended default=true)**

* `WINDOW_TOP_MEMBER`: `TemporalWindow → (Entity|Value)` for top-N members across categories

  * metadata: `{rank, salience, member_node_type, category_key}`
* `WINDOW_TOP_CLUSTER`: `TemporalWindow → SemanticCluster` ranked by deterministic score

  * metadata: `{rank, score, category_key}`

**Member evidence drill-down (optional; capped, default=true)**

* `MEMBER_EVIDENCED_BY`: `(Entity|Value) → Assertion` for assertions contributing to that member in-window

  * metadata: `{window_start_utc, window_end_utc, granularity, mention_count_in_window}`
  * enforce deterministic caps to prevent blow-up.

#### 6.5.7 Reserved future edges

* `MERGED_INTO`: `Entity → Entity`
* `CANONICAL_NAME_CHANGED`: `Entity → …`

---

### 6.6 Temporal Profile Visualization (optional; designed for “self + monthly clusters”)

#### 6.6.0 Summary

**Goal:** For focal entity (esp. SELF), build time-partitioned summaries that cluster salient linked **Entity and Value** members into pinned semantic categories (projects/interests/problems/tasks/desires/plans/…).

**Deterministic method**

1. Resolve focal actor(s)
2. Partition eligible assertions into aligned windows
3. Compute within-window salience for candidate members (Entity + Value)
4. Classify members into categories via tiered deterministic strategies
5. Materialize profile/window/cluster nodes + edges (and optional navigation shortcuts)

Determinism: stable sorts everywhere; fixed tie-breaks; optional LLM fallback must be seeded and majority-voted with logged IO.

---

#### 6.6.1 Configuration requirements

If `enable_temporal_profiles=true`, `temporal_profile_config.yaml` must exist (else deterministic config error), unless an explicit “defaults allowed” policy is specified.

Config shape remains as originally specified, with these expectations:

* category keys are **pinned** (recommended: `project, interest, problem, task, desire, plan, unclassified`)
* a default granularity (recommended: `month`) is present
* salience weights are specified (or defaulted deterministically)
* optional toggles for embedding/LLM tiers are pinned and seeded

**temporal_profile_config shape (conceptual):**

```yaml
temporal_profile_config:

  # Keyword rules per category
  strong_rule_keywords:
    project: [ ... ]
    interest: [ ... ]
    problem: [ ... ]
    task: [ ... ]
    desire: [ ... ]
    plan: [ ... ]

  # Embedding-based predicate clustering
  predicate_cluster_categories:
    project: [ ...seed predicates... ]
    interest: [ ... ]
    problem: [ ... ]
```

---

#### 6.6.2 Phase T0 — Actor/SELF resolution (deterministic)

Resolve focal entity list:

* If `config.temporal_profile_focal_entities` provided: use as-is.
* Else try SELF:

  * `entities.entity_key='__SELF__' AND entity_type='PERSON' AND status='active'` (expect ≤1; if >1, pick lowest `entity_id` and log)
* Else fallback: choose subject entity with most eligible assertions:

  * `T.status IN config.allowed_assertion_statuses`
  * `T.valid_from_utc` present
  * tie-break `entity_id ASC`

Record resolution mode in profile metadata.

---

#### 6.6.3 Phase T1 — Time window partitioning

For each focal entity `E` and granularity `G`:

1. Collect eligible assertions where:

   * `A.subject_entity_id = E`
   * `T.status IN allowed_assertion_statuses`
   * `T.valid_from_utc` present
2. Determine `[range_start, range_end]` from min/max parsed timestamps (not lexicographic strings).
3. Generate aligned, non-overlapping windows covering the range (UTC-aligned to month/quarter/year boundaries).
4. Assign each assertion to exactly one window by parsed datetime comparison:

   * `window_start <= valid_from < window_end`
5. Drop windows with fewer than `window_min_assertions`.

Output `TemporalWindow` records with deterministic `window_source_id = sha256(JCS([E, start, end, G]))`.

---

#### 6.6.4 Phase T2 — Window salience recomputation (Entity + Value)

For each window:

**Candidate members**

* Entity members: unique `object_entity_id` in window assertions
* Value members: unique `Value.source_id` for literal objects in window assertions

For each candidate member compute:

* **frequency:** (# window assertions mentioning member) / (total assertions in window)
* **recency:** deterministic time-based recency from `valid_from_utc` timestamps (exponential decay optional; seeded constants)
* **confidence:** mean `confidence_final` over mentions
* **window_salience:** weighted combination using config weights

---

#### 6.6.5 Phase T3 — Semantic clustering (tiered classification)

Categories are pinned (recommended):
`project, interest, problem, task, desire, plan, unclassified`

For each member-in-window with `window_salience > 0`:

1. Evidence predicates: predicate labels from assertions linking `E → member` within the window.
2. Optional additional signals (deterministic):

   * `assertions.modality` (intention/preference informative)
   * `predicates.category` if populated
   * for Value members only: bounded keyword rules over value text (deterministic, capped)
3. Classification tiers (deterministic chain):

   * Tier 1: keyword rules over predicate labels (vote + deterministic tie-break)
   * Tier 2 (optional): seeded predicate embedding clustering to category seeds + threshold
   * Tier 3 (optional): seeded multi-run LLM with consensus threshold; log prompts/responses
4. If none: assign `unclassified` tier `0`.

---

#### 6.6.6 Phase T4 — Profile node and edge generation

Create nodes (using §6.2 rules) and edges (using §6.3 rules):

**TemporalProfile**

* one per `(entity_id, granularity)` with ≥1 window
* metadata: `{entity_id, granularity, window_count, total_assertions, date_range, profile_version, resolution_mode}`

**TemporalWindow**

* one per window
* metadata: `{entity_id, granularity, bounds, assertion_count, cluster_count, top_members/top_entities summary}`

**SemanticCluster**

* one per `(window, category_key)` with ≥1 member
* metadata: `{category_key, member_count, avg_salience, avg_classification_confidence, tier_distribution}`

**Edges**

* required:

  * `HAS_PROFILE`: `Entity → TemporalProfile`
  * `HAS_WINDOW`: `TemporalProfile → TemporalWindow` (metadata: `{window_index}`)
  * `HAS_CLUSTER`: `TemporalWindow → SemanticCluster`
  * `CLUSTER_CONTAINS`: `SemanticCluster → (Entity|Value)` (rank + salience + tier metadata)
* optional:

  * `WINDOW_INCLUDES`: `TemporalWindow → Assertion`
  * `EVOLVES_TO`: `SemanticCluster(Wn) → SemanticCluster(Wn+1)` for same category across adjacent windows (metadata may include overlap stats)
  * `WINDOW_PRECEDES`, `WINDOW_TOP_MEMBER`, `WINDOW_TOP_CLUSTER`
  * `CLUSTER_OF_CATEGORY` / `WINDOW_HAS_CATEGORY`
  * `MEMBER_EVIDENCED_BY` (capped)

---

### 6.7 Configuration parameters (Stage 6)

**Core graph generation**

* `include_message_nodes` (bool)
* `include_lexicon_nodes` (bool)
* `include_detection_tier_metadata` (bool)
* `conflict_pairwise_max_n` (int)
* `include_inverse_lifecycle_edges` (bool)

**Temporal profiles**

* `enable_temporal_profiles` (bool)
* `temporal_profile_config_path` (path; required if enabled unless defaults policy)
* `temporal_profile_focal_entities` (list[str] | null)
* `allowed_assertion_statuses` (list[str]) for profile eligibility (recommended default: `["active"]`)
* `include_window_assertion_edges` (bool)
* `include_cluster_evolution_edges` (bool)

**Visualization-focused optional expansions**

* `include_value_members_in_profiles` (bool, default true)
* `include_category_nodes` (bool, default true)
* `include_window_sequence_edges` (bool, default true)
* `include_window_top_edges` (bool, default true)
* `window_top_n_members` (int, default 10)
* `window_top_n_clusters` (int, default 6)
* `include_time_mention_nodes` (bool, default false)
* `include_time_qualifier_edges` (bool, default false)
* `include_member_evidence_edges` (bool, default false)
* `member_evidence_edge_cap_per_window` (int, default 2000; deterministic truncation)

---

### 6.8 Stage 6 transaction and completion

**Algorithm**

1. Begin transaction.
2. (Re)create `graph_nodes`, `graph_edges`.
3. Materialize core nodes (§6.4) then core edges (§6.5.1–§6.5.5).
4. If temporal profiles enabled:

   * validate config (or apply deterministic defaults if policy allows)
   * run T0–T4 (§6.6)
   * emit required profile anchors + optional navigation shortcuts (§6.5.6)
5. Log stats (and optionally write to `build_meta` notes if present), commit.
6. On any error: rollback fully.

**Completion criteria**

* Every assertion has an Assertion node.
* Lifecycle/valid-time edges reflect `assertion_temporalized`.
* Conflicts representable without mandatory O(n²) edges (group + membership sufficient).
* If temporal profiles enabled:

  * focal entity connects via `HAS_PROFILE`
  * windows exist and are time-aligned; clusters exist by category
  * cluster members include **Entity + Value** when enabled
  * optional top shortcuts make “what mattered” queryable without deep traversal

**Stage 6 statistics to log**

* `nodes_by_type`, `edges_by_type`
* `entities_with_salience`
* `assertions_by_status`
* `assertions_by_detection_tier`
* `conflict_groups_by_type`
* `lexicon_terms_linked` (if enabled)
* `temporal_profiles_generated`, `temporal_windows_generated` (if enabled)
* `semantic_clusters_by_category`, `classifications_by_tier` (if enabled)
* `llm_classification_calls` (if enabled)
* optional: `time_mentions_linked`, `window_top_edges_emitted`, `member_evidence_edges_emitted`, `profile_members_by_type`
