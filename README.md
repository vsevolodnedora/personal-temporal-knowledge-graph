# Personal Bitemporal Knowledge Graph from ChatGPT Conversations

> **Transform your ChatGPT conversation history into a queryable, temporal knowledge graph with deterministic, auditable extraction**

A principled pipeline for extracting structured knowledge from conversational AI interactions, implementing semantic-lossless preservation, bitemporal reasoning, and role-aware trust weighting.

## Overview

This project converts ChatGPT `conversations.json` exports into a fully materialized knowledge graph that tracks:
- **What was discussed** (entities, relationships, assertions)
- **When things were true** (valid-time) vs. when we learned them (transaction-time)
- **How facts evolved** (supersessions, retractions, corrections)
- **What mattered when** (temporal profiles with semantic clustering)

The system is designed for **reproducibility and auditability**: every extraction decision is logged, every ID is deterministic, and every stage can be replayed with identical results.

## Key Principles

1. **Lossless Preservation First**: All raw JSON stored in canonical form (RFC 8785 JCS)
2. **Offsets are Sacred**: Text spans must exactly match source or be marked NULL
3. **Deterministic Everywhere**: Same input always produces same output (stable sorts, pinned models, seeded LLMs)
4. **Role-Aware Trust**: User statements weighted higher than assistant claims
5. **Transactional Stages**: Each stage commits fully or rolls back completely

## Features

- ✅ **Entity Detection**: Email, URLs, NER names, personal lexicon terms
- ✅ **Time Resolution**: Conservative temporal grounding with interval support
- ✅ **Detector Reliability Tiers**: Structured detectors (email, URL) prioritized over statistical methods (NER)
- ✅ **Assertion Extraction**: Hybrid rule-based + optional LLM extraction with corroboration detection
- ✅ **Bitemporal Reasoning**: Functional invalidation, supersession tracking, conflict detection
- ✅ **Temporal Profiles**: Monthly "what mattered" summaries with semantic clustering (projects, interests, tasks, etc.)
- ✅ **Graph Materialization**: Ready-to-visualize nodes and edges with rich metadata

## Repository Structure

```
.
├── data/
│   ├── figures/          # Output plots and visualizations
│   ├── metadata/         # Configuration YAML files
│   │   ├── export_mapping.yaml
│   │   ├── invalidation_rules.yaml
│   │   └── temporal_profile_config.yaml
│   ├── output/           # Generated database (kg.db)
│   └── raw/              # Input conversations.json
│
├── docs/
│   └── project_plan.md  # Detailed technical specification (Stages 1-6)
│
├── tkg/                  # Main source directory
│   ├── stage_1_load_preprocess_pipeline.py
│   ├── stage_2A_entity_time_detection_pipeline.py
│   ├── stage_2B_personal_lexicon_entity_consolidation_pipeline.py
│   ├── stage_3_entity_canonicalization_pipeline.py
│   ├── stage_4_assertion_extraction_and_grounding_pipeline.py
│   ├── stage_5_temporal_reasoning_pipeline.py
│   └── stage_6_graph_materialization_pipeline.py
│
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- SQLite 3.35+
- spaCy model: `python -m spacy download en_core_web_sm`

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/chatgpt-knowledge-graph.git
cd chatgpt-knowledge-graph

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

1. **Export your ChatGPT conversations** (Settings → Data Controls → Export Data)
2. **Place `conversations.json`** in `data/raw/`
3. **Run the full pipeline:**

```bash
# Stage 1: Import and preprocessing
python tkg/stage_1_load_preprocess_pipeline.py \
    --input data/raw/conversations.json \
    --output data/output/kg.db

# Stage 2A: Entity and time detection
python tkg/stage_2A_entity_time_detection_pipeline.py --db data/output/kg.db

# Stage 2B: Personal lexicon and entity consolidation
python tkg/stage_2B_personal_lexicon_entity_consolidation_pipeline.py --db data/output/kg.db

# Stage 3: Entity canonicalization
python tkg/stage_3_entity_canonicalization_pipeline.py --db data/output/kg.db

# Stage 4: Assertion extraction and grounding
python tkg/stage_4_assertion_extraction_and_grounding_pipeline.py --db data/output/kg.db

# Stage 5: Temporal reasoning
python tkg/stage_5_temporal_reasoning_pipeline.py \
    --db data/output/kg.db \
    --rules data/metadata/invalidation_rules.yaml

# Stage 6: Graph materialization
python tkg/stage_6_graph_materialization_pipeline.py \
    --db data/output/kg.db \
    --temporal-profile-config data/metadata/temporal_profile_config.yaml
```

4. **Query your knowledge graph** using SQLite or your preferred graph visualization tool!

---

## Stage Documentation

### Stage 1: Raw Ingest & Preprocessing

**Purpose**: Import conversations with semantic-lossless preservation, flatten hierarchy, compute message threading, and detect code fences/blockquotes.

**What it does**:
- Parses `conversations.json` using configurable export mapping
- Flattens into 3 core tables: `conversations`, `messages`, `message_parts`
- Computes deterministic tree paths for message threading
- Extracts text with exact character offsets
- Detects Markdown code fences and blockquotes for later exclusion
- Stores all raw JSON in JCS-canonical form

**Usage**:
```bash
python tkg/stage_1_load_preprocess_pipeline.py \
    --input data/raw/conversations.json \
    --output data/output/kg.db \
    --export_mapping data/metadata/export_mapping.yaml
```

**Key Options**:
- `--input`: Path to ChatGPT export JSON (default: `../data/raw/conversations.json`)
- `--output`: Output SQLite database path (default: `../data/output/kg.db`)
- `--export_mapping`: Export format mapping YAML (default: `../data/metadata/export_mapping.yaml`)

**Output Tables**: `conversations`, `messages`, `message_parts`

---

### Stage 2A: Entity & Time Detection Layer

**Purpose**: Detect entity mentions (emails, URLs, names, etc.) and time expressions with conservative resolution, using offset-correct spans.

**What it does**:
- Runs structured detectors: EMAIL, URL, DOI, UUID, HASH_HEX, IP_ADDRESS, PHONE, FILEPATH, BARE_DOMAIN
- Optional detectors: arXiv, CVE, ORCID, @handles, #hashtags
- Runs NER (spaCy) with configurable model and label filtering
- Detects and resolves time mentions (instants, intervals, relative expressions)
- Excludes code fences and optionally blockquotes from detection
- Resolves overlaps with deterministic winner selection (confidence × detector tier × span length)
- Stores all candidates for auditability, emits only winners

**Usage**:
```bash
python tkg/stage_2A_entity_time_detection_pipeline.py \
    --db data/output/kg.db \
    --ner-model en_core_web_sm \
    --ignore-blockquotes
```

**Key Options**:
- `--db`: SQLite database path (default: `../data/output/kg.db`)
- `--id-namespace`: UUID namespace for deterministic IDs (default: `550e8400-e29b-41d4-a716-446655440000`)
- `--timezone`: Default timezone for time resolution (default: `UTC`)
- `--no-ner`: Disable NER detection
- `--ner-model`: spaCy model name (default: `en_core_web_sm`)
- `--ignore-blockquotes`: Exclude blockquotes from entity detection
- `--enable-handles`: Enable @handle detection
- `--enable-hashtags`: Enable #hashtag detection
- `-v, --verbose`: Enable verbose logging

**Output Tables**: `entity_mention_candidates`, `entity_mentions`, `time_mentions`, `ner_model_runs` (if NER enabled)

---

### Stage 2B: Personal Lexicon & Entity Consolidation

**Purpose**: Learn personal terms from corpus patterns (project names, acronyms, nicknames) and consolidate entity mentions into canonical entities.

**What it does**:
- **Lexicon Induction**: Extracts candidates via deterministic patterns (TitleCase, ALLCAPS, CamelCase, #hashtags, @handles, "quoted phrases")
- Aggregates candidates with role-weighted counting (user=1.0, assistant=0.5)
- Filters by thresholds (min mentions, min conversations, max code-likeness, min context diversity)
- Selects top-N terms via composite scoring
- **LexiconMatch Detection**: Scans corpus with learned terms, emits new mentions with verified offsets
- **Entity Consolidation**: Creates canonical entities with normalized keys, links all mentions
- Seeds reserved **SELF** entity for user representation
- Computes entity salience scores

**Usage**:
```bash
python tkg/stage_2B_personal_lexicon_entity_consolidation_pipeline.py \
    --db data/output/kg.db \
    --min-mentions 3 \
    --min-conversations 2 \
    --max-terms 1000
```

**Key Options**:
- `--db`: SQLite database path (default: `../data/output/kg.db`)
- `--namespace`: UUID namespace for IDs (default: `550e8400-e29b-41d4-a716-446655440000`)
- `--min-mentions`: Minimum user-weighted mentions for term selection (default: `3`)
- `--min-conversations`: Minimum conversation count for term selection (default: `2`)
- `--max-terms`: Maximum lexicon terms to select (default: `1000`)
- `--denylist`: Path to additional denylist file (optional)
- `--disable-lexicon`: Skip lexicon induction, only run entity consolidation

**Output Tables**: `entities`, `lexicon_builds`, `lexicon_term_candidates`, `lexicon_terms` (+ updated `entity_mentions`)

---

### Stage 3: Entity Canonicalization Layer

**Purpose**: Refine entity canonical names using detector-weighted, role-weighted mention evidence.

**What it does**:
- Aggregates all surface forms per entity across mentions
- Applies **detector reliability tiers**:
  - Tier 1 (highest): EMAIL, URL, DOI, UUID, IP_ADDRESS, PHONE
  - Tier 2: HASH_HEX, FILEPATH, BARE_DOMAIN, arXiv, CVE, ORCID
  - Tier 3: LEXICON:* (learned terms)
  - Tier 4 (lowest): NER:* (statistical models)
- Computes composite weighted score per surface:
  - `score = (role_weight × detector_weight × confidence_boost × mention_count)`
- Selects canonical name via deterministic tie-breaks (score → detector tier → frequency → temporal order → lexicographic)
- Applies lexicon bonus for CUSTOM_TERM entities
- Tracks all changes in audit history

**Usage**:
```bash
python tkg/stage_3_entity_canonicalization_pipeline.py \
    --db data/output/kg.db \
    --trust-user 1.0 \
    --trust-assistant 0.5 \
    --detector-tier1 1.0 \
    --detector-tier4 0.6
```

**Key Options**:
- `--db`: SQLite database path (default: `../data/output/kg.db`)
- `--namespace`: UUID namespace (default: `550e8400-e29b-41d4-a716-446655440000`)
- `--trust-user`: Trust weight for user mentions (default: `1.0`)
- `--trust-assistant`: Trust weight for assistant mentions (default: `0.5`)
- `--detector-tier1` through `--detector-tier4`: Detector tier weights (defaults: 1.0, 0.9, 0.8, 0.6)
- `--tier-confidence-bonus`: Per-tier confidence bonus (default: `0.05`)
- `--lexicon-bonus`: Bonus for lexicon-preferred surface (default: `0.2`)
- `--salience-priority`: Enable salience-adjusted processing order
- `--method`: Canonicalization method (default: `detector_role_weighted`)
- `-v, --verbose`: Enable verbose logging

**Output Tables**: `entity_canonicalization_runs`, `entity_canonical_name_history` (+ updated `entities.canonical_name`)

---

### Stage 4: Assertion Extraction & Grounding Layer

**Purpose**: Extract semantic assertions (subject-predicate-object triples) from messages and ground them to canonical entities.

**What it does**:
- **Extraction**: Hybrid approach combining:
  - Rule-based patterns (deterministic, fast)
  - Optional LLM extraction (with full I/O logging, multi-run consensus)
- **Grounding**: Links subject/object references to entities via:
  - Span-based resolution (preferred: exact character offsets)
  - Canonical name matching
  - Alias matching
  - Fuzzy string similarity (Jaro-Winkler)
  - Fallback: create new entity
- **Captures detector tier** for each resolved entity
- **Corroboration detection**: Searches for user evidence supporting assistant claims
- **Confidence composition**: Combines extraction confidence, grounding quality, detector reliability, trust weights, and salience
- **Temporal qualifier linking**: Links assertions to time mentions when unambiguous
- **Temporal bounds validation**: Warns if assertion precedes entity's first appearance
- **Retraction detection**: Extracts explicit corrections and negations

**Usage**:
```bash
python tkg/stage_4_assertion_extraction_and_grounding_pipeline.py \
    --db data/output/kg.db \
    --k-context 5 \
    --trust-user 1.0 \
    --trust-assistant-corroborated 0.9 \
    --trust-assistant-uncorroborated 0.5
```

**Key Options**:
- `--db`: SQLite database path (default: `../data/output/kg.db`)
- `--enable-llm`: Enable LLM-based assertion extraction (default: disabled)
- `--llm-model`: LLM model name (default: `claude-sonnet-4-20250514`)
- `--k-context`: Number of prior messages for context (default: `5`)
- `--trust-user`: Trust weight for user assertions (default: `1.0`)
- `--trust-assistant-corroborated`: Trust weight for corroborated assistant assertions (default: `0.9`)
- `--trust-assistant-uncorroborated`: Trust weight for uncorroborated assistant assertions (default: `0.5`)
- `--ignore-blockquotes`: Exclude blockquotes from extraction
- `--upsert-policy`: Deduplication policy: `keep_highest_confidence` (default), `keep_first`, `keep_all`
- `--fuzzy-threshold`: String similarity threshold for entity linking (default: `0.85`)
- `--coref-window`: Window size for corroboration detection (default: `10`)
- `--detector-grounding-bonus`: Per-tier grounding confidence bonus (default: `0.05`)
- `--enable-salience-boost`: Use salience in confidence scoring
- `--salience-bonus`: Max salience boost factor (default: `0.1`)
- `--enable-temporal-validation`: Validate against entity temporal bounds (default: enabled)
- `--temporal-penalty`: Confidence penalty for temporal violations (default: `0.1`)
- `-v, --verbose`: Enable verbose logging

**Output Tables**: `predicates`, `assertions`, `retractions`, `llm_extraction_runs`, `llm_extraction_calls` (if LLM enabled)

---

### Stage 5: Temporal Reasoning Layer

**Purpose**: Assign valid-time to assertions, compute lifecycle status, and apply functional invalidation rules.

**What it does**:
- **Valid-Time Assignment**:
  - Links assertions to time mentions via proximity-based alignment
  - Falls back to message timestamp (if high quality) as non-world-time hint
  - Interprets temporal qualifiers (at, since, until, during)
  - Conservatively marks unanchored assertions as `valid_time_type='unknown'`
- **Lifecycle Management**:
  - Projects Stage 4 correction supersessions
  - Applies user retractions (where unambiguous)
  - Applies negation closures (where unambiguous)
  - Runs functional invalidation per configured rules (e.g., "person can only have one location at a time")
- **Supersession Logic**:
  - Groups assertions by timepoint and object signature
  - Uses detector tier as tie-breaker when confidence is close
  - Creates temporal SUPERSEDES edges
  - Closes prior assertions by setting `valid_to_utc`
- **Conflict Detection**:
  - Detects object disagreements, ambiguous negations, close confidence, same-time duplicates
  - Creates conflict groups with deterministic membership

**Usage**:
```bash
python tkg/stage_5_temporal_reasoning_pipeline.py \
    --db data/output/kg.db \
    --rules data/metadata/invalidation_rules.yaml \
    --threshold-close 0.7 \
    --confidence-margin 0.01
```

**Key Options**:
- `--db`: SQLite database path (default: `../data/output/kg.db`)
- `--rules`: Path to `invalidation_rules.yaml` (optional; loads default rule if missing)
- `--time-link-proximity`: Max character distance for time linking (default: `200`)
- `--threshold-close`: Confidence threshold for closing assertions (default: `0.7`)
- `--confidence-margin`: Minimum confidence advantage for supersession (default: `0.01`)
- `--no-fallback`: Disable `asserted_at` fallback for `valid_from`
- `--no-detector-tier-tiebreak`: Disable detector tier as supersession tie-breaker
- `--use-salience-tiebreak`: Enable entity salience in conflict resolution
- `--namespace`: UUID namespace (default: `550e8400-e29b-41d4-a716-446655440000`)
- `-v, --verbose`: Enable verbose logging

**Output Tables**: `assertion_temporalized`, `invalidation_rules` (seeded), `conflict_groups`, `conflict_members`

---

### Stage 6: Graph Materialization

**Purpose**: Materialize read-only graph tables (`graph_nodes`, `graph_edges`) for visualization and export.

**What it does**:
- **Core Graph Generation**:
  - Creates nodes: Entity, Predicate, Assertion, Value (for literal objects), TimeInterval
  - Creates semantic edges: HAS_SUBJECT, HAS_PREDICATE, HAS_OBJECT
  - Creates temporal edges: VALID_IN, VALID_UNTIL_HINT, SUPERSEDES, RETRACTED_BY, NEGATED_BY
  - Creates conflict edges: HAS_CONFLICT_MEMBER, optional pairwise CONFLICTS_WITH
- **Optional Provenance**:
  - Message nodes + ASSERTED_IN edges
  - LexiconTerm nodes + DERIVED_FROM_LEXICON edges
  - TimeMention nodes + QUALIFIED_BY_TIME edges
- **Temporal Profiles** (optional, recommended):
  - Resolves focal entity (SELF or highest-salience)
  - Partitions assertions into aligned time windows (typically monthly)
  - Computes window salience for linked entities AND literal values
  - Classifies members into semantic categories (project, interest, problem, task, desire, plan, unclassified) via:
    - Tier 1: Keyword rules over predicate labels
    - Tier 2 (optional): Predicate embedding clustering
    - Tier 3 (optional): LLM classification with consensus
  - Creates profile nodes: TemporalProfile → TemporalWindow → SemanticCluster → (Entity|Value)
  - Optional navigation shortcuts: WINDOW_TOP_MEMBER, WINDOW_TOP_CLUSTER, MEMBER_EVIDENCED_BY

**Usage**:
```bash
python tkg/stage_6_graph_materialization_pipeline.py \
    --db data/output/kg.db \
    --temporal-profile-config data/metadata/temporal_profile_config.yaml \
    --window-top-n-members 10 \
    --windowing-mode mention_first \
    --evergreen-penalty 0.3
```

**Key Options**:
- `--db`: SQLite database path (default: `../data/output/kg.db`)
- `--temporal-profile-config`: Path to profile config YAML (default: `../data/metadata/temporal_profile_config.yaml`)
- `--id-namespace`: UUID namespace (default: `550e8400-e29b-41d4-a716-446655440000`)
- `--no-message-nodes`: Disable message node generation
- `--no-lexicon-nodes`: Disable lexicon term node generation
- `--no-temporal-profiles`: Disable temporal profile generation
- `--include-time-mentions`: Include time mention nodes
- `--include-time-qualifier-edges`: Include time qualifier edges
- `--include-member-evidence-edges`: Include member evidence drill-down edges
- `--include-cluster-evolution`: Include cluster evolution edges across windows
- `--include-window-assertions`: Include window-to-assertion edges
- `--no-value-members`: Exclude literal values from profiles (only entities)
- `--window-top-n-members`: Top members per window for visualization (default: `10`)
- `--cluster-top-n-members`: Max members per cluster (default: `50`)
- `--member-evidence-cap`: Max evidence edges per member per window (default: `50`)
- `--windowing-mode`: Window assignment mode: `mention_first` (default), `mention_any`, `validity_overlap`
- `--evergreen-penalty`: Penalty strength 0.0-1.0 for items appearing in many months (default: `0.3`)
- `--no-topic-nodes`: Disable Topic node generation

**Output Tables**: `graph_nodes`, `graph_edges`

---

## Configuration Files

### `export_mapping.yaml`

Maps ChatGPT export JSON structure to pipeline expectations. Customize if your export format differs.

```yaml
format_version: "1.0"
conversation_id_path: "/id"
conversation_title_path: "/title"
messages_path: "/mapping"
messages_is_mapping: true
message_id_path: "/id"
message_role_path: "/author/role"
# ... (see data/metadata/export_mapping.yaml for full example)
```

### `invalidation_rules.yaml`

Defines functional invalidation rules for temporal reasoning (e.g., "person can only work at one company at a time").

```yaml
rules:
  - predicate_label: "works_at"
    subject_entity_type: "PERSON"
    is_functional: true
    invalidation_policy: "close_previous_on_newer_state"
    notes: "A person can only work at one company at a time"
# ... (see data/metadata/invalidation_rules.yaml for full example)
```

### `temporal_profile_config.yaml`

Configures temporal profile generation: semantic categories, keyword rules, embedding seeds, LLM settings.

```yaml
temporal_profile_config:
  granularity: "month"
  
  strong_rule_keywords:
    project:
      - "project"
      - "building"
      - "working on"
    interest:
      - "interested in"
      - "fascinated by"
    # ... (see data/metadata/temporal_profile_config.yaml for full example)
```

---

## Output Database Schema

The pipeline produces a single SQLite database (`kg.db`) with tables organized by stage:

### Stage 1 Tables
- `conversations`: Conversation metadata with raw JSON
- `messages`: Flattened messages with tree paths, extracted text, code fence/blockquote detection
- `message_parts`: Individual message content parts (text, images, files, tool calls)

### Stage 2 Tables
- `entity_mention_candidates`: All detector outputs (audit trail)
- `entity_mentions`: Winning mentions after overlap resolution
- `time_mentions`: Detected and resolved time expressions
- `entities`: Canonical entities with salience scores
- `lexicon_builds`, `lexicon_term_candidates`, `lexicon_terms`: Personal lexicon artifacts
- `ner_model_runs`: NER execution logs (if enabled)

### Stage 3 Tables
- `entity_canonicalization_runs`: Canonicalization execution metadata
- `entity_canonical_name_history`: Audit trail of name changes

### Stage 4 Tables
- `predicates`: Normalized relation vocabulary
- `assertions`: Grounded semantic claims with confidence scoring
- `retractions`: Detected corrections and negations
- `llm_extraction_runs`, `llm_extraction_calls`: LLM extraction artifacts (if enabled)

### Stage 5 Tables
- `assertion_temporalized`: Valid-time and lifecycle status per assertion
- `invalidation_rules`: Functional invalidation rules
- `conflict_groups`, `conflict_members`: Detected conflicts with membership

### Stage 6 Tables
- `graph_nodes`: Materialized nodes (Entity, Predicate, Assertion, Value, TimeInterval, TemporalProfile, etc.)
- `graph_edges`: Materialized edges (HAS_SUBJECT, VALID_IN, SUPERSEDES, HAS_CLUSTER, etc.)

All tables store metadata in JCS-canonical JSON format for reproducibility.

---

## Example Queries

### Find all entities mentioned
```sql
SELECT DISTINCT e.canonical_name, e.entity_type, e.salience_score
FROM entities e
JOIN entity_mentions em ON e.entity_id = em.entity_id
JOIN messages m ON em.message_id = m.message_id
WHERE m.role = 'user'
ORDER BY e.salience_score DESC NULLS LAST
LIMIT 50;
```

### Find your active assertions about a specific entity
```sql
SELECT 
    p.canonical_label AS predicate,
    COALESCE(eo.canonical_name, a.object_value) AS object,
    at.valid_from_utc,
    at.valid_to_utc,
    a.confidence_final
FROM assertions a
JOIN assertion_temporalized at ON a.assertion_id = at.assertion_id
JOIN entities es ON a.subject_entity_id = es.entity_id
JOIN predicates p ON a.predicate_id = p.predicate_id
LEFT JOIN entities eo ON a.object_entity_id = eo.entity_id
WHERE es.canonical_name = 'YourEntityName'
  AND at.status = 'active'
ORDER BY at.valid_from_utc DESC;
```

### Find what were working on in a specific month
```sql
SELECT 
    gn_entity.label AS item,
    gn_cluster.metadata_json->>'$.category_key' AS category,
    CAST(gn_entity.metadata_json->>'$.window_salience' AS REAL) AS salience
FROM graph_nodes gn_window
JOIN graph_edges ge_has_cluster ON ge_has_cluster.src_node_id = gn_window.node_id
JOIN graph_nodes gn_cluster ON gn_cluster.node_id = ge_has_cluster.dst_node_id
JOIN graph_edges ge_contains ON ge_contains.src_node_id = gn_cluster.node_id
JOIN graph_nodes gn_entity ON gn_entity.node_id = ge_contains.dst_node_id
WHERE gn_window.node_type = 'TemporalWindow'
  AND gn_window.metadata_json->>'$.window_start_utc' LIKE '2024-03%'
  AND ge_has_cluster.edge_type = 'HAS_CLUSTER'
  AND ge_contains.edge_type = 'CLUSTER_CONTAINS'
ORDER BY salience DESC
LIMIT 20;
```

---

## Troubleshooting

### "No conversations found in input file"
- Verify your `conversations.json` is valid JSON
- Check `export_mapping.yaml` paths match your export structure
- Try `--verbose` flag for detailed parsing logs

### "NER model not found"
```bash
python -m spacy download en_core_web_sm
```

### "Temporal profiles generated: 0"
- Verify `temporal_profile_config.yaml` exists and is valid
- Check that Stage 5 produced assertions with `status='active'` and `valid_from_utc` non-null
- Try `--verbose` to see eligibility filtering details

### "Database locked" errors
- Close any SQLite browsers/tools connected to `kg.db`
- Ensure no other pipeline processes are running

### Low entity counts after Stage 2B consolidation
- Expected: overlapping mentions get merged into single entities
- Check `entity_mention_candidates` table to see raw detector outputs
- Adjust Stage 2A detector settings if needed (e.g., `--enable-handles`, `--enable-hashtags`)

---

## Project Status

This project represents a **learning marathon** rather than a production-ready system. The implementation closely follows the detailed specification in `docs/project_plan.md`, but there are known areas where theory outran implementation:

### Known Limitations
- Some temporal reasoning edge cases not fully tested
- LLM extraction (Stage 4) optional and lightly exercised
- Graph visualization tooling not included (outputs are SQLite tables)
- Embedding-based classification (Stage 6) implemented but not extensively tuned
- Limited testing on non-English conversations

### Open Issues
See [GitHub Issues](your-issues-link) for:
- Edge case handling in temporal supersession
- Performance optimization for large conversation histories (>10K messages)
- Additional detector types (e.g., ISBNs, ISSNs, more domain-specific patterns)
- Graph export formats (Neo4j, GraphML, etc.)

---

## Contributing

This is primarily a personal learning project, but I'm happy to discuss:
- Design choices and trade-offs
- Knowledge graph principles and temporal reasoning
- Bugs or inconsistencies between documentation and implementation
- Ideas for extensions or improvements

Feel free to:
- Open issues for bugs or questions
- Submit PRs for fixes (please discuss significant changes first)
- Fork and adapt for your own use cases

---

## Acknowledgments

This project synthesizes ideas from:
- **Bitemporal Data Management** (Snodgrass, Jensen)
- **Knowledge Graph Construction** best practices (Noy, McGuinness)
- **Deterministic Systems Design** principles
- The ChatGPT conversation export format and community

Special thanks to anyone who reads `docs/project_plan.md` in its entirety—you're braver than most.

---

## License

MIT License - see LICENSE file for details.

---

**Questions? Feedback?** Open an issue or reach out!