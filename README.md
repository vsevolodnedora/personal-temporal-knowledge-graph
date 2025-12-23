# Personal Temporal Knowledge Graph from ChatGPT Conversations

Build a semantic-lossless, bitemporal personal knowledge graph from ChatGPT `conversations.json` export with deterministic, auditable transformations.

## Overview

Extracts structured knowledge from ChatGPT conversation history, preserving full provenance and temporal evolution. Converts nested JSON conversations into a queryable SQLite database, then applies NLP pipelines to build a temporal knowledge graph tracking entities, relationships, and their evolution over time.

## Key Principles

- **Semantic-lossless**: Never discard export content; preserve unknown structures as JCS-canonical JSON
- **Deterministic**: Same input always produces same output (pinned versions, stable ordering, canonical serialization)
- **Bitemporal**: Track both transaction time (message timestamps) and valid time (real-world claim periods)
- **Auditable**: Every derived record references source message_id with character-level spans
- **Incremental**: Staged pipeline with transactional commits (rollback on failure)

## Technical Stack

- **Storage**: SQLite with strict schema enforcement
- **Canonicalization**: RFC 8785 (JCS) for deterministic JSON serialization
- **Identifiers**: UUIDv5 namespace-based for reproducible IDs
- **Timestamps**: UTC ISO-8601 with milliseconds (`YYYY-MM-DDTHH:MM:SS.sssZ`)
- **JSON Pointers**: RFC 6901 for flexible schema mapping

## Installation

```bash
git clone https://github.com/vsevolodnedora/personal-temporal-knowledge-graph.git
cd personal-temporal-knowledge-graph
pip install -r requirements.txt
```

## Usage

### 1. Export Your Conversations

1. ChatGPT Settings → Data Controls → Export Data
2. Download and extract the ZIP file
3. Locate `conversations.json`

### 2. Configure Export Mapping

Create or verify `export_mapping.yaml` matches your export format:

```yaml
format_version: "1.0"
conversation_id_path: "/id"
conversation_title_path: "/title"
messages_path: "/mapping"
messages_is_mapping: true
# ... (see export_mapping.yaml for full schema)
```

### 3. Run Pipeline

```bash
python my_pipeline.py \
  --input conversations.json \
  --output kg.db \
  --mapping export_mapping.yaml
```

Or programmatically:

```python
from tkg.extraction_pipeline import run_pipeline
from pathlib import Path

run_pipeline(
    input_file_path=Path("../data/raw/conversations.json"),
    output_file_path=Path("../data/output/kg.db"),
    export_mapping_path=Path("../data/metadata/export_mapping.yaml")
)
```

## Pipeline Stages

### **Stage 0: Preprocessing**
- Compute input SHA-256 hashes
- Validate export mapping and config
- Generate build fingerprint for reproducibility

### **Stage 1: Raw Ingestion (Current)**
- Parse conversations and messages with flexible schema mapping
- Generate deterministic IDs (conversation_id, message_id)
- Preserve full raw JSON (JCS-canonical)
- Extract text content and detect code fences/blockquotes
- Compute tree paths for conversation branching

### Stage 2: Entity Recognition (Planned)
- Named Entity Recognition with span preservation
- Entity linking and deduplication
- Coreference resolution

### Stage 3: Assertion Extraction (Planned)
- LLM-based fact extraction from user and assistant messages
- Subject-predicate-object triple generation
- Confidence scoring and source attribution

### Stage 4: Temporal Reasoning (Planned)
- Temporal expression detection and normalization
- Valid time interval assignment
- Conflict detection and resolution (functional predicates, negations)

### Stage 5: Graph Materialization (Planned)
- Export to graph format (nodes, edges)
- Visualization preparation
- Query interface

## Database Schema

**conversations**: Conversation metadata with raw JSON preservation
**messages**: Individual messages with tree structure and text extraction
**message_parts**: Structured content parts (text, images, tool calls)
**entities**: Extracted entities with types and aliases *(Stage 2+)*
**assertions**: Subject-predicate-object triples with temporal validity *(Stage 3+)*
**predicates**: Relationship types with metadata *(Stage 3+)*

## Configuration

All deterministic parameters defined in `config.yaml`:

```yaml
KG_NS_UUID: "550e8400-e29b-41d4-a716-446655440000"
anchor_timezone: "Europe/Berlin"
llm_seed: 0
llm_temperature: 0
enable_coreference: false
enable_ner: false
enable_llm_assertion_extraction: true
# ... (see full config schema in docs: TBA)
```

## Key Features

- **Flexible Schema Mapping**: Adapts to different ChatGPT export formats via YAML config
- **Tree Structure Preservation**: Maintains conversation branching (regenerated responses, tangents)
- **Code Fence Detection**: Identifies and masks code blocks for accurate entity extraction
- **Deterministic IDs**: Reproducible UUIDv5 generation from canonical representations
- **Full Provenance**: Character-level spans reference source text

## Requirements

- Python 3.10+
- SQLite 3.35+
- Dependencies: see `requirements.txt`

## Output

SQLite database with:
- Full conversation history (losslessly preserved)
- Extracted entities and relationships
- Temporal evolution tracking
- Conflict detection and resolution
- Graph-ready materialization

## Project Status

✅ Stage 1: Raw ingestion complete  
🚧 Stage 2-5: In development  

## License

MIT

## Contributing

Issues and PRs welcome. Please maintain deterministic behavior and full test coverage for any changes.