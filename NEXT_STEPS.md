# Next Steps for lorch Development

This document tracks TODOs, roadmap, and implementation priorities for the lorch orchestrator.

## 🎯 Current Status

**Version:** 0.1.0
**Status:** Production-ready orchestrator with vault storage
**Last Updated:** 2025-11-12

### ✅ Completed

- [x] **Pipeline orchestration framework**
- [x] **Vault storage**: Time-series with LATEST pointers
- [x] **CLI interface**: run, extract, list, status, validate, clean
- [x] **Granular extraction**: `lorch extract <tap>` for individual sources
- [x] **Discovery commands**: `lorch list extractors/jobs/transforms/mappings`
- [x] **Extract stage**: Meltano integration with chunked targets (12 extractors)
- [x] **Canonize stage**: LATEST-only, deterministic, idempotent
- [x] **Per-account canonical output**: `canonical/{source}/{account}.jsonl`
- [x] **LATEST pointer system**: Failed runs don't update pointer (safety)
- [x] **Index stage**: Stub implementation
- [x] **Configuration management**: YAML-based with vault paths
- [x] **Structured logging**: JSON format, no PHI
- [x] **Error handling with retries**: Exponential backoff
- [x] **PHI security validation**: 700/600 permissions enforced
- [x] **Transform registry structure**
- [x] **Manifest tracking**: Records, checksums, compression stats
- [x] **Comprehensive documentation**

### ⚠️ Partially Complete

- [ ] **Transform library**: Only 1/4 transforms exist
  - ✅ Gmail → canonical email
  - ❌ Exchange → canonical email
  - ❌ Dataverse contact → canonical contact
  - ❌ Dataverse session → canonical clinical_session
- [ ] **Vector-projector integration**: Stub mode only (just copies files)
- [ ] **Testing infrastructure**: No unit or integration tests yet

---

## 🚀 Phase 1: Complete Transform Library (Priority: HIGH)

**Goal:** Create all necessary transforms to support the data pipeline.

### Task 1.1: Create Exchange → Canonical Email Transform

**Status:** 🔴 TODO
**Effort:** 2-4 hours
**Prerequisites:** Access to Exchange message schema

**Steps:**

1. **Examine Exchange message structure:**
   ```bash
   cd /home/user/meltano-ingest
   head -1 tests/fixtures/exchange_messages.jsonl | jq .
   ```

2. **Create JSONata transform:**
   ```bash
   cd /home/user/transforms/email
   # Copy and modify Gmail transform as starting point
   cp gmail_to_canonical_v1.jsonata exchange_to_canonical_v1.jsonata
   ```

3. **Map Exchange fields to canonical:**
   - `id` → `message_id`
   - `subject` → `subject`
   - `from.emailAddress.address` → `from`
   - `toRecipients[].emailAddress.address` → `to[]`
   - `receivedDateTime` → `received_at`
   - `body.content` → `body`

4. **Create metadata file:**
   ```bash
   cp gmail_to_canonical_v1.meta.yaml exchange_to_canonical_v1.meta.yaml
   # Update:
   #   - name: exchange_to_canonical
   #   - input_schema: iglu:com.microsoft/graph_email/jsonschema/1-0-0
   #   - checksum: (calculate new checksum)
   ```

5. **Test transform:**
   ```bash
   cd /home/user/canonizer
   source .venv/bin/activate
   cat /home/user/meltano-ingest/tests/fixtures/exchange_messages.jsonl | \
     can transform run \
     --meta /home/user/transforms/email/exchange_to_canonical_v1.meta.yaml
   ```

6. **Verify output matches canonical schema**

### Task 1.2: Create Dataverse Contact → Canonical Contact Transform

**Status:** 🔴 TODO
**Effort:** 2-3 hours
**Prerequisites:** Access to Dataverse contact schema

**Steps:**

1. **Examine Dataverse contact structure:**
   ```bash
   head -1 /home/user/phi-data/meltano-extracts/contacts.jsonl | jq .
   ```

2. **Create transform directory:**
   ```bash
   mkdir -p /home/user/transforms/contact
   ```

3. **Create JSONata transform:**
   ```bash
   cd /home/user/transforms/contact
   vi dataverse_contact_to_canonical_v1.jsonata
   ```

4. **Map Dataverse fields:**
   - `contactid` → `contact_id`
   - `firstname` → `first_name`
   - `lastname` → `last_name`
   - `emailaddress1` → `email`
   - `telephone1` → `phone`
   - `address1_*` → `address {}`
   - `birthdate` → `birth_date`
   - `createdon` → `created_at`

5. **Create metadata file with checksum**

6. **Test and verify**

### Task 1.3: Create Dataverse Session → Canonical Clinical Session Transform

**Status:** 🔴 TODO
**Effort:** 2-3 hours
**Prerequisites:** Access to Dataverse session schema

**Steps:**

1. **Examine Dataverse session structure**
2. **Create transform directory:**
   ```bash
   mkdir -p /home/user/transforms/clinical_session
   ```
3. **Create JSONata transform**
4. **Map Dataverse session fields to canonical**
5. **Create metadata file**
6. **Test and verify**

### Task 1.4: Define Canonical Schemas

**Status:** 🔴 TODO
**Effort:** 3-5 hours
**Prerequisites:** Understanding of all data types

**Create JSON Schema files for:**

```bash
/home/user/transforms/schemas/canonical/
├── email_v1-0-0.json
├── contact_v1-0-0.json
└── clinical_session_v1-0-0.json
```

**Schema should include:**
- Required vs optional fields
- Data types and formats
- Field descriptions
- Examples

---

## 🗄️ Phase 2: Implement Vector-Projector Integration (Priority: HIGH)

**Goal:** Replace stub with full SQLite indexing and inode storage.

### Task 2.1: Implement SQLite Indexing

**Status:** 🔴 TODO
**Effort:** 1-2 days
**Prerequisites:** SQLite 3.35+ (JSON support)

**Steps:**

1. **Design database schema:**
   ```sql
   CREATE TABLE objects (
     id TEXT PRIMARY KEY,
     type TEXT NOT NULL,
     file_path TEXT NOT NULL,
     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
     metadata JSON,
     UNIQUE(type, id)
   );

   CREATE INDEX idx_type ON objects(type);
   CREATE INDEX idx_created_at ON objects(created_at);
   ```

2. **Implement database module:**
   ```bash
   cd /home/user/vector-projector
   # Create vector_projector/db.py
   ```

3. **Add insert/query functions:**
   - `insert_object(obj_id, obj_type, file_path, metadata)`
   - `query_objects(obj_type, filters, limit)`
   - `get_object(obj_id)`

4. **Update index stage to use SQLite:**
   - Modify `/home/user/lorch/lorch/stages/index.py`
   - Replace file copy with SQLite insert
   - Store files in inode-style directory structure

### Task 2.2: Implement Inode-Style Storage

**Status:** 🔴 TODO
**Effort:** 4-8 hours

**Steps:**

1. **Implement hash-based directory structure:**
   ```python
   def get_storage_path(obj_id: str) -> Path:
       hash_prefix = obj_id[:2]
       hash_suffix = obj_id[2:4]
       return Path(f"objects/{hash_prefix}/{hash_suffix}/{obj_id}.json")
   ```

2. **Implement storage functions:**
   - `store_object(obj_id, data) -> Path`
   - `retrieve_object(obj_id) -> dict`
   - `list_objects(obj_type) -> List[Path]`

3. **Add registry files:**
   ```bash
   /home/user/phi-data/vector-store/
   ├── objects/           # Inode storage
   ├── registry/          # Per-type registries
   │   ├── email.json
   │   └── contact.json
   └── metadata.db        # SQLite index
   ```

### Task 2.3: Create Vector-Projector CLI

**Status:** 🔴 TODO
**Effort:** 4-6 hours

**Commands to implement:**

```bash
vector-projector pull --input /path/to/canonical/ --type email
vector-projector query --type email --limit 10
vector-projector get OBJECT_ID
vector-projector info
```

---

## 🧪 Phase 3: Add Testing Infrastructure (Priority: MEDIUM)

**Goal:** Achieve >80% test coverage.

### Task 3.1: Create Test Infrastructure

**Status:** 🔴 TODO
**Effort:** 1 day

**Steps:**

1. **Create test directory structure:**
   ```bash
   cd /home/user/lorch
   mkdir -p tests/{unit,integration,fixtures}
   ```

2. **Create pytest configuration:**
   ```bash
   # pyproject.toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_classes = ["Test*"]
   python_functions = ["test_*"]
   ```

3. **Create test fixtures:**
   ```bash
   # tests/fixtures/
   ├── pipeline_config.yaml
   ├── gmail_messages.jsonl
   ├── exchange_messages.jsonl
   └── dataverse_contacts.jsonl
   ```

### Task 3.2: Unit Tests

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Test modules to create:**

- `tests/unit/test_config.py` - Configuration loading and validation
- `tests/unit/test_utils.py` - Utility functions
- `tests/unit/test_stages_extract.py` - Extract stage
- `tests/unit/test_stages_canonize.py` - Canonize stage
- `tests/unit/test_stages_index.py` - Index stage
- `tests/unit/test_pipeline.py` - Pipeline orchestration

**Example test:**

```python
def test_config_loads_successfully():
    config = load_config(Path("tests/fixtures/pipeline_config.yaml"))
    assert config.name == "test-pipeline"
    assert len(config.stages) == 3

def test_extract_stage_validates():
    stage = ExtractStage(config, logger)
    stage.validate()  # Should not raise
```

### Task 3.3: Integration Tests

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Test scenarios:**

- `test_full_pipeline_with_test_data()` - End-to-end with fixtures
- `test_canonize_stage_with_real_transforms()` - Real canonizer integration
- `test_error_handling_and_retries()` - Failure scenarios
- `test_phi_permission_enforcement()` - Security validation

---

## 📊 Phase 4: Monitoring and Observability (Priority: MEDIUM)

**Goal:** Add metrics, alerting, and dashboard.

### Task 4.1: Metrics Collection

**Status:** 🔴 TODO
**Effort:** 1-2 days

**Metrics to collect:**

- Pipeline execution duration
- Records processed per stage
- Success/failure rates
- Error types and frequencies
- Stage-specific metrics

**Implementation:**

```python
# lorch/metrics.py
class MetricsCollector:
    def record_pipeline_run(self, result: PipelineResult):
        """Record metrics for pipeline run."""
        pass

    def record_stage_execution(self, stage_name: str, result: StageResult):
        """Record metrics for stage execution."""
        pass

    def export_metrics(self) -> Dict:
        """Export metrics in Prometheus format."""
        pass
```

### Task 4.2: Alerting

**Status:** 🔴 TODO
**Effort:** 1 day

**Alert conditions:**

- Pipeline failure
- Stage taking longer than threshold
- Zero records extracted
- Permission errors

**Implementation:**

- Email alerts (SMTP)
- Slack webhooks
- PagerDuty integration (optional)

### Task 4.3: Dashboard

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Dashboard features:**

- Pipeline run history
- Success/failure visualization
- Record count trends
- Error log viewer
- Stage duration charts

**Tech stack:**

- Streamlit or Dash (Python-based)
- SQLite for metrics storage
- Plotly for charts

---

## 🚀 Phase 5: Advanced Features (Priority: LOW)

### Task 5.1: Parallel Stage Execution

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Goal:** Run independent extractions in parallel.

**Implementation:**

```python
# config/pipeline.yaml
behavior:
  parallel_execution: true
  max_workers: 4

# lorch/pipeline.py
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(stage.run) for stage in stages]
    results = [f.result() for f in futures]
```

### Task 5.2: Incremental Extraction

**Status:** 🔴 TODO
**Effort:** 1-2 days

**Goal:** Only extract new/changed records.

**Features:**

- State tracking per source
- Watermark-based extraction
- Checkpoint and resume

### Task 5.3: Data Quality Checks

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Checks to implement:**

- Schema validation
- Null value detection
- Duplicate record detection
- Referential integrity checks
- Data freshness monitoring

---

## 🌐 Phase 6: Cloud Migration Preparation (Priority: LOW)

### Task 6.1: Containerization

**Status:** 🔴 TODO
**Effort:** 1-2 days

**Create Dockerfiles:**

```bash
# /home/user/lorch/Dockerfile
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN pip install .
CMD ["lorch", "run"]
```

### Task 6.2: Cloud Workflow Definition

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Create Cloud Workflows YAML:**

```yaml
# workflows/phi-pipeline.yaml
main:
  steps:
    - extract: ...
    - canonize: ...
    - index: ...
```

### Task 6.3: Terraform/IaC

**Status:** 🔴 TODO
**Effort:** 3-5 days

**Infrastructure as Code:**

- Cloud Run Jobs
- Cloud Functions
- BigQuery datasets
- GCS buckets
- IAM policies
- Secrets

---

## 📚 Phase 7: Documentation Enhancements

### Task 7.1: API Documentation

**Status:** 🔴 TODO
**Effort:** 1-2 days

**Generate API docs:**

```bash
# Using Sphinx
cd /home/user/lorch
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs/api
sphinx-apidoc -o docs/api lorch/
make html
```

### Task 7.2: Tutorial Videos/Guides

**Status:** 🔴 TODO
**Effort:** 2-3 days

**Create guides for:**

- Getting started tutorial
- Creating transforms guide
- Troubleshooting common issues
- Cloud migration walkthrough

### Task 7.3: Example Transforms

**Status:** 🔴 TODO
**Effort:** 1 day

**Create example transforms for:**

- Different email providers
- Various CRM systems
- Common data formats

---

## 🎯 Recommended Implementation Order

### Sprint 1 (Week 1)
1. **Task 1.1:** Create Exchange transform (Day 1-2)
2. **Task 1.2:** Create Dataverse contact transform (Day 2-3)
3. **Task 1.3:** Create Dataverse session transform (Day 3-4)
4. **Task 1.4:** Define canonical schemas (Day 4-5)

### Sprint 2 (Week 2)
5. **Task 3.1:** Create test infrastructure (Day 1)
6. **Task 3.2:** Write unit tests (Day 2-4)
7. **Task 3.3:** Write integration tests (Day 4-5)

### Sprint 3 (Week 3)
8. **Task 2.1:** Implement SQLite indexing (Day 1-2)
9. **Task 2.2:** Implement inode storage (Day 3-4)
10. **Task 2.3:** Create vector-projector CLI (Day 4-5)

### Sprint 4 (Week 4)
11. **Task 4.1:** Metrics collection (Day 1-2)
12. **Task 4.2:** Alerting (Day 3)
13. **Task 4.3:** Dashboard (Day 3-5)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for:

- Development setup
- Code style guidelines
- Pull request process
- Testing requirements

---

## 📞 Questions?

For questions about this roadmap:

1. Check [docs/architecture.md](docs/architecture.md) for design decisions
2. Review [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
3. See [README.md](README.md) for current status

---

**Last Updated:** 2025-11-12
**Maintained By:** lorch development team
