# What's Next for lorch

## Current Status (v0.1.0)

**Extraction Pipeline: ✅ COMPLETE**

lorch successfully handles data ingestion with:
- Vault storage with time-series + LATEST pointers
- Tool adapter pattern for meltano integration
- Time-based extraction CLI (`--last 7d`, `--since`, `--from/--to`)
- Granular extraction per account
- Manifest tracking with checksums
- Deterministic, idempotent canonization from vault
- 12 configured extractors (Gmail, Exchange, Dataverse, Sheets, QuickBooks, etc.)

**Status: We are ingestion-only right now**

---

## Phase 1: Complete Transform Library (NEXT)

**Priority: HIGH**
**Goal:** Transform raw extracted data to canonical schemas

### Current State
- ✅ Gmail → canonical email (EXISTS)
- ❌ Exchange → canonical email (MISSING)
- ❌ Dataverse contact → canonical contact (MISSING)
- ❌ Dataverse session → canonical clinical_session (MISSING)

### Work Required

#### 1.1 Exchange → Canonical Email Transform
**Effort:** 2-4 hours

```bash
# Examine Exchange message structure
head -1 vault/email/exchange/*/dt=*/run_id=*/part-000.jsonl.gz | zcat | jq .

# Create transform
cd /home/user/transforms/email
cp gmail_to_canonical_v1.jsonata exchange_to_canonical_v1.jsonata
# Edit to map Exchange fields → canonical email schema

# Create metadata
cp gmail_to_canonical_v1.meta.yaml exchange_to_canonical_v1.meta.yaml
# Update name, input_schema, checksum

# Test
cat vault/email/exchange/*/dt=*/run_id=*/part-000.jsonl.gz | \
  zcat | \
  can transform run --meta /home/user/transforms/email/exchange_to_canonical_v1.meta.yaml
```

**Deliverable:** Working Exchange transform that maps to same canonical email schema as Gmail

#### 1.2 Dataverse Contact → Canonical Contact Transform
**Effort:** 2-3 hours

```bash
# Examine Dataverse contact structure
head -1 vault/crm/dataverse/*/dt=*/run_id=*/part-000.jsonl.gz | zcat | jq .

# Create transform directory
mkdir -p /home/user/transforms/contact

# Create JSONata transform
vi /home/user/transforms/contact/dataverse_contact_to_canonical_v1.jsonata

# Map fields:
#   contactid → contact_id
#   firstname → first_name
#   lastname → last_name
#   emailaddress1 → email
#   telephone1 → phone
#   address1_* → address {}
#   birthdate → birth_date
#   createdon → created_at

# Create metadata and test
```

**Deliverable:** Working Dataverse contact transform

#### 1.3 Dataverse Session → Canonical Clinical Session Transform
**Effort:** 2-3 hours

Similar process to 1.2, but for clinical session data.

**Deliverable:** Working Dataverse session transform

#### 1.4 Define Canonical Schemas (JSON Schema)
**Effort:** 3-5 hours

```bash
mkdir -p /home/user/transforms/schemas/canonical/
# Create:
#   - email_v1-0-0.json
#   - contact_v1-0-0.json
#   - clinical_session_v1-0-0.json
```

Each schema should include:
- Required vs optional fields
- Data types and formats
- Field descriptions
- Examples

**Deliverable:** JSON Schema files for all 3 canonical types

---

## Phase 2: Implement Vector-Projector Integration

**Priority: HIGH**
**Goal:** Replace stub with full SQLite indexing and object storage

### Current State
- ✅ Index stage exists (stub mode - just copies files)
- ❌ SQLite indexing (MISSING)
- ❌ Inode-style storage (MISSING)
- ❌ Query API (MISSING)

### Work Required

#### 2.1 SQLite Indexing
**Effort:** 1-2 days

Design schema:
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

Implement:
- `vector_projector/db.py` with insert/query functions
- Update `lorch/stages/index.py` to use SQLite
- Store files in inode-style directory structure

#### 2.2 Inode-Style Storage
**Effort:** 4-8 hours

```python
def get_storage_path(obj_id: str) -> Path:
    hash_prefix = obj_id[:2]
    hash_suffix = obj_id[2:4]
    return Path(f"objects/{hash_prefix}/{hash_suffix}/{obj_id}.json")
```

Directory structure:
```
/home/user/phi-data/vector-store/
├── objects/           # Inode storage
│   ├── ab/
│   │   ├── cd/
│   │   │   └── abcd1234.json
├── registry/          # Per-type registries
│   ├── email.json
│   └── contact.json
└── metadata.db        # SQLite index
```

#### 2.3 Vector-Projector CLI
**Effort:** 4-6 hours

```bash
vector-projector pull --input /path/to/canonical/ --type email
vector-projector query --type email --limit 10
vector-projector get OBJECT_ID
vector-projector info
```

---

## Phase 3: Add Testing Infrastructure

**Priority: MEDIUM**
**Goal:** Achieve >80% test coverage

### Work Required

#### 3.1 Test Infrastructure Setup
**Effort:** 1 day

```bash
mkdir -p tests/{unit,integration,fixtures}
# Create pytest configuration
# Create test fixtures
```

#### 3.2 Unit Tests
**Effort:** 2-3 days

Test modules:
- `tests/unit/test_config.py`
- `tests/unit/test_utils.py`
- `tests/unit/test_stages_extract.py`
- `tests/unit/test_stages_canonize.py`
- `tests/unit/test_stages_index.py`
- `tests/unit/test_pipeline.py`
- `tests/unit/test_vault_pointers.py` ← Important!

#### 3.3 Integration Tests
**Effort:** 2-3 days

Test scenarios:
- End-to-end pipeline with fixtures
- Vault LATEST pointer logic
- Canonization determinism
- Error handling and retries
- PHI permission enforcement

---

## Recommended Order

### Sprint 1 (Week 1) - Transforms
1. **Day 1-2:** Exchange → canonical email
2. **Day 2-3:** Dataverse contact → canonical contact
3. **Day 3-4:** Dataverse session → canonical clinical_session
4. **Day 4-5:** Define canonical JSON schemas

### Sprint 2 (Week 2) - Vector-Projector
5. **Day 1-2:** SQLite indexing
6. **Day 3-4:** Inode-style storage
7. **Day 4-5:** Vector-projector CLI

### Sprint 3 (Week 3) - Testing
8. **Day 1:** Test infrastructure setup
9. **Day 2-4:** Unit tests
10. **Day 4-5:** Integration tests

---

## Success Metrics

**Phase 1 Complete When:**
- [ ] All 3 transforms exist and tested
- [ ] Canonical schemas documented
- [ ] `lorch run --stage canonize` produces all 3 canonical types

**Phase 2 Complete When:**
- [ ] SQLite database indexes all canonical objects
- [ ] Inode storage working with 700/600 permissions
- [ ] `vector-projector query` returns results

**Phase 3 Complete When:**
- [ ] >80% code coverage
- [ ] All tests passing
- [ ] CI/CD pipeline configured

---

## Notes

- **We are ingestion-only right now** - vault extraction is complete
- **Next blocker:** Need transforms to make data useful
- **After transforms:** Vector-projector makes data queryable
- **Then testing:** Ensure everything is solid

See [NEXT_STEPS.md](../NEXT_STEPS.md) for full roadmap with additional phases (monitoring, cloud migration, etc.)

---

**Last Updated:** 2025-11-14
