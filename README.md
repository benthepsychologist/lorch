# lorch - Local Orchestrator

**Local Orchestrator** for the PHI data pipeline. Coordinates three-stage data processing from extraction to indexed storage.

## 🎯 Overview

lorch orchestrates the local-first data pipeline with time-series vault storage:

```
┌─────────────┐      ┌────────────┐      ┌──────────────────┐
│   meltano   │  →   │ canonizer  │  →   │ vector-projector │
│  (extract)  │      │(transform) │      │    (index)       │
└─────────────┘      └────────────┘      └──────────────────┘
 Chunked Vault    Canonical/Account      SQLite + Files
```

### Pipeline Stages

1. **Extract** - Pulls data from 12 sources (Gmail, Exchange, Dataverse, Sheets, QuickBooks)
   - Writes to time-series vault with LATEST pointer
   - Chunked, compressed, auditable
2. **Canonize** - Transforms LATEST vault runs to canonical schemas via JSONata
   - Deterministic: same LATEST → same output
   - Idempotent: clears and rebuilds per account
3. **Index** - Stores canonical data in local document store with SQLite indexing

### Key Features

- ✅ **Vault Storage**: Time-series extraction with LATEST pointers
- ✅ **Deterministic**: Canonization always produces same output from LATEST
- ✅ **Idempotent**: Safe to rerun - clears and rebuilds
- ✅ **Auditable**: Historical runs preserved for audit trail
- ✅ **Security-First**: PHI data protection with 700/600 permissions
- ✅ **Granular Control**: Extract individual accounts or run batch jobs
- ✅ **Error Handling**: Retry logic, failed runs don't update LATEST
- ✅ **Structured Logging**: JSON logs with no PHI data
- ✅ **Cloud-Ready**: Design enables easy cloud migration

## 🚀 Quick Start

### Installation

```bash
cd /home/user/lorch

# Create virtual environment
uv venv
source .venv/bin/activate

# Install lorch
uv pip install -e .

# Verify installation
lorch --version
```

### Basic Usage

```bash
# List available extractors
lorch list extractors

# Extract from individual source
lorch extract tap-gmail--acct1-personal

# List available transforms
lorch list transforms

# List configured mappings
lorch list mappings

# Run canonization (processes LATEST runs only)
lorch run --stage canonize

# Validate configuration
lorch validate

# Check status
lorch status

# Clean outputs
lorch clean --stage canonize
```

### First Run

```bash
# 1. Validate setup
lorch validate

# 2. List what's available
lorch list extractors

# 3. Extract from a single source
lorch extract tap-gmail--acct1-personal

# 4. Run canonization (LATEST-only, idempotent)
lorch run --stage canonize

# 5. Check logs
tail -f logs/pipeline-*.log
```

## 📖 Documentation

Comprehensive documentation in the `docs/` directory:

- **[docs/architecture.md](docs/architecture.md)** - System architecture and design
- **[docs/configuration.md](docs/configuration.md)** - Configuration reference
- **[docs/cloud-migration.md](docs/cloud-migration.md)** - Cloud migration guide
- **[NEXT_STEPS.md](NEXT_STEPS.md)** - TODOs and implementation roadmap
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guide

## 🔧 Configuration

Default configuration: `config/pipeline.yaml`

### Key Configuration Sections

```yaml
pipeline:
  name: phi-local-pipeline
  version: 1.0.0

stages:
  extract:
    type: meltano
    repo_path: /home/user/meltano-ingest
    job: ingest-all-accounts
    output_dir: /home/user/phi-data/meltano-extracts

  canonize:
    type: canonizer
    repo_path: /home/user/canonizer
    transform_registry: /home/user/transforms
    mappings:
      - source_pattern: "*gmail*.jsonl"
        transform: email/gmail_to_canonical_v1
```

See [docs/configuration.md](docs/configuration.md) for complete reference.

## 🗄️ Vault Structure

lorch uses a time-series vault with LATEST pointers for deterministic, idempotent processing:

```
vault/{domain}/{source}/{account}/
├── dt=2025-11-12/run_id=20251112T205433Z/
│   ├── manifest.json               # Run metadata, checksums
│   ├── part-000.jsonl.gz          # Compressed data chunks
│   └── part-001.jsonl.gz
├── dt=2025-11-13/run_id=20251113T103045Z/  # Next day's run
│   ├── manifest.json
│   └── part-000.jsonl.gz
└── LATEST.json  ← Points to most recent successful run
```

**LATEST.json** format:
```json
{
  "dt": "2025-11-13",
  "run_id": "20251113T103045Z",
  "updated_at": "2025-11-13T10:31:05.123456+00:00",
  "records": 295
}
```

**Key Properties:**
- **Time-Series**: All historical runs preserved
- **LATEST Pointer**: Determines which run to canonize
- **Deterministic**: Same LATEST → same canonical output
- **Idempotent**: Rerun canonize safely, clears and rebuilds
- **Safe**: Failed runs don't update LATEST pointer

**Workflow:**
1. `lorch extract` → creates new `dt=/run_id=` directory
2. On success → updates `LATEST.json` to point to new run
3. `lorch run --stage canonize` → processes LATEST run only
4. Historical runs kept for audit, but ignored by canonize

## 📊 CLI Commands

### `lorch extract`

Run a single extractor (auto-selects chunked target).

```bash
lorch extract TAP_NAME [OPTIONS]

Options:
  --target TEXT     Override target (default: auto-select)
  --config PATH     Custom configuration file
  --verbose         Enable debug logging
```

**Examples:**

```bash
# Extract from single Gmail account
lorch extract tap-gmail--acct1-personal

# Extract from Exchange
lorch extract tap-msgraph-mail--ben-mensio

# Extract from all Gmail accounts (run 3 times)
lorch extract tap-gmail--acct1-personal
lorch extract tap-gmail--acct2-business1
lorch extract tap-gmail--acct3-bfarmstrong

# Override target
lorch extract tap-dataverse --target target-jsonl
```

**Output:**
- Creates: `vault/{domain}/{source}/{account}/dt=YYYY-MM-DD/run_id=TIMESTAMP/`
- Updates: `LATEST.json` on success
- Shows: Manifest summary with records, size, parts

### `lorch list`

Discover available extractors, transforms, jobs, and mappings.

```bash
lorch list COMMAND

Commands:
  extractors    List all configured meltano taps
  jobs          List all configured meltano jobs
  transforms    List available transforms
  mappings      List source → transform mappings
```

**Examples:**

```bash
# List all extractors
lorch list extractors

# List meltano jobs
lorch list jobs

# List available transforms
lorch list transforms

# List configured mappings
lorch list mappings
```

### `lorch run`

Execute pipeline stages.

```bash
lorch run [OPTIONS]

Options:
  --stage TEXT       Run specific stage (extract|canonize|index)
  --config PATH      Custom configuration file
  --dry-run          Validate without executing
  --verbose          Enable debug logging
```

**Examples:**

```bash
# Run full pipeline
lorch run

# Run single stage
lorch run --stage extract

# Dry run with validation
lorch run --dry-run

# Verbose logging
lorch run --verbose

# Custom config
lorch run --config /path/to/pipeline.yaml
```

### `lorch status`

Show pipeline state and last run information.

```bash
lorch status
```

**Output:**

```
Pipeline Status: phi-local-pipeline v1.0.0

Last Run: 2025-11-12 10:30:45
Status: ✅ SUCCESS
Duration: 4m 32s

Stages:
  ✅ extract      1m 45s    27 records
  ✅ canonize     2m 12s    27 records
  ✅ index          35s    27 records

Logs: logs/pipeline-2025-11-12.log
```

### `lorch validate`

Validate configuration and dependencies.

```bash
lorch validate [OPTIONS]

Options:
  --skip-permissions    Skip PHI directory permission checks
```

**Checks:**
- Configuration file syntax
- Component paths exist
- Transform registry accessible
- Output directories writable
- PHI directory permissions (700)

### `lorch clean`

Clean stage outputs.

```bash
lorch clean [OPTIONS]

Options:
  --stage TEXT    Clean specific stage output
  --all           Clean all stage outputs
  --dry-run       Show what would be deleted
```

**Examples:**

```bash
# Clean specific stage
lorch clean --stage canonize

# Clean all (prompts for confirmation)
lorch clean --all

# Dry run
lorch clean --all --dry-run
```

## 🔐 Security

### PHI Data Protection

All PHI data is stored with restrictive permissions:

```bash
# Required permissions (enforced by lorch)
chmod 700 /home/user/phi-data
chmod 700 /home/user/phi-data/meltano-extracts
chmod 700 /home/user/phi-data/canonical
chmod 700 /home/user/phi-data/vector-store
```

lorch validates these permissions before each run.

### Logging

Logs contain **no PHI data** - only:
- Timestamps and durations
- Record counts
- File paths
- Error messages (sanitized)

Logs are written to `logs/pipeline-YYYY-MM-DD.log` with structured JSON format.

## 🎯 Current Status

### ✅ Implemented (v0.1.0)

- [x] **Pipeline orchestration framework**
- [x] **Vault storage**: Time-series with LATEST pointers
- [x] **Extract stage**: Meltano integration with chunked targets
- [x] **Granular extraction**: `lorch extract <tap>` for individual sources
- [x] **Discovery commands**: `lorch list extractors/jobs/transforms/mappings`
- [x] **Canonize stage**: LATEST-only, deterministic, idempotent
- [x] **Per-account canonical output**: `canonical/{source}/{account}.jsonl`
- [x] **Index stage**: Stub (file copy)
- [x] **CLI interface**: run, extract, list, status, validate, clean
- [x] **Structured logging**: JSON logs with no PHI
- [x] **Error handling**: Retries, failed runs don't update LATEST
- [x] **Configuration management**: YAML-based
- [x] **PHI security validation**: 700/600 permissions enforced
- [x] **Transform registry**: `/home/user/transforms/`
- [x] **Manifest tracking**: Records, checksums, compression stats

### ⚠️ Partially Implemented

- [ ] **Transform library**: Only 1/4 exist
  - ✅ Gmail → canonical email
  - ❌ Exchange → canonical email
  - ❌ Dataverse contact → canonical contact
  - ❌ Dataverse session → canonical clinical_session
- [ ] **Vector-projector integration**: Stub mode (just copies files)
- [ ] **Tests**: No unit or integration tests yet

### 🚧 TODO

See [NEXT_STEPS.md](NEXT_STEPS.md) for complete roadmap.

**Immediate next steps:**

1. **Create missing transforms** (3 remaining):
   - Exchange → canonical email
   - Dataverse contact → canonical contact
   - Dataverse session → canonical clinical_session

2. **Implement full vector-projector**:
   - SQLite indexing by message_id/entity_id
   - Inode-style file storage
   - Query API

3. **Add tests**:
   - Unit tests for stages
   - Integration tests for full pipeline
   - Vault LATEST pointer logic tests

4. **Add monitoring**:
   - Metrics collection (records/run, duration, errors)
   - Alerting on failures
   - Dashboard (optional)

## 🛠️ Development

### Project Structure

```
lorch/
├── lorch/                      # Main package
│   ├── __init__.py            # Package entry point
│   ├── cli.py                 # CLI interface
│   ├── pipeline.py            # Orchestrator
│   ├── config.py              # Configuration
│   ├── utils.py               # Utilities
│   └── stages/                # Pipeline stages
│       ├── base.py            # Base classes
│       ├── extract.py         # Meltano stage
│       ├── canonize.py        # Canonizer stage
│       └── index.py           # Vector-projector stage
├── config/
│   └── pipeline.yaml          # Configuration
├── docs/                       # Documentation
├── logs/                       # Runtime logs
├── tests/                      # Tests (to be added)
└── README.md                   # This file
```

### Running Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=lorch tests/
```

### Code Style

```bash
# Format code
black lorch/

# Lint code
ruff check lorch/
```

## 🐛 Troubleshooting

### Pipeline Fails at Extract Stage

**Issue:** Meltano job fails

**Solution:**
1. Check Meltano authentication:
   ```bash
   cd /home/user/meltano-ingest
   source .venv/bin/activate
   meltano invoke tap-gmail--acct1-personal
   ```
2. Verify `.env` file has valid credentials
3. Check logs: `tail -f /home/user/meltano-ingest/.meltano/logs/meltano.log`

### Canonization Errors

**Issue:** Transform fails or produces invalid output

**Solution:**
1. Check transform exists:
   ```bash
   ls -la /home/user/transforms/email/gmail_to_canonical_v1.meta.yaml
   ```
2. Validate transform manually:
   ```bash
   cd /home/user/canonizer
   source .venv/bin/activate
   cat test.json | can transform run \
     --meta /home/user/transforms/email/gmail_to_canonical_v1.meta.yaml
   ```
3. Check canonizer logs for detailed errors

### Permission Denied Errors

**Issue:** Cannot write to PHI directories

**Solution:**
```bash
chmod 700 /home/user/phi-data
chmod 700 /home/user/phi-data/meltano-extracts
chmod 700 /home/user/phi-data/canonical
chmod 700 /home/user/phi-data/vector-store
```

### Missing Transform Files

**Issue:** "Transform metadata not found"

**Solution:**

See [docs/transforms.md](docs/transforms.md) for guide on creating transforms.

Quick fix:
```bash
# Create missing transform
cp /home/user/transforms/email/gmail_to_canonical_v1.jsonata \
   /home/user/transforms/email/exchange_to_canonical_v1.jsonata

# Update metadata
cp /home/user/transforms/email/gmail_to_canonical_v1.meta.yaml \
   /home/user/transforms/email/exchange_to_canonical_v1.meta.yaml

# Edit to match Exchange schema
```

## 🌐 Cloud Migration

This pipeline is designed for easy cloud migration. See [docs/cloud-migration.md](docs/cloud-migration.md) for:

- Component mapping (local → cloud services)
- Deployment steps
- Cost estimates
- Security configurations

**Local → Cloud:**
- `meltano` → Cloud Run Job
- `canonizer` → Cloud Function
- `vector-projector` → BigQuery
- `lorch` → Cloud Workflows
- Files → GCS buckets

## 📚 Related Repositories

- **[meltano-ingest](../meltano-ingest/)** - Data extraction with Meltano
- **[canonizer](../canonizer/)** - JSON transformation engine
- **[vector-projector](../vector-projector/)** - Local document store
- **[transforms](../transforms/)** - Transform registry

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

### Quick Contribution Guide

1. Create feature branch
2. Implement changes with tests
3. Run `black` and `ruff` for formatting
4. Update documentation
5. Submit pull request

### Areas Needing Help

- [ ] Create missing transforms (Exchange, Dataverse)
- [ ] Implement vector-projector integration
- [ ] Add unit tests
- [ ] Create integration tests
- [ ] Add monitoring/metrics
- [ ] Write migration scripts for cloud

## 📝 License

Proprietary - For internal use only. Contains PHI data handling logic.

## 📞 Support

For issues or questions:

- **Documentation**: See `docs/` directory
- **Configuration**: [docs/configuration.md](docs/configuration.md)
- **Architecture**: [docs/architecture.md](docs/architecture.md)
- **Next Steps**: [NEXT_STEPS.md](NEXT_STEPS.md)
- **Cloud Migration**: [docs/cloud-migration.md](docs/cloud-migration.md)

## 🎓 For LLM Assistants

If you're an LLM helping to develop this project:

1. **Read these files first:**
   - [NEXT_STEPS.md](NEXT_STEPS.md) - What needs to be done
   - [docs/architecture.md](docs/architecture.md) - How it works
   - [CONTRIBUTING.md](CONTRIBUTING.md) - Development guidelines

2. **Current priorities:**
   - Create missing transforms in `/home/user/transforms/`
   - Implement vector-projector integration
   - Add comprehensive tests

3. **Code style:**
   - Use Python 3.12+ features
   - Type hints for all functions
   - Docstrings in Google format
   - Black formatting, Ruff linting

4. **Security:**
   - Never log PHI data
   - Validate permissions before file operations
   - Sanitize error messages

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

**Version:** 0.1.0
**Status:** Production-ready orchestrator, missing some transforms
**Last Updated:** 2025-11-12
