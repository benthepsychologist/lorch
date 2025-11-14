---
version: "0.1"
tier: C
title: Tool Adapter Pattern + Fix Gmail Extraction
owner: benthepsychologist
goal: Implement tool adapter pattern for lorch orchestration and fix Gmail extraction issues
labels: [architecture, refactor, bugfix]
project_slug: lorch
spec_version: 1.0.0
created: 2025-11-13T12:28:01.751123+00:00
updated: 2025-11-13T12:30:00.000000+00:00
orchestrator_contract: "standard"
repo:
  working_branch: "feat/tool-adapter-pattern"
---

# Tool Adapter Pattern + Fix Gmail Extraction

## Objective

> Implement tool adapter pattern for orchestration and fix Gmail extraction bugs

Lorch needs a clean separation between orchestration logic and tool-specific execution. Instead of a complex plugin registry, implement lightweight tool adapters that wrap external tools (meltano, canonizer, vector-projector, GCS, etc.) with validation and testing logic.

Also fix current Gmail extraction issues:
- Query filter not working (extracting 1+ year instead of 1 day)
- Base64 image bloat (131MB files)
- Target name mismatches

## Acceptance Criteria

- [ ] Tool adapter pattern documented and implemented
- [ ] `lorch config show meltano` command displays meltano configuration
- [ ] `lorch config sync meltano` syncs from meltano.yml to lorch cache
- [ ] `lorch tools validate meltano` validates tap-target pairs before execution
- [ ] Gmail extraction works correctly (1-day filter, text-only, ~1-5MB files)
- [ ] Extract stage uses meltano adapter with validation
- [ ] No protected paths modified
- [ ] All tests pass

## Context

### Background

**Problem:** Lorch currently calls meltano directly without validation, leading to bugs like:
1. Query filter (`messages.q: after:2025/11/12`) ignored because `message_list` stream was excluded
2. Extracting full base64-encoded images/attachments (131MB for what should be 5MB)
3. Wrong target name selected by auto-selector
4. No validation before running expensive extractions

**Current Architecture Issues:**
- No separation between orchestration (lorch) and execution (meltano)
- Config lives only in meltano.yml (lorch can't validate without running meltano)
- Testing logic scattered across stages
- Hard to add new tools (canonizer, vector-projector, GCS) without duplicating patterns

**Why Tool Adapters Over Plugin Registry:**

Initially considered a plugin registry system with `config/registry/`, `plugins/extractors/`, etc.
**Rejected because:**
- Too complex for 3-4 tools (meltano, canonizer, vector-projector)
- Future tools (GCS buckets, event pipelines) don't fit "plugin" model
- Over-abstraction: treating everything as extractors/loaders is wrong
- Premature generalization

**Tool Adapter Pattern is better because:**
- Simpler: Just wrapper classes, no registry abstraction
- Flexible: Each tool can have completely different API (CLI, library, cloud API, streaming)
- Right abstraction: "Tools" = anything lorch orchestrates
- Solves actual problems: validation, testing, config sync
- Grows naturally: Add adapters as needed for new tools

### Architecture: Tool Adapter Pattern

```
lorch/
├── config/
│   ├── pipeline.yaml              # Orchestration config (current)
│   └── tools/                     # NEW: Per-tool cached config
│       ├── meltano.yaml          # Synced from meltano.yml
│       ├── canonizer.yaml        # Canonizer config (future)
│       └── vector_projector.yaml # Vector-proj config (future)
├── lorch/
│   ├── tools/                     # NEW: Tool adapters
│   │   ├── __init__.py
│   │   ├── base.py               # Base adapter interface
│   │   ├── meltano.py            # Meltano adapter
│   │   ├── canonizer.py          # Canonizer adapter (future)
│   │   └── vector_projector.py   # Vector-proj adapter (future)
│   ├── stages/
│   │   ├── extract.py            # UPDATED: Use meltano adapter
│   │   ├── canonize.py           # UPDATED: Use canonizer adapter (future)
│   │   └── index.py              # UPDATED: Use vector-proj adapter (future)
│   └── cli.py                     # UPDATED: Add config/tools commands
└── tests/
    └── tools/                     # NEW: Adapter tests
        └── test_meltano.py
```

**Tool Adapter Interface (Base Class):**

```python
# lorch/tools/base.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

class ToolAdapter(ABC):
    """Base class for tool adapters."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self.load_config()

    @abstractmethod
    def load_config(self) -> Dict[str, Any]:
        """Load tool configuration."""
        pass

    @abstractmethod
    def sync_config(self) -> None:
        """Sync config from tool's native source."""
        pass

    @abstractmethod
    def validate(self) -> Dict[str, Any]:
        """Validate tool setup. Returns dict with 'valid': bool, 'errors': list."""
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute tool command."""
        pass
```

**Meltano Adapter Specification:**

```python
# lorch/tools/meltano.py
class MeltanoAdapter(ToolAdapter):
    """Adapter for meltano extract tool."""

    def __init__(self, meltano_dir: Path, config_cache: Path):
        self.meltano_dir = meltano_dir  # /home/user/meltano-ingest
        self.config_cache = config_cache  # config/tools/meltano.yaml
        super().__init__(config_cache)

    def sync_config(self) -> None:
        """Sync from meltano.yml to config/tools/meltano.yaml"""
        # Parse /home/user/meltano-ingest/meltano.yml
        # Extract: extractors, loaders, jobs
        # Write to config/tools/meltano.yaml

    def validate_task(self, tap: str, target: str) -> Dict[str, Any]:
        """Validate a tap-target pair before execution"""
        errors = []

        # Check tap exists
        if tap not in self.config['extractors']:
            errors.append(f"Tap '{tap}' not found")

        # Check target exists
        if target not in self.config['loaders']:
            errors.append(f"Target '{target}' not found")

        # Check specific validations (Gmail example)
        if 'gmail' in tap:
            tap_config = self.config['extractors'][tap]

            # If messages.q filter is set, message_list must be enabled
            if tap_config.get('config', {}).get('messages.q'):
                select = tap_config.get('select', [])
                if '!message_list.*.*' in select:
                    errors.append(
                        f"Tap '{tap}' has messages.q filter but message_list stream is excluded. "
                        "Filter will not work. Either remove messages.q or enable message_list stream."
                    )

        return {'valid': len(errors) == 0, 'errors': errors}

    def run_task(self, tap: str, target: str, **kwargs) -> subprocess.CompletedProcess:
        """Execute meltano run <tap> <target>"""
        # Validate first
        validation = self.validate_task(tap, target)
        if not validation['valid']:
            raise ValueError(f"Task validation failed: {validation['errors']}")

        # Build command
        cmd = [
            str(self.meltano_dir / '.venv' / 'bin' / 'meltano'),
            'run',
            tap,
            target
        ]

        # Execute
        return subprocess.run(cmd, cwd=self.meltano_dir, **kwargs)
```

### New Commands

```bash
# Display tool configuration (from cached config)
lorch config show meltano
lorch config show canonizer

# Sync tool configuration from source
lorch config sync meltano     # FROM meltano.yml TO config/tools/meltano.yaml

# List available tools
lorch tools list

# Validate tool configuration
lorch tools validate meltano
lorch tools validate canonizer
```

### Constraints

- No edits to meltano-ingest repository (only read meltano.yml) - document required changes instead
- Config sync is one-way: meltano.yml → lorch cache (for now)
- Keep it simple: Start with meltano adapter only
- No breaking changes to existing `lorch extract` command

### Gmail Extraction Issues to Document

**Issue 1: Query filter not working**
- Root cause: `message_list` stream excluded but that's where `messages.q` filter applies
- Required fix: Re-enable `message_list.*.*` in tap-gmail--acct1-personal select
- Location: `/home/user/meltano-ingest/meltano.yml` lines 82-89 (will be documented, not edited)

**Issue 2: Base64 bloat**
- Root cause: `format=full` returns entire email with inline images as base64
- Required fix: Exclude `messages.raw` field to remove base64-encoded full email
- Add to select: `!messages.raw` (will be documented, not edited)

**Issue 3: Wrong target name**
- Root cause: lorch CLI auto-selector uses wrong target names
- Fix: Update `_select_chunked_target()` in `/home/user/lorch/lorch/cli.py` (this file CAN be edited)
- Change: `target-jsonl-chunked--acct1-ben-mensio` → `target-jsonl-chunked--gmail-ben-mensio`

## Plan

### Step 1: Create Tool Adapter Base Class [G0: Design Approval]

**Prompt:**

Create the base tool adapter interface in `lorch/tools/base.py`.

Implement:
- `ToolAdapter` abstract base class
- Methods: `load_config()`, `sync_config()`, `validate()`, `execute()`
- Type hints and docstrings
- Simple error handling

**Outputs:**

- `lorch/tools/__init__.py`
- `lorch/tools/base.py`

---

### Step 2: Implement Meltano Adapter [G0: Design Approval]

**Prompt:**

Create `lorch/tools/meltano.py` with `MeltanoAdapter` class.

Implement:
1. `sync_config()` - Parse meltano.yml, extract extractors/loaders/jobs, write to config/tools/meltano.yaml
2. `validate_task(tap, target)` - Check tap/target exist, validate Gmail-specific rules
3. `run_task(tap, target)` - Execute meltano run with validation
4. `load_config()` - Load from config/tools/meltano.yaml

**Outputs:**

- `lorch/tools/meltano.py`
- `config/tools/` directory created

---

### Step 3: Add CLI Commands [G1: Code Readiness]

**Prompt:**

Update `lorch/cli.py` to add new commands:
- `lorch config show <tool>` - Display tool configuration
- `lorch config sync <tool>` - Sync tool config from source
- `lorch tools list` - List available adapters
- `lorch tools validate <tool>` - Validate tool configuration

Commands should:
- Use click decorators
- Handle errors gracefully
- Provide clear output

**Commands:**

```bash
ruff check lorch/
```

**Outputs:**

- `lorch/cli.py` (updated)

---

### Step 4: Document Required Gmail Config Changes [G1: Code Readiness]

**Prompt:**

Document the required changes to tap-gmail--acct1-personal configuration.

Create `docs/gmail-config-fixes.md` documenting what needs to be changed in `/home/user/meltano-ingest/meltano.yml`:

1. Re-enable message_list stream (remove `!message_list.*.*` exclusion)
2. Add `!messages.raw` exclusion to reduce base64 bloat
3. Verify `messages.q: after:2025/11/12` is set

Include example YAML showing the updated select configuration:
```yaml
select:
  - message_list.*.*     # Needed for query filtering
  - messages.*.*         # Message content
  - '!messages.raw'      # Exclude base64-encoded full email
```

**Outputs:**

- `docs/gmail-config-fixes.md` (new)

---

### Step 5: Fix Lorch Target Auto-Selector [G1: Code Readiness]

**Prompt:**

Fix `_select_chunked_target()` function in `/home/user/lorch/lorch/cli.py` (lines ~434-475).

Update target name mappings:
- Line ~438: `target-jsonl-chunked--acct1-ben-mensio` → `target-jsonl-chunked--gmail-ben-mensio`
- Line ~442: `target-jsonl-chunked--acct2-ben-mensio` → `target-jsonl-chunked--gmail-drben`
- Line ~446: `target-jsonl-chunked--acct3-ben-mensio` → `target-jsonl-chunked--gmail-ben-personal`

**Outputs:**

- `lorch/cli.py` (updated)

---

### Step 6: Update Extract Stage to Use Adapter [G1: Code Readiness]

**Prompt:**

Update `lorch/stages/extract.py` to use MeltanoAdapter:

1. Import MeltanoAdapter
2. Before running extraction, validate task with adapter
3. Use adapter.run_task() instead of direct subprocess call
4. Log validation results

**Commands:**

```bash
ruff check lorch/stages/
pytest tests/stages/test_extract.py -v
```

**Outputs:**

- `lorch/stages/extract.py` (updated)

---

### Step 7: Test Gmail Extraction [G2: Pre-Release]

**Prompt:**

Test the fixed Gmail extraction:

1. Clean up failed extraction: `rm -rf /home/user/phi-data/vault/email/gmail/ben-personal/dt=2025-11-13/`
2. Run sync: `lorch config sync meltano`
3. Validate: `lorch tools validate meltano`
4. Run extraction: `lorch extract tap-gmail--acct1-personal`
5. Verify results:
   - Small file size (~1-5MB for 1 day)
   - Fast completion (1-2 minutes)
   - Correct date range (only after 2025/11/12)
   - No base64 bloat

**Commands:**

```bash
# Clean failed run
rm -rf /home/user/phi-data/vault/email/gmail/ben-personal/dt=2025-11-13/

# Test new commands
cd /home/user/lorch
source .venv/bin/activate
lorch config sync meltano
lorch config show meltano
lorch tools validate meltano

# Run extraction
time lorch extract tap-gmail--acct1-personal

# Check results
ls -lh /home/user/phi-data/vault/email/gmail/ben-personal/dt=*/run_id=*/
zcat /home/user/phi-data/vault/email/gmail/ben-personal/dt=*/run_id=*/part-000.jsonl.gz | head -n 1 | jq '.internalDate'
```

**Validation:**

- ✓ Extraction completes in < 3 minutes
- ✓ File size < 10MB
- ✓ Messages dated after 2025/11/12
- ✓ No validation errors from adapter

**Outputs:**

- Working extraction with validated config
- Test results documented

---

### Step 8: Documentation [G3: Post-Implementation]

**Prompt:**

Update documentation:

1. Add "Tool Adapter Pattern" section to README.md
2. Document new commands in CLI help
3. Add example adapter usage
4. Document how to add new tool adapters

**Outputs:**

- `README.md` (updated)
- `docs/tool-adapters.md` (new)

---

### Step 9: Decision Log [G4: Final Approval]

**Prompt:**

Document architectural decision:
- Why tool adapters over plugin registry
- Tradeoffs considered
- Future extensibility plan

**Outputs:**

- `artifacts/governance/decision-log.md`

## Models & Tools

**Tools:** bash, pytest, ruff, jq

**Models:** (to be filled by defaults)

## Repository

**Branch:** `feat/tool-adapter-pattern`

**Merge Strategy:** squash
