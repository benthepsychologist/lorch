# Vault Storage Architecture

## Overview

lorch uses a time-series vault storage system with LATEST pointers to provide deterministic, idempotent data processing while maintaining a complete audit trail.

## Design Principles

1. **Time-Series**: All extraction runs are preserved with timestamps
2. **LATEST Pointer**: Single source of truth for canonization
3. **Deterministic**: Same LATEST → same canonical output
4. **Idempotent**: Safe to rerun canonization
5. **Auditable**: Complete history for compliance
6. **Safe**: Failed runs don't update LATEST

## Directory Structure

```
vault/{domain}/{source}/{account}/
├── dt=2025-11-12/run_id=20251112T205433Z/
│   ├── manifest.json               # Run metadata
│   ├── part-000.jsonl.gz          # Data chunk 1
│   ├── part-001.jsonl.gz          # Data chunk 2
│   └── _tmp/                       # Temp files during write
├── dt=2025-11-13/run_id=20251113T103045Z/
│   ├── manifest.json
│   └── part-000.jsonl.gz
├── dt=2025-11-14/run_id=20251114T094521Z/
│   ├── manifest.json
│   └── part-000.jsonl.gz
└── LATEST.json  ← Points to most recent successful run
```

### Path Components

- **domain**: High-level category (email, stripe, questionnaires, etc.)
- **source**: Specific source type (gmail, exchange, dataverse, etc.)
- **account**: Account identifier (ben-mensio, drben, benthepsychologist, etc.)
- **dt**: Date partition (YYYY-MM-DD)
- **run_id**: UTC timestamp (YYYYMMDDTHHMMSSz)

### Examples

```
vault/email/gmail/ben-mensio/
vault/email/exchange/booking-mensio/
vault/stripe/benthepsychologist/
vault/questionnaires/google-sheets/followup/
```

## LATEST.json Format

The LATEST.json marker file points to the most recent successful extraction run:

```json
{
  "dt": "2025-11-14",
  "run_id": "20251114T094521Z",
  "updated_at": "2025-11-14T09:45:30.123456+00:00",
  "records": 295
}
```

### Fields

- **dt**: Date partition of the pointed run
- **run_id**: Run ID of the pointed run
- **updated_at**: When this run completed (ISO 8601 UTC)
- **records**: Total record count in this run

### Permissions

- File: 600 (owner read/write only)
- Location: `vault/{domain}/{source}/{account}/LATEST.json`

## manifest.json Format

Each run directory contains a manifest.json with complete metadata:

```json
{
  "run_id": "20251114T094521Z",
  "source": "email/gmail",
  "account": "ben-mensio",
  "stream": "messages",
  "status": "completed",
  "started_utc": "2025-11-14T09:45:21.123456+00:00",
  "ended_utc": "2025-11-14T09:45:30.123456+00:00",
  "parts": [
    {
      "path": "part-000.jsonl.gz",
      "seq": 0,
      "size_compressed": 321500,
      "size_uncompressed": 2774479,
      "record_count": 295,
      "md5": "ecfe6350c4afd2fab8ccb46127837f58",
      "sha256": "609bfe293f2cdf83e03ad49caba247edbdc64264311e47f5362fd3458f2c1161"
    }
  ],
  "totals": {
    "parts": 1,
    "records": 295,
    "size_compressed": 321500,
    "size_uncompressed": 2774479
  }
}
```

### Status Values

- **completed**: Run finished successfully, LATEST may be updated
- **aborted**: Run failed, LATEST will not be updated

## Extraction Workflow

### 1. Start Extraction

```bash
lorch extract tap-gmail--acct1-personal
```

- Auto-selects target: `target-jsonl-chunked--acct1-ben-mensio`
- Sets `RUN_ID` environment variable (UTC timestamp)
- Calls meltano: `meltano run tap-gmail--acct1-personal target-jsonl-chunked--acct1-ben-mensio`

### 2. Meltano Writes to Vault

target-jsonl-chunked creates:
1. Run directory: `vault/email/gmail/ben-mensio/dt=YYYY-MM-DD/run_id=TIMESTAMP/`
2. Temporary directory: `_tmp/` for atomic writes
3. Data chunks: `_tmp/part-NNN.jsonl.gz` (gzip compressed)
4. On success: Move chunks from `_tmp/` to run directory
5. Write manifest.json with status="completed"

### 3. lorch Updates LATEST

After successful extraction:
1. Find manifests with this run_id
2. Check status == "completed"
3. Extract dt and run_id from path
4. Write/update LATEST.json in account directory
5. Set 600 permissions

### 4. Failed Runs

If extraction fails:
1. Partial run directory may exist
2. manifest.json has status="aborted" or missing
3. **LATEST.json NOT updated** (safety!)
4. Historical runs unaffected
5. Canonization continues using previous LATEST

## Canonization Workflow

### 1. Start Canonization

```bash
lorch run --stage canonize
```

### 2. Discover LATEST Runs

For each mapping (email/gmail, email/exchange, etc.):
1. Find all account directories under source
2. Read LATEST.json from each account
3. Build path to manifest: `dt={dt}/run_id={run_id}/manifest.json`
4. **Ignore all non-LATEST runs**

### 3. Process LATEST Manifest

For each LATEST manifest:
1. Read manifest.json
2. Get list of parts (chunks)
3. Build per-account output path: `canonical/{source}/{account}.jsonl`
4. **Clear existing output file** (idempotency)
5. Process each part in sequence:
   - Decompress gzip
   - Pipe to canonizer transform
   - Append to output file

### 4. Deterministic Output

- Same LATEST → same parts → same canonical output
- Rerunning produces identical results
- No historical runs processed
- Clear and rebuild strategy

## State Management

### Extraction State (Meltano)

Meltano maintains state for incremental extraction:
- State stored in: `.meltano/state/`
- Tracks: last sync timestamp, cursor values
- Incremental: Only new/updated records extracted

Example:
- Day 1: Extract 1000 messages (full)
- Day 2: Extract 50 messages (incremental)
- Day 3: Extract 30 messages (incremental)

Each creates new vault run, LATEST points to most recent.

### Canonization State (LATEST)

lorch uses LATEST pointer for canonization state:
- LATEST points to which run to process
- Canonize doesn't track state internally
- Rerunning canonize = rebuild from LATEST

Example:
- Day 1: LATEST → run_id=ABC (1000 records)
- Day 2: LATEST → run_id=DEF (50 records)
- Day 3: LATEST → run_id=GHI (30 records)

Each time canonize runs, it processes only the LATEST run (ignoring previous).

## Audit Trail

### Historical Runs

All runs preserved:
```
vault/email/gmail/ben-mensio/
├── dt=2025-11-12/run_id=ABC/  # Week 1
├── dt=2025-11-13/run_id=DEF/  # Week 2
├── dt=2025-11-14/run_id=GHI/  # Week 3
└── LATEST.json → GHI
```

### Why Keep History?

1. **Compliance**: Audit trail for PHI access
2. **Debugging**: Compare runs to identify issues
3. **Recovery**: Rollback to previous run if needed
4. **Analysis**: Track data growth over time

### Cleanup (Future)

For now: Keep all runs
Future: Add retention policy
```bash
# Future command (not implemented)
lorch clean --vault --older-than 90d
```

## Idempotency Guarantees

### Extraction

Running extract multiple times:
1. First run: Creates new vault run, updates LATEST
2. Second run: Creates another vault run, updates LATEST to new run
3. Old runs preserved but ignored

Safe to rerun - each run gets unique run_id.

### Canonization

Running canonize multiple times with same LATEST:
1. First run: Clears output, rebuilds from LATEST
2. Second run: Clears output, rebuilds from LATEST (identical)
3. Third run: Clears output, rebuilds from LATEST (identical)

Safe to rerun - deterministic output.

## Security

### File Permissions

- **Vault directories**: 700 (owner only)
- **Run directories**: 700
- **Data files**: 600 (owner read/write only)
- **LATEST.json**: 600
- **manifest.json**: 600

Enforced by:
- target-jsonl-chunked on write
- lorch validation before read
- System-level PHI directory checks

### PHI Protection

- No PHI in filenames (only IDs)
- No PHI in manifests (only counts)
- No PHI in LATEST.json
- Data encrypted at rest (filesystem level)
- Restrictive permissions prevent access

## Performance

### Compression

- Gzip level 1 (fastest)
- Typical compression: 80-90% reduction
- Trade-off: Speed over max compression

### Chunking

- Default: 50MB compressed per chunk
- Large datasets → multiple parts
- Enables streaming processing
- Parallel decompression possible (future)

### Manifest Overhead

- Small JSON files (~1KB)
- Read once per account
- Cached in memory during processing
- Negligible performance impact

## Best Practices

### Extraction

1. **Run regularly**: Daily/hourly depending on source
2. **Monitor LATEST**: Verify updates after each run
3. **Check manifests**: Validate record counts
4. **Verify permissions**: Ensure 700/600 enforced

### Canonization

1. **Process LATEST only**: Don't implement --all-runs yet
2. **Verify output**: Check canonical file created
3. **Rerun if needed**: Idempotent, safe to retry
4. **Monitor determinism**: Same LATEST = same output

### Maintenance

1. **Disk space**: Monitor vault growth
2. **Retention**: Plan future cleanup strategy
3. **Backups**: Backup vault periodically
4. **Monitoring**: Track run counts, sizes, durations

## Troubleshooting

### LATEST.json Not Updated

**Symptom**: Extraction runs but LATEST.json unchanged

**Causes**:
- Extraction failed (manifest status != "completed")
- Permissions issue (can't write LATEST.json)
- Wrong vault path in configuration

**Solution**:
```bash
# Check manifest status
cat vault/email/gmail/account/dt=DATE/run_id=ID/manifest.json | jq .status

# Check permissions
ls -la vault/email/gmail/account/LATEST.json

# Manually update LATEST if needed
python3 -c "
import json
from pathlib import Path
latest = {
  'dt': '2025-11-14',
  'run_id': '20251114T094521Z',
  'updated_at': '2025-11-14T09:45:30+00:00',
  'records': 295
}
Path('vault/email/gmail/account/LATEST.json').write_text(json.dumps(latest, indent=2))
"
chmod 600 vault/email/gmail/account/LATEST.json
```

### Canonization Processes Wrong Run

**Symptom**: Canonical output doesn't match expected data

**Cause**: LATEST.json points to old run

**Solution**:
```bash
# Check LATEST pointer
cat vault/email/gmail/account/LATEST.json

# Check what runs exist
ls -la vault/email/gmail/account/dt=*/run_id=*/manifest.json

# Update LATEST to specific run
# (see manual update above)
```

### Corrupted Gzip Files

**Symptom**: `Not a gzipped file` error during canonization

**Cause**: Extraction interrupted mid-write

**Solution**:
```bash
# Remove corrupted run
rm -rf vault/email/gmail/account/dt=DATE/run_id=ID/

# Rerun extraction
lorch extract tap-gmail--account
```

LATEST won't point to corrupted run if extraction failed.

## Future Enhancements

### Planned

- [ ] `--all-runs` flag for canonization (process all historical)
- [ ] `--since DATE` flag (process runs since date)
- [ ] Retention policy and cleanup commands
- [ ] Compression level configuration
- [ ] Chunk size configuration
- [ ] Parallel chunk processing
- [ ] Incremental canonization (upsert mode)

### Not Planned

- ❌ Symlinks (portability concerns)
- ❌ Merge historical runs (complexity)
- ❌ Deduplication in canonize (meltano handles this)

## References

- [target-jsonl-chunked](../../meltano-ingest/targets/target-jsonl-chunked/)
- [lorch/stages/canonize.py](../lorch/stages/canonize.py)
- [lorch/cli.py](../lorch/cli.py)
- [config/pipeline.yaml](../config/pipeline.yaml)
