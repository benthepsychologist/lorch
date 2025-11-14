# Time-Based Extraction with lorch

## Overview

The `lorch extract` command now supports convenient time-based filtering flags that make it easy to extract data for specific time ranges without manually editing configuration files.

## New CLI Flags

### 1. `--last <duration>`

Extract data from the last N days/weeks/months.

**Syntax:**
- `7d` = last 7 days
- `2w` = last 2 weeks
- `1m` = last 1 month (30 days)
- `3y` = last 3 years

**Examples:**
```bash
# Extract last 7 days from Gmail
lorch extract tap-gmail--acct1-personal --last 7d

# Extract last 30 days from Exchange
lorch extract tap-msgraph-mail--ben-mensio --last 30d

# Extract last 2 weeks from Dataverse
lorch extract tap-dataverse --last 2w
```

### 2. `--since <date>`

Extract data since a specific date (inclusive).

**Formats:**
- Absolute: `2025-11-01`, `2025/11/01`
- Relative: `7d`, `2w`, `1m`

**Examples:**
```bash
# Extract since November 1st
lorch extract tap-gmail--acct1-personal --since 2025-11-01

# Extract since 7 days ago
lorch extract tap-gmail--acct1-personal --since 7d
```

### 3. `--from <date> --to <date>`

Extract data for a specific date range (inclusive).

**Note:** Both `--from` and `--to` must be specified together.

**Examples:**
```bash
# Extract November 1-15
lorch extract tap-gmail--acct1-personal --from 2025-11-01 --to 2025-11-15

# Extract last week using relative dates
lorch extract tap-msgraph-mail--ben-mensio --from 14d --to 7d
```

### 4. `-q, --query <query>`

Provide a custom query string in provider-specific syntax. This overrides all time-based flags.

**Examples:**
```bash
# Gmail: custom query with label filter
lorch extract tap-gmail--acct1-personal -q "label:inbox after:2025/11/12"

# Gmail: search for specific subject
lorch extract tap-gmail--acct1-personal -q "subject:invoice after:2025/11/01"
```

## Provider-Specific Query Formats

The tool automatically detects the provider type and formats queries appropriately:

### Gmail

```
# Single date (since)
after:2025/11/01

# Date range
after:2025/11/01 before:2025/11/15
```

### Exchange/Microsoft Graph

```
# Single date (since)
receivedDateTime ge 2025-11-01T00:00:00Z

# Date range
receivedDateTime ge 2025-11-01T00:00:00Z and receivedDateTime le 2025-11-15T23:59:59Z
```

### Dataverse

```
# Single date (since)
modifiedon ge 2025-11-01

# Date range
modifiedon ge 2025-11-01 and modifiedon le 2025-11-15
```

## How It Works

1. **Query Building:** The CLI parses your time flags and builds a provider-specific query string
2. **Environment Variables:** The query is passed to Meltano via environment variables:
   - Gmail: `TAP_GMAIL_MESSAGES_Q`
   - Exchange: `TAP_MSGRAPH_MAIL_FILTER`
   - Dataverse: `TAP_DATAVERSE_FILTER`
3. **Meltano Execution:** Meltano runs with the query filter applied

## Validation Rules

The CLI enforces mutual exclusivity to prevent conflicts:

- ✗ Cannot use `--last` with `--since`, `--from`, or `--to`
- ✗ Cannot use `--since` with `--from` or `--to`
- ✗ Must specify both `--from` and `--to` together
- ✓ Custom `-q` overrides all time-based flags

**Examples of invalid usage:**
```bash
# ERROR: Cannot mix --last with --since
lorch extract tap-gmail--acct1-personal --last 7d --since 2025-11-01

# ERROR: Must specify both --from and --to
lorch extract tap-gmail--acct1-personal --from 2025-11-01
```

## Complete Examples

### Example 1: Backfill Last Month

```bash
# Extract last 30 days from all Gmail accounts
lorch extract tap-gmail--acct1-personal --last 30d
lorch extract tap-gmail--acct2-business1 --last 30d
lorch extract tap-gmail--acct3-bfarmstrong --last 30d
```

### Example 2: Extract Specific Week

```bash
# Extract November 1-7, 2025
lorch extract tap-msgraph-mail--ben-mensio --from 2025-11-01 --to 2025-11-07
```

### Example 3: Recent Data Only

```bash
# Extract last 7 days for testing
lorch extract tap-gmail--acct1-personal --last 7d
```

### Example 4: Custom Query with Additional Filters

```bash
# Gmail: Only inbox messages from last 14 days
lorch extract tap-gmail--acct1-personal -q "label:inbox after:2025/10/30"

# Gmail: Unread messages from last week
lorch extract tap-gmail--acct1-personal -q "is:unread after:2025/11/06"
```

## Checking Query in Output

When you run `lorch extract` with a time filter, the query string is displayed in the output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
         Extracting: tap-gmail--acct1-personal
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ℹ Auto-selected target: target-jsonl-chunked--gmail-ben-mensio
ℹ Command: meltano run tap-gmail--acct1-personal target-jsonl-chunked--gmail-ben-mensio
ℹ Run ID: 20251113T120000Z
ℹ Query: after:2025/11/06
ℹ Working directory: /home/user/meltano-ingest
ℹ Vault directory: /home/user/phi-data/vault
```

## Implementation Details

### Date Parsing Utilities

- `parse_relative_date(relative)` - Parse "7d", "2w", "1m" to datetime
- `parse_date_string(date_str)` - Parse absolute or relative dates
- `format_date_for_provider(provider, from_dt, to_dt)` - Format for Gmail/Exchange/Dataverse
- `detect_provider_from_tap_name(tap_name)` - Auto-detect provider type

### Files Modified

- `lorch/utils.py` - Added date parsing and formatting utilities (~180 lines)
- `lorch/cli.py` - Added CLI options and query building logic (~60 lines)

## Future Enhancements

Potential future improvements:

- [ ] Support for more granular time units (hours, minutes)
- [ ] Preset date ranges (e.g., `--today`, `--yesterday`, `--this-week`)
- [ ] Save common queries as named presets
- [ ] Support for timezone-aware queries
- [ ] Validation against provider-specific query limits

## Related Documentation

- [Meltano Configuration](meltano_configuration.md)
- [Vault Structure](vault.md)
- [Extract Stage](stages/extract.md)

---

**Last Updated:** 2025-11-13
