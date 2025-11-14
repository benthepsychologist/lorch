"""
CLI interface for lorch pipeline orchestrator.

Provides commands: run, status, validate, clean.
"""

from pathlib import Path

import click

from lorch import __version__
from lorch.config import load_config
from lorch.pipeline import Pipeline
from lorch.utils import (
    format_duration,
    print_banner,
    print_error,
    print_info,
    print_success,
    print_warning,
)


@click.group()
@click.version_option(version=__version__, prog_name="lorch")
def main():
    """
    lorch - Local Orchestrator for PHI data pipeline.

    Coordinates extract → canonize → index stages.
    """
    pass


@main.command()
@click.option(
    "--stage",
    type=click.Choice(["extract", "canonize", "index"], case_sensitive=False),
    help="Run specific stage only",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file (default: config/pipeline.yaml)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without executing",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable debug logging",
)
def run(stage, config, dry_run, verbose):
    """
    Run pipeline stages.

    Examples:

      # Run full pipeline
      lorch run

      # Run single stage
      lorch run --stage extract

      # Dry run (validation only)
      lorch run --dry-run

      # Verbose logging
      lorch run --verbose

      # Custom config
      lorch run --config /path/to/pipeline.yaml
    """
    try:
        # Load configuration
        pipeline_config = load_config(config) if config else load_config()
        pipeline = Pipeline(pipeline_config)

        # Determine stages to run
        stages_to_run = [stage] if stage else None

        # Run pipeline
        result = pipeline.run(
            stages=stages_to_run,
            dry_run=dry_run,
            verbose=verbose,
        )

        # Exit with appropriate code
        if result.success:
            exit(0)
        else:
            exit(1)

    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        exit(1)


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def status(config):
    """
    Show pipeline status and last run information.

    Examples:

      # Show status
      lorch status

      # Status with custom config
      lorch status --config /path/to/pipeline.yaml
    """
    try:
        # Load configuration
        pipeline_config = load_config(config) if config else load_config()
        pipeline = Pipeline(pipeline_config)

        # Get last run status
        last_run = pipeline.status()

        if not last_run:
            print_info("No previous pipeline runs found")
            exit(0)

        # Display status
        print_info(f"Pipeline Status: {pipeline_config.name} v{pipeline_config.version}\n")

        status_symbol = "✅" if last_run.success else "❌"
        status_text = "SUCCESS" if last_run.success else "FAILED"

        print(f"Last Run: {last_run.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Status: {status_symbol} {status_text}")
        print(f"Duration: {format_duration(last_run.duration_seconds)}")

        if last_run.error_message:
            print(f"Error: {last_run.error_message}")

        print()

        if last_run.stages:
            print("Stages:")
            for stage_name, stage_result in last_run.stages.items():
                stage_symbol = "✅" if stage_result.success else "❌"
                duration = format_duration(stage_result.duration_seconds)
                records = stage_result.records_processed

                print(f"  {stage_symbol} {stage_name:<12} {duration:>8}  {records:>6} records")

        # Show log location
        log_file = pipeline_config.get_log_file_path()
        if log_file.exists():
            print(f"\nLogs: {log_file}")

    except Exception as e:
        print_error(f"Could not retrieve status: {e}")
        exit(1)


@main.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
@click.option(
    "--skip-permissions",
    is_flag=True,
    help="Skip PHI directory permission checks",
)
def validate(config, skip_permissions):
    """
    Validate pipeline configuration and dependencies.

    Checks:
    - Configuration file syntax
    - Component paths exist
    - Transform registry accessible
    - Output directories writable
    - PHI directory permissions

    Examples:

      # Validate default config
      lorch validate

      # Validate custom config
      lorch validate --config /path/to/pipeline.yaml

      # Skip permission checks
      lorch validate --skip-permissions
    """
    try:
        # Load configuration
        pipeline_config = load_config(config) if config else load_config()
        pipeline = Pipeline(pipeline_config)

        # Validate
        pipeline.validate(skip_permissions=skip_permissions)

        print_success("✓ Pipeline configuration is valid")
        exit(0)

    except Exception as e:
        print_error(f"Validation failed: {e}")
        exit(1)


@main.command()
@click.option(
    "--stage",
    type=click.Choice(["extract", "canonize", "index"], case_sensitive=False),
    help="Clean specific stage output only",
)
@click.option(
    "--all",
    "clean_all",
    is_flag=True,
    help="Clean all stage outputs (use with caution)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be deleted without deleting",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
@click.confirmation_option(
    prompt="Are you sure you want to delete pipeline outputs? This cannot be undone.",
)
def clean(stage, clean_all, dry_run, config):
    """
    Clean stage outputs.

    WARNING: This permanently deletes output files.

    Examples:

      # Clean specific stage
      lorch clean --stage extract

      # Clean all stages (prompts for confirmation)
      lorch clean --all

      # Dry run (show what would be deleted)
      lorch clean --all --dry-run
    """
    try:
        import shutil

        # Load configuration
        pipeline_config = load_config(config) if config else load_config()

        # Determine stages to clean
        if stage:
            stages_to_clean = [stage]
        elif clean_all:
            stages_to_clean = ["extract", "canonize", "index"]
        else:
            print_error("Must specify --stage or --all")
            exit(1)

        # Clean each stage
        for stage_name in stages_to_clean:
            stage_config = pipeline_config.get_stage(stage_name)

            if not stage_config:
                print_error(f"Stage not found: {stage_name}")
                continue

            output_dir = stage_config.output_dir

            if not output_dir.exists():
                print_info(f"{stage_name}: output directory does not exist ({output_dir})")
                continue

            # Count files
            files = list(output_dir.glob("*"))

            if not files:
                print_info(f"{stage_name}: no files to clean")
                continue

            if dry_run:
                print_info(f"{stage_name}: would delete {len(files)} files from {output_dir}")
                for f in files[:10]:  # Show first 10
                    print(f"  - {f.name}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")
            else:
                print_info(f"{stage_name}: deleting {len(files)} files from {output_dir}")

                for f in files:
                    try:
                        if f.is_dir():
                            shutil.rmtree(f)
                        else:
                            f.unlink()
                    except Exception as e:
                        print_error(f"Could not delete {f.name}: {e}")

                print_success(f"{stage_name}: cleaned {len(files)} files")

        exit(0)

    except Exception as e:
        print_error(f"Clean failed: {e}")
        exit(1)


@main.command()
@click.argument("tap_name")
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
@click.option(
    "--target",
    default=None,
    help="Target to use (auto-selects chunked target if not specified)",
)
@click.option(
    "-q",
    "--query",
    default=None,
    help="Custom query string (provider-specific syntax, overrides time flags)",
)
@click.option(
    "--since",
    default=None,
    help="Extract since date (YYYY-MM-DD or relative like '7d', '2w', '1m')",
)
@click.option(
    "--from",
    "from_date",
    default=None,
    help="Extract from date (YYYY-MM-DD or relative)",
)
@click.option(
    "--to",
    "to_date",
    default=None,
    help="Extract to date (YYYY-MM-DD or relative)",
)
@click.option(
    "--last",
    default=None,
    help="Extract last N days/weeks/months (e.g., '7d', '2w', '1m')",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output",
)
def extract(tap_name, config, target, query, since, from_date, to_date, last, verbose):
    """
    Run a single meltano extractor with target-jsonl-chunked.

    Automatically selects the appropriate chunked target based on tap name.
    Supports time-based filtering with convenient flags.

    Examples:

      # Extract from single Gmail account (auto-selects chunked target)
      lorch extract tap-gmail--acct1-personal

      # Extract last 7 days from Gmail
      lorch extract tap-gmail--acct1-personal --last 7d

      # Extract since specific date
      lorch extract tap-gmail--acct1-personal --since 2025-11-01

      # Extract date range
      lorch extract tap-gmail--acct1-personal --from 2025-11-01 --to 2025-11-15

      # Custom query (provider-specific syntax)
      lorch extract tap-gmail--acct1-personal -q "label:inbox after:2025/11/12"

      # Extract from Exchange with time filter
      lorch extract tap-msgraph-mail--ben-mensio --last 30d

      # Override target
      lorch extract tap-dataverse --target target-jsonl
    """
    try:
        import subprocess
        import time
        import os
        from lorch.utils import (
            parse_date_string,
            format_date_for_provider,
            detect_provider_from_tap_name,
        )

        # Validate query flags (mutual exclusivity)
        if last and (since or from_date or to_date):
            print_error("Cannot use --last with --since, --from, or --to")
            exit(1)

        if since and (from_date or to_date):
            print_error("Cannot use --since with --from or --to")
            exit(1)

        if (from_date and not to_date) or (to_date and not from_date):
            print_error("Must specify both --from and --to together")
            exit(1)

        # Build query string
        query_string = None

        if query:
            # Custom query takes precedence
            query_string = query
        elif last or since or from_date:
            # Build query from time flags
            provider = detect_provider_from_tap_name(tap_name)

            try:
                if last:
                    # Parse relative date
                    from_dt = parse_date_string(last)
                    to_dt = None
                    query_string = format_date_for_provider(provider, from_dt, to_dt)
                elif since:
                    # Parse since date
                    from_dt = parse_date_string(since)
                    to_dt = None
                    query_string = format_date_for_provider(provider, from_dt, to_dt)
                elif from_date and to_date:
                    # Parse date range
                    from_dt = parse_date_string(from_date)
                    to_dt = parse_date_string(to_date)
                    query_string = format_date_for_provider(provider, from_dt, to_dt)
            except ValueError as e:
                print_error(f"Invalid date format: {e}")
                exit(1)

        # Load configuration
        pipeline_config = load_config(config) if config else load_config()

        # Get extract stage config
        extract_config = pipeline_config.get_stage("extract")
        if not extract_config:
            print_error("Extract stage not configured")
            exit(1)

        # Auto-select chunked target if not specified
        if target is None:
            target = _select_chunked_target(tap_name)
            print_info(f"Auto-selected target: {target}")

        # Set RUN_ID environment variable for chunked targets
        run_id = os.environ.get("RUN_ID")
        if not run_id:
            from datetime import datetime
            run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            os.environ["RUN_ID"] = run_id

        # Set query environment variable if specified
        if query_string:
            provider = detect_provider_from_tap_name(tap_name)

            if provider == "gmail":
                # Gmail uses messages.q nested config
                os.environ["TAP_GMAIL_MESSAGES_Q"] = query_string
            elif provider == "exchange":
                # Exchange/msgraph uses filter parameter
                os.environ["TAP_MSGRAPH_MAIL_FILTER"] = query_string
            elif provider == "dataverse":
                # Dataverse uses filter parameter
                os.environ["TAP_DATAVERSE_FILTER"] = query_string
            else:
                # Generic query parameter
                os.environ["TAP_QUERY"] = query_string

        # Build command
        meltano_bin = extract_config.venv_path / "bin" / "meltano"
        command = [str(meltano_bin), "run", tap_name, target]

        print_banner(f"Extracting: {tap_name}")
        print_info(f"Command: meltano run {tap_name} {target}")
        print_info(f"Run ID: {run_id}")
        if query_string:
            print_info(f"Query: {query_string}")
        print_info(f"Working directory: {extract_config.repo_path}")
        print_info(f"Vault directory: {extract_config.output_dir}\n")

        start_time = time.time()

        # Run meltano
        result = subprocess.run(
            command,
            cwd=extract_config.repo_path,
            capture_output=not verbose,
            text=True,
            check=False,
        )

        duration = time.time() - start_time

        if result.returncode != 0:
            print_error(f"\nExtraction failed (exit code {result.returncode})")
            if not verbose and result.stderr:
                print_error(f"Error: {result.stderr[:500]}")
            exit(1)

        # Find and display manifests
        print_success(f"\nExtraction completed in {format_duration(duration)}")

        # Update LATEST pointers for successful runs
        _update_latest_pointers(extract_config.output_dir, run_id)

        _display_vault_summary(extract_config.output_dir, run_id)

        exit(0)

    except Exception as e:
        print_error(f"Extraction failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        exit(1)


def _select_chunked_target(tap_name: str) -> str:
    """Select appropriate chunked target based on tap name."""
    # Gmail accounts
    if "gmail--acct1" in tap_name:
        return "target-jsonl-chunked--gmail-ben-mensio"
    elif "gmail--acct2" in tap_name:
        return "target-jsonl-chunked--gmail-drben"
    elif "gmail--acct3" in tap_name:
        return "target-jsonl-chunked--gmail-ben-personal"

    # Exchange/MS Graph accounts
    elif "msgraph-mail--ben-mensio" in tap_name:
        return "target-jsonl-chunked--ben-mensio-exchange"
    elif "msgraph-mail--booking" in tap_name:
        return "target-jsonl-chunked--booking-mensio"
    elif "msgraph-mail--info" in tap_name:
        return "target-jsonl-chunked--info-mensio"
    elif "msgraph-mail--ben-efs" in tap_name:
        return "target-jsonl-chunked--ben-efs"

    # Dataverse
    elif "dataverse" in tap_name:
        return "target-jsonl-chunked--dataverse"

    # Google Sheets
    elif "google-sheets--initial-1" in tap_name:
        return "target-jsonl-chunked--google-sheets-initial-1"
    elif "google-sheets--initial-2" in tap_name:
        return "target-jsonl-chunked--google-sheets-initial-2"
    elif "google-sheets--followup" in tap_name:
        return "target-jsonl-chunked--google-sheets-followup"

    # QuickBooks
    elif "quickbooks--ben-personal" in tap_name:
        return "target-jsonl-chunked--quickbooks-ben-personal"
    elif "quickbooks--mensio-cad" in tap_name:
        return "target-jsonl-chunked--quickbooks-mensio-cad"

    else:
        # Fallback to generic chunked target
        return "target-jsonl-chunked"


def _update_latest_pointers(vault_root: Path, run_id: str):
    """
    Update LATEST.json pointers for all successful runs.

    Creates/updates LATEST.json in each account directory pointing to the
    latest successful run.
    """
    import json
    import re
    from builtins import list as builtin_list

    # Find all manifests with this run_id
    manifests = builtin_list(vault_root.rglob(f"run_id={run_id}/manifest.json"))

    for manifest_path in manifests:
        try:
            # Read manifest to check status
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            status = manifest.get("status")
            if status != "completed":
                continue  # Skip failed runs

            # Extract dt and run_id from path
            # e.g., vault/email/gmail/ben-mensio/dt=2025-11-12/run_id=20251112T205433Z/manifest.json
            run_dir = manifest_path.parent
            run_id_dir = run_dir.name  # run_id=20251112T205433Z
            dt_dir = run_dir.parent.name  # dt=2025-11-12
            account_dir = run_dir.parent.parent  # vault/email/gmail/ben-mensio/

            # Extract values
            dt_match = re.match(r"dt=(.+)", dt_dir)
            run_id_match = re.match(r"run_id=(.+)", run_id_dir)

            if not dt_match or not run_id_match:
                continue

            dt = dt_match.group(1)
            run_id_value = run_id_match.group(1)

            # Create LATEST.json in account directory
            latest_file = account_dir / "LATEST.json"
            latest_data = {
                "dt": dt,
                "run_id": run_id_value,
                "updated_at": manifest.get("ended_utc"),
                "records": manifest.get("totals", {}).get("records", 0),
            }

            with open(latest_file, "w") as f:
                json.dump(latest_data, f, indent=2)

            # Set secure permissions (600)
            latest_file.chmod(0o600)

        except Exception as e:
            print_warning(f"Could not update LATEST pointer for {manifest_path}: {e}")


def _display_vault_summary(vault_root: Path, run_id: str):
    """Display summary of extracted data from vault manifests."""
    import json
    from builtins import list as builtin_list

    print_info("\nVault Summary:")

    # Find manifests with this run_id
    manifests = builtin_list(vault_root.rglob(f"run_id={run_id}/manifest.json"))

    if not manifests:
        print_warning("No manifests found (extraction may have failed)")
        return

    for manifest_path in manifests:
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            source = manifest.get("source", "unknown")
            account = manifest.get("account", "unknown")
            totals = manifest.get("totals", {})
            records = totals.get("records", 0)
            size_compressed = totals.get("size_compressed", 0)
            size_mb = size_compressed / (1024 * 1024)
            parts = totals.get("parts", 0)

            print(f"\n  {source}/{account}:")
            print(f"    Records: {records}")
            print(f"    Size: {size_mb:.2f} MB (compressed)")
            print(f"    Parts: {parts}")
            print(f"    Location: {manifest_path.parent}")

        except Exception as e:
            print_warning(f"Could not read manifest: {manifest_path}: {e}")


@main.group()
def list():
    """
    List available extractors, jobs, transforms, and mappings.

    Examples:

      # List all extractors
      lorch list extractors

      # List all jobs
      lorch list jobs

      # List all transforms
      lorch list transforms

      # List configured mappings
      lorch list mappings
    """
    pass


@list.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def extractors(config):
    """
    List all configured meltano extractors (taps).

    Shows all tap instances configured in meltano.yml.
    """
    try:
        import subprocess
        import yaml

        # Load configuration
        pipeline_config = load_config(config) if config else load_config()

        # Get meltano repo path
        extract_config = pipeline_config.get_stage("extract")
        if not extract_config:
            print_error("Extract stage not configured")
            exit(1)

        meltano_path = extract_config.repo_path / "meltano.yml"

        if not meltano_path.exists():
            print_error(f"Meltano configuration not found: {meltano_path}")
            exit(1)

        # Parse meltano.yml
        with open(meltano_path, "r") as f:
            meltano_config = yaml.safe_load(f)

        # Get all extractors
        extractors_list = []
        plugins = meltano_config.get("plugins", {})
        extractor_plugins = plugins.get("extractors", [])

        for extractor in extractor_plugins:
            name = extractor.get("name", "")
            variant = extractor.get("variant", "")
            inherit_from = extractor.get("inherit_from", "")

            # Determine type
            if "gmail" in name.lower():
                tap_type = "Gmail"
            elif "msgraph" in name.lower() or "exchange" in name.lower():
                tap_type = "Exchange"
            elif "dataverse" in name.lower():
                tap_type = "Dataverse"
            elif "stripe" in name.lower():
                tap_type = "Stripe"
            elif "sheets" in name.lower():
                tap_type = "Google Sheets"
            else:
                tap_type = "Other"

            # Skip base configurations
            if inherit_from:
                extractors_list.append({
                    "name": name,
                    "type": tap_type,
                    "base": inherit_from,
                })

        print_info(f"Configured Extractors ({len(extractors_list)} total):\n")

        # Group by type
        by_type = {}
        for ext in extractors_list:
            tap_type = ext["type"]
            if tap_type not in by_type:
                by_type[tap_type] = []
            by_type[tap_type].append(ext["name"])

        for tap_type, names in sorted(by_type.items()):
            print(f"\n{tap_type}:")
            for name in sorted(names):
                print(f"  - {name}")

        exit(0)

    except Exception as e:
        print_error(f"Failed to list extractors: {e}")
        exit(1)


@list.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def jobs(config):
    """
    List all configured meltano jobs.

    Jobs are bundles of extractors that run together.
    """
    try:
        import subprocess

        # Load configuration
        pipeline_config = load_config(config) if config else load_config()

        # Get meltano repo path
        extract_config = pipeline_config.get_stage("extract")
        if not extract_config:
            print_error("Extract stage not configured")
            exit(1)

        # Run meltano job list
        meltano_bin = extract_config.venv_path / "bin" / "meltano"

        result = subprocess.run(
            [str(meltano_bin), "job", "list"],
            cwd=extract_config.repo_path,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print_error(f"Failed to list jobs: {result.stderr}")
            exit(1)

        print_info("Configured Jobs:\n")
        print(result.stdout)

        exit(0)

    except Exception as e:
        print_error(f"Failed to list jobs: {e}")
        exit(1)


@list.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def transforms(config):
    """
    List all available transforms.

    Shows transforms from the transform registry.
    """
    try:
        import subprocess

        # Load configuration
        pipeline_config = load_config(config) if config else load_config()

        # Get canonize config
        canonize_config = pipeline_config.get_stage("canonize")
        if not canonize_config:
            print_error("Canonize stage not configured")
            exit(1)

        transform_registry = Path(canonize_config.get("transform_registry"))
        can_bin = canonize_config.venv_path / "bin" / "can"

        if not can_bin.exists():
            print_error(f"Canonizer executable not found: {can_bin}")
            exit(1)

        # Run can transform list
        result = subprocess.run(
            [str(can_bin), "transform", "list", "--dir", str(transform_registry)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            print_error(f"Failed to list transforms: {result.stderr}")
            exit(1)

        print(result.stdout)

        exit(0)

    except Exception as e:
        print_error(f"Failed to list transforms: {e}")
        exit(1)


@list.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def mappings(config):
    """
    List configured source → transform mappings.

    Shows which source files get transformed by which transforms.
    """
    try:
        # Load configuration
        pipeline_config = load_config(config) if config else load_config()

        # Get canonize config
        canonize_config = pipeline_config.get_stage("canonize")
        if not canonize_config:
            print_error("Canonize stage not configured")
            exit(1)

        mappings_list = canonize_config.get("mappings", [])

        if not mappings_list:
            print_info("No mappings configured")
            exit(0)

        print_info(f"Configured Mappings ({len(mappings_list)} total):\n")

        for i, mapping in enumerate(mappings_list, 1):
            source_pattern = mapping.get("source_pattern", "")
            transform = mapping.get("transform", "")
            output_name = mapping.get("output_name", "")

            print(f"{i}. {source_pattern}")
            print(f"   → Transform: {transform}")
            print(f"   → Output: {output_name}.jsonl")
            print()

        exit(0)

    except Exception as e:
        print_error(f"Failed to list mappings: {e}")
        exit(1)


@main.group()
def config():
    """
    Manage tool configurations.

    Sync and display configuration for external tools (meltano, canonizer, etc.).

    Examples:

      # Show meltano configuration
      lorch config show meltano

      # Sync from meltano.yml to lorch cache
      lorch config sync meltano
    """
    pass


@config.command()
@click.argument("tool", type=click.Choice(["meltano", "canonizer", "vector-projector"], case_sensitive=False))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def show(tool, config):
    """
    Display tool configuration from cache.

    Shows the cached configuration for the specified tool.
    Run 'lorch config sync <tool>' first if cache is empty.
    """
    try:
        from datetime import datetime
        import yaml
        from lorch.tools.meltano import MeltanoAdapter
        from lorch.tools.canonizer import CanonizerAdapter
        from lorch.tools.vector_projector import VectorProjectorAdapter

        # Load pipeline configuration
        pipeline_config = load_config(config) if config else load_config()

        if tool == "meltano":
            # Get meltano repo path
            extract_config = pipeline_config.get_stage("extract")
            if not extract_config:
                print_error("Extract stage not configured")
                exit(1)

            # Create adapter
            adapter = MeltanoAdapter(
                meltano_dir=extract_config.repo_path,
                config_cache=Path("config/tools/meltano.yaml"),
            )

            # Check if config is synced
            if not adapter.config or not adapter.config.get("extractors"):
                print_warning("Config cache is empty. Run 'lorch config sync meltano' first.")
                exit(1)

            # Display config
            print_banner(f"Meltano Configuration (cached)")
            print_info(f"Cache: {adapter.config_path}\n")

            synced_at = adapter.config.get("synced_at")
            if synced_at:
                print(f"Synced at: {synced_at}\n")

            extractors = adapter.config.get("extractors", {})
            loaders = adapter.config.get("loaders", {})
            jobs = adapter.config.get("jobs", [])

            print(f"Extractors: {len(extractors)}")
            print(f"Loaders: {len(loaders)}")
            print(f"Jobs: {len(jobs)}")

            # Show extractors grouped by type
            print("\n" + "=" * 60)
            print("EXTRACTORS")
            print("=" * 60 + "\n")

            for name in sorted(extractors.keys()):
                print(f"  {name}")

            # Show loaders
            print("\n" + "=" * 60)
            print("LOADERS")
            print("=" * 60 + "\n")

            for name in sorted(loaders.keys()):
                print(f"  {name}")

            # Show jobs
            if jobs:
                print("\n" + "=" * 60)
                print("JOBS")
                print("=" * 60 + "\n")

                for job in jobs:
                    job_name = job.get("name", "unnamed")
                    tasks = job.get("tasks", [])
                    print(f"  {job_name}: {len(tasks)} tasks")

        elif tool == "canonizer":
            # Get canonizer config
            canonize_config = pipeline_config.get_stage("canonize")
            if not canonize_config:
                print_error("Canonize stage not configured")
                exit(1)

            # Create adapter
            adapter = CanonizerAdapter(
                canonizer_dir=canonize_config.repo_path,
                transform_registry=Path(canonize_config.get("transform_registry")),
                config_cache=Path("config/tools/canonizer.yaml"),
            )

            # Check if config is synced
            if not adapter.config or not adapter.config.get("transforms"):
                print_warning("Config cache is empty. Run 'lorch config sync canonizer' first.")
                exit(1)

            # Display config
            print_banner(f"Canonizer Configuration (cached)")
            print_info(f"Cache: {adapter.config_path}\n")

            transforms = adapter.config.get("transforms", {})
            transform_registry = adapter.config.get("transform_registry")

            print(f"Transform Registry: {transform_registry}")
            print(f"Transforms Discovered: {len(transforms)}\n")

            # Show transforms grouped by category
            print("=" * 60)
            print("TRANSFORMS")
            print("=" * 60 + "\n")

            for name in sorted(transforms.keys()):
                transform = transforms[name]
                print(f"  {name}")
                print(f"    Input:  {transform.get('input_schema', 'unknown')}")
                print(f"    Output: {transform.get('output_schema', 'unknown')}")
                print(f"    Version: {transform.get('version', 'unknown')}")
                print()

        elif tool == "vector-projector":
            # Get vector-projector config
            index_config = pipeline_config.get_stage("index")
            if not index_config:
                print_error("Index stage not configured")
                exit(1)

            # Create adapter
            adapter = VectorProjectorAdapter(
                vector_store_dir=index_config.output_dir,
                config_cache=Path("config/tools/vector_projector.yaml"),
            )

            # Display config (stub)
            print_banner(f"Vector-Projector Configuration (stub)")
            print_info(f"Cache: {adapter.config_path}\n")

            print_warning("Vector-projector is in stub mode (file copying only)")
            print(f"Status: {adapter.config.get('status', 'not_implemented')}")
            print(f"Vector Store: {adapter.config.get('vector_store_dir')}")
            print("\nFuture features:")
            print("  - SQLite indexing")
            print("  - Inode-style storage")
            print("  - Query API")

        exit(0)

    except Exception as e:
        print_error(f"Failed to show config: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


@config.command()
@click.argument("tool", type=click.Choice(["meltano", "canonizer", "vector-projector"], case_sensitive=False))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
def sync(tool, config):
    """
    Sync tool configuration from source to lorch cache.

    Reads the tool's native configuration file and caches it
    in lorch's config/tools/ directory for validation and inspection.
    """
    try:
        from datetime import datetime
        import yaml
        from lorch.tools.meltano import MeltanoAdapter
        from lorch.tools.canonizer import CanonizerAdapter
        from lorch.tools.vector_projector import VectorProjectorAdapter

        # Load pipeline configuration
        pipeline_config = load_config(config) if config else load_config()

        if tool == "meltano":
            # Get meltano repo path
            extract_config = pipeline_config.get_stage("extract")
            if not extract_config:
                print_error("Extract stage not configured")
                exit(1)

            print_info(f"Syncing meltano configuration...")
            print_info(f"Source: {extract_config.repo_path / 'meltano.yml'}")
            print_info(f"Cache: config/tools/meltano.yaml\n")

            # Create adapter
            adapter = MeltanoAdapter(
                meltano_dir=extract_config.repo_path,
                config_cache=Path("config/tools/meltano.yaml"),
            )

            # Sync config
            adapter.sync_config()

            # Update synced_at timestamp
            import yaml
            with open(adapter.config_path, "r") as f:
                cached = yaml.safe_load(f)

            cached["synced_at"] = datetime.utcnow().isoformat() + "Z"

            with open(adapter.config_path, "w") as f:
                yaml.safe_dump(cached, f, default_flow_style=False, sort_keys=False)

            # Reload to get counts
            adapter.config = adapter.load_config()

            extractors = adapter.config.get("extractors", {})
            loaders = adapter.config.get("loaders", {})
            jobs = adapter.config.get("jobs", [])

            print_success(f"✓ Synced {len(extractors)} extractors, {len(loaders)} loaders, {len(jobs)} jobs")
            print_info(f"Cache saved to: {adapter.config_path}")

        elif tool == "canonizer":
            # Get canonizer config
            canonize_config = pipeline_config.get_stage("canonize")
            if not canonize_config:
                print_error("Canonize stage not configured")
                exit(1)

            print_info(f"Syncing canonizer configuration...")
            print_info(f"Source: {canonize_config.get('transform_registry')}")
            print_info(f"Cache: config/tools/canonizer.yaml\n")

            # Create adapter
            adapter = CanonizerAdapter(
                canonizer_dir=canonize_config.repo_path,
                transform_registry=Path(canonize_config.get("transform_registry")),
                config_cache=Path("config/tools/canonizer.yaml"),
            )

            # Sync config (discovers transforms from registry)
            adapter.sync_config()

            # Reload to get counts
            adapter.config = adapter.load_config()

            transforms = adapter.config.get("transforms", {})
            discovered_count = adapter.config.get("discovered_count", 0)

            print_success(f"✓ Discovered {discovered_count} transforms")
            print_info(f"Cache saved to: {adapter.config_path}")

        elif tool == "vector-projector":
            # Get vector-projector config
            index_config = pipeline_config.get_stage("index")
            if not index_config:
                print_error("Index stage not configured")
                exit(1)

            print_info(f"Syncing vector-projector configuration...")
            print_warning("Vector-projector is in stub mode - no configuration to sync")
            print_info(f"Cache: config/tools/vector_projector.yaml\n")

            # Create adapter
            adapter = VectorProjectorAdapter(
                vector_store_dir=index_config.output_dir,
                config_cache=Path("config/tools/vector_projector.yaml"),
            )

            # Sync config (no-op for stub)
            adapter.sync_config()

            print_info("✓ Stub mode - no sync needed")

        exit(0)

    except Exception as e:
        print_error(f"Failed to sync config: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


@main.group()
def tools():
    """
    Manage tool adapters.

    List and validate tool adapters for external tools.

    Examples:

      # List available tool adapters
      lorch tools list

      # Validate meltano configuration
      lorch tools validate meltano
    """
    pass


@tools.command()
def list():
    """
    List available tool adapters.

    Shows all tool adapters available in lorch for orchestrating
    external tools (meltano, canonizer, vector-projector, etc.).
    """
    try:
        print_banner("Available Tool Adapters")

        adapters = [
            {
                "name": "meltano",
                "description": "Meltano extract/load tool",
                "status": "implemented",
            },
            {
                "name": "canonizer",
                "description": "Data canonization with JSONata transforms",
                "status": "implemented",
            },
            {
                "name": "vector-projector",
                "description": "Vector store indexing (stub mode)",
                "status": "stub",
            },
        ]

        for adapter in adapters:
            if adapter["status"] == "implemented":
                status_symbol = "✓"
            elif adapter["status"] == "stub":
                status_symbol = "◐"
            else:
                status_symbol = "○"
            status_text = adapter["status"].upper()

            print(f"\n{status_symbol} {adapter['name']}")
            print(f"  Description: {adapter['description']}")
            print(f"  Status: {status_text}")

        print()

        exit(0)

    except Exception as e:
        print_error(f"Failed to list tools: {e}")
        exit(1)


@tools.command()
@click.argument("tool", type=click.Choice(["meltano", "canonizer", "vector-projector"], case_sensitive=False))
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Custom configuration file",
)
@click.option(
    "--tap",
    help="Validate specific tap-target pair (requires --target)",
)
@click.option(
    "--target",
    help="Target for tap-target validation (requires --tap)",
)
def validate(tool, config, tap, target):
    """
    Validate tool configuration and setup.

    Checks that the tool is properly configured and ready to use.
    Can also validate specific tap-target pairs for meltano.

    Examples:

      # Validate meltano setup
      lorch tools validate meltano

      # Validate specific tap-target pair
      lorch tools validate meltano --tap tap-gmail--acct1-personal --target target-jsonl-chunked--gmail-ben-mensio
    """
    try:
        from lorch.tools.meltano import MeltanoAdapter
        from lorch.tools.canonizer import CanonizerAdapter
        from lorch.tools.vector_projector import VectorProjectorAdapter

        # Load pipeline configuration
        pipeline_config = load_config(config) if config else load_config()

        if tool == "meltano":
            # Get meltano repo path
            extract_config = pipeline_config.get_stage("extract")
            if not extract_config:
                print_error("Extract stage not configured")
                exit(1)

            # Create adapter
            adapter = MeltanoAdapter(
                meltano_dir=extract_config.repo_path,
                config_cache=Path("config/tools/meltano.yaml"),
            )

            print_banner(f"Validating {tool}")

            # General validation
            validation = adapter.validate()

            if validation["errors"]:
                print_error("Validation failed:")
                for error in validation["errors"]:
                    print(f"  ✗ {error}")
                exit(1)

            if validation["warnings"]:
                print_warning("Warnings:")
                for warning in validation["warnings"]:
                    print(f"  ⚠ {warning}")
                print()

            print_success("✓ General validation passed")

            # Task-specific validation
            if tap and target:
                print_info(f"\nValidating task: {tap} → {target}")

                task_validation = adapter.validate_task(tap, target)

                if task_validation["errors"]:
                    print_error("\nTask validation failed:")
                    for error in task_validation["errors"]:
                        print(f"  ✗ {error}")
                    exit(1)

                if task_validation["warnings"]:
                    print_warning("\nTask warnings:")
                    for warning in task_validation["warnings"]:
                        print(f"  ⚠ {warning}")

                print_success("\n✓ Task validation passed")

            elif tap or target:
                print_error("Both --tap and --target must be specified together")
                exit(1)

        elif tool == "canonizer":
            # Get canonizer config
            canonize_config = pipeline_config.get_stage("canonize")
            if not canonize_config:
                print_error("Canonize stage not configured")
                exit(1)

            # Create adapter
            adapter = CanonizerAdapter(
                canonizer_dir=canonize_config.repo_path,
                transform_registry=Path(canonize_config.get("transform_registry")),
                config_cache=Path("config/tools/canonizer.yaml"),
            )

            print_banner(f"Validating {tool}")

            # Validation
            validation = adapter.validate()

            if validation["errors"]:
                print_error("Validation failed:")
                for error in validation["errors"]:
                    print(f"  ✗ {error}")
                exit(1)

            print_success("✓ Canonizer validation passed")
            print_info(f"  Canonizer repo: {canonize_config.repo_path}")
            print_info(f"  Transform registry: {canonize_config.get('transform_registry')}")

        elif tool == "vector-projector":
            # Get vector-projector config
            index_config = pipeline_config.get_stage("index")
            if not index_config:
                print_error("Index stage not configured")
                exit(1)

            # Create adapter
            adapter = VectorProjectorAdapter(
                vector_store_dir=index_config.output_dir,
                config_cache=Path("config/tools/vector_projector.yaml"),
            )

            print_banner(f"Validating {tool}")

            # Validation (stub - always valid)
            validation = adapter.validate()

            print_warning("Vector-projector is in stub mode (file copying only)")
            print_success("✓ Stub validation passed")
            print_info(f"  Vector store: {index_config.output_dir}")
            print_info("\nFuture validation will check:")
            print("  - SQLite database")
            print("  - Inode storage directory structure")
            print("  - Query API availability")

        exit(0)

    except Exception as e:
        print_error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
