"""Canonizer tool adapter for lorch."""

import gzip
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from lorch.tools.base import ToolAdapter


class CanonizerAdapter(ToolAdapter):
    """
    Adapter for canonizer transform tool.

    This adapter provides validation, configuration sync, and execution
    capabilities for canonizing raw data to canonical schemas using JSONata transforms.

    Supports vault structure with manifests, LATEST pointers, and gzip-compressed chunks.
    """

    def __init__(
        self,
        canonizer_dir: Path,
        transform_registry: Path,
        config_cache: Optional[Path] = None,
    ):
        """
        Initialize CanonizerAdapter.

        Args:
            canonizer_dir: Path to canonizer repo directory (e.g., /home/user/canonizer)
            transform_registry: Path to transform registry (e.g., /home/user/transforms)
            config_cache: Path to cached config file (defaults to config/tools/canonizer.yaml)
        """
        self.canonizer_dir = Path(canonizer_dir)
        self.transform_registry = Path(transform_registry)
        self.venv_path = self.canonizer_dir / ".venv"
        self.can_bin = self.venv_path / "bin" / "can"

        if config_cache is None:
            config_cache = Path("config/tools/canonizer.yaml")

        super().__init__(config_cache)

    def load_config(self) -> Dict[str, Any]:
        """
        Load tool configuration from the cached config file.

        Returns:
            Dictionary containing transforms configuration

        Raises:
            ValueError: If config file is invalid
        """
        if not self.config_path.exists():
            return {
                "transforms": {},
                "canonizer_dir": str(self.canonizer_dir),
                "transform_registry": str(self.transform_registry),
            }

        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                return config or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {self.config_path}: {e}")

    def sync_config(self) -> None:
        """
        Sync configuration from transform registry to lorch's cached config.

        Discovers all transforms in the registry and writes them to cache.

        Raises:
            FileNotFoundError: If transform registry doesn't exist
        """
        if not self.transform_registry.exists():
            raise FileNotFoundError(
                f"Transform registry not found: {self.transform_registry}"
            )

        # Discover all .meta.yaml files in transform registry
        transforms = {}

        for meta_file in self.transform_registry.rglob("*.meta.yaml"):
            try:
                with open(meta_file, "r") as f:
                    meta = yaml.safe_load(f)

                transform_name = meta.get("name", meta_file.stem.replace(".meta", ""))
                transform_path = str(meta_file.relative_to(self.transform_registry))

                transforms[transform_name] = {
                    "meta_file": transform_path,
                    "input_schema": meta.get("input_schema", "unknown"),
                    "output_schema": meta.get("output_schema", "unknown"),
                    "version": meta.get("version", "1.0.0"),
                }
            except Exception as e:
                # Skip invalid meta files
                continue

        # Write config cache
        config = {
            "canonizer_dir": str(self.canonizer_dir),
            "transform_registry": str(self.transform_registry),
            "transforms": transforms,
            "discovered_count": len(transforms),
        }

        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def validate(self) -> Dict[str, Any]:
        """
        Validate canonizer tool setup.

        Returns:
            Dict with 'valid': bool, 'errors': list

        Checks:
        - Canonizer repo exists
        - Virtual environment exists
        - Canonizer executable exists
        - Transform registry exists
        """
        errors = []

        # Check canonizer repo
        if not self.canonizer_dir.exists():
            errors.append(f"Canonizer repo not found: {self.canonizer_dir}")

        # Check venv
        if not self.venv_path.exists():
            errors.append(f"Canonizer venv not found: {self.venv_path}")

        # Check canonizer executable
        if not self.can_bin.exists():
            errors.append(f"Canonizer executable not found: {self.can_bin}")

        # Check transform registry
        if not self.transform_registry.exists():
            errors.append(f"Transform registry not found: {self.transform_registry}")

        return {"valid": len(errors) == 0, "errors": errors}

    def find_latest_manifests(
        self, vault_root: Path, source_path: str, logger=None
    ) -> List[Path]:
        """
        Find LATEST manifest.json files in vault for a given source path.

        Only processes manifests pointed to by LATEST.json markers in account
        directories. This ensures deterministic, idempotent canonization.

        Args:
            vault_root: Vault root directory
            source_path: Source path pattern (e.g., "email/gmail")
            logger: Optional logger for debug messages

        Returns:
            List of manifest.json file paths from LATEST runs only
        """
        manifests = []
        source_dir = vault_root / source_path

        if not source_dir.exists():
            return manifests

        # Find all account directories under this source
        # e.g., vault/email/gmail/ben-mensio/, vault/email/gmail/drben/, etc.
        for account_dir in source_dir.iterdir():
            if not account_dir.is_dir():
                continue

            # Look for LATEST.json marker
            latest_marker = account_dir / "LATEST.json"

            if not latest_marker.exists():
                if logger:
                    logger.debug(f"No LATEST.json found in {account_dir}, skipping")
                continue

            try:
                # Read LATEST.json to get dt and run_id
                with open(latest_marker, "r") as f:
                    latest_data = json.load(f)

                dt = latest_data.get("dt")
                run_id = latest_data.get("run_id")

                if not dt or not run_id:
                    if logger:
                        logger.warning(f"Invalid LATEST.json in {account_dir}")
                    continue

                # Build path to manifest
                manifest_path = (
                    account_dir / f"dt={dt}" / f"run_id={run_id}" / "manifest.json"
                )

                if not manifest_path.exists():
                    if logger:
                        logger.warning(
                            f"LATEST points to non-existent run: {manifest_path}"
                        )
                    continue

                manifests.append(manifest_path)

                if logger:
                    logger.debug(
                        f"Found LATEST manifest for {account_dir.name}: {dt}/{run_id}"
                    )

            except Exception as e:
                if logger:
                    logger.warning(f"Could not read LATEST.json in {account_dir}: {e}")

        return manifests

    def transform_from_manifest(
        self,
        manifest_path: Path,
        transform_name: str,
        output_dir: Path,
        logger=None,
    ) -> Dict[str, Any]:
        """
        Transform data from a vault manifest with LATEST pointer support.

        Processes all parts in the manifest, decompressing gzip chunks and
        piping through canonizer. Clears existing output file for idempotency.

        Args:
            manifest_path: Path to manifest.json
            transform_name: Transform name (e.g., "email/gmail_to_canonical_v1")
            output_dir: Output directory
            logger: Optional logger for status messages

        Returns:
            Dict with:
                - records: Total records processed
                - output_file: Path to output file
                - account: Account identifier
                - source: Source identifier

        Raises:
            FileNotFoundError: If transform or manifest parts not found
            RuntimeError: If canonizer fails
        """
        # Read manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        run_dir = manifest_path.parent
        parts = manifest.get("parts", [])

        if not parts:
            if logger:
                logger.warning(f"No parts found in manifest: {manifest_path}")
            return {
                "records": 0,
                "output_file": None,
                "account": manifest.get("account", "unknown"),
                "source": manifest.get("source", "unknown"),
            }

        if logger:
            logger.info(
                f"Processing {len(parts)} part(s) from manifest: {manifest_path.name}"
            )

        # Build transform metadata path
        transform_meta = self.transform_registry / f"{transform_name}.meta.yaml"

        if not transform_meta.exists():
            raise FileNotFoundError(f"Transform metadata not found: {transform_meta}")

        # Build output file path (per-account for idempotency)
        # Extract account from manifest
        account = manifest.get("account", "unknown")
        source = manifest.get("source", "unknown").replace(
            "/", "_"
        )  # email/gmail → email_gmail

        # Output: canonical/email_gmail/ben-mensio.jsonl
        account_output_dir = output_dir / source
        account_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = account_output_dir / f"{account}.jsonl"

        # Clear existing output for this account (idempotency: rebuild from LATEST)
        if output_file.exists():
            if logger:
                logger.debug(f"Clearing existing canonical output: {output_file}")
            output_file.unlink()

        # Process all parts in sequence
        total_records = 0

        for part in sorted(parts, key=lambda p: p.get("seq", 0)):
            part_path = run_dir / part["path"]

            if not part_path.exists():
                if logger:
                    logger.warning(f"Part file not found: {part_path}")
                continue

            # Transform this part
            records = self._transform_gzip_part(
                part_path=part_path,
                transform_meta=transform_meta,
                output_file=output_file,
                logger=logger,
            )

            total_records += records

        return {
            "records": total_records,
            "output_file": str(output_file),
            "account": account,
            "source": source,
        }

    def _transform_gzip_part(
        self,
        part_path: Path,
        transform_meta: Path,
        output_file: Path,
        logger=None,
    ) -> int:
        """
        Transform a single gzip-compressed JSONL part.

        Args:
            part_path: Path to part-NNN.jsonl.gz file
            transform_meta: Transform metadata file
            output_file: Output file path
            logger: Optional logger

        Returns:
            Number of records processed

        Raises:
            RuntimeError: If canonizer fails
        """
        # Build command
        command = [
            str(self.can_bin),
            "transform",
            "run",
            "--meta",
            str(transform_meta),
        ]

        if logger:
            logger.debug(f"Transforming part: {part_path.name}")

        # Decompress and stream to canonizer stdin
        with gzip.open(part_path, "rt") as gz_file:
            input_data = gz_file.read()

        # Execute canonizer
        result = subprocess.run(
            command,
            input=input_data,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = (
                f"Canonizer failed on {part_path.name} with exit code {result.returncode}"
            )
            if result.stderr:
                error_msg += f": {result.stderr[:500]}"
            raise RuntimeError(error_msg)

        # Append output to file
        with open(output_file, "a") as f:
            f.write(result.stdout)

        # Count records in output
        records = result.stdout.count("\n")

        return records

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute canonizer command (generic interface).

        For canonizer, use specific methods like transform_from_manifest() instead.
        """
        raise NotImplementedError(
            "Use transform_from_manifest() method for canonizer operations"
        )
