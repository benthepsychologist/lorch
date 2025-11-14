"""
Canonize stage: Transform source data to canonical format.

Uses canonizer CLI to apply JSONata transforms.
Supports vault structure with manifests and gzip-compressed chunks.
"""

import gzip
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

from lorch.config import StageConfig
from lorch.stages.base import Stage, StageResult
from lorch.utils import count_jsonl_records, validate_file_is_jsonl


class CanonizeStage(Stage):
    """
    Canonize stage using canonizer.

    Transforms source JSONL files to canonical format using JSONata transforms.
    """

    def validate(self) -> None:
        """Validate canonizer installation and transform registry."""
        # Check canonizer repo exists
        if not self.config.repo_path.exists():
            raise FileNotFoundError(
                f"Canonizer repo not found: {self.config.repo_path}"
            )

        # Check venv exists
        if not self.config.venv_path.exists():
            raise FileNotFoundError(
                f"Canonizer venv not found: {self.config.venv_path}"
            )

        # Check canonizer executable
        can_bin = self.config.venv_path / "bin" / "can"
        if not can_bin.exists():
            raise FileNotFoundError(
                f"Canonizer executable not found: {can_bin}"
            )

        # Check transform registry exists
        transform_registry = Path(self.config.get("transform_registry"))
        if not transform_registry.exists():
            raise FileNotFoundError(
                f"Transform registry not found: {transform_registry}"
            )

        # Validate vault directory exists (don't check for specific files)
        if not self.config.input_dir.exists():
            raise FileNotFoundError(f"Vault directory does not exist: {self.config.input_dir}")

        # Validate output directory
        self._validate_output_dir()

        # Validate mappings are configured
        mappings = self.config.get("mappings", [])
        if not mappings:
            raise ValueError("No transform mappings configured")

    def execute(self) -> StageResult:
        """Execute canonization transforms on vault data."""
        mappings = self.config.get("mappings", [])
        vault_root = self.config.input_dir
        output_dir = self.config.output_dir
        transform_registry = Path(self.config.get("transform_registry"))

        self.logger.info(
            f"Starting canonization with {len(mappings)} mappings from vault",
            extra={
                "stage": self.name,
                "event": "canonize_started",
                "metadata": {"mappings_count": len(mappings), "vault_root": str(vault_root)},
            },
        )

        total_records = 0
        output_files = []
        errors = []

        # Process each mapping
        for mapping in mappings:
            source_path = mapping["source_pattern"]  # e.g., "email/gmail"
            transform_name = mapping["transform"]
            output_name = mapping.get("output_name", "canonical")

            # Find manifests in vault matching source path
            manifests = self._find_manifests(vault_root, source_path)

            if not manifests:
                self.logger.warning(
                    f"No manifests found for source: {source_path}",
                    extra={
                        "stage": self.name,
                        "event": "no_manifests_found",
                        "metadata": {"source_path": source_path},
                    },
                )
                continue

            self.logger.info(
                f"Found {len(manifests)} manifest(s) for {source_path}",
                extra={
                    "stage": self.name,
                    "event": "manifests_discovered",
                    "metadata": {
                        "source_path": source_path,
                        "manifest_count": len(manifests),
                    },
                },
            )

            # Process each manifest
            for manifest_path in manifests:
                try:
                    records_processed = self._transform_from_manifest(
                        manifest_path=manifest_path,
                        transform_name=transform_name,
                        transform_registry=transform_registry,
                        output_dir=output_dir,
                        output_name=output_name,
                    )

                    total_records += records_processed
                    self.logger.info(
                        f"Transformed {records_processed} records from {manifest_path.parent}",
                        extra={
                            "stage": self.name,
                            "event": "manifest_transformed",
                            "metadata": {
                                "manifest": str(manifest_path),
                                "records": records_processed,
                            },
                        },
                    )

                except Exception as e:
                    error_msg = f"Failed to transform manifest {manifest_path}: {e}"
                    errors.append(error_msg)
                    self.logger.error(
                        error_msg,
                        extra={
                            "stage": self.name,
                            "event": "transform_error",
                            "metadata": {"manifest": str(manifest_path), "error": str(e)},
                        },
                    )

        # Collect output files
        output_files = list(output_dir.glob("*.jsonl"))

        # Check if we had errors
        if errors and not output_files:
            # All transforms failed
            return StageResult(
                stage_name=self.name,
                success=False,
                duration_seconds=0,
                error_message=f"{len(errors)} transform(s) failed: {errors[0]}",
            )

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,  # Will be set by base class
            records_processed=total_records,
            output_files=output_files,
            metadata={
                "transform_registry": str(transform_registry),
                "mappings_applied": len(mappings),
                "manifests_processed": total_records,
                "errors": len(errors),
            },
        )

    def _transform_file(
        self,
        input_file: Path,
        transform_name: str,
        transform_registry: Path,
        output_dir: Path,
        output_name: str,
    ) -> int:
        """
        Transform a single JSONL file.

        Args:
            input_file: Input JSONL file
            transform_name: Transform name (e.g., "email/gmail_to_canonical_v1")
            transform_registry: Transform registry directory
            output_dir: Output directory
            output_name: Output file base name

        Returns:
            Number of records processed

        Raises:
            Exception: If transformation fails
        """
        # Build transform metadata path
        transform_meta = transform_registry / f"{transform_name}.meta.yaml"

        if not transform_meta.exists():
            raise FileNotFoundError(f"Transform metadata not found: {transform_meta}")

        # Build output file path
        from datetime import datetime
        date_str = datetime.now().strftime("%Y%m%d")
        output_file = output_dir / f"{output_name}-{date_str}.jsonl"

        # Build command
        can_bin = self.config.venv_path / "bin" / "can"
        command = [
            str(can_bin),
            "transform",
            "run",
            "--meta",
            str(transform_meta),
            "--input",
            str(input_file),
        ]

        self.logger.debug(f"Executing: {' '.join(command)}")

        # Execute canonizer
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_msg = f"Canonizer failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr[:500]}"
            raise RuntimeError(error_msg)

        # Write output to file (append mode for multiple inputs)
        with open(output_file, "a") as f:
            f.write(result.stdout)

        # Count records in output
        records = result.stdout.count("\n")

        # Validate output is valid JSONL
        if not validate_file_is_jsonl(output_file):
            self.logger.warning(
                f"Output file {output_file} contains invalid JSONL",
                extra={
                    "stage": self.name,
                    "event": "invalid_jsonl",
                    "metadata": {"file": str(output_file)},
                },
            )

        return records

    def _find_manifests(self, vault_root: Path, source_path: str) -> List[Path]:
        """
        Find LATEST manifest.json files in vault for a given source path.

        Only processes manifests pointed to by LATEST.json markers in account
        directories. This ensures deterministic, idempotent canonization.

        Args:
            vault_root: Vault root directory
            source_path: Source path pattern (e.g., "email/gmail")

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
                self.logger.debug(f"No LATEST.json found in {account_dir}, skipping")
                continue

            try:
                # Read LATEST.json to get dt and run_id
                with open(latest_marker, "r") as f:
                    latest_data = json.load(f)

                dt = latest_data.get("dt")
                run_id = latest_data.get("run_id")

                if not dt or not run_id:
                    self.logger.warning(f"Invalid LATEST.json in {account_dir}")
                    continue

                # Build path to manifest
                manifest_path = account_dir / f"dt={dt}" / f"run_id={run_id}" / "manifest.json"

                if not manifest_path.exists():
                    self.logger.warning(
                        f"LATEST points to non-existent run: {manifest_path}"
                    )
                    continue

                manifests.append(manifest_path)

                self.logger.debug(
                    f"Found LATEST manifest for {account_dir.name}: {dt}/{run_id}"
                )

            except Exception as e:
                self.logger.warning(
                    f"Could not read LATEST.json in {account_dir}: {e}"
                )

        return manifests

    def _transform_from_manifest(
        self,
        manifest_path: Path,
        transform_name: str,
        transform_registry: Path,
        output_dir: Path,
        output_name: str,
    ) -> int:
        """
        Transform data from a vault manifest.

        Args:
            manifest_path: Path to manifest.json
            transform_name: Transform name (e.g., "email/gmail_to_canonical_v1")
            transform_registry: Transform registry directory
            output_dir: Output directory
            output_name: Output file base name

        Returns:
            Number of records processed
        """
        # Read manifest
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        run_dir = manifest_path.parent
        parts = manifest.get("parts", [])

        if not parts:
            self.logger.warning(f"No parts found in manifest: {manifest_path}")
            return 0

        self.logger.info(
            f"Processing {len(parts)} part(s) from manifest",
            extra={
                "stage": self.name,
                "event": "manifest_parts_found",
                "metadata": {
                    "manifest": str(manifest_path),
                    "parts_count": len(parts),
                    "total_records": manifest.get("totals", {}).get("records", 0),
                },
            },
        )

        # Build transform metadata path
        transform_meta = transform_registry / f"{transform_name}.meta.yaml"

        if not transform_meta.exists():
            raise FileNotFoundError(f"Transform metadata not found: {transform_meta}")

        # Build output file path (per-account for idempotency)
        # Extract account from manifest
        account = manifest.get("account", "unknown")
        source = manifest.get("source", "unknown").replace("/", "_")  # email/gmail → email_gmail

        # Output: canonical/email_gmail/ben-mensio.jsonl
        account_output_dir = output_dir / source
        account_output_dir.mkdir(parents=True, exist_ok=True)

        output_file = account_output_dir / f"{account}.jsonl"

        # Clear existing output for this account (idempotency: rebuild from LATEST)
        if output_file.exists():
            self.logger.debug(f"Clearing existing canonical output: {output_file}")
            output_file.unlink()

        # Process all parts in sequence
        total_records = 0

        for part in sorted(parts, key=lambda p: p.get("seq", 0)):
            part_path = run_dir / part["path"]

            if not part_path.exists():
                self.logger.warning(f"Part file not found: {part_path}")
                continue

            # Transform this part
            records = self._transform_gzip_part(
                part_path=part_path,
                transform_meta=transform_meta,
                output_file=output_file,
            )

            total_records += records

        return total_records

    def _transform_gzip_part(
        self,
        part_path: Path,
        transform_meta: Path,
        output_file: Path,
    ) -> int:
        """
        Transform a single gzip-compressed JSONL part.

        Args:
            part_path: Path to part-NNN.jsonl.gz file
            transform_meta: Transform metadata file
            output_file: Output file path

        Returns:
            Number of records processed
        """
        # Build command
        can_bin = self.config.venv_path / "bin" / "can"
        command = [
            str(can_bin),
            "transform",
            "run",
            "--meta",
            str(transform_meta),
        ]

        self.logger.debug(f"Transforming part: {part_path.name}")

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
            error_msg = f"Canonizer failed on {part_path.name} with exit code {result.returncode}"
            if result.stderr:
                error_msg += f": {result.stderr[:500]}"
            raise RuntimeError(error_msg)

        # Append output to file
        with open(output_file, "a") as f:
            f.write(result.stdout)

        # Count records in output
        records = result.stdout.count("\n")

        return records

    def cleanup(self) -> None:
        """Clean up temporary files if needed."""
        # No cleanup needed for canonize stage
        pass
