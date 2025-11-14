"""
Canonize stage: Transform source data to canonical format.

Uses CanonizerAdapter to apply JSONata transforms with vault LATEST pointer support.
"""

from pathlib import Path

from lorch.config import StageConfig
from lorch.stages.base import Stage, StageResult
from lorch.tools.canonizer import CanonizerAdapter


class CanonizeStage(Stage):
    """
    Canonize stage using CanonizerAdapter.

    Transforms vault data to canonical format using JSONata transforms.
    Delegates transform execution to CanonizerAdapter.
    """

    def __init__(self, config: StageConfig, logger):
        """Initialize stage with CanonizerAdapter."""
        super().__init__(config, logger)

        # Initialize CanonizerAdapter
        canonizer_dir = self.config.repo_path
        transform_registry = Path(self.config.get("transform_registry"))

        self.adapter = CanonizerAdapter(
            canonizer_dir=canonizer_dir,
            transform_registry=transform_registry,
        )

    def validate(self) -> None:
        """Validate canonizer installation via adapter."""
        # Validate using adapter
        validation = self.adapter.validate()

        if not validation["valid"]:
            errors = "\n".join(validation["errors"])
            raise ValueError(f"Canonizer validation failed:\n{errors}")

        # Validate vault directory exists
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Vault directory does not exist: {self.config.input_dir}"
            )

        # Validate output directory
        self._validate_output_dir()

        # Validate mappings are configured
        mappings = self.config.get("mappings", [])
        if not mappings:
            raise ValueError("No transform mappings configured")

    def execute(self) -> StageResult:
        """Execute canonization transforms on vault data via adapter."""
        mappings = self.config.get("mappings", [])
        vault_root = self.config.input_dir
        output_dir = self.config.output_dir

        self.logger.info(
            f"Starting canonization with {len(mappings)} mappings from vault",
            extra={
                "stage": self.name,
                "event": "canonize_started",
                "metadata": {
                    "mappings_count": len(mappings),
                    "vault_root": str(vault_root),
                },
            },
        )

        total_records = 0
        output_files = []
        errors = []

        # Process each mapping
        for mapping in mappings:
            source_path = mapping["source_pattern"]  # e.g., "email/gmail"
            transform_name = mapping["transform"]

            # Find LATEST manifests using adapter
            manifests = self.adapter.find_latest_manifests(
                vault_root=vault_root, source_path=source_path, logger=self.logger
            )

            if not manifests:
                self.logger.warning(
                    f"No LATEST manifests found for source: {source_path}",
                    extra={
                        "stage": self.name,
                        "event": "no_manifests_found",
                        "metadata": {"source_path": source_path},
                    },
                )
                continue

            self.logger.info(
                f"Found {len(manifests)} LATEST manifest(s) for {source_path}",
                extra={
                    "stage": self.name,
                    "event": "manifests_discovered",
                    "metadata": {
                        "source_path": source_path,
                        "manifest_count": len(manifests),
                    },
                },
            )

            # Process each manifest using adapter
            for manifest_path in manifests:
                try:
                    result = self.adapter.transform_from_manifest(
                        manifest_path=manifest_path,
                        transform_name=transform_name,
                        output_dir=output_dir,
                        logger=self.logger,
                    )

                    records_processed = result["records"]
                    total_records += records_processed

                    if result["output_file"]:
                        output_files.append(result["output_file"])

                    self.logger.info(
                        f"Transformed {records_processed} records from {result['account']}",
                        extra={
                            "stage": self.name,
                            "event": "manifest_transformed",
                            "metadata": {
                                "manifest": str(manifest_path),
                                "account": result["account"],
                                "records": records_processed,
                                "output_file": result["output_file"],
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
                            "metadata": {
                                "manifest": str(manifest_path),
                                "error": str(e),
                            },
                        },
                    )

        # Check if we had errors
        if errors and total_records == 0:
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
                "transform_registry": str(self.adapter.transform_registry),
                "mappings_applied": len(mappings),
                "errors": len(errors),
            },
        )

    def cleanup(self) -> None:
        """Clean up temporary files if needed."""
        # No cleanup needed for canonize stage
        pass
