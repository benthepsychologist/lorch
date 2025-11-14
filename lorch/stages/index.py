"""
Index stage: Store canonical data in local document store.

Uses VectorProjectorAdapter (stub) to index canonical data.
Future: Will use full vector-projector with SQLite + inode storage.
"""

from pathlib import Path

from lorch.config import StageConfig
from lorch.stages.base import Stage, StageResult
from lorch.tools.vector_projector import VectorProjectorAdapter


class IndexStage(Stage):
    """
    Index stage using VectorProjectorAdapter (stub).

    Delegates indexing operations to VectorProjectorAdapter.
    Current: Adapter stub copies files.
    Future: Adapter will use SQLite + inode storage when vector-projector is ready.
    """

    def __init__(self, config: StageConfig, logger):
        """Initialize stage with VectorProjectorAdapter stub."""
        super().__init__(config, logger)

        # Initialize VectorProjectorAdapter (stub)
        vector_store_dir = self.config.output_dir

        self.adapter = VectorProjectorAdapter(
            vector_store_dir=vector_store_dir,
        )

    def validate(self) -> None:
        """Validate index stage prerequisites via adapter."""
        # Validate using adapter
        validation = self.adapter.validate()

        if not validation["valid"]:
            errors = "\n".join(validation["errors"])
            raise ValueError(f"Vector-projector validation failed:\n{errors}")

        # Validate output directory
        self._validate_output_dir()

        # Note: We don't validate input files here since they may not exist yet
        # (they're created by the canonize stage)

    def execute(self) -> StageResult:
        """Execute index stage via adapter (stub mode - copies files)."""
        input_dir = self.config.input_dir
        output_dir = self.config.output_dir

        self.logger.info(
            "Starting index stage (stub mode - copying files via adapter)",
            extra={
                "stage": self.name,
                "event": "index_started",
                "metadata": {"mode": "stub"},
            },
        )

        # Use adapter to copy files (stub implementation)
        result = self.adapter.copy_files(
            input_dir=input_dir,
            output_dir=output_dir,
            logger=self.logger,
        )

        files_copied = result["files_copied"]
        records_processed = result["records_processed"]

        if files_copied == 0:
            self.logger.warning(
                "No files found for indexing",
                extra={
                    "stage": self.name,
                    "event": "no_input_files",
                },
            )
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_seconds=0,
                records_processed=0,
                metadata={"mode": "stub", "files_processed": 0},
            )

        self.logger.info(
            f"Index stage completed: {files_copied} files, {records_processed} records",
            extra={
                "stage": self.name,
                "event": "index_completed",
                "metadata": {
                    "files": files_copied,
                    "records": records_processed,
                },
            },
        )

        # Convert output file paths to Path objects
        output_files = [Path(f) for f in result["output_files"]]

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_seconds=0,  # Will be set by base class
            records_processed=records_processed,
            output_files=output_files,
            metadata={
                "mode": "stub",
                "files_processed": files_copied,
                "manifest": result["manifest_file"],
            },
        )

    def cleanup(self) -> None:
        """Clean up temporary files if needed."""
        # No cleanup needed for stub mode
        pass
