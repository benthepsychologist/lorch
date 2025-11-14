"""Vector-projector tool adapter for lorch (STUB IMPLEMENTATION)."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from lorch.tools.base import ToolAdapter
from lorch.utils import count_jsonl_records


class VectorProjectorAdapter(ToolAdapter):
    """
    Adapter for vector-projector indexing tool (STUB).

    Current: Minimal stub that copies files to destination.
    Future: Will provide SQLite indexing, inode-style storage, and query API.

    TODO: Implement when vector-projector tool is ready:
    - SQLite database with object metadata
    - Inode-style file storage (objects/ab/cd/abcd1234.json)
    - Query API for searching indexed objects
    - Per-type registries
    """

    def __init__(
        self,
        vector_store_dir: Path,
        config_cache: Optional[Path] = None,
    ):
        """
        Initialize VectorProjectorAdapter (stub).

        Args:
            vector_store_dir: Path to vector store directory (e.g., /home/user/phi-data/vector-store)
            config_cache: Path to cached config file (defaults to config/tools/vector_projector.yaml)
        """
        self.vector_store_dir = Path(vector_store_dir)

        if config_cache is None:
            config_cache = Path("config/tools/vector_projector.yaml")

        super().__init__(config_cache)

    def load_config(self) -> Dict[str, Any]:
        """
        Load tool configuration (stub - returns minimal config).

        Returns:
            Dictionary with minimal configuration
        """
        return {
            "mode": "stub",
            "vector_store_dir": str(self.vector_store_dir),
            "status": "not_implemented",
        }

    def sync_config(self) -> None:
        """
        Sync configuration (stub - no-op).

        Future: Will sync vector-projector configuration when tool is available.
        """
        # TODO: Implement when vector-projector has configuration
        pass

    def validate(self) -> Dict[str, Any]:
        """
        Validate vector-projector setup (stub - always valid).

        Returns:
            Dict with 'valid': True (stub mode always valid)

        Future: Will check:
        - Vector-projector installation
        - SQLite database
        - Directory permissions
        """
        # Stub mode is always valid (just copies files)
        return {"valid": True, "errors": [], "mode": "stub"}

    def copy_files(
        self,
        input_dir: Path,
        output_dir: Path,
        logger=None,
    ) -> Dict[str, Any]:
        """
        Copy canonical files to vector store (stub implementation).

        Args:
            input_dir: Input directory with canonical JSONL files
            output_dir: Output directory in vector store
            logger: Optional logger

        Returns:
            Dict with:
                - files_copied: Number of files copied
                - records_processed: Total records
                - output_files: List of output file paths
                - manifest_file: Path to manifest file

        Future: Will be replaced with:
        - index_objects() - Insert objects into SQLite + inode storage
        - query_objects() - Query indexed objects
        - get_object() - Retrieve object by ID
        """
        if logger:
            logger.info("Vector-projector stub: copying files to vector store")

        # Get input files (including subdirectories)
        input_files = list(input_dir.rglob("*.jsonl"))

        if not input_files:
            if logger:
                logger.warning(f"No JSONL files found in {input_dir}")
            return {
                "files_copied": 0,
                "records_processed": 0,
                "output_files": [],
                "manifest_file": None,
            }

        # Copy files
        total_records = 0
        output_files = []

        for input_file in input_files:
            try:
                # Generate output filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                # Preserve subdirectory structure
                relative_path = input_file.relative_to(input_dir)
                output_subdir = output_dir / relative_path.parent
                output_subdir.mkdir(parents=True, exist_ok=True)

                output_file = output_subdir / f"{input_file.stem}-{timestamp}{input_file.suffix}"

                # Copy file
                shutil.copy2(input_file, output_file)
                output_files.append(output_file)

                # Count records
                records = count_jsonl_records(output_file)
                total_records += records

                if logger:
                    logger.info(
                        f"Copied {records} records from {relative_path}"
                    )

            except Exception as e:
                if logger:
                    logger.error(f"Failed to copy {input_file}: {e}")

        # Create manifest
        manifest_file = output_dir / "manifest.json"
        self._create_manifest(manifest_file, output_files, total_records, output_dir)

        if logger:
            logger.info(
                f"Stub indexing complete: {len(output_files)} files, {total_records} records"
            )

        return {
            "files_copied": len(output_files),
            "records_processed": total_records,
            "output_files": [str(f) for f in output_files],
            "manifest_file": str(manifest_file),
        }

    def _create_manifest(
        self,
        manifest_file: Path,
        output_files: List[Path],
        total_records: int,
        output_dir: Path,
    ) -> None:
        """
        Create manifest file with indexing metadata.

        Args:
            manifest_file: Path to manifest file
            output_files: List of indexed files
            total_records: Total record count
            output_dir: Output directory for relative paths
        """
        manifest = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "mode": "stub",
            "total_records": total_records,
            "files": [
                {
                    "path": str(f.relative_to(output_dir)),
                    "size_bytes": f.stat().st_size,
                    "records": count_jsonl_records(f),
                }
                for f in output_files
            ],
        }

        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute vector-projector command (generic interface).

        For stub, use copy_files() method instead.
        """
        raise NotImplementedError(
            "Use copy_files() method for stub operations. "
            "Future: Will support index_objects(), query_objects(), get_object()"
        )
