"""
Utility functions for lorch pipeline.

Includes logging, retries, validation, and file operations.
"""

import json
import logging
import os
import stat
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


# Global console for pretty output
console = Console()


def setup_logging(log_file: Path, log_level: str = "INFO", log_format: str = "structured", console_output: bool = True) -> logging.Logger:
    """
    Set up logging for pipeline execution.

    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_format: "structured" (JSON) or "pretty" (human-readable)
        console_output: Also log to console

    Returns:
        Configured logger
    """
    # Create log directory if needed
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Set permissions on log directory
    log_file.parent.chmod(0o700)

    # Create logger
    logger = logging.getLogger("lorch")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []  # Clear existing handlers

    # File handler
    if log_format == "structured":
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(StructuredFormatter())
    else:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    logger.addHandler(file_handler)

    # Console handler
    if console_output:
        if log_format == "pretty":
            console_handler = RichHandler(rich_tracebacks=True, show_time=False)
        else:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(levelname)s: %(message)s")
            )

        logger.addHandler(console_handler)

    return logger


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "stage"):
            log_data["stage"] = record.stage
        if hasattr(record, "event"):
            log_data["event"] = record.event
        if hasattr(record, "metadata"):
            log_data["metadata"] = record.metadata

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    backoff_seconds: int = 60,
    backoff_multiplier: float = 2.0,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        func: Function to retry
        max_attempts: Maximum number of attempts
        backoff_seconds: Initial backoff time in seconds
        backoff_multiplier: Multiplier for each retry
        logger: Logger for retry messages

    Returns:
        Result of successful function call

    Raises:
        Exception: If all retries exhausted
    """
    attempt = 1
    wait_time = backoff_seconds

    while attempt <= max_attempts:
        try:
            if logger:
                logger.info(f"Attempt {attempt}/{max_attempts}")
            return func()

        except Exception as e:
            if attempt == max_attempts:
                if logger:
                    logger.error(f"All {max_attempts} attempts failed: {e}")
                raise

            if logger:
                logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {wait_time}s..."
                )

            time.sleep(wait_time)
            wait_time *= backoff_multiplier
            attempt += 1


def validate_phi_permissions(path: Path, required_mode: str = "700") -> None:
    """
    Validate that PHI directory has correct permissions.

    Args:
        path: Path to PHI directory
        required_mode: Required permission mode (e.g., "700")

    Raises:
        PermissionError: If permissions are incorrect
    """
    if not path.exists():
        raise FileNotFoundError(f"PHI directory does not exist: {path}")

    stat_info = path.stat()
    current_mode = stat.S_IMODE(stat_info.st_mode)
    required_mode_int = int(required_mode, 8)

    if current_mode != required_mode_int:
        raise PermissionError(
            f"PHI directory {path} has insecure permissions {oct(current_mode)}. "
            f"Expected {oct(required_mode_int)}. Run: chmod {required_mode} {path}"
        )


def validate_file_is_jsonl(file_path: Path) -> bool:
    """
    Validate that a file is valid JSONL format.

    Args:
        file_path: Path to JSONL file

    Returns:
        True if valid, False otherwise
    """
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                json.loads(line)  # Will raise if invalid JSON
        return True
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False


def count_jsonl_records(file_path: Path) -> int:
    """
    Count records in a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        Number of records (non-empty lines)
    """
    count = 0
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def sanitize_error_message(error: Exception, max_length: int = 500) -> str:
    """
    Sanitize error message to avoid logging PHI.

    Args:
        error: Exception to sanitize
        max_length: Maximum message length

    Returns:
        Sanitized error message
    """
    message = str(error)

    # Truncate if too long
    if len(message) > max_length:
        message = message[:max_length] + "..."

    # Remove potential email addresses
    import re
    message = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", message)

    # Remove potential phone numbers
    message = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", message)

    return message


def ensure_directory_permissions(directory: Path, mode: int = 0o700) -> None:
    """
    Ensure directory exists with correct permissions.

    Args:
        directory: Directory path
        mode: Permission mode (default: 0o700)
    """
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(mode)


def get_file_checksum(file_path: Path) -> str:
    """
    Calculate SHA256 checksum of file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of SHA256 checksum
    """
    import hashlib

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "1m 23s", "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours = minutes // 60
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def print_banner(title: str) -> None:
    """
    Print a banner to console.

    Args:
        title: Banner title
    """
    console.rule(f"[bold blue]{title}[/bold blue]")


def print_success(message: str) -> None:
    """Print success message to console."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str) -> None:
    """Print error message to console."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str) -> None:
    """Print warning message to console."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str) -> None:
    """Print info message to console."""
    console.print(f"[bold cyan]ℹ[/bold cyan] {message}")


def parse_relative_date(relative: str) -> datetime:
    """
    Parse relative date string to absolute datetime.

    Args:
        relative: Relative date string like "7d", "2w", "1m", "3y"
                  - d = days
                  - w = weeks
                  - m = months (30 days)
                  - y = years (365 days)

    Returns:
        datetime object representing the date

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_relative_date("7d")  # 7 days ago
        >>> parse_relative_date("2w")  # 2 weeks ago
        >>> parse_relative_date("1m")  # 1 month (30 days) ago
    """
    from datetime import timedelta
    import re

    match = re.match(r"^(\d+)([dwmy])$", relative.lower())
    if not match:
        raise ValueError(
            f"Invalid relative date format: '{relative}'. "
            "Expected format: <number><unit> (e.g., '7d', '2w', '1m', '3y')"
        )

    value = int(match.group(1))
    unit = match.group(2)

    now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    if unit == "d":
        return now - timedelta(days=value)
    elif unit == "w":
        return now - timedelta(weeks=value)
    elif unit == "m":
        return now - timedelta(days=value * 30)  # Approximate month
    elif unit == "y":
        return now - timedelta(days=value * 365)  # Approximate year
    else:
        raise ValueError(f"Unknown unit: {unit}")


def parse_date_string(date_str: str) -> datetime:
    """
    Parse date string to datetime.

    Accepts:
    - Absolute dates: "2025-11-01", "2025/11/01"
    - Relative dates: "7d", "2w", "1m"

    Args:
        date_str: Date string

    Returns:
        datetime object

    Raises:
        ValueError: If format is invalid
    """
    import re

    # Check if it's a relative date
    if re.match(r"^\d+[dwmy]$", date_str.lower()):
        return parse_relative_date(date_str)

    # Try parsing absolute date formats
    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"]:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    raise ValueError(
        f"Invalid date format: '{date_str}'. "
        "Expected: YYYY-MM-DD, YYYY/MM/DD, or relative (e.g., '7d')"
    )


def format_date_for_provider(
    provider: str,
    from_dt: Optional[datetime] = None,
    to_dt: Optional[datetime] = None,
) -> str:
    """
    Format date range as query string for different providers.

    Args:
        provider: Provider type ("gmail", "exchange", "dataverse")
        from_dt: Start date (inclusive)
        to_dt: End date (inclusive)

    Returns:
        Provider-specific query string

    Examples:
        Gmail:
            - after:2025/11/01
            - after:2025/11/01 before:2025/11/15

        Exchange:
            - receivedDateTime ge 2025-11-01T00:00:00Z
            - receivedDateTime ge 2025-11-01T00:00:00Z and receivedDateTime le 2025-11-15T23:59:59Z

        Dataverse:
            - modifiedon ge 2025-11-01
            - modifiedon ge 2025-11-01 and modifiedon le 2025-11-15
    """
    if not from_dt and not to_dt:
        return ""

    provider = provider.lower()

    if provider == "gmail":
        parts = []
        if from_dt:
            parts.append(f"after:{from_dt.strftime('%Y/%m/%d')}")
        if to_dt:
            parts.append(f"before:{to_dt.strftime('%Y/%m/%d')}")
        return " ".join(parts)

    elif provider == "exchange" or provider == "msgraph":
        parts = []
        if from_dt:
            parts.append(f"receivedDateTime ge {from_dt.strftime('%Y-%m-%dT00:00:00Z')}")
        if to_dt:
            # End of day
            to_dt_end = to_dt.replace(hour=23, minute=59, second=59)
            parts.append(f"receivedDateTime le {to_dt_end.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        return " and ".join(parts)

    elif provider == "dataverse":
        parts = []
        if from_dt:
            parts.append(f"modifiedon ge {from_dt.strftime('%Y-%m-%d')}")
        if to_dt:
            parts.append(f"modifiedon le {to_dt.strftime('%Y-%m-%d')}")
        return " and ".join(parts)

    else:
        # Generic format (ISO date)
        parts = []
        if from_dt:
            parts.append(f"from:{from_dt.strftime('%Y-%m-%d')}")
        if to_dt:
            parts.append(f"to:{to_dt.strftime('%Y-%m-%d')}")
        return " ".join(parts)


def detect_provider_from_tap_name(tap_name: str) -> str:
    """
    Detect provider type from tap name.

    Args:
        tap_name: Tap name (e.g., "tap-gmail--acct1-personal", "tap-msgraph-mail--ben-mensio")

    Returns:
        Provider name ("gmail", "exchange", "dataverse", "unknown")
    """
    tap_lower = tap_name.lower()

    if "gmail" in tap_lower:
        return "gmail"
    elif "msgraph" in tap_lower or "exchange" in tap_lower:
        return "exchange"
    elif "dataverse" in tap_lower:
        return "dataverse"
    else:
        return "unknown"
