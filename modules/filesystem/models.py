"""Data models for filesystem functionality."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileInfo:
    """Represents information about a file or directory."""

    name: str
    path: str
    is_directory: bool
    size: int
    modified_time: float
    created_time: float
    extension: Optional[str] = None

    @classmethod
    def from_path(cls, path: Path) -> "FileInfo":
        """Create FileInfo from a Path object."""
        stat = path.stat()
        extension = path.suffix if path.is_file() else None

        return cls(name=path.name, path=str(path), is_directory=path.is_dir(), size=stat.st_size, modified_time=stat.st_mtime, created_time=stat.st_ctime, extension=extension)

    @property
    def modified_time_formatted(self) -> str:
        """Return formatted modification time."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.modified_time))

    @property
    def created_time_formatted(self) -> str:
        """Return formatted creation time."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_time))

    @property
    def size_formatted(self) -> str:
        """Return human-readable file size."""
        if self.size < 1024:
            return f"{self.size} bytes"
        elif self.size < 1024 * 1024:
            return f"{self.size / 1024:.1f} KB"
        elif self.size < 1024 * 1024 * 1024:
            return f"{self.size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.size / (1024 * 1024 * 1024):.1f} GB"


@dataclass
class FileOperation:
    """Represents the result of a file operation."""

    success: bool
    message: str
    path: str
    operation_type: str  # "read", "write", "list", "delete"
    timestamp: float
    metadata: Dict[str, Any]

    @classmethod
    def success_result(cls, operation_type: str, path: str, message: str = "", metadata: Optional[Dict[str, Any]] = None) -> "FileOperation":
        """Create a successful operation result."""
        return cls(success=True, message=message, path=path, operation_type=operation_type, timestamp=time.time(), metadata=metadata or {})

    @classmethod
    def error_result(cls, operation_type: str, path: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> "FileOperation":
        """Create a failed operation result."""
        return cls(success=False, message=message, path=path, operation_type=operation_type, timestamp=time.time(), metadata=metadata or {})

    @property
    def timestamp_formatted(self) -> str:
        """Return formatted timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))


@dataclass
class DirectoryListing:
    """Represents a directory listing with files and subdirectories."""

    path: str
    files: List[FileInfo]
    directories: List[FileInfo]
    total_size: int
    listing_time: float

    @classmethod
    def create_now(cls, path: str, files: List[FileInfo], directories: List[FileInfo]) -> "DirectoryListing":
        """Create a DirectoryListing with current timestamp."""
        total_size = sum(file.size for file in files)

        return cls(path=path, files=files, directories=directories, total_size=total_size, listing_time=time.time())

    @property
    def total_files(self) -> int:
        """Return total number of files."""
        return len(self.files)

    @property
    def total_directories(self) -> int:
        """Return total number of directories."""
        return len(self.directories)

    @property
    def total_size_formatted(self) -> str:
        """Return human-readable total size."""
        if self.total_size < 1024:
            return f"{self.total_size} bytes"
        elif self.total_size < 1024 * 1024:
            return f"{self.total_size / 1024:.1f} KB"
        elif self.total_size < 1024 * 1024 * 1024:
            return f"{self.total_size / (1024 * 1024):.1f} MB"
        else:
            return f"{self.total_size / (1024 * 1024 * 1024):.1f} GB"

    @property
    def listing_time_formatted(self) -> str:
        """Return formatted listing time."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.listing_time))
