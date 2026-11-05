"""Read files out of an export, whether it is a .zip or an unzipped folder.

Exports nest differently by year and language, so every parser looks files up
by *file name* anywhere inside, never by a fixed folder path.
"""

from __future__ import annotations

import fnmatch
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


class Bundle:
    """A read-only view over a zip file or a folder."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._zip: zipfile.ZipFile | None = None
        if self.path.is_file():
            if not zipfile.is_zipfile(self.path):
                raise ValueError(f"not a zip or folder: {self.path}")
            self._zip = zipfile.ZipFile(self.path)
            names = [n for n in self._zip.namelist() if not n.endswith("/")]
        else:
            names = [p.relative_to(self.path).as_posix()
                     for p in self.path.rglob("*") if p.is_file()]
        # macOS zips carry resource-fork junk under __MACOSX/ and ._ files.
        self.names = sorted(n for n in names
                            if not n.startswith("__MACOSX/")
                            and not PurePosixPath(n).name.startswith("._"))

    def find(self, pattern: str) -> list[str]:
        """Files whose base name matches a glob (case-insensitive) or compiled regex."""
        out = []
        for n in self.names:
            base = PurePosixPath(n).name
            if isinstance(pattern, re.Pattern):
                if pattern.fullmatch(base):
                    out.append(n)
            elif fnmatch.fnmatch(base.lower(), pattern.lower()):
                out.append(n)
        return out

    def read_bytes(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()

    def read_text(self, name: str) -> str:
        # utf-8-sig drops the byte-order mark some exports start with.
        return self.read_bytes(name).decode("utf-8-sig", errors="replace")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "Bundle":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_FORMATS = (
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S %Z",      # reddit: 2023-01-15 12:34:56 UTC
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",            # spotify basic endTime
    "%a %b %d %H:%M:%S %z %Y",   # twitter created_at
)


def parse_time(value) -> datetime | None:
    """Parse the timestamp styles these exports use. Always returns UTC-aware, or None."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, timezone.utc)
    s = str(value).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    for fmt in _FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def sort_key(item) -> tuple:
    """Oldest first, undated items last."""
    w = item.when
    return (w is None, w or datetime.min.replace(tzinfo=timezone.utc))
