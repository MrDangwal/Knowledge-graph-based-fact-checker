from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Iterable, List

from app.config import settings

SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


class KBStorage:
    def __init__(self, base_dir: str | None = None) -> None:
        self.base_dir = Path(base_dir or settings.data_dir).resolve()
        self.kb_dir = self.base_dir / "kb_files"
        self.kb_dir.mkdir(parents=True, exist_ok=True)

    def list_files(self) -> List[Path]:
        return sorted(self.kb_dir.glob("*.txt"))

    def clear(self) -> None:
        for file_path in self.list_files():
            file_path.unlink(missing_ok=True)

    def save_files(self, files: Iterable[tuple[str, bytes]]) -> List[Path]:
        saved: List[Path] = []
        for filename, content in files:
            if filename.endswith(".zip"):
                saved.extend(self._extract_zip(content))
                continue
            if not filename.endswith(".txt"):
                continue
            safe_name = SAFE_FILENAME_RE.sub("_", os.path.basename(filename))
            if not safe_name:
                continue
            target = self.kb_dir / safe_name
            target.write_bytes(content)
            saved.append(target)
        return saved

    def _extract_zip(self, content: bytes) -> List[Path]:
        saved: List[Path] = []
        zip_path = self.kb_dir / "upload.zip"
        zip_path.write_bytes(content)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.is_dir():
                    continue
                if not member.filename.endswith(".txt"):
                    continue
                safe_name = SAFE_FILENAME_RE.sub("_", os.path.basename(member.filename))
                if not safe_name:
                    continue
                target = self.kb_dir / safe_name
                with zip_ref.open(member) as source:
                    target.write_bytes(source.read())
                saved.append(target)
        zip_path.unlink(missing_ok=True)
        return saved
