from pathlib import Path
from hashlib import sha256


class PromptRepository:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self, section: str) -> str:
        file_path = self.directory / f"{section}.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Prompt section does not exist: {file_path}"
            )

        return file_path.read_text(
            encoding="utf-8"
        ).strip()

    def fingerprint(self, section: str) -> str:
        """Stable content version for reports and production traces."""
        return sha256(self.load(section).encode("utf-8")).hexdigest()[:16]
