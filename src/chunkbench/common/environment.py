"""Runtime environment fingerprinting."""

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_commit(project_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_fingerprint(project_root: Path) -> dict[str, Any]:
    """Return reproducibility-relevant runtime metadata."""
    dependencies = {}
    for package in ("numpy", "pandas", "PyYAML", "chunkbench"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "machine": platform.machine(),
        "git_commit": _git_commit(project_root),
        "dependencies": dependencies,
    }
