"""Release integrity: Git snapshot, package identity, and notebook imports."""
import json
from pathlib import Path
import subprocess
import tomllib
import zipfile

import pytest

from scripts.package_delivery import package
from xiyuan_mvp import __version__


def test_single_version_definition():
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["dynamic"] == ["version"]
    assert config["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "xiyuan_mvp.__version__"
    result = subprocess.run(
        [__import__("sys").executable, "-m", "xiyuan_mvp.cli", "--version"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == __version__


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True)
    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("config", "user.name", "Release Test")
    git("config", "user.email", "release-test@example.invalid")
    (root / "source.py").write_bytes(b"print('frozen')\n")
    git("add", "source.py")
    git("commit", "-qm", "snapshot")
    return root, git


def test_source_archive_is_only_committed_content(repo, tmp_path):
    root, git = repo
    (root / "private.json").write_text("not for delivery", encoding="utf-8")
    output = tmp_path / "release.zip"
    result = package(output, root=root)
    with zipfile.ZipFile(output) as archive:
        assert "private.json" not in archive.namelist()
        assert archive.read("source.py") == b"print('frozen')\n"
        assert json.loads(archive.read("SOURCE_COMMIT.json"))["commit"] == git("rev-parse", "HEAD").strip()
        assert json.loads(archive.read("SOURCE_MANIFEST.json"))["source.py"]
    assert result["files"] == 1
    with pytest.raises(FileExistsError):
        package(output, root=root)


def test_source_archive_rejects_uncommitted_tracked_changes(repo, tmp_path):
    root, _ = repo
    (root / "source.py").write_text("uncommitted", encoding="utf-8")
    with pytest.raises(subprocess.CalledProcessError):
        package(tmp_path / "dirty.zip", root=root)
