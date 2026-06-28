"""Tests for the benchmark fetch/verify/cache module (no network)."""

import hashlib
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest

import evaldata.loaders.benchmarks.fetch as fetch
from evaldata.loaders.benchmarks.fetch import (
    BenchmarkSource,
    cached_dataset_path,
    fetch_benchmark,
)


def _make_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE items (id INTEGER, name TEXT)")
    con.execute("INSERT INTO items VALUES (1, 'a')")
    con.commit()
    con.close()


def _build_benchmark_zip(
    tmp: Path,
    *,
    cases: int,
    databases_dirname: str,
    nested_databases_zip: bool,
    wrapper_name: str,
    db_name: str = "shop",
    corrupt: bool = False,
) -> Path:
    """Build a benchmark archive under `wrapper_name/`, with `dev.json` and databases.

    When `nested_databases_zip=True`, the databases directory is packed as a nested zip and a
    `__MACOSX/` dir is included (exercises both extraction and cleanup).
    """
    staging = tmp / "staging"
    wrapper = staging / wrapper_name
    db_dir = wrapper / databases_dirname / db_name
    db_dir.mkdir(parents=True)
    db_path = db_dir / f"{db_name}.sqlite"
    if corrupt:
        db_path.write_text("not a sqlite database")
    else:
        _make_db(db_path)

    records = [{"db_id": db_name, "question": f"q{i}", "evidence": "", "SQL": "SELECT 1"} for i in range(cases)]
    (wrapper / "dev.json").write_text(json.dumps(records))

    if nested_databases_zip:
        nested = wrapper / f"{databases_dirname}.zip"
        with zipfile.ZipFile(nested, "w") as zf:
            for file in (wrapper / databases_dirname).rglob("*"):
                zf.write(file, file.relative_to(wrapper))
        shutil.rmtree(wrapper / databases_dirname)

        # An ignorable macOS metadata folder at the top level.
        (wrapper / "__MACOSX").mkdir()
        (wrapper / "__MACOSX" / "junk").write_text("ignore me")

    archive = tmp / "dev.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for file in staging.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(staging))
    return archive


def _build_bird_zip(tmp: Path, *, cases: int, db_name: str = "shop", corrupt: bool = False) -> Path:
    """Build a BIRD-shaped dev.zip with a nested dev_databases.zip wrapper and a __MACOSX dir."""
    return _build_benchmark_zip(
        tmp,
        cases=cases,
        databases_dirname="dev_databases",
        nested_databases_zip=True,
        wrapper_name="dev_20240627",
        db_name=db_name,
        corrupt=corrupt,
    )


def _build_spider_zip(tmp: Path, *, cases: int, db_name: str = "shop", corrupt: bool = False) -> Path:
    """Build a Spider-shaped zip with a plain database/ dir under a spider_data/ wrapper."""
    return _build_benchmark_zip(
        tmp,
        cases=cases,
        databases_dirname="database",
        nested_databases_zip=False,
        wrapper_name="spider_data",
        db_name=db_name,
        corrupt=corrupt,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stand up a fake bird archive + monkeypatched _download, returning useful handles."""
    cache = tmp_path / "cache"
    archive = _build_bird_zip(tmp_path, cases=3)
    real_sha = _sha256(archive)

    calls = {"n": 0}

    def fake_download(url: str, dest: Path, *, progress: bool) -> str:
        calls["n"] += 1
        shutil.copyfile(archive, dest)
        return _sha256(dest)

    monkeypatch.setattr(fetch, "_download", fake_download)

    def install(*, archive_sha256: str | None, expected_cases: int = 3) -> None:
        fetch.SOURCES["bird"] = BenchmarkSource(
            name="bird",
            url="https://example.invalid/dev.zip",
            archive_sha256=archive_sha256,
            expected_cases=expected_cases,
            split="dev",
            license="CC BY-SA 4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            databases_dirname="dev_databases",
            nested_databases_zip=True,
        )

    original = fetch.SOURCES["bird"]
    yield {"cache": cache, "real_sha": real_sha, "calls": calls, "install": install, "archive": archive}
    fetch.SOURCES["bird"] = original


@pytest.fixture
def fake_spider_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Stand up a fake Spider archive + monkeypatched _download, returning useful handles."""
    cache = tmp_path / "cache"
    archive = _build_spider_zip(tmp_path, cases=3)
    real_sha = _sha256(archive)

    calls = {"n": 0}

    def fake_download(url: str, dest: Path, *, progress: bool) -> str:
        calls["n"] += 1
        shutil.copyfile(archive, dest)
        return _sha256(dest)

    monkeypatch.setattr(fetch, "_download", fake_download)

    def install(*, archive_sha256: str | None, expected_cases: int = 3) -> None:
        fetch.SOURCES["spider"] = BenchmarkSource(
            name="spider",
            url="https://example.invalid/spider_data.zip",
            archive_sha256=archive_sha256,
            expected_cases=expected_cases,
            split="dev",
            license="CC BY-SA 4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            databases_dirname="database",
            nested_databases_zip=False,
        )

    original = fetch.SOURCES["spider"]
    yield {"cache": cache, "real_sha": real_sha, "calls": calls, "install": install, "archive": archive}
    fetch.SOURCES["spider"] = original


@pytest.mark.unit
class TestFetchBenchmark:
    def test_pinned_hash_match_caches_and_returns_root(self, fake_source: dict) -> None:
        fake_source["install"](archive_sha256=fake_source["real_sha"])
        root = fetch_benchmark("bird", cache_dir=fake_source["cache"])
        assert (root / "dev.json").is_file()
        assert (root / "dev_databases").is_dir()
        assert (root / ".evaldata-meta.json").is_file()

    def test_pinned_hash_mismatch_raises_and_writes_no_cache(self, fake_source: dict) -> None:
        fake_source["install"](archive_sha256="0" * 64)
        with pytest.raises(RuntimeError, match="does not match the pinned"):
            fetch_benchmark("bird", cache_dir=fake_source["cache"])
        assert cached_dataset_path("bird", cache_dir=fake_source["cache"]) is None

    def test_unpinned_untrusted_refuses_before_downloading(self, fake_source: dict) -> None:
        fake_source["install"](archive_sha256=None)
        with pytest.raises(RuntimeError, match="not yet pinned"):
            fetch_benchmark("bird", trust=False, cache_dir=fake_source["cache"])
        # Fails fast: no download, no cache.
        assert fake_source["calls"]["n"] == 0
        assert cached_dataset_path("bird", cache_dir=fake_source["cache"]) is None

    def test_unpinned_trusted_caches(self, fake_source: dict) -> None:
        fake_source["install"](archive_sha256=None)
        root = fetch_benchmark("bird", trust=True, cache_dir=fake_source["cache"])
        assert (root / "dev.json").is_file()

    def test_expected_cases_mismatch_raises(self, fake_source: dict) -> None:
        fake_source["install"](archive_sha256=fake_source["real_sha"], expected_cases=99)
        with pytest.raises(RuntimeError, match="wrong dataset version"):
            fetch_benchmark("bird", cache_dir=fake_source["cache"])
        assert cached_dataset_path("bird", cache_dir=fake_source["cache"]) is None

    def test_corrupt_sqlite_fails_integrity(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = tmp_path / "cache"
        archive = _build_bird_zip(tmp_path, cases=3, corrupt=True)

        monkeypatch.setattr(
            fetch,
            "_download",
            lambda url, dest, *, progress: (shutil.copyfile(archive, dest), _sha256(dest))[1],
        )
        original = fetch.SOURCES["bird"]
        fetch.SOURCES["bird"] = BenchmarkSource(
            name="bird",
            url="https://example.invalid/dev.zip",
            archive_sha256=_sha256(archive),
            expected_cases=3,
            split="dev",
            license="CC BY-SA 4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            databases_dirname="dev_databases",
            nested_databases_zip=True,
        )
        try:
            with pytest.raises(RuntimeError, match="SQLite|integrity"):
                fetch_benchmark("bird", cache_dir=cache)
        finally:
            fetch.SOURCES["bird"] = original

    def test_valid_cache_skips_second_download(self, fake_source: dict) -> None:
        fake_source["install"](archive_sha256=fake_source["real_sha"])
        fetch_benchmark("bird", cache_dir=fake_source["cache"])
        assert fake_source["calls"]["n"] == 1
        fetch_benchmark("bird", cache_dir=fake_source["cache"])
        assert fake_source["calls"]["n"] == 1


@pytest.mark.unit
class TestFetchSpider:
    def test_pinned_hash_match_caches_and_returns_root(self, fake_spider_source: dict) -> None:
        fake_spider_source["install"](archive_sha256=fake_spider_source["real_sha"])
        root = fetch_benchmark("spider", cache_dir=fake_spider_source["cache"])
        assert (root / "dev.json").is_file()
        assert (root / "database").is_dir()
        assert (root / ".evaldata-meta.json").is_file()

    def test_unpinned_untrusted_refuses_before_downloading(self, fake_spider_source: dict) -> None:
        fake_spider_source["install"](archive_sha256=None)
        with pytest.raises(RuntimeError, match="not yet pinned"):
            fetch_benchmark("spider", trust=False, cache_dir=fake_spider_source["cache"])
        assert fake_spider_source["calls"]["n"] == 0
        assert cached_dataset_path("spider", cache_dir=fake_spider_source["cache"]) is None

    def test_expected_cases_mismatch_raises(self, fake_spider_source: dict) -> None:
        fake_spider_source["install"](archive_sha256=fake_spider_source["real_sha"], expected_cases=99)
        with pytest.raises(RuntimeError, match="wrong dataset version"):
            fetch_benchmark("spider", cache_dir=fake_spider_source["cache"])
        assert cached_dataset_path("spider", cache_dir=fake_spider_source["cache"]) is None

    def test_corrupt_sqlite_fails_integrity(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cache = tmp_path / "cache"
        archive = _build_spider_zip(tmp_path, cases=3, corrupt=True)

        monkeypatch.setattr(
            fetch,
            "_download",
            lambda url, dest, *, progress: (shutil.copyfile(archive, dest), _sha256(dest))[1],
        )
        original = fetch.SOURCES["spider"]
        fetch.SOURCES["spider"] = BenchmarkSource(
            name="spider",
            url="https://example.invalid/spider_data.zip",
            archive_sha256=_sha256(archive),
            expected_cases=3,
            split="dev",
            license="CC BY-SA 4.0",
            license_url="https://creativecommons.org/licenses/by-sa/4.0/",
            databases_dirname="database",
            nested_databases_zip=False,
        )
        try:
            with pytest.raises(RuntimeError, match="SQLite|integrity"):
                fetch_benchmark("spider", cache_dir=cache)
        finally:
            fetch.SOURCES["spider"] = original


@pytest.mark.unit
def test_non_zip_download_raises_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache"
    html = tmp_path / "quota.html"
    html.write_bytes(b"<!DOCTYPE html><html>Google Drive quota exceeded</html>")

    monkeypatch.setattr(
        fetch,
        "_download",
        lambda url, dest, *, progress: (shutil.copyfile(html, dest), _sha256(dest))[1],
    )
    original = fetch.SOURCES["spider"]
    fetch.SOURCES["spider"] = BenchmarkSource(
        name="spider",
        url="https://example.invalid/spider_data.zip",
        archive_sha256=_sha256(html),
        expected_cases=3,
        split="dev",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
        databases_dirname="database",
        nested_databases_zip=False,
    )
    try:
        with pytest.raises(RuntimeError, match="not a valid zip"):
            fetch_benchmark("spider", cache_dir=cache)
    finally:
        fetch.SOURCES["spider"] = original


@pytest.mark.unit
def test_unknown_name_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        fetch_benchmark("nope", cache_dir=tmp_path)


@pytest.mark.unit
def test_cached_dataset_path_none_when_absent(tmp_path: Path) -> None:
    assert cached_dataset_path("bird", cache_dir=tmp_path) is None
    assert cached_dataset_path("nope", cache_dir=tmp_path) is None
