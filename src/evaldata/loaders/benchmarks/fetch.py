"""Download, verify, and cache text-to-SQL benchmark archives for the loaders to read."""

import hashlib
import json
import shutil
import sqlite3
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

_META_NAME = ".evaldata-meta.json"
_USER_AGENT = "evaldata"
_DOWNLOAD_TIMEOUT_SECONDS = 600
_CHUNK_BYTES = 1 << 20


@dataclass(frozen=True)
class BenchmarkSource:
    """A downloadable benchmark archive and the invariants used to verify it.

    Attributes:
        name: The dataset key (e.g. `"bird"`).
        url: The HTTPS URL of the official archive. May be mutable upstream, which is why
            `archive_sha256` and `expected_cases` are checked after download.
        archive_sha256: The pinned SHA-256 of the archive bytes, or `None` when not yet
            pinned (the first fetch must pass `trust=True` to accept and reveal the hash).
        expected_cases: The number of records the dataset's split file must contain.
        split: The split file stem the loader reads (e.g. `"dev"`).
        license: The dataset license identifier.
        license_url: A URL describing the license.
    """

    name: str
    url: str
    archive_sha256: str | None
    expected_cases: int
    split: str
    license: str
    license_url: str


SOURCES: dict[str, BenchmarkSource] = {
    "bird": BenchmarkSource(
        name="bird",
        url="https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        archive_sha256="cdd6d19faeb45a23970b98d3ef6c40a87987c95459c2cf12076897a60cf5a630",
        expected_cases=1534,
        split="dev",
        license="CC BY-SA 4.0",
        license_url="https://creativecommons.org/licenses/by-sa/4.0/",
    ),
}


def cache_root(cache_dir: Path | None = None) -> Path:
    """Resolve the base cache directory.

    Args:
        cache_dir: An explicit cache directory; takes precedence over everything else.

    Returns:
        `cache_dir` if given, else `$EVALDATA_CACHE_DIR` if set, else the per-user cache
        directory for `evaldata`.

    Raises:
        RuntimeError: If `platformdirs` is needed but not installed.
    """
    import os

    if cache_dir is not None:
        return Path(cache_dir)
    env = os.environ.get("EVALDATA_CACHE_DIR")
    if env:
        return Path(env)
    try:
        import platformdirs
    except ImportError as e:
        msg = "platformdirs is required to locate the cache; install evaldata[benchmarks]"
        raise RuntimeError(msg) from e
    return Path(platformdirs.user_cache_dir("evaldata"))


def _datasets_dir(name: str, *, cache_dir: Path | None) -> Path:
    """Return the directory holding all cached copies of dataset `name`."""
    return cache_root(cache_dir) / "datasets" / name


def _is_valid_cache(root: Path, source: BenchmarkSource) -> bool:
    """Whether `root` is a complete cached copy of `source`'s dataset.

    Args:
        root: A candidate cached dataset root.
        source: The benchmark source the copy should satisfy.

    Returns:
        `True` if `root` holds both the cache marker and the split JSON.
    """
    return (root / _META_NAME).is_file() and (root / f"{source.split}.json").is_file()


def cached_dataset_path(name: str, *, cache_dir: Path | None = None) -> Path | None:
    """Return the cached dataset root for `name`, if a valid copy exists.

    Args:
        name: The dataset key.
        cache_dir: An explicit cache directory to look in, else the default cache root.

    Returns:
        The normalized dataset root (the directory containing the split JSON) if a valid
        cached copy exists, else `None`. Unknown names also return `None`.
    """
    source = SOURCES.get(name)
    if source is None:
        return None
    parent = _datasets_dir(name, cache_dir=cache_dir)
    if not parent.is_dir():
        return None
    for child in sorted(parent.iterdir()):
        if child.is_dir() and _is_valid_cache(child, source):
            return child
    return None


def _download(url: str, dest: Path, *, progress: bool) -> str:
    """Stream `url` to `dest`, returning the hex SHA-256 of the bytes written.

    Args:
        url: The HTTPS URL to download.
        dest: The file path to write the archive to.
        progress: Whether to render a progress bar on a TTY.

    Returns:
        The hex SHA-256 digest of the downloaded bytes.
    """
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 - https only, fixed sources
    with urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        total = int(response.headers.get("Content-Length", 0)) or None
        bar = _progress_bar(total) if progress else None
        try:
            with dest.open("wb") as out:
                for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                    out.write(chunk)
                    digest.update(chunk)
                    if bar is not None:
                        bar[0].update(bar[1], advance=len(chunk))
        finally:
            if bar is not None:
                bar[0].stop()
    return digest.hexdigest()


def _progress_bar(total: int | None):  # noqa: ANN202 - internal, returns (Progress, TaskID) or None
    """Start a `rich` download progress bar on a TTY, or return `None`.

    Args:
        total: The total byte count, or `None` when the response has no `Content-Length`.

    Returns:
        A `(Progress, TaskID)` pair on a TTY, else `None`.
    """
    from rich.console import Console
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn

    console = Console()
    if not console.is_terminal:
        return None
    bar = Progress(TextColumn("downloading"), BarColumn(), DownloadColumn(), console=console)
    bar.start()
    return bar, bar.add_task("download", total=total)


def _verify_hash(source: BenchmarkSource, computed: str, temp: Path, *, trust: bool) -> None:
    """Check the downloaded archive's hash against the source's policy.

    Args:
        source: The benchmark source whose hash policy applies.
        computed: The hex SHA-256 of the downloaded bytes.
        temp: The temp archive path, deleted before raising on failure.
        trust: Whether the caller accepts an unpinned source's bytes.

    Raises:
        RuntimeError: If a pinned hash mismatches, or the source is unpinned and `trust`
            is `False`. The error reveals the computed hash so it can be pinned.
    """
    if source.archive_sha256 is not None:
        if computed != source.archive_sha256:
            temp.unlink(missing_ok=True)
            msg = (
                f"{source.name}: archive SHA-256 {computed} does not match the pinned "
                f"{source.archive_sha256}; possible tampering or a changed upstream version"
            )
            raise RuntimeError(msg)
        return
    if not trust:
        temp.unlink(missing_ok=True)
        msg = (
            f"{source.name}: archive is not yet pinned (SHA-256 {computed}). Verify its "
            f"provenance, then re-run with --trust to accept it and pin the hash in SOURCES."
        )
        raise RuntimeError(msg)


def _normalize_layout(extracted: Path, source: BenchmarkSource) -> Path:
    """Flatten an extracted archive to a directory that the loader can read directly.

    Locates `<split>.json` (possibly under a wrapper folder), extracts a nested
    `<split>_databases.zip` if present, and ignores `__MACOSX`.

    Args:
        extracted: The directory the archive was unzipped into.
        source: The benchmark source (its `split` names the expected files).

    Returns:
        The directory directly containing `<split>.json` and `<split>_databases/`.

    Raises:
        RuntimeError: If the split JSON or databases directory cannot be located.
    """
    split = source.split
    matches = [p for p in extracted.rglob(f"{split}.json") if "__MACOSX" not in p.parts]
    if not matches:
        msg = f"{source.name}: {split}.json not found in the archive"
        raise RuntimeError(msg)
    root = matches[0].parent

    databases_zip = root / f"{split}_databases.zip"
    if databases_zip.is_file():
        with zipfile.ZipFile(databases_zip) as zf:
            zf.extractall(root)
        databases_zip.unlink()

    macosx = root / "__MACOSX"
    if macosx.is_dir():
        shutil.rmtree(macosx)

    if not (root / f"{split}_databases").is_dir():
        msg = f"{source.name}: {split}_databases/ not found after extraction"
        raise RuntimeError(msg)
    return root


def _sqlite_files(databases: Path) -> Iterator[Path]:
    """Yield every `*.sqlite` file under `databases`."""
    yield from databases.rglob("*.sqlite")


def _validate(root: Path, source: BenchmarkSource) -> None:
    """Check the normalized dataset's case count and SQLite integrity.

    Args:
        root: The normalized dataset root.
        source: The benchmark source carrying the expected invariants.

    Raises:
        RuntimeError: If the record count differs from `expected_cases`, or any SQLite
            database fails `PRAGMA integrity_check`.
    """
    records = json.loads((root / f"{source.split}.json").read_text(encoding="utf-8"))
    if len(records) != source.expected_cases:
        msg = (
            f"{source.name}: {source.split}.json has {len(records)} cases, "
            f"expected {source.expected_cases}; wrong dataset version"
        )
        raise RuntimeError(msg)

    for db_path in _sqlite_files(root / f"{source.split}_databases"):
        try:
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                result = con.execute("PRAGMA integrity_check").fetchone()
            finally:
                con.close()
        except sqlite3.DatabaseError as e:
            msg = f"{source.name}: {db_path.name} is not a valid SQLite database ({e})"
            raise RuntimeError(msg) from e
        if result is None or result[0] != "ok":
            msg = f"{source.name}: {db_path.name} failed integrity_check"
            raise RuntimeError(msg)


def fetch_benchmark(
    name: str,
    *,
    force: bool = False,
    trust: bool = False,
    cache_dir: Path | None = None,
    progress: bool = True,
) -> Path:
    """Download, verify, and cache a benchmark dataset, returning its loader-ready root.

    Resolves the source, downloads the archive (verifying its SHA-256 by policy), extracts
    and normalizes the layout, validates the case count and SQLite integrity, then atomically
    installs the result into the content-addressed cache. Prints a one-line license notice.

    Args:
        name: The dataset key (e.g. `"bird"`).
        force: Re-download even if a valid cached copy exists.
        trust: Accept an unpinned source's bytes (required the first time a source is fetched).
        cache_dir: An explicit cache directory, else the default cache root.
        progress: Show a download progress bar on a TTY.

    Returns:
        The cached dataset root: a directory directly containing `<split>.json` and
        `<split>_databases/`.

    Raises:
        ValueError: If `name` is not a known source.
        RuntimeError: On a hash mismatch, an untrusted unpinned source, a missing layout,
            a case-count mismatch, or a SQLite integrity failure.
    """  # noqa: DOC502
    source = SOURCES.get(name)
    if source is None:
        available = ", ".join(sorted(SOURCES))
        msg = f"unknown benchmark {name!r}; available: {available}"
        raise ValueError(msg)

    if not force:
        cached = cached_dataset_path(name, cache_dir=cache_dir)
        if cached is not None:
            return cached

    # Refuse an unpinned source before downloading, so a forgotten --trust doesn't pull the
    # whole archive only to reject it (and force a second download on the corrective re-run).
    if source.archive_sha256 is None and not trust:
        msg = (
            f"{source.name}: archive is not yet pinned. Verify its provenance, then re-run "
            f"with --trust to download it and reveal the SHA-256 to pin in SOURCES."
        )
        raise RuntimeError(msg)

    parent = _datasets_dir(name, cache_dir=cache_dir)
    parent.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(prefix=f"evaldata-{name}-") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "archive.zip"
        computed = _download(source.url, archive, progress=progress)
        _verify_hash(source, computed, archive, trust=trust)

        extracted = tmp_path / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
        archive.unlink()

        root = _normalize_layout(extracted, source)
        _validate(root, source)

        meta = {
            "name": source.name,
            "sha256": computed,
            "url": source.url,
            "expected_cases": source.expected_cases,
            "license": source.license,
            "split": source.split,
        }
        (root / _META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        destination = parent / computed[:16]
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(root), str(destination))

    print(f"{source.name}: {source.license} — source {source.url}")  # noqa: T201 - user-facing notice
    if source.archive_sha256 is None:
        # Surface the hash so an unpinned source can be locked to this version in SOURCES.
        print(f"{source.name}: pin this version by setting archive_sha256={computed!r}")  # noqa: T201
    return destination
