"""Benchmark loaders: turn standard text-to-SQL datasets into runnable `EvalCase`s.

Each loader reads a user-downloaded dataset directory (the datasets are not redistributed) and
yields `EvalCase`s whose expected outcome is the benchmark's gold query (`GoldQuery`), to be
scored with `ExecutionAccuracy`. Spider and BIRD both ship a per-`db_id` SQLite database, so
both run on the `sqlite` platform.
"""

from evaldata.loaders.benchmarks.bird import load_bird
from evaldata.loaders.benchmarks.fetch import (
    SOURCES,
    BenchmarkSource,
    cache_root,
    cached_dataset_path,
    fetch_benchmark,
)
from evaldata.loaders.benchmarks.spider import load_spider

__all__ = [
    "SOURCES",
    "BenchmarkSource",
    "cache_root",
    "cached_dataset_path",
    "fetch_benchmark",
    "load_bird",
    "load_spider",
]
