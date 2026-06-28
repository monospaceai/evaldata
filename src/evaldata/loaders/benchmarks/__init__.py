"""Loaders for standard text-to-SQL benchmark datasets (Spider, BIRD).

Each loader reads a dataset directory and yields `EvalCase`s with a `GoldQuery` expected,
suitable for scoring with `ExecutionAccuracy`.
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
