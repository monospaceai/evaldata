"""Pytest session bootstrap for local developer runs."""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
