"""Refuse to commit a vcr cassette that still contains a credential.

Deny-list header scrubbing fails open when a provider adds a new secret-bearing field, so this
is the independent second net: it scans recorded cassettes for credential markers and exits
non-zero if any survive. Run over cassette files by the pre-commit hook of the same name.
"""

import re
import sys

# Credential markers that must never appear in a committed cassette. Kept provider-agnostic and
# free of any real account identifier so the scanner itself leaks nothing.
_FORBIDDEN = [
    re.compile(r"snowflake token", re.IGNORECASE),
    re.compile(r"\bauthorization\b\s*:", re.IGNORECASE),
    re.compile(r"\bbearer\s", re.IGNORECASE),
    re.compile(r"set-cookie", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}"),  # JWT header segment
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),  # OpenAI-style key
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bprivate_key\b", re.IGNORECASE),
]


def scan(path: str) -> list[str]:
    """Return the credential markers found in the cassette at `path`.

    Args:
        path: The cassette file to scan.

    Returns:
        A list of human-readable findings, empty when the cassette is clean.
    """
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return [f"{path}: matched /{pattern.pattern}/" for pattern in _FORBIDDEN if pattern.search(text)]


def main(paths: list[str]) -> int:
    """Scan `paths` and report any findings.

    Args:
        paths: The cassette files to scan.

    Returns:
        `1` if any credential marker was found, else `0`.
    """
    findings = [finding for path in paths for finding in scan(path)]
    for finding in findings:
        print(finding, file=sys.stderr)
    if findings:
        print("\nrefusing to commit: cassette(s) appear to contain a credential", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
