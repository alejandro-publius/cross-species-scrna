"""The README states how many tests there are. Keep it true.

A count written into prose goes stale the moment a test is added, because
nothing reads prose. The README said `(5 passed)` while the documented command
runs 14 -- nearly three times the promise, in the very first thing the
quickstart tells a reader to run.

So the number is derived here rather than trusted, and derived in a way that
gives the same answer in every environment. `tests/test_prep.py` skips itself
when scanpy is absent, which is deliberate -- CI runs a fast lane without the
single-cell stack (see .github/workflows/ci.yml) -- so a plain collection
returns 11 there and 14 under `uv sync && uv run pytest`. A guard that simply
read the local collection would therefore pass in one environment and fail in
the other, or, worse, skip itself in the only environment that gates a push.

Instead: count what collection found, then add back the tests in any module
collection did not reach, read straight from its source. The total is the size
of the suite as a whole, which is what the README describes.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPO_ROOT / "tests"

# (path relative to repo root, why its counts are frozen).
# A changelog entry records what was true at a release and must not be updated
# to match the present, so it is declared here rather than silently skipped.
HISTORICAL_COUNTS: list[tuple[str, str]] = [
    ("CHANGELOG.md", "release notes quote the counts that were true at the time"),
]

_COUNT = re.compile(r"\(?\b(\d{1,4})\s+(?:tests?|passed)\b\)?")


def _collect() -> tuple[int, set[str]]:
    """Return (tests collected here, the test files collection actually reached)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        pytest.fail(f"collection failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    match = re.search(r"^(\d+) tests? collected", proc.stdout, re.MULTILINE)
    if match is None:
        pytest.fail(f"could not read a collected count from:\n{proc.stdout[-2000:]}")
    reached = {
        line.split("::", 1)[0].strip()
        for line in proc.stdout.splitlines()
        if "::" in line and line.strip().startswith("tests/")
    }
    return int(match.group(1)), reached


def _tests_defined_in(path: Path) -> int:
    """Count test functions in a module collection never reached.

    Read from source, because the module cannot be imported in this
    environment -- that is precisely why it was skipped.
    """
    source = path.read_text(encoding="utf-8")
    if "parametrize" in source:
        pytest.fail(
            f"{path.name} was skipped in this environment and uses parametrize, so "
            "its test count cannot be read from source. Add it to the CI lane, or "
            "teach this guard the expansion."
        )
    tree = ast.parse(source)
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    )


def suite_size() -> int:
    """The whole suite, independent of which optional extras are installed."""
    collected, reached = _collect()
    unreached = [
        path
        for path in sorted(TESTS_DIR.glob("test_*.py"))
        if str(path.relative_to(REPO_ROOT)) not in reached
    ]
    return collected + sum(_tests_defined_in(path) for path in unreached)


def _live_claims() -> list[tuple[Path, int, str]]:
    """Every docs line that states a pytest count, with its number."""
    frozen = {(REPO_ROOT / rel).resolve() for rel, _ in HISTORICAL_COUNTS}
    out: list[tuple[Path, int, str]] = []
    for path in sorted(REPO_ROOT.rglob("*.md")):
        if {".git", "node_modules", ".venv"} & set(path.parts):
            continue
        if path.resolve() in frozen:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            # Only lines that also name the command, so an unrelated "5 passed"
            # in prose is not compared against the suite total.
            if "pytest" not in line:
                continue
            for match in _COUNT.finditer(line):
                out.append((path, int(match.group(1)), line.strip()))
    return out


def test_every_stated_pytest_count_is_the_current_one():
    total = suite_size()
    claims = _live_claims()
    assert claims, (
        "no docs line states a pytest count any more. If that is deliberate, "
        "delete this test; if a line was reworded, this check just went blind."
    )
    stale = [
        f"{path.relative_to(REPO_ROOT)}: says {stated}, suite has {total} -- {line!r}"
        for path, stated, line in claims
        if stated != total
    ]
    assert not stale, "stale test counts:\n  " + "\n  ".join(stale)
