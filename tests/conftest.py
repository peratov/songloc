"""Test isolation.

`app.config` resolves SONGLOC_DATA_DIR at import time, and `app.store` opens its
SQLite connection at import time from the path that produced. By the time a test
body runs, the test module's own top-level `from app.providers import ...` has
already pulled `app.config` in, so setting the variable inside a test is too late
to move anything -- it lands in the repo's real data/ directory.

pytest imports conftest before it imports any test module, which is the only
point early enough to redirect it. Everything below must therefore run at import
time, and must not import `app`.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

_DATA_DIR = Path(tempfile.mkdtemp(prefix="songloc-tests-"))

# Before app.config is imported by anything. Overwrites rather than defaults:
# a developer with SONGLOC_DATA_DIR pointing at real state should not have the
# suite write into it.
os.environ["SONGLOC_DATA_DIR"] = str(_DATA_DIR)

# The keystore lives under the data dir too, so this also stops the suite from
# reading real API keys out of data/credentials.json -- the mock providers must
# pass with no credentials configured.


def pytest_sessionfinish(session, exitstatus):
    # app.store opens its SQLite connection at import time and holds it open for
    # the process. Windows refuses to unlink an open file, so close it first or
    # the temp directory leaks a songloc.db every run.
    store = sys.modules.get("app.store")
    if store is not None:
        store._conn.close()
    shutil.rmtree(_DATA_DIR, ignore_errors=True)
