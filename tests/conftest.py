"""Test-wide setup.

The safety net (``crystalline.ui.safety``) swallows exceptions escaping Qt
virtual overrides, which is exactly right in the app and exactly wrong here: a
guard that quietly ate a real bug would let the suite pass over the very faults
it exists to survive. Strict mode makes every guard re-raise, so a broken
``paintEvent`` fails a test instead of being reported and shrugged off.

The tests that exercise the net itself turn strict mode off around what they
are testing.
"""

import pytest


@pytest.fixture(autouse=True, scope="session")
def _guards_re_raise():
    safety = pytest.importorskip("crystalline.ui.safety")
    safety.strict(True)
    yield
    safety.strict(False)
