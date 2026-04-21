import sys
import os

# Ensure the tests directory itself is on sys.path so `fixtures` is importable
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CT_DIR = os.path.dirname(_TESTS_DIR)
_EXP_DIR = os.path.dirname(_CT_DIR)
_REPO = os.path.dirname(_EXP_DIR)

for p in [_TESTS_DIR, _CT_DIR, _EXP_DIR, _REPO,
          os.path.join(_REPO, 'round2_trained'),
          os.path.join(_REPO, 'round1_constructed')]:
    if p not in sys.path:
        sys.path.insert(0, p)
