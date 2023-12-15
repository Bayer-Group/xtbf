import sys

def pytest_ignore_collect(path):
        if str(path).endswith("dc_featurizer.py"):
            return True
