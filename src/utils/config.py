import json
from copy import deepcopy

def load_config(path):
    with open(path) as f:
        return json.load(f)


def merge_config(base, updates):
    cfg = deepcopy(base)
    cfg.update(updates)
    return cfg