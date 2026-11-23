# Copyright (c) 2026 the UroJEPA authors.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""YAML configs with `_base_` inheritance and command-line overrides.

    python finetune.py --config configs/finetune/tasks/stone_sepsis.yaml \
        --opts seed=1 model.pretrained=/ckpt/urojepa_dagger.pth.tar data.train_fraction=0.25
"""

import copy
import os

import yaml


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set_by_dotted_key(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def load_config(path, overrides=()):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("_base_", None)
    if base is not None:
        base_path = base if os.path.isabs(base) else os.path.join(os.path.dirname(path), base)
        cfg = _deep_merge(load_config(base_path), cfg)
    return apply_overrides(cfg, overrides)


def apply_overrides(cfg, overrides=()):
    """Apply `key.subkey=value` overrides (values are parsed as YAML)."""
    for item in overrides or ():
        if "=" not in item:
            raise ValueError(f"Override {item!r} must look like key.subkey=value")
        key, value = item.split("=", 1)
        _set_by_dotted_key(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def add_config_args(parser):
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--opts", nargs="*", default=[], help="config overrides, e.g. seed=1 optimization.lr=5e-4"
    )
    return parser
