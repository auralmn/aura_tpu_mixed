#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import json
from typing import Dict, Any, List


def registry_default_path() -> str:
    return os.path.join('tools', 'registry.json')


def load_registry(path: str = None) -> Dict[str, Any]:
    path = path or registry_default_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def save_registry(reg: Dict[str, Any], path: str = None) -> bool:
    path = path or registry_default_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(reg, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def register_tool(name: str, spec: Dict[str, Any], path: str = None) -> bool:
    reg = load_registry(path)
    reg[name] = spec
    return save_registry(reg, path)


def list_tools(path: str = None) -> List[str]:
    reg = load_registry(path)
    return sorted(list(reg.keys()))


def get_tool(name: str, path: str = None) -> Dict[str, Any]:
    reg = load_registry(path)
    return reg.get(name, {})
