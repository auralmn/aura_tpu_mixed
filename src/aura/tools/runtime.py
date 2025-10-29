#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import importlib
from typing import Any, Dict

from .registry import get_tool, registry_default_path


def run_registered_tool(name: str, kwargs: Dict[str, Any] | None = None) -> Any:
    spec = get_tool(name, registry_default_path())
    module = spec.get('module')
    call = spec.get('callable', 'main')
    if not module:
        raise ValueError(f"tool not found: {name}")
    mod = importlib.import_module(module)
    fn = getattr(mod, call, None)
    if fn is None:
        raise AttributeError(f"callable '{call}' not found in module '{module}'")
    return fn(**(kwargs or {}))
