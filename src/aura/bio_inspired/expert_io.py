#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from flax import serialization


def save_params(params: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = serialization.to_bytes(params)
    with open(path, 'wb') as f:
        f.write(data)


def load_params(empty_vars: Any, path: str) -> Any:
    with open(path, 'rb') as f:
        data = f.read()
    params = serialization.from_bytes(empty_vars, data)
    return params
