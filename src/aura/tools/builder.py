#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Optional


def create_skeleton_tool(name: str, description: Optional[str] = None) -> str:
    """
    Create a skeleton tool file under aura.tools.user/<name>.py with a main(**kwargs) entrypoint.
    Returns the relative module file path.
    """
    safe = "".join([c if c.isalnum() or c in ('_',) else '_' for c in name.strip()])
    pkg_dir = os.path.join(os.path.dirname(__file__), 'user')
    os.makedirs(pkg_dir, exist_ok=True)
    init_file = os.path.join(pkg_dir, '__init__.py')
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("")
    mod_path = os.path.join(pkg_dir, f"{safe}.py")
    if not os.path.exists(mod_path):
        desc = (description or 'A user-defined tool.').replace('"', '\"')
        content = (
            f"#!/usr/bin/env python3\n"
            f"# Auto-generated tool: {safe}\n\n"
            f"\"\"\"\n{desc}\n\"\"\"\n\n"
            f"from typing import Any, Dict\n\n"
            f"def main(**kwargs: Dict[str, Any]) -> Dict[str, Any]:\n"
            f"    \"\"\"Entry point for tool execution. Modify as needed.\"\"\"\n"
            f"    return {{'ok': True, 'tool': '{safe}', 'kwargs': dict(kwargs)}}\n"
        )
        with open(mod_path, 'w', encoding='utf-8') as f:
            f.write(content)
    return mod_path
