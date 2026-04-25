import subprocess
import sys


def test_top_level_import_keeps_formal_core_without_optional_ml_dependencies():
    script = r'''
import builtins
import importlib

real_import = builtins.__import__
blocked = {"torch", "networkx"}


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if root in blocked:
        raise ImportError(f"blocked optional dependency: {root}")
    return real_import(name, globals, locals, fromlist, level)


builtins.__import__ = guarded_import
package = importlib.import_module("topos_ai")
formal = importlib.import_module("topos_ai.formal_category")
assert hasattr(package, "FiniteCategory")
assert hasattr(formal, "PresheafTopos")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
