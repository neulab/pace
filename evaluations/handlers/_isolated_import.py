"""Import a vendored benchmark's top-level modules without clashing with another
benchmark's modules of the same name.

planbench and visualwebbench both ship a top-level `utils` package. Python caches
modules by name, so when one process runs both (evaluations/score_new_model.py runs
every handler in-process), whichever ran second would get the first one's `utils`.
"""

import contextlib
import sys
import threading

_LOCK = threading.Lock()


@contextlib.contextmanager
def isolated_imports(path, names):
    """Import the top-level modules `names` (and their submodules) from `path` inside
    the block. Same-named modules already loaded are set aside for the block and put
    back afterwards; the objects imported inside the block keep working."""
    def owned(mod):
        return any(mod == n or mod.startswith(n + ".") for n in names)

    with _LOCK:
        saved = {m: sys.modules.pop(m) for m in list(sys.modules) if owned(m)}
        sys.path.insert(0, path)
        try:
            yield
        finally:
            sys.path.remove(path)
            for m in [m for m in sys.modules if owned(m)]:
                del sys.modules[m]
            sys.modules.update(saved)
