#!/usr/bin/env python3
"""Minimal BFCL scoring runner, executed under ``evaluations/.bfcl-venv/bin/python``.

Why this exists
---------------
BFCL requires ``tree-sitter`` 0.21.x, which hard-conflicts with the main env
(repobench / codebleu need 0.23.x). ``evaluations/install.sh`` step [2/2] therefore
builds a dedicated virtualenv at ``evaluations/.bfcl-venv`` containing only BFCL's
dependencies. Running BFCL in the main env fails with
``No module named 'tree_sitter_javascript'``.

``evaluations/score_new_model.py`` runs this file as a subprocess under that venv
python so BFCL is scored with the authentic ``bfcl_eval`` checkers (AST /
multi-turn / relevance / agentic), without importing the rest of
``evaluations/run.py`` (whose ``lm_eval`` etc. are not installed in the venv).
``evaluations/handlers/bfcl.py`` only imports ``os`` / ``sys`` at module scope; all
heavy imports happen inside ``_run_bfcl``, so importing it here is cheap and safe.

Input
-----
A JSON object on **stdin** (preferred), and/or the same keys as ``--flags`` on
argv. stdin wins for any key it provides::

    {"model_name", "base_url", "api_key", "subtask", "instance_id"}

Credentials arrive via stdin so they never appear in argv / ``ps`` output or on
disk.

Output
------
On success: the JSON-encoded list returned by ``_run_bfcl`` (e.g. ``[{...}]``) on
stdout, exit 0. Any noise printed by ``bfcl_eval`` / inference is redirected to
stderr so stdout carries only that JSON.

On error: a JSON object ``{"error": ..., "traceback": ...}`` on stdout, exit 1.
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

# Make the `evaluations` package importable regardless of the current directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_KEYS = ("model_name", "base_url", "api_key", "subtask", "instance_id")


def _parse_inputs() -> dict:
    """Collect the 5 required params from argv and/or a JSON object on stdin."""
    ap = argparse.ArgumentParser(description="Score one BFCL instance.")
    for k in _KEYS:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=None)
    args = ap.parse_args()
    params = {k: getattr(args, k) for k in _KEYS}

    # stdin (JSON) fills in / overrides anything not passed on argv. Preferred for
    # credentials so they never land in argv / `ps` output or on disk.
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            data = json.loads(raw)
            for k in _KEYS:
                if data.get(k) is not None:
                    params[k] = data[k]

    missing = [k for k in _KEYS if params.get(k) in (None, "")]
    if missing:
        raise ValueError(f"missing required parameter(s): {', '.join(missing)}")
    return params


def main() -> None:
    try:
        params = _parse_inputs()
        # Imported lazily and only after inputs validate: pulls in bfcl_eval, which
        # only resolves correctly under the .bfcl-venv interpreter.
        from evaluations.handlers.bfcl import _run_bfcl

        # Keep stdout clean: route any prints from bfcl_eval / inference to stderr
        # so the only thing on stdout is our final JSON result.
        with contextlib.redirect_stdout(sys.stderr):
            result = _run_bfcl(
                model_name=params["model_name"],
                base_url=params["base_url"],
                api_key=params["api_key"],
                subtask=params["subtask"],
                instance_id=str(params["instance_id"]),
            )
    except Exception as exc:  # noqa: BLE001 - report every failure as JSON
        import traceback

        json.dump(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()
        sys.exit(1)

    # `result` embeds model_result_raw / metadata that may hold non-JSON types;
    # default=str keeps serialization total without losing scalar fields like
    # 'valid' that score_new_model.extract() reads.
    json.dump(result, sys.stdout, default=str)
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
