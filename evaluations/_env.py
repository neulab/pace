"""Optional `.env` loading for the evaluation entry points.

Credentials (`API_KEY`, `BASE_URL`, `HF_HUB_ENABLE_HF_TRANSFER`, ...) are read
from the process environment. To avoid re-`export`-ing them every shell session,
you may instead put them in a repo-root `.env` file (git-ignored):

    API_KEY=sk-...
    BASE_URL=https://cmu.litellm.ai

`load_env()` is called at the top of the CLI entry points (`run.py`,
`score_new_model.py`) to populate the environment from that file. It is a no-op
when the file is absent or `python-dotenv` is not installed, and it never
overrides variables already set in the real environment — so an explicit
`export` (or a SLURM/CI job env) still wins over `.env`.

Only the CLI boundary loads `.env`; the library layer (`run_instance` and the
handlers) keeps taking credentials as explicit arguments, with no hidden global
state.
"""
from pathlib import Path

# repo root = parent of evaluations/
_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> None:
    """Populate os.environ from <repo-root>/.env if present. Best-effort no-op."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return  # python-dotenv not installed -> rely on real env vars only
    # override=False: real environment variables take precedence over .env
    load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=False)
