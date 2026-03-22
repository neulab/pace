from pathlib import Path


def delete_tokens_json_files(directory: str) -> None:
    """
    Delete all files ending with '_tokens.json' under the given directory recursively.

    Args:
        directory: Root directory to search.
    """
    root = Path(directory).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    for file_path in root.rglob("*_tokens.json"):
        if file_path.is_file():
            file_path.unlink()
            print(f"Deleted: {file_path}")
            
import os
dirs = ["/home/yueqis/proxybench/proxy-bench/results/raw_results/lifbench/" + dir for dir in os.listdir("/home/yueqis/proxybench/proxy-bench/results/raw_results/lifbench/")]
for d in dirs:
    delete_tokens_json_files(d)