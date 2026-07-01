from pathlib import Path
import shutil


def copy_files_out_of_zero(src_root: str, dst_root: str, zero_dir_name: str = "0", overwrite: bool = False):
    """
    Copy files from:
        src_root/<model>/<zero_dir_name>/*
    to:
        dst_root/<model>/*

    Example:
        src_root/Claude-Opus-4.5/0/selfrepair_1.json
        -> dst_root/Claude-Opus-4.5/selfrepair_1.json

    Args:
        src_root: Source root directory containing model subdirectories.
        dst_root: Destination root directory.
        zero_dir_name: The intermediate subdirectory name, default is "0".
        overwrite: Whether to overwrite destination files if they already exist.
    """
    src_root = Path(src_root).expanduser().resolve()
    dst_root = Path(dst_root).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    for model_dir in src_root.iterdir():
        if not model_dir.is_dir():
            continue

        zero_dir = model_dir / zero_dir_name
        if not zero_dir.is_dir():
            print(f"Skip: {zero_dir} does not exist")
            continue

        target_model_dir = dst_root / model_dir.name
        target_model_dir.mkdir(parents=True, exist_ok=True)

        for item in zero_dir.iterdir():
            if not item.is_file():
                continue

            dst_file = target_model_dir / item.name

            if dst_file.exists():
                if overwrite:
                    if dst_file.is_file():
                        dst_file.unlink()
                    else:
                        raise IsADirectoryError(f"Destination exists and is not a file: {dst_file}")
                else:
                    print(f"Skip existing file: {dst_file}")
                    continue

            shutil.copy2(str(item), str(dst_file))
            print(f"Copied: {item} -> {dst_file}")


copy_files_out_of_zero(
    src_root="/home/yueqis/proxybench/LiveCodeBench/output",
    dst_root="/home/yueqis/proxybench/proxy-bench/results/raw_results/livecodebench",
)