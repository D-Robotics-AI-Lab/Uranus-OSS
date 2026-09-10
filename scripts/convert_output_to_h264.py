#!/usr/bin/env python3
"""Convert MP4 files below a directory to browser-friendly H.264 MP4 files.

By default, files are read from ``output/`` and written to ``output/h264/``.
The destination is deliberately separate from the source tree so that running
the script repeatedly does not convert its own output files.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("output"),
        help="Directory containing source MP4 files (default: output)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Destination directory (default: <input_dir>/h264)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace H.264 files that already exist",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 quality setting, lower is higher quality (default: 23)",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        choices=("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"),
        help="libx264 encoding speed/size preset (default: medium)",
    )
    return parser.parse_args()


def convert_file(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
    crf: int,
    preset: str,
) -> bool:
    if destination.exists() and not overwrite:
        print(f"SKIP  {source} -> {destination} (already exists; use --overwrite)")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Encode to a temporary file first, so a failed ffmpeg run cannot leave a
    # seemingly complete destination file behind.
    # Keep the .mp4 suffix so ffmpeg can infer the output container.
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True)
        temporary.replace(destination)
    except (OSError, subprocess.CalledProcessError) as error:
        temporary.unlink(missing_ok=True)
        print(f"FAIL  {source}: {error}", file=sys.stderr)
        return False

    print(f"OK    {source} -> {destination}")
    return True


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = (args.output_dir or input_dir / "h264").resolve()

    if shutil.which("ffmpeg") is None:
        print("错误：找不到 ffmpeg，请先安装 ffmpeg。", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"错误：输入目录不存在：{input_dir}", file=sys.stderr)
        return 2
    if args.crf < 0 or args.crf > 51:
        print("错误：--crf 必须在 0 到 51 之间。", file=sys.stderr)
        return 2

    sources = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() == ".mp4"
        and (path != output_dir and output_dir not in path.parents)
    )
    if not sources:
        print(f"在 {input_dir} 下没有找到 MP4 文件。")
        return 0

    succeeded = 0
    failed = 0
    for source in sources:
        relative = source.relative_to(input_dir)
        destination = output_dir / relative.with_suffix(".mp4")
        if convert_file(
            source,
            destination,
            overwrite=args.overwrite,
            crf=args.crf,
            preset=args.preset,
        ):
            succeeded += 1
        else:
            failed += 1

    print(f"完成：成功/跳过 {succeeded} 个，失败 {failed} 个。输出目录：{output_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
