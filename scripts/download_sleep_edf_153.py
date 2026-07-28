"""从 PhysioNet 下载并校验 Sleep-EDF Expanded 的 153 条 SC 记录。"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


# PhysioNet 官方页面提供的公开 S3 镜像；相较网页文件端点更适合并发与断点续传。
BASE_URL = "https://physionet-open.s3.amazonaws.com/sleep-edfx/1.0.0"
DEFAULT_ROOT = Path("datasets/sleep-edf-expanded-1.0.0")
METADATA_FILES = ("RECORDS", "SHA256SUMS.txt", "SC-subjects.xls")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def request_bytes(relative_path: str) -> bytes:
    response = requests.get(f"{BASE_URL}/{relative_path}", timeout=(15, 120))
    response.raise_for_status()
    return response.content


def load_manifest(root: Path) -> tuple[list[str], dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        metadata_path = root / name
        if not metadata_path.exists():
            metadata_path.write_bytes(request_bytes(name))

    checksums = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        checksum, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = checksum

    # PhysioNet 的 RECORDS 只索引 PSG 主记录；配套 Hypnogram 需要从完整
    # SHA-256 清单中选择，否则会静默漏掉全部人工睡眠阶段标注。
    records = sorted(
        path
        for path in checksums
        if path.startswith("sleep-cassette/") and path.endswith(".edf")
    )
    psg_count = sum(path.endswith("-PSG.edf") for path in records)
    hypnogram_count = sum(path.endswith("-Hypnogram.edf") for path in records)
    if (psg_count, hypnogram_count) != (153, 153):
        raise RuntimeError(
            "官方 Sleep-Cassette 文件数异常："
            f"PSG={psg_count}, Hypnogram={hypnogram_count}"
        )
    return records, checksums


def download_one(root: Path, relative_path: str, expected_hash: str) -> str:
    destination = root / relative_path
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and sha256(destination) == expected_hash:
        return "cached"
    if destination.exists():
        destination.unlink()

    for attempt in range(1, 4):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                f"{BASE_URL}/{relative_path}",
                headers=headers,
                stream=True,
                timeout=(15, 120),
            ) as response:
                response.raise_for_status()
                append = offset > 0 and response.status_code == 206
                mode = "ab" if append else "wb"
                with partial.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)

            if sha256(partial) != expected_hash:
                partial.unlink(missing_ok=True)
                raise RuntimeError("SHA-256 不匹配")
            partial.replace(destination)
            return "downloaded"
        except (requests.RequestException, OSError, RuntimeError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        help="仅下载指定记录前缀，例如 SC4001；可重复指定。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records, checksums = load_manifest(args.root)
    if args.record:
        prefixes = tuple(args.record)
        records = [path for path in records if Path(path).name.startswith(prefixes)]
    if not records:
        raise RuntimeError("没有匹配的 Sleep-Cassette 文件")

    psg_count = sum(path.endswith("-PSG.edf") for path in records)
    hypnogram_count = sum(path.endswith("-Hypnogram.edf") for path in records)
    print(
        f"目标文件 {len(records)} 个：PSG={psg_count}，Hypnogram={hypnogram_count}，"
        f"保存到 {args.root.resolve()}",
        flush=True,
    )

    failures: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(download_one, args.root, path, checksums[path]): path
            for path in records
        }
        for index, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                result = future.result()
                print(f"[{index}/{len(records)}] {result}: {path}", flush=True)
            except BaseException as exc:  # 保留其他下载任务并汇总失败项
                failures.append((path, exc))
                print(f"[{index}/{len(records)}] failed: {path}: {exc}", flush=True)

    if failures:
        print(f"失败 {len(failures)} 个；重新执行同一命令可断点续传。", file=sys.stderr)
        return 1
    print("全部文件已通过官方 SHA-256 校验。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
