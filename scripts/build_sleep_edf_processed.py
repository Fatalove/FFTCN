"""构建里程碑 3 的完整 Sleep-EDF-153 预处理数据与可复核清单。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

# 直接运行 scripts/*.py 时，把仓库根目录加入模块搜索路径。
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from data.sleep_edf_preprocessor import preprocess_record, save_processed_record, split_subjects
from data.sleep_edf_reader import discover_records, read_record


DEFAULT_RAW_ROOT = Path("datasets/sleep-edf-expanded-1.0.0/sleep-cassette")
DEFAULT_OUTPUT_ROOT = Path("datasets/sleep-edf-153-processed-v1")
DEFAULT_ARTIFACT = Path("reproduction_artifacts/milestone_03/full_dataset_summary.json")
EXPECTED_CLASS_COUNTS = {0: 65951, 1: 21522, 2: 69132, 3: 13039, 4: 25835}


def build_dataset(raw_root: Path, output_root: Path) -> dict[str, object]:
    records = discover_records(raw_root)
    split = split_subjects(records, seed=42)
    subjects_by_split = {
        "train": set(split.train),
        "validation": set(split.validation),
        "test": set(split.test),
    }
    assignment = {
        subject_id: split_name
        for split_name, subject_ids in subjects_by_split.items()
        for subject_id in subject_ids
    }

    split_stats = {
        split_name: {
            "subjects": sorted(subject_ids),
            "records": [],
            "epoch_count": 0,
            "class_counts": Counter(),
        }
        for split_name, subject_ids in subjects_by_split.items()
    }
    total_counts: Counter[int] = Counter()

    for paths in tqdm(records, desc="preprocess", ncols=100):
        split_name = assignment[paths.subject_id]
        raw = read_record(paths)
        processed = preprocess_record(raw)

        non_wake = (processed.labels != 0).nonzero()[0]
        leading_wake = int(non_wake[0])
        trailing_wake = int(len(processed.labels) - non_wake[-1] - 1)
        if leading_wake > 60 or trailing_wake > 60:
            raise RuntimeError(
                f"{paths.record_id} wake trim failed: {leading_wake}/{trailing_wake}"
            )

        output_path = save_processed_record(processed, output_root / split_name)
        counts = Counter(int(label) for label in processed.labels)
        total_counts.update(counts)
        split_stats[split_name]["epoch_count"] += len(processed.labels)
        split_stats[split_name]["class_counts"].update(counts)
        split_stats[split_name]["records"].append(
            {
                "record_id": paths.record_id,
                "subject_id": paths.subject_id,
                "night_id": paths.night_id,
                "epochs": len(processed.labels),
                "class_counts": {str(key): counts.get(key, 0) for key in range(5)},
                "file": str(output_path.resolve()),
            }
        )

    if len(records) != 153:
        raise RuntimeError(f"expected 153 records, got {len(records)}")
    if dict(sorted(total_counts.items())) != EXPECTED_CLASS_COUNTS:
        raise RuntimeError(
            f"class-count mismatch: {dict(sorted(total_counts.items()))}"
        )

    for stats in split_stats.values():
        stats["subject_count"] = len(stats["subjects"])
        stats["record_count"] = len(stats["records"])
        stats["class_counts"] = {
            str(key): stats["class_counts"].get(key, 0) for key in range(5)
        }

    return {
        "dataset": "Sleep-EDF-153 processed v1",
        "source_root": str(raw_root.resolve()),
        "output_root": str(output_root.resolve()),
        "split_seed": 42,
        "record_count": len(records),
        "subject_count": len(assignment),
        "epoch_count": sum(total_counts.values()),
        "class_counts": {str(key): total_counts.get(key, 0) for key in range(5)},
        "splits": split_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_dataset(args.raw_root, args.output_root)
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"processed dataset: {args.output_root.resolve()}")
    print(f"summary: {args.artifact.resolve()}")


if __name__ == "__main__":
    main()
