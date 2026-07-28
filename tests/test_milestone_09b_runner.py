from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from scripts import run_milestone_09b as runner
from training.sleep_edf_full_run import StageTrainingResult


class Milestone09BRunnerTest(unittest.TestCase):
    def test_cache_progress_counts_each_cwt_and_switches_stage_label(self) -> None:
        bar = mock.Mock()
        wave_transform = mock.Mock(side_effect=lambda raw: raw + 1)
        with mock.patch.object(runner, "tqdm", return_value=bar):
            progress = runner._CacheWaveProgress(
                wave_transform,
                {"train": 2, "validation": 1},
            )
            outputs = [
                progress(np.array([value], dtype=np.float32))
                for value in (1, 2, 3)
            ]
            progress.close()

        self.assertEqual([int(output[0]) for output in outputs], [2, 3, 4])
        self.assertEqual(bar.update.call_count, 3)
        self.assertEqual(
            [call.args[0] for call in bar.set_description.call_args_list],
            ["cache train CWT", "cache validation CWT"],
        )
        bar.close.assert_called_once_with()

    def test_fixed_configs_keep_preflight_and_formal_outputs_separate(self) -> None:
        preflight = runner._config(
            runner.PREFLIGHT_OUTPUT,
            raw_epochs=1,
            wave_epochs=1,
            fusion_epochs=1,
        )
        formal = runner._config(
            runner.FORMAL_OUTPUT,
            raw_epochs=20,
            wave_epochs=20,
            fusion_epochs=50,
        )

        self.assertNotEqual(preflight.output_dir, formal.output_dir)
        self.assertEqual(
            (
                formal.raw_pretrain_epochs,
                formal.wave_pretrain_epochs,
                formal.fusion_finetune_epochs,
            ),
            (20, 20, 50),
        )
        self.assertEqual(
            (formal.raw_batch_size, formal.wave_batch_size, formal.fusion_batch_size),
            (128, 128, 32),
        )
        self.assertEqual(formal.seed, 0)
        self.assertEqual(formal.num_workers, 0)

    def test_cache_status_requires_all_files_and_fixed_sample_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir) / "cache"
            split_dir = cache_root / "train"
            split_dir.mkdir(parents=True)
            np.save(split_dir / "raw.npy", np.zeros((2, 1, 3000), np.float32))
            np.save(split_dir / "wave.npy", np.zeros((2, 1, 30, 60), np.float16))
            np.save(split_dir / "labels.npy", np.zeros((2,), np.int64))
            manifest = {
                "sample_count": 2,
                "raw_shape": [2, 1, 3000],
                "raw_dtype": "float32",
                "wave_shape": [2, 1, 30, 60],
                "wave_dtype": "float16",
                "labels_shape": [2],
                "labels_dtype": "int64",
                "records": [{"record_id": "r0", "start": 0, "stop": 2}],
            }
            (split_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            with (
                mock.patch.object(runner, "CACHE_ROOT", cache_root),
                mock.patch.object(
                    runner, "EXPECTED_CACHE_COUNTS", {"train": 2}
                ),
            ):
                self.assertTrue(runner._cache_status()["complete"])
                (split_dir / "wave.npy").write_bytes(b"")
                self.assertFalse(runner._cache_status()["complete"])

                np.save(
                    split_dir / "wave.npy",
                    np.zeros((2, 1, 30, 60), np.float16),
                )
                manifest.pop("wave_shape")
                (split_dir / "manifest.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                self.assertFalse(runner._cache_status()["complete"])

    def test_resume_checkpoints_are_collected_from_one_output_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "preflight"
            for stage in runner.STAGE_NAMES:
                path = output_dir / stage / "last.pt"
                path.parent.mkdir(parents=True)
                path.touch()

            checkpoints = runner._last_checkpoints(output_dir, require_all=True)

        self.assertEqual(set(checkpoints), set(runner.STAGE_NAMES))
        self.assertTrue(all(path.is_relative_to(output_dir) for path in checkpoints.values()))

    def test_formal_report_blocks_a_second_formal_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            formal_output = root / "formal"
            formal_output.mkdir()
            (formal_output / "formal_test_report.json").write_text(
                "{}",
                encoding="utf-8",
            )
            resume2_report = root / "resume2.json"
            resume2_report.write_text("{}", encoding="utf-8")

            with (
                mock.patch.object(runner, "FORMAL_OUTPUT", formal_output),
                mock.patch.object(runner, "RESUME2_REPORT", resume2_report),
                mock.patch.object(runner, "_require_cache"),
                mock.patch.object(runner, "run_formal_experiment") as formal_run,
            ):
                with self.assertRaisesRegex(RuntimeError, "拒绝再次运行"):
                    runner.run_formal(resume=False)

        formal_run.assert_not_called()

    def test_overfit_resolves_auto_before_calling_core_function(self) -> None:
        batch = ((torch.zeros(2, 1, 3000),), torch.zeros(2, dtype=torch.long))
        dataset = mock.Mock()
        model = mock.Mock()
        config = runner._config(
            runner.PREFLIGHT_OUTPUT,
            raw_epochs=1,
            wave_epochs=1,
            fusion_epochs=1,
        )
        overfit_result = SimpleNamespace(
            losses=(1.0, 0.5),
            peak_cuda_memory_mib=None,
        )

        with (
            mock.patch.object(
                runner, "SleepEDFSequenceDataset", return_value=dataset
            ),
            mock.patch.object(runner, "SleepEDFRawFeatureNet", return_value=model),
            mock.patch.object(runner.torch.optim, "Adam", return_value=mock.Mock()),
            mock.patch.object(
                runner, "build_reproducible_loader", return_value=[batch]
            ),
            mock.patch.object(
                runner, "_resolve_device", return_value=torch.device("cpu")
            ),
            mock.patch.object(
                runner, "overfit_single_batch", return_value=overfit_result
            ) as overfit,
        ):
            result = runner._run_one_overfit_stage("raw", config)

        self.assertTrue(result["loss_decreased"])
        self.assertEqual(overfit.call_args.kwargs["device"], torch.device("cpu"))
        dataset.close.assert_called_once_with()

    def test_epoch1_resume_accepts_directory_created_before_first_checkpoint(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            preflight = root / "preflight"
            (preflight / "raw_pretrain").mkdir(parents=True)
            overfit_report = root / "overfit.json"
            overfit_report.write_text(
                json.dumps(
                    {
                        "stages": {
                            stage: {"loss_decreased": True}
                            for stage in ("raw", "wave", "fusion")
                        }
                    }
                ),
                encoding="utf-8",
            )
            epoch1_report = root / "epoch1.json"
            stage_results = {
                stage: StageTrainingResult(
                    stage=stage,
                    best_checkpoint=preflight / stage / "best.pt",
                    last_checkpoint=preflight / stage / "last.pt",
                    history_path=preflight / stage / "history.json",
                    history=({"epoch": 0},),
                )
                for stage in runner.STAGE_NAMES
            }

            with (
                mock.patch.object(runner, "PREFLIGHT_OUTPUT", preflight),
                mock.patch.object(runner, "OVERFIT_REPORT", overfit_report),
                mock.patch.object(runner, "EPOCH1_REPORT", epoch1_report),
                mock.patch.object(runner, "_require_cache"),
                mock.patch.object(
                    runner, "run_full_training", return_value=stage_results
                ) as full_training,
                mock.patch("builtins.print"),
            ):
                runner.run_epoch1(resume=True)

            self.assertTrue(epoch1_report.is_file())
            self.assertEqual(
                full_training.call_args.kwargs["resume_checkpoints"],
                {},
            )

    def test_formal_resume_restarts_when_only_empty_stage_directory_exists(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            formal_output = root / "formal"
            (formal_output / "raw_pretrain").mkdir(parents=True)
            resume2_report = root / "resume2.json"
            resume2_report.write_text("{}", encoding="utf-8")
            formal_result = SimpleNamespace(
                report_path=formal_output / "formal_test_report.json",
                training_seconds=1.0,
                peak_cuda_memory_mib=None,
                metrics=SimpleNamespace(
                    accuracy=0.1,
                    macro_f1=0.1,
                    kappa=0.0,
                ),
            )

            with (
                mock.patch.object(runner, "FORMAL_OUTPUT", formal_output),
                mock.patch.object(runner, "RESUME2_REPORT", resume2_report),
                mock.patch.object(runner, "_require_cache"),
                mock.patch.object(runner, "_cache_build_seconds", return_value=2.0),
                mock.patch.object(
                    runner, "run_formal_experiment", return_value=formal_result
                ) as formal_run,
                mock.patch("builtins.print"),
            ):
                runner.run_formal(resume=True)

            self.assertIsNone(
                formal_run.call_args.kwargs["resume_checkpoints"]
            )


if __name__ == "__main__":
    unittest.main()
