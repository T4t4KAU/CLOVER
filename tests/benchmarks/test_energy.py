from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks import energy


class EnergyProfilerIntelTest(unittest.TestCase):
    def test_rapl_meter_reads_joules_and_handles_wraparound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            energy_path = root / "energy_uj"
            energy_path.write_text("900")
            (root / "max_energy_range_uj").write_text("1000")

            meter = energy._IntelRaplMeter(energy_path)

            self.assertAlmostEqual(meter.read_j(), 0.0009)
            self.assertAlmostEqual(meter.delta_j(0.0009, 0.0001), 0.0002)

    def test_integrates_nvml_power_samples(self) -> None:
        joules = energy._integrate_power_samples(
            [(0.0, 10.0), (1.0, 20.0), (2.0, 10.0)],
            start_time=0.0,
            end_time=2.0,
        )

        self.assertAlmostEqual(joules, 30.0)

    def test_power_summary_reports_memory_and_rapl_details(self) -> None:
        summary = energy.PowerSummary(
            samples=1,
            elapsed_seconds=2.0,
            cpu_joules=10.0,
            gpu_joules=4.0,
            ane_joules=0.0,
            memory_joules=3.0,
            cpu_core_joules=6.0,
            cpu_uncore_joules=4.0,
            rapl_domains={"package": 10.0, "core": 6.0, "dram": 3.0},
        )

        payload = summary.to_dict()

        self.assertEqual(payload["memory_joules"], 3.0)
        self.assertEqual(payload["total_joules"], 17.0)
        self.assertEqual(payload["memory_avg_watts"], 1.5)
        self.assertEqual(payload["cpu_core_joules"], 6.0)
        self.assertEqual(payload["cpu_uncore_joules"], 4.0)
        self.assertEqual(payload["rapl_domains"]["dram"], 3.0)

    def test_intel_sampler_splits_rapl_domains(self) -> None:
        class FakeRapl:
            def read_j(self) -> dict[str, float]:
                return {"package": 100.0, "core": 60.0, "dram": 30.0}

            def delta_j(
                self,
                start: dict[str, float],
                end: dict[str, float],
            ) -> dict[str, float]:
                del start, end
                return {
                    "package": 10.0,
                    "core": 6.0,
                    "dram": 3.0,
                    "psys": 14.0,
                }

        with patch.object(energy, "_IntelRaplDomainMeters", FakeRapl):
            sampler = energy._IntelEnergySampler(
                interval_s=0.001,
                include_nvml=False,
            )
            summary = sampler.measure_window(0.0)

        self.assertEqual(summary.cpu_joules, 10.0)
        self.assertEqual(summary.memory_joules, 3.0)
        self.assertEqual(summary.cpu_core_joules, 6.0)
        self.assertEqual(summary.cpu_uncore_joules, 4.0)
        self.assertEqual(summary.psys_joules, 14.0)

    def test_collects_nvml_only_energy_samples(self) -> None:
        class FakeNvmlSampler:
            def __init__(self, *, interval_s: float) -> None:
                self.interval_s = interval_s

            def measure_window(self, seconds: float, stop_event: object | None = None) -> energy.PowerSummary:
                del seconds, stop_event
                return energy.PowerSummary(
                    samples=1,
                    elapsed_seconds=0.5,
                    cpu_joules=0.0,
                    gpu_joules=7.0,
                    ane_joules=0.0,
                )

            def close(self) -> None:
                pass

        with patch.object(energy, "_NvmlEnergySampler", FakeNvmlSampler):
            summary = energy._collect_fixed_sample_count(
                sample_ms=100,
                sample_count=2,
                backend=energy.NVIDIA_NVML_BACKEND,
                password=None,
            )

        self.assertEqual(summary.samples, 2)
        self.assertEqual(summary.cpu_joules, 0.0)
        self.assertEqual(summary.gpu_joules, 14.0)
        self.assertEqual(energy._samplers_for_backend(energy.NVIDIA_NVML_BACKEND), "nvidia_nvml")

    def test_detects_linux_rapl_cpu_only_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            energy_path = Path(tmp) / "energy_uj"
            energy_path.write_text("123456")

            with (
                patch.object(energy.platform, "system", return_value="Linux"),
                patch.object(energy.platform, "machine", return_value="x86_64"),
                patch.object(energy.platform, "release", return_value="6.0"),
                patch.object(energy.platform, "version", return_value="#1"),
                patch.object(energy.platform, "processor", return_value=""),
                patch.object(energy, "_linux_cpu_brand", return_value="Intel(R) Xeon"),
                patch.object(energy, "_find_rapl_energy_path", return_value=energy_path),
                patch.object(energy, "_nvml_available", return_value=False),
            ):
                platform_info = energy.detect_hardware_platform()

        self.assertTrue(platform_info.supported)
        self.assertEqual(platform_info.backend, energy.INTEL_RAPL_BACKEND)
        self.assertEqual(platform_info.cpu_brand, "Intel(R) Xeon")

    def test_detects_linux_rapl_nvml_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            energy_path = Path(tmp) / "energy_uj"
            energy_path.write_text("123456")

            with (
                patch.object(energy.platform, "system", return_value="Linux"),
                patch.object(energy.platform, "machine", return_value="x86_64"),
                patch.object(energy.platform, "release", return_value="6.0"),
                patch.object(energy.platform, "version", return_value="#1"),
                patch.object(energy.platform, "processor", return_value=""),
                patch.object(energy, "_linux_cpu_brand", return_value="Intel(R) Xeon"),
                patch.object(energy, "_find_rapl_energy_path", return_value=energy_path),
                patch.object(energy, "_nvml_available", return_value=True),
            ):
                platform_info = energy.detect_hardware_platform()

        self.assertTrue(platform_info.supported)
        self.assertEqual(platform_info.backend, energy.INTEL_RAPL_NVML_BACKEND)

    def test_detects_linux_nvml_only_backend_without_rapl(self) -> None:
        with (
            patch.object(energy.platform, "system", return_value="Linux"),
            patch.object(energy.platform, "machine", return_value="x86_64"),
            patch.object(energy.platform, "release", return_value="6.0"),
            patch.object(energy.platform, "version", return_value="#1"),
            patch.object(energy.platform, "processor", return_value=""),
            patch.object(energy, "_linux_cpu_brand", return_value="Intel(R) Xeon"),
            patch.object(energy, "_find_rapl_energy_path", return_value=None),
            patch.object(energy, "_nvml_available", return_value=True),
        ):
            platform_info = energy.detect_hardware_platform()

        self.assertTrue(platform_info.supported)
        self.assertEqual(platform_info.backend, energy.NVIDIA_NVML_BACKEND)

    def test_linux_without_rapl_is_unsupported(self) -> None:
        with (
            patch.object(energy.platform, "system", return_value="Linux"),
            patch.object(energy.platform, "machine", return_value="x86_64"),
            patch.object(energy.platform, "release", return_value="6.0"),
            patch.object(energy.platform, "version", return_value="#1"),
            patch.object(energy.platform, "processor", return_value=""),
            patch.object(energy, "_linux_cpu_brand", return_value="Intel(R) Xeon"),
            patch.object(energy, "_find_rapl_energy_path", return_value=None),
            patch.object(energy, "_nvml_available", return_value=False),
        ):
            platform_info = energy.detect_hardware_platform()

        self.assertFalse(platform_info.supported)
        self.assertEqual(platform_info.backend, "unsupported")
        self.assertEqual(platform_info.unsupported_reason, "Intel RAPL energy_uj not found")


if __name__ == "__main__":
    unittest.main()
