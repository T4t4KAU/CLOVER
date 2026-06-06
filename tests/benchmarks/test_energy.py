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

    def test_linux_without_rapl_is_unsupported(self) -> None:
        with (
            patch.object(energy.platform, "system", return_value="Linux"),
            patch.object(energy.platform, "machine", return_value="x86_64"),
            patch.object(energy.platform, "release", return_value="6.0"),
            patch.object(energy.platform, "version", return_value="#1"),
            patch.object(energy.platform, "processor", return_value=""),
            patch.object(energy, "_linux_cpu_brand", return_value="Intel(R) Xeon"),
            patch.object(energy, "_find_rapl_energy_path", return_value=None),
        ):
            platform_info = energy.detect_hardware_platform()

        self.assertFalse(platform_info.supported)
        self.assertEqual(platform_info.backend, "unsupported")
        self.assertEqual(platform_info.unsupported_reason, "Intel RAPL energy_uj not found")


if __name__ == "__main__":
    unittest.main()
