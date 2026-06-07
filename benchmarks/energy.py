"""Optional benchmark energy profiling helpers."""

from __future__ import annotations

import ctypes
import os
import platform
import plistlib
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


POWERMETRICS_BACKEND = "powermetrics_apple_silicon"
POWERMETRICS_SAMPLERS = "cpu_power,gpu_power,ane_power"
INTEL_RAPL_BACKEND = "intel_rapl"
INTEL_RAPL_NVML_BACKEND = "intel_rapl_nvml"
NVIDIA_NVML_BACKEND = "nvidia_nvml"
INTEL_RAPL_SAMPLERS = "intel_rapl"
INTEL_RAPL_NVML_SAMPLERS = "intel_rapl,nvidia_nvml"
NVIDIA_NVML_SAMPLERS = "nvidia_nvml"


@dataclass(frozen=True)
class HardwarePlatform:
    system: str
    machine: str
    release: str
    version: str
    model_identifier: str | None
    cpu_brand: str | None
    backend: str
    supported: bool
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "machine": self.machine,
            "release": self.release,
            "version": self.version,
            "model_identifier": self.model_identifier,
            "cpu_brand": self.cpu_brand,
            "backend": self.backend,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class PowerSummary:
    samples: int
    elapsed_seconds: float
    cpu_joules: float
    gpu_joules: float
    ane_joules: float
    memory_joules: float = 0.0
    cpu_core_joules: float | None = None
    cpu_uncore_joules: float | None = None
    psys_joules: float | None = None
    rapl_domains: dict[str, float] | None = None

    @property
    def total_joules(self) -> float:
        return (
            self.cpu_joules
            + self.gpu_joules
            + self.ane_joules
            + self.memory_joules
        )

    def avg_watts(self, joules: float) -> float:
        return joules / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "samples": self.samples,
            "elapsed_seconds": self.elapsed_seconds,
            "cpu_joules": self.cpu_joules,
            "gpu_joules": self.gpu_joules,
            "ane_joules": self.ane_joules,
            "memory_joules": self.memory_joules,
            "total_joules": self.total_joules,
            "cpu_avg_watts": self.avg_watts(self.cpu_joules),
            "gpu_avg_watts": self.avg_watts(self.gpu_joules),
            "ane_avg_watts": self.avg_watts(self.ane_joules),
            "memory_avg_watts": self.avg_watts(self.memory_joules),
            "total_avg_watts": self.avg_watts(self.total_joules),
            "total_wh": self.total_joules / 3600.0,
        }
        if self.cpu_core_joules is not None:
            payload["cpu_core_joules"] = self.cpu_core_joules
            payload["cpu_core_avg_watts"] = self.avg_watts(self.cpu_core_joules)
        if self.cpu_uncore_joules is not None:
            payload["cpu_uncore_joules"] = self.cpu_uncore_joules
            payload["cpu_uncore_avg_watts"] = self.avg_watts(
                self.cpu_uncore_joules
            )
        if self.psys_joules is not None:
            payload["psys_joules"] = self.psys_joules
            payload["psys_avg_watts"] = self.avg_watts(self.psys_joules)
        if self.rapl_domains:
            payload["rapl_domains"] = dict(sorted(self.rapl_domains.items()))
        return payload


class EnergyProfiler:
    """Measure gross eval energy on supported local hardware.

    On Apple Silicon macOS this uses powermetrics CPU/GPU/ANE subsystem
    estimates. On Linux Intel hosts this uses RAPL package energy for CPU and,
    when available, NVIDIA NVML instantaneous power for GPU integration. On
    Linux GPU-only hosts where RAPL is unavailable but NVML is readable, this
    records GPU energy only. On other platforms it records the hardware
    metadata and reports that direct measurement is unsupported.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        sample_ms: int = 500,
        baseline_seconds: float = 0.0,
        password_env: str = "CLOVER_POWERMETRICS_PASSWORD",
    ) -> None:
        self.enabled = enabled
        self.sample_ms = sample_ms
        self.baseline_seconds = baseline_seconds
        self.password_env = password_env
        self.platform = detect_hardware_platform()
        self.summary: dict[str, Any] | None = None
        self._started = 0.0
        self._baseline: PowerSummary | None = None
        self._sample_summaries: list[PowerSummary] = []
        self._sample_errors: list[str] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "EnergyProfiler":
        if not self.enabled:
            self.summary = {
                "enabled": False,
                "hardware_platform": self.platform.to_dict(),
            }
            return self
        if not self.platform.supported:
            self.summary = {
                "enabled": True,
                "status": "unsupported",
                "hardware_platform": self.platform.to_dict(),
            }
            return self
        if self.sample_ms <= 0:
            self.summary = {
                "enabled": True,
                "status": "error",
                "error": "sample_ms must be positive",
                "hardware_platform": self.platform.to_dict(),
            }
            return self

        password = (
            _password_from_env(self.password_env)
            if self.platform.backend == POWERMETRICS_BACKEND
            else None
        )
        try:
            if self.baseline_seconds > 0:
                self._baseline = _collect_fixed_window(
                    sample_ms=self.sample_ms,
                    seconds=self.baseline_seconds,
                    backend=self.platform.backend,
                    password=password,
                )
            self._thread = threading.Thread(
                target=self._sample_loop,
                args=(password,),
                name="clover-energy-profiler",
                daemon=True,
            )
            self._started = time.perf_counter()
            self._thread.start()
        except Exception as exc:  # noqa: BLE001 - keep eval usable without energy.
            self.summary = {
                "enabled": True,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "hardware_platform": self.platform.to_dict(),
            }
            self._cleanup()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.summary is not None and self.summary.get("status") in {
            "error",
            "unsupported",
        }:
            return
        if not self.enabled or self._thread is None:
            return

        measured_wall_seconds = time.perf_counter() - self._started
        self._stop_event.set()
        self._thread.join(timeout=max(5.0, self.sample_ms / 1000.0 * 4))

        try:
            power = _combine_power_summaries(self._sample_summaries)
            payload: dict[str, Any] = {
                "enabled": True,
                "status": "ok" if power.samples > 0 else "no_samples",
                "backend": self.platform.backend,
                "samplers": _samplers_for_backend(self.platform.backend),
                "sample_ms": self.sample_ms,
                "hardware_platform": self.platform.to_dict(),
                "measured_wall_seconds": measured_wall_seconds,
                "gross": power.to_dict(),
            }
            if self._sample_errors:
                payload["sample_errors"] = self._sample_errors[:5]
            if self._baseline is not None:
                baseline = self._baseline.to_dict()
                baseline_w = self._baseline.avg_watts(self._baseline.total_joules)
                net_j = power.total_joules - baseline_w * power.elapsed_seconds
                payload["baseline"] = baseline
                payload["baseline_adjusted"] = {
                    "total_joules": net_j,
                    "total_wh": net_j / 3600.0,
                    "baseline_total_avg_watts": baseline_w,
                }
            self.summary = payload
        except Exception as parse_exc:  # noqa: BLE001
            self.summary = {
                "enabled": True,
                "status": "error",
                "error": f"{type(parse_exc).__name__}: {parse_exc}",
                "hardware_platform": self.platform.to_dict(),
                "measured_wall_seconds": measured_wall_seconds,
            }
        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        self._stop_event.set()

    def _sample_loop(self, password: str | None) -> None:
        if _is_intel_energy_backend(self.platform.backend):
            self._intel_sample_loop()
            return
        if self.platform.backend == NVIDIA_NVML_BACKEND:
            self._nvml_sample_loop()
            return

        while not self._stop_event.is_set():
            try:
                summary = _collect_fixed_sample_count(
                    sample_ms=self.sample_ms,
                    sample_count=1,
                    backend=self.platform.backend,
                    password=password,
                )
                if summary.samples > 0:
                    self._sample_summaries.append(summary)
            except Exception as exc:  # noqa: BLE001
                self._sample_errors.append(f"{type(exc).__name__}: {exc}")
                self._stop_event.set()

    def _intel_sample_loop(self) -> None:
        interval_s = min(0.1, max(0.02, self.sample_ms / 1000.0 / 4.0))
        try:
            sampler = _IntelEnergySampler(
                interval_s=interval_s,
                include_nvml=self.platform.backend == INTEL_RAPL_NVML_BACKEND,
            )
        except Exception as exc:  # noqa: BLE001
            self._sample_errors.append(f"{type(exc).__name__}: {exc}")
            self._stop_event.set()
            return

        try:
            while not self._stop_event.is_set():
                summary = sampler.measure_window(
                    self.sample_ms / 1000.0,
                    stop_event=self._stop_event,
                )
                if summary.samples > 0:
                    self._sample_summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            self._sample_errors.append(f"{type(exc).__name__}: {exc}")
            self._stop_event.set()
        finally:
            sampler.close()

    def _nvml_sample_loop(self) -> None:
        interval_s = min(0.1, max(0.02, self.sample_ms / 1000.0 / 4.0))
        try:
            sampler = _NvmlEnergySampler(interval_s=interval_s)
        except Exception as exc:  # noqa: BLE001
            self._sample_errors.append(f"{type(exc).__name__}: {exc}")
            self._stop_event.set()
            return

        try:
            while not self._stop_event.is_set():
                summary = sampler.measure_window(
                    self.sample_ms / 1000.0,
                    stop_event=self._stop_event,
                )
                if summary.samples > 0:
                    self._sample_summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            self._sample_errors.append(f"{type(exc).__name__}: {exc}")
            self._stop_event.set()
        finally:
            sampler.close()


def detect_hardware_platform() -> HardwarePlatform:
    system = platform.system()
    machine = platform.machine()
    release = platform.release()
    version = platform.version()
    model_identifier = _sysctl_value("hw.model") if system == "Darwin" else None
    if system == "Darwin":
        cpu_brand = _sysctl_value("machdep.cpu.brand_string") or platform.processor() or None
    elif system == "Linux":
        cpu_brand = _linux_cpu_brand() or platform.processor() or None
    else:
        cpu_brand = platform.processor() or None

    if system == "Linux":
        rapl_path = _find_rapl_energy_path()
        if rapl_path is None:
            if _nvml_available():
                return HardwarePlatform(
                    system=system,
                    machine=machine,
                    release=release,
                    version=version,
                    model_identifier=model_identifier,
                    cpu_brand=cpu_brand,
                    backend=NVIDIA_NVML_BACKEND,
                    supported=True,
                )
            return HardwarePlatform(
                system=system,
                machine=machine,
                release=release,
                version=version,
                model_identifier=model_identifier,
                cpu_brand=cpu_brand,
                backend="unsupported",
                supported=False,
                unsupported_reason="Intel RAPL energy_uj not found",
            )
        try:
            _IntelRaplMeter(rapl_path).read_j()
        except Exception as exc:  # noqa: BLE001 - surface a concise capability probe.
            if _nvml_available():
                return HardwarePlatform(
                    system=system,
                    machine=machine,
                    release=release,
                    version=version,
                    model_identifier=model_identifier,
                    cpu_brand=cpu_brand,
                    backend=NVIDIA_NVML_BACKEND,
                    supported=True,
                )
            return HardwarePlatform(
                system=system,
                machine=machine,
                release=release,
                version=version,
                model_identifier=model_identifier,
                cpu_brand=cpu_brand,
                backend="unsupported",
                supported=False,
                unsupported_reason=f"Intel RAPL energy_uj is not readable: {exc}",
            )
        backend = INTEL_RAPL_NVML_BACKEND if _nvml_available() else INTEL_RAPL_BACKEND
        return HardwarePlatform(
            system=system,
            machine=machine,
            release=release,
            version=version,
            model_identifier=model_identifier,
            cpu_brand=cpu_brand,
            backend=backend,
            supported=True,
        )

    if system != "Darwin":
        return HardwarePlatform(
            system=system,
            machine=machine,
            release=release,
            version=version,
            model_identifier=model_identifier,
            cpu_brand=cpu_brand,
            backend="unsupported",
            supported=False,
            unsupported_reason=(
                "supported energy backends require macOS powermetrics "
                "or Linux Intel RAPL"
            ),
        )
    if machine not in {"arm64", "aarch64"}:
        return HardwarePlatform(
            system=system,
            machine=machine,
            release=release,
            version=version,
            model_identifier=model_identifier,
            cpu_brand=cpu_brand,
            backend="unsupported",
            supported=False,
            unsupported_reason="Apple Silicon powermetrics backend requires arm64",
        )
    if shutil.which("powermetrics") is None:
        return HardwarePlatform(
            system=system,
            machine=machine,
            release=release,
            version=version,
            model_identifier=model_identifier,
            cpu_brand=cpu_brand,
            backend="unsupported",
            supported=False,
            unsupported_reason="powermetrics executable not found",
        )
    return HardwarePlatform(
        system=system,
        machine=machine,
        release=release,
        version=version,
        model_identifier=model_identifier,
        cpu_brand=cpu_brand,
        backend=POWERMETRICS_BACKEND,
        supported=True,
    )


def parse_plists(raw: bytes) -> list[dict[str, Any]]:
    docs = []
    for part in raw.split(b"\x00"):
        start = 0
        while True:
            xml_start = part.find(b"<?xml", start)
            if xml_start < 0:
                break
            xml_end = part.find(b"</plist>", xml_start)
            if xml_end < 0:
                break
            xml_end += len(b"</plist>")
            docs.append(plistlib.loads(part[xml_start:xml_end]))
            start = xml_end
    return docs


def summarize_power(samples: list[dict[str, Any]]) -> PowerSummary:
    cpu_j = gpu_j = ane_j = elapsed_s = 0.0
    for sample in samples:
        elapsed = float(sample.get("elapsed_ns", 0)) / 1e9
        processor = sample.get("processor", {})
        elapsed_s += elapsed
        cpu_j += _subsystem_energy_j(processor, "cpu", elapsed)
        gpu_j += _subsystem_energy_j(processor, "gpu", elapsed)
        ane_j += _subsystem_energy_j(processor, "ane", elapsed)
    return PowerSummary(
        samples=len(samples),
        elapsed_seconds=elapsed_s,
        cpu_joules=cpu_j,
        gpu_joules=gpu_j,
        ane_joules=ane_j,
    )


def _combine_power_summaries(summaries: list[PowerSummary]) -> PowerSummary:
    return PowerSummary(
        samples=sum(summary.samples for summary in summaries),
        elapsed_seconds=sum(summary.elapsed_seconds for summary in summaries),
        cpu_joules=sum(summary.cpu_joules for summary in summaries),
        gpu_joules=sum(summary.gpu_joules for summary in summaries),
        ane_joules=sum(summary.ane_joules for summary in summaries),
        memory_joules=sum(summary.memory_joules for summary in summaries),
        cpu_core_joules=_sum_optional(
            summary.cpu_core_joules for summary in summaries
        ),
        cpu_uncore_joules=_sum_optional(
            summary.cpu_uncore_joules for summary in summaries
        ),
        psys_joules=_sum_optional(summary.psys_joules for summary in summaries),
        rapl_domains=_combine_domain_totals(
            summary.rapl_domains for summary in summaries
        ),
    )


def _sum_optional(values: Any) -> float | None:
    total = 0.0
    seen = False
    for value in values:
        if value is None:
            continue
        total += float(value)
        seen = True
    return total if seen else None


def _combine_domain_totals(values: Any) -> dict[str, float] | None:
    totals: dict[str, float] = {}
    for domains in values:
        if not domains:
            continue
        for key, value in domains.items():
            totals[str(key)] = totals.get(str(key), 0.0) + float(value)
    return totals or None


class _IntelRaplMeter:
    def __init__(self, energy_path: Path | None = None) -> None:
        path = energy_path or _find_rapl_energy_path()
        if path is None:
            raise RuntimeError("cannot find Intel RAPL energy_uj")
        self.path = path
        max_path = self.path.parent / "max_energy_range_uj"
        self.max_uj = _read_int(max_path) if max_path.exists() else 0

    def read_j(self) -> float:
        return int(self.path.read_text().strip()) / 1_000_000.0

    def delta_j(self, start_j: float, end_j: float) -> float:
        if end_j >= start_j:
            return end_j - start_j
        if self.max_uj <= 0:
            return 0.0
        return (self.max_uj / 1_000_000.0 - start_j) + end_j


class _IntelRaplDomainMeters:
    def __init__(self) -> None:
        paths = _find_rapl_domain_energy_paths()
        if not paths:
            package_path = _find_rapl_energy_path()
            if package_path is None:
                raise RuntimeError("cannot find Intel RAPL energy_uj")
            paths = {"package": package_path}
        self.meters = {
            name: _IntelRaplMeter(path)
            for name, path in sorted(paths.items())
        }

    def read_j(self) -> dict[str, float]:
        return {name: meter.read_j() for name, meter in self.meters.items()}

    def delta_j(
        self,
        start: dict[str, float],
        end: dict[str, float],
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        for name, meter in self.meters.items():
            if name not in start or name not in end:
                continue
            output[name] = max(0.0, meter.delta_j(start[name], end[name]))
        return output


class _NvmlMeter:
    def __init__(self, device_index: int = 0) -> None:
        self.lib: Any | None = None
        for name in ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if self.lib is None:
            raise RuntimeError("cannot load libnvidia-ml")

        try:
            ret = self.lib.nvmlInit_v2()
        except AttributeError as exc:
            raise RuntimeError("nvmlInit_v2 symbol not found") from exc
        if ret != 0:
            raise RuntimeError(f"nvmlInit_v2 failed: {ret}")
        self.handle = ctypes.c_void_p()
        try:
            ret = self.lib.nvmlDeviceGetHandleByIndex_v2(
                device_index,
                ctypes.byref(self.handle),
            )
        except AttributeError as exc:
            self.close()
            raise RuntimeError("nvmlDeviceGetHandleByIndex_v2 symbol not found") from exc
        if ret != 0:
            self.close()
            raise RuntimeError(f"nvmlDeviceGetHandleByIndex_v2 failed: {ret}")

    def power_w(self) -> float:
        if self.lib is None:
            return 0.0
        power_mw = ctypes.c_uint()
        try:
            ret = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power_mw))
        except AttributeError:
            return 0.0
        if ret != 0:
            return 0.0
        return power_mw.value / 1000.0

    def close(self) -> None:
        if self.lib is None:
            return
        shutdown = getattr(self.lib, "nvmlShutdown", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:
                pass
        self.lib = None


class _IntelEnergySampler:
    def __init__(self, *, interval_s: float, include_nvml: bool) -> None:
        self.rapl = _IntelRaplDomainMeters()
        self.gpu: _NvmlMeter | None
        if not include_nvml:
            self.gpu = None
        else:
            try:
                self.gpu = _NvmlMeter()
            except RuntimeError:
                self.gpu = None
        self.interval_s = interval_s

    def close(self) -> None:
        if self.gpu is not None:
            self.gpu.close()
            self.gpu = None

    def measure_window(
        self,
        seconds: float,
        stop_event: threading.Event | None = None,
    ) -> PowerSummary:
        rapl_start_j = self.rapl.read_j()
        t0 = time.perf_counter()
        deadline = t0 + max(0.0, seconds)
        gpu_samples: list[tuple[float, float]] = []
        if self.gpu is not None:
            gpu_samples.append((t0, self.gpu.power_w()))

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            sleep_s = min(self.interval_s, remaining)
            if stop_event is None:
                time.sleep(sleep_s)
            elif stop_event.wait(sleep_s):
                break
            if self.gpu is not None:
                gpu_samples.append((time.perf_counter(), self.gpu.power_w()))

        t1 = time.perf_counter()
        rapl_end_j = self.rapl.read_j()
        rapl_delta = self.rapl.delta_j(rapl_start_j, rapl_end_j)
        cpu_package_j = rapl_delta.get("package", 0.0)
        cpu_core_j = rapl_delta.get("core")
        cpu_uncore_j = (
            max(0.0, cpu_package_j - cpu_core_j)
            if cpu_core_j is not None
            else None
        )
        memory_j = rapl_delta.get("dram", 0.0)
        gpu_j = _integrate_power_samples(gpu_samples, start_time=t0, end_time=t1)
        return PowerSummary(
            samples=1,
            elapsed_seconds=t1 - t0,
            cpu_joules=cpu_package_j,
            gpu_joules=gpu_j,
            ane_joules=0.0,
            memory_joules=memory_j,
            cpu_core_joules=cpu_core_j,
            cpu_uncore_joules=cpu_uncore_j,
            psys_joules=rapl_delta.get("psys"),
            rapl_domains=rapl_delta,
        )


class _NvmlEnergySampler:
    def __init__(self, *, interval_s: float) -> None:
        self.gpu = _NvmlMeter()
        self.interval_s = interval_s

    def close(self) -> None:
        self.gpu.close()

    def measure_window(
        self,
        seconds: float,
        stop_event: threading.Event | None = None,
    ) -> PowerSummary:
        t0 = time.perf_counter()
        deadline = t0 + max(0.0, seconds)
        gpu_samples: list[tuple[float, float]] = [(t0, self.gpu.power_w())]

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            sleep_s = min(self.interval_s, remaining)
            if stop_event is None:
                time.sleep(sleep_s)
            elif stop_event.wait(sleep_s):
                break
            gpu_samples.append((time.perf_counter(), self.gpu.power_w()))

        t1 = time.perf_counter()
        gpu_j = _integrate_power_samples(gpu_samples, start_time=t0, end_time=t1)
        return PowerSummary(
            samples=1,
            elapsed_seconds=t1 - t0,
            cpu_joules=0.0,
            gpu_joules=gpu_j,
            ane_joules=0.0,
        )


def _collect_fixed_window(
    *,
    sample_ms: int,
    seconds: float,
    backend: str,
    password: str | None,
) -> PowerSummary:
    sample_count = max(1, int(seconds * 1000 / sample_ms + 0.999))
    return _collect_fixed_sample_count(
        sample_ms=sample_ms,
        sample_count=sample_count,
        backend=backend,
        password=password,
    )


def _collect_fixed_sample_count(
    *,
    sample_ms: int,
    sample_count: int,
    backend: str,
    password: str | None,
) -> PowerSummary:
    if backend == POWERMETRICS_BACKEND:
        return _collect_powermetrics_fixed_sample_count(
            sample_ms=sample_ms,
            sample_count=sample_count,
            password=password,
        )
    if _is_intel_energy_backend(backend):
        return _collect_intel_fixed_sample_count(
            sample_ms=sample_ms,
            sample_count=sample_count,
            include_nvml=backend == INTEL_RAPL_NVML_BACKEND,
        )
    if backend == NVIDIA_NVML_BACKEND:
        return _collect_nvml_fixed_sample_count(
            sample_ms=sample_ms,
            sample_count=sample_count,
        )
    raise RuntimeError(f"unsupported energy backend: {backend}")


def _collect_powermetrics_fixed_sample_count(
    *,
    sample_ms: int,
    sample_count: int,
    password: str | None,
) -> PowerSummary:
    with tempfile.NamedTemporaryFile(prefix="clover-energy-baseline-", suffix=".plist") as output:
        proc = _start_powermetrics(
            sample_ms=sample_ms,
            sample_count=sample_count,
            password=password,
            output_file=Path(output.name),
        )
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(errors="replace").strip())
        output.seek(0)
        return summarize_power(parse_plists(output.read()))


def _collect_intel_fixed_sample_count(
    *,
    sample_ms: int,
    sample_count: int,
    include_nvml: bool,
) -> PowerSummary:
    interval_s = min(0.1, max(0.02, sample_ms / 1000.0 / 4.0))
    sampler = _IntelEnergySampler(interval_s=interval_s, include_nvml=include_nvml)
    try:
        summaries = [
            sampler.measure_window(sample_ms / 1000.0)
            for _ in range(max(0, sample_count))
        ]
    finally:
        sampler.close()
    return _combine_power_summaries(summaries)


def _collect_nvml_fixed_sample_count(
    *,
    sample_ms: int,
    sample_count: int,
) -> PowerSummary:
    interval_s = min(0.1, max(0.02, sample_ms / 1000.0 / 4.0))
    sampler = _NvmlEnergySampler(interval_s=interval_s)
    try:
        summaries = [
            sampler.measure_window(sample_ms / 1000.0)
            for _ in range(max(0, sample_count))
        ]
    finally:
        sampler.close()
    return _combine_power_summaries(summaries)


def _start_powermetrics(
    *,
    sample_ms: int,
    sample_count: int,
    password: str | None,
    output_file: Path,
) -> subprocess.Popen[bytes]:
    cmd = ["sudo"]
    if password is None:
        cmd.append("-n")
    else:
        cmd.extend(["-S", "-p", ""])
    cmd.extend(
        [
            "powermetrics",
            "--samplers",
            POWERMETRICS_SAMPLERS,
            "-i",
            str(sample_ms),
            "-n",
            str(sample_count),
            "-f",
            "plist",
            "-o",
            str(output_file),
        ]
    )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if password is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    if password is not None and proc.stdin is not None:
        proc.stdin.write((password + "\n").encode())
        proc.stdin.close()
        proc.stdin = None
    return proc


def _subsystem_energy_j(processor: dict[str, Any], name: str, elapsed_s: float) -> float:
    energy_key = f"{name}_energy"
    power_key = f"{name}_power"
    if energy_key in processor:
        return float(processor[energy_key]) / 1000.0
    if power_key in processor:
        return float(processor[power_key]) * elapsed_s / 1000.0
    return 0.0


def _password_from_env(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _samplers_for_backend(backend: str) -> str:
    if backend == POWERMETRICS_BACKEND:
        return POWERMETRICS_SAMPLERS
    if backend == INTEL_RAPL_NVML_BACKEND:
        return INTEL_RAPL_NVML_SAMPLERS
    if backend == INTEL_RAPL_BACKEND:
        return INTEL_RAPL_SAMPLERS
    if backend == NVIDIA_NVML_BACKEND:
        return NVIDIA_NVML_SAMPLERS
    return "unsupported"


def _is_intel_energy_backend(backend: str) -> bool:
    return backend in {INTEL_RAPL_BACKEND, INTEL_RAPL_NVML_BACKEND}


def _find_rapl_energy_path() -> Path | None:
    candidates = [
        Path("/sys/class/powercap/intel-rapl:0/energy_uj"),
        Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"),
    ]
    for path in candidates:
        if path.exists():
            return path

    root = Path("/sys/class/powercap")
    if not root.exists():
        return None

    energy_paths = sorted(root.glob("intel-rapl*/energy_uj"))
    for path in energy_paths:
        name = _read_text(path.parent / "name")
        if name and name.startswith("package-"):
            return path
    return energy_paths[0] if energy_paths else None


def _find_rapl_domain_energy_paths() -> dict[str, Path]:
    root = Path("/sys/class/powercap")
    if not root.exists():
        return {}
    output: dict[str, Path] = {}
    for directory in sorted(root.glob("intel-rapl*")):
        if not directory.is_dir():
            continue
        energy_path = directory / "energy_uj"
        if not energy_path.exists():
            continue
        raw_name = _read_text(directory / "name") or directory.name
        name = _normalize_rapl_domain_name(raw_name)
        if name in output:
            name = _unique_rapl_domain_name(name, output)
        output[name] = energy_path
    return output


def _normalize_rapl_domain_name(name: str) -> str:
    normalized = name.strip().lower()
    if normalized.startswith("package-"):
        return "package"
    if normalized in {"core", "dram", "uncore", "psys"}:
        return normalized
    return "".join(
        char if char.isalnum() else "_"
        for char in normalized
    ).strip("_") or "unknown"


def _unique_rapl_domain_name(name: str, existing: dict[str, Path]) -> str:
    index = 1
    while f"{name}_{index}" in existing:
        index += 1
    return f"{name}_{index}"


def _nvml_available() -> bool:
    meter: _NvmlMeter | None = None
    try:
        meter = _NvmlMeter()
        meter.power_w()
        return True
    except RuntimeError:
        return False
    finally:
        if meter is not None:
            meter.close()


def _integrate_power_samples(
    samples: list[tuple[float, float]],
    *,
    start_time: float,
    end_time: float,
) -> float:
    if not samples or end_time <= start_time:
        return 0.0

    points = [
        (min(max(timestamp, start_time), end_time), max(0.0, power_w))
        for timestamp, power_w in samples
    ]
    points.sort(key=lambda item: item[0])

    timeline: list[tuple[float, float]] = [(start_time, points[0][1])]
    for timestamp, power_w in points:
        if start_time < timestamp < end_time:
            timeline.append((timestamp, power_w))
    if timeline[-1][0] < end_time:
        timeline.append((end_time, points[-1][1]))

    total_j = 0.0
    for (last_time, last_power), (timestamp, power_w) in zip(timeline, timeline[1:]):
        if timestamp > last_time:
            total_j += ((last_power + power_w) / 2.0) * (timestamp - last_time)
    return total_j


def _linux_cpu_brand() -> str | None:
    text = _read_text(Path("/proc/cpuinfo"))
    if text is None:
        return None
    for line in text.splitlines():
        if line.lower().startswith("model name"):
            _, _, value = line.partition(":")
            return value.strip() or None
    return None


def _read_int(path: Path) -> int:
    return int(path.read_text().strip())


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _sysctl_value(name: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", name],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
