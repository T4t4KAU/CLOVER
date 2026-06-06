"""Small ModelScope loading helpers used by benchmark downloaders."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any


def load_modelscope_rows(
    *,
    repo_id: str,
    subset_name: str | None,
    splits: Sequence[str],
) -> list[dict[str, Any]]:
    """Load rows from ModelScope with MsDataset.load."""

    MsDataset = _import_ms_dataset()
    rows: list[dict[str, Any]] = []
    for split in splits:
        kwargs: dict[str, Any] = {"split": split}
        if subset_name:
            kwargs["subset_name"] = subset_name
        dataset = MsDataset.load(repo_id, **kwargs)
        for row in _iter_dataset_rows(dataset):
            payload = _row_to_dict(row)
            payload.setdefault("split", split)
            rows.append(payload)
    return rows


def download_modelscope_dataset_snapshot(
    *,
    repo_id: str,
    local_dir: str | Path | None = None,
) -> Path:
    """Download a ModelScope dataset repository snapshot and return its path."""

    try:
        from modelscope import dataset_snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised by user envs.
        raise RuntimeError(
            "Missing optional dependency 'modelscope'. Install it with "
            "`pip install modelscope` or `pip install .[modelscope]`."
        ) from exc

    kwargs: dict[str, Any] = {}
    if local_dir is not None:
        kwargs["local_dir"] = str(Path(local_dir).expanduser())
    return Path(dataset_snapshot_download(repo_id, **kwargs)).expanduser().resolve()


def _import_ms_dataset() -> Any:
    try:
        from modelscope.msdatasets import MsDataset

        return MsDataset
    except ImportError:
        try:
            from modelscope import MsDataset

            return MsDataset
        except ImportError as exc:  # pragma: no cover - exercised by user envs.
            raise RuntimeError(
                "Missing optional dependency 'modelscope'. Install it with "
                "`pip install modelscope` or `pip install .[modelscope]`."
            ) from exc


def _iter_dataset_rows(dataset: Any) -> Iterable[Any]:
    if hasattr(dataset, "to_hf_dataset"):
        dataset = dataset.to_hf_dataset()
    if isinstance(dataset, dict):
        for value in dataset.values():
            yield from _iter_dataset_rows(value)
        return
    for row in dataset:
        yield row


def _row_to_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception as exc:
        raise TypeError(f"ModelScope row is not dict-like: {type(row)!r}") from exc
