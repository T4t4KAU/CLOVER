"""Automatic lifecycle policy for executor-owned artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResourceLifecycle:
    """Track when temporary node artifacts can be released."""

    configured: bool = False
    remaining_consumers: dict[str, int] = field(default_factory=dict)
    retained_artifacts: set[str] = field(default_factory=set)

    def configure(
        self,
        *,
        consumers_by_artifact: dict[str, int],
        retained_artifacts: set[str],
    ) -> None:
        self.remaining_consumers = {
            str(name): int(count)
            for name, count in consumers_by_artifact.items()
            if int(count) > 0
        }
        self.retained_artifacts = {str(name) for name in retained_artifacts}
        self.configured = True

    def is_retained(self, name: str) -> bool:
        return name in self.retained_artifacts

    def releasable_after_put(self, name: str) -> bool:
        return self._can_release(name)

    def mark_consumed(self, names: list[str] | tuple[str, ...]) -> list[str]:
        if not self.configured:
            return []
        releasable: list[str] = []
        for name in names:
            if name not in self.remaining_consumers:
                continue
            self.remaining_consumers[name] -= 1
            if self.remaining_consumers[name] <= 0:
                self.remaining_consumers.pop(name, None)
                if self._can_release(name):
                    releasable.append(name)
        return releasable

    def _can_release(self, name: str) -> bool:
        if not self.configured:
            return False
        if name in self.retained_artifacts:
            return False
        return self.remaining_consumers.get(name, 0) <= 0
