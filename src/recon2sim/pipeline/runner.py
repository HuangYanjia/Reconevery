from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recon2sim.adapters import REGISTRY
from recon2sim.adapters.base import StageContext
from recon2sim.config import PipelineConfig
from recon2sim.reporting.logging import configure_run_logger
from recon2sim.storage import atomic_write_json, atomic_write_yaml

STATUSES = {"pending", "running", "succeeded", "failed", "skipped"}


def now() -> str:
    return datetime.now(UTC).isoformat()


class PipelineRunner:
    def __init__(self, config: PipelineConfig, input_dir: Path, run_dir: Path) -> None:
        self.config = config
        self.input_dir = input_dir
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"
        self.logger = configure_run_logger(run_dir)

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            import json

            return json.loads(self.manifest_path.read_text())
        return {"created_at": now(), "stages": {}}

    def save_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def order(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []

        def visit(s: str) -> None:
            if s in seen:
                return
            for dep in self.config.stages[s].depends_on:
                visit(dep)
            seen.add(s)
            out.append(s)

        for stage in self.config.stages:
            visit(stage)
        return out

    def selected(self, from_stage: str | None, until_stage: str | None) -> list[str]:
        ordered = self.order()
        start = ordered.index(from_stage) if from_stage else 0
        end = ordered.index(until_stage) + 1 if until_stage else len(ordered)
        return ordered[start:end]

    def run(
        self, *, resume: bool = False, from_stage: str | None = None, until_stage: str | None = None
    ) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(
            self.run_dir / "resolved_config.yaml", self.config.model_dump(mode="json")
        )
        manifest = self.load_manifest()
        chosen = set(self.selected(from_stage, until_stage))
        for stage in self.order():
            cfg = self.config.stages[stage]
            entry = manifest["stages"].setdefault(stage, {"status": "pending"})
            if stage not in chosen or not cfg.enabled:
                entry.update(
                    status="skipped", start_time=entry.get("start_time"), end_time=now(), error=None
                )
                self.save_manifest(manifest)
                continue
            adapter_cls = REGISTRY.get(cfg.adapter.name)
            if adapter_cls is None:
                raise ValueError(f"Unknown adapter {cfg.adapter.name!r} for stage {stage}")
            ctx = StageContext(stage, self.input_dir, self.run_dir, cfg, self.config.seed)
            sig = ctx.signature()
            if resume and entry.get("status") == "succeeded" and entry.get("signature") == sig:
                entry.update(
                    status="skipped",
                    skipped_reason="already succeeded with matching signature",
                    end_time=now(),
                )
                self.save_manifest(manifest)
                continue
            entry.update(
                status="running", start_time=now(), end_time=None, error=None, signature=sig
            )
            self.save_manifest(manifest)
            self.logger.info("stage started", extra={"stage": stage})
            try:
                adapter = adapter_cls()
                adapter.prepare(ctx)
                result = adapter.run(ctx)
                collected = adapter.collect(ctx)
                artifacts = [a.__dict__ for a in [*result.artifacts, *collected]]
                entry.update(
                    status="succeeded",
                    end_time=now(),
                    artifacts=artifacts,
                    metrics=result.metrics,
                    error=None,
                )
                self.logger.info("stage succeeded", extra={"stage": stage})
            except Exception as exc:
                entry.update(status="failed", end_time=now(), error=f"{type(exc).__name__}: {exc}")
                self.save_manifest(manifest)
                self.logger.error(f"stage failed: {exc}", extra={"stage": stage})
                raise
            self.save_manifest(manifest)
        return manifest
