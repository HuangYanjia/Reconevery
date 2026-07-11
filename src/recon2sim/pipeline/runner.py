from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import TypeAdapter
from recon2sim.adapters import REGISTRY, ArtifactRecord, OutputSpec
from recon2sim.adapters.base import Adapter, StageContext
from recon2sim.config import PipelineConfig
from recon2sim.images import validate_png
from recon2sim.reporting.logging import configure_run_logger
from recon2sim.storage import atomic_write_json, atomic_write_yaml

Manifest = dict[str, Any]
StageEntry = dict[str, Any]


class PipelineConfigurationError(ValueError):
    """Raised when the configured DAG or requested stage range is invalid."""


class OutputValidationError(RuntimeError):
    """Raised when an adapter does not satisfy its declared output contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _directory_snapshot(root: Path) -> list[dict[str, str | int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist or is not a directory: {root}")
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


class PipelineRunner:
    def __init__(self, config: PipelineConfig, input_dir: Path, run_dir: Path) -> None:
        self.config = config
        self.input_dir = input_dir
        self.run_dir = run_dir
        self.manifest_path = run_dir / "manifest.json"
        self.logger = configure_run_logger(run_dir)

    def load_manifest(self) -> Manifest:
        if not self.manifest_path.exists():
            return {
                "schema_version": "0.1.0",
                "created_at": _now(),
                "updated_at": _now(),
                "stages": {},
            }
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"run manifest is not valid JSON: {self.manifest_path}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("stages"), dict):
            raise ValueError(f"run manifest has an invalid structure: {self.manifest_path}")
        return cast(Manifest, raw)

    def save_manifest(self, manifest: Manifest) -> None:
        manifest["updated_at"] = _now()
        atomic_write_json(self.manifest_path, manifest)

    def order(self) -> list[str]:
        stages = self.config.stages
        for stage_name, stage in stages.items():
            unknown = [dependency for dependency in stage.depends_on if dependency not in stages]
            if unknown:
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} has unknown dependencies: {unknown}; "
                    f"known stages are {list(stages)}"
                )

        state: dict[str, int] = {}
        stack: list[str] = []
        ordered: list[str] = []

        def visit(stage_name: str) -> None:
            status = state.get(stage_name, 0)
            if status == 2:
                return
            if status == 1:
                cycle_start = stack.index(stage_name)
                cycle = [*stack[cycle_start:], stage_name]
                raise PipelineConfigurationError(
                    f"pipeline dependency cycle detected: {' -> '.join(cycle)}"
                )
            state[stage_name] = 1
            stack.append(stage_name)
            for dependency in stages[stage_name].depends_on:
                visit(dependency)
            stack.pop()
            state[stage_name] = 2
            ordered.append(stage_name)

        for stage_name in stages:
            visit(stage_name)
        return ordered

    def selected(self, from_stage: str | None, until_stage: str | None) -> list[str]:
        ordered = self.order()
        if from_stage is not None and from_stage not in self.config.stages:
            raise PipelineConfigurationError(
                f"from-stage {from_stage!r} does not exist; choose one of {ordered}"
            )
        if until_stage is not None and until_stage not in self.config.stages:
            raise PipelineConfigurationError(
                f"until-stage {until_stage!r} does not exist; choose one of {ordered}"
            )
        start = ordered.index(from_stage) if from_stage is not None else 0
        end = ordered.index(until_stage) if until_stage is not None else len(ordered) - 1
        if start > end:
            raise PipelineConfigurationError(
                f"from-stage {from_stage!r} occurs after until-stage {until_stage!r}"
            )
        return ordered[start : end + 1]

    def run(
        self,
        *,
        resume: bool = False,
        from_stage: str | None = None,
        until_stage: str | None = None,
    ) -> Manifest:
        if not self.input_dir.is_dir():
            raise FileNotFoundError(
                f"input directory does not exist or is not a directory: {self.input_dir}"
            )
        ordered = self.order()
        selected = self.selected(from_stage, until_stage)
        chosen = set(selected)
        manifest = self.load_manifest()
        self._validate_execution_dependencies(manifest, chosen)
        self._validate_adapters(chosen)

        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_yaml(
            self.run_dir / "resolved_config.yaml", self.config.model_dump(mode="json")
        )
        stages = self._stage_entries(manifest)
        for stage_name in ordered:
            stage_config = self.config.stages[stage_name]
            entry = self._stage_entry(stages, stage_name)
            if stage_name not in chosen:
                entry["last_execution"] = "not_selected"
                self.save_manifest(manifest)
                continue
            if not stage_config.enabled:
                if entry.get("status") != "succeeded":
                    entry["status"] = "skipped"
                entry["last_execution"] = "disabled"
                entry["end_time"] = _now()
                self.save_manifest(manifest)
                continue

            adapter_class = REGISTRY[stage_config.adapter.name]
            adapter = adapter_class()
            signature, signature_inputs = self._stage_signature(stage_name, adapter, manifest)
            if (
                resume
                and entry.get("status") == "succeeded"
                and entry.get("signature") == signature
                and self._artifacts_current(entry)
            ):
                entry["last_execution"] = "cache_hit"
                entry["cache_checked_at"] = _now()
                entry["error"] = None
                self.save_manifest(manifest)
                self.logger.info("stage cache hit", extra={"stage": stage_name})
                continue

            health = adapter.healthcheck()
            if not health.ok:
                raise RuntimeError(
                    f"adapter {adapter.name!r} healthcheck failed for stage "
                    f"{stage_name!r}: {health.message}"
                )
            entry.update(
                status="running",
                last_execution="executing",
                start_time=_now(),
                end_time=None,
                error=None,
                signature=signature,
                signature_inputs=signature_inputs,
                adapter_name=adapter.name,
                adapter_version=adapter.version,
                attempts=[],
            )
            self.save_manifest(manifest)
            self.logger.info("stage started", extra={"stage": stage_name})
            try:
                records, metrics = self._execute_with_retries(adapter, stage_name, entry, manifest)
            except Exception as exc:
                entry.update(
                    status="failed",
                    last_execution="failed",
                    end_time=_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.save_manifest(manifest)
                self.logger.error(f"stage failed: {exc}", extra={"stage": stage_name})
                raise

            execution_count = int(entry.get("execution_count", 0)) + 1
            output_signature = _stable_digest(
                [
                    {"relative_path": record.relative_path, "sha256": record.sha256}
                    for record in records
                ]
            )
            execution_signature = _stable_digest(
                {
                    "signature": signature,
                    "output_signature": output_signature,
                    "execution_count": execution_count,
                }
            )
            entry.update(
                status="succeeded",
                last_execution="executed",
                end_time=_now(),
                artifacts=[record.model_dump(mode="json") for record in records],
                metrics=metrics,
                error=None,
                execution_count=execution_count,
                output_signature=output_signature,
                execution_signature=execution_signature,
            )
            self.save_manifest(manifest)
            self.logger.info("stage succeeded", extra={"stage": stage_name})
        return manifest

    def _execute_with_retries(
        self,
        adapter: Adapter,
        stage_name: str,
        entry: StageEntry,
        manifest: Manifest,
    ) -> tuple[list[ArtifactRecord], dict[str, float | int | str | bool]]:
        stage_config = self.config.stages[stage_name]
        total_attempts = stage_config.adapter.retries + 1
        last_error: Exception | None = None
        for attempt in range(1, total_attempts + 1):
            attempt_entry: dict[str, Any] = {
                "attempt": attempt,
                "start_time": _now(),
                "status": "running",
            }
            attempts = cast(list[dict[str, Any]], entry["attempts"])
            attempts.append(attempt_entry)
            self.save_manifest(manifest)
            context = StageContext(
                stage_name=stage_name,
                input_dir=self.input_dir,
                run_dir=self.run_dir,
                config=stage_config,
                seed=self.config.seed,
                attempt=attempt,
            )
            try:
                adapter.prepare(context)
                declared = adapter.expected_outputs(context)
                result = adapter.run(context)
                records = self._validate_outputs(
                    stage_name,
                    adapter.name,
                    [*declared, *result.outputs],
                )
                attempt_entry.update(status="succeeded", end_time=_now(), error=None)
                self.save_manifest(manifest)
                return records, result.metrics
            except Exception as exc:
                last_error = exc
                details = getattr(exc, "details", None)
                attempt_entry.update(
                    status="failed",
                    end_time=_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(details, dict):
                    attempt_entry["details"] = details
                self.save_manifest(manifest)
                if attempt < total_attempts:
                    self.logger.warning(
                        f"stage attempt {attempt} failed; retrying: {exc}",
                        extra={"stage": stage_name},
                    )
        if last_error is None:
            raise RuntimeError(f"stage {stage_name!r} did not execute")
        raise last_error

    def _validate_outputs(
        self,
        stage_name: str,
        adapter_name: str,
        specs: list[OutputSpec],
    ) -> list[ArtifactRecord]:
        unique_specs: dict[str, OutputSpec] = {}
        for spec in specs:
            existing = unique_specs.get(spec.relative_path)
            if existing is not None and existing != spec:
                raise OutputValidationError(
                    f"stage {stage_name!r} declared conflicting metadata for output "
                    f"{spec.relative_path!r}"
                )
            unique_specs[spec.relative_path] = spec
        if not unique_specs:
            raise OutputValidationError(
                f"stage {stage_name!r} declared no outputs; every stage must declare outputs"
            )

        records: list[ArtifactRecord] = []
        for relative_path, spec in sorted(unique_specs.items()):
            pure_path = PurePosixPath(relative_path)
            if pure_path.is_absolute() or ".." in pure_path.parts or relative_path in {"", "."}:
                raise OutputValidationError(
                    f"stage {stage_name!r} declared unsafe output path {relative_path!r}"
                )
            path = self.run_dir / relative_path
            if not path.is_file():
                raise OutputValidationError(
                    f"stage {stage_name!r} did not produce required output {relative_path!r}"
                )
            try:
                self._validate_output_content(path, spec)
            except Exception as exc:
                raise OutputValidationError(
                    f"stage {stage_name!r} produced invalid output {relative_path!r}: {exc}"
                ) from exc
            records.append(
                ArtifactRecord(
                    relative_path=relative_path,
                    artifact_type=spec.artifact_type,
                    media_type=spec.media_type,
                    sha256=_sha256(path),
                    size_bytes=path.stat().st_size,
                    producer_stage=stage_name,
                    producer_adapter=adapter_name,
                    source_type=spec.source_type,
                    schema_identifier=spec.schema_identifier,
                )
            )
        return records

    @staticmethod
    def _validate_output_content(path: Path, spec: OutputSpec) -> None:
        if spec.validation in {"json", "scene_ir"}:
            text = path.read_text(encoding="utf-8")
            if spec.model is not None:
                spec.model.model_validate_json(text)
            else:
                TypeAdapter(dict[str, Any]).validate_json(text)
        elif spec.validation == "png":
            validate_png(path)
        elif spec.validation == "obj":
            text = path.read_text(encoding="utf-8")
            if not any(line.startswith("v ") for line in text.splitlines()):
                raise ValueError("OBJ contains no vertices")
            if not any(line.startswith("f ") for line in text.splitlines()):
                raise ValueError("OBJ contains no faces")

    def _stage_signature(
        self, stage_name: str, adapter: Adapter, manifest: Manifest
    ) -> tuple[str, dict[str, Any]]:
        stage_config = self.config.stages[stage_name]
        upstream_artifacts: dict[str, list[dict[str, str | int]]] = {}
        upstream_signatures: dict[str, str] = {}
        stages = self._stage_entries(manifest)
        for dependency in stage_config.depends_on:
            entry = self._stage_entry(stages, dependency)
            artifacts: list[dict[str, str | int]] = []
            for raw_record in cast(list[dict[str, Any]], entry.get("artifacts", [])):
                record = ArtifactRecord.model_validate(raw_record)
                path = self.run_dir / record.relative_path
                if not path.is_file():
                    raise FileNotFoundError(
                        f"stage {stage_name!r} requires missing artifact "
                        f"{record.relative_path!r} from stage {dependency!r}"
                    )
                artifacts.append(
                    {
                        "relative_path": record.relative_path,
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            upstream_artifacts[dependency] = artifacts
            execution_signature = entry.get("execution_signature")
            if not isinstance(execution_signature, str):
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} requires stage {dependency!r}, but that stage has "
                    "no successful execution signature"
                )
            upstream_signatures[dependency] = execution_signature

        signature_inputs: dict[str, Any] = {
            "stage": stage_name,
            "stage_configuration": stage_config.model_dump(mode="json"),
            "adapter_name": adapter.name,
            "adapter_version": adapter.version,
            "seed": self.config.seed,
            "input_artifacts": upstream_artifacts,
            "upstream_execution_signatures": upstream_signatures,
            "source_input_files": _directory_snapshot(self.input_dir)
            if not stage_config.depends_on
            else [],
        }
        return _stable_digest(signature_inputs), signature_inputs

    def _validate_execution_dependencies(self, manifest: Manifest, chosen: set[str]) -> None:
        stages = self._stage_entries(manifest)
        for stage_name in chosen:
            stage = self.config.stages[stage_name]
            if not stage.enabled:
                continue
            for dependency in stage.depends_on:
                dependency_config = self.config.stages[dependency]
                if not dependency_config.enabled:
                    allowed = self.config.allow_existing_artifacts_for_disabled_dependencies
                    available = self._stage_available(stages, dependency)
                    if not allowed or not available:
                        raise PipelineConfigurationError(
                            f"enabled stage {stage_name!r} depends on disabled stage "
                            f"{dependency!r}; enable it, or set "
                            "allow_existing_artifacts_for_disabled_dependencies=true after a "
                            "successful run with intact artifacts"
                        )
                elif dependency not in chosen and not self._stage_available(stages, dependency):
                    raise PipelineConfigurationError(
                        f"stage {stage_name!r} depends on stage {dependency!r}, which is outside "
                        "the selected range and has no intact successful artifacts; start from "
                        f"{dependency!r} or run the complete pipeline first"
                    )

    def _validate_adapters(self, chosen: set[str]) -> None:
        for stage_name in chosen:
            stage = self.config.stages[stage_name]
            if stage.enabled and stage.adapter.name not in REGISTRY:
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} uses unknown adapter {stage.adapter.name!r}; "
                    f"available adapters are {sorted(REGISTRY)}"
                )

    def _stage_available(self, stages: dict[str, StageEntry], stage_name: str) -> bool:
        entry = self._stage_entry(stages, stage_name)
        return entry.get("status") == "succeeded" and self._artifacts_current(entry)

    def _artifacts_current(self, entry: StageEntry) -> bool:
        raw_artifacts = entry.get("artifacts")
        if not isinstance(raw_artifacts, list) or not raw_artifacts:
            return False
        try:
            records = [ArtifactRecord.model_validate(item) for item in raw_artifacts]
        except Exception:
            return False
        return all(
            (self.run_dir / record.relative_path).is_file()
            and _sha256(self.run_dir / record.relative_path) == record.sha256
            for record in records
        )

    @staticmethod
    def _stage_entries(manifest: Manifest) -> dict[str, StageEntry]:
        return cast(dict[str, StageEntry], manifest["stages"])

    @staticmethod
    def _stage_entry(stages: dict[str, StageEntry], stage_name: str) -> StageEntry:
        return stages.setdefault(stage_name, {"status": "pending", "last_execution": "never"})
