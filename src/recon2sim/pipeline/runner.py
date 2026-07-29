from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import TypeAdapter

from recon2sim.adapters import REGISTRY, ArtifactRecord, InputSpec, OutputSpec
from recon2sim.adapters.base import Adapter, StageContext
from recon2sim.config import PipelineConfig
from recon2sim.images import validate_binary_mask_png, validate_png
from recon2sim.reporting.logging import configure_run_logger
from recon2sim.storage import atomic_write_json, atomic_write_yaml

Manifest = dict[str, Any]
StageEntry = dict[str, Any]
_FICLONE = 0x40049409


@dataclass(frozen=True)
class _InputSnapshot:
    source: Path
    sha256: str
    relative_path: str
    verify_after_execution: bool


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


def _source_snapshot(root: Path) -> list[dict[str, str | int]]:
    if root.is_file():
        return [
            {
                "relative_path": root.name,
                "sha256": _sha256(root),
                "size_bytes": root.stat().st_size,
            }
        ]
    if not root.is_dir():
        raise FileNotFoundError(f"input path does not exist: {root}")
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
        if not self.input_dir.exists():
            raise FileNotFoundError(f"input path does not exist: {self.input_dir}")
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
            artifacts_current = self._artifacts_current(entry)
            exact_cache_hit = entry.get("signature") == signature
            migrated_cache_hit = self._can_migrate_selective_signature(
                entry,
                signature_inputs,
            )
            if (
                resume
                and entry.get("status") == "succeeded"
                and artifacts_current
                and (exact_cache_hit or migrated_cache_hit)
            ):
                if migrated_cache_hit:
                    output_signature = entry.get("output_signature")
                    if not isinstance(output_signature, str):
                        raise PipelineConfigurationError(
                            f"stage {stage_name!r} cannot migrate a cache entry without an "
                            "output signature"
                        )
                    entry.update(
                        signature=signature,
                        signature_inputs=signature_inputs,
                        execution_signature=_stable_digest(
                            {
                                "signature": signature,
                                "output_signature": output_signature,
                            }
                        ),
                    )
                entry["last_execution"] = "cache_hit"
                entry["cache_checked_at"] = _now()
                entry["error"] = None
                self.save_manifest(manifest)
                self.logger.info("stage cache hit", extra={"stage": stage_name})
                continue

            health_context = StageContext(
                stage_name=stage_name,
                input_dir=self.input_dir,
                run_dir=self.run_dir,
                canonical_run_dir=self.run_dir,
                config=stage_config,
                seed=self.config.seed,
                attempt=0,
            )
            health = adapter.healthcheck(health_context)
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
            except (KeyboardInterrupt, SystemExit) as exc:
                entry.update(
                    status="interrupted",
                    last_execution="interrupted",
                    end_time=_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                self.save_manifest(manifest)
                self.logger.warning("stage interrupted", extra={"stage": stage_name})
                raise
            except BaseException as exc:
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
        last_error: BaseException | None = None
        for _ in range(total_attempts):
            attempt = self._next_attempt_number(stage_name)
            workspace = self.run_dir / "work" / stage_name / f"attempt_{attempt}"
            workspace.mkdir(parents=True, exist_ok=False)
            attempt_entry: dict[str, Any] = {
                "attempt": attempt,
                "workspace": workspace.relative_to(self.run_dir).as_posix(),
                "start_time": _now(),
                "status": "running",
            }
            attempts = cast(list[dict[str, Any]], entry["attempts"])
            attempts.append(attempt_entry)
            self.save_manifest(manifest)
            context = StageContext(
                stage_name=stage_name,
                input_dir=self.input_dir,
                run_dir=workspace,
                canonical_run_dir=self.run_dir,
                config=stage_config,
                seed=self.config.seed,
                attempt=attempt,
            )
            try:
                materialized, snapshots = self._populate_attempt_workspace(
                    workspace,
                    stage_name,
                    manifest,
                    adapter,
                    context,
                )
                attempt_entry["materialized_inputs"] = materialized
                self.save_manifest(manifest)
                adapter.prepare(context)
                declared = adapter.expected_outputs(context)
                try:
                    result = adapter.run(context)
                except BaseException:
                    self._verify_input_snapshots(stage_name, snapshots)
                    raise
                self._verify_input_snapshots(stage_name, snapshots)
                records = self._validate_outputs(
                    stage_name,
                    adapter.name,
                    [*declared, *result.outputs],
                    root=workspace,
                )
                previous_records = [
                    ArtifactRecord.model_validate(item)
                    for item in cast(list[dict[str, Any]], entry.get("artifacts", []))
                ]
                self._promote_outputs(workspace, records, previous_records)
                attempt_entry.update(status="succeeded", end_time=_now(), error=None)
                self.save_manifest(manifest)
                return records, result.metrics
            except (KeyboardInterrupt, SystemExit) as exc:
                details = getattr(exc, "details", None)
                attempt_entry.update(
                    status="interrupted",
                    end_time=_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(details, dict):
                    attempt_entry["details"] = details
                self._preserve_attempt_logs(workspace)
                self.save_manifest(manifest)
                raise
            except BaseException as exc:
                last_error = exc
                details = getattr(exc, "details", None)
                attempt_entry.update(
                    status="failed",
                    end_time=_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                if isinstance(details, dict):
                    attempt_entry["details"] = details
                self._preserve_attempt_logs(workspace)
                self.save_manifest(manifest)
                if len(attempts) < total_attempts:
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
        *,
        root: Path | None = None,
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
            path = (root or self.run_dir) / relative_path
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

    def _next_attempt_number(self, stage_name: str) -> int:
        stage_work = self.run_dir / "work" / stage_name
        existing = []
        if stage_work.is_dir():
            for path in stage_work.iterdir():
                if path.is_dir() and path.name.startswith("attempt_"):
                    try:
                        existing.append(int(path.name.removeprefix("attempt_")))
                    except ValueError:
                        continue
        return max(existing, default=0) + 1

    def _populate_attempt_workspace(
        self,
        workspace: Path,
        stage_name: str,
        manifest: Manifest,
        adapter: Adapter,
        context: StageContext,
    ) -> tuple[list[dict[str, Any]], list[_InputSnapshot]]:
        stages = self._stage_entries(manifest)
        ancestors = self._ancestor_stages(stage_name)
        records: dict[str, ArtifactRecord] = {}
        for producer_stage, entry in stages.items():
            if producer_stage not in ancestors or entry.get("status") != "succeeded":
                continue
            for raw_record in cast(list[dict[str, Any]], entry.get("artifacts", [])):
                record = ArtifactRecord.model_validate(raw_record)
                records[record.relative_path] = record

        specs = self._adapter_input_specs(adapter, context)
        if specs is None:
            specs = [
                InputSpec(
                    relative_path=record.relative_path,
                    artifact_type=record.artifact_type,
                    expected_sha256=record.sha256,
                )
                for record in sorted(records.values(), key=lambda item: item.relative_path)
            ]

        materialized: list[dict[str, Any]] = []
        snapshots: list[_InputSnapshot] = []
        destinations: set[str] = set()
        for spec in specs:
            self._validate_input_spec(stage_name, spec)
            if spec.relative_path in destinations:
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} declared duplicate input destination "
                    f"{spec.relative_path!r}"
                )
            destinations.add(spec.relative_path)

            source_record: ArtifactRecord | None = None
            if spec.source_path is not None:
                source = spec.source_path.expanduser()
                if not source.is_absolute():
                    source = Path.cwd() / source
                source = source.resolve()
                source_kind = "external_configuration"
            else:
                source_artifact_path = spec.source_artifact_path or spec.relative_path
                source_record = records.get(source_artifact_path)
                if source_record is None:
                    if not spec.required:
                        continue
                    raise FileNotFoundError(
                        f"stage {stage_name!r} requires undeclared or missing upstream artifact "
                        f"{source_artifact_path!r}"
                    )
                if (
                    spec.artifact_type != source_record.artifact_type
                    and spec.artifact_type != "any"
                ):
                    raise PipelineConfigurationError(
                        f"stage {stage_name!r} requested artifact "
                        f"{source_artifact_path!r} as {spec.artifact_type!r}, but its type is "
                        f"{source_record.artifact_type!r}"
                    )
                source = self.run_dir / source_record.relative_path
                source_kind = "ancestor_artifact"

            if not source.is_file():
                if not spec.required:
                    continue
                raise FileNotFoundError(f"stage {stage_name!r} requires missing input {source}")
            if source.is_symlink():
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} input source must not be a symlink: {source}"
                )
            source_hash = _sha256(source)
            expected_hash = spec.expected_sha256 or (
                source_record.sha256 if source_record is not None else None
            )
            if expected_hash is not None and source_hash != expected_hash:
                raise FileNotFoundError(
                    f"stage {stage_name!r} requires stale input {source}; "
                    f"expected sha256 {expected_hash}, found {source_hash}"
                )

            if spec.materialization_mode != "reference_only":
                destination = workspace / spec.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._materialize_input(source, destination, spec.materialization_mode)
                if _sha256(destination) != source_hash:
                    raise RuntimeError(
                        f"stage {stage_name!r} input copy verification failed for "
                        f"{spec.relative_path!r}"
                    )
            materialized.append(
                {
                    "relative_path": spec.relative_path,
                    "source": (
                        source_record.relative_path if source_record is not None else str(source)
                    ),
                    "source_kind": source_kind,
                    "artifact_type": spec.artifact_type,
                    "sha256": source_hash,
                    "size_bytes": source.stat().st_size,
                    "materialization_mode": spec.materialization_mode,
                }
            )
            snapshots.append(
                _InputSnapshot(
                    source=source,
                    sha256=source_hash,
                    relative_path=spec.relative_path,
                    verify_after_execution=source_record is not None,
                )
            )
        return materialized, snapshots

    @staticmethod
    def _adapter_input_specs(
        adapter: Adapter,
        context: StageContext,
    ) -> list[InputSpec] | None:
        method = getattr(adapter, "required_inputs", None)
        if method is None:
            return None
        provider = cast(Callable[[StageContext], list[InputSpec]], method)
        return provider(context)

    @staticmethod
    def _validate_input_spec(stage_name: str, spec: InputSpec) -> None:
        path = PurePosixPath(spec.relative_path)
        if path.is_absolute() or ".." in path.parts or spec.relative_path in {"", "."}:
            raise PipelineConfigurationError(
                f"stage {stage_name!r} declared unsafe input path {spec.relative_path!r}"
            )
        if spec.source_artifact_path is not None:
            source_path = PurePosixPath(spec.source_artifact_path)
            if (
                source_path.is_absolute()
                or ".." in source_path.parts
                or spec.source_artifact_path in {"", "."}
            ):
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} declared unsafe source artifact path "
                    f"{spec.source_artifact_path!r}"
                )
        if spec.source_path is not None and spec.source_artifact_path is not None:
            raise PipelineConfigurationError(
                f"stage {stage_name!r} input {spec.relative_path!r} cannot declare both "
                "source_path and source_artifact_path"
            )
        if spec.expected_sha256 is not None and (
            len(spec.expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in spec.expected_sha256)
        ):
            raise PipelineConfigurationError(
                f"stage {stage_name!r} input {spec.relative_path!r} has an invalid sha256"
            )

    @staticmethod
    def _materialize_input(source: Path, destination: Path, mode: str) -> None:
        if mode == "reference_only":
            return
        if mode == "copy":
            shutil.copy2(source, destination)
            return
        try:
            with source.open("rb") as input_file, destination.open("wb") as output_file:
                fcntl.ioctl(output_file.fileno(), _FICLONE, input_file.fileno())
            shutil.copystat(source, destination)
        except OSError:
            destination.unlink(missing_ok=True)
            shutil.copy2(source, destination)

    @staticmethod
    def _verify_input_snapshots(
        stage_name: str,
        snapshots: list[_InputSnapshot],
    ) -> None:
        for snapshot in snapshots:
            if not snapshot.verify_after_execution:
                continue
            if not snapshot.source.is_file() or _sha256(snapshot.source) != snapshot.sha256:
                raise RuntimeError(
                    f"stage {stage_name!r} modified canonical upstream input "
                    f"{snapshot.relative_path!r}; the attempt was rejected"
                )

    def _promote_outputs(
        self,
        workspace: Path,
        records: list[ArtifactRecord],
        previous_records: list[ArtifactRecord],
    ) -> None:
        promotion_root = self.run_dir / ".promotion" / workspace.parent.name / workspace.name
        if promotion_root.exists():
            shutil.rmtree(promotion_root)
        staged_root = promotion_root / "staged"
        backup_root = promotion_root / "backup"
        for record in records:
            source = workspace / record.relative_path
            staged = staged_root / record.relative_path
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)

        current_paths = {record.relative_path for record in records}
        previous_paths = {record.relative_path for record in previous_records}
        affected_paths = sorted(current_paths | previous_paths)
        previously_present: set[str] = set()
        for relative_path in affected_paths:
            canonical = self.run_dir / relative_path
            if not canonical.is_file():
                continue
            previously_present.add(relative_path)
            backup = backup_root / relative_path
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical, backup)

        try:
            for record in records:
                staged = staged_root / record.relative_path
                destination = self.run_dir / record.relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                self._replace_promoted_output(staged, destination)
            for relative_path in previous_paths - current_paths:
                obsolete = self.run_dir / relative_path
                if obsolete.is_file():
                    obsolete.unlink()
        except BaseException as promotion_error:
            try:
                for relative_path in affected_paths:
                    destination = self.run_dir / relative_path
                    if relative_path in previously_present:
                        backup = backup_root / relative_path
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        rollback_copy = destination.with_name(f".{destination.name}.rollback")
                        shutil.copy2(backup, rollback_copy)
                        os.replace(rollback_copy, destination)
                    elif destination.is_file():
                        destination.unlink()
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"output promotion failed and rollback was incomplete: {rollback_error}"
                ) from promotion_error
            raise
        finally:
            shutil.rmtree(promotion_root, ignore_errors=True)

    @staticmethod
    def _replace_promoted_output(source: Path, destination: Path) -> None:
        os.replace(source, destination)

    def _ancestor_stages(self, stage_name: str) -> set[str]:
        ancestors: set[str] = set()

        def collect(current: str) -> None:
            for dependency in self.config.stages[current].depends_on:
                if dependency not in ancestors:
                    ancestors.add(dependency)
                    collect(dependency)

        collect(stage_name)
        return ancestors

    def _preserve_attempt_logs(self, workspace: Path) -> None:
        logs_root = workspace / "logs"
        if not logs_root.is_dir():
            return
        for source in sorted(logs_root.rglob("*")):
            if not source.is_file():
                continue
            relative_path = source.relative_to(workspace)
            destination = self.run_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.attempt-copy")
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)

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
        elif spec.validation == "binary_png":
            validate_binary_mask_png(path)
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
        signature_context = StageContext(
            stage_name=stage_name,
            input_dir=self.input_dir,
            run_dir=self.run_dir,
            canonical_run_dir=self.run_dir,
            config=stage_config,
            seed=self.config.seed,
            attempt=0,
        )
        declared_specs = self._adapter_input_specs(adapter, signature_context)
        if declared_specs is not None:
            return self._selective_stage_signature(
                stage_name,
                adapter,
                manifest,
                declared_specs,
            )

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
            "source_input_files": _source_snapshot(self.input_dir)
            if not stage_config.depends_on
            else [],
        }
        signature_inputs["external_input_files"] = []
        return _stable_digest(signature_inputs), signature_inputs

    def _selective_stage_signature(
        self,
        stage_name: str,
        adapter: Adapter,
        manifest: Manifest,
        declared_specs: list[InputSpec],
    ) -> tuple[str, dict[str, Any]]:
        stage_config = self.config.stages[stage_name]
        ancestors = self._ancestor_stages(stage_name)
        stages = self._stage_entries(manifest)
        records: dict[str, tuple[str, ArtifactRecord]] = {}
        for producer_stage, entry in stages.items():
            if producer_stage not in ancestors or entry.get("status") != "succeeded":
                continue
            for raw_record in cast(list[dict[str, Any]], entry.get("artifacts", [])):
                record = ArtifactRecord.model_validate(raw_record)
                records[record.relative_path] = (producer_stage, record)

        declared_inputs: list[dict[str, str | int | bool]] = []
        destinations: set[str] = set()
        for spec in declared_specs:
            self._validate_input_spec(stage_name, spec)
            if spec.relative_path in destinations:
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} declared duplicate input destination "
                    f"{spec.relative_path!r}"
                )
            destinations.add(spec.relative_path)

            if spec.source_path is not None:
                source = spec.source_path.expanduser()
                if not source.is_absolute():
                    source = Path.cwd() / source
                source = source.resolve()
                source_artifact_path = str(source)
                source_kind = "external_configuration"
                producer_stage = "external"
                source_record: ArtifactRecord | None = None
            else:
                source_artifact_path = spec.source_artifact_path or spec.relative_path
                resolved = records.get(source_artifact_path)
                if resolved is None:
                    if not spec.required:
                        continue
                    raise FileNotFoundError(
                        f"stage {stage_name!r} requires undeclared or missing upstream artifact "
                        f"{source_artifact_path!r}"
                    )
                producer_stage, source_record = resolved
                if (
                    spec.artifact_type != source_record.artifact_type
                    and spec.artifact_type != "any"
                ):
                    raise PipelineConfigurationError(
                        f"stage {stage_name!r} requested artifact "
                        f"{source_artifact_path!r} as {spec.artifact_type!r}, but its type is "
                        f"{source_record.artifact_type!r}"
                    )
                source = self.run_dir / source_record.relative_path
                source_kind = "ancestor_artifact"

            if not source.is_file():
                if not spec.required:
                    continue
                raise FileNotFoundError(f"stage {stage_name!r} requires missing input {source}")
            if source.is_symlink():
                raise PipelineConfigurationError(
                    f"stage {stage_name!r} input source must not be a symlink: {source}"
                )
            source_hash = _sha256(source)
            expected_hash = spec.expected_sha256 or (
                source_record.sha256 if source_record is not None else None
            )
            if expected_hash is not None and source_hash != expected_hash:
                raise FileNotFoundError(
                    f"stage {stage_name!r} requires stale input {source}; "
                    f"expected sha256 {expected_hash}, found {source_hash}"
                )
            signature_record: dict[str, str | int | bool] = {
                "relative_path": spec.relative_path,
                "source": source_artifact_path,
                "source_kind": source_kind,
                "producer_stage": producer_stage,
                "artifact_type": spec.artifact_type,
                "sha256": source_hash,
                "size_bytes": source.stat().st_size,
                "materialization_mode": spec.materialization_mode,
                "include_producer_signature": spec.include_producer_signature,
            }
            if source_record is not None and spec.include_producer_signature:
                producer_entry = self._stage_entry(stages, producer_stage)
                producer_signature = producer_entry.get("execution_signature")
                if not isinstance(producer_signature, str):
                    raise PipelineConfigurationError(
                        f"stage {stage_name!r} requires stage {producer_stage!r}, but that "
                        "producer has no successful execution signature"
                    )
                signature_record["producer_execution_signature"] = producer_signature
            declared_inputs.append(signature_record)

        signature_inputs: dict[str, Any] = {
            "stage": stage_name,
            "stage_configuration": stage_config.model_dump(mode="json"),
            "adapter_name": adapter.name,
            "adapter_version": adapter.version,
            "seed": self.config.seed,
            "declared_inputs": sorted(
                declared_inputs,
                key=lambda item: (
                    str(item["relative_path"]),
                    str(item["source"]),
                ),
            ),
            "source_input_files": _source_snapshot(self.input_dir)
            if not stage_config.depends_on
            else [],
        }
        return _stable_digest(signature_inputs), signature_inputs

    @staticmethod
    def _can_migrate_selective_signature(
        entry: StageEntry,
        new_inputs: dict[str, Any],
    ) -> bool:
        declared = new_inputs.get("declared_inputs")
        legacy = entry.get("signature_inputs")
        if (
            not isinstance(declared, list)
            or not isinstance(legacy, dict)
            or legacy.get("stage") != new_inputs.get("stage")
            or legacy.get("stage_configuration") != new_inputs.get("stage_configuration")
            or legacy.get("adapter_name") != new_inputs.get("adapter_name")
            or legacy.get("adapter_version") != new_inputs.get("adapter_version")
            or legacy.get("seed") != new_inputs.get("seed")
        ):
            return False

        legacy_declared = legacy.get("declared_inputs")
        if isinstance(legacy_declared, list):
            comparable_fields = (
                "relative_path",
                "source",
                "source_kind",
                "producer_stage",
                "artifact_type",
                "sha256",
                "size_bytes",
                "materialization_mode",
            )
            old_by_identity = {
                tuple(raw.get(field) for field in comparable_fields): raw
                for raw in legacy_declared
                if isinstance(raw, dict)
            }
            for raw in declared:
                if not isinstance(raw, dict):
                    return False
                identity = tuple(raw.get(field) for field in comparable_fields)
                prior = old_by_identity.get(identity)
                if prior is None:
                    return False
                producer_signature = raw.get("producer_execution_signature")
                if producer_signature is not None and (
                    not isinstance(producer_signature, str)
                    or prior.get("producer_execution_signature") != producer_signature
                ):
                    return False
            return True

        legacy_artifacts: dict[str, tuple[str, int]] = {}
        raw_artifact_groups = legacy.get("input_artifacts")
        if isinstance(raw_artifact_groups, dict):
            for raw_group in raw_artifact_groups.values():
                if not isinstance(raw_group, list):
                    continue
                for raw in raw_group:
                    if not isinstance(raw, dict):
                        continue
                    path = raw.get("relative_path")
                    sha256 = raw.get("sha256")
                    size = raw.get("size_bytes")
                    if isinstance(path, str) and isinstance(sha256, str) and isinstance(size, int):
                        legacy_artifacts[path] = (sha256, size)
        legacy_external: dict[str, tuple[str, int]] = {}
        raw_external = legacy.get("external_input_files")
        if isinstance(raw_external, list):
            for raw in raw_external:
                if not isinstance(raw, dict):
                    continue
                path = raw.get("relative_path")
                sha256 = raw.get("sha256")
                size = raw.get("size_bytes")
                if isinstance(path, str) and isinstance(sha256, str) and isinstance(size, int):
                    legacy_external[path] = (sha256, size)
        legacy_producer_signatures = legacy.get("upstream_execution_signatures")
        if not isinstance(legacy_producer_signatures, dict):
            legacy_producer_signatures = {}

        for raw in declared:
            if not isinstance(raw, dict):
                return False
            source_kind = raw.get("source_kind")
            lookup = (
                legacy_external if source_kind == "external_configuration" else legacy_artifacts
            )
            lookup_path = (
                raw.get("relative_path")
                if source_kind == "external_configuration"
                else raw.get("source")
            )
            sha256 = raw.get("sha256")
            size = raw.get("size_bytes")
            if (
                not isinstance(lookup_path, str)
                or not isinstance(sha256, str)
                or not isinstance(size, int)
                or lookup.get(lookup_path) != (sha256, size)
            ):
                return False
            producer_signature = raw.get("producer_execution_signature")
            producer_stage = raw.get("producer_stage")
            if producer_signature is not None and (
                not isinstance(producer_signature, str)
                or not isinstance(producer_stage, str)
                or legacy_producer_signatures.get(producer_stage) != producer_signature
            ):
                return False
        return True

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
