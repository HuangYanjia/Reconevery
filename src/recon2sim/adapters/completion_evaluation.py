from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from PIL import Image
from pydantic import Field

from recon2sim.adapters.base import (
    HealthcheckResult,
    InputSpec,
    OutputSpec,
    StageContext,
    StageResult,
)
from recon2sim.adapters.completion_candidates import SAM3D_COMMIT
from recon2sim.adapters.completion_common import (
    CompletionWorkerConfig,
    completion_healthcheck,
    resolve_worker_python,
    worker_command,
)
from recon2sim.adapters.completion_registration import GENERATION_MANIFESTS
from recon2sim.adapters.ingest import ProcessExecutionError, run_process
from recon2sim.artifacts import (
    CameraReconstruction,
    CandidateEvaluationManifest,
    CandidateEvaluationRequest,
    CandidateFailureClassification,
    CandidateGenerationManifest,
    CandidateRegistrationManifest,
    CandidateRepresentationParityArtifact,
    CandidateRepresentationParityView,
    CompletionCropManifest,
    CompletionEvidencePackage,
    CompletionEvidenceSplit,
    CompletionWorkerManifest,
    DenseDepthManifest,
    DenseUndistortionManifest,
    DenseWorkspaceManifest,
    ObjectCompletionCandidate,
    SegmentationTrackingArtifact,
)
from recon2sim.completion import sha256_file
from recon2sim.completion_parity import (
    backend_layout_world_matrix,
    binary_mask_metrics,
    flatten_matrix,
    invert_rigid_matrix,
    target_mask_metrics,
    world_from_camera_matrix,
)
from recon2sim.storage import atomic_write_json


def _render_parity_representation(
    *,
    context: StageContext,
    config: CompletionEvaluationAdapterConfig,
    candidate_id: str,
    view_id: str,
    representation: str,
    asset_path: str,
    asset_format: str,
    transform: list[float],
    camera: dict[str, object],
) -> tuple[dict[str, object], list[OutputSpec]]:
    python = resolve_worker_python(config.sam3d_renderer_python or "")
    if python is None:
        raise RuntimeError("configured SAM 3D renderer Python was not found")
    relative_root = (
        Path("reconstruction")
        / "completion"
        / "candidates"
        / candidate_id
        / "representation_parity"
        / view_id
        / representation
    )
    output_dir = context.path(*relative_root.parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "render_request.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": "0.1.0",
            "asset_path": asset_path,
            "asset_format": asset_format,
            "matrix_world_from_candidate": [
                transform[index : index + 4] for index in range(0, 16, 4)
            ],
            "camera": camera,
            "official_checkout_path": config.sam3d_official_checkout_path,
            "official_code_commit": SAM3D_COMMIT,
            "alpha_threshold": 0.001,
        },
    )
    run_process(
        [
            python,
            "-m",
            config.sam3d_renderer_module,
            "render",
            "--request",
            str(request_path),
            "--input-root",
            str(context.run_dir.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
        ],
        context=context,
        name=f"sam3d_parity_{candidate_id}_{view_id}_{representation}",
        log_directory="reconstruction/completion/raw/parity_logs",
    )
    manifest_path = output_dir / "render_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = (request_path, manifest_path, output_dir / "rgba.png", output_dir / "valid.png")
    if any(not path.is_file() for path in required):
        raise RuntimeError("SAM 3D representation renderer omitted a required artifact")
    width = camera.get("width")
    height = camera.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        raise RuntimeError("representation parity camera dimensions must be integers")
    expected_size = (width, height)
    with (
        Image.open(output_dir / "rgba.png") as rgba,
        Image.open(output_dir / "valid.png") as valid,
    ):
        if rgba.mode != "RGBA" or rgba.size != expected_size:
            raise RuntimeError("representation renderer emitted an invalid RGBA image")
        if valid.mode != "L" or valid.size != expected_size:
            raise RuntimeError("representation renderer emitted an invalid visibility mask")
    output_specs = [
        OutputSpec(
            path.relative_to(context.run_dir).as_posix(),
            "candidate_representation_parity_file",
            "image/png" if path.suffix == ".png" else "application/json",
            "completion_evaluation",
            validation="exists" if path.suffix == ".png" else "json",
        )
        for path in required
    ]
    depth_name = manifest.get("depth_path")
    if depth_name:
        depth_path = output_dir / str(depth_name)
        if not depth_path.is_file():
            raise RuntimeError("representation renderer declared a missing depth output")
        output_specs.append(
            OutputSpec(
                depth_path.relative_to(context.run_dir).as_posix(),
                "candidate_representation_depth",
                "application/octet-stream",
                "completion_evaluation",
                validation="exists",
            )
        )
    return manifest, output_specs


class CompletionEvaluationAdapterConfig(CompletionWorkerConfig):
    worker_module: str = "completion_evaluation_worker"
    docker_image: str = "reconevery/completion-evaluation:phase5b"
    minimum_validation_views: int = Field(default=2, ge=1)
    minimum_mask_iou: float = Field(default=0.25, ge=0, le=1)
    minimum_mask_precision: float = Field(default=0.60, ge=0, le=1)
    maximum_median_relative_depth_residual: float = Field(default=0.08, ge=0)
    minimum_depth_inlier_fraction: float = Field(default=0.50, ge=0, le=1)
    maximum_negative_space_violation_ratio: float = Field(default=0.10, ge=0, le=1)
    maximum_front_of_scene_violation_ratio: float = Field(default=0.05, ge=0, le=1)
    minimum_recall_gain_over_measured_baseline: float = Field(default=0.05, ge=-1, le=1)
    maximum_precision_drop_from_measured_baseline: float = Field(default=0.15, ge=0, le=1)
    sam3d_renderer_python: str | None = None
    sam3d_renderer_module: str = "sam3d_objects_worker"
    sam3d_official_checkout_path: str | None = None
    parity_minimum_silhouette_iou: float = Field(default=0.80, ge=0, le=1)
    parity_minimum_bbox_iou: float = Field(default=0.80, ge=0, le=1)
    parity_maximum_normalized_centroid_distance: float = Field(default=0.05, ge=0)


class CompletionCandidateEvaluationAdapter:
    name = "completion_candidate_evaluation"
    version = "0.1.3"

    def required_inputs(self, context: StageContext) -> list[InputSpec]:
        split = CompletionEvidenceSplit.model_validate_json(
            context.canonical_path("reconstruction", "completion", "evidence_split.json").read_text(
                encoding="utf-8"
            )
        )
        tracks = SegmentationTrackingArtifact.model_validate_json(
            context.canonical_path("observations", "object_tracks.json").read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(
            context.canonical_path("reconstruction", "dense", "depth_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        workspace = DenseWorkspaceManifest.model_validate_json(
            context.canonical_path("reconstruction", "dense", "workspace_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        crops = CompletionCropManifest.model_validate_json(
            context.canonical_path("reconstruction", "completion", "crop_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        diagnostic_frames = {
            frame_id
            for item in split.objects
            for frame_id in (
                *item.generation_anchor_frames,
                *item.registration_fitting_frames,
                *item.heldout_validation_frames,
            )
        }
        specs = [
            InputSpec("camera/reconstruction.json", "camera_reconstruction"),
            InputSpec("observations/object_tracks.json", "segmentation_tracking"),
            InputSpec("reconstruction/dense/depth_manifest.json", "dense_depth_manifest"),
            InputSpec(
                "reconstruction/dense/undistortion_manifest.json",
                "dense_undistortion_manifest",
            ),
            InputSpec(
                "reconstruction/dense/workspace_manifest.json",
                "dense_workspace_manifest",
            ),
            InputSpec(
                "reconstruction/completion/evidence/evidence_package.json",
                "completion_evidence_package",
            ),
            InputSpec(
                "reconstruction/completion/evidence_split.json",
                "completion_evidence_split",
            ),
            InputSpec(
                "reconstruction/completion/crop_manifest.json",
                "completion_crop_manifest",
            ),
            InputSpec(
                "reconstruction/completion/registration_manifest.json",
                "candidate_registration_manifest",
            ),
        ]
        specs.extend(
            InputSpec(anchor.crop_path, "completion_evidence_file")
            for anchor in crops.anchors
            if anchor.frame_id in diagnostic_frames
        )
        for path in GENERATION_MANIFESTS.values():
            specs.append(InputSpec(path, "candidate_generation_manifest"))
            generation = CandidateGenerationManifest.model_validate_json(
                context.canonical_path(*Path(path).parts).read_text(encoding="utf-8")
            )
            for candidate in generation.candidates:
                specs.extend(
                    InputSpec(
                        asset.relative_path,
                        (
                            "completion_evidence_file"
                            if asset.relative_path.startswith("reconstruction/completion/evidence/")
                            else (
                                "measured_object_geometry_file"
                                if candidate.backend.value == "measured_partial_baseline"
                                else "completion_candidate_file"
                            )
                        ),
                        materialization_mode="reflink_or_copy",
                    )
                    for asset in candidate.native_assets
                )
        specs.extend(
            InputSpec(observation.mask_path, "canonical_object_mask")
            for track in tracks.tracks
            for observation in track.observations
            if observation.frame_id in diagnostic_frames
        )
        for record in depth.records:
            if record.frame_id in diagnostic_frames:
                specs.extend(
                    [
                        InputSpec(record.depth_path, "dense_mvs_workspace_file"),
                        InputSpec(record.normal_path, "dense_mvs_workspace_file"),
                        InputSpec(
                            record.consistency_graph_path,
                            "dense_mvs_workspace_file",
                        ),
                    ]
                )
        specs.extend(
            InputSpec(
                frame.workspace_filename,
                "dense_mvs_workspace_file",
            )
            for frame in workspace.frames
            if frame.frame_id in diagnostic_frames
        )
        return [replace(spec, include_producer_signature=False) for spec in specs]

    def healthcheck(self, context: StageContext | None = None) -> HealthcheckResult:
        return completion_healthcheck(
            context,
            CompletionEvaluationAdapterConfig,
            worker_name="completion held-out evaluation worker",
        )

    def prepare(self, context: StageContext) -> None:
        context.path("reconstruction", "completion", "raw", "evaluation_logs").mkdir(
            parents=True, exist_ok=True
        )

    def expected_outputs(self, context: StageContext) -> list[OutputSpec]:
        root = "reconstruction/completion"
        outputs = [
            OutputSpec(
                f"{root}/evaluation_request.json",
                "candidate_evaluation_request",
                "application/json",
                "completion_evaluation",
                validation="json",
                model=CandidateEvaluationRequest,
            ),
            OutputSpec(
                f"{root}/evaluation_worker_manifest.json",
                "completion_worker_manifest",
                "application/json",
                "completion_evaluation",
                validation="json",
                model=CompletionWorkerManifest,
            ),
            OutputSpec(
                f"{root}/evaluation_manifest.json",
                "candidate_evaluation_manifest",
                "application/json",
                "completion_evaluation",
                validation="json",
                model=CandidateEvaluationManifest,
            ),
        ]
        registration_path = context.canonical_path(
            "reconstruction", "completion", "registration_manifest.json"
        )
        if registration_path.is_file():
            registration = CandidateRegistrationManifest.model_validate_json(
                registration_path.read_text(encoding="utf-8")
            )
            candidate_anchors = {
                candidate.candidate_id: candidate.anchor_frame_id
                for path in GENERATION_MANIFESTS.values()
                for candidate in CandidateGenerationManifest.model_validate_json(
                    context.canonical_path(*Path(path).parts).read_text(encoding="utf-8")
                ).candidates
            }
            for item in registration.registrations:
                if item.frozen_transform is None:
                    continue
                groups = {
                    "anchor": [candidate_anchors[item.candidate_id]],
                    "fitting": item.fitting_frame_ids,
                    "heldout": item.heldout_frame_ids,
                }
                for group, frame_ids in groups.items():
                    for frame_id in frame_ids:
                        outputs.append(
                            OutputSpec(
                                f"{root}/renders/{item.candidate_id}/{group}/{frame_id}.png",
                                f"candidate_{group}_render",
                                "image/png",
                                "completion_evaluation",
                                validation="png",
                            )
                        )
        return outputs

    def _run_representation_parity(
        self,
        *,
        context: StageContext,
        config: CompletionEvaluationAdapterConfig,
        evaluation: CandidateEvaluationManifest,
        registration: CandidateRegistrationManifest,
        candidates: dict[str, ObjectCompletionCandidate],
    ) -> tuple[CandidateEvaluationManifest, list[OutputSpec]]:
        if config.sam3d_renderer_python is None:
            return evaluation, []
        if not config.sam3d_official_checkout_path:
            raise RuntimeError(
                "sam3d_official_checkout_path is required when representation parity is enabled"
            )
        camera = CameraReconstruction.model_validate_json(
            context.path("camera", "reconstruction.json").read_text(encoding="utf-8")
        )
        undistortion = DenseUndistortionManifest.model_validate_json(
            context.path(
                "reconstruction",
                "dense",
                "undistortion_manifest.json",
            ).read_text(encoding="utf-8")
        )
        crops = CompletionCropManifest.model_validate_json(
            context.path(
                "reconstruction",
                "completion",
                "crop_manifest.json",
            ).read_text(encoding="utf-8")
        )
        crop_by_object_frame = {
            (anchor.object_id, anchor.frame_id): anchor for anchor in crops.anchors
        }
        pose_by_frame = {pose.frame_id: pose for pose in camera.poses}
        dense_by_frame = {record.frame_id: record for record in undistortion.records}
        registration_by_id = {
            item.candidate_id: item
            for item in registration.registrations
            if item.frozen_transform is not None
        }
        parity_by_candidate: dict[
            str,
            tuple[str, bool, CandidateFailureClassification | None],
        ] = {}
        outputs: list[OutputSpec] = []
        for candidate_id, candidate_value in sorted(candidates.items()):
            candidate = candidate_value
            if candidate.backend.value != "sam3d_objects":
                continue
            assets = {asset.format.value: asset for asset in candidate.native_assets}
            gaussian = assets.get("gaussian_splat_ply")
            glb = assets.get("pbr_glb") or assets.get("mesh_glb")
            registration_item = registration_by_id.get(candidate_id)
            if gaussian is None or glb is None or registration_item is None:
                continue
            assert registration_item.frozen_transform is not None
            frozen = list(registration_item.frozen_transform.matrix_world_from_candidate)
            views: list[
                tuple[
                    str,
                    str,
                    list[float],
                    str,
                    dict[str, object] | None,
                    str | None,
                ]
            ] = []
            try:
                backend_camera = candidate.backend_anchor_camera
                crop = crop_by_object_frame[(candidate.object_id, candidate.anchor_frame_id)]
                if backend_camera is not None:
                    layout_transform = flatten_matrix(
                        backend_layout_world_matrix(
                            candidate.backend_predicted_layout,
                            [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        )
                    )
                    fx, fy, cx, cy = backend_camera.pixel_intrinsics
                    views.append(
                        (
                            "backend_canonical_anchor",
                            candidate.anchor_frame_id,
                            layout_transform,
                            "backend_predicted_layout_in_official_crop_camera",
                            {
                                "camera_from_world": [
                                    [1.0, 0.0, 0.0, 0.0],
                                    [0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],
                                ],
                                "width": backend_camera.width,
                                "height": backend_camera.height,
                                "fx": fx,
                                "fy": fy,
                                "cx": cx,
                                "cy": cy,
                                "near": 1e-6,
                                "far": 1e6,
                            },
                            crop.crop_path,
                        )
                    )
            except (KeyError, TypeError, ValueError):
                pass
            views.append(
                (
                    "registered_anchor",
                    candidate.anchor_frame_id,
                    frozen,
                    "frozen_registration",
                    None,
                    None,
                )
            )
            views.extend(
                (
                    f"fitting_{index:02d}",
                    frame_id,
                    frozen,
                    "frozen_registration",
                    None,
                    None,
                )
                for index, frame_id in enumerate(registration_item.fitting_frame_ids[:2])
            )
            parity_views: list[CandidateRepresentationParityView] = []
            for (
                view_id,
                frame_id,
                transform,
                transform_source,
                camera_override,
                target_crop_path,
            ) in views:
                if camera_override is None:
                    pose = pose_by_frame[frame_id]
                    camera_from_world = invert_rigid_matrix(
                        world_from_camera_matrix(
                            pose.transform_world_from_camera.rotation_xyzw,
                            pose.transform_world_from_camera.translation,
                        )
                    )
                    dense = dense_by_frame[frame_id]
                    width, height = dense.dense_dimensions
                    fx, fy, cx, cy = dense.dense_intrinsics
                    target_camera: dict[str, object] = {
                        "camera_from_world": camera_from_world,
                        "width": width,
                        "height": height,
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                        "near": 1e-6,
                        "far": 1e6,
                    }
                else:
                    target_camera = camera_override
                    width_value = target_camera["width"]
                    height_value = target_camera["height"]
                    if not isinstance(width_value, int) or not isinstance(height_value, int):
                        raise RuntimeError("backend anchor camera dimensions must be integers")
                    width, height = width_value, height_value
                manifests = {}
                valid_paths = {}
                for representation, asset in (("gaussian", gaussian), ("glb", glb)):
                    manifest, render_outputs = _render_parity_representation(
                        context=context,
                        config=config,
                        candidate_id=candidate_id,
                        view_id=view_id,
                        representation=representation,
                        asset_path=asset.relative_path,
                        asset_format=asset.format.value,
                        transform=transform,
                        camera=target_camera,
                    )
                    outputs.extend(render_outputs)
                    manifests[representation] = manifest
                    valid_paths[representation] = context.path(
                        "reconstruction",
                        "completion",
                        "candidates",
                        candidate_id,
                        "representation_parity",
                        view_id,
                        representation,
                        "valid.png",
                    )
                with (
                    Image.open(valid_paths["gaussian"]).convert("L") as gaussian_mask,
                    Image.open(valid_paths["glb"]).convert("L") as glb_mask,
                ):
                    silhouette, bbox_iou, centroid, gaussian_count, glb_count = binary_mask_metrics(
                        gaussian_mask.getdata(),
                        glb_mask.getdata(),
                        width,
                        height,
                    )
                target_metrics: dict[str, tuple[float, float, float] | None] = {
                    "gaussian": None,
                    "glb": None,
                }
                if target_crop_path is not None:
                    with Image.open(context.path(*Path(target_crop_path).parts)).convert(
                        "RGBA"
                    ) as target_rgba:
                        target = target_rgba.getchannel("A")
                        if target.size != (width, height):
                            raise RuntimeError(
                                "official backend anchor camera does not match the anchor crop"
                            )
                        for representation in ("gaussian", "glb"):
                            with Image.open(valid_paths[representation]).convert("L") as rendered:
                                target_metrics[representation] = target_mask_metrics(
                                    rendered.getdata(),
                                    target.getdata(),
                                )
                gaussian_target = target_metrics["gaussian"]
                glb_target = target_metrics["glb"]
                parity_views.append(
                    CandidateRepresentationParityView(
                        view_id=view_id,
                        frame_id=frame_id,
                        transform_source=transform_source,
                        gaussian_valid_pixel_count=gaussian_count,
                        glb_valid_pixel_count=glb_count,
                        silhouette_iou=silhouette,
                        projected_bbox_iou=bbox_iou,
                        normalized_centroid_distance=centroid,
                        gaussian_depth_available=bool(manifests["gaussian"].get("depth_path")),
                        glb_depth_available=bool(manifests["glb"].get("depth_path")),
                        gaussian_target_mask_precision=(
                            gaussian_target[0] if gaussian_target is not None else None
                        ),
                        gaussian_target_mask_recall=(
                            gaussian_target[1] if gaussian_target is not None else None
                        ),
                        gaussian_target_mask_iou=(
                            gaussian_target[2] if gaussian_target is not None else None
                        ),
                        glb_target_mask_precision=(
                            glb_target[0] if glb_target is not None else None
                        ),
                        glb_target_mask_recall=(glb_target[1] if glb_target is not None else None),
                        glb_target_mask_iou=(glb_target[2] if glb_target is not None else None),
                    )
                )
            failure_reasons = []
            if any(
                view.silhouette_iou < config.parity_minimum_silhouette_iou for view in parity_views
            ):
                failure_reasons.append("minimum_silhouette_iou")
            if any(
                view.projected_bbox_iou < config.parity_minimum_bbox_iou for view in parity_views
            ):
                failure_reasons.append("minimum_bbox_iou")
            if any(
                view.normalized_centroid_distance is None
                or view.normalized_centroid_distance
                > config.parity_maximum_normalized_centroid_distance
                for view in parity_views
            ):
                failure_reasons.append("maximum_normalized_centroid_distance")
            parity = CandidateRepresentationParityArtifact(
                candidate_id=candidate_id,
                gaussian_asset_id=gaussian.asset_id,
                gaussian_asset_path=gaussian.relative_path,
                glb_asset_id=glb.asset_id,
                glb_asset_path=glb.relative_path,
                official_code_commit=SAM3D_COMMIT,
                renderer="official_sam3d_gaussian_gsplat_and_nvdiffrast_glb",
                views=parity_views,
                minimum_silhouette_iou=config.parity_minimum_silhouette_iou,
                minimum_bbox_iou=config.parity_minimum_bbox_iou,
                maximum_normalized_centroid_distance=(
                    config.parity_maximum_normalized_centroid_distance
                ),
                accepted=not failure_reasons,
                failure_reasons=failure_reasons,
                transform_transfer_permitted=not failure_reasons,
                warnings=[
                    "native Gaussian depth is unavailable in the pinned official gsplat renderer"
                ],
            )
            parity_path = context.path(
                "reconstruction",
                "completion",
                "candidates",
                candidate_id,
                "representation_parity.json",
            )
            atomic_write_json(parity_path, parity)
            parity_relative = parity_path.relative_to(context.run_dir).as_posix()
            outputs.append(
                OutputSpec(
                    parity_relative,
                    "candidate_representation_parity",
                    "application/json",
                    "completion_evaluation",
                    validation="json",
                    model=CandidateRepresentationParityArtifact,
                )
            )
            canonical = next(
                (view for view in parity.views if view.view_id == "backend_canonical_anchor"),
                None,
            )
            canonical_failure = None
            if canonical is not None:
                if canonical.glb_valid_pixel_count == 0:
                    canonical_failure = CandidateFailureClassification.EMPTY_CANDIDATE_RENDER
                elif (canonical.glb_target_mask_iou or 0.0) <= 1e-8:
                    canonical_failure = CandidateFailureClassification.BACKEND_EXPORT_INVALID
            parity_by_candidate[candidate_id] = (
                parity_relative,
                parity.accepted,
                canonical_failure,
            )
        updated_evaluations = []
        for item in evaluation.evaluations:
            parity_record = parity_by_candidate.get(item.candidate_id)
            if parity_record is None:
                updated_evaluations.append(item)
                continue
            parity_relative_path, parity_accepted, canonical_failure = parity_record
            failed_gates = list(item.failed_gates)
            updates: dict[str, object] = {
                "representation_parity_path": parity_relative_path,
                "representation_parity_accepted": parity_accepted,
            }
            if canonical_failure is not None:
                failed_gates.append("backend_canonical_anchor_sanity")
                updates.update(
                    {
                        "failure_classification": canonical_failure,
                        "failed_gates": sorted(set(failed_gates)),
                        "passed_hard_gates": False,
                    }
                )
            updated_evaluations.append(item.model_copy(update=updates))
        return (
            evaluation.model_copy(update={"evaluations": updated_evaluations}),
            outputs,
        )

    def run(self, context: StageContext) -> StageResult:
        config = CompletionEvaluationAdapterConfig.model_validate(context.config.adapter.config)
        root = context.path("reconstruction", "completion")
        registration_path = root / "registration_manifest.json"
        package_path = root / "evidence" / "evidence_package.json"
        split_path = root / "evidence_split.json"
        tracks_path = context.path("observations", "object_tracks.json")
        camera_path = context.path("camera", "reconstruction.json")
        depth_path = context.path("reconstruction", "dense", "depth_manifest.json")
        undistortion_path = context.path("reconstruction", "dense", "undistortion_manifest.json")
        workspace_path = context.path("reconstruction", "dense", "workspace_manifest.json")
        registration = CandidateRegistrationManifest.model_validate_json(
            registration_path.read_text(encoding="utf-8")
        )
        CompletionEvidencePackage.model_validate_json(package_path.read_text(encoding="utf-8"))
        split = CompletionEvidenceSplit.model_validate_json(split_path.read_text(encoding="utf-8"))
        tracks = SegmentationTrackingArtifact.model_validate_json(
            tracks_path.read_text(encoding="utf-8")
        )
        depth = DenseDepthManifest.model_validate_json(depth_path.read_text(encoding="utf-8"))
        workspace = DenseWorkspaceManifest.model_validate_json(
            workspace_path.read_text(encoding="utf-8")
        )
        dense_frames = {item.frame_id: item for item in workspace.frames}
        observations = {
            track.object_id: {item.frame_id: item for item in track.observations}
            for track in tracks.tracks
        }
        depth_by_id = {item.frame_id: item for item in depth.records}

        def evaluation_inputs(frame_ids: list[str], object_id: str) -> dict[str, object]:
            return {
                "frame_ids": frame_ids,
                "mask_paths": {
                    frame_id: observations[object_id][frame_id].mask_path for frame_id in frame_ids
                },
                "depth_paths": {
                    frame_id: depth_by_id[frame_id].depth_path for frame_id in frame_ids
                },
                "normal_paths": {
                    frame_id: depth_by_id[frame_id].normal_path for frame_id in frame_ids
                },
                "dense_depth_hashes": {
                    frame_id: depth_by_id[frame_id].depth_sha256 for frame_id in frame_ids
                },
                "dense_image_paths": {
                    frame_id: dense_frames[frame_id].workspace_filename for frame_id in frame_ids
                },
            }

        anchor_inputs: dict[str, dict[str, object]] = {}
        fitting_inputs: dict[str, dict[str, object]] = {}
        heldout_inputs: dict[str, dict[str, object]] = {}
        for item in split.objects:
            anchor_inputs[item.object_id] = evaluation_inputs(
                item.generation_anchor_frames,
                item.object_id,
            )
            fitting_inputs[item.object_id] = evaluation_inputs(
                item.registration_fitting_frames,
                item.object_id,
            )
            heldout_inputs[item.object_id] = evaluation_inputs(
                item.heldout_validation_frames,
                item.object_id,
            )
        manifest_paths = dict(GENERATION_MANIFESTS)
        manifest_hashes = {
            name: sha256_file(context.path(*Path(path).parts))
            for name, path in manifest_paths.items()
        }
        request = CandidateEvaluationRequest(
            registration_manifest_path="reconstruction/completion/registration_manifest.json",
            registration_manifest_sha256=sha256_file(registration_path),
            evidence_package_path=("reconstruction/completion/evidence/evidence_package.json"),
            evidence_package_sha256=sha256_file(package_path),
            evidence_split_path="reconstruction/completion/evidence_split.json",
            evidence_split_sha256=sha256_file(split_path),
            generation_manifest_paths=manifest_paths,
            generation_manifest_hashes=manifest_hashes,
            segmentation_tracking_path="observations/object_tracks.json",
            segmentation_tracking_sha256=sha256_file(tracks_path),
            camera_reconstruction_path="camera/reconstruction.json",
            camera_reconstruction_sha256=sha256_file(camera_path),
            dense_depth_manifest_path="reconstruction/dense/depth_manifest.json",
            dense_depth_manifest_sha256=sha256_file(depth_path),
            dense_undistortion_manifest_path=("reconstruction/dense/undistortion_manifest.json"),
            dense_undistortion_manifest_sha256=sha256_file(undistortion_path),
            anchor_inputs=anchor_inputs,
            fitting_inputs=fitting_inputs,
            heldout_inputs=heldout_inputs,
            evaluation_configuration={
                "occlusion_policy": "dense_depth_visibility_v1",
                "minimum_validation_views": config.minimum_validation_views,
                "minimum_mask_iou": config.minimum_mask_iou,
                "minimum_mask_precision": config.minimum_mask_precision,
                "maximum_median_relative_depth_residual": (
                    config.maximum_median_relative_depth_residual
                ),
                "minimum_depth_inlier_fraction": config.minimum_depth_inlier_fraction,
                "maximum_negative_space_violation_ratio": (
                    config.maximum_negative_space_violation_ratio
                ),
                "maximum_front_of_scene_violation_ratio": (
                    config.maximum_front_of_scene_violation_ratio
                ),
                "minimum_recall_gain_over_measured_baseline": (
                    config.minimum_recall_gain_over_measured_baseline
                ),
                "maximum_precision_drop_from_measured_baseline": (
                    config.maximum_precision_drop_from_measured_baseline
                ),
                "heldout_only": True,
                "transforms_frozen": True,
                "fake_mode": config.fake_mode,
            },
            output_directory="reconstruction/completion",
            seed=context.seed,
        )
        request_path = root / "evaluation_request.json"
        atomic_write_json(request_path, request)
        try:
            run_process(
                worker_command(
                    context,
                    config,
                    "evaluate",
                    "reconstruction/completion/evaluation_request.json",
                    "reconstruction/completion",
                ),
                context=context,
                name="completion_evaluation_worker",
                log_directory="reconstruction/completion/raw/evaluation_logs",
            )
        except ProcessExecutionError as exc:
            if "out of memory" in exc.result.stderr.lower():
                raise RuntimeError("completion evaluation worker ran out of memory") from exc
            raise RuntimeError(str(exc)) from exc
        worker = CompletionWorkerManifest.model_validate_json(
            (root / "evaluation_worker_manifest.json").read_text(encoding="utf-8")
        )
        evaluation = CandidateEvaluationManifest.model_validate_json(
            (root / "evaluation_manifest.json").read_text(encoding="utf-8")
        )
        if worker.request_sha256 != sha256_file(request_path):
            raise RuntimeError("evaluation worker request hash mismatch")
        if evaluation.registration_manifest_sha256 != request.registration_manifest_sha256:
            raise RuntimeError("evaluation worker changed the registration lineage")
        registered = {
            item.candidate_id: item
            for item in registration.registrations
            if item.frozen_transform is not None
        }
        split_by_object = {item.object_id: item for item in split.objects}
        candidates = {
            candidate.candidate_id: candidate
            for path in GENERATION_MANIFESTS.values()
            for candidate in CandidateGenerationManifest.model_validate_json(
                context.path(*Path(path).parts).read_text(encoding="utf-8")
            ).candidates
        }
        evaluation, parity_outputs = self._run_representation_parity(
            context=context,
            config=config,
            evaluation=evaluation,
            registration=registration,
            candidates=candidates,
        )
        atomic_write_json(root / "evaluation_manifest.json", evaluation)
        for evaluation_item in evaluation.evaluations:
            if evaluation_item.candidate_id not in registered:
                raise RuntimeError("evaluation references an unregistered candidate")
            if (
                evaluation_item.heldout_frame_ids
                != split_by_object[evaluation_item.object_id].heldout_validation_frames
            ):
                raise RuntimeError("evaluation did not use the declared held-out frames")
            candidate = candidates[evaluation_item.candidate_id]
            registration_item = registered[evaluation_item.candidate_id]
            if (
                registration_item.registration_asset_id != evaluation_item.registration_asset_id
                or registration_item.registration_asset_path
                != evaluation_item.registration_asset_path
            ):
                raise RuntimeError("evaluation changed the registered representation")
            declared_assets = {
                asset.asset_id: asset.relative_path for asset in candidate.native_assets
            }
            for asset_id, asset_path in (
                (
                    evaluation_item.registration_asset_id,
                    evaluation_item.registration_asset_path,
                ),
                (
                    evaluation_item.evaluation_asset_id,
                    evaluation_item.evaluation_asset_path,
                ),
                (
                    evaluation_item.selection_asset_id,
                    evaluation_item.selection_asset_path,
                ),
            ):
                if declared_assets.get(asset_id) != asset_path:
                    raise RuntimeError("evaluation references an undeclared candidate asset")
            if evaluation_item.metrics.validation_view_count != len(
                evaluation_item.heldout_frame_ids
            ):
                raise RuntimeError("evaluation view count does not match held-out evidence")
            if set(evaluation_item.render_paths) != set(evaluation_item.heldout_frame_ids):
                raise RuntimeError("evaluation renders do not cover every held-out frame")
            expected_split = split_by_object[evaluation_item.object_id]
            if set(evaluation_item.anchor_render_paths) != {
                candidates[evaluation_item.candidate_id].anchor_frame_id
            }:
                raise RuntimeError("anchor diagnostics do not cover the candidate anchor")
            if set(evaluation_item.fitting_render_paths) != set(
                expected_split.registration_fitting_frames
            ):
                raise RuntimeError("fitting diagnostics do not cover fitting evidence")
            for render_path in (
                *evaluation_item.anchor_render_paths.values(),
                *evaluation_item.fitting_render_paths.values(),
                *evaluation_item.render_paths.values(),
            ):
                if not context.path(*Path(render_path).parts).is_file():
                    raise RuntimeError(f"evaluation render is missing: {render_path}")
        return StageResult(
            outputs=parity_outputs,
            metrics={
                "evaluated_candidates": len(evaluation.evaluations),
                "passing_candidates": sum(
                    item.passed_hard_gates for item in evaluation.evaluations
                ),
            },
        )
