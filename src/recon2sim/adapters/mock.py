from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recon2sim.adapters.base import ArtifactRecord, HealthcheckResult, StageContext, StageResult
from recon2sim.ir import *
from recon2sim.storage import atomic_write_json, atomic_write_text


def prov(adapter: str, cfg: dict[str, Any], ins: list[str], outs: list[str]) -> ProvenanceRecord:
    return ProvenanceRecord(
        adapter_name=adapter,
        adapter_version="0.1.0",
        configuration=cfg,
        input_artifact_paths=ins,
        output_artifact_paths=outs,
        timestamp=datetime.now(UTC),
        confidence=ConfidenceRecord(score=0.9, method="deterministic_mock"),
        source=GeometrySourceType.MOCK,
    )


def obj(path: Path, name: str) -> None:
    atomic_write_text(path, f"# mock {name}\no cube\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")


class MockAdapter:
    name = "mock"

    def healthcheck(self) -> HealthcheckResult:
        return HealthcheckResult(True, "mock ready")

    def prepare(self, context: StageContext) -> None:
        context.run_dir.mkdir(parents=True, exist_ok=True)

    def collect(self, context: StageContext) -> list[ArtifactRecord]:
        return []


class MockIngestAdapter(MockAdapter):
    name = "mock_ingest"

    def run(self, context: StageContext) -> StageResult:
        (context.path("inputs")).mkdir(exist_ok=True)
        frames = context.path("frames")
        frames.mkdir(exist_ok=True)
        srcs = sorted(context.input_dir.glob("frames/*")) or sorted(context.input_dir.glob("*"))
        arts = []
        for i, src in enumerate(srcs[:3] or []):
            if src.is_file():
                dst = frames / f"frame_{i:03d}{src.suffix or '.txt'}"
                shutil.copyfile(src, dst)
                arts.append(ArtifactRecord(str(dst.relative_to(context.run_dir)), "frame", "mock"))
        if not arts:
            for i in range(3):
                dst = frames / f"frame_{i:03d}.txt"
                atomic_write_text(dst, f"mock frame {i}\n")
                arts.append(ArtifactRecord(str(dst.relative_to(context.run_dir)), "frame", "mock"))
        atomic_write_json(
            context.path("inputs", "manifest.json"),
            {"source": "mock", "frames": [a.path for a in arts]},
        )
        return StageResult(arts, {"frame_count": len(arts)})


class MockCameraRecoveryAdapter(MockAdapter):
    name = "mock_camera_recovery"

    def run(self, context: StageContext) -> StageResult:
        frames = sorted(context.path("frames").glob("frame_*"))
        out = context.path("camera", "reconstruction.json")
        data = {
            "source": "mock",
            "camera_id": "cam0",
            "intrinsics": {"width": 640, "height": 480, "fx": 500, "fy": 500, "cx": 320, "cy": 240},
            "poses": [
                {"frame_id": f.stem, "translation_m": [0, i * 0.05, 1]}
                for i, f in enumerate(frames)
            ],
        }
        atomic_write_json(out, data)
        return StageResult([ArtifactRecord("camera/reconstruction.json", "camera", "mock")])


class MockSegmentationTrackingAdapter(MockAdapter):
    name = "mock_segmentation_tracking"

    def run(self, context: StageContext) -> StageResult:
        masks = context.path("observations", "masks")
        masks.mkdir(parents=True, exist_ok=True)
        obs = []
        for f in sorted(context.path("frames").glob("frame_*")):
            for oid in ["floor", "table", "cup", "cabinet", "drawer"]:
                m = masks / f"{f.stem}_{oid}.txt"
                atomic_write_text(m, "mock mask\n")
                obs.append(
                    {
                        "object_id": oid,
                        "frame_id": f.stem,
                        "bbox_xywh": [10, 10, 100, 100],
                        "mask_path": str(m.relative_to(context.run_dir)),
                        "confidence": {"score": 0.8, "method": "mock"},
                    }
                )
        out = context.path("observations", "object_tracks.json")
        atomic_write_json(out, {"source": "mock", "observations": obs})
        return StageResult([ArtifactRecord("observations/object_tracks.json", "tracks", "mock")])


class MockGlobalReconstructionAdapter(MockAdapter):
    name = "mock_global_reconstruction"

    def run(self, context: StageContext) -> StageResult:
        out = context.path("reconstruction", "global", "room.obj")
        obj(out, "room")
        return StageResult([ArtifactRecord("reconstruction/global/room.obj", "mesh", "mock")])


class MockObjectReconstructionAdapter(MockAdapter):
    name = "mock_object_reconstruction"

    def run(self, context: StageContext) -> StageResult:
        arts = []
        for name in ["floor", "table", "cup", "cabinet", "drawer"]:
            p = context.path("reconstruction", "objects", f"{name}.obj")
            obj(p, name)
            arts.append(ArtifactRecord(str(p.relative_to(context.run_dir)), "mesh", "mock"))
        return StageResult(arts)


class MockSceneIRAssemblyAdapter(MockAdapter):
    name = "mock_scene_ir_assembly"

    def run(self, context: StageContext) -> StageResult:
        cfg = context.config.adapter.config
        p = prov(
            self.name,
            cfg,
            ["camera/reconstruction.json", "observations/object_tracks.json"],
            ["scene_ir/scene.json"],
        )
        geom = []
        coll = []
        mats = []
        objects = []
        specs = [
            ("floor", AssetType.STATIC_STRUCTURE, True),
            ("table", AssetType.STATIC_STRUCTURE, True),
            ("cup", AssetType.RIGID, False),
            ("cabinet", AssetType.ARTICULATED, True),
            ("drawer", AssetType.ARTICULATED, False),
        ]
        for oid, typ, is_static in specs:
            g = f"geom_{oid}"
            c = f"coll_{oid}"
            geom.append(
                GeometryAsset(
                    asset_id=g,
                    asset_type=typ,
                    uri=f"reconstruction/objects/{oid}.obj",
                    format="obj",
                    source=GeometrySourceType.MOCK,
                    provenance=p,
                )
            )
            coll.append(
                CollisionAsset(
                    collision_id=c,
                    uri=f"reconstruction/objects/{oid}.obj",
                    format="obj",
                    source=GeometrySourceType.MOCK,
                    provenance=p,
                )
            )
            mats.append(
                MaterialAsset(
                    material_id=f"mat_{oid}",
                    name=f"mock {oid}",
                    base_color_rgba=(0.5, 0.5, 0.5, 1),
                    provenance=p,
                )
            )
            art = None
            if oid == "cabinet":
                art = Articulation(
                    articulation_id="cabinet_articulation",
                    links=[
                        Link(link_id="cabinet_body", name="body"),
                        Link(link_id="drawer_link", name="drawer"),
                    ],
                    joints=[
                        Joint(
                            joint_id="drawer_slide",
                            parent_link_id="cabinet_body",
                            child_link_id="drawer_link",
                            joint_type="prismatic",
                            axis_xyz=(1, 0, 0),
                            limits=(0, 0.4),
                        )
                    ],
                )
            objects.append(
                ObjectInstance(
                    object_id=oid,
                    name=oid,
                    asset_type=typ,
                    geometry_asset_ids=[g],
                    material_asset_ids=[f"mat_{oid}"],
                    collision_asset_ids=[c],
                    physics=PhysicsProperties(
                        mass_kg=None if is_static else 0.2, friction=0.5, is_static=is_static
                    ),
                    articulation=art,
                    provenance=[p],
                    confidence=ConfidenceRecord(score=0.85, method="mock"),
                )
            )
        cam = Camera(
            camera_id="cam0",
            model="pinhole",
            intrinsics=CameraIntrinsics(width=640, height=480, fx=500, fy=500, cx=320, cy=240),
            provenance=p,
        )
        relations = [
            SceneRelation(
                relation_type=RelationType.SUPPORTED_BY,
                subject_id="table",
                object_id="floor",
                confidence=ConfidenceRecord(score=0.9, method="mock"),
                provenance=p,
            ),
            SceneRelation(
                relation_type=RelationType.SUPPORTED_BY,
                subject_id="cup",
                object_id="table",
                confidence=ConfidenceRecord(score=0.9, method="mock"),
                provenance=p,
            ),
            SceneRelation(
                relation_type=RelationType.PART_OF,
                subject_id="drawer",
                object_id="cabinet",
                confidence=ConfidenceRecord(score=0.8, method="mock"),
                provenance=p,
            ),
        ]
        scene = SceneIR(
            metadata=SceneMetadata(
                scene_id="tabletop_demo",
                name="Mock tabletop",
                source=GeometrySourceType.MOCK,
                provenance=[p],
            ),
            cameras=[cam],
            objects=objects,
            geometry_assets=geom,
            material_assets=mats,
            collision_assets=coll,
            relations=relations,
        )
        atomic_write_json(context.path("scene_ir", "scene.json"), scene.model_dump(mode="json"))
        return StageResult([ArtifactRecord("scene_ir/scene.json", "scene_ir", "mock")])


class MockSceneCompilerAdapter(MockAdapter):
    name = "mock_scenesmith_compiler"

    def run(self, context: StageContext) -> StageResult:
        pkg = context.path("compiled", "scene_package")
        pkg.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pkg / "package.json",
            {"source": "mock", "scene_ir": "scene_ir/scene.json", "exports": ["mock_scene.obj"]},
        )
        obj(pkg / "mock_scene.obj", "compiled_scene")
        return StageResult(
            [ArtifactRecord("compiled/scene_package/package.json", "compiled_scene", "mock")]
        )


class MockPhysicsValidatorAdapter(MockAdapter):
    name = "mock_physics_validator"

    def run(self, context: StageContext) -> StageResult:
        report = ValidationReport(
            scene_id="tabletop_demo",
            passed=True,
            issues=[
                ValidationIssue(
                    severity="warning",
                    code="MOCK_LOW_FIDELITY_COLLISION",
                    message="Mock collision meshes are illustrative only.",
                    object_id="cup",
                )
            ],
        )
        atomic_write_json(context.path("validation", "report.json"), report.model_dump(mode="json"))
        return StageResult([ArtifactRecord("validation/report.json", "validation", "mock")])


class MockExportAdapter(MockAdapter):
    name = "mock_export"

    def run(self, context: StageContext) -> StageResult:
        atomic_write_json(
            context.path("export_manifest.json"),
            {"source": "mock", "compiled": "compiled/scene_package"},
        )
        return StageResult([ArtifactRecord("export_manifest.json", "export", "mock")])
