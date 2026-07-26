from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from articulated_retrieval_worker import __version__
from articulated_retrieval_worker.ranking import rank_records


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrieve(request: dict[str, object], input_root: Path) -> None:
    started = time.monotonic()
    output_path = input_root / str(request["output_path"])
    index_path_value = request.get("asset_index_path")
    if index_path_value is None:
        candidates: list[dict[str, object]] = []
        index_hash = None
        warnings = ["no immutable local asset index configured"]
    else:
        index_path = input_root / str(index_path_value)
        if sha256(index_path) != request["asset_index_sha256"]:
            raise ValueError("local articulated asset index hash mismatch")
        index = read_json(index_path)
        measured = read_json(input_root / str(request["measured_motion_path"]))
        prompts = read_json(input_root / str(request["part_prompt_manifest_path"]))
        objects = prompts.get("objects")
        if not isinstance(objects, list) or not objects:
            raise ValueError("part prompt manifest contains no object")
        semantic_label = str(objects[0]["semantic_label"])
        hypotheses = measured.get("joint_hypotheses")
        if not isinstance(hypotheses, list):
            raise ValueError("measured motion has no joint hypotheses")
        joint_types = {str(item["joint_type"]) for item in hypotheses}
        records = index.get("records")
        if not isinstance(records, list):
            raise ValueError("asset index contains no records")
        ranked = rank_records(
            [item for item in records if isinstance(item, dict)],
            semantic_label=semantic_label,
            observed_joint_types=joint_types,
            observed_part_count=len(hypotheses),
        )
        family = str(request["source_family"])
        selected_assets_value = request.get("selected_assets", [])
        if not isinstance(selected_assets_value, list):
            raise ValueError("selected_assets must be a list")
        selected_assets = {
            str(item["asset_id"]): item for item in selected_assets_value if isinstance(item, dict)
        }
        candidates = []
        for score, terms, record in ranked:
            asset_id = str(record["asset_id"])
            selected = selected_assets.get(asset_id)
            if selected is None:
                continue
            source_candidate_path = input_root / str(selected["source_candidate_path"])
            source_candidate = read_json(source_candidate_path)
            mapping_value = selected.get("visual_path_mapping")
            if not isinstance(mapping_value, dict):
                raise ValueError("selected articulated asset is missing visual path mapping")
            mapping = {str(key): str(value) for key, value in mapping_value.items()}
            candidate_id = f"{request['articulated_object_id']}__{family}__{asset_id}"
            candidate_directory = output_path.parent / "candidates" / candidate_id / "retrieved"
            rewritten: dict[str, str] = {}
            for position, (source_relative, materialized_relative) in enumerate(
                sorted(mapping.items())
            ):
                source = input_root / materialized_relative
                if not source.is_file():
                    raise ValueError(f"selected articulated visual is missing: {source}")
                destination = candidate_directory / (
                    f"{position:03d}{source.suffix.lower() or '.bin'}"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                rewritten[source_relative] = destination.relative_to(input_root).as_posix()
            links = source_candidate.get("links")
            if not isinstance(links, list) or not links:
                raise ValueError("retrieved candidate bundle contains no links")
            for link_value in links:
                if not isinstance(link_value, dict):
                    raise ValueError("retrieved candidate link must be an object")
                paths = link_value.get("visual_asset_paths")
                if not isinstance(paths, list):
                    raise ValueError("retrieved candidate link has no visual paths")
                translated = []
                hashes = {}
                for path_value in paths:
                    original = str(path_value)
                    if original not in rewritten:
                        raise ValueError(
                            f"candidate visual {original!r} is absent from the local index"
                        )
                    translated.append(rewritten[original])
                    hashes[rewritten[original]] = sha256(input_root / rewritten[original])
                link_value["visual_asset_paths"] = translated
                link_value["visual_asset_hashes"] = hashes
            native_paths_value = source_candidate.get("native_output_paths", [])
            if not isinstance(native_paths_value, list):
                raise ValueError("candidate native_output_paths must be a list")
            native_paths = [rewritten[str(path)] for path in native_paths_value]
            source_candidate.update(
                {
                    "candidate_id": candidate_id,
                    "articulated_object_id": request["articulated_object_id"],
                    "source_family": family,
                    "source_asset_id": asset_id,
                    "native_output_paths": native_paths,
                    "native_output_hashes": {
                        path: sha256(input_root / path) for path in native_paths
                    },
                    "license_record": record["license_record"],
                    "production_selectable": bool(
                        record.get("license_record", {}).get("production_selectable", False)
                    ),
                    "provenance": {
                        "adapter_name": f"{family}_retrieval",
                        "adapter_version": __version__,
                        "configuration": {"retrieval_score": score},
                        "input_artifact_paths": [
                            str(selected["source_candidate_path"]),
                            *sorted(mapping.values()),
                        ],
                        "output_artifact_paths": sorted(rewritten.values()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                        "confidence": {
                            "score": max(0.0, min(1.0, score)),
                            "method": "deterministic_articulated_asset_retrieval",
                            "notes": "RGB appearance is not used for ranking",
                        },
                        "source": "retrieved",
                    },
                }
            )
            bundle_path = candidate_directory.parent / "candidate.json"
            write_json(bundle_path, source_candidate)
            bundle_relative = bundle_path.relative_to(input_root).as_posix()
            visual_paths = sorted(rewritten.values())
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "source_family": family,
                    "source_asset_id": asset_id,
                    "retrieval_score": score,
                    "evidence_terms": terms,
                    "production_selectable": source_candidate["production_selectable"],
                    "candidate_bundle_path": bundle_relative,
                    "candidate_bundle_sha256": sha256(bundle_path),
                    "visual_asset_paths": visual_paths,
                    "visual_asset_hashes": {
                        path: sha256(input_root / path) for path in visual_paths
                    },
                }
            )
            if len(candidates) >= int(request["maximum_candidates"]):
                break
        index_hash = request["asset_index_sha256"]
        warnings = (
            [] if candidates else ["no selected local asset had a normalized candidate bundle"]
        )
    family = str(request["source_family"])
    write_json(
        output_path,
        {
            "schema_version": "0.1.0",
            "articulated_object_id": request["articulated_object_id"],
            "measured_motion_sha256": request["measured_motion_sha256"],
            "candidates": candidates,
            "artvip_index_sha256": index_hash if family == "artvip" else None,
            "partnet_index_sha256": (index_hash if family == "partnet_mobility" else None),
            "runtime_seconds": time.monotonic() - started,
            "warnings": warnings,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("healthcheck", "retrieve"))
    parser.add_argument("--request")
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.action == "healthcheck":
        print(f"articulated_retrieval_worker {__version__}: offline-only")
        return 0
    if not args.request or not args.input_root:
        parser.error("retrieve requires --request and --input-root")
    retrieve(read_json(Path(args.request).resolve()), Path(args.input_root).resolve())
    return 0
