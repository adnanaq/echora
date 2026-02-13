#!/usr/bin/env python3
"""
Generate checked-in Python gRPC stubs for vector and enrichment services.
"""

from __future__ import annotations

import subprocess
import sys
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTO_ROOT = REPO_ROOT / "protos"

TARGETS = [
    {
        "name": "vector_service",
        "protos": [
            PROTO_ROOT / "vector_service" / "v1" / "vector_common.proto",
            PROTO_ROOT / "vector_service" / "v1" / "vector_admin.proto",
            PROTO_ROOT / "vector_service" / "v1" / "vector_search.proto",
        ],
        "proto_include": PROTO_ROOT / "vector_service",
        "out_root": REPO_ROOT / "apps" / "vector_service" / "src" / "vector_proto",
        "rewrites": (
            ("from v1 import", "from vector_proto.v1 import"),
            ("import v1.", "import vector_proto.v1."),
            ("from vector_service.v1 import", "from vector_proto.v1 import"),
        ),
    },
    {
        "name": "enrichment_service",
        "proto": PROTO_ROOT / "enrichment_service" / "v1" / "enrichment_service.proto",
        "proto_include": PROTO_ROOT / "enrichment_service",
        "out_root": REPO_ROOT / "apps" / "enrichment_service" / "src" / "enrichment_proto",
        "rewrites": (
            ("from v1 import", "from enrichment_proto.v1 import"),
            ("import v1.", "import enrichment_proto.v1."),
            ("from enrichment_service.v1 import", "from enrichment_proto.v1 import"),
        ),
    },
    # Agent proto generation is disabled for now.
    # {
    #     "name": "agent_service",
    #     "optional": True,
    #     "protos": [
    #         PROTO_ROOT / "agent_service" / "v1" / "agent_search.proto",
    #     ],
    #     "proto_include": PROTO_ROOT / "agent_service",
    #     "out_root": REPO_ROOT / "apps" / "agent_service" / "src" / "agent_proto",
    #     "rewrites": (
    #         ("from v1 import", "from agent_proto.v1 import"),
    #         ("import v1.", "import agent_proto.v1."),
    #         ("from agent.v1 import", "from agent_proto.v1 import"),
    #         ("from agent_service.v1 import", "from agent_proto.v1 import"),
    #     ),
    # },
]


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _rewrite_generated_imports(
    out_root: Path, rewrites: tuple[tuple[str, str], ...]
) -> None:
    if not rewrites:
        return
    for generated_file in list(out_root.rglob("*_pb2.py")) + list(
        out_root.rglob("*_pb2_grpc.py")
    ):
        text = generated_file.read_text(encoding="utf-8")
        updated = text
        for before, after in rewrites:
            updated = updated.replace(before, after)
        if updated != text:
            generated_file.write_text(updated, encoding="utf-8")


def main() -> int:
    missing_required: list[Path] = []
    per_target_protos: dict[str, list[Path]] = {}
    for entry in TARGETS:
        protos: list[Path]
        if "protos" in entry:
            protos = [Path(p) for p in entry["protos"]]
        else:
            protos = [Path(entry["proto"])]
        missing = [p for p in protos if not p.exists()]
        if missing:
            if entry.get("optional", False):
                print(
                    f"Skipping optional proto target '{entry['name']}' "
                    f"(missing: {', '.join(str(p) for p in missing)})"
                )
                continue
            missing_required.extend(missing)
            continue
        per_target_protos[entry["name"]] = protos

    if missing_required:
        print("Missing proto files:", file=sys.stderr)
        for path in missing_required:
            print(f"- {path}", file=sys.stderr)
        return 2

    for entry in TARGETS:
        if entry["name"] not in per_target_protos:
            continue
        out_root = Path(entry["out_root"])
        protos = per_target_protos[entry["name"]]
        proto_include = Path(entry["proto_include"])
        proto_rel = [p.relative_to(proto_include) for p in protos]
        v1_dir = out_root / "v1"
        if v1_dir.exists():
            shutil.rmtree(v1_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I",
            str(proto_include),
            f"--python_out={out_root}",
            f"--grpc_python_out={out_root}",
            *(str(p) for p in proto_rel),
        ]
        _run(cmd)
        _rewrite_generated_imports(out_root, entry.get("rewrites", ()))
        init_root = out_root / "__init__.py"
        init_v1 = out_root / "v1" / "__init__.py"
        if not init_root.exists():
            init_root.write_text('"""Generated proto package."""\n', encoding="utf-8")
        if not init_v1.exists():
            init_v1.write_text('"""Generated proto v1 package."""\n', encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
