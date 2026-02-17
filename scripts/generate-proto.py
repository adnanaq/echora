#!/usr/bin/env python3
"""
Generate checked-in Python gRPC stubs for vector and enrichment services.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
PROTO_ROOT = REPO_ROOT / "protos"

TARGETS = [
    {
        "name": "shared_proto",
        "protos": [
            PROTO_ROOT / "shared_proto" / "v1" / "error.proto",
            PROTO_ROOT / "shared_proto" / "v1" / "anime.proto",
        ],
        "proto_include": PROTO_ROOT,
        "out_root": REPO_ROOT / "libs" / "common" / "src" / "shared_proto",
        "relocate_shared_proto_v1_to_v1": True,
        "rewrites": (
            ("from v1 import", "from shared_proto.v1 import"),
            ("import v1.", "import shared_proto.v1."),
            ("from common.v1 import", "from shared_proto.v1 import"),
        ),
    },
    {
        "name": "vector_service",
        "protos": [
            PROTO_ROOT / "vector_service" / "v1" / "vector_admin.proto",
            PROTO_ROOT / "vector_service" / "v1" / "vector_search.proto",
        ],
        "proto_include": PROTO_ROOT / "vector_service",
        "proto_extra_includes": [PROTO_ROOT],
        "out_root": REPO_ROOT / "apps" / "vector_service" / "src" / "vector_proto",
        "rewrites": (
            ("from v1 import", "from vector_proto.v1 import"),
            ("import v1.", "import vector_proto.v1."),
            ("from vector_service.v1 import", "from vector_proto.v1 import"),
            ("from common.v1 import", "from shared_proto.v1 import"),
        ),
    },
    {
        "name": "enrichment_service",
        "protos": [
            PROTO_ROOT / "enrichment_service" / "v1" / "enrichment_service.proto",
        ],
        "proto_include": PROTO_ROOT / "enrichment_service",
        "proto_extra_includes": [PROTO_ROOT],
        "out_root": REPO_ROOT
        / "apps"
        / "enrichment_service"
        / "src"
        / "enrichment_proto",
        "rewrites": (
            ("from v1 import", "from enrichment_proto.v1 import"),
            ("import v1.", "import enrichment_proto.v1."),
            ("from enrichment_service.v1 import", "from enrichment_proto.v1 import"),
            ("from common.v1 import", "from shared_proto.v1 import"),
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


class _CommandFailedError(RuntimeError):
    """Raised when a subprocess command exits with a non-zero code."""

    def __init__(self, cmd: list[str], returncode: int) -> None:
        """Initialise with the failed command and its exit code.

        Args:
            cmd: The command list that was executed.
            returncode: Non-zero exit code returned by the subprocess.
        """
        self.cmd = cmd
        self.returncode = returncode
        super().__init__(f"Command failed (exit {returncode}): {' '.join(cmd)}")


def _run(cmd: list[str]) -> None:
    """Run a subprocess command relative to the repo root.

    Args:
        cmd: Command and arguments to execute.

    Raises:
        _CommandFailedError: If the process exits with a non-zero return code.
    """
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)  # noqa: S603
    if proc.returncode != 0:
        print(
            f"Command failed (exit {proc.returncode}): {' '.join(cmd)}",
            file=sys.stderr,
        )
        raise _CommandFailedError(cmd, proc.returncode)


def _validate_rewrite_rules(rewrites: tuple[tuple[str, str], ...]) -> None:
    """Validate rewrite rule ordering to avoid accidental double rewrites.

    Args:
        rewrites: Ordered tuple of (before, after) replacement rules.

    Raises:
        ValueError: If an ``after`` string can match a later rule's ``before``.
    """
    for i, (_before_i, after_i) in enumerate(rewrites):
        for before_j, _after_j in rewrites[i + 1 :]:
            if before_j in after_i:
                raise ValueError(
                    "Invalid rewrite rule order: "
                    f"after='{after_i}' contains later before='{before_j}'"
                )


def _rewrite_generated_imports(
    out_root: Path, rewrites: tuple[tuple[str, str], ...]
) -> None:
    """Rewrite import statements in generated pb2 files to use package-scoped paths.

    Args:
        out_root: Directory containing the generated ``*_pb2.py``, ``*_pb2_grpc.py``,
            and ``*_pb2.pyi`` files.
        rewrites: Ordered sequence of ``(before, after)`` string replacement pairs.
    """
    if not rewrites:
        return
    # NOTE: rewrite order is significant. Ensure each "after" string does not
    # match any later rule's "before" pattern to avoid double rewriting.
    for generated_file in list(out_root.rglob("*_pb2.py")) + list(
        out_root.rglob("*_pb2_grpc.py")
    ) + list(out_root.rglob("*_pb2.pyi")):
        text = generated_file.read_text(encoding="utf-8")
        updated = text
        for before, after in rewrites:
            updated = updated.replace(before, after)
        if updated != text:
            generated_file.write_text(updated, encoding="utf-8")


def main() -> int:
    """Validate proto files exist, then regenerate all configured stub targets.

    Returns:
        0 on success, 2 on a configuration or missing-file error, or the
        non-zero exit code of a failed ``grpc_tools.protoc`` invocation.
    """
    missing_required: list[Path] = []
    per_target_protos: dict[str, list[Path]] = {}
    for entry in TARGETS:
        protos = [Path(p) for p in entry["protos"]]
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

    try:
        for entry in TARGETS:
            if entry["name"] not in per_target_protos:
                continue
            out_root = Path(entry["out_root"])
            protos = per_target_protos[entry["name"]]
            proto_include = Path(entry["proto_include"])
            proto_extra_includes = [
                Path(include) for include in entry.get("proto_extra_includes", [])
            ]
            proto_rel = [p.relative_to(proto_include) for p in protos]
            v1_dir = out_root / "v1"
            if v1_dir.exists():
                shutil.rmtree(v1_dir)
            if entry.get("relocate_shared_proto_v1_to_v1", False):
                shared_dir = out_root / "shared_proto"
                if shared_dir.exists():
                    shutil.rmtree(shared_dir)
            out_root.mkdir(parents=True, exist_ok=True)
            rewrites = entry.get("rewrites", ())
            _validate_rewrite_rules(rewrites)
            cmd = [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                "-I",
                str(proto_include),
                *(
                    include_arg
                    for include in proto_extra_includes
                    for include_arg in ("-I", str(include))
                ),
                f"--python_out={out_root}",
                f"--pyi_out={out_root}",
                f"--grpc_python_out={out_root}",
                *(str(p) for p in proto_rel),
            ]
            _run(cmd)
            if entry.get("relocate_shared_proto_v1_to_v1", False):
                generated_shared_v1 = out_root / "shared_proto" / "v1"
                if generated_shared_v1.exists():
                    shutil.move(str(generated_shared_v1), str(v1_dir))
                    shutil.rmtree(out_root / "shared_proto", ignore_errors=True)
            _rewrite_generated_imports(out_root, rewrites)
            init_root = out_root / "__init__.py"
            v1_dir.mkdir(parents=True, exist_ok=True)
            init_v1 = v1_dir / "__init__.py"
            if not init_root.exists():
                init_root.write_text(
                    '"""Generated proto package."""\n', encoding="utf-8"
                )
            if not init_v1.exists():
                init_v1.write_text('"""Generated proto v1 package."""\n', encoding="utf-8")
    except _CommandFailedError as exc:
        return exc.returncode
    except ValueError as exc:
        print(f"Proto generation configuration error: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
