#!/usr/bin/env python3
"""Validate conformance between anime Pydantic models and shared proto schema."""

from __future__ import annotations

import re
import sys
from enum import Enum

from pydantic import BaseModel

from common.models import anime as anime_models
from shared_proto.v1 import anime_pb2

# ProductionStaff accepts dynamic role fields by design.
_SKIP_MODEL_MESSAGES = {"ProductionStaff"}


def _module_model_classes() -> list[type[BaseModel]]:
    """Return all Pydantic BaseModel subclasses defined in the anime models module.

    Returns:
        Sorted list of BaseModel subclasses whose ``__module__`` matches the
        anime models module.
    """
    model_classes: list[type[BaseModel]] = []
    for obj in vars(anime_models).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseModel)
            and obj is not BaseModel
            and obj.__module__ == anime_models.__name__
        ):
            model_classes.append(obj)
    return sorted(model_classes, key=lambda cls: cls.__name__)


def _module_enum_classes() -> list[type[Enum]]:
    """Return all Enum subclasses defined in the anime models module.

    Returns:
        Sorted list of Enum subclasses whose ``__module__`` matches the
        anime models module.
    """
    enum_classes: list[type[Enum]] = []
    for obj in vars(anime_models).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Enum)
            and obj is not Enum
            and obj.__module__ == anime_models.__name__
        ):
            enum_classes.append(obj)
    return sorted(enum_classes, key=lambda cls: cls.__name__)


def _enum_prefix(enum_name: str) -> str:
    """Convert a CamelCase enum name to the SCREAMING_SNAKE prefix used in protobuf.

    Args:
        enum_name: CamelCase enum class name (e.g., ``AnimeStatus``).

    Returns:
        SCREAMING_SNAKE prefix expected in proto enum value names
        (e.g., ``ANIME_STATUS``).
    """
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", enum_name).upper()
    return snake


def _check_messages() -> list[str]:
    """Check that every Pydantic model class has a matching proto message with identical field names.

    Returns:
        List of failure description strings; empty if all checks pass.
    """
    proto_messages = anime_pb2.DESCRIPTOR.message_types_by_name
    failures: list[str] = []

    for model_class in _module_model_classes():
        if model_class.__name__ in _SKIP_MODEL_MESSAGES:
            continue

        proto_message = proto_messages.get(model_class.__name__)
        if proto_message is None:
            failures.append(
                f"Missing proto message for model class '{model_class.__name__}'."
            )
            continue

        model_field_names = set(model_class.model_fields.keys())
        proto_field_names = set(proto_message.fields_by_name.keys())

        missing_in_proto = sorted(model_field_names - proto_field_names)
        extra_in_proto = sorted(proto_field_names - model_field_names)
        if missing_in_proto or extra_in_proto:
            failures.append(
                f"{model_class.__name__}: "
                f"missing_in_proto={missing_in_proto}, "
                f"extra_in_proto={extra_in_proto}"
            )
    return failures


def _check_enums() -> list[str]:
    """Check that every Python Enum class has a matching proto enum with identical value names.

    Returns:
        List of failure description strings; empty if all checks pass.
    """
    proto_enums = anime_pb2.DESCRIPTOR.enum_types_by_name
    failures: list[str] = []

    for enum_class in _module_enum_classes():
        proto_enum = proto_enums.get(enum_class.__name__)
        if proto_enum is None:
            failures.append(f"Missing proto enum for model enum '{enum_class.__name__}'.")
            continue

        prefix = _enum_prefix(enum_class.__name__)
        proto_names = set(proto_enum.values_by_name.keys())
        expected_enum_names = {f"{prefix}_{member.name}" for member in enum_class}

        missing_in_proto = sorted(expected_enum_names - proto_names)
        extra_in_proto = sorted(
            name
            for name in proto_names
            if name != f"{prefix}_UNSPECIFIED" and name not in expected_enum_names
        )
        if missing_in_proto or extra_in_proto:
            failures.append(
                f"{enum_class.__name__}: "
                f"missing_in_proto={missing_in_proto}, "
                f"extra_in_proto={extra_in_proto}"
            )
    return failures


def main() -> int:
    """Run all model/proto contract checks and report results.

    Returns:
        0 on success, 1 if any drift was detected.
    """
    failures = _check_messages() + _check_enums()
    if failures:
        print("Anime model/proto contract drift detected:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Anime model/proto contract check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
