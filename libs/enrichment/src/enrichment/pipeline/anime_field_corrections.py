"""Apply approved FieldCorrection protobuf overrides onto merged entity messages."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any, cast

from google.protobuf import wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto  # ty: ignore[unresolved-import]
from google.protobuf.descriptor import Descriptor, FieldDescriptor
from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp  # ty: ignore[unresolved-import]
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import FieldCorrection

logger = logging.getLogger(__name__)

_WRAPPER_NAMES: tuple[str, ...] = (
    "StringValue",
    "Int32Value",
    "Int64Value",
    "BoolValue",
    "FloatValue",
    "DoubleValue",
)

_WRAPPER_TYPES_BY_FULL_NAME: dict[str, type[Message]] = {
    wrapper_cls.DESCRIPTOR.full_name: wrapper_cls
    for wrapper_name in _WRAPPER_NAMES
    if (wrapper_cls := getattr(wrappers_pb2, wrapper_name)).DESCRIPTOR is not None
}

_MESSAGE_TYPES_BY_FULL_NAME: dict[str, type[Message]] = {
    descriptor.full_name: message_cls
    for message_cls in (
        anime_pb2.AiredDates,
        anime_pb2.Broadcast,
        anime_pb2.AnimeHiatus,
        anime_pb2.AnimeImages,
        anime_pb2.ScoreCalculations,
        Timestamp,
    )
    if (descriptor := message_cls.DESCRIPTOR) is not None
}

_PROTECTED_FIELD_PATHS: frozenset[str] = frozenset(
    {"id", "anime_id", "entity_type", "roles", "anime_ids"}
)


def _unpack_any_value(any_value: AnyProto) -> Any:
    """Unpack a ``google.protobuf.Any`` wrapper or known message into a Python value.

    Args:
        any_value: Packed protobuf ``Any`` payload.

    Returns:
        Unpacked scalar or protobuf message, or ``None`` if unsupported.
    """
    type_name = any_value.type_url.rsplit("/", 1)[-1]
    if (wrapper_cls := _WRAPPER_TYPES_BY_FULL_NAME.get(type_name)) is not None:
        wrapper_message = wrapper_cls()
        wrapper_message.ParseFromString(any_value.value)
        return getattr(wrapper_message, "value")
    if (message_cls := _MESSAGE_TYPES_BY_FULL_NAME.get(type_name)) is not None:
        nested_message = message_cls()
        nested_message.ParseFromString(any_value.value)
        return nested_message

    logger.warning(f"Unsupported FieldCorrection Any type_url: {any_value.type_url!r}")
    return None


def _is_string_map_field(field_descriptor: FieldDescriptor) -> bool:
    """Check whether a protobuf field descriptor is a ``map<string, string>``.

    Args:
        field_descriptor: Protobuf field descriptor to inspect.

    Returns:
        ``True`` when the field is a string-keyed, string-valued protobuf map.
    """
    if (
        field_descriptor.type != FieldDescriptor.TYPE_MESSAGE
        or not field_descriptor.is_repeated
    ):
        return False
    message_type = field_descriptor.message_type
    if message_type is None or not message_type.GetOptions().map_entry:
        return False
    key_field = message_type.fields_by_name.get("key")
    value_field = message_type.fields_by_name.get("value")
    return (
        key_field is not None
        and value_field is not None
        and key_field.type == FieldDescriptor.TYPE_STRING
        and value_field.type == FieldDescriptor.TYPE_STRING
    )


def _resolve_enum_number(
    field_descriptor: FieldDescriptor,
    value: Any,
) -> int | None:
    """Resolve a string or integer correction value into a protobuf enum number.

    Args:
        field_descriptor: Enum ``FieldDescriptor`` on the target message.
        value: Candidate enum name string or integer enum number.

    Returns:
        Matching protobuf enum number, or ``None`` if the value is invalid.
    """
    enum_type = field_descriptor.enum_type
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value in enum_type.values_by_number else None
    if not isinstance(value, str):
        return None
    enum_value = enum_type.values_by_name.get(value)
    if enum_value is None:
        normalized = value.strip().upper().replace(" ", "_")
        enum_prefix = re.sub(r"(?<!^)(?=[A-Z])", "_", enum_type.name).upper()
        enum_value = enum_type.values_by_name.get(f"{enum_prefix}_{normalized}")
    return enum_value.number if enum_value is not None else None


def _assign_leaf_field(
    target_message: Message,
    field_path: str,
    leaf_name: str,
    value: Any,
) -> None:
    """Assign a validated correction value onto a singular leaf protobuf field.

    Args:
        target_message: Protobuf message owning the leaf field.
        field_path: Full dotted path used in warning logs.
        leaf_name: Name of the leaf field on ``target_message``.
        value: Unpacked correction value.
    """
    message_descriptor = cast(Descriptor, target_message.DESCRIPTOR)
    field_descriptor = message_descriptor.fields_by_name.get(leaf_name)
    if field_descriptor is None:
        logger.warning(f"Ignoring unknown correction path {field_path!r}")
        return
    if field_descriptor.is_repeated:
        logger.warning(f"Ignoring correction for repeated or map field {field_path!r}")
        return
    if field_descriptor.type == FieldDescriptor.TYPE_ENUM:
        enum_number = _resolve_enum_number(field_descriptor, value)
        if enum_number is None:
            logger.warning(
                "Ignoring invalid enum correction %r for path %r", value, field_path
            )
            return
        setattr(target_message, leaf_name, enum_number)
        return
    if field_descriptor.type == FieldDescriptor.TYPE_MESSAGE:
        expected_name = field_descriptor.message_type.full_name
        if (
            not isinstance(value, Message)
            or value.DESCRIPTOR.full_name != expected_name
        ):
            logger.warning(
                "Ignoring mismatched message correction for path %r (expected %s)",
                field_path,
                expected_name,
            )
            return
        getattr(target_message, leaf_name).CopyFrom(value)
        return
    try:
        setattr(target_message, leaf_name, value)
    except TypeError, ValueError:
        logger.warning(
            "Ignoring mismatched scalar correction %r for path %r", value, field_path
        )


def _apply_single_correction(
    target_message: Message,
    field_path: str,
    value: Any,
) -> None:
    """Apply one unpacked field correction onto ``target_message`` at ``field_path``.

    Supports singular scalar, enum, and message leaves as well as single-key
    updates on ``map<string, string>`` fields (such as ``titles.en`` or
    ``spoilers.description``). Protected identity and derived fields are refused.

    Args:
        target_message: Target protobuf message mutated in place.
        field_path: Dotted field path on the message (e.g. ``'episode_count'``).
        value: Unpacked correction value.
    """
    segments = [segment for segment in field_path.split(".") if segment]
    if not segments:
        return
    if segments[0] in _PROTECTED_FIELD_PATHS:
        logger.warning(f"Refusing correction for protected field path {field_path!r}")
        return

    current_message: Message = target_message
    for index, segment in enumerate(segments[:-1]):
        current_descriptor = cast(Descriptor, current_message.DESCRIPTOR)
        field_descriptor = current_descriptor.fields_by_name.get(segment)
        if field_descriptor is None:
            logger.warning(f"Ignoring unknown correction path {field_path!r}")
            return
        if _is_string_map_field(field_descriptor) and index == len(segments) - 2:
            if not isinstance(value, str):
                logger.warning(
                    "Ignoring non-string value %r for string map path %r",
                    value,
                    field_path,
                )
                return
            getattr(current_message, segment)[segments[-1]] = value
            return
        if (
            field_descriptor.type != FieldDescriptor.TYPE_MESSAGE
            or field_descriptor.is_repeated
        ):
            logger.warning(
                "Ignoring correction traversing non-message segment %r in %r",
                segment,
                field_path,
            )
            return
        current_message = getattr(current_message, segment)

    _assign_leaf_field(current_message, field_path, segments[-1], value)


def apply_field_corrections(
    target_message: Message,
    corrections: Sequence[FieldCorrection],
) -> None:
    """Apply approved ``FieldCorrection`` overrides in ``approved_at`` order.

    Corrections are stably sorted by ``(approved_at.seconds, approved_at.nanos)``
    so the most recently approved override wins when multiple corrections target
    the same field.

    Args:
        target_message: Target protobuf message mutated in place.
        corrections: Sequence of approved ``FieldCorrection`` messages.
    """
    ordered_corrections = sorted(
        corrections,
        key=lambda correction: (
            correction.approved_at.seconds,
            correction.approved_at.nanos,
        ),
    )
    for correction in ordered_corrections:
        if not correction.field_path or not correction.HasField("value"):
            continue
        unpacked = _unpack_any_value(correction.value)
        if unpacked is None:
            continue
        _apply_single_correction(target_message, correction.field_path, unpacked)
