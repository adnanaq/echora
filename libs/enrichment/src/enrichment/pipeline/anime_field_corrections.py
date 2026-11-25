"""Apply approved FieldCorrection protobuf overrides onto merged Anime messages."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import Any

from google.protobuf import wrappers_pb2
from google.protobuf.any_pb2 import Any as AnyProto  # ty: ignore[unresolved-import]
from google.protobuf.message import Message
from google.protobuf.timestamp_pb2 import Timestamp  # ty: ignore[unresolved-import]
from shared_proto.v1 import anime_pb2
from shared_proto.v1.merge_pb2 import FieldCorrection

logger = logging.getLogger(__name__)


def _unpack_any_value(any_value: AnyProto) -> Any:
    """Unpack a ``google.protobuf.Any`` wrapper or known message into a Python value.

    Args:
        any_value: Packed protobuf ``Any`` payload.

    Returns:
        Unpacked scalar or protobuf message, or ``None`` if unsupported.
    """
    wrapper_names = (
        "StringValue",
        "Int32Value",
        "Int64Value",
        "BoolValue",
        "FloatValue",
        "DoubleValue",
    )
    for wrapper_name in wrapper_names:
        wrapper_cls = getattr(wrappers_pb2, wrapper_name)
        if any_value.Is(wrapper_cls.DESCRIPTOR):
            msg = wrapper_cls()
            any_value.Unpack(msg)
            return msg.value

    message_types = (
        anime_pb2.AiredDates,
        anime_pb2.Broadcast,
        anime_pb2.AnimeHiatus,
        anime_pb2.AnimeImages,
        anime_pb2.ScoreCalculations,
        Timestamp,
    )
    for msg_cls in message_types:
        if any_value.Is(msg_cls.DESCRIPTOR):
            msg = msg_cls()
            any_value.Unpack(msg)
            return msg

    logger.warning(f"Unsupported FieldCorrection Any type_url: {any_value.type_url!r}")
    return None


def _apply_single_correction(target_msg: Message, field_path: str, value: Any) -> None:
    """Apply one unpacked field correction onto ``target_msg`` at ``field_path``.

    Args:
        target_msg: Target protobuf message mutated in place.
        field_path: Dotted field path on the message (e.g. ``'episode_count'``).
        value: Unpacked correction value.
    """
    parts = [part for part in field_path.split(".") if part]
    if not parts:
        return
    target: Any = target_msg
    for attr in parts[:-1]:
        if not hasattr(target, attr):
            logger.warning(f"Ignoring unknown correction path {field_path!r}")
            return
        target = getattr(target, attr)

    leaf = parts[-1]
    field_desc = target.DESCRIPTOR.fields_by_name.get(leaf)
    if field_desc is None:
        logger.warning(f"Ignoring unknown correction path {field_path!r}")
        return

    if field_desc.type == field_desc.TYPE_ENUM and isinstance(value, str):
        enum_value = field_desc.enum_type.values_by_name.get(value)
        if enum_value is None:
            normalized = value.strip().upper().replace(" ", "_")
            enum_prefix = re.sub(
                r"(?<!^)(?=[A-Z])", "_", field_desc.enum_type.name
            ).upper()
            prefixed = f"{enum_prefix}_{normalized}"
            enum_value = field_desc.enum_type.values_by_name.get(prefixed)
        if enum_value is not None:
            setattr(target, leaf, enum_value.number)
        return

    if field_desc.type == field_desc.TYPE_MESSAGE:
        getattr(target, leaf).CopyFrom(value)
        return

    setattr(target, leaf, value)


def apply_field_corrections(
    target_msg: Message,
    corrections: Sequence[FieldCorrection],
) -> None:
    """Apply approved ``FieldCorrection`` overrides onto a merged protobuf message.

    Args:
        target_msg: Target protobuf message mutated in place.
        corrections: Sequence of approved ``FieldCorrection`` messages.
    """
    for correction in corrections:
        if not correction.field_path or not correction.HasField("value"):
            continue
        unpacked = _unpack_any_value(correction.value)
        if unpacked is None:
            continue
        _apply_single_correction(target_msg, correction.field_path, unpacked)
