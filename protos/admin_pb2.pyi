from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GetStatsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetStatsResponse(_message.Message):
    __slots__ = ("collection_name", "total_documents", "vector_size", "distance_metric", "status", "additional_stats")
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    TOTAL_DOCUMENTS_FIELD_NUMBER: _ClassVar[int]
    VECTOR_SIZE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_METRIC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ADDITIONAL_STATS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    total_documents: int
    vector_size: int
    distance_metric: str
    status: str
    additional_stats: _struct_pb2.Struct
    def __init__(self, collection_name: _Optional[str] = ..., total_documents: _Optional[int] = ..., vector_size: _Optional[int] = ..., distance_metric: _Optional[str] = ..., status: _Optional[str] = ..., additional_stats: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...
