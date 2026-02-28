from shared_proto.v1 import error_pb2 as _error_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SearchRequest(_message.Message):
    __slots__ = ("query_text", "image", "entity_type", "limit", "filters")
    QUERY_TEXT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    ENTITY_TYPE_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    FILTERS_FIELD_NUMBER: _ClassVar[int]
    query_text: str
    image: bytes
    entity_type: str
    limit: int
    filters: _struct_pb2.Struct
    def __init__(self, query_text: _Optional[str] = ..., image: _Optional[bytes] = ..., entity_type: _Optional[str] = ..., limit: _Optional[int] = ..., filters: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class SearchData(_message.Message):
    __slots__ = ("id", "similarity_score", "payload_json")
    ID_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_SCORE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    id: str
    similarity_score: float
    payload_json: str
    def __init__(self, id: _Optional[str] = ..., similarity_score: _Optional[float] = ..., payload_json: _Optional[str] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ("data", "error")
    DATA_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[SearchData]
    error: _error_pb2.ErrorDetails
    def __init__(self, data: _Optional[_Iterable[_Union[SearchData, _Mapping]]] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...
