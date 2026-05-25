from shared_proto.v1 import error_pb2 as _error_pb2
from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FilterOperator(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILTER_OPERATOR_UNSPECIFIED: _ClassVar[FilterOperator]
    FILTER_OPERATOR_EQ: _ClassVar[FilterOperator]
    FILTER_OPERATOR_NE: _ClassVar[FilterOperator]
    FILTER_OPERATOR_IN: _ClassVar[FilterOperator]
    FILTER_OPERATOR_NOT_IN: _ClassVar[FilterOperator]
    FILTER_OPERATOR_RANGE: _ClassVar[FilterOperator]

class FilterClause(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILTER_CLAUSE_UNSPECIFIED: _ClassVar[FilterClause]
    FILTER_CLAUSE_MUST: _ClassVar[FilterClause]
    FILTER_CLAUSE_MUST_NOT: _ClassVar[FilterClause]
    FILTER_CLAUSE_SHOULD: _ClassVar[FilterClause]
FILTER_OPERATOR_UNSPECIFIED: FilterOperator
FILTER_OPERATOR_EQ: FilterOperator
FILTER_OPERATOR_NE: FilterOperator
FILTER_OPERATOR_IN: FilterOperator
FILTER_OPERATOR_NOT_IN: FilterOperator
FILTER_OPERATOR_RANGE: FilterOperator
FILTER_CLAUSE_UNSPECIFIED: FilterClause
FILTER_CLAUSE_MUST: FilterClause
FILTER_CLAUSE_MUST_NOT: FilterClause
FILTER_CLAUSE_SHOULD: FilterClause

class FilterCondition(_message.Message):
    __slots__ = ("field", "operator", "value", "clause")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    OPERATOR_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    CLAUSE_FIELD_NUMBER: _ClassVar[int]
    field: str
    operator: FilterOperator
    value: _struct_pb2.Value
    clause: FilterClause
    def __init__(self, field: _Optional[str] = ..., operator: _Optional[_Union[FilterOperator, str]] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ..., clause: _Optional[_Union[FilterClause, str]] = ...) -> None: ...

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
    filters: _containers.RepeatedCompositeFieldContainer[FilterCondition]
    def __init__(self, query_text: _Optional[str] = ..., image: _Optional[bytes] = ..., entity_type: _Optional[str] = ..., limit: _Optional[int] = ..., filters: _Optional[_Iterable[_Union[FilterCondition, _Mapping]]] = ...) -> None: ...

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
