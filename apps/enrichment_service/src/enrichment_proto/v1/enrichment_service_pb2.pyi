from shared_proto.v1 import error_pb2 as _error_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("healthy", "service", "details_json", "error")
    HEALTHY_FIELD_NUMBER: _ClassVar[int]
    SERVICE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    healthy: bool
    service: str
    details_json: str
    error: _error_pb2.ErrorDetails
    def __init__(self, healthy: bool = ..., service: _Optional[str] = ..., details_json: _Optional[str] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...

class RunPipelineRequest(_message.Message):
    __slots__ = ("file_path", "index", "title", "agent_dir", "skip_services", "only_services")
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    AGENT_DIR_FIELD_NUMBER: _ClassVar[int]
    SKIP_SERVICES_FIELD_NUMBER: _ClassVar[int]
    ONLY_SERVICES_FIELD_NUMBER: _ClassVar[int]
    file_path: str
    index: int
    title: str
    agent_dir: str
    skip_services: _containers.RepeatedScalarFieldContainer[str]
    only_services: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, file_path: _Optional[str] = ..., index: _Optional[int] = ..., title: _Optional[str] = ..., agent_dir: _Optional[str] = ..., skip_services: _Optional[_Iterable[str]] = ..., only_services: _Optional[_Iterable[str]] = ...) -> None: ...

class RunPipelineResponse(_message.Message):
    __slots__ = ("success", "output_path", "result_json", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    RESULT_JSON_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    output_path: str
    result_json: str
    error: _error_pb2.ErrorDetails
    def __init__(self, success: bool = ..., output_path: _Optional[str] = ..., result_json: _Optional[str] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...
