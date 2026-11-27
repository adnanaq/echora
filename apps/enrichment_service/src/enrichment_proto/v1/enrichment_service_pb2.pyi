import datetime

from google.protobuf import duration_pb2 as _duration_pb2
from shared_proto.v1 import error_pb2 as _error_pb2
from shared_proto.v1 import merge_pb2 as _merge_pb2
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
    def __init__(self, healthy: _Optional[bool] = ..., service: _Optional[str] = ..., details_json: _Optional[str] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...

class RunPipelineRequest(_message.Message):
    __slots__ = ("file_path", "index", "title", "agent_dir", "skip_services", "only_services", "skip_characters", "skip_episodes")
    FILE_PATH_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    AGENT_DIR_FIELD_NUMBER: _ClassVar[int]
    SKIP_SERVICES_FIELD_NUMBER: _ClassVar[int]
    ONLY_SERVICES_FIELD_NUMBER: _ClassVar[int]
    SKIP_CHARACTERS_FIELD_NUMBER: _ClassVar[int]
    SKIP_EPISODES_FIELD_NUMBER: _ClassVar[int]
    file_path: str
    index: int
    title: str
    agent_dir: str
    skip_services: _containers.RepeatedScalarFieldContainer[str]
    only_services: _containers.RepeatedScalarFieldContainer[str]
    skip_characters: bool
    skip_episodes: bool
    def __init__(self, file_path: _Optional[str] = ..., index: _Optional[int] = ..., title: _Optional[str] = ..., agent_dir: _Optional[str] = ..., skip_services: _Optional[_Iterable[str]] = ..., only_services: _Optional[_Iterable[str]] = ..., skip_characters: _Optional[bool] = ..., skip_episodes: _Optional[bool] = ...) -> None: ...

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
    def __init__(self, success: _Optional[bool] = ..., output_path: _Optional[str] = ..., result_json: _Optional[str] = ..., error: _Optional[_Union[_error_pb2.ErrorDetails, _Mapping]] = ...) -> None: ...

class FetchPageRequest(_message.Message):
    __slots__ = ("page_key", "max_age")
    PAGE_KEY_FIELD_NUMBER: _ClassVar[int]
    MAX_AGE_FIELD_NUMBER: _ClassVar[int]
    page_key: _merge_pb2.PageKey
    max_age: _duration_pb2.Duration
    def __init__(self, page_key: _Optional[_Union[_merge_pb2.PageKey, _Mapping]] = ..., max_age: _Optional[_Union[datetime.timedelta, _duration_pb2.Duration, _Mapping]] = ...) -> None: ...

class FetchPageResponse(_message.Message):
    __slots__ = ("page",)
    PAGE_FIELD_NUMBER: _ClassVar[int]
    page: _merge_pb2.StoredPage
    def __init__(self, page: _Optional[_Union[_merge_pb2.StoredPage, _Mapping]] = ...) -> None: ...
