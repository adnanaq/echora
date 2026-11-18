from shared_proto.v1 import merge_pb2 as _merge_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StorePageRequest(_message.Message):
    __slots__ = ("page",)
    PAGE_FIELD_NUMBER: _ClassVar[int]
    page: _merge_pb2.StoredPage
    def __init__(self, page: _Optional[_Union[_merge_pb2.StoredPage, _Mapping]] = ...) -> None: ...

class StorePageResponse(_message.Message):
    __slots__ = ("has_changed",)
    HAS_CHANGED_FIELD_NUMBER: _ClassVar[int]
    has_changed: bool
    def __init__(self, has_changed: _Optional[bool] = ...) -> None: ...
