import functools
import hashlib
import json
import re
import types
from collections.abc import Iterator
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from common.config import CacheConfig, RedisConfig
from http_cache import configure_cache
from http_cache.result_cache import (
    _compute_schema_hash,
    _generate_cache_key,
    _normalize_batch_args,
    _stable_repr,
    cached_result,
)
from redis.asyncio import Redis
from redis.asyncio.client import Pipeline
from redis.exceptions import RedisError
from redis.typing import EncodableT, ExpiryT, KeysT, KeyT

MAX_KEY_LENGTH = 200
HEX_DIGITS = set("0123456789abcdef")

_SELECTORS = {"title": "//h1"}
_DELAY = 1.0


def _read_selector() -> str:
    return _SELECTORS["title"]


def _read_selector_in_nested_function() -> str:
    def read() -> str:
        return _SELECTORS["title"]

    return read()


def _wait() -> float:
    return _DELAY


def _strip_title(text: str) -> str:
    return text.strip()


def _lower_title(text: str) -> str:
    return text.lower()


_title_cleaner = _strip_title


def _clean_title(text: str) -> str:
    return _title_cleaner(text)


def _extract_title(html: str) -> str:
    return _clean_title(html)


def _passed_through(func: Any) -> Any:
    @functools.wraps(func)
    def call(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return call


_decorated_cleaner = _passed_through(_clean_title)


def _clean_with_decorated_helper(text: str) -> str:
    return _decorated_cleaner(text)


async def _fetch_page(url: str) -> str:
    return url


async def _fetch_other_page(url: str) -> str:
    return url.upper()


_page_fetcher = _fetch_page


async def _read_fetched_page(url: str) -> str:
    return await _page_fetcher(url)


_library_function = json.dumps


def _call_library(value: object) -> str:
    return _library_function(value)


def _trim_front(text: str) -> str:
    return _trim_back(text[1:]) if text else text


def _trim_back(text: str) -> str:
    return _trim_front(text[:-1]) if text else text


def _with_module_data(func: Any, **data: object) -> Any:
    return types.FunctionType(
        func.__code__, {**func.__globals__, **data}, func.__name__
    )


def _text(value: EncodableT | KeyT) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _seconds(time: ExpiryT) -> int:
    return int(time.total_seconds()) if isinstance(time, timedelta) else time


class InMemoryPipeline(Pipeline):
    def __init__(self, store: InMemoryRedis) -> None:
        self.store = store
        self.queued: list[tuple[str, int, str]] = []

    def __del__(self, _warn: Any = None, _grl: Any = None) -> None:
        return None

    async def __aenter__(self) -> InMemoryPipeline:
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        return None

    def setex(self, name: KeyT, time: ExpiryT, value: EncodableT) -> InMemoryPipeline:
        self.queued.append((_text(name), _seconds(time), _text(value)))
        return self

    async def execute(self, raise_on_error: bool = True) -> list[bool]:
        for name, time, value in self.queued:
            self.store.values[name] = value
            self.store.ttls[name] = time
        return [True] * len(self.queued)


class InMemoryRedis(Redis):
    def __init__(self, failing: frozenset[str] = frozenset()) -> None:
        self.values: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.reads: list[str] = []
        self.failing = failing

    def __del__(self, _warn: Any = None, _grl: Any = None) -> None:
        return None

    async def get(self, name: KeyT) -> str | None:
        self._fail_if("get")
        self.reads.append(_text(name))
        return self.values.get(_text(name))

    async def mget(self, keys: KeysT, *args: EncodableT) -> list[bytes | str | None]:
        self._fail_if("mget")
        names = [_text(key) for key in ([keys] if isinstance(keys, str) else keys)]
        self.reads.extend(names)
        return [self.values.get(name) for name in names]

    async def setex(self, name: KeyT, time: ExpiryT, value: EncodableT) -> bool:
        self._fail_if("setex")
        self.values[_text(name)] = _text(value)
        self.ttls[_text(name)] = _seconds(time)
        return True

    def pipeline(
        self, transaction: bool = True, shard_hint: str | None = None
    ) -> InMemoryPipeline:
        self._fail_if("pipeline")
        return InMemoryPipeline(self)

    def _fail_if(self, command: str) -> None:
        if command in self.failing:
            raise RedisError(f"{command} failed")


class EmptyMgetRedis(InMemoryRedis):
    async def mget(self, keys: KeysT, *args: EncodableT) -> list[bytes | str | None]:
        return []


def _serve(store: Redis, **cache_fields: Any) -> AbstractContextManager[object]:
    cache = configure_cache(
        CacheConfig(cache_enabled=True, **cache_fields),
        RedisConfig(redis_url="redis://localhost:6379/15"),
    )
    return patch.object(cache, "get_result_client", autospec=True, return_value=store)


@pytest.fixture
def redis_store() -> Iterator[InMemoryRedis]:
    store = InMemoryRedis()
    with _serve(store):
        yield store


def _key(prefix: str, schema_hash: str, *args: Any, **kwargs: Any) -> str:
    return _generate_cache_key(MAX_KEY_LENGTH, prefix, schema_hash, *args, **kwargs)


def _is_hash(value: str) -> bool:
    return len(value) == 16 and set(value) <= HEX_DIGITS


def test_compute_schema_hash_regular_function_returns_16_hex_characters() -> None:
    def fetch() -> str:
        return "test"

    assert _is_hash(_compute_schema_hash(fetch))


def test_compute_schema_hash_same_function_returns_same_hash() -> None:
    def fetch() -> str:
        return "test"

    assert _compute_schema_hash(fetch) == _compute_schema_hash(fetch)


def test_compute_schema_hash_different_functions_return_different_hashes() -> None:
    def first() -> str:
        return "first"

    def second() -> str:
        return "second"

    assert _compute_schema_hash(first) != _compute_schema_hash(second)


def test_compute_schema_hash_changed_body_returns_different_hash() -> None:
    def fetch_v1() -> str:
        return "version 1"

    def fetch_v2() -> str:
        return "version 2"

    assert _compute_schema_hash(fetch_v1) != _compute_schema_hash(fetch_v2)


def test_compute_schema_hash_lambda_returns_16_hex_characters() -> None:
    increment = lambda number: number + 1  # noqa: E731

    assert _is_hash(_compute_schema_hash(increment))


def test_compute_schema_hash_builtin_hashes_its_name() -> None:
    assert _compute_schema_hash(len) == hashlib.sha256(b"len").hexdigest()[:16]


def test_compute_schema_hash_multiline_function_returns_16_hex_characters() -> None:
    def add(first: int, second: int) -> int:
        total = first + second
        if total > 10:
            return total * 2
        return total

    assert _is_hash(_compute_schema_hash(add))


def test_compute_schema_hash_decorated_function_returns_16_hex_characters() -> None:
    def forward(func):  # type: ignore[no-untyped-def]
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            return func(*args, **kwargs)

        return wrapper

    @forward
    def decorated() -> str:
        return "decorated"

    assert _is_hash(_compute_schema_hash(decorated))


def test_compute_schema_hash_none_dependencies_matches_no_dependencies() -> None:
    def fetch() -> str:
        return "main"

    assert _compute_schema_hash(fetch) == _compute_schema_hash(fetch, None)


def test_compute_schema_hash_empty_dependencies_matches_no_dependencies() -> None:
    def fetch() -> str:
        return "main"

    assert _compute_schema_hash(fetch) == _compute_schema_hash(fetch, [])


def test_compute_schema_hash_added_dependency_returns_different_hash() -> None:
    def parse() -> str:
        return "dep_v1"

    def fetch() -> str:
        return parse()

    assert _compute_schema_hash(fetch) != _compute_schema_hash(fetch, [parse])


def test_compute_schema_hash_changed_dependency_returns_different_hash() -> None:
    def fetch() -> str:
        return "main"

    def parse_v1() -> str:
        return "schema_v1"

    def parse_v2() -> str:
        return "schema_v2"

    assert _compute_schema_hash(fetch, [parse_v1]) != _compute_schema_hash(
        fetch, [parse_v2]
    )


def test_compute_schema_hash_each_dependency_changes_hash() -> None:
    def fetch() -> str:
        return "main"

    def parse_a() -> str:
        return "a"

    def parse_b() -> str:
        return "b"

    hashes = {
        _compute_schema_hash(fetch, [parse_a]),
        _compute_schema_hash(fetch, [parse_b]),
        _compute_schema_hash(fetch, [parse_a, parse_b]),
    }

    assert len(hashes) == 3


def test_compute_schema_hash_same_dependencies_return_same_hash() -> None:
    def parse() -> str:
        return "stable_dep"

    def fetch() -> str:
        return parse()

    assert _compute_schema_hash(fetch, [parse]) == _compute_schema_hash(fetch, [parse])


def test_compute_schema_hash_builtin_dependency_returns_16_hex_characters() -> None:
    def fetch() -> int:
        return len([])

    assert _is_hash(_compute_schema_hash(fetch, [len]))


def test_compute_schema_hash_lambda_dependency_returns_16_hex_characters() -> None:
    double = lambda number: number * 2  # noqa: E731

    def fetch(number: int) -> int:
        return double(number)

    assert _is_hash(_compute_schema_hash(fetch, [double]))


def test_compute_schema_hash_dependency_without_source_still_changes_hash() -> None:
    def parse() -> str:
        return "good"

    assert _compute_schema_hash(parse, [parse]) != _compute_schema_hash(
        parse, [parse, len]
    )


def test_generate_cache_key_without_arguments_returns_prefix_and_hash() -> None:
    assert _key("fetch", "abc12345") == "result_cache:fetch:abc12345"


@pytest.mark.parametrize(
    ("args", "suffix"),
    [
        (("arg1",), "arg1"),
        (("arg1", "arg2", "arg3"), "arg1:arg2:arg3"),
        ((42,), "42"),
        ((3.14,), "3.14"),
        ((True,), "True"),
        ((False,), "False"),
        (({"key": "value"},), '{"key": "value"}'),
        (([1, 2, 3],), "[1, 2, 3]"),
        (({"z": 1, "a": 2, "m": 3},), '{"a": 2, "m": 3, "z": 1}'),
    ],
)
def test_generate_cache_key_positional_arguments_appended_in_order(
    args: tuple[Any, ...], suffix: str
) -> None:
    assert _key("fetch", "abc12345", *args) == f"result_cache:fetch:abc12345:{suffix}"


@pytest.mark.parametrize(
    ("kwargs", "suffix"),
    [
        ({"param1": "value1", "param2": "value2"}, "param1=value1:param2=value2"),
        ({"optional": None}, "optional=None"),
        ({"items": [{"id": 1}, {"id": 2}]}, 'items=[{"id": 1}, {"id": 2}]'),
        (
            {"options": {"nested": {"key": "value"}, "list": [1, 2]}},
            'options={"list": [1, 2], "nested": {"key": "value"}}',
        ),
    ],
)
def test_generate_cache_key_keyword_arguments_appended_as_name_value(
    kwargs: dict[str, Any], suffix: str
) -> None:
    assert (
        _key("fetch", "abc12345", **kwargs) == f"result_cache:fetch:abc12345:{suffix}"
    )


def test_generate_cache_key_keyword_order_returns_same_key() -> None:
    assert _key("fetch", "abc12345", z="last", a="first", m="middle") == _key(
        "fetch", "abc12345", a="first", m="middle", z="last"
    )


def test_generate_cache_key_same_inputs_return_same_key() -> None:
    assert _key("fetch", "hash123", "arg1", 42, param="value") == _key(
        "fetch", "hash123", "arg1", 42, param="value"
    )


def test_generate_cache_key_different_schema_hash_returns_different_key() -> None:
    first = _key("fetch", "hash111", "arg1")
    second = _key("fetch", "hash222", "arg1")

    assert first == "result_cache:fetch:hash111:arg1"
    assert second == "result_cache:fetch:hash222:arg1"


def test_generate_cache_key_longer_than_limit_hashes_arguments() -> None:
    long_arg = "x" * MAX_KEY_LENGTH
    full_key = f"fetch:abc12345:{long_arg}"

    key = _key("fetch", "abc12345", long_arg)

    assert key == (
        f"result_cache:fetch:abc12345:{hashlib.sha256(full_key.encode()).hexdigest()}"
    )


def test_generate_cache_key_custom_limit_keeps_key_within_it() -> None:
    long_arg = "x" * 300

    assert long_arg in _generate_cache_key(500, "fetch", "abc12345", long_arg)
    assert long_arg not in _generate_cache_key(100, "fetch", "abc12345", long_arg)


def test_generate_cache_key_datetime_argument_uses_repr() -> None:
    moment = datetime(2025, 1, 20, 12, 30, 45)

    assert _key("fetch", "abc12345", moment) == (
        f"result_cache:fetch:abc12345:{moment!r}"
    )


def test_generate_cache_key_datetime_keyword_uses_repr() -> None:
    moment = datetime(2025, 1, 20, 12, 30, 45)

    assert _key("fetch", "abc12345", timestamp=moment) == (
        f"result_cache:fetch:abc12345:timestamp={moment!r}"
    )


def test_generate_cache_key_custom_object_argument_uses_repr() -> None:
    class Item:
        def __repr__(self) -> str:
            return "Item(value='test123')"

    assert _key("fetch", "abc12345", Item()) == (
        "result_cache:fetch:abc12345:Item(value='test123')"
    )


def test_normalize_batch_args_bare_values_wrapped_in_tuples() -> None:
    assert _normalize_batch_args(["a", "b"], None) == ([("a",), ("b",)], [{}, {}])


def test_normalize_batch_args_tuples_kept() -> None:
    normalized, _ = _normalize_batch_args([("a", 1), ("b", 2)], None)

    assert normalized == [("a", 1), ("b", 2)]


def test_normalize_batch_args_given_kwargs_list_returned_unchanged() -> None:
    kwargs_list = [{"x": 1}, {"x": 2}]

    _, returned = _normalize_batch_args(["a", "b"], kwargs_list)

    assert returned is kwargs_list


def test_normalize_batch_args_kwargs_list_length_mismatch_raises_value_error() -> None:
    with pytest.raises(ValueError, match="kwargs_list length"):
        _normalize_batch_args(["a", "b"], [{}])


async def test_cached_result_second_call_returns_cached_value_without_calling(
    redis_store: InMemoryRedis,
) -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, Any]:
        calls.append(item_id)
        return {"id": item_id, "data": "test_data"}

    first = await fetch("item1")
    second = await fetch("item1")

    assert first == second == {"id": "item1", "data": "test_data"}
    assert calls == ["item1"]
    assert list(redis_store.values) == [
        _key("test", _compute_schema_hash(fetch.__wrapped__), "item1")
    ]


async def test_cached_result_caching_off_calls_function_every_time() -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        calls.append(item_id)
        return {"id": item_id}

    assert await fetch("item1") == await fetch("item1")
    assert calls == ["item1", "item1"]


async def test_cached_result_without_service_or_ttl_keeps_result_one_day(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch("item1")

    assert list(redis_store.ttls.values()) == [86400]


async def test_cached_result_ttl_given_keeps_result_that_long(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(service="mal", key_prefix="test", ttl=3600)
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch("item1")

    assert list(redis_store.ttls.values()) == [3600]


@pytest.mark.parametrize(
    ("service", "expected"),
    [("mal", 604800), ("animeschedule", 86400), ("unknown", 86400)],
)
async def test_cached_result_service_keeps_result_for_service_ttl(
    redis_store: InMemoryRedis, service: str, expected: int
) -> None:
    @cached_result(service=service, key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch("item1")

    assert list(redis_store.ttls.values()) == [expected]


async def test_cached_result_service_ttl_read_from_configured_cache() -> None:
    store = InMemoryRedis()

    @cached_result(service="mal", key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with _serve(store, ttl_mal=120):
        await fetch("item1")

    assert list(store.ttls.values()) == [120]


async def test_cached_result_configured_key_limit_hashes_long_arguments() -> None:
    store = InMemoryRedis()

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with _serve(store, max_cache_key_length=40):
        await fetch("item-" * 10)

    assert "item-" not in next(iter(store.values))


async def test_cached_result_without_key_prefix_uses_function_name(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result()
    async def my_custom_function(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await my_custom_function("item1")

    assert redis_store.reads[0].startswith("result_cache:my_custom_function:")


async def test_cached_result_none_result_not_stored(
    redis_store: InMemoryRedis,
) -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str] | None:
        calls.append(item_id)
        return None

    assert await fetch("item1") is None
    assert await fetch("item1") is None
    assert calls == ["item1", "item1"]
    assert redis_store.values == {}


async def test_cached_result_read_error_calls_function() -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        calls.append(item_id)
        return {"id": item_id, "data": "test"}

    with _serve(InMemoryRedis(failing=frozenset({"get"}))):
        result = await fetch("item1")

    assert result == {"id": "item1", "data": "test"}
    assert calls == ["item1"]


async def test_cached_result_write_error_returns_result_after_one_call() -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        calls.append(item_id)
        return {"id": item_id, "data": "test"}

    with _serve(InMemoryRedis(failing=frozenset({"setex"}))):
        result = await fetch("item1")

    assert result == {"id": "item1", "data": "test"}
    assert calls == ["item1"]


async def test_cached_result_unserializable_result_returned_after_one_call(
    redis_store: InMemoryRedis,
) -> None:
    calls: list[str] = []
    moment = datetime(2025, 1, 20)

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, Any]:
        calls.append(item_id)
        return {"id": item_id, "at": moment}

    result = await fetch("item1")

    assert result == {"id": "item1", "at": moment}
    assert calls == ["item1"]
    assert redis_store.values == {}


async def test_cached_result_corrupted_entry_calls_function_and_replaces_it(
    redis_store: InMemoryRedis,
) -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        calls.append(item_id)
        return {"id": item_id, "data": "test"}

    key = _key("test", _compute_schema_hash(fetch.__wrapped__), "item1")
    redis_store.values[key] = "{invalid json data"

    assert await fetch("item1") == {"id": "item1", "data": "test"}
    assert calls == ["item1"]
    assert json.loads(redis_store.values[key]) == {"id": "item1", "data": "test"}


async def test_cached_result_different_arguments_stored_separately(
    redis_store: InMemoryRedis,
) -> None:
    calls: list[str] = []

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        calls.append(item_id)
        return {"id": item_id, "data": f"data_{item_id}"}

    assert (await fetch("item1"))["id"] == "item1"
    assert (await fetch("item2"))["id"] == "item2"
    assert calls == ["item1", "item2"]
    assert len(redis_store.values) == 2


async def test_cached_result_key_holds_prefix_schema_hash_and_argument(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch("item1")

    namespace, prefix, schema_hash, argument = redis_store.reads[0].split(":")
    assert (namespace, prefix, argument) == ("result_cache", "test", "item1")
    assert _is_hash(schema_hash)


def test_cached_result_keeps_function_name_and_docstring() -> None:
    @cached_result(key_prefix="test")
    async def my_function(item_id: str) -> dict[str, str]:
        """My function docstring."""
        return {"id": item_id}

    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "My function docstring."


async def test_cached_result_nested_result_returned_unchanged_from_cache(
    redis_store: InMemoryRedis,
) -> None:
    payload = {
        "nested": {"key": "value", "list": [1, 2, 3]},
        "array": ["a", "b", "c"],
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
    }

    @cached_result(key_prefix="test")
    async def fetch(item_id: str) -> dict[str, Any]:
        return {"id": item_id, **payload}

    first = await fetch("item1")
    second = await fetch("item1")

    assert first == second == {"id": "item1", **payload}
    assert json.loads(next(iter(redis_store.values.values())))["nested"]["list"] == [
        1,
        2,
        3,
    ]


async def test_cached_result_changed_function_uses_different_key(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="versioned_func")
    async def fetch_v1(item_id: str) -> dict[str, str]:
        return {"id": item_id, "version": "1"}

    @cached_result(key_prefix="versioned_func")
    async def fetch_v2(item_id: str) -> dict[str, str]:
        return {"id": item_id, "version": "2", "new_field": "data"}

    await fetch_v1("item1")
    await fetch_v2("item1")

    assert redis_store.reads[0] != redis_store.reads[1]


async def test_cached_result_changed_dependency_uses_different_key(
    redis_store: InMemoryRedis,
) -> None:
    def schema_v1() -> list[str]:
        return ["//div[@class='title']"]

    def schema_v2() -> list[str]:
        return ["//h1[@class='title']"]

    @cached_result(key_prefix="page_data", dependencies=[schema_v1])
    async def fetch_v1(url: str) -> dict[str, str]:
        return {"url": url}

    @cached_result(key_prefix="page_data", dependencies=[schema_v2])
    async def fetch_v2(url: str) -> dict[str, str]:
        return {"url": url}

    await fetch_v1("https://example.com")
    await fetch_v2("https://example.com")

    assert redis_store.reads[0] != redis_store.reads[1]


async def test_cached_result_same_dependencies_use_same_key_every_call(
    redis_store: InMemoryRedis,
) -> None:
    def parse_title() -> str:
        return "parse_title"

    @cached_result(key_prefix="stable", dependencies=[parse_title])
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch("item1")
    await fetch("item1")

    assert redis_store.reads[0] == redis_store.reads[1]


async def test_cache_batch_get_returns_cached_values_and_missing_positions(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    schema_hash = _compute_schema_hash(fetch.__wrapped__)
    keys = [
        _key("test_batch", schema_hash, item) for item in ("item1", "item2", "item3")
    ]
    redis_store.values[keys[0]] = json.dumps({"id": "item1"})
    redis_store.values[keys[2]] = "{bad json"

    cached, missing = await fetch.cache_batch_get([("item1",), ("item2",), ("item3",)])

    assert cached == [{"id": "item1"}, None, None]
    assert missing == [1, 2]
    assert redis_store.reads == keys


async def test_cache_batch_get_keyword_arguments_match_single_call_keys(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str, *, page: int) -> dict[str, Any]:
        return {"id": item_id, "page": page}

    await fetch("item1", page=2)

    cached, missing = await fetch.cache_batch_get(["item1"], [{"page": 2}])

    assert cached == [{"id": "item1", "page": 2}]
    assert missing == []


async def test_cache_batch_get_caching_off_returns_every_position_missing() -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    assert await fetch.cache_batch_get([("item1",), ("item2",)]) == (
        [None, None],
        [0, 1],
    )


async def test_cache_batch_get_empty_arguments_returns_empty_lists() -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    assert await fetch.cache_batch_get([]) == ([], [])


async def test_cache_batch_get_read_error_returns_every_position_missing() -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with _serve(InMemoryRedis(failing=frozenset({"mget"}))):
        result = await fetch.cache_batch_get([("a",), ("b",)])

    assert result == ([None, None], [0, 1])


async def test_cache_batch_get_empty_reply_returns_every_position_missing() -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with _serve(EmptyMgetRedis()):
        result = await fetch.cache_batch_get([("a",), ("b",)])

    assert result == ([None, None], [0, 1])


async def test_cache_batch_set_stores_values_with_ttl_and_skips_none(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test_batch", ttl=123)
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch.cache_batch_set(
        [("item1",), ("item2",), ("item3",)],
        [{"id": "item1"}, None, {"id": "item3"}],
    )

    schema_hash = _compute_schema_hash(fetch.__wrapped__)
    assert redis_store.ttls == {
        _key("test_batch", schema_hash, "item1"): 123,
        _key("test_batch", schema_hash, "item3"): 123,
    }
    assert await fetch.cache_batch_get(["item1", "item3"]) == (
        [{"id": "item1"}, {"id": "item3"}],
        [],
    )


async def test_cache_batch_set_service_stores_values_for_service_ttl(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(service="anisearch", key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch.cache_batch_set(["item1"], [{"id": "item1"}])

    assert list(redis_store.ttls.values()) == [604800]


async def test_cache_batch_set_empty_arguments_stores_nothing(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch.cache_batch_set([], [])

    assert redis_store.values == {}


async def test_cache_batch_set_values_length_mismatch_raises_value_error() -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with pytest.raises(ValueError, match="values length"):
        await fetch.cache_batch_set([("a",), ("b",)], [{"id": "a"}])


async def test_cache_batch_set_caching_off_stores_nothing() -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch.cache_batch_set([("a",)], [{"id": "a"}])

    assert await fetch.cache_batch_get([("a",)]) == ([None], [0])


async def test_cache_batch_set_unserializable_value_skipped_others_stored(
    redis_store: InMemoryRedis,
) -> None:
    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, Any]:
        return {"id": item_id}

    await fetch.cache_batch_set(
        ["a", "b"], [{"id": "a", "at": datetime(2025, 1, 20)}, {"id": "b"}]
    )

    assert await fetch.cache_batch_get(["a", "b"]) == ([None, {"id": "b"}], [0])


async def test_cache_batch_set_pipeline_error_raises_nothing() -> None:
    store = InMemoryRedis(failing=frozenset({"pipeline"}))

    @cached_result(key_prefix="test_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    with _serve(store):
        await fetch.cache_batch_set([("a",)], [{"id": "a"}])

    assert store.values == {}


@pytest.mark.parametrize("func", [_read_selector, _read_selector_in_nested_function])
def test_compute_schema_hash_changed_module_data_returns_different_hash(
    func: Any,
) -> None:
    changed = _with_module_data(func, _SELECTORS={"title": "//h2"})

    assert _compute_schema_hash(func) != _compute_schema_hash(changed)


def test_compute_schema_hash_changed_module_data_in_dependency_returns_different_hash() -> (
    None
):
    def fetch() -> str:
        return "fetched"

    changed = _with_module_data(_read_selector, _SELECTORS={"title": "//h2"})

    assert _compute_schema_hash(fetch, [_read_selector]) != _compute_schema_hash(
        fetch, [changed]
    )


def test_compute_schema_hash_changed_number_constant_returns_same_hash() -> None:
    assert _compute_schema_hash(_wait) == _compute_schema_hash(
        _with_module_data(_wait, _DELAY=2.0)
    )


def test_compute_schema_hash_changed_helper_returns_different_hash() -> None:
    changed = _with_module_data(_clean_title, _title_cleaner=_lower_title)

    assert _compute_schema_hash(_clean_title) != _compute_schema_hash(changed)


def test_compute_schema_hash_changed_helper_of_helper_returns_different_hash() -> None:
    changed = _with_module_data(
        _extract_title,
        _clean_title=_with_module_data(_clean_title, _title_cleaner=_lower_title),
    )

    assert _compute_schema_hash(_extract_title) != _compute_schema_hash(changed)


def test_compute_schema_hash_changed_helper_of_decorated_helper_returns_different_hash() -> (
    None
):
    changed = _with_module_data(
        _clean_with_decorated_helper,
        _decorated_cleaner=_passed_through(
            _with_module_data(_clean_title, _title_cleaner=_lower_title)
        ),
    )

    assert _compute_schema_hash(_clean_with_decorated_helper) != _compute_schema_hash(
        changed
    )


def test_compute_schema_hash_changed_async_helper_returns_same_hash() -> None:
    changed = _with_module_data(_read_fetched_page, _page_fetcher=_fetch_other_page)

    assert _compute_schema_hash(_read_fetched_page) == _compute_schema_hash(changed)


def test_compute_schema_hash_changed_function_from_other_package_returns_same_hash() -> (
    None
):
    changed = _with_module_data(_call_library, _library_function=json.loads)

    assert _compute_schema_hash(_call_library) == _compute_schema_hash(changed)


def test_compute_schema_hash_helpers_calling_each_other_returns_hash() -> None:
    assert _is_hash(_compute_schema_hash(_trim_front))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (frozenset({"b", "a"}), "['a', 'b']"),
        ({"z": {"b", "a"}}, "{'z': ['a', 'b']}"),
        (("x", {"b", "a"}), "('x', ['a', 'b'])"),
        (re.compile(r"\d+"), "re.compile('\\\\d+')"),
    ],
)
def test_stable_repr_sets_are_sorted(value: object, expected: str) -> None:
    assert _stable_repr(value) == expected
