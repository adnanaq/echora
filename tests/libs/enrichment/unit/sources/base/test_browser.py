import asyncio
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import create_autospec, patch

import pytest
import zendriver
from enrichment.sources.base import browser as browser_module
from enrichment.sources.base.browser import (
    OWNER_FILE_NAME,
    UNOWNED_PROFILE_GRACE_SECONDS,
    BrowserPoolSizeError,
    browser_session,
    close_browser,
    configure_browser_pool,
    reap_orphans,
)

SOURCES_DIR = Path(browser_module.__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def unlimited_pool() -> Iterator[None]:
    browser_module._max_browsers = None
    browser_module._slots_by_loop.clear()
    yield
    browser_module._max_browsers = None
    browser_module._slots_by_loop.clear()


@pytest.fixture
def started_browsers() -> Iterator[list[zendriver.Browser]]:
    browsers: list[zendriver.Browser] = []

    async def start_fake_browser(**_: object) -> zendriver.Browser:
        browser = create_autospec(zendriver.Browser, instance=True)
        browsers.append(browser)
        return browser

    with patch("zendriver.start", autospec=True, side_effect=start_fake_browser):
        yield browsers


@pytest.fixture
def stand_in_processes() -> Iterator[list[subprocess.Popen[bytes]]]:
    processes: list[subprocess.Popen[bytes]] = []
    yield processes
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.wait()


def _start_stand_in(processes: list, profile: Path) -> subprocess.Popen[bytes]:
    title = f"chrome --type=renderer --user-data-dir={profile} --lang=en"
    process = subprocess.Popen(  # noqa: S603  (fixed test arguments)
        ["/bin/bash", "-c", 'exec -a "$0" sleep 300', title]
    )
    processes.append(process)
    deadline = time.monotonic() + 5
    while (
        f"--user-data-dir={profile}".encode()
        not in Path(f"/proc/{process.pid}/cmdline").read_bytes()
    ):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    return process


def _dead_owner() -> str:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    return f"{finished.pid} 1"


def _profile(
    temp_dir: Path, name: str, owner: str | None = None, age: float = 0
) -> Path:
    profile = temp_dir / f"uc_{name}"
    profile.mkdir()
    if owner is not None:
        (profile / OWNER_FILE_NAME).write_text(owner)
    if age:
        old = time.time() - age
        os.utime(profile, (old, old))
    return profile


def test_configure_browser_pool_below_one_raises_browser_pool_size_error() -> None:
    with pytest.raises(BrowserPoolSizeError, match="at least 1"):
        configure_browser_pool(0)


async def test_browser_session_cap_never_exceeds_limit_and_runs_every_session(
    started_browsers: list[zendriver.Browser],
) -> None:
    configure_browser_pool(2)
    open_now = 0
    most_open = 0

    async def use_browser() -> None:
        nonlocal open_now, most_open
        async with browser_session(headless=True):
            open_now += 1
            most_open = max(most_open, open_now)
            await asyncio.sleep(0.01)
            open_now -= 1

    await asyncio.gather(*(use_browser() for _ in range(7)))

    assert most_open == 2
    assert len(started_browsers) == 7


async def test_browser_session_waiting_for_slot_has_not_started_browser(
    started_browsers: list[zendriver.Browser],
) -> None:
    configure_browser_pool(1)
    release = asyncio.Event()

    async def hold_slot() -> None:
        async with browser_session(headless=True):
            await release.wait()

    holder = asyncio.create_task(hold_slot())
    await asyncio.sleep(0.01)
    waiter = asyncio.create_task(hold_slot())
    await asyncio.sleep(0.01)

    assert len(started_browsers) == 1
    release.set()
    await asyncio.gather(holder, waiter)
    assert len(started_browsers) == 2


async def test_browser_session_block_raises_releases_slot_and_closes_browser(
    started_browsers: list[zendriver.Browser],
) -> None:
    configure_browser_pool(1)

    with pytest.raises(RuntimeError, match="crawl failed"):
        async with browser_session(headless=True):
            raise RuntimeError("crawl failed")
    async with asyncio.timeout(1):
        async with browser_session(headless=True):
            pass

    started_browsers[0].stop.assert_awaited_once_with()


async def test_browser_session_passes_headless_to_zendriver_start(
    started_browsers: list[zendriver.Browser],
) -> None:
    async with browser_session(headless=False):
        pass

    zendriver.start.assert_awaited_once_with(headless=False)


async def test_browser_session_restart_closes_old_browser_and_keeps_slot(
    started_browsers: list[zendriver.Browser],
) -> None:
    configure_browser_pool(1)

    async with browser_session(headless=False) as session:
        first = session.browser
        second = await session.restart()

    assert second is session.browser
    assert second is not first
    first.stop.assert_awaited_once_with()
    second.stop.assert_awaited_once_with()


async def test_close_browser_stop_raises_returns_without_error() -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.stop.side_effect = RuntimeError("connection gone")

    await close_browser(browser)

    browser.stop.assert_awaited_once_with()


def test_reap_orphans_dead_owner_stops_its_processes_and_removes_profile(
    tmp_path: Path, stand_in_processes: list
) -> None:
    profile = _profile(tmp_path, "abandoned", owner=_dead_owner())
    process = _start_stand_in(stand_in_processes, profile)

    assert reap_orphans(tmp_path) == 1

    assert process.wait(timeout=5) is not None
    assert not profile.exists()


def test_reap_orphans_live_owner_beside_abandoned_keeps_live_profile(
    tmp_path: Path, stand_in_processes: list
) -> None:
    live_owner = f"{os.getpid()} {browser_module._process_start_ticks(os.getpid())}"
    live = _profile(tmp_path, "live", owner=live_owner)
    live_process = _start_stand_in(stand_in_processes, live)
    abandoned = _profile(tmp_path, "abandoned", owner=_dead_owner())

    assert reap_orphans(tmp_path) == 1

    assert live.exists()
    assert live_process.poll() is None
    assert not abandoned.exists()


def test_reap_orphans_unowned_profile_removed_only_when_unused_and_old(
    tmp_path: Path, stand_in_processes: list
) -> None:
    old_unused = _profile(
        tmp_path, "old_unused", age=UNOWNED_PROFILE_GRACE_SECONDS + 60
    )
    new_unused = _profile(tmp_path, "new_unused")
    old_used = _profile(tmp_path, "old_used", age=UNOWNED_PROFILE_GRACE_SECONDS + 60)
    used_process = _start_stand_in(stand_in_processes, old_used)

    assert reap_orphans(tmp_path) == 1

    assert not old_unused.exists()
    assert new_unused.exists()
    assert old_used.exists()
    assert used_process.poll() is None


def test_reap_orphans_process_without_zendriver_profile_is_never_touched(
    tmp_path: Path, stand_in_processes: list
) -> None:
    other_profile = tmp_path / "chrome_profile"
    other_profile.mkdir()
    process = _start_stand_in(stand_in_processes, other_profile)

    assert reap_orphans(tmp_path) == 0

    assert process.poll() is None
    assert other_profile.exists()


def _crawler_sources() -> list[Path]:
    return [
        path
        for path in SOURCES_DIR.rglob("*.py")
        if path.resolve() != Path(browser_module.__file__).resolve()
    ]


def test_crawler_sources_start_browsers_only_through_browser_session() -> None:
    offenders = [
        f"{path.relative_to(SOURCES_DIR)}:{number}"
        for path in _crawler_sources()
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"\b(zd|zendriver)\.start\(|\.stop\(\)", line)
    ]

    assert offenders == []


def test_crawler_sources_keep_each_site_headless_setting() -> None:
    sessions: dict[str, list[str]] = {}
    for path in _crawler_sources():
        site = path.relative_to(SOURCES_DIR).parts[0]
        sessions.setdefault(site, []).extend(
            re.findall(r"browser_session\(headless=(True|False)\)", path.read_text())
        )

    assert sessions["anisearch"] == ["False"] * 8
    assert sessions["anidb"] == ["False"]
    assert sessions["mal"] == ["True"] * 7
    assert sessions["anime_planet"] == ["True"] * 4
