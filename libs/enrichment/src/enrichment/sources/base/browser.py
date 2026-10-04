"""Start and close the crawlers' zendriver browsers in one place.

Every crawler opens its browser through `browser_session()`, which:

* closes it in tens of milliseconds. zendriver's `Browser.stop()` sends
  `terminate()` and then checks the process every 0.25 s, so even a browser that
  exits at once costs about half a second. Here Chrome is asked to close and its
  process is polled every few milliseconds first, so `stop()` finds it gone.
* holds a slot from a process-wide limit while the browser is open, so concurrent
  enrichments cannot start an unbounded number of Chromes.
* writes the owning process into the browser's temporary profile, so
  `reap_orphans()` can stop Chromes and delete profiles left behind by a run that
  was killed before its `finally` blocks ran.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import zendriver
from zendriver import cdp

logger = logging.getLogger(__name__)

PROFILE_PREFIX = "uc_"
OWNER_FILE_NAME = "echora_owner"
UNOWNED_PROFILE_GRACE_SECONDS = 600

_CLOSE_COMMAND_TIMEOUT_SECONDS = 1.0
_EXIT_POLL_SECONDS = 0.005
_EXIT_DEADLINE_SECONDS = 2.0
_PROC = Path("/proc")
# Chrome rewrites its process title, so /proc/<pid>/cmdline holds the switches
# as one space-separated string rather than one argument each.
_PROFILE_ARGUMENT = re.compile(rb"--user-data-dir=(\S+)")

_max_browsers: int | None = None
_slots_by_loop: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, asyncio.Semaphore
] = weakref.WeakKeyDictionary()


class BrowserPoolSizeError(ValueError):
    """Raised when the browser limit is below one."""

    def __init__(self, max_browsers: int) -> None:
        super().__init__(f"max_browsers must be at least 1, got {max_browsers}")


@dataclass
class BrowserSession:
    """An open browser and the settings it was started with."""

    headless: bool
    browser: zendriver.Browser

    async def restart(self) -> zendriver.Browser:
        """Close the current browser and start a new one in the same slot.

        Used after Chrome crashes mid-batch; the new browser keeps the slot so a
        restart never lets another session start in between.
        """
        await close_browser(self.browser)
        self.browser = await _start_browser(self.headless)
        return self.browser


def configure_browser_pool(max_browsers: int) -> None:
    """Limit how many browsers can be open at once in this process.

    Call once at startup. Calling again with the same limit does nothing;
    a different limit applies to sessions opened after the call.

    Raises:
        BrowserPoolSizeError: If ``max_browsers`` is below 1.
    """
    global _max_browsers
    if max_browsers < 1:
        raise BrowserPoolSizeError(max_browsers)
    if max_browsers != _max_browsers:
        _max_browsers = max_browsers
        _slots_by_loop.clear()


@asynccontextmanager
async def browser_session(*, headless: bool) -> AsyncIterator[BrowserSession]:
    """Open a browser for the length of the block and always close it.

    Waits for a free slot before Chrome is started, so a waiting session holds
    no browser. The slot is released however the block ends.

    Args:
        headless: Start Chrome without a window. AniSearch and AniDB pages need
            a window, so their crawlers pass ``False``.

    Yields:
        The session; its ``browser`` is replaced by ``restart()``.
    """
    slots = _browser_slots()
    async with slots if slots is not None else nullcontext():
        session = BrowserSession(headless, await _start_browser(headless))
        try:
            yield session
        finally:
            await close_browser(session.browser)


async def close_browser(browser: Any) -> None:
    """Close a browser quickly, whether it is running, exited or crashed.

    Never raises: a failure to close is logged and the profile cleanup in
    zendriver's ``stop()`` still runs.
    """
    process = getattr(browser, "_process", None)
    if isinstance(process, subprocess.Popen):
        if process.poll() is None:
            children = _descendant_pids(process.pid)
            await _request_close(browser, process)
            await _wait_for_exit(process)
        else:
            children = set(_profile_users().get(_profile_path(browser), []))
        await _wait_for_children(children)
        await _close_connection(browser)
    try:
        await browser.stop()
    except Exception as error:
        logger.debug(f"browser stop failed: {error}")


def reap_orphans(temp_dir: Path | None = None) -> int:
    """Stop browsers and delete profiles left behind by dead crawler processes.

    A profile written by `browser_session()` names its owning process; when that
    process is gone, every Chrome still using the profile is stopped and the
    profile deleted. A profile without an owner is deleted only when no process
    uses it and it is older than ``UNOWNED_PROFILE_GRACE_SECONDS``, so a profile
    another process has just created is left alone. Chromes not using a
    zendriver profile are never touched.

    Linux only; elsewhere it does nothing.

    Returns:
        The number of profiles deleted.
    """
    if not _PROC.is_dir():
        return 0
    temp_dir = temp_dir or Path(tempfile.gettempdir())
    users = _profile_users()
    reaped = 0
    for profile in temp_dir.glob(f"{PROFILE_PREFIX}*"):
        if profile.is_dir() and _is_abandoned(profile, users.get(str(profile), [])):
            _stop_processes(users.get(str(profile), []))
            shutil.rmtree(profile, ignore_errors=True)
            reaped += 1
    if reaped:
        logger.info(f"removed {reaped} browser profile(s) left by dead crawler runs")
    return reaped


def _browser_slots() -> asyncio.Semaphore | None:
    if _max_browsers is None:
        return None
    loop = asyncio.get_running_loop()
    slots = _slots_by_loop.get(loop)
    if slots is None:
        slots = _slots_by_loop[loop] = asyncio.Semaphore(_max_browsers)
    return slots


async def _start_browser(headless: bool) -> zendriver.Browser:
    browser = await zendriver.start(headless=headless)
    _write_profile_owner(browser)
    return browser


async def _request_close(browser: Any, process: subprocess.Popen[bytes]) -> None:
    try:
        await asyncio.wait_for(
            browser.connection.send(cdp.browser.close()),
            _CLOSE_COMMAND_TIMEOUT_SECONDS,
        )
    except Exception:
        process.terminate()


async def _wait_for_exit(process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + _EXIT_DEADLINE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        await asyncio.sleep(_EXIT_POLL_SECONDS)
    if process.poll() is None:
        logger.warning(f"browser {process.pid} did not exit in time; killing it")
        process.kill()
        await asyncio.to_thread(process.wait)


async def _wait_for_children(pids: set[int]) -> None:
    deadline = time.monotonic() + _EXIT_DEADLINE_SECONDS
    while time.monotonic() < deadline and any(_is_running(pid) for pid in pids):
        await asyncio.sleep(_EXIT_POLL_SECONDS)
    still_running = [pid for pid in pids if _is_running(pid)]
    if still_running:
        logger.warning(
            f"browser processes {still_running} outlived Chrome; killing them"
        )
        _stop_processes(still_running)


def _descendant_pids(root_pid: int) -> set[int]:
    if not _PROC.is_dir():
        return set()
    children_of: dict[int, list[int]] = {}
    for entry in _PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        parent = int(stat.rsplit(")", 1)[1].split()[1])
        children_of.setdefault(parent, []).append(int(entry.name))
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        for child in children_of.get(pending.pop(), []):
            descendants.add(child)
            pending.append(child)
    return descendants


async def _close_connection(browser: Any) -> None:
    connection = getattr(browser, "connection", None)
    if connection is None or connection.closed:
        return
    try:
        await connection.aclose()
    except Exception as error:
        logger.debug(f"closing browser connection failed: {error}")


def _profile_path(browser: Any) -> str:
    profile = getattr(getattr(browser, "config", None), "user_data_dir", None)
    return os.path.normpath(profile) if isinstance(profile, str) else ""


def _write_profile_owner(browser: Any) -> None:
    profile = _profile_path(browser)
    if not Path(profile).name.startswith(PROFILE_PREFIX):
        return
    pid = os.getpid()
    try:
        (Path(profile) / OWNER_FILE_NAME).write_text(
            f"{pid} {_process_start_ticks(pid)}"
        )
    except OSError as error:
        logger.debug(f"could not mark browser profile {profile}: {error}")


def _is_abandoned(profile: Path, pids: list[int]) -> bool:
    try:
        owner_pid, owner_start = (profile / OWNER_FILE_NAME).read_text().split()
    except OSError, ValueError:
        age = time.time() - profile.stat().st_mtime
        return not pids and age > UNOWNED_PROFILE_GRACE_SECONDS
    return _process_start_ticks(int(owner_pid)) != _parse_ticks(owner_start)


def _parse_ticks(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def _process_start_ticks(pid: int) -> int | None:
    try:
        stat = (_PROC / str(pid) / "stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[19])


def _profile_users() -> dict[str, list[int]]:
    users: dict[str, list[int]] = {}
    if not _PROC.is_dir():
        return users
    for entry in _PROC.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command_line = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        match = _PROFILE_ARGUMENT.search(command_line)
        if match:
            profile = os.path.normpath(match.group(1).decode(errors="replace"))
            users.setdefault(profile, []).append(int(entry.name))
    return users


def _stop_processes(pids: list[int]) -> None:
    for sent_signal in (signal.SIGTERM, signal.SIGKILL):
        for pid in pids:
            try:
                os.kill(pid, sent_signal)
            except ProcessLookupError, PermissionError:
                continue
        deadline = time.monotonic() + _EXIT_DEADLINE_SECONDS
        while time.monotonic() < deadline and any(_is_running(pid) for pid in pids):
            time.sleep(0.05)
        if not any(_is_running(pid) for pid in pids):
            return


def _is_running(pid: int) -> bool:
    try:
        state = (_PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[0]
    except OSError:
        return False
    return state != "Z"
