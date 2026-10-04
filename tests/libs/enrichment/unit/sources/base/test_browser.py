import asyncio
import functools
import http.server
import itertools
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import call, create_autospec, patch

import pytest
import zendriver
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment.sources.base import browser as browser_module
from enrichment.sources.base.browser import (
    BACKGROUND_DOWNLOAD_SWITCHES,
    OWNER_FILE_NAME,
    UNOWNED_PROFILE_GRACE_SECONDS,
    BrowserPoolSizeError,
    browser_session,
    close_browser,
    configure_browser_pool,
    reap_orphans,
)
from zendriver import cdp
from zendriver.core.config import find_executable

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
        browser.main_tab = create_autospec(zendriver.Tab, instance=True)
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


def _start_stand_in(
    processes: list, profile: Path, *, ignore_terminate: bool = False
) -> subprocess.Popen[bytes]:
    title = f"chrome --type=renderer --user-data-dir={profile} --lang=en"
    script = 'exec -a "$0" sleep 300'
    if ignore_terminate:
        script = f"trap '' TERM; {script}"
    process = subprocess.Popen(  # noqa: S603  (fixed test arguments)
        ["/bin/bash", "-c", script, title]
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


async def test_browser_session_starts_chrome_with_headless_setting_and_background_switches(
    started_browsers: list[zendriver.Browser],
) -> None:
    async with browser_session(headless=False):
        pass

    zendriver.start.assert_awaited_once_with(
        headless=False, browser_args=list(BACKGROUND_DOWNLOAD_SWITCHES)
    )


async def test_browser_session_allowed_site_resolves_only_that_domain_and_subdomains(
    started_browsers: list[zendriver.Browser],
) -> None:
    async with browser_session(headless=True, allowed_site="myanimelist.net"):
        pass

    switches = zendriver.start.call_args.kwargs["browser_args"]
    assert switches == [
        *BACKGROUND_DOWNLOAD_SWITCHES,
        "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE myanimelist.net, "
        "EXCLUDE *.myanimelist.net",
    ]


async def test_browser_session_blocking_off_ignores_allowed_site(
    started_browsers: list[zendriver.Browser],
) -> None:
    async with browser_session(
        headless=True, allowed_site="myanimelist.net", block_unused_resources=False
    ):
        pass

    assert zendriver.start.call_args.kwargs["browser_args"] == list(
        BACKGROUND_DOWNLOAD_SWITCHES
    )


async def test_browser_session_blocking_off_one_session_leaves_others_blocking(
    started_browsers: list[zendriver.Browser],
) -> None:
    with patch.object(
        browser_module, "install_static_resource_blocking", autospec=True
    ) as install_blocking:
        async with browser_session(headless=True, block_unused_resources=False):
            pass
        async with browser_session(headless=True):
            pass

    install_blocking.assert_awaited_once_with(started_browsers[1].main_tab)


async def test_browser_session_restart_keeps_blocking_off(
    started_browsers: list[zendriver.Browser],
) -> None:
    with patch.object(
        browser_module, "install_static_resource_blocking", autospec=True
    ) as install_blocking:
        async with browser_session(
            headless=True, block_unused_resources=False
        ) as session:
            await session.restart()

    assert len(started_browsers) == 2
    install_blocking.assert_not_awaited()


async def test_browser_session_restart_keeps_allowed_site(
    started_browsers: list[zendriver.Browser],
) -> None:
    async with browser_session(
        headless=True, allowed_site="myanimelist.net"
    ) as session:
        await session.restart()

    first_switches, restart_switches = (
        started.kwargs["browser_args"] for started in zendriver.start.call_args_list
    )
    assert restart_switches == first_switches


async def test_browser_session_blocking_setup_fails_closes_browser_and_releases_slot(
    started_browsers: list[zendriver.Browser],
) -> None:
    configure_browser_pool(1)

    with (
        patch.object(
            browser_module,
            "install_static_resource_blocking",
            autospec=True,
            side_effect=RuntimeError("connection closed"),
        ),
        pytest.raises(RuntimeError, match="connection closed"),
    ):
        async with browser_session(headless=True):
            pass
    async with asyncio.timeout(1):
        async with browser_session(headless=True):
            pass

    started_browsers[0].stop.assert_awaited_once_with()
    assert len(started_browsers) == 2


async def test_browser_session_clearance_site_loads_clearance_on_start_and_restart(
    started_browsers: list[zendriver.Browser],
) -> None:
    with patch.object(browser_module, "load_clearance", autospec=True) as load:
        async with browser_session(
            headless=False, clearance_site="anidb.net"
        ) as session:
            await session.restart()

    assert load.await_args_list == [
        call(started_browsers[0], "anidb.net"),
        call(started_browsers[1], "anidb.net"),
    ]


async def test_browser_session_without_clearance_site_loads_no_clearance(
    started_browsers: list[zendriver.Browser],
) -> None:
    with patch.object(browser_module, "load_clearance", autospec=True) as load:
        async with browser_session(headless=True):
            pass

    load.assert_not_awaited()


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


async def test_close_browser_child_ignores_terminate_keeps_event_loop_running(
    tmp_path: Path, stand_in_processes: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(browser_module, "_EXIT_DEADLINE_SECONDS", 0.3)
    profile = _profile(tmp_path, "crashed")
    child = _start_stand_in(stand_in_processes, profile, ignore_terminate=True)
    exited = subprocess.Popen([sys.executable, "-c", "pass"])
    exited.wait()
    browser = create_autospec(zendriver.Browser, instance=True)
    browser._process = exited
    browser.config = create_autospec(zendriver.Config, instance=True)
    browser.config.user_data_dir = str(profile)
    tick_times: list[float] = []

    async def record_ticks() -> None:
        while True:
            tick_times.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticking = asyncio.create_task(record_ticks())
    await close_browser(browser)
    tick_times.append(time.monotonic())
    ticking.cancel()

    assert child.wait(timeout=5) == -9
    assert (
        max(later - earlier for earlier, later in itertools.pairwise(tick_times)) < 0.1
    )


async def test_browser_session_start_time_unreadable_writes_no_owner_file(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, "unreadable")
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.main_tab = create_autospec(zendriver.Tab, instance=True)
    browser.config = create_autospec(zendriver.Config, instance=True)
    browser.config.user_data_dir = str(profile)

    with (
        patch("zendriver.start", autospec=True, return_value=browser),
        patch.object(
            browser_module, "_process_start_ticks", autospec=True, return_value=None
        ),
    ):
        async with browser_session(headless=True):
            assert not (profile / OWNER_FILE_NAME).exists()


def test_reap_orphans_profile_removed_during_scan_returns_without_error(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path, "removed")
    real_is_dir = Path.is_dir

    def removed_after_check(path: Path) -> bool:
        found = real_is_dir(path)
        if path == profile:
            shutil.rmtree(path)
        return found

    with patch.object(Path, "is_dir", autospec=True, side_effect=removed_after_check):
        assert reap_orphans(tmp_path) == 0


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
            re.findall(r"browser_session\(headless=(True|False)\b", path.read_text())
        )

    assert sessions.get("anisearch", []) == []
    assert sessions["anidb"] == ["False"]
    assert sessions["mal"] == ["True"] * 7
    assert sessions["anime_planet"] == ["True"] * 4


def test_crawler_sources_use_allowed_site_on_mal_only() -> None:
    sites_with_rule = {
        path.relative_to(SOURCES_DIR).parts[0]
        for path in _crawler_sources()
        if "allowed_site=" in path.read_text()
    }
    mal_sessions = sum(
        path.read_text().count("allowed_site=MAL_DOMAIN")
        for path in (SOURCES_DIR / "mal").glob("*.py")
    )

    assert sites_with_rule == {"mal"}
    assert mal_sessions == 7


RUNS = 10


def _browser_installed() -> bool:
    try:
        find_executable()
    except FileNotFoundError:
        return False
    return True


requires_browser = pytest.mark.skipif(
    not _browser_installed(), reason="no Chrome or Chromium installed"
)


@pytest.fixture
def browser_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.fixture
def local_page(tmp_path: Path) -> str:
    page = tmp_path / "page.html"
    page.write_text("<html><body><p id=x>ready</p></body></html>")
    return page.as_uri()


def _process_tree(root_pid: int) -> set[int]:
    tree = {root_pid}
    grew = True
    while grew:
        grew = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) in tree:
                continue
            try:
                stat = (entry / "stat").read_text()
            except OSError:
                continue
            if int(stat.rsplit(")", 1)[1].split()[1]) in tree:
                tree.add(int(entry.name))
                grew = True
    return tree


def _still_present(pids: set[int]) -> list[str]:
    present = []
    for pid in pids:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        present.append(f"{pid}:{state}")
    return present


@pytest.mark.integration
@requires_browser
@pytest.mark.usefixtures("browser_home")
async def test_browser_session_close_takes_under_half_of_zendriver_stop(
    local_page: str,
) -> None:
    session_closes, zendriver_stops = [], []
    for _ in range(RUNS):
        async with browser_session(headless=True) as session:
            await session.browser.get(local_page)
            started = time.perf_counter()
        session_closes.append(time.perf_counter() - started)
        browser = await zendriver.start(headless=True)
        await browser.get(local_page)
        started = time.perf_counter()
        await browser.stop()
        zendriver_stops.append(time.perf_counter() - started)

    assert statistics.median(session_closes) < statistics.median(zendriver_stops) / 2, (
        session_closes,
        zendriver_stops,
    )


@pytest.mark.integration
@requires_browser
@pytest.mark.usefixtures("browser_home")
async def test_browser_session_close_leaves_no_process_or_profile(
    local_page: str,
) -> None:
    async with browser_session(headless=True) as session:
        await session.browser.get(local_page)
        tree = _process_tree(session.browser._process_pid)
        profile = Path(session.browser.config.user_data_dir)

    assert len(tree) > 1
    assert _still_present(tree) == []
    assert not profile.exists()


@pytest.mark.integration
@requires_browser
@pytest.mark.usefixtures("browser_home")
async def test_browser_session_chrome_killed_closes_without_error(
    local_page: str,
) -> None:
    async with browser_session(headless=True) as session:
        await session.browser.get(local_page)
        tree = _process_tree(session.browser._process_pid)
        session.browser._process.kill()
        session.browser._process.wait()
        started = time.perf_counter()

    assert time.perf_counter() - started < 3
    assert [entry for entry in _still_present(tree) if not entry.endswith(":Z")] == []
    assert not Path(session.browser.config.user_data_dir).exists()


@pytest.mark.integration
@requires_browser
@pytest.mark.usefixtures("browser_home")
async def test_browser_session_inside_enrichment_pipeline_leaves_no_browser_after_exit(
    local_page: str,
) -> None:
    async with EnrichmentPipeline():
        async with browser_session(headless=True) as session:
            await session.browser.get(local_page)
            tree = _process_tree(session.browser._process_pid)

    assert _still_present(tree) == []


SITE_PAGE = """<html><head>
<link rel=stylesheet href=style.css>
<style>@font-face{font-family:F;src:url(font.woff2)} body{font-family:F}</style>
<script src=app.js></script>
</head><body><p id=x>ready</p><img src=picture.png><video src=clip.mp4 autoplay muted></video>
<script>
fetch('data.json').then(response => response.json()).then(data => {document.body.dataset.fetched = data.ok});
const request = new XMLHttpRequest(); request.open('GET', 'data.json');
request.onload = () => {document.body.dataset.xhr = request.status}; request.send();
</script></body></html>"""


class QuietRequestHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def local_site(tmp_path: Path) -> Iterator[str]:
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(SITE_PAGE)
    (site / "app.js").write_text("document.documentElement.dataset.script = 'ran';")
    (site / "data.json").write_text('{"ok": "yes"}')
    for name in ("style.css", "font.woff2", "picture.png", "clip.mp4", "favicon.ico"):
        (site / name).write_bytes(b"x" * 100)
    handler = functools.partial(QuietRequestHandler, directory=str(site))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/index.html"
    server.shutdown()
    server.server_close()
    thread.join()


async def _load_site(
    url: str, *, block_unused_resources: bool = True
) -> tuple[dict[str, str], dict[str, object]]:
    kinds: dict[str, str] = {}
    outcomes: dict[str, str] = {}

    async def requested(event: cdp.network.RequestWillBeSent) -> None:
        kinds[event.request_id] = event.type_.value if event.type_ else "Other"

    async def loaded(event: cdp.network.LoadingFinished) -> None:
        outcomes.setdefault(event.request_id, "loaded")

    async def failed(event: cdp.network.LoadingFailed) -> None:
        outcomes[event.request_id] = event.error_text

    async with browser_session(
        headless=True, block_unused_resources=block_unused_resources
    ) as session:
        tab = session.browser.main_tab
        tab.add_handler(cdp.network.RequestWillBeSent, requested)
        tab.add_handler(cdp.network.LoadingFinished, loaded)
        tab.add_handler(cdp.network.LoadingFailed, failed)
        page = await session.browser.get(url)
        async with asyncio.timeout(10):
            while not await page.evaluate("document.body?.dataset.xhr || ''"):
                await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        page_state = {
            "text": await page.evaluate("document.getElementById('x').textContent"),
            "script": await page.evaluate("document.documentElement.dataset.script"),
            "fetch": await page.evaluate("document.body.dataset.fetched"),
            "xhr": await page.evaluate("document.body.dataset.xhr"),
        }
    by_kind = {kinds[request]: outcome for request, outcome in outcomes.items()}
    return by_kind, page_state


@pytest.mark.integration
@requires_browser
@pytest.mark.usefixtures("browser_home")
async def test_browser_session_default_fails_static_requests_and_loads_documents_scripts_and_data(
    local_site: str,
) -> None:
    outcomes, page_state = await _load_site(local_site)

    assert {
        kind: outcomes[kind] for kind in ("Image", "Font", "Media", "Stylesheet")
    } == {
        kind: "net::ERR_BLOCKED_BY_CLIENT.Inspector"
        for kind in ("Image", "Font", "Media", "Stylesheet")
    }
    assert {
        kind: outcomes[kind] for kind in ("Document", "Script", "XHR", "Fetch")
    } == {kind: "loaded" for kind in ("Document", "Script", "XHR", "Fetch")}
    assert page_state == {
        "text": "ready",
        "script": "ran",
        "fetch": "yes",
        "xhr": "200",
    }


@pytest.mark.integration
@requires_browser
@pytest.mark.usefixtures("browser_home")
async def test_browser_session_blocking_off_loads_every_request(
    local_site: str,
) -> None:
    outcomes, page_state = await _load_site(local_site, block_unused_resources=False)

    assert set(outcomes.values()) == {"loaded"}
    assert {"Image", "Font", "Stylesheet", "Document", "Script", "XHR", "Fetch"} <= (
        outcomes.keys()
    )
    assert page_state == {
        "text": "ready",
        "script": "ran",
        "fetch": "yes",
        "xhr": "200",
    }
