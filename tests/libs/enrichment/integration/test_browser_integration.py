import statistics
import time
from pathlib import Path

import pytest
import zendriver
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment.sources.base.browser import browser_session
from zendriver.core.config import find_executable

pytestmark = pytest.mark.integration

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


@pytest.fixture(autouse=True)
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


@requires_browser
async def test_browser_session_close_takes_under_third_of_zendriver_stop(
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

    assert statistics.median(session_closes) < statistics.median(zendriver_stops) / 3, (
        session_closes,
        zendriver_stops,
    )


@requires_browser
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


@requires_browser
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


@requires_browser
async def test_enrichment_pipeline_exit_leaves_no_browser_running(
    local_page: str,
) -> None:
    async with EnrichmentPipeline():
        async with browser_session(headless=True) as session:
            await session.browser.get(local_page)
            tree = _process_tree(session.browser._process_pid)

    assert _still_present(tree) == []
