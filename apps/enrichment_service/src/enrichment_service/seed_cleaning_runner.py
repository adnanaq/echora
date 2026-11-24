"""Run seed cleaning and delegate open-case Jev judging to agent_service."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path

import grpc
from agent_proto.v1 import agent_service_pb2, agent_service_pb2_grpc
from common.config import EnrichmentConfig
from http_cache import HTTPCacheManager
from shared_proto.v1 import review_cases_pb2
from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    ReviewCase,
    ReviewCaseKind,
    is_review_file_locked,
    replace_file_lines,
)
from work_matching.review_case_messages import convert_case_to_message

logger = logging.getLogger(__name__)

_EXIT_CODE_FILE_NAME = "review_run.exit_code"
_REVIEW_FILE_NAME = "review.jsonl"
_JUDGE_TIMEOUT_SECONDS = 600.0


class SeedCleaningRunner:
    """Manage background seed-cleaning runs and track status in each data directory."""

    def __init__(
        self,
        *,
        enrichment: EnrichmentConfig,
        cache: HTTPCacheManager,
        agent_service_address: str,
    ) -> None:
        """Initialize shared enrichment, cache, and agent service settings.

        Args:
            enrichment: Enrichment configuration used for reading page synopses.
            cache: Shared HTTP cache manager for synopsis lookups.
            agent_service_address: ``host:port`` address of ``agent_service``.
        """
        self._enrichment = enrichment
        self._cache = cache
        self._agent_service_address = agent_service_address
        self._run_lock = threading.RLock()
        self._active_tasks: dict[Path, asyncio.Task[None]] = {}

    def start_run(
        self,
        *,
        seed_file: Path,
        data_dir: Path,
        no_fetch: bool = True,
        refetch_stored: bool = False,
    ) -> review_cases_pb2.RunState:
        """Start a background seed-cleaning run when none is active for ``data_dir``.

        Args:
            seed_file: Path to the seed release JSON file.
            data_dir: Directory holding ``review.jsonl`` and ``provider_pages.jsonl``.
            no_fetch: Skip network page fetches before cleaning.
            refetch_stored: Re-fetch already stored pages when fetching is enabled.

        Returns:
            Current ``RunState`` marked as running.
        """
        with self._run_lock:
            if self._is_running(data_dir):
                return review_cases_pb2.RunState(is_running=True)
            (data_dir / _EXIT_CODE_FILE_NAME).unlink(missing_ok=True)
            task = asyncio.create_task(
                self._run_and_record(
                    seed_file,
                    data_dir,
                    no_fetch=no_fetch,
                    refetch_stored=refetch_stored,
                )
            )
            self._active_tasks[data_dir] = task
        return review_cases_pb2.RunState(is_running=True)

    def read_run_state(self, data_dir: Path) -> review_cases_pb2.RunState:
        """Report whether a run is active in ``data_dir`` and the last exit code.

        Args:
            data_dir: Directory holding ``review.jsonl`` and ``review_run.exit_code``.

        Returns:
            Current ``RunState`` protobuf message.
        """
        with self._run_lock:
            if self._is_running(data_dir):
                return review_cases_pb2.RunState(is_running=True)
            exit_code = _read_exit_code(data_dir / _EXIT_CODE_FILE_NAME)
            if exit_code is None:
                return review_cases_pb2.RunState(is_running=False)
            return review_cases_pb2.RunState(is_running=False, last_exit_code=exit_code)

    def _is_running(self, data_dir: Path) -> bool:
        active_task = self._active_tasks.get(data_dir)
        if active_task is not None and not active_task.done():
            return True
        return is_review_file_locked(data_dir / _REVIEW_FILE_NAME)

    async def _run_and_record(
        self,
        seed_file: Path,
        data_dir: Path,
        *,
        no_fetch: bool,
        refetch_stored: bool,
    ) -> None:
        exit_code = 0
        try:
            await self._execute_run(
                seed_file,
                data_dir,
                no_fetch=no_fetch,
                refetch_stored=refetch_stored,
            )
        except Exception:
            logger.exception("Seed cleaning run failed for %s", data_dir)
            exit_code = 1
        finally:
            replace_file_lines(data_dir / _EXIT_CODE_FILE_NAME, [str(exit_code)])
            with self._run_lock:
                self._active_tasks.pop(data_dir, None)

    async def _execute_run(
        self,
        seed_file: Path,
        data_dir: Path,
        *,
        no_fetch: bool,
        refetch_stored: bool,
    ) -> None:
        from seed_cleaning import run_seed_cleaning

        del no_fetch, refetch_stored
        seed_document = json.loads(seed_file.read_text(encoding="utf-8"))
        known_work_entries = KnownWorkEntriesFromSeed(seed_document["data"])
        store = ProviderPageStore(data_dir / "provider_pages.jsonl")
        await asyncio.to_thread(
            run_seed_cleaning,
            seed_document,
            seed_file,
            known_work_entries,
            store,
            data_dir,
        )
        await self._judge_open_cases(data_dir)

    async def _judge_open_cases(self, data_dir: Path) -> None:
        review_path = data_dir / _REVIEW_FILE_NAME
        if not review_path.exists():
            return
        open_cases = [
            ReviewCase.model_validate_json(line)
            for line in review_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target_cases = [
            case
            for case in open_cases
            if case.answer is None and case.kind is not ReviewCaseKind.LINKED_PAGE
        ]
        if not target_cases:
            return
        synopses = await self._read_case_synopses(target_cases)
        request = agent_service_pb2.JudgeOpenCasesRequest(
            review_cases=[convert_case_to_message(case) for case in target_cases],
            synopses_by_page_key=synopses,
        )
        async with grpc.aio.insecure_channel(self._agent_service_address) as channel:
            stub = agent_service_pb2_grpc.AgentServiceStub(channel)
            await stub.JudgeOpenCases(request, timeout=_JUDGE_TIMEOUT_SECONDS)

    async def _read_case_synopses(self, cases: list[ReviewCase]) -> dict[str, str]:
        from enrichment.pipeline.api_fetcher import ApiFetcher
        from enrichment.pipeline.page_synopsis_reader import PageSynopsisReader

        synopsis_reader = PageSynopsisReader(ApiFetcher(self._enrichment, self._cache))
        pages_by_key = {page.key: page for case in cases for page in case.pages}
        synopses: dict[str, str] = {}
        for page_key, page in pages_by_key.items():
            synopsis = await synopsis_reader.read_synopsis(page)
            if synopsis:
                synopses[page_key] = synopsis
        return synopses


def _read_exit_code(exit_code_path: Path) -> int | None:
    if not exit_code_path.exists():
        return None
    text = exit_code_path.read_text(encoding="utf-8").strip()
    return int(text) if text else None
