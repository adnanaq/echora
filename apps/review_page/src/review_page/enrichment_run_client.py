"""Start seed-cleaning runs and read run status via EnrichmentService gRPC."""

import logging
from pathlib import Path
from typing import Protocol

import grpc
from enrichment_proto.v1 import enrichment_service_pb2, enrichment_service_pb2_grpc
from review_page.errors import EnrichmentServiceUnavailableError
from shared_proto.v1 import review_cases_pb2

logger = logging.getLogger(__name__)

_RPC_TIMEOUT_SECONDS = 5.0


class SeedCleaningRunClient(Protocol):
    """Start seed-cleaning runs and report active run status."""

    def start_run(self, seed_file: Path, data_dir: Path) -> review_cases_pb2.RunState:
        """Start a ``--no-fetch`` seed-cleaning run for ``seed_file`` in ``data_dir``.

        Args:
            seed_file: Path to the seed release JSON file.
            data_dir: Directory containing ``review.jsonl`` and run outputs.

        Returns:
            Current ``RunState`` protobuf message.
        """

    def get_run_state(self, data_dir: Path) -> review_cases_pb2.RunState:
        """Return the current run state for ``data_dir``.

        Args:
            data_dir: Directory containing ``review.jsonl`` and run outputs.

        Returns:
            Current ``RunState`` protobuf message.
        """


class GrpcSeedCleaningRunClient:
    """Call ``EnrichmentService/StartSeedCleaning`` and ``GetSeedCleaningStatus``."""

    def __init__(
        self,
        target_address: str,
        *,
        timeout_seconds: float = _RPC_TIMEOUT_SECONDS,
    ) -> None:
        """Configure the EnrichmentService target address and RPC timeout.

        Args:
            target_address: ``host:port`` address of the EnrichmentService server.
            timeout_seconds: Maximum seconds to wait for each RPC.
        """
        self._target_address = target_address
        self._timeout_seconds = timeout_seconds

    def start_run(self, seed_file: Path, data_dir: Path) -> review_cases_pb2.RunState:
        """Request ``StartSeedCleaning`` with ``no_fetch=True`` over gRPC.

        Args:
            seed_file: Path to the seed release JSON file.
            data_dir: Directory containing ``review.jsonl`` and run outputs.

        Returns:
            Current ``RunState`` returned by ``EnrichmentService``.
        """
        request = enrichment_service_pb2.StartSeedCleaningRequest(
            seed_file=str(seed_file),
            data_dir=str(data_dir),
            no_fetch=True,
        )
        try:
            with grpc.insecure_channel(self._target_address) as channel:
                stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(channel)
                return stub.StartSeedCleaning(request, timeout=self._timeout_seconds)
        except grpc.RpcError as error:
            logger.warning("EnrichmentService unavailable for start_run: %s", error)
            raise EnrichmentServiceUnavailableError(self._target_address) from error

    def get_run_state(self, data_dir: Path) -> review_cases_pb2.RunState:
        """Query ``GetSeedCleaningStatus`` over gRPC, returning idle on failure.

        Args:
            data_dir: Directory containing ``review.jsonl`` and run outputs.

        Returns:
            Current ``RunState``, or idle if ``EnrichmentService`` is unreachable.
        """
        request = enrichment_service_pb2.GetSeedCleaningStatusRequest(
            data_dir=str(data_dir)
        )
        try:
            with grpc.insecure_channel(self._target_address) as channel:
                stub = enrichment_service_pb2_grpc.EnrichmentServiceStub(channel)
                return stub.GetSeedCleaningStatus(
                    request, timeout=self._timeout_seconds
                )
        except grpc.RpcError as error:
            logger.warning("EnrichmentService unavailable for run state: %s", error)
            return review_cases_pb2.RunState(is_running=False)
