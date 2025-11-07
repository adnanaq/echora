"""
gRPC service implementation for Admin tasks.
"""
import grpc
import logging
from google.protobuf.struct_pb2 import Struct

from protos import admin_pb2
from protos import admin_pb2_grpc

logger = logging.getLogger(__name__)

class AdminService(admin_pb2_grpc.AdminServiceServicer):
    """
    Provides the gRPC implementation for the AdminService.
    """

    async def GetStats(self, request: admin_pb2.GetStatsRequest, context) -> admin_pb2.GetStatsResponse:
        """
        Handles the gRPC GetStats request.
        Fetches statistics from the Qdrant vector database.
        """
        # Import globals module for shared state
        from src import globals as app_globals

        if not app_globals.qdrant_client:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Qdrant client not available.")
            return admin_pb2.GetStatsResponse()

        try:
            logger.info("gRPC GetStats received request.")
            stats = await app_globals.qdrant_client.get_stats()

            if "error" in stats:
                raise Exception(f"Failed to retrieve stats: {stats['error']}")

            # Convert the additional_stats dict to a Protobuf Struct
            additional_stats_struct = Struct()
            additional_stats_dict = {
                "optimizer_status": stats.get("optimizer_status"),
                "indexed_vectors_count": stats.get("indexed_vectors_count"),
                "points_count": stats.get("points_count"),
            }
            additional_stats_struct.update(additional_stats_dict)

            return admin_pb2.GetStatsResponse(
                collection_name=stats.get("collection_name", "unknown"),
                total_documents=stats.get("total_documents", 0),
                vector_size=stats.get("vector_size", 0),
                distance_metric=stats.get("distance_metric", "unknown"),
                status=stats.get("status", "unknown"),
                additional_stats=additional_stats_struct,
            )

        except Exception as e:
            logger.error(f"gRPC GetStats failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"An internal error occurred: {e}")
            return admin_pb2.GetStatsResponse()
