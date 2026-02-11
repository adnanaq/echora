pex_binary(
    name="agent_service",
    entry_point="agent_service.main",
    dependencies=["//apps/agent_service/src/agent_service"],
    # This service pulls in large ML deps (torch/transformers). Default zipapp
    # layout can exceed Python zipimport limits; packed layout avoids that.
    layout="packed",
)
