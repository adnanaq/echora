python_requirements(
    name="reqs0",
    source="pyproject.toml",
    module_mapping={
        "open-clip-torch": ["open_clip"],
        "protobuf": ["google.protobuf"],
        "flagembedding": ["FlagEmbedding"],
    },
)

files(
    name="rust_workspace",
    sources=[
        "Cargo.toml",
        "Cargo.lock",
        ".cargo/config.toml",
        ".sqlx/*.json",
        "protos/**/*.proto",
    ],
)
