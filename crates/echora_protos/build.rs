//! Compiles shared protobuf and gRPC service definitions once for the Rust workspace.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../protos/postgres_service/v1/postgres_service.proto");
    println!("cargo:rerun-if-changed=../../protos/enrichment_service/v1/enrichment_service.proto");
    println!("cargo:rerun-if-changed=../../protos/shared_proto/v1/anime.proto");
    println!("cargo:rerun-if-changed=../../protos/shared_proto/v1/merge.proto");

    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc_path);
    prost_config.btree_map(["."]);

    tonic_prost_build::configure().compile_with_config(
        prost_config,
        &[
            "../../protos/postgres_service/v1/postgres_service.proto",
            "../../protos/enrichment_service/v1/enrichment_service.proto",
        ],
        &["../../protos"],
    )?;
    Ok(())
}
