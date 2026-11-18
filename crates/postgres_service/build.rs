//! Compiles protobuf definitions for `postgres_service`.

// Rust guideline compliant 2026-10-09

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../protos/postgres_service/v1/postgres_service.proto");
    println!("cargo:rerun-if-changed=../../protos/shared_proto/v1/merge.proto");

    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    let mut prost_config = prost_build::Config::new();
    prost_config.protoc_executable(protoc_path);

    tonic_prost_build::configure().compile_with_config(
        prost_config,
        &["../../protos/postgres_service/v1/postgres_service.proto"],
        &["../../protos"],
    )?;
    Ok(())
}
