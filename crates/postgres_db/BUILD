"""PostgreSQL database library crate sources for Docker image builds."""

files(
    name="migrations",
    sources=["migrations/**/*.sql"],
)

files(
    name="sources",
    sources=["Cargo.toml", "build.rs", "src/**/*.rs", "migrations/**/*.sql"],
)
