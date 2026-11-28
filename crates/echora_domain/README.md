# `echora_domain`

Domain types that every Echora Rust service shares, so the PostgreSQL Service, the Temporal workflows and the Update Service agree on their names, rules and wire forms. A concept the services pass between each other belongs here, while infrastructure such as settings and telemetry stays in [`common`](../common/README.md).

## Contents

- `Provider`: an external site whose anime pages the catalogue stores. `Provider::ALL` lists every provider in the order of the `provider` lookup table.
- `EntityType`: a work, character or episode, which is also what a provider page describes. The variants follow the `shared_proto.v1.EntityType` numbers.
- `Address` and `PageKey`: one provider page. A page key is built only through `PageKey::new`, which checks the address against its provider's rules for that entity type, so a MAL episode address is `<work id>:<episode number>` and a Kitsu address may not hold `/`, `?`, `#` or whitespace.
- `KeyError`: why a provider name, entity type or address was refused. `is_unknown_provider`, `is_unknown_entity_type`, `is_unspecified_entity_type` and `is_invalid_address` tell the cases apart, so each service answers with its own status code.

Every value is written by its lookup name (`mal`, `anime_planet`, `episode`), the same name the PostgreSQL lookup tables and the protos use.

## Features

Each feature adds the conversions one consumer needs and changes nothing else.

| Feature | Adds | Used by |
| --- | --- | --- |
| `proto` | `From` and `TryFrom` for the `shared_proto.v1` `EntityType` and `PageKey` messages, and `TryFrom<i32>` for a message's `entity_type` number | `postgres_db`, `postgres_service`, `update_service` |
| `serde` | `Serialize` and `Deserialize`. A page key is an object with `provider`, `entity_type` and `address` fields, and reading one checks its address | `temporal_workflows` |
| `sqlx` | `Type` and `Decode` for reading PostgreSQL `text` lookup columns | `postgres_db` |

## Layout

- `src/provider.rs` holds `Provider` and its address rules.
- `src/entity_type.rs` holds `EntityType`.
- `src/page_key.rs` holds `Address` and `PageKey`.
- `src/error.rs` holds `KeyError`.
- `src/proto.rs` and `src/sql.rs` hold the `proto` and `sqlx` conversions.
