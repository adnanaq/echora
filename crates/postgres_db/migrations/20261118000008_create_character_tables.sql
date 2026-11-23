-- Character tables, views, indexes, appearances by work page, and approved field corrections.

CREATE TABLE character (
  id                        uuid PRIMARY KEY,
  name                      text NOT NULL,
  name_native               text,
  description               text,
  favorites                 int CHECK (favorites >= 0),
  version                   bigint NOT NULL DEFAULT 1,
  merged_record_hash        text,
  hash_format               smallint,
  merge_version             int,
  merge_requested_at        timestamptz,
  merged_at                 timestamptz,
  refreshed_at              timestamptz,
  refresh_failed_at         timestamptz,
  next_refresh_at           timestamptz,
  refresh_failures          int NOT NULL DEFAULT 0 CHECK (refresh_failures >= 0),
  deleted_at                timestamptz,
  merged_into_character_id  uuid REFERENCES character(id),
  created_at                timestamptz NOT NULL DEFAULT now(),
  updated_at                timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX character_merge_requested_idx
  ON character(merge_requested_at)
  WHERE merge_requested_at IS NOT NULL;

CREATE INDEX character_merged_into_idx
  ON character(merged_into_character_id)
  WHERE merged_into_character_id IS NOT NULL;

CREATE INDEX character_refresh_order_idx
  ON character((coalesce(next_refresh_at, '-infinity'::timestamptz)), id)
  WHERE deleted_at IS NULL;

CREATE VIEW character_visible AS
  SELECT *
  FROM character
  WHERE merged_at IS NOT NULL AND deleted_at IS NULL;

-- Character.sources: each character page belongs to exactly one character.
CREATE TABLE character_source (
  character_id  uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  provider      text NOT NULL,
  entity_type   text NOT NULL DEFAULT 'character' CHECK (entity_type = 'character'),
  address       text NOT NULL,
  PRIMARY KEY (provider, entity_type, address),
  CONSTRAINT character_source_provider_key UNIQUE (character_id, provider),
  FOREIGN KEY (provider, entity_type, address)
    REFERENCES provider_page(provider, entity_type, address) ON DELETE RESTRICT
);

-- Character.animeography: each work the character's pages list, held by the work's page,
-- with the character's role there. The work is found by joining the page to anime_source,
-- so a work stored after the character, or a page that moves, needs no change here.
-- The page has no foreign key to provider_page, since the work's page may not be fetched yet.
CREATE TABLE character_animeography (
  character_id         uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  anime_page_provider  text NOT NULL,
  anime_page_address   text NOT NULL,
  role                 text NOT NULL REFERENCES character_role(value),
  PRIMARY KEY (character_id, anime_page_provider, anime_page_address)
);

-- A work's cast, and the characters to update when a page becomes a work's source.
CREATE INDEX character_animeography_anime_page_idx
  ON character_animeography(anime_page_provider, anime_page_address);

CREATE TABLE character_name_variation (
  character_id    uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  name_variation  text NOT NULL,
  PRIMARY KEY (character_id, name_variation)
);

CREATE TABLE character_nickname (
  character_id  uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  nickname      text NOT NULL,
  PRIMARY KEY (character_id, nickname)
);

CREATE TABLE character_trait (
  character_id  uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  trait         text NOT NULL REFERENCES trait(name),
  PRIMARY KEY (character_id, trait)
);

-- Characters by trait.
CREATE INDEX character_trait_trait_idx ON character_trait(trait, character_id);

CREATE TABLE character_image (
  character_id  uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  source        text NOT NULL,
  PRIMARY KEY (character_id, source)
);

CREATE TABLE character_attribute (
  character_id     uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  attribute_name   text NOT NULL,
  attribute_value  text NOT NULL,
  PRIMARY KEY (character_id, attribute_name)
);

-- Character.spoilers, the description spoiler included as spoiler_name 'description'.
CREATE TABLE character_spoiler (
  character_id   uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  spoiler_name   text NOT NULL,
  spoiler_value  text NOT NULL,
  PRIMARY KEY (character_id, spoiler_name)
);

-- Character.mangaography: each manga or novel the character's pages list, held by its
-- provider identifier, with the character's role there. The source material is found by
-- joining the identifier to source_material_xref, so source material stored after the
-- character still reaches it.
CREATE TABLE character_mangaography (
  character_id                 uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  source_material_provider     text NOT NULL,
  source_material_external_id  text NOT NULL,
  role                         text NOT NULL REFERENCES character_role(value),
  PRIMARY KEY (character_id, source_material_provider, source_material_external_id)
);

-- The characters of a source material, joined from its cross-references.
CREATE INDEX character_mangaography_source_material_idx
  ON character_mangaography(source_material_provider, source_material_external_id);

-- Approved per-field overrides applied after every character merge.
CREATE TABLE character_field_correction (
  id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  character_id         uuid NOT NULL REFERENCES character(id) ON DELETE CASCADE,
  field_path           text NOT NULL,
  value_message        text NOT NULL,
  value                bytea NOT NULL,
  approval_request_id  bigint REFERENCES approval_request(id) ON DELETE SET NULL,
  approved_by          text NOT NULL,
  approved_at          timestamptz NOT NULL DEFAULT now(),
  UNIQUE (character_id, field_path)
);

CREATE INDEX character_field_correction_approval_idx
  ON character_field_correction(approval_request_id)
  WHERE approval_request_id IS NOT NULL;
