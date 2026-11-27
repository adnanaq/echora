-- Works, work-to-page links, approved field corrections, visible view, and outbox.

CREATE TABLE anime (
  id                   uuid PRIMARY KEY,
  title                text,
  title_english        text,
  title_japanese       text,
  type                 text REFERENCES anime_type(value),
  status               text REFERENCES anime_status(value),
  rating               text NOT NULL DEFAULT 'UNKNOWN' REFERENCES anime_rating(value),
  season               text REFERENCES anime_season(value),
  source_material      text REFERENCES source_material_type(value),
  year                 smallint CHECK (year >= 1900),
  month                smallint CHECK (month BETWEEN 1 AND 12),
  episode_count        int NOT NULL DEFAULT 0 CHECK (episode_count >= 0),
  duration_seconds     int CHECK (duration_seconds >= 0),
  is_nsfw              boolean,
  country_of_origin    text REFERENCES country(code),
  synopsis             text,
  background           text,
  aired_from           timestamptz,
  aired_to             timestamptz,
  score_weighted       numeric(4, 2) CHECK (score_weighted BETWEEN 0 AND 10),
  score_mean           numeric(4, 2) CHECK (score_mean BETWEEN 0 AND 10),
  score_median         numeric(4, 2) CHECK (score_median BETWEEN 0 AND 10),
  version              bigint NOT NULL DEFAULT 1,
  merged_record_hash   text,
  hash_format          smallint,
  merge_version        int,
  merge_requested_at   timestamptz,
  merged_at            timestamptz,
  refreshed_at         timestamptz,
  refresh_failed_at    timestamptz,
  next_refresh_at      timestamptz,
  refresh_failures     int NOT NULL DEFAULT 0 CHECK (refresh_failures >= 0),
  deleted_at           timestamptz,
  merged_into_anime_id uuid REFERENCES anime(id),
  created_at           timestamptz NOT NULL DEFAULT now(),
  updated_at           timestamptz NOT NULL DEFAULT now(),
  CHECK (merged_at IS NULL OR (title IS NOT NULL AND type IS NOT NULL AND status IS NOT NULL))
);

CREATE INDEX anime_merge_requested_idx
  ON anime(merge_requested_at)
  WHERE merge_requested_at IS NOT NULL;

CREATE INDEX anime_merged_into_idx
  ON anime(merged_into_anime_id)
  WHERE merged_into_anime_id IS NOT NULL;

-- Works by country, and the foreign key check when an unused country is deleted
CREATE INDEX anime_country_of_origin_idx
  ON anime(country_of_origin)
  WHERE country_of_origin IS NOT NULL;

CREATE INDEX anime_refresh_order_idx
  ON anime((coalesce(next_refresh_at, '-infinity'::timestamptz)), id)
  WHERE deleted_at IS NULL;

-- The daily score refresh reads ongoing and recently finished works.
CREATE INDEX anime_score_sync_idx ON anime(status, aired_to)
  WHERE deleted_at IS NULL AND status IN ('ONGOING', 'FINISHED');

CREATE VIEW anime_visible AS
  SELECT *
  FROM anime
  WHERE merged_at IS NOT NULL AND deleted_at IS NULL;

-- Anime.sources: the pages merged into each work. A page is the source of at most
-- one work, and a work has at most one source page per provider.
CREATE TABLE anime_source (
  anime_id   uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  provider   text NOT NULL,
  entity_type  text NOT NULL DEFAULT 'anime' CHECK (entity_type = 'anime'),
  address    text NOT NULL,
  PRIMARY KEY (provider, entity_type, address),
  CONSTRAINT anime_source_provider_key UNIQUE (anime_id, provider),
  FOREIGN KEY (provider, entity_type, address)
    REFERENCES provider_page(provider, entity_type, address) ON DELETE RESTRICT
);

-- Anime.partial_sources: pages listing a larger work this work is part of, linked for
-- reference and never merged. A page can be a partial source of several works.
CREATE TABLE anime_partial_source (
  anime_id   uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  provider   text NOT NULL,
  entity_type  text NOT NULL DEFAULT 'anime' CHECK (entity_type = 'anime'),
  address    text NOT NULL,
  PRIMARY KEY (anime_id, provider, entity_type, address),
  FOREIGN KEY (provider, entity_type, address)
    REFERENCES provider_page(provider, entity_type, address) ON DELETE RESTRICT
);

CREATE INDEX anime_partial_source_page_idx
  ON anime_partial_source(provider, entity_type, address);

-- Approved per-field overrides applied after every merge.
CREATE TABLE anime_field_correction (
  id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id            uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  field_path          text NOT NULL,
  value_message       text NOT NULL,
  value               bytea NOT NULL,
  approval_request_id bigint REFERENCES approval_request(id) ON DELETE SET NULL,
  approved_by         text NOT NULL,
  approved_at         timestamptz NOT NULL DEFAULT now(),
  UNIQUE (anime_id, field_path)
);

CREATE INDEX anime_field_correction_approval_idx
  ON anime_field_correction(approval_request_id)
  WHERE approval_request_id IS NOT NULL;

-- Transactional outbox for NATS catalogue and notification events.
CREATE TABLE outbox (
  id               uuid PRIMARY KEY DEFAULT uuidv7(),
  subject          text NOT NULL,
  entity_id        uuid NOT NULL,
  version          bigint NOT NULL,
  payload          bytea NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  claimed_until    timestamptz,
  published_at     timestamptz,
  publish_failures int NOT NULL DEFAULT 0,
  parked_at        timestamptz,
  traceparent      text
);

CREATE INDEX outbox_unpublished_idx
  ON outbox(id)
  WHERE published_at IS NULL AND parked_at IS NULL;
