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
  duration             int CHECK (duration >= 0),
  nsfw                 boolean,
  country_of_origin    text CHECK (char_length(country_of_origin) = 2),
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

CREATE INDEX anime_refresh_order_idx
  ON anime((coalesce(next_refresh_at, '-infinity'::timestamptz)), id)
  WHERE deleted_at IS NULL;

CREATE VIEW anime_visible AS
  SELECT *
  FROM anime
  WHERE merged_at IS NOT NULL AND deleted_at IS NULL;

-- Which provider pages make up each work.
CREATE TABLE anime_provider_page (
  anime_id   uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  provider   text NOT NULL,
  page_kind  text NOT NULL DEFAULT 'anime' CHECK (page_kind = 'anime'),
  address    text NOT NULL,
  role       text NOT NULL REFERENCES anime_page_role(value),
  PRIMARY KEY (anime_id, provider, page_kind, address),
  FOREIGN KEY (provider, page_kind, address)
    REFERENCES provider_page(provider, page_kind, address) ON DELETE RESTRICT
);

CREATE UNIQUE INDEX anime_provider_page_one_source_uniq
  ON anime_provider_page(provider, address)
  WHERE role = 'source';

CREATE UNIQUE INDEX anime_provider_page_source_per_provider_uniq
  ON anime_provider_page(anime_id, provider)
  WHERE role = 'source';

CREATE INDEX anime_provider_page_page_idx
  ON anime_provider_page(provider, page_kind, address);

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
