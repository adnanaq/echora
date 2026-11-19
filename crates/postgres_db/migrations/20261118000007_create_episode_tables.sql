-- Episode tables, views, indexes, list entries, and approved field corrections.

CREATE TABLE episode (
  id                  uuid PRIMARY KEY,
  anime_id            uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  episode_number      int NOT NULL CHECK (episode_number >= 0),
  season_number       int CHECK (season_number > 0),
  title               text,
  title_japanese      text,
  title_romaji        text,
  synopsis            text,
  description         text,
  aired_at            timestamptz,
  aired_notified_at   timestamptz,
  duration_seconds    int CHECK (duration_seconds >= 0),
  is_filler           boolean NOT NULL DEFAULT false,
  is_recap            boolean NOT NULL DEFAULT false,
  score               numeric(5, 2) CHECK (score >= 0),
  version             bigint NOT NULL DEFAULT 1,
  merged_record_hash  text,
  hash_format         smallint,
  merge_version       int,
  merge_requested_at  timestamptz,
  merged_at           timestamptz,
  refreshed_at        timestamptz,
  refresh_failed_at   timestamptz,
  next_refresh_at     timestamptz,
  refresh_failures    int NOT NULL DEFAULT 0 CHECK (refresh_failures >= 0),
  deleted_at          timestamptz,
  created_at          timestamptz NOT NULL DEFAULT now(),
  updated_at          timestamptz NOT NULL DEFAULT now()
);

-- One undeleted episode per number: a removed episode never blocks its number being listed again.
CREATE UNIQUE INDEX episode_number_uniq
  ON episode(anime_id, episode_number)
  WHERE deleted_at IS NULL;

CREATE INDEX episode_merge_requested_idx
  ON episode(merge_requested_at)
  WHERE merge_requested_at IS NOT NULL;

CREATE INDEX episode_refresh_order_idx
  ON episode((coalesce(next_refresh_at, '-infinity'::timestamptz)), id)
  WHERE deleted_at IS NULL;

CREATE INDEX episode_aired_at_idx
  ON episode(aired_at)
  WHERE aired_at IS NOT NULL;

CREATE VIEW episode_visible AS
  SELECT *
  FROM episode
  WHERE merged_at IS NOT NULL AND deleted_at IS NULL;

-- Each episode page belongs to exactly one episode.
CREATE TABLE episode_provider_page (
  episode_id  uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  provider    text NOT NULL,
  page_kind   text NOT NULL DEFAULT 'episode' CHECK (page_kind = 'episode'),
  address     text NOT NULL,
  PRIMARY KEY (provider, page_kind, address),
  UNIQUE (episode_id, provider),
  FOREIGN KEY (provider, page_kind, address)
    REFERENCES provider_page(provider, page_kind, address) ON DELETE RESTRICT
);

-- What one provider's work page says about an episode, sliced out by the work's merge.
CREATE TABLE episode_list_entry (
  episode_id        uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  provider          text NOT NULL REFERENCES provider(value),
  title             text,
  aired_at          timestamptz,
  duration_seconds  int CHECK (duration_seconds >= 0),
  is_filler         boolean,
  is_recap          boolean,
  PRIMARY KEY (episode_id, provider)
);

CREATE TABLE episode_title (
  episode_id  uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  language    text NOT NULL,
  title       text NOT NULL,
  PRIMARY KEY (episode_id, language)
);

CREATE TABLE episode_image (
  episode_id  uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  url         text NOT NULL,
  PRIMARY KEY (episode_id, url)
);

CREATE TABLE episode_streaming (
  episode_id  uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  platform    text NOT NULL,
  url         text NOT NULL,
  PRIMARY KEY (episode_id, platform)
);

-- Characters in an episode, by their character page until ECHO-104 creates the character table.
CREATE TABLE episode_character (
  episode_id               uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  character_page_provider  text NOT NULL,
  character_page_address   text NOT NULL,
  character_id             uuid,
  role                     text NOT NULL REFERENCES character_role(value),
  PRIMARY KEY (episode_id, character_page_provider, character_page_address)
);

CREATE INDEX episode_character_unresolved_idx
  ON episode_character(character_page_provider, character_page_address)
  WHERE character_id IS NULL;

-- Approved per-field overrides applied after every episode merge.
CREATE TABLE episode_field_correction (
  id                  bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  episode_id          uuid NOT NULL REFERENCES episode(id) ON DELETE CASCADE,
  field_path          text NOT NULL,
  value_message       text NOT NULL,
  value               bytea NOT NULL,
  approval_request_id bigint REFERENCES approval_request(id) ON DELETE SET NULL,
  approved_by         text NOT NULL,
  approved_at         timestamptz NOT NULL DEFAULT now(),
  UNIQUE (episode_id, field_path)
);

CREATE INDEX episode_field_correction_approval_idx
  ON episode_field_correction(approval_request_id)
  WHERE approval_request_id IS NOT NULL;
