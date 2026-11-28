-- ECHO-101: Opening and ending themes, trailers, external sources, streaming, statistics, broadcast, and hiatus history.

-- Anime.opening_themes and Anime.ending_themes, one table each.
CREATE TABLE anime_opening_theme (
  id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  position  int NOT NULL,
  title     text NOT NULL,
  artist    text,
  CONSTRAINT anime_opening_theme_position_key UNIQUE (anime_id, position)
);

-- ThemeSong.episodes of an opening.
CREATE TABLE anime_opening_theme_episode (
  opening_theme_id  bigint NOT NULL REFERENCES anime_opening_theme(id) ON DELETE CASCADE,
  range_start       int NOT NULL CHECK (range_start >= 0),
  range_end         int CHECK (range_end >= range_start),
  PRIMARY KEY (opening_theme_id, range_start)
);

CREATE TABLE anime_ending_theme (
  id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  position  int NOT NULL,
  title     text NOT NULL,
  artist    text,
  CONSTRAINT anime_ending_theme_position_key UNIQUE (anime_id, position)
);

-- ThemeSong.episodes of an ending.
CREATE TABLE anime_ending_theme_episode (
  ending_theme_id  bigint NOT NULL REFERENCES anime_ending_theme(id) ON DELETE CASCADE,
  range_start      int NOT NULL CHECK (range_start >= 0),
  range_end        int CHECK (range_end >= range_start),
  PRIMARY KEY (ending_theme_id, range_start)
);

-- One row per trailer video, shared by every work that lists it.
CREATE TABLE trailer (
  source            text PRIMARY KEY,
  title             text,
  thumbnail_source  text
);

CREATE TABLE anime_trailer (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source    text NOT NULL REFERENCES trailer(source),
  PRIMARY KEY (anime_id, source)
);

CREATE INDEX anime_trailer_source_idx ON anime_trailer(source, anime_id);

-- One row per external URL, links and streaming pages alike. Its platform and
-- language describe the URL, whichever work lists it.
CREATE TABLE external_source (
  source    text PRIMARY KEY,
  platform  text NOT NULL,
  language  text REFERENCES language(code)
);

CREATE INDEX external_source_language_idx
  ON external_source(language)
  WHERE language IS NOT NULL;

-- Anime.external_sources; the label is the provider's own name for the link, per work.
CREATE TABLE anime_external_source (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source    text NOT NULL REFERENCES external_source(source),
  label     text,
  PRIMARY KEY (anime_id, source)
);

CREATE INDEX anime_external_source_source_idx ON anime_external_source(source, anime_id);

-- Anime.streaming_sources; the platform is read from external_source.
CREATE TABLE anime_streaming_source (
  id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id             uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source               text NOT NULL REFERENCES external_source(source),
  region               text,
  is_free              boolean,
  is_premium_required  boolean,
  has_dub              boolean,
  CONSTRAINT anime_streaming_source_region_key UNIQUE NULLS NOT DISTINCT (anime_id, source, region)
);

CREATE INDEX anime_streaming_source_source_idx ON anime_streaming_source(source);

-- StreamingEntry.subtitle_languages. Constraints are named, since the generated
-- names pass PostgreSQL's 63-byte limit.
CREATE TABLE anime_streaming_source_subtitle_language (
  streaming_source_id  bigint CONSTRAINT anime_streaming_subtitle_source_not_null NOT NULL,
  language             text NOT NULL,
  CONSTRAINT anime_streaming_subtitle_pkey PRIMARY KEY (streaming_source_id, language),
  CONSTRAINT anime_streaming_subtitle_source_fkey FOREIGN KEY (streaming_source_id)
    REFERENCES anime_streaming_source(id) ON DELETE CASCADE,
  CONSTRAINT anime_streaming_subtitle_language_fkey FOREIGN KEY (language)
    REFERENCES language(code)
);

CREATE INDEX anime_streaming_subtitle_language_idx
  ON anime_streaming_source_subtitle_language(language);

CREATE TABLE anime_statistics (
  anime_id    uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  provider    text NOT NULL REFERENCES provider(value),
  score       numeric(4,2) CHECK (score BETWEEN 0 AND 10),
  scored_by   int CHECK (scored_by >= 0),
  members     int CHECK (members >= 0),
  favorites   int CHECK (favorites >= 0),
  rank        int CHECK (rank > 0),
  popularity  int CHECK (popularity > 0),
  PRIMARY KEY (anime_id, provider)
);

-- Statistics.contextual_ranks
CREATE TABLE anime_statistics_contextual_rank (
  anime_id     uuid NOT NULL,
  provider     text NOT NULL,
  context      text NOT NULL,
  rank         int NOT NULL CHECK (rank > 0),
  is_all_time  boolean,
  format       text,
  season       text REFERENCES anime_season(value),
  year         smallint CHECK (year >= 1900),
  PRIMARY KEY (anime_id, provider, context),
  FOREIGN KEY (anime_id, provider)
    REFERENCES anime_statistics(anime_id, provider) ON DELETE CASCADE
);

CREATE TABLE anime_broadcast (
  anime_id             uuid PRIMARY KEY REFERENCES anime(id) ON DELETE CASCADE,
  weekday              smallint CHECK (weekday BETWEEN 1 AND 7),
  timezone             text,
  jp_time              time,
  sub_time             time,
  dub_time             time,
  sub_delay_days       int,
  dub_delay_days       int,
  premiere_jp          timestamptz,
  premiere_sub         timestamptz,
  premiere_dub         timestamptz,
  next_episode_at      timestamptz,
  next_episode_number  int CHECK (next_episode_number >= 0)
);

-- Episode tracking finds works whose next episode airs soon.
CREATE INDEX anime_broadcast_next_episode_idx
  ON anime_broadcast(next_episode_at)
  WHERE next_episode_at IS NOT NULL;

CREATE TABLE anime_hiatus (
  id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id      uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  reason        text,
  hiatus_from   date,
  hiatus_until  date,
  recorded_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX anime_hiatus_latest_idx ON anime_hiatus(anime_id, recorded_at DESC);
