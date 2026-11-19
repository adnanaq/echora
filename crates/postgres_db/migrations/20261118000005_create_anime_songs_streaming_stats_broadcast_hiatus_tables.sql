-- ECHO-101: Theme songs, trailers, external links, streaming, statistics, broadcast, and hiatus history.

CREATE TABLE anime_theme_song (
  id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  kind      text NOT NULL REFERENCES theme_song_kind(value),
  position  int NOT NULL,
  title     text NOT NULL,
  artist    text,
  UNIQUE (anime_id, kind, position)
);

CREATE TABLE theme_song_episode_range (
  theme_song_id  bigint NOT NULL REFERENCES anime_theme_song(id) ON DELETE CASCADE,
  range_start    int NOT NULL CHECK (range_start >= 0),
  range_end      int CHECK (range_end >= range_start),
  PRIMARY KEY (theme_song_id, range_start)
);

CREATE TABLE anime_trailer (
  anime_id       uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source         text NOT NULL,
  title          text,
  thumbnail_url  text,
  PRIMARY KEY (anime_id, source)
);

CREATE TABLE anime_external_link (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source    text NOT NULL,
  platform  text NOT NULL,
  label     text,
  language  text,
  PRIMARY KEY (anime_id, source)
);

CREATE TABLE anime_streaming (
  id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id             uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  platform             text NOT NULL,
  source               text NOT NULL,
  region               text,
  is_free              boolean,
  is_premium_required  boolean,
  has_dub              boolean,
  UNIQUE NULLS NOT DISTINCT (anime_id, source, region)
);

CREATE TABLE anime_streaming_subtitle_language (
  streaming_id  bigint NOT NULL REFERENCES anime_streaming(id) ON DELETE CASCADE,
  language      text NOT NULL,
  PRIMARY KEY (streaming_id, language)
);

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

CREATE TABLE anime_contextual_rank (
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

CREATE TABLE anime_hiatus (
  id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  anime_id      uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  reason        text,
  hiatus_from   date,
  hiatus_until  date,
  recorded_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX anime_hiatus_latest_idx ON anime_hiatus(anime_id, recorded_at DESC);
