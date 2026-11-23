-- Lookup tables seeded from canonical enums and storage design, plus shared name tables.

-- Enums of the canonical models
CREATE TABLE anime_type (value text PRIMARY KEY);
INSERT INTO anime_type VALUES
  ('CM'), ('MOVIE'), ('MUSIC'), ('ONA'), ('OTHER'), ('OVA'), ('PV'),
  ('SPECIAL'), ('TV'), ('TV SHORT'), ('TV SPECIAL'), ('UNKNOWN');

CREATE TABLE anime_status (value text PRIMARY KEY);
INSERT INTO anime_status VALUES
  ('CANCELLED'), ('FINISHED'), ('ONGOING'), ('UNKNOWN'), ('UPCOMING');

CREATE TABLE anime_rating (value text PRIMARY KEY);
INSERT INTO anime_rating VALUES
  ('G - All Ages'), ('PG - Children'), ('PG-13 - Teens 13 or older'),
  ('R - 17+ (violence & profanity)'), ('R+ - Mild Nudity'), ('Rx - Hentai'), ('UNKNOWN');

CREATE TABLE anime_season (value text PRIMARY KEY);
INSERT INTO anime_season VALUES ('SPRING'), ('SUMMER'), ('FALL'), ('WINTER');

-- Shared by anime.source_material and source_material.type
CREATE TABLE source_material_type (value text PRIMARY KEY);
INSERT INTO source_material_type VALUES
  ('BOOK'), ('CARD GAME'), ('DOUJINSHI'), ('4-KOMA'), ('GAME'), ('LIGHT NOVEL'),
  ('MANGA'), ('MANHUA'), ('MANHWA'), ('MIXED MEDIA'), ('MUSIC'), ('NOVEL'),
  ('ONE SHOT'), ('ORIGINAL'), ('OTHER'), ('PICTURE BOOK'), ('RADIO'), ('UNKNOWN'),
  ('VISUAL NOVEL'), ('WEB MANGA'), ('WEB NOVEL'), ('WEBTOON'), ('FAIRY TALE'),
  ('SHORT STORY'), ('COMIC'), ('LIVE ACTION'), ('ILLUSTRATION'), ('WESTERN MEDIA');

-- rank orders the roles from strongest to weakest, so a character listed twice for one work
-- or source material shows its strongest role.
CREATE TABLE character_role (
  value  text PRIMARY KEY,
  rank   smallint NOT NULL UNIQUE CHECK (rank > 0)
);
INSERT INTO character_role (value, rank) VALUES
  ('MAIN', 1), ('SUPPORTING', 2), ('BACKGROUND', 3), ('UNKNOWN', 4);

CREATE TABLE company_role (value text PRIMARY KEY);
INSERT INTO company_role VALUES ('STUDIO'), ('PRODUCER'), ('LICENSOR'), ('UNKNOWN');

-- Values use underscores, exactly as AnimeRelationType defines them
CREATE TABLE anime_relation_type (value text PRIMARY KEY);
INSERT INTO anime_relation_type VALUES
  ('ADAPTATION'), ('ALTERNATIVE_VERSION'), ('ALTERNATIVE_SETTING'), ('CHARACTER'),
  ('CROSSOVER'), ('FULL_STORY'), ('OTHER'), ('PARENT_STORY'), ('PREQUEL'),
  ('SAME_FRANCHISE'), ('SAME_SETTING'), ('SEQUEL'), ('SIDE_STORY'), ('SPIN_OFF'), ('SUMMARY');

CREATE TABLE source_material_relation_type (value text PRIMARY KEY);
INSERT INTO source_material_relation_type VALUES
  ('ADAPTATION'), ('SOURCE'), ('ALTERNATIVE'), ('SPIN_OFF'), ('CHARACTER'),
  ('SEQUEL'), ('PREQUEL'), ('PARENT_STORY'), ('SIDE_STORY'), ('OTHER');

CREATE TABLE entity_type (value text PRIMARY KEY);
INSERT INTO entity_type VALUES ('anime'), ('character'), ('episode');

-- Values of the storage design
CREATE TABLE provider (value text PRIMARY KEY);
INSERT INTO provider VALUES
  ('mal'), ('anilist'), ('kitsu'), ('anidb'), ('anime_planet'), ('anisearch'),
  ('animeschedule'), ('syoboi'),
  ('ann'), ('livechart'), ('simkl'),
  ('animecountdown');

CREATE TABLE anime_image_type (value text PRIMARY KEY);
INSERT INTO anime_image_type VALUES ('cover'), ('poster'), ('banner');

CREATE TABLE approval_kind (value text PRIMARY KEY);
INSERT INTO approval_kind VALUES
  ('review_case'), ('risky_change'), ('suggested_edit'), ('dead_page'),
  ('character_match'), ('link_check');

CREATE TABLE approval_status (value text PRIMARY KEY);
INSERT INTO approval_status VALUES ('open'), ('approved'), ('rejected');

CREATE TABLE graph_node_type (value text PRIMARY KEY);
INSERT INTO graph_node_type VALUES
  ('anime'), ('character'), ('source_material'), ('company'), ('person');

-- Edge types of graph_edge: anime relation types, company roles, and edges between
-- works, characters, and source material.
CREATE TABLE graph_edge_type (value text PRIMARY KEY);
INSERT INTO graph_edge_type VALUES
  ('ADAPTATION'), ('ALTERNATIVE_VERSION'), ('ALTERNATIVE_SETTING'), ('CHARACTER'),
  ('CROSSOVER'), ('FULL_STORY'), ('OTHER'), ('PARENT_STORY'), ('PREQUEL'),
  ('SAME_FRANCHISE'), ('SAME_SETTING'), ('SEQUEL'), ('SIDE_STORY'), ('SPIN_OFF'), ('SUMMARY'),
  ('STUDIO'), ('PRODUCER'), ('LICENSOR'), ('UNKNOWN'),
  ('FEATURES'), ('APPEARS_IN'), ('ADAPTED_FROM'), ('WAS_ADAPTED_TO');

-- Shared names, one row per distinct name
CREATE TABLE genre (name text PRIMARY KEY);
CREATE TABLE tag (name text PRIMARY KEY);
CREATE TABLE demographic (name text PRIMARY KEY);
CREATE TABLE content_warning (name text PRIMARY KEY);
CREATE TABLE trait (name text PRIMARY KEY);

-- Language and country codes, one row per code in use, added by the save that first
-- uses one. The canonical model has already standardized every code, so a row holds
-- only the key, and a display name is read from the code, never stored.
CREATE TABLE language (
  code text PRIMARY KEY CHECK (code <> '')
);

CREATE TABLE country (
  code text PRIMARY KEY CHECK (code ~ '^[A-Z]{2}$')
);
