-- ECHO-100: Work names, classification, themes, and images tables.

CREATE TABLE anime_title (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  language  text NOT NULL,
  title     text NOT NULL,
  PRIMARY KEY (anime_id, language)
);

CREATE TABLE anime_synonym (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  synonym   text NOT NULL,
  PRIMARY KEY (anime_id, synonym)
);

CREATE TABLE anime_genre (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  genre     text NOT NULL REFERENCES genre(name),
  PRIMARY KEY (anime_id, genre)
);

CREATE INDEX anime_genre_genre_idx ON anime_genre(genre, anime_id);

CREATE TABLE anime_tag (
  anime_id  uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  tag       text NOT NULL REFERENCES tag(name),
  PRIMARY KEY (anime_id, tag)
);

CREATE INDEX anime_tag_tag_idx ON anime_tag(tag, anime_id);

CREATE TABLE anime_demographic (
  anime_id     uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  demographic  text NOT NULL REFERENCES demographic(name),
  PRIMARY KEY (anime_id, demographic)
);

CREATE INDEX anime_demographic_demographic_idx ON anime_demographic(demographic, anime_id);

CREATE TABLE anime_content_warning (
  anime_id         uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  content_warning  text NOT NULL REFERENCES content_warning(name),
  PRIMARY KEY (anime_id, content_warning)
);

CREATE INDEX anime_content_warning_content_warning_idx
  ON anime_content_warning(content_warning, anime_id);

CREATE TABLE anime_theme (
  anime_id     uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  name         text NOT NULL,
  description  text,
  PRIMARY KEY (anime_id, name)
);

CREATE TABLE anime_image (
  anime_id    uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  image_type  text NOT NULL REFERENCES anime_image_type(value),
  source      text NOT NULL,
  PRIMARY KEY (anime_id, image_type, source)
);
