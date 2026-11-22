-- ECHO-102: Relations, source material, companies, and graph_edge projection.

-- Anime.related_anime, stored by the target's page. The target work is found by
-- joining that page to anime_source, so nothing here changes when the page moves.
CREATE TABLE anime_related_anime (
  id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  from_anime_id     uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  relation_type     text NOT NULL REFERENCES anime_relation_type(value),
  to_page_provider  text NOT NULL,
  to_page_address   text NOT NULL,
  CONSTRAINT anime_related_anime_target_key
    UNIQUE (from_anime_id, relation_type, to_page_provider, to_page_address)
);

-- The relations pointing at a page, for the join to its work and for recomputing
-- the pairs of a page that moves. Lookups by from_anime_id use the unique constraint.
CREATE INDEX anime_related_anime_target_page_idx
  ON anime_related_anime(to_page_provider, to_page_address);

CREATE TABLE anime_related_anime_claim (
  from_anime_id     uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  provider          text NOT NULL REFERENCES provider(value),
  relation_type     text NOT NULL REFERENCES anime_relation_type(value),
  to_page_provider  text NOT NULL,
  to_page_address   text NOT NULL,
  PRIMARY KEY (from_anime_id, provider, relation_type, to_page_provider, to_page_address)
);

CREATE INDEX anime_related_anime_claim_target_idx
  ON anime_related_anime_claim(to_page_provider, to_page_address);

CREATE TABLE source_material (
  id              uuid PRIMARY KEY DEFAULT uuidv7(),
  title           text NOT NULL,
  title_english   text,
  title_japanese  text,
  type            text NOT NULL REFERENCES source_material_type(value),
  status          text REFERENCES anime_status(value),
  score           numeric(4,2) CHECK (score BETWEEN 0 AND 10),
  chapters        int CHECK (chapters >= 0),
  volumes         int CHECK (volumes >= 0),
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE source_material_xref (
  provider            text NOT NULL REFERENCES provider(value),
  external_id         text NOT NULL,
  source_material_id  uuid NOT NULL REFERENCES source_material(id) ON DELETE CASCADE,
  source              text NOT NULL,
  PRIMARY KEY (provider, external_id),
  UNIQUE (source_material_id, provider)
);

CREATE TABLE source_material_image (
  source_material_id  uuid NOT NULL REFERENCES source_material(id) ON DELETE CASCADE,
  source              text NOT NULL,
  PRIMARY KEY (source_material_id, source)
);

-- Anime.related_source_material
CREATE TABLE anime_related_source_material (
  anime_id            uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source_material_id  uuid NOT NULL REFERENCES source_material(id) ON DELETE CASCADE,
  relation_type       text NOT NULL REFERENCES source_material_relation_type(value),
  PRIMARY KEY (anime_id, source_material_id, relation_type)
);

CREATE INDEX anime_related_source_material_work_idx
  ON anime_related_source_material(source_material_id);

CREATE TABLE company (
  id           uuid PRIMARY KEY DEFAULT uuidv7(),
  name         text NOT NULL,
  description  text
);

CREATE TABLE company_xref (
  provider    text NOT NULL REFERENCES provider(value),
  source      text NOT NULL,
  company_id  uuid NOT NULL REFERENCES company(id) ON DELETE CASCADE,
  PRIMARY KEY (provider, source),
  UNIQUE (company_id, provider)
);

CREATE TABLE anime_company (
  anime_id    uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  company_id  uuid NOT NULL REFERENCES company(id) ON DELETE RESTRICT,
  role        text NOT NULL REFERENCES company_role(value),
  PRIMARY KEY (anime_id, company_id, role)
);

CREATE INDEX anime_company_company_idx ON anime_company(company_id);

CREATE TABLE graph_edge (
  from_id         uuid NOT NULL,
  from_node_type  text NOT NULL REFERENCES graph_node_type(value),
  to_id           uuid NOT NULL,
  to_node_type    text NOT NULL REFERENCES graph_node_type(value),
  edge_type       text NOT NULL REFERENCES graph_edge_type(value),
  PRIMARY KEY (from_id, to_id, edge_type)
);

CREATE INDEX graph_edge_from_idx ON graph_edge(from_id, from_node_type, edge_type);
CREATE INDEX graph_edge_to_idx   ON graph_edge(to_id,   to_node_type,   edge_type);
