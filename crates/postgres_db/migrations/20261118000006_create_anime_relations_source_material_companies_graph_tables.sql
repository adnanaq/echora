-- ECHO-102: Relations, source material, companies, and graph_edge projection.

CREATE TABLE anime_relation (
  id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  from_anime_id     uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  relation_type     text NOT NULL REFERENCES anime_relation_type(value),
  to_anime_id       uuid REFERENCES anime(id) ON DELETE CASCADE,
  to_page_provider  text NOT NULL,
  to_page_address   text NOT NULL,
  CHECK (from_anime_id <> to_anime_id),
  UNIQUE (from_anime_id, relation_type, to_page_provider, to_page_address)
);

CREATE INDEX anime_relation_from_idx ON anime_relation(from_anime_id);
CREATE INDEX anime_relation_to_idx ON anime_relation(to_anime_id);

CREATE INDEX anime_relation_unresolved_idx
  ON anime_relation(to_page_provider, to_page_address)
  WHERE to_anime_id IS NULL;

CREATE TABLE anime_relation_claim (
  from_anime_id     uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  provider          text NOT NULL REFERENCES provider(value),
  relation_type     text NOT NULL REFERENCES anime_relation_type(value),
  to_page_provider  text NOT NULL,
  to_page_address   text NOT NULL,
  PRIMARY KEY (from_anime_id, provider, relation_type, to_page_provider, to_page_address)
);

CREATE INDEX anime_relation_claim_target_idx
  ON anime_relation_claim(to_page_provider, to_page_address);

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

CREATE TABLE anime_source_material_relation (
  anime_id            uuid NOT NULL REFERENCES anime(id) ON DELETE CASCADE,
  source_material_id  uuid NOT NULL REFERENCES source_material(id) ON DELETE CASCADE,
  relation_type       text NOT NULL REFERENCES source_material_relation_type(value),
  PRIMARY KEY (anime_id, source_material_id, relation_type)
);

CREATE INDEX anime_source_material_relation_work_idx
  ON anime_source_material_relation(source_material_id);

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
  from_id    uuid NOT NULL,
  from_type  text NOT NULL REFERENCES graph_node_type(value),
  to_id      uuid NOT NULL,
  to_type    text NOT NULL REFERENCES graph_node_type(value),
  edge_type  text NOT NULL REFERENCES graph_edge_type(value),
  PRIMARY KEY (from_id, to_id, edge_type)
);

CREATE INDEX graph_edge_from_idx ON graph_edge(from_id, from_type, edge_type);
CREATE INDEX graph_edge_to_idx   ON graph_edge(to_id,   to_type,   edge_type);
