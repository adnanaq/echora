-- Provider page storage and approval queue tables.

-- One row per provider page, keyed by its page key.
CREATE TABLE provider_page (
  provider        text NOT NULL REFERENCES provider(value),
  entity_type       text NOT NULL REFERENCES entity_type(value),
  address         text NOT NULL,
  fetched_at      timestamptz,
  is_dead         boolean NOT NULL DEFAULT false,
  missing_since   timestamptz,
  missing_last_at timestamptz,
  missing_answers int NOT NULL DEFAULT 0 CHECK (missing_answers >= 0),
  responses_hash  text,
  fetch_failed_at timestamptz,
  fetch_failures  int NOT NULL DEFAULT 0 CHECK (fetch_failures >= 0),
  PRIMARY KEY (provider, entity_type, address),
  CHECK ((fetched_at IS NULL) = (responses_hash IS NULL))
);

-- Every response that makes up a page, stored compressed at rest.
CREATE TABLE provider_page_response (
  provider     text NOT NULL,
  entity_type    text NOT NULL,
  address      text NOT NULL,
  request_name text NOT NULL,
  content_type text NOT NULL,
  body         bytea NOT NULL,
  PRIMARY KEY (provider, entity_type, address, request_name),
  FOREIGN KEY (provider, entity_type, address)
    REFERENCES provider_page(provider, entity_type, address) ON DELETE CASCADE
);

-- Store pre-compressed bodies out of line without TOAST re-compressing them.
ALTER TABLE provider_page_response ALTER COLUMN body SET STORAGE EXTERNAL;

-- Shared approval queue for dead-page reviews and future approval workflows.
CREATE TABLE approval_request (
  id               bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  kind             text NOT NULL REFERENCES approval_kind(value),
  proposal_message text NOT NULL,
  proposal         bytea NOT NULL,
  status           text NOT NULL DEFAULT 'open' REFERENCES approval_status(value),
  created_by       text NOT NULL,
  created_at       timestamptz NOT NULL DEFAULT now(),
  decided_by       text,
  decided_at       timestamptz,
  decision_message text,
  decision         bytea
);

CREATE INDEX approval_request_open_idx
  ON approval_request(created_at)
  WHERE status = 'open';
