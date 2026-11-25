-- Scheduling indexes for daily score sync and upcoming episode broadcast lookups.

CREATE INDEX anime_score_sync_idx ON anime(status, aired_to)
  WHERE deleted_at IS NULL AND status IN ('ONGOING', 'FINISHED');

CREATE INDEX anime_broadcast_next_episode_idx
  ON anime_broadcast(next_episode_at)
  WHERE next_episode_at IS NOT NULL;
