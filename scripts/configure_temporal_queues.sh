#!/usr/bin/env bash
# Configure per-provider Temporal activity queue RPS limits and fairness weights.
# Idempotent: safe to re-run after deploys or Temporal restarts.

set -euo pipefail

TEMPORAL_ADDRESS="${TEMPORAL_ADDRESS:-localhost:7233}"
TEMPORAL_NAMESPACE="${TEMPORAL_NAMESPACE:-echora}"
TEMPORAL_BIN="${TEMPORAL_BIN:-temporal}"

configure_provider_queue() {
  local queue="$1"
  local rps="$2"
  local reason="$3"

  "$TEMPORAL_BIN" task-queue config set \
    --address "$TEMPORAL_ADDRESS" \
    --namespace "$TEMPORAL_NAMESPACE" \
    --task-queue "$queue" \
    --task-queue-type activity \
    --queue-rps-limit "$rps" \
    --queue-rps-limit-reason "$reason" \
    --fairness-key-weight "sweep=1.0" \
    --fairness-key-weight "seed=1.0"
}

configure_provider_queue "provider-anidb"         "0.00174" "AniDB daily budget: 150 requests/day"
configure_provider_queue "provider-mal"           "0.10"    "MAL: 1 req / 3s across ~3 requests per page"
configure_provider_queue "provider-anisearch"     "0.16"    "AniSearch: 1 req / 3s across ~2 requests per page"
configure_provider_queue "provider-anime_planet"  "0.33"    "Anime-Planet: 1 req / 1.5s across ~2 requests per page"
configure_provider_queue "provider-anilist"       "0.50"    "AniList: GraphQL pacing well below 90 req/min limit"
configure_provider_queue "provider-animeschedule" "1.0"     "AnimeSchedule: REST API pacing"
configure_provider_queue "provider-kitsu"         "2.0"     "Kitsu: REST API with 0.1s pagination delay"
configure_provider_queue "provider-syoboi"        "1.0"     "Syoboi: JSON API pacing"

echo "Configured all 8 provider-* queues in namespace $TEMPORAL_NAMESPACE at $TEMPORAL_ADDRESS."
