---
title: Update Service
date: 2026-02-19
tags:
  - services
  - update-service
  - temporal
status: planned
related:
  - "[[temporal_infrastructure]]"
  - "[[event_driven_architecture]]"
  - "[[event_schema_specification]]"
  - "[[Architecture Index]]"
---

# Update Service

## Overview

The Update Service is a set of **Temporal workflows**, written in Rust with the Temporal Rust SDK, that keep every work, episode and character current. A workflow decides which pages are due and runs every step as a Temporal activity, so each refresh leaves a full history in Temporal and resumes where it stopped after a crash.

Works, episodes and characters are separate entities. Each has its own provider pages, its own refresh workflow and its own events, and all three refresh the same way.

The workflows own the order of the steps, and nothing else. Every step is a gRPC call to the service that owns it:

- **Enrichment Service** for anything enrichment does: fetching a page, and mapping and merging an entity's pages.
- **Agent Service** for anything AI does, such as writing a synopsis.
- **PostgreSQL Service** for storing and loading data. It never calls another service itself.

Temporal retries a failed activity, so every gRPC method an activity calls must be safe to run twice. `StorePage` ignores a fetch older than the one it holds, and every save checks the entity's version and writes nothing when the entity did not change.

Every change from a refresh applies automatically for now. Sending risky changes to review is planned, once what counts as risky is decided.

> [!note] Why Temporal
> The service runs thousands of refreshes and episode timers, each firing at its own time, must resume after a crash without repeating finished work, and keeps a history of every run. See [[temporal_infrastructure|Temporal Infrastructure]].

---

## How a Refresh Runs

> **Linear:** ECHO-111, ECHO-112

Each entity has one refresh workflow with a fixed ID:

| Entity | Workflow | ID |
|---|---|---|
| Work | `RefreshAnimeWorkflow` | `refresh-anime-<anime_id>` |
| Episode | `RefreshEpisodeWorkflow` | `refresh-episode-<episode_id>` |
| Character | `RefreshCharacterWorkflow` | `refresh-character-<character_id>` |

A work's refresh runs these activities:

```
for each source page of the work that is due, on that provider's task queue:
  refresh_page(page_key)
    EnrichmentService.FetchPage      fetch the page's responses, as received
    PostgresService.StorePage        store them with fetched_at; an older fetch is ignored
    returns                          the page key, and whether the page changed

if any page changed, the work's pages were relinked, or a merge was requested:
  merge_work(anime_id)
    PostgresService.LoadWork         pages, corrections, AI text, known episodes and
                                     characters, and the work's version
    EnrichmentService.MergeWork      map the pages, merge them, apply corrections and AI text;
                                     known episodes and characters keep their IDs, new ones
                                     get a UUIDv7
    PostgresService.SaveWork         save if the version still matches; the same transaction
                                     creates new episodes and characters and writes the outbox rows
    returns                          the new episode and character IDs, and whether the
                                     provider synopses changed

then, through start_workflows on the update-service queue:
  start RefreshEpisodeWorkflow for each new episode
  start RefreshCharacterWorkflow for each new character
  if the provider synopses changed: start SynopsisWorkflow

before ending:
  if merge_requested arrived during the merge, merge again
```

An episode's or a character's refresh is the same with its own pages: `refresh_page` for each due page, then `LoadEpisode`, `MergeEpisode`, `SaveEpisode`, or `LoadCharacter`, `MergeCharacter`, `SaveCharacter`.

A partial-source page is linked to a work but never merged into it, so a change to it starts no merge.

### Starting other workflows

> **Linear:** ECHO-111

Workflows never start other workflows as child workflows. A `start_workflows` activity starts refreshes, tracked episodes and synopses as top-level workflows through the Temporal client, a batch of IDs per call, each with the `USE_EXISTING` conflict policy:

- **`USE_EXISTING` needs the client.** Child workflows do not support a workflow ID conflict policy, so a second start for an entity already refreshing would fail instead of joining the running refresh.
- **Nothing ends with its starter.** A child workflow's default parent close policy terminates it when the parent ends or continues as new, which would end a sweep's or a seed load's refreshes in flight. A workflow started through the client has no parent.
- **No child limit.** A workflow can have at most 2,000 child workflows open, and each one adds events to its history. A long series such as One Piece lists more than 1,100 episodes. Starts made inside an activity add only that activity's input and result.

The activity is safe to retry, since a start repeated with `USE_EXISTING` returns the workflow already running.

### Only references pass through Temporal

Temporal keeps every activity's input and result in the workflow's history. It warns when a single payload passes 512 KB, fails at 2 MB, and stops a workflow whose history reaches 50 MB. A raw provider page, such as the character list of a long series, can pass those sizes.

So the large data never leaves an activity. `refresh_page` fetches and stores a page in one activity, and `merge_work` loads, merges and saves in another. Each returns only keys, IDs and flags. This is Temporal's claim check pattern, with PostgreSQL as the store the references point to.

When a MAL page shows a new episode, the refresh fetches that one page and merges the work again from what PostgreSQL already holds. No other provider is fetched, and nothing runs again from scratch.

The same workflow serves two other cases:

- **First enrichment of a new work.** Every page is due, so every page is fetched, then the work is merged once, and each episode and character it lists starts its own first refresh.
- **A page moving between works**, after a seed release or a review answer. The links change in PostgreSQL, and each affected work is merged again from its stored pages. Only a page new to the catalogue is fetched.

### One door per entity

> **Linear:** ECHO-111

Only an entity's own refresh workflow merges and saves it. Temporal allows at most one open run per workflow ID, so two merges of one entity never overlap.

- **Starting.** Refreshes start with the `USE_EXISTING` conflict policy. A second start for an entity already refreshing gets the running refresh back instead of an error, so fifty users opening the same old work start one refresh, not fifty.
- **Asking for another merge.** A workflow that changes a work's inputs without fetching, such as `SynopsisWorkflow` after saving new text, sends the work's refresh a `merge_requested` signal with signal-with-start. If no refresh is running, the signal starts one. If one is running, the signal handler only sets a flag. After each merge, the workflow checks the flag and merges again while it is set, and checks it once more before it ends. A signal that arrives while the workflow is completing makes the server reject the completion and run the workflow again with the signal, so the final check sees it and no request is lost.
- **The version check.** Every load returns the entity's version, and every save writes only if the version still matches. A save that finds another change in between returns `ABORTED`, and the merge activity loads, merges and saves again within the same attempt. This catches anything outside the workflow, such as a seed load relinking pages, a new AI-written synopsis or a new correction, since each of these increases the version.

### Provider rate limits

> **Linear:** ECHO-113, ECHO-119

Every provider has its own Temporal task queue for `refresh_page`, and each queue has a rate limit on activities per second, set on the queue itself with `temporal task-queue config set --queue-rps-limit` (see [[temporal_infrastructure#Rate limits|Temporal Infrastructure]]). Temporal enforces that limit on the server, across every worker polling the queue, so adding workers never raises a provider's request rate. A limit set on the queue takes precedence over the workers' own `max_task_queue_activities_per_second`, so differently configured workers cannot fight over it.

All triggers share these queues, so their order is set with Temporal's task queue priority and fairness:

| Work | Priority key | Fairness key |
|---|---|---|
| Opened by a user | 1 | `user` |
| Episode tracking at air time | 2 | `episode` |
| Weekly sweep, daily scores | 3 | `sweep` |
| Seed load | 3 | `seed` |

A lower priority key always goes first. Within priority 3, the keys `sweep` and `seed` both have the weight 1.0, so when both have work waiting, each gets half of every provider's rate. A 39,000-work seed load therefore does not stop the sweep, and the sweep does not stop the seed load. The weights are set where each workflow starts, and can be changed without a deploy with `temporal task-queue config set --fairness-key-weight`.

> [!warning] AniDB's daily limit is not handled yet
> AniDB bans on daily request volume, about 200 requests a day in earlier runs. The design assumes no daily limit for now. A daily budget for AniDB, or a longer refresh interval for AniDB pages, is decided later.

### Activity timeouts

> **Linear:** ECHO-111

Starting values, tuned once real runs are measured:

| Activity | Start-to-close | Heartbeat | Retries |
|---|---|---|---|
| `refresh_page` | 5 minutes (browser-crawled pages are slow) | 1 minute | Exponential backoff, at most 5 attempts |
| `merge_work`, `merge_episode`, `merge_character` | 2 minutes | — | At most 5 attempts; `ABORTED` is retried inside the attempt |
| `start_workflows` | 1 minute | — | Exponential backoff, at most 5 attempts |
| `write_synopsis` | 10 minutes | 1 minute | 3 attempts |

- **Heartbeats during a gRPC call.** `refresh_page` and `write_synopsis` each wait on one long gRPC call. While it runs, the activity records a heartbeat every 20 seconds and watches `ctx.cancelled()`, with `tokio::select!` over the call, a heartbeat interval and the cancellation. Without this loop, Temporal would time the activity out after the 1-minute heartbeat timeout. With it, a crashed worker is noticed within a minute instead of after 5 or 10, and a cancelled workflow stops the call.
- **`ABORTED` inside the merge.** A Temporal retry policy has one attempt limit for every error, so a version conflict cannot get its own limit there. The merge activity therefore loads, merges and saves again itself when `SaveWork` returns `ABORTED`, up to 10 times in one attempt. Conflicts last only as long as the other change, so this ends quickly, and the policy's 5 attempts count only other failures.
- **Permanent errors are not retried.** An activity returns `ApplicationFailure::non_retryable` for a gRPC status that a retry cannot fix: `INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION`, `PERMISSION_DENIED` and `UNIMPLEMENTED`. `UNAVAILABLE`, `DEADLINE_EXCEEDED`, `RESOURCE_EXHAUSTED` and `INTERNAL` stay retryable.

---

## What Triggers a Refresh

A work is **older** when it first aired more than `OLDER_AFTER_YEARS` (30) years ago. An older work, and its episodes and characters, refresh only when a user opens their page. A work with no known year counts as not older. A character counts as older only when every work it appears in is older.

| Trigger | Entities | When |
|---|---|---|
| Weekly sweep | Works, episodes and characters that are not older | Not refreshed for `REFRESH_AFTER_DAYS` (7) |
| Opened by a user | Any work, episode or character | The user opens its page and it was not refreshed for `REFRESH_AFTER_DAYS` or more |
| Episode tracking | Episodes of ongoing and upcoming works | At each episode's expected air time, then 1, 3 and 7 days after it aired |
| Daily scores | Ongoing works and works finished in the last 90 days | Once a day, the pages that report scores (MAL, AniList, Kitsu) |

An entity a user opens is served from PostgreSQL at once, and its refresh runs behind it with priority 1. The refreshed data shows on the next view.

### Weekly sweep

> **Linear:** ECHO-117

One scheduled `SweepWorkflow` covers the whole catalogue, instead of one schedule per entity. For each kind, it asks the PostgreSQL Service for the entities due (`ListDueWorks`, `ListDueEpisodes`, `ListDueCharacters`), a page at a time, and starts a refresh for each.

- **Fixed cutoff.** A sweep reads every entity not refreshed before the time it started, paging by `(refreshed_at, id)`. An entity whose refresh fails is not picked again in the same sweep.
- **Bounded.** The sweep starts a page of refreshes with `start_workflows`, each tagged with the search attribute `RefreshTrigger = 'sweep'`. Before the next page, it waits until fewer than `SWEEP_MAX_PENDING_STARTS` (500) of those are still running, counted with a visibility count query, so the queues never hold the whole catalogue at once. It continues as new between pages, so its own history stays small.
- **One sweep at a time.** The schedules use the `Skip` overlap policy, Temporal's default, set explicitly. A sweep still running when the next one is due is not started a second time, and the next sweep picks up whatever is then due.

The daily score refresh runs the same way, paging through `ListScoreWorks`: works that are ongoing, or finished within the lookback. A finished work without an end date is left to the weekly sweep.

### Episode tracking

> **Linear:** ECHO-118

An episode has little information before it airs, so it is not refreshed until then. Ongoing and upcoming works keep their episodes watched by their status and their episodes' expected air times:

1. **The week it airs.** A daily `EpisodeTrackingWorkflow` asks for every episode expected in the next `EPISODE_WINDOW_DAYS` (7) days (`ListUpcomingEpisodes`) and starts a `track-episode-<anime_id>-<episode_number>` workflow for each, with `USE_EXISTING`.
2. **At air time.** The tracking workflow sleeps on a durable Temporal timer until the expected air time. It then refreshes the work's pages that report episodes (MAL, AniList), which confirms the episode and creates it if the work did not list it yet, and refreshes the episode itself.
3. **Not aired yet.** When the episode is not there, it waits `EPISODE_RETRY_WAIT_HOURS` (2) and tries again, up to `EPISODE_MAX_RETRIES` (3) times, then raises an alert and ends. The next daily run picks the episode up again with its new expected time.
4. **After airing.** Providers fill in titles, synopses and scores over the following days, so it refreshes the episode again 1, 3 and 7 days after it aired, then ends.

When the expected air time changes, as with a delay or a hiatus, the tracking workflow sees the new time when it wakes and sleeps again until then. When a work's status becomes finished, it has no upcoming episodes, so no new tracking starts. A long-running series such as One Piece is never assumed finished.

---

## AI-Written Synopsis

> **Linear:** ECHO-96, ECHO-115

Echora never shows a provider's synopsis. A work's synopsis is written by the Agent Service from every provider's synopsis and a web search, so the text is Echora's own. Until the Agent Service exists, works show no synopsis.

Provider synopses are still fetched and kept in the stored pages, as the input for writing.

### When it is written

A synopsis mostly changes before a work airs. An announced work has a line or two at most, and providers fill in a full synopsis around the first episodes. After that it rarely changes, so a work needs one or two writes over its life.

- **Started by a merge.** `merge_work` reports when the provider synopses differ from the ones the current synopsis was written from, and the work's refresh then starts a `SynopsisWorkflow`. The saved synopsis keeps a hash of its inputs, so a refresh that changes no provider synopsis calls no AI.
- **Only with enough to go on.** Below a minimum amount of provider text, nothing is written, and the next change tries again.
- **One run per work.** The workflow ID is `synopsis-anime-<anime_id>`, with the `USE_EXISTING` conflict policy.

```
SynopsisWorkflow
  write_synopsis, on the agent task queue:
    PostgresService.LoadSynopsisSources   provider synopses, title, type, year, status
    AgentService.WriteSynopsis            web search, then write the synopsis
    PostgresService.SaveGeneratedText     text, sources, model, prompt version, input hash
  signal-with-start refresh-anime-<anime_id> with merge_requested
                                          the work's own refresh merges and saves it,
                                          and SaveWork sends anime.updated
```

### What a work shows

`MergeWork` sets the synopsis from, in order:

1. an approved correction of the field,
2. the latest AI-written synopsis,
3. nothing.

The merge never falls back to a provider's synopsis.

### Rules for the written text

- **Written, not copied.** The Agent Service rejects text too close to any provider synopsis or fetched web page, and writes again.
- **Sources kept.** The web pages used are stored with the text.
- **Rate limited.** The agent task queue has its own activity rate limit, so a seed load does not exceed the model's or the search API's limits.

### Still open

- Whether search may embed provider synopses internally while a work has no written one, or embeds no synopsis until then.
- Whether every written synopsis goes to the approval queue, a sample does, or none.
- How close to a source counts as copied, and the minimum amount of provider text.
- How `merge_work` reports that the provider synopses changed: no field in `MergeWorkResponse` or `SaveWorkResponse` carries it yet. Decided in ECHO-105.

Character attributes are planned the same way: the Agent Service extracts typed values from provider descriptions and web results, a disagreement it cannot settle becomes an approval item, and a value is extracted again only when its sources change.

---

## Providers Fetched

`EnrichmentService.FetchPage` uses the fetchers in `libs/enrichment`, one page at a time, and returns the responses as received. The mappers run later, inside each merge.

| Provider | How its page is read |
|---|---|
| MAL | MAL pages, read by the MAL crawlers |
| AniList | AniList GraphQL |
| Kitsu | Kitsu REST API |
| AniDB | AniDB HTTP API |
| Anime-Planet | Anime-Planet pages |
| aniSearch | aniSearch pages |
| AnimeSchedule | AnimeSchedule API |
| Syoboi | Syoboi Calendar API, for broadcast schedules and delays; AniList when the work has no Syoboi ID |

---

## Events

The workflows publish nothing themselves. After a save commits, the PostgreSQL Service's outbox sends the entity's `created`, `updated` or `deleted` event on NATS, as a thin notice that the vector service and notifications read back from. The episode-aired and status notifications go out the same way. See [[event_driven_architecture|Event-Driven Architecture]] and [[event_schema_specification|Event Schema Specification]].

---

## Temporal Worker Configuration

> **Linear:** ECHO-95

| Setting | Value |
|---|---|
| Task queues | `update-service` for workflows, merges and `start_workflows`, one `provider-<name>` queue per provider for `refresh_page`, `agent` for AI activities |
| Workers | A Rust SDK worker polls one task queue, so `crates/update_service` runs one worker per queue in one process, sharing one client. Each registers only what its queue runs: the `update-service` worker the workflows and the merge and start activities, each provider worker `refresh_page`, and the `agent` worker `write_synopsis`. Every replica runs the same set, so all workers polling a queue register the same types. |
| Schedules | Weekly sweep, daily scores and daily episode tracking, each with the `Skip` overlap policy |
| Search attribute | `RefreshTrigger` (keyword: `user`, `episode`, `sweep`, `seed`), set on each refresh start, used to count the refreshes a sweep or seed load still has running |
| Workflow types | `RefreshAnimeWorkflow`, `RefreshEpisodeWorkflow`, `RefreshCharacterWorkflow`, `SweepWorkflow` (weekly sweep, daily scores), `EpisodeTrackingWorkflow` (daily) with one `track-episode` workflow per episode, `SynopsisWorkflow`, `SeedReleaseWorkflow` (loads a cleaned seed, see [[ingestion_pipeline|Ingestion Pipeline]]) |
| Workflow IDs | `refresh-anime-<id>`, `refresh-episode-<id>`, `refresh-character-<id>`, `track-episode-<anime_id>-<episode_number>`, `synopsis-anime-<id>`, all with conflict policy `USE_EXISTING`; `seed-release` with conflict policy `FAIL`, so a second seed load is refused while one runs |
| Rate limit | A queue rate limit per provider queue, set with `temporal task-queue config set` from that provider's pacing |
| Priority and fairness | Priority keys 1 to 3 and fairness keys as in [[#Provider rate limits]] |
| Temporal namespace | `echora` |

The workflows and activities live in `crates/temporal_workflows/`, and `crates/update_service/` runs them as a Temporal worker. See [[temporal_infrastructure|Temporal Infrastructure]] for server setup, and its [[temporal_infrastructure#Writing Workflow Code|Writing Workflow Code]] section for the Temporal Developer Skill to use when building these workflows.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TEMPORAL_HOST` | `localhost:7233` | Temporal server address |
| `TEMPORAL_NAMESPACE` | `echora` | Temporal namespace |
| `REFRESH_AFTER_DAYS` | `7` | Age after which a work, episode or character counts as due, for the sweep and for opened pages |
| `OLDER_AFTER_YEARS` | `30` | Works first aired longer ago than this, with their episodes and characters, refresh only when opened |
| `EPISODE_WINDOW_DAYS` | `7` | How far ahead episode tracking starts |
| `EPISODE_FOLLOW_UP_DAYS` | `1,3,7` | Days after airing that an episode is refreshed again |
| `EPISODE_MAX_RETRIES` | `3` | Tries at air time before an alert is raised |
| `EPISODE_RETRY_WAIT_HOURS` | `2` | Wait between tries at air time |
| `SCORE_SYNC_HOUR` | `6` | UTC hour of the daily score refresh |
| `SCORE_SYNC_LOOKBACK_DAYS` | `90` | Days after a work finishes that its scores keep refreshing |
| `SWEEP_MAX_PENDING_STARTS` | `500` | Refreshes a sweep or seed load keeps pending at once |

---

## Related Documentation

- [[temporal_infrastructure|Temporal Infrastructure]] — Temporal server setup, priority and fairness settings
- [[event_driven_architecture|Event-Driven Architecture]] — The outbox and the NATS notices sent after an entity changes
- [[event_schema_specification|Event Schema Specification]] — The gRPC methods each activity calls
- [[ingestion_pipeline|Ingestion Pipeline]] — How a seed release becomes works

---

**Status**: Planned | **Last Updated**: 2026-11-15
