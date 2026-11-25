---
title: Update Service
date: 2026-11-15
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

Every change from a refresh applies automatically for now. Sending risky changes to review is planned, once what counts as risky is decided. (P-ECHO-15)

> [!note] Why Temporal
> The service runs thousands of refreshes and episode timers, each firing at its own time, must resume after a crash without repeating finished work, and keeps a history of every run. See [[temporal_infrastructure|Temporal Infrastructure]].

---

## How a Refresh Runs

Each entity has one refresh workflow with a fixed ID:

| Entity | Workflow | ID |
|---|---|---|
| Work | `RefreshAnimeWorkflow` | `refresh-anime-<anime_id>` (ECHO-111) |
| Episode | `RefreshEpisodeWorkflow` | `refresh-episode-<episode_id>` (ECHO-112) |
| Character | `RefreshCharacterWorkflow` | `refresh-character-<character_id>` (ECHO-112) |

A work's refresh runs these activities (ECHO-111):

```
list_pages(anime_id)
  PostgresService.ListEntityPages    every page of the work with its role, fetched_at,
                                     is_dead and last failure; never-fetched pages have
                                     no fetched_at, so they are due

for each source page that is due, or that a fetch_requested signal names,
on that provider's task queue, all at once:
  refresh_page(page_key, max_age)
    EnrichmentService.FetchPage      fetch the page's responses, as received; the response
                                     cache answers only with a response younger than max_age
    PostgresService.StorePage        store them with fetched_at; an older fetch is ignored;
                                     a page that failed every attempt records the failure
    returns                          the page key, and whether the page changed

merge when every page has finished, or MERGE_FIRST_WAIT_MINUTES (10) after the first
page was stored, whichever comes first, if a page changed, the work's pages were
relinked, or a merge was requested:
  merge_anime(anime_id)
    PostgresService.LoadAnime         pages, corrections, AI text, known episodes,
                                     and the work's version
    EnrichmentService.MergeAnime      map the pages, merge them, apply corrections and AI text;
                                     known episodes keep their IDs, new ones get a UUIDv7
    PostgresService.SaveAnime         save if the version still matches; the same transaction
                                     creates new episodes and writes the outbox rows
    returns                          the new episode IDs, and whether the provider
                                     synopses changed

each page that finishes after that merge sets the merge flag again; pages that finish
close together are merged once

then, through start_entity_workflows on the update-service queue:
  start RefreshEpisodeWorkflow for each new or changed episode, after each merge
  start WriteSynopsisWorkflow once every page has finished or failed, if any merge of this
  refresh reported changed provider synopses

before ending, while merge_requested or fetch_requested is set: fetch and merge again

record_refresh(anime_id, outcome)
  PostgresService.RecordRefresh      sets next_refresh_at: REFRESH_AFTER_DAYS ahead on success,
                                     changed or not; the first failed page's retry time when
                                     pages failed; a backoff when the refresh failed
```

An episode's or a character's refresh is the same with its own pages: `ListEntityPages`, `refresh_page` for each due page, `LoadEpisode`, `MergeEpisode`, `SaveEpisode` or `LoadCharacter`, `MergeCharacter`, `SaveCharacter`, then `RecordRefresh`. (ECHO-112)

~~An episode's load also returns its list entries, what each provider's work page says about it, since only MAL and aniSearch have a page per episode. (ECHO-103)~~

The episode merges its own pages and those entries together, so an episode with no page of its own still merges. (ECHO-109)

- **Due pages.** A page is due when it was never fetched, or was fetched longer than `REFRESH_AFTER_DAYS` ago, and is not dead. A page whose fetches keep failing waits `2^failures` hours, at most a week, before it is tried again, and `RecordRefresh` makes the work due again at that time, so a failed page is not left until the next weekly sweep. (ECHO-111)
- **Requested pages.** A page a `fetch_requested` signal names is fetched when its `fetched_at` is older than the request's `max_age`, whatever `REFRESH_AFTER_DAYS` says. A `max_age` of 0 always fetches. (ECHO-111)
- **The synopsis waits for the pages.** An early merge may hold only one or two fast providers, and the next merge, minutes later, would change the provider synopses again. `WriteSynopsisWorkflow` therefore starts once every page of the refresh has finished or failed, so one refresh asks for at most one synopsis. Episode and character refreshes start after each merge, since a repeated start joins the one already running. (ECHO-111)
- **The first merge does not wait for the slowest provider.** Every provider's page is valid input on its own, since every mapper sets a title, a type and a status, so the work can be saved with the pages that are in. A page that is still waiting, such as an AniDB page behind AniDB's slow queue, merges when it arrives, while the workflow is still open. Most works still merge once, since the fast providers arrive well within the wait. (ECHO-111)
- **Requested merges.** A change made in the database, such as an approved correction, a page move or new AI-written text, needs a merge even when nothing is refreshing the entity.
  - ~~The change sets `merge_requested_at`, and the save that follows clears it. (ECHO-98)~~
  - A pass every 15 minutes finds the entities still marked, so the change is merged even when the `merge_requested` signal was lost. (ECHO-117)
- **Merge version.** A refresh also merges, without fetching, when the entity was last merged by older mapping or merge code (`merge_version`), so a mapping fix reaches every entity, including those the sweep otherwise leaves alone. (ECHO-111)
- **A page the provider says is gone stays.**
  - Seed releases never remove pages. (ECHO-107)
  - `FetchPage` reports the provider's own "not found" as `is_not_found`. (ECHO-99)
  - ~~`StorePage` marks the page missing, keeps its last data in the merge, and after two answers a day apart sends it for review as a `dead_page`. With the provider's `DEAD_PAGE_REVIEW` setting turned off, once reviews keep confirming, it marks the page dead directly. (ECHO-97)~~
  - Otherwise only an approval marks it dead. (P-ECHO-15)
- **A failed page does not fail the refresh.** `refresh_page` records the failure through `StorePage` and returns, and the merge uses the pages that were stored. Only a failed merge or save fails the refresh, which `RecordRefresh` then records. (ECHO-111)

A partial-source page is linked to a work but never merged into it, so a change to it starts no merge. (ECHO-111)

A refresh fetches only a work's source pages, so a partial-source page that no work holds as a source is never fetched, and cannot be found dead. Whether such pages are fetched, and by which workflow, is still to be decided. (ECHO-111)

### Starting other workflows

> **Linear:** ECHO-111

Workflows never start other workflows as child workflows. A `start_entity_workflows` activity starts refreshes, tracked episodes and synopses as top-level workflows through the Temporal client, a batch of IDs per call, each with the `USE_EXISTING` conflict policy:

- **`USE_EXISTING` needs the client.** Child workflows do not support a workflow ID conflict policy, so a second start for an entity already refreshing would fail instead of joining the running refresh.
- **Nothing ends with its starter.** A child workflow's default parent close policy terminates it when the parent ends or continues as new, which would end a sweep's or a seed load's refreshes in flight. A workflow started through the client has no parent.
- **No child limit.** A workflow can have at most 2,000 child workflows open, and each one adds events to its history. A long series such as One Piece lists more than 1,100 episodes. Starts made inside an activity add only that activity's input and result.

The activity is safe to retry, since a start repeated with `USE_EXISTING` returns the workflow already running. A refresh started by another workflow carries its starter's `RefreshTrigger` and priority, so a seed load's episode refreshes count against the seed load's limit, and a user's request stays first.

### Only references pass through Temporal

Temporal keeps every activity's input and result in the workflow's history. It warns when a single payload passes 512 KB, fails at 2 MB, and stops a workflow whose history reaches 50 MB. A raw provider page, such as the character list of a long series, can pass those sizes.

So the large data never leaves an activity. `refresh_page` fetches and stores a page in one activity, and `merge_anime` loads, merges and saves in another. Each returns only keys, IDs and flags. This is Temporal's claim check pattern, with PostgreSQL as the store the references point to. (ECHO-111)

Inside an activity the pages still travel over gRPC, and they are large. The saved One Piece pages are 2.7 MB from AniDB and 1.8 MB from aniSearch's episode list, among others, past gRPC's 4 MB default for one message. A merge needs every page in memory at once, so streaming one page per message would add code without lowering memory. So each side sets its message limit explicitly to 64 MB, uses gRPC's gzip compression, which shrinks HTML and XML several times over, and records the largest message size it handles as a metric:

- ~~the PostgreSQL Service (ECHO-97),~~
- the Enrichment Service (ECHO-99),
- the Update Service's clients (ECHO-111).

When a MAL page shows a new episode, the refresh fetches that one page and merges the work again from what PostgreSQL already holds. No other provider is fetched, and nothing runs again from scratch. (ECHO-111)

The same workflow serves two other cases:

- **First enrichment of a new work.** Every page is due, so every page is fetched, then the work is merged once, and each episode it lists starts its own first refresh. Its characters come from the character list and refresh on their own. (ECHO-111) (ECHO-135)
- **A page moving between works**, after a seed release or a review answer.
  - The links change in PostgreSQL. (ECHO-107)
  - Each affected work is merged again from its stored pages, and only a page new to the catalogue is fetched. (ECHO-111)

### One door per entity

> **Linear:** ECHO-111

Only an entity's own refresh workflow merges and saves it. Temporal allows at most one open run per workflow ID, so two merges of one entity never overlap.

- **Starting.** Refreshes start with the `USE_EXISTING` conflict policy. A second start for an entity already refreshing gets the running refresh back instead of an error, so fifty users opening the same old work start one refresh, not fifty.
- **Asking for specific pages.** A trigger that needs pages that are not due, such as episode tracking at air time or the daily score refresh, sends a `fetch_requested` signal with signal-with-start, naming the providers, a `max_age` and its priority. It starts the refresh when none is running, and a running refresh adds the pages to its work instead of dropping the request, as a plain `USE_EXISTING` start would. Later activities run at the highest priority the refresh has received, so a user's request is not stuck behind a seed load's.
- **Asking for another merge.** A workflow that changes a work's inputs without fetching, such as `WriteSynopsisWorkflow` after saving new text, sends the work's refresh a `merge_requested` signal with signal-with-start. If no refresh is running, the signal starts one. If one is running, the signal handler only sets a flag. After each merge, the workflow checks the flag and merges again while it is set, and checks it once more before it ends. A signal that arrives while the workflow is completing makes the server reject the completion and run the workflow again with the signal, so the final check sees it and no request is lost.
- **The version check.** Every load returns the entity's version, and every save writes only if the version still matches. A save that finds another change in between returns `ABORTED`, and the merge activity loads, merges and saves again within the same attempt. This catches anything outside the workflow, such as a seed load relinking pages, a new AI-written synopsis or a new correction, since each of these increases the version.

### Provider rate limits

> **Linear:** ECHO-113

Every provider has its own Temporal task queue for `refresh_page`, and each queue has a rate limit on activities per second, set on the queue itself with `temporal task-queue config set --queue-rps-limit` (see [[temporal_infrastructure#Rate limits|Temporal Infrastructure]]). Temporal enforces that limit on the server, across every worker polling the queue, so adding workers never raises a provider's request rate. A limit set on the queue takes precedence over the workers' own `max_task_queue_activities_per_second`, so differently configured workers cannot fight over it.

All triggers share these queues, so their order is set with Temporal's task queue priority and fairness:

| Work | Priority key | Fairness key |
|---|---|---|
| Opened by a user | 1 | `user` |
| Episode tracking at air time | 2 | `episode` |
| Requested merges (approvals, page moves, new text) | 2 | `review` |
| Weekly sweep, daily scores | 3 | `sweep` |
| Seed load | 3 | `seed` |
| Re-merge after a merge-version raise | 4 | `remerge` |

A lower priority key always goes first. Within priority 3, the keys `sweep` and `seed` both have the weight 1.0, so when both have work waiting, each gets half of every provider's rate. A 39,000-work seed load therefore does not stop the sweep, and the sweep does not stop the seed load. The weights are set where each workflow starts, and can be changed without a deploy with `temporal task-queue config set --fairness-key-weight`.

### Providers with a daily budget

AniDB bans on daily request volume, about 200 requests a day in earlier runs, so it gets a budget of 150 a day. Its 16,370 work pages then take about 110 days to go round once, and a weekly cycle would add about 2,300 a day to a queue that drains 150. Work refreshes would stay open for months waiting on AniDB, and the sweep's limit of running refreshes would fill with them. (ECHO-119)

So budgeted providers are fetched apart from work refreshes:

- **Refreshes skip them.** A work's refresh fetches every due page except those of budgeted providers (`BUDGETED_PROVIDERS`, `anidb`), and merges with what the store already holds for them. (ECHO-117)
- **One budgeted fetcher per provider.** A long-running `FetchBudgetedPagesWorkflow` (`budgeted-fetch-anidb`) takes pages in due order from `ListDuePages`, fetches them through `refresh_page` on the provider's queue, and sends the page's work `merge_requested` when a page changed. It continues as new between batches. (ECHO-119)
- **An even pace, not a burst.** The `provider-anidb` queue's rate limit is 150 a day, one request about every 10 minutes. Pages beyond the budget wait in the queue for the next free slot, and a day of bursts is what gets an IP banned. (ECHO-113)
- **Visits first.**
  - When a user opens a work whose AniDB page is due, the backend's refresh request reaches the fetcher as a `page_requested` signal. (P-ECHO-16)
  - The fetcher moves that page to the front of its queue. (ECHO-119)

In practice each AniDB page refreshes about every 110 days, which is what the budget allows. Another provider that gets a daily cap joins the same way. (ECHO-119)

### One fetch process per provider

> **Linear:** ECHO-113, ECHO-99

Request pacing in `libs/enrichment` (`polite_http.py`) holds requests to a site apart across one process, not across processes, and a queue's rate limit counts pages, while one MAL page can be several requests. So each provider's pages are fetched by exactly one Enrichment process, and the queue's limit is set from that provider's pace divided by its requests per page. Merging holds no provider pacing, so merge processes scale freely. Fetching and merging run as two deployments of the Enrichment image: merges are CPU-heavy and would otherwise block the fetches' event loop, stalling heartbeats and health checks. The workers for browser-crawled providers (`provider-mal`, `provider-anime-planet`, `provider-anisearch`) cap how many activities they run at once, and the Enrichment Service sets its browser pool limit at startup, so slow crawls cannot pile up browsers until the process runs out of memory.

### Activity timeouts

Starting values, tuned once real runs are measured:

| Activity | Start-to-close | Heartbeat | Retries |
|---|---|---|---|
| `refresh_page` | 5 minutes (browser-crawled pages are slow) | 1 minute | Exponential backoff, at most 5 attempts (ECHO-111) |
| `list_pages`, `record_refresh` | 30 seconds | — | Exponential backoff, at most 5 attempts (ECHO-111) |
| `merge_anime` | 2 minutes | — | At most 5 attempts; `ABORTED` is retried inside the attempt (ECHO-111) |
| `merge_episode`, `merge_character` | 2 minutes | — | At most 5 attempts; `ABORTED` is retried inside the attempt (ECHO-112) |
| `start_entity_workflows` | 1 minute | — | Exponential backoff, at most 5 attempts (ECHO-111) |
| `write_synopsis` | 10 minutes | 1 minute | 3 attempts (ECHO-115) |

- **Heartbeats during a gRPC call.** `refresh_page` and `write_synopsis` each wait on one long gRPC call. While it runs, the activity records a heartbeat every 20 seconds and watches `ctx.cancelled()`, with `tokio::select!` over the call, a heartbeat interval and the cancellation. Without this loop, Temporal would time the activity out after the 1-minute heartbeat timeout. With it, a crashed worker is noticed within a minute instead of after 5 or 10, and a cancelled workflow stops the call.
  - `refresh_page` runs this loop. (ECHO-111)
  - `write_synopsis` runs this loop. (ECHO-115)
- **`ABORTED` inside the merge.** A Temporal retry policy has one attempt limit for every error, so a version conflict cannot get its own limit there. The merge activity therefore loads, merges and saves again itself when `SaveAnime` returns `ABORTED`, up to 10 times in one attempt. Conflicts last only as long as the other change, so this ends quickly, and the policy's 5 attempts count only other failures. (ECHO-111)
- **Bans fail fast.** After a block, `polite_http` sends nothing more to that site from the process. A fetch then fails at once without a request, and is recorded as a page failure with backoff, so a ban is never pressed further. Workers also set their own `max_task_queue_activities_per_second` from the provider's pace, which applies when a queue has no limit set, such as in a new environment before the limit script ran. (ECHO-111)
- **Permanent errors are not retried.** An activity returns `ApplicationFailure::non_retryable` for a gRPC status that a retry cannot fix: `INVALID_ARGUMENT`, `NOT_FOUND`, `FAILED_PRECONDITION`, `PERMISSION_DENIED`, `UNIMPLEMENTED`, and `RESOURCE_EXHAUSTED`, which our services return only for a message over the 64 MB limit. `UNAVAILABLE`, `DEADLINE_EXCEEDED` and `INTERNAL` stay retryable, and a provider's throttling arrives as `UNAVAILABLE`. (ECHO-111)

---

## What Triggers a Refresh

A work is **older** when it first aired more than `ON_DEMAND_REFRESH_AGE_YEARS` (30) years ago. An older work, and its episodes and characters, refresh only when a user opens their page. A work with no known year counts as not older. A character counts as older only when every work it appears in is older. (ECHO-117)

| Trigger | Entities | When |
|---|---|---|
| Weekly sweep | Works that are not older, with every provider's statistics; episodes and characters of ongoing and upcoming works | Due under `next_refresh_at`, `REFRESH_AFTER_DAYS` (7) after the last success (ECHO-117) |
| ~~Work's merge~~ | ~~Episodes of finished works~~ | ~~The work's weekly merge changed what it says about them: a changed list entry (ECHO-103)~~ |
| Character load | New characters | `CreateCharacter` created them from the character list (ECHO-135) |
| Character discovery | Known characters a new or refreshed work lists | The work's character list names a character whose appearances do not include that work yet; its refresh gets `fetch_requested` (ECHO-135) |
| Opened by a user | Any work, episode or character | The user opens its page and it is due. Bots never start a refresh, and the backend caps refresh starts per minute (P-ECHO-16) |
| Episode tracking | Episodes of ongoing and upcoming works | At each episode's expected air time, then 1, 3 and 7 days after it aired (ECHO-118) |
| Daily scores | Ongoing works and works finished in the last 90 days | Once a day, the pages that report scores (MAL, AniList, Kitsu) (ECHO-117) |

An entity a user opens is served from PostgreSQL at once, and its refresh runs behind it with priority 1. The refreshed data shows on the next view. (P-ECHO-16)

### Weekly sweep

> **Linear:** ECHO-117

One scheduled `SweepCatalogueWorkflow` covers the whole catalogue, instead of one schedule per entity. For each kind, it asks the PostgreSQL Service for the entities due (`ListDueAnime`, `ListDueEpisodes`, `ListDueCharacters`), a page at a time, and starts a refresh for each.

- **Fixed cutoff.** A sweep reads every entity whose `next_refresh_at` is before the time it started, paging by `(next_refresh_at, id)`. Every refresh that finishes, failed or not, moves `next_refresh_at` past that cutoff, so no entity is picked twice in one sweep.
- **Bounded.** The sweep starts a page of refreshes with `start_entity_workflows`, each tagged with the search attribute `RefreshTrigger = 'sweep'`. Before the next page, it waits until fewer than `MAX_PENDING_BULK_REFRESHES` (500) of those are still running, counted with a visibility count query, so the queues never hold the whole catalogue at once. It continues as new between pages, so its own history stays small.
- **One sweep at a time.** The schedules use the `Skip` overlap policy, Temporal's default, set explicitly. A sweep still running when the next one is due is not started a second time, and the next sweep picks up whatever is then due.

Requested merges have their own schedule, every 15 minutes, since a reviewer's approval should not wait a week. It reads `ListRequestedMerges` for each kind and sends each entity's refresh `merge_requested` with signal-with-start (`USE_EXISTING`, priority 2, fairness key `review`). A caller that changed the entity usually sent the signal already, so this pass only catches signals that were lost, or changes no caller signals.

The daily score refresh runs the same way, paging through `ListScoreAnime`: works that are ongoing, or finished within the lookback. A finished work without an end date is left to the weekly sweep. It sends each work's refresh a `fetch_requested` signal for the pages that report scores (MAL, AniList, Kitsu), with a `max_age` of a day, since those pages are not due yet.

### Episode tracking

> **Linear:** ECHO-118

An episode has little information before it airs, so it is not refreshed until then. Ongoing and upcoming works keep their episodes watched by their status and their episodes' expected air times:

1. **The week it airs.** A daily `TrackEpisodesWorkflow` asks for every episode expected in the next `EPISODE_TRACKING_WINDOW_DAYS` (7) days (`ListUpcomingEpisodes`) and starts a `track-episode-<anime_id>-<episode_number>` workflow for each, with `USE_EXISTING`.
2. **At air time.** The tracking workflow sleeps on a durable Temporal timer until the expected air time. It then sends the work's refresh a `fetch_requested` signal for the pages that report episodes (MAL, AniList) with a `max_age` of 0, so the cache cannot answer, which confirms the episode and creates it if the work did not list it yet, and refreshes the episode itself the same way.
3. **Not aired yet.** The tracker learns the outcome from PostgreSQL, not from the refresh: after `EPISODE_RETRY_WAIT_HOURS` (2) it reads whether the episode is stored as aired. Once it is, the tracker calls `MarkEpisodeAired`, which writes the aired notification once, even when no save changed anything. When it is not, it asks again, up to `EPISODE_MAX_RETRIES` (3) times, then raises an alert and ends. The next daily run picks the episode up again with its new expected time.
4. **After airing.** Providers fill in titles, synopses and scores over the following days, so it refreshes the episode again 1, 3 and 7 days after it aired, then ends.

When the expected air time changes, as with a delay or a hiatus, the tracking workflow sees the new time when it wakes and sleeps again until then. When a work's status becomes finished, it has no upcoming episodes, so no new tracking starts. A long-running series such as One Piece is never assumed finished.

---

## AI-Written Synopsis

Echora never shows a provider's synopsis. A work's synopsis is written by the Agent Service from every provider's synopsis and a web search, so the text is Echora's own. Until the Agent Service exists, works show no synopsis. (P-ECHO-14)

~~Provider synopses are still fetched and kept in the stored pages, as the input for writing. (ECHO-97)~~

### When it is written

A synopsis mostly changes before a work airs. An announced work has a line or two at most, and providers fill in a full synopsis around the first episodes. After that it rarely changes, so a work needs one or two writes over its life. (P-ECHO-14)

- **Started by a merge.**
  - `merge_anime` reports when the provider synopses differ from the ones the current synopsis was written from. (ECHO-105)
  - The work's refresh then starts a `WriteSynopsisWorkflow`. (ECHO-111)
  - The saved synopsis keeps a hash of its inputs, so a refresh that changes no provider synopsis calls no AI. (ECHO-115)
- **Only with enough to go on.** Below a minimum amount of provider text, nothing is written, and the next change tries again. (ECHO-115)
- **Every attempt is remembered.** `SaveGeneratedText` records the input hash and outcome even when nothing was written, so a merge with the same provider synopses does not start the workflow again. (ECHO-115)
- **One run per work.** The workflow ID is `synopsis-anime-<anime_id>`, with the `USE_EXISTING` conflict policy. (ECHO-115)

```
WriteSynopsisWorkflow                          (ECHO-115)
  write_synopsis, on the agent-service task queue:
    PostgresService.LoadSynopsisSources   provider synopses, title, type, year, status
    AgentService.WriteSynopsis            web search, then write the synopsis
    PostgresService.SaveGeneratedText     text, sources, model, prompt version, input hash
  signal-with-start refresh-anime-<anime_id> with merge_requested
                                          the work's own refresh merges and saves it,
                                          and SaveAnime sends anime.updated
```

### What a work shows

> **Linear:** ECHO-105

`MergeAnime` sets the synopsis from, in order:

1. an approved correction of the field,
2. the latest AI-written synopsis,
3. nothing.

The merge never falls back to a provider's synopsis.

### Rules for the written text

- **Written, not copied.** The Agent Service rejects text too close to any provider synopsis or fetched web page, and writes again. (ECHO-96)
- **Sources kept.** The web pages used are stored with the text. (ECHO-115)
- **Rate limited.** The agent-service task queue has its own activity rate limit, so a seed load does not exceed the model's or the search API's limits. (ECHO-115)

### Still open

- Whether search may embed provider synopses internally while a work has no written one, or embeds no synopsis until then. (ECHO-115)
- Whether every written synopsis goes to the approval queue, a sample does, or none. (ECHO-115)
- How close to a source counts as copied. (ECHO-96)
- The minimum amount of provider text. (ECHO-115)
- How `merge_anime` reports that the provider synopses changed: no field in `MergeAnimeResponse` or `SaveAnimeResponse` carries it yet. (ECHO-105)

Character attributes are planned the same way: the Agent Service extracts typed values from provider descriptions and web results, a disagreement it cannot settle becomes an approval item, and a value is extracted again only when its sources change. (P-ECHO-19)

---

## Franchise Link Check

> **Linear:** P-ECHO-22

A scheduled `CheckFranchiseLinksWorkflow` looks for franchise links a reviewer should check, and sends them to the approval queue. What it looks for, and why no rule can decide the links on its own, is in [[Database Schema#Link check|Database Schema]].

- **When.** Weekly, and after each seed release, with the `Skip` overlap policy like the sweep, so only one check runs at a time.
- **One activity.** `check_franchise_links` runs on the `update-service` queue and makes three calls:
  1. it streams every franchise link from `PostgresService.ListFranchiseLinks`;
  2. it streams them on into `EnrichmentService.FindFranchiseLinkCandidates`, which runs the check in Python;
  3. it sends the candidates to `PostgresService.CreateLinkCheckApprovals`, which writes one `link_check` approval request per candidate and skips any candidate that already has an open request.
- **The links never enter Temporal.** Every franchise link in the catalogue is too much for one payload, and Temporal keeps every activity input and result in the workflow's history. So the links move between the two services inside the one activity, which returns only the number of items it created, as `refresh_page` does with pages (see [[#Only references pass through Temporal]]).
- **Heartbeats.** The activity records a heartbeat while its calls run, the same way `refresh_page` does, so a crashed worker is noticed within a minute.
- **Answers need no workflow of their own.** `DecideApproval` writes the reviewer's correction, crossover mark or confirmation and sets `merge_requested_at` on both works. The requested-merge schedule then merges them again, and the franchises are computed again from the corrected links.

## Providers Fetched

> **Linear:** ~~ECHO-89~~, ECHO-94, ECHO-99

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

The workflows publish nothing themselves. After a save commits, the PostgreSQL Service's outbox sends the entity's `created`, `updated` or `deleted` event on NATS, as a thin notice that the vector service and notifications read back from. The episode-aired and status notifications go out the same way. See [[event_driven_architecture|Event-Driven Architecture]] and [[event_schema_specification|Event Schema Specification]]. (ECHO-106)

---

## Temporal Worker Configuration

| Setting | Value |
|---|---|
| Task queues | `update-service` for workflows, merges and `start_entity_workflows`, one `provider-<name>` queue per provider for `refresh_page`, `agent-service` for AI activities (ECHO-95) |
| Workers | A Rust SDK worker polls one task queue, so `crates/update_service` runs one worker per queue in one process, sharing one client. Each registers only what its queue runs: the `update-service` worker the workflows and the merge and start activities, each provider worker `refresh_page`, and the `agent-service` worker `write_synopsis`. Every replica runs the same set, so all workers polling a queue register the same types. (ECHO-95) |
| Schedules | Weekly sweep and daily scores, each with the `Skip` overlap policy (ECHO-117) |
| Schedules | Daily episode tracking, with the `Skip` overlap policy (ECHO-118) |
| Search attribute | `RefreshTrigger` (keyword: `user`, `episode`, `sweep`, `seed`), set on each refresh start, used to count the refreshes a sweep or seed load still has running (ECHO-95) |
| Workflow types | `RefreshAnimeWorkflow`, `RefreshEpisodeWorkflow`, `RefreshCharacterWorkflow`, `SweepCatalogueWorkflow` (weekly sweep, daily scores), `TrackEpisodesWorkflow` (daily) with one `track-episode` workflow per episode, `WriteSynopsisWorkflow`, `LoadSeedReleaseWorkflow` (loads a cleaned seed, see [[ingestion_pipeline|Ingestion Pipeline]]), `FetchBudgetedPagesWorkflow` (one per budgeted provider) (ECHO-95) |
| Workflow IDs | `refresh-anime-<id>`, `refresh-episode-<id>`, `refresh-character-<id>`, `track-episode-<anime_id>-<episode_number>`, `synopsis-anime-<id>`, `budgeted-fetch-<provider>`, all with conflict policy `USE_EXISTING`; `seed-release` with conflict policy `FAIL`, so a second seed load is refused while one runs (ECHO-95) |
| Rate limit | A queue rate limit per provider queue, set with `temporal task-queue config set` from that provider's pacing (ECHO-113) |
| Priority and fairness | Priority keys 1 to 3 and fairness keys as in [[#Provider rate limits]] (ECHO-113) |
| Temporal namespace | `echora` (ECHO-95) |

The workflows and activities live in `crates/temporal_workflows/`, and `crates/update_service/` runs them as a Temporal worker. See [[temporal_infrastructure|Temporal Infrastructure]] for server setup, and its [[temporal_infrastructure#Writing Workflow Code|Writing Workflow Code]] section for the Temporal Developer Skill to use when building these workflows. (ECHO-95)

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `TEMPORAL_HOST` | `localhost:7233` | Temporal server address (ECHO-95) |
| `TEMPORAL_NAMESPACE` | `echora` | Temporal namespace (ECHO-95) |
| `REFRESH_AFTER_DAYS` | `7` | Age after which a work, episode or character counts as due, for the sweep and for opened pages (ECHO-111) |
| `MERGE_VERSION` | — | Version of the mapping and merge code; raised with a fix that should reach every entity (ECHO-111) |
| ~~`DEAD_PAGE_REVIEW`~~ | ~~`true` per provider~~ | ~~Missing pages go to review before they are marked dead; turned off for a provider once reviews keep confirming (ECHO-97)~~ |
| `MERGE_FIRST_WAIT_MINUTES` | `10` | How long a refresh waits after its first stored page before merging without the pages still in flight (ECHO-111) |
| `BUDGETED_PROVIDERS` | `anidb` | Providers fetched by a budgeted fetcher instead of by work refreshes (ECHO-119) |
| `ANIDB_DAILY_BUDGET` | `150` | AniDB requests a day; the `provider-anidb` queue's rate limit is set from it (ECHO-113) |
| `ON_DEMAND_REFRESH_AGE_YEARS` | `30` | Works first aired longer ago than this, with their episodes and characters, refresh only when opened (ECHO-117) |
| `EPISODE_TRACKING_WINDOW_DAYS` | `7` | How far ahead episode tracking starts (ECHO-118) |
| `EPISODE_FOLLOW_UP_DAYS` | `1,3,7` | Days after airing that an episode is refreshed again (ECHO-118) |
| `EPISODE_MAX_RETRIES` | `3` | Tries at air time before an alert is raised (ECHO-118) |
| `EPISODE_RETRY_WAIT_HOURS` | `2` | Wait between tries at air time (ECHO-118) |
| `SCORE_SYNC_HOUR` | `6` | UTC hour of the daily score refresh (ECHO-117) |
| `SCORE_SYNC_LOOKBACK_DAYS` | `90` | Days after a work finishes that its scores keep refreshing (ECHO-117) |
| `MAX_PENDING_BULK_REFRESHES` | `500` | Refreshes a sweep or seed load keeps pending at once (ECHO-117) |

---

## Related Documentation

- [[temporal_infrastructure|Temporal Infrastructure]] — Temporal server setup, priority and fairness settings
- [[event_driven_architecture|Event-Driven Architecture]] — The outbox and the NATS notices sent after an entity changes
- [[event_schema_specification|Event Schema Specification]] — The gRPC methods each activity calls
- [[ingestion_pipeline|Ingestion Pipeline]] — How a seed release becomes works

---

**Status**: Planned | **Last Updated**: 2026-11-15
