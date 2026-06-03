---
title: Syoboi Calendar Integration
date: 2026-02-26
tags:
  - services
  - update-service
  - schedule
  - external-api
  - syoboi
status: active
related:
  - "[[update_service]]"
  - "[[ingestion_pipeline]]"
  - "[[event_driven_architecture]]"
---

# Syoboi Calendar Integration

## Overview

[Syoboi Calendar](https://cal.syoboi.jp/) (しょぼいカレンダー) is a community-maintained Japanese anime TV broadcast schedule database. It is the authoritative source for per-slot broadcast timing data — the only external source in Echora's stack that provides:

| Capability | Syoboi | AniList | Jikan/MAL |
|---|---|---|---|
| Broadcast slot time (per channel) | ✓ | ✗ (single timestamp) | ✗ |
| Delay in seconds (`StOffset`) | ✓ | ✗ | ✗ |
| Schedule change warning flag (`Warn`) | ✓ | ✗ | ✗ |
| Episode subtitle per channel | ✓ | partial | partial |
| Confirmation episode actually aired | ✗ | ✓ | ✓ |

Syoboi is used exclusively for **broadcast schedule tracking**. It does not replace AniList or Jikan for episode airing confirmation — those still cover streaming and reflect actual episode availability.

---

## Coverage

| Scope | Coverage |
|---|---|
| arm.json total entries | 35,667 |
| Entries with `syobocal_tid` | 6,573 (18%) |
| Currently-airing Japanese TV anime | ~100% |
| Movies, OVAs, streaming-only | Low — use AniList fallback |

The 18% global coverage figure is misleading: the 82% without TIDs are overwhelmingly titles that never aired on Japanese terrestrial TV (OVAs, movies, streaming-only, niche catalogue). For ongoing TV anime, Syoboi coverage is near-total.

---

## ID Mapping via arm.json

Syoboi uses its own title ID (TID) space. There is no direct AniDB → TID lookup.

**arm project** ([kawaiioverflow/arm](https://github.com/kawaiioverflow/arm)) is a community-maintained mapping database:

```json
{
  "mal_id": 59978,
  "anilist_id": 182255,
  "annict_id": 14068,
  "syobocal_tid": 7629
}
```

Fields: `mal_id`, `anilist_id`, `annict_id`, `syobocal_tid` — **no AniDB ID**.

### ID Chain for Echora

```
Jikan/MAL fetch (Stage 2) → MAL ID
  └─ arm.json (keyed by mal_id) → syobocal_tid
       └─ Syoboi ProgLookup API
```

MAL ID is already available in the enrichment pipeline from the Stage 2 Jikan fetch. The arm.json lookup is one additional resolution step at the end of Stage 4.

### arm.json Usage

arm.json is a flat JSON array. Build an in-memory index at startup:

```python
import json, httpx

ARM_URL = "https://raw.githubusercontent.com/kawaiioverflow/arm/master/arm.json"

async def load_arm_index() -> dict[int, dict]:
    """Returns {mal_id: entry} index. Cache this for the session."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(ARM_URL)
        arm_data = resp.json()
    return {
        entry["mal_id"]: entry
        for entry in arm_data
        if entry.get("mal_id")
    }

def resolve_syobocal_tid(arm_index: dict, mal_id: int | None) -> int | None:
    if mal_id is None:
        return None
    entry = arm_index.get(mal_id)
    return entry.get("syobocal_tid") if entry else None
```

arm.json is refreshed once per day in production (cached in the Broadcast Schedule Monitor or as a shared service dependency). Individual enrichment runs cache it for the session.

---

## API Reference

### db.php — Stable XML API

The only API endorsed for production use. `json.php` is explicitly documented as internal and subject to breaking changes without notice — do not use it.

**Base URL**: `https://cal.syoboi.jp/db.php`

All times are **JST (UTC+9)**. Pass JST datetimes in Range and StTime parameters.

No authentication required. Include a descriptive `User-Agent` header. No official public rate limit is published; use ≤ 1 req/s for bulk operations.

---

#### TitleLookup

Retrieve title metadata for one or more TIDs.

```
GET /db.php?Command=TitleLookup&TID={tid}
```

Parameters:

| Param | Description |
|---|---|
| `TID` | Comma-separated TIDs, or `*` for all |
| `LastUpdate` | Date range filter on last modification |
| `Fields` | Comma-separated field names to return (omit for all) |

Response fields:

| Field | Description |
|---|---|
| `TID` | Syoboi title ID |
| `Title` | Japanese title |
| `ShortTitle` | Abbreviated title |
| `TitleYomi` | Phonetic reading (hiragana) |
| `Cat` | Category (1 = anime) |
| `FirstYear`, `FirstMonth` | Season start year and month |
| `FirstCh` | First broadcast channel **name** (text, not ChID) |
| `SubTitles` | Episode list: `*{count}*{subtitle}\n...` |
| `TitleFlag` | Title flags bitmask |
| `LastUpdate` | Last modified timestamp (JST) |

`FirstCh` is stored as a human-readable channel name, not a numeric ID. Resolve it to a ChID via ChLookup by matching `ChName`.

Example: Frieren S2 (TID=7629)
```xml
<TID>7629</TID>
<Title>葬送のフリーレン(第2期)</Title>
<FirstYear>2026</FirstYear>
<FirstMonth>1</FirstMonth>
<FirstCh>日本テレビ</FirstCh>
<SubTitles>
  *29*じゃあ行こうか
  *34*討伐要請
</SubTitles>
```

---

#### ProgLookup

Retrieve broadcast schedule slots. Returns up to 5,000 rows per request.

```
GET /db.php?Command=ProgLookup&TID={tid}&Range={start}-{end}&JOIN=SubTitles
```

Parameters:

| Param | Description |
|---|---|
| `TID` | Comma-separated TIDs |
| `ChID` | Comma-separated channel IDs (omit for all channels) |
| `Range` | JST datetime range: `YYYYMMDD_HHMMSS-YYYYMMDD_HHMMSS` (both required). Matches: `Range_start < EdTime AND StTime < Range_end` |
| `Count` | Comma-separated episode numbers |
| `JOIN=SubTitles` | Include `STSubTitle` from subtitle table |
| `Fields` | Limit output fields |

Response fields per `ProgItem`:

| Field | Type | Description |
|---|---|---|
| `PID` | int | Unique program slot ID (globally unique per channel per episode) |
| `TID` | int | Title ID |
| `ChID` | int | Channel ID |
| `StTime` | datetime (JST) | Broadcast start time |
| `EdTime` | datetime (JST) | Broadcast end time |
| `StOffset` | int (seconds) | Delay from `StTime` (0 = on schedule, positive = delayed) |
| `Count` | int | Episode number within the series |
| `SubTitle` | string | Episode subtitle (often empty; prefer `STSubTitle`) |
| `STSubTitle` | string | Episode subtitle from subtitle table (requires `JOIN=SubTitles`) |
| `Flag` | int | Broadcast flags bitmask |
| `Warn` | int | Warning flag: 1 = schedule uncertainty or recent change |
| `Deleted` | int | 1 if this slot was removed |
| `LastUpdate` | datetime (JST) | Last modified timestamp |
| `Revision` | int | Number of times this entry has been edited |

> [!important] One episode → multiple ProgItems
> Each episode is broadcast on multiple channels (key station + regional affiliates + satellite repeats). A single `ProgLookup` without `ChID` filter returns one row per channel per episode. Always filter by `ChID` to get the canonical primary broadcast slot.

> [!note] `StOffset` vs rescheduling
> `StOffset` reflects a live delay on the night of broadcast (the show started N seconds late). It does NOT signal a future date change. A future date change is detected by comparing `StTime` against the previously-stored expected air time.

---

#### ChLookup

Retrieve channel metadata.

```
GET /db.php?Command=ChLookup&ChID={ids}
```

Key response fields: `ChID`, `ChName`, `ChGID` (channel group).

**Channel group (ChGID) guide:**

| GID | Type | Example channels |
|---|---|---|
| 1 | Terrestrial key stations (Tokyo) | NTV, TBS, TV Asahi, Fuji TV, TV Tokyo |
| 4 | NHK | NHK General, NHK-E |
| 5 | BS broadcast | BS11, WOWOW, NHK BS |
| 6 | Satellite/Cable specialist | AT-X |
| 8, 13 | Regional affiliates | YTV (Osaka), CTV (Nagoya) |

For canonical primary broadcast, prefer GID=1 (terrestrial Tokyo key stations). Among GID=1 channels, select the one with the earliest `StTime` for the episode.

Frieren S2 broadcast example (ep 34):

| ChID | ChName | ChGID | StTime (JST) |
|---|---|---|---|
| 4 | 日本テレビ | 1 | 2026-02-27 23:00 |
| 54 | 読売テレビ | 8 | 2026-02-27 23:00 |
| 80 | 中京テレビ | 13 | 2026-02-27 23:00 |
| 20 | AT-X | 6 | 2026-02-28 21:00 |

Primary = ChID 4 (NTV, GID=1).

---

## Echora Integration

### Phase 1: Enrichment — Store `syobocal_tid`

During **Stage 4 (Cross-Source Merge)** of the ingestion pipeline:

1. MAL ID is already present from the Stage 2 Jikan fetch
2. Resolve via arm.json index: `arm_index.get(mal_id, {}).get("syobocal_tid")`
3. Store `syobocal_tid` on the anime record (nullable)
4. Resolve and store `primary_channel_id` (ChID) via TitleLookup → ChLookup

If `syobocal_tid` is null (OVA, movie, streaming-only), the Update Service falls back to AniList `nextAiringEpisode` for schedule data.

**Fields added to anime record:**

| Field | Source | Notes |
|---|---|---|
| `syobocal_tid` | arm.json | Nullable; null = no Syoboi coverage |
| `syobocal_primary_ch_id` | ChLookup | ChID of canonical broadcast channel |

---

### Phase 2: Update Service — SyoboiCalendarAdapter

`SyoboiCalendarAdapter` is added to the source adapter registry alongside `JikanAdapter` and `AniListAdapter`. It is responsible exclusively for broadcast schedule data.

```python
class SyoboiCalendarAdapter:
    BASE_URL = "https://cal.syoboi.jp/db.php"

    async def get_upcoming_schedule(
        self,
        syobocal_tid: int,
        primary_ch_id: int,
        days_ahead: int = 14,
    ) -> list[ProgItem]:
        """
        Returns ProgItems for the next `days_ahead` days, filtered to
        the primary broadcast channel. Sorted by StTime ascending.
        """
        now_jst = datetime.now(JST)
        range_end = now_jst + timedelta(days=days_ahead)
        params = {
            "Command": "ProgLookup",
            "TID": syobocal_tid,
            "ChID": primary_ch_id,
            "Range": f"{fmt_jst(now_jst)}-{fmt_jst(range_end)}",
            "JOIN": "SubTitles",
        }
        # parse XML response → list[ProgItem]
        ...

    async def get_episode_slot(
        self,
        syobocal_tid: int,
        primary_ch_id: int,
        episode_number: int,
    ) -> ProgItem | None:
        """
        Returns the broadcast slot for a specific episode number.
        Used at T-24h and T-1h for direct schedule verification.
        """
        params = {
            "Command": "ProgLookup",
            "TID": syobocal_tid,
            "ChID": primary_ch_id,
            "Count": episode_number,
            "JOIN": "SubTitles",
        }
        # parse XML response → ProgItem or None
        ...
```

---

### Broadcast Schedule Monitor

The daily sweep uses `SyoboiCalendarAdapter.get_upcoming_schedule()` as the primary schedule source for anime with a `syobocal_tid`.

Schedule change detection:

```
for each ONGOING anime with syobocal_tid:
  slots = syoboi_adapter.get_upcoming_schedule(tid, ch_id, days=14)
  for slot in slots:
    stored = pg.get_expected_air_time(anime_id, slot.count)
    if slot.st_time != stored.expected_air_time:
      emit AnimeUpdatedEvent(changed_fields=["broadcast"])
      update PG → KV → reschedule Temporal workflow
    if slot.warn == 1:
      flag in PG for heightened monitoring (next daily sweep re-checks)
```

`Warn=1` is treated as an advisory signal — not a confirmed delay. A schedule date change is the only trigger for rescheduling.

---

### Episode Check-in Workflow — T-24h and T-1h Direct Query

At T-24h and T-1h timers, the workflow calls `get_episode_slot()` directly instead of reading from NATS KV, because KV is only as fresh as the last daily sweep.

```
T-24h timer fires
  ├── syoboi.get_episode_slot(tid, ch_id, episode_number)
  ├── slot.st_time != expected_time → reschedule workflow, no notification
  ├── slot.st_time == expected_time, slot.warn == 1 → send notification, flag for T-1h check
  └── slot.st_time == expected_time, slot.warn == 0 → send EpisodeUpcomingNotification(T-24h)

T-1h timer fires
  └── same pattern; last safety net before notification window
```

---

### T=0 — Syoboi Does Not Confirm Airing

Syoboi tracks broadcast slots, not actual episode availability. At T=0, use `JikanAdapter` or `AniListAdapter` to poll for episode metadata becoming available. Syoboi is not queried at T=0.

---

## Limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| 18% global TID coverage | Movies/OVAs/streaming-only have no TID | AniList fallback for null `syobocal_tid` |
| Times in JST | Timezone bugs if not handled explicitly | Use `zoneinfo.ZoneInfo("Asia/Tokyo")` |
| json.php is internal/unstable | Cannot use for production queries | Use db.php only |
| Warn=1 is advisory, not confirmed | Cannot mechanically block notifications on Warn alone | Treat as signal for heightened monitoring |
| Multiple slots per episode | Wrong channel → wrong time | Always filter by `primary_ch_id` |
| STSubTitle may lag | Subtitle empty until broadcaster announces | Expected; not a blocking concern |

---

## Related Documentation

- [[update_service|Update Service]] — EpisodeAirTrackingWorkflow and source adapter pattern
- [[ingestion_pipeline|Ingestion Pipeline]] — Stage 4 enrichment where `syobocal_tid` is resolved
- [[event_driven_architecture|Event-Driven Architecture]] — Episode check-in workflow and broadcast KV

---

**Status**: Active | **Last Updated**: 2026-02-26
