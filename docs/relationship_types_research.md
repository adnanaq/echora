# Relationship Types Research

**Date**: 2025-10-19
**Purpose**: Document relationship type structures across all sources before fixing

---

## MAL/Jikan Structure

**Discovery**: MAL uses two-level hierarchy

**Structure**:

```
Relationship Category
  └─ Entries with Format Types
```

**Example from One Piece**:

```
Side Story:
  - One Piece: Taose! Kaizoku Ganzack (OVA)
  - One Piece Movie 01 (Movie)
  - One Piece: Umi no Heso no Daibouken-hen (TV Special)
  - One Piece Movie 02 (Movie)
  - One Piece: Jango no Dance Carnival (Special)

When viewing "One Piece: Taose! Kaizoku Ganzack":
Full Story:
  - One Piece (TV)
```

**MAL Relationship Categories**:

- Adaptation
- Side Story
- Summary
- Spin-off
- Alternative Version
- Alternative Setting
- Character
- Full Story
- Parent Story
- Sequel
- Prequel
- Other

**MAL Entry Types** (format of the related anime):

- TV
- Movie
- OVA
- Special
- TV Special
- Music
- PV
- CM
- ONA

**Source Relation Types**

- Adaptation (Novel)
- Adaptation (Light Novel)
- Adaptation (Visual Novel)
- Adaptation (Manga)
- Adaptation (Manhwa)

**Current Implementation**: We store category in `relation_type` but lose the entry type (OVA, Movie, etc.)

**What We Should Store**: Both the relationship category AND the entry type

---

## AnimePlanet Structure

**Discovery**: AnimePlanet also uses two-level hierarchy

**AnimePlanet Relationship Categories**:

- Sequel
- Prequel
- Condensed Version
- Alternate Universe
- Same Franchise
- Recap
- Side Story
- Other Franchise

**AnimePlanet Entry Types** (format):

- TV
- Music Video
- Other
- OVA
- Movie
- TV Special
- Web

**Source Relation Types**:

- Original Manga

**Fields in Data**:

- `relation_type`: The category (e.g., "same_franchise", "sequel")
- `relation_subtype`: The specific type (needs verification if this is format or relationship refinement)

---

## AniList Structure

**Format**: Single field `relationType` (SCREAMING_SNAKE_CASE)

**Examples**: "SIDE_STORY", "SEQUEL", "PREQUEL", "ADAPTATION"

**Question**: Does the node have format information we're ignoring?

**Need to investigate**: Check if `node.type` or similar field exists

---

## AniSearch Structure

**Discovery**: AniSearch also uses two-level hierarchy

**AniSearch Relationship Categories**:

- Summary
- Side Story
- Shared Universe
- Alternate Environment
- Crossover
- Other
- Character
- Original Work
- Sequel
- Prequel

**AniSearch Entry Types** (format):

- TV-Series
- Music Video
- Manga
- TV-Special
- Web
- Other
- Movie
- CM
- OVA

---

## AnimSchedule Structure

**Format**: Category-based grouping

**Relationship Types**:

- Side Story
- Prequel
- Sequel
- Other
- Alternative
- Parent

**Format Types**:

- Movie
- ONA
- OVA
- Special
- TV
- Music
- TV Short (Chinese)
- TV Short
- Movie (Chinese)
- ONA (Chinese)
- Special (Chinese)
- OVA (Chinese)
- TV (Chinese)

**Source Relation Types**:

- 4-koma Manga
- Book
- Card Game
- Doujinshi
- Game
- Light Novel
- Manga
- Music
- Novel
- Original
- Other
- Picture book
- Video Game
- Visual Novel
- Web Manga
- Web Novel

---

## AniDB Structure

**Discovery**: AniDB uses a two-level hierarchy for related anime, where the relationship type is explicitly defined, and the format type is available for the related entry itself.

**AniDB Relationship Categories** (from `<relatedanime>`'s `type` attribute):

- Summary
- Side Story
- Same Setting
- Sequel
- Prequel
- Other
- Parent Story
- Full Story
- Alternate Setting
- Alternative Version

**AniDB Entry Types** (format of the anime entry, from main `<anime>`'s `<type>` tag):

- TV Series
- Movie
- OVA
- Music Video
- Web
- Other
- Unknown
- TV Special
- Special

**Note**: The `Web` type in AniDB often corresponds to ONA (Original Net Animation) or Web Series.

---

## Current Schema

**src/models/anime.py**:

```
RelationEntry:
  - title: str
  - relation_type: str (no enum)
  - url: str

RelatedAnimeEntry:
  - title: str
  - relation_type: str (no enum)
  - url: str
```

**Issues**:

- No enum for relationship types
- No field for entry format/type
- Losing format information from Jikan

---

## To Research

1. Check AnimePlanet temp files - what is `relation_subtype`?
2. Check AniList temp files - is there format information?
3. Check AniSearch temp files - is there format information?
4. Check AnimSchedule temp files - relationship vs format?
5. Look at Jikan temp files - verify two-level structure

---

## Questions

1. Should we add a `format` or `type` field to relation models?
2. Should we create enum for relationship types?
3. Should we reuse existing `AnimeType` enum or create new one?
4. Do we need to store both relationship and format for all sources?
