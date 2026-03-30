"""AniList typed models for character edges.

These are returned by the paginated characters query and contain one character
node per edge, with role and all voice actor role data across all languages.

Description parsing
-------------------
AniList character descriptions embed structured metadata as AniList-flavoured
markdown at the top of the prose field:

    __Height:__ 174 cm
    __Bounty:__ ~!500,000,000!~

    Usopp is a liar...

``AniListCharacterNode`` normalises this on validation:
- ``description_prose``       — remaining prose (markdown links simplified)
- ``description_attributes``  — non-spoiler key/value pairs
- ``description_spoilers``    — pairs whose value was wrapped in ``~! … !~``
"""
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator

_ATTR_LINE_RE = re.compile(r"^__(.+?):__\s*(.*)")
_SPOILER_RE = re.compile(r"~!\s*(.*?)\s*!~", re.DOTALL)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


class AniListCharacterName(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    full: str | None = None
    native: str | None = None
    alternative: list[str] = Field(default_factory=list)
    alternative_spoiler: list[str] = Field(default_factory=list, alias="alternativeSpoiler")


class AniListCharacterImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    large: str | None = None
    medium: str | None = None


class AniListFuzzyDate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    year: int | None = None
    month: int | None = None
    day: int | None = None


class AniListStaffName(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    full: str | None = None
    native: str | None = None


class AniListStaffImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    large: str | None = None
    medium: str | None = None


class AniListVoiceActor(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: int
    name: AniListStaffName | None = None
    language_v2: str | None = Field(None, alias="languageV2")
    image: AniListStaffImage | None = None
    site_url: str | None = Field(None, alias="siteUrl")


class AniListVoiceActorRole(BaseModel):
    """A single entry from CharacterEdge.voiceActorRoles — one VA with language/role info."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    voice_actor: AniListVoiceActor | None = Field(None, alias="voiceActor")
    role_notes: str | None = Field(None, alias="roleNotes")
    dub_group: str | None = Field(None, alias="dubGroup")


class AniListCharacterNode(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: int
    name: AniListCharacterName | None = None
    image: AniListCharacterImage | None = None
    description: str | None = None
    gender: str | None = None
    date_of_birth: AniListFuzzyDate | None = Field(None, alias="dateOfBirth")
    age: str | None = None
    blood_type: str | None = Field(None, alias="bloodType")
    favourites: int | None = None
    site_url: str | None = Field(None, alias="siteUrl")

    # Computed from description — populated by _parse_description validator
    description_prose: str | None = Field(None, exclude=True)
    description_attributes: dict[str, str] = Field(default_factory=dict, exclude=True)
    description_spoilers: dict[str, str] = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def _parse_description(self) -> "AniListCharacterNode":
        """Split AniList markdown description into prose, attributes, and spoilers."""
        if not self.description:
            return self

        lines = self.description.split("\n")
        attributes: dict[str, str] = {}
        spoilers: dict[str, str] = {}
        prose_lines: list[str] = []
        in_prose = False

        for line in lines:
            if in_prose:
                prose_lines.append(line)
                continue
            stripped = line.strip()
            if not stripped:
                # Blank lines inside the attribute block are skipped
                continue
            m = _ATTR_LINE_RE.match(stripped)
            if m:
                key = m.group(1).strip().lower().replace(" ", "_")
                raw_value = m.group(2).strip()
                spoiler_match = _SPOILER_RE.fullmatch(raw_value)
                if spoiler_match:
                    spoilers[key] = spoiler_match.group(1).strip()
                else:
                    clean_value = _SPOILER_RE.sub(lambda s: s.group(1).strip(), raw_value)
                    attributes[key] = clean_value
            else:
                in_prose = True
                prose_lines.append(line)

        prose: str | None = "\n".join(prose_lines).lstrip("\n").strip() or None
        if prose:
            prose = _MD_LINK_RE.sub(r"\1", prose)
            prose = _SPOILER_RE.sub(lambda s: s.group(1).strip(), prose)
            prose = prose.strip() or None

        self.description_prose = prose
        self.description_attributes = attributes
        self.description_spoilers = spoilers
        return self


class AniListCharacterEdge(BaseModel):
    """A single edge from the AniList characters paginated connection."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    node: AniListCharacterNode
    role: str | None = None
    voice_actor_roles: list[AniListVoiceActorRole] = Field(default_factory=list, alias="voiceActorRoles")
