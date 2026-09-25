"""Build the canonical company list from a provider's per-role lists.

Every mapper knows its own provider's roles and builds `CompanyEntry` lists for
them. This turns those into the one list the model stores, where a company
appears once carrying every role that provider gave it.

That last part matters for the providers that credit one company twice: Kitsu
files Madhouse as both producer and studio on Death Note, and AniDB reports Toei
Animation under both `Animation Work` and `Work` on One Piece. Two entries for
one company would be a duplicate the merge then has to undo.
"""

from __future__ import annotations

from common.models.anime import CompanyEntry, CompanyRole
from enrichment.sources.base.external_links import normalize_link_url


def companies_from_roles(
    *,
    studios: list[CompanyEntry] | None = None,
    producers: list[CompanyEntry] | None = None,
    licensors: list[CompanyEntry] | None = None,
) -> list[CompanyEntry]:
    """Collapse one provider's role lists into the canonical company list.

    Args:
        studios: Companies this provider calls studios.
        producers: Companies this provider calls producers.
        licensors: Companies this provider calls licensors.

    Returns:
        One entry per company, carrying every role this provider gave it and
        every source URL it supplied, in the order the company was first seen.
        Names are matched exactly here - reconciling spellings is the merge's
        job, and within one provider the spelling does not vary.
    """
    combined: dict[str, CompanyEntry] = {}

    for role, entries in (
        (CompanyRole.STUDIO, studios),
        (CompanyRole.PRODUCER, producers),
        (CompanyRole.LICENSOR, licensors),
    ):
        for entry in entries or []:
            existing = combined.get(entry.name)
            if existing is None:
                combined[entry.name] = CompanyEntry(
                    name=entry.name,
                    roles=[role],
                    description=entry.description,
                    sources=list(entry.sources),
                )
                continue
            if role not in existing.roles:
                existing.roles.append(role)
            known = {normalize_link_url(url) for url in existing.sources}
            existing.sources.extend(
                url for url in entry.sources if normalize_link_url(url) not in known
            )
            existing.description = existing.description or entry.description

    return list(combined.values())
