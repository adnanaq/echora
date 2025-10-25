#!/usr/bin/env python3
"""
AniList Helper for AI Enrichment Integration

Test script to fetch and analyze AniList data for anime entries using GraphQL API.
"""

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AniListEnrichmentHelper:
    """Helper for AniList data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize AniList enrichment helper."""
        self.base_url = "https://graphql.anilist.co"
        self.session = None
        self.rate_limit_remaining = 90
        self.rate_limit_reset = None

    async def _make_request(
        self, query: str, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make GraphQL request to AniList API."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"query": query, "variables": variables or {}}

        if not self.session:
            # No timeout - we want ALL data, even if it takes minutes
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None)
            )

        try:
            if self.rate_limit_remaining < 5:
                logging.info(
                    f"Rate limit low ({self.rate_limit_remaining}), waiting 60 seconds..."
                )
                await asyncio.sleep(60)
                self.rate_limit_remaining = 90

            async with self.session.post(
                self.base_url, json=payload, headers=headers
            ) as response:
                if "X-RateLimit-Remaining" in response.headers:
                    self.rate_limit_remaining = int(
                        response.headers["X-RateLimit-Remaining"]
                    )

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 60))
                    logger.warning(
                        f"Rate limit exceeded. Waiting {retry_after} seconds..."
                    )
                    await asyncio.sleep(retry_after)
                    return await self._make_request(query, variables)

                response.raise_for_status()
                data = await response.json()
                if "errors" in data:
                    logger.error(f"AniList GraphQL errors: {data['errors']}")
                    return {}
                return data.get("data", {})
        except Exception as e:
            logger.error(f"AniList API request failed: {e}")
            return {}

    def _get_media_query_fields(self) -> str:
        return """
        id
        idMal
        title {
          romaji
          english
          native
          userPreferred
        }
        description(asHtml: false)
        source
        format
        episodes
        duration
        status
        season
        seasonYear
        countryOfOrigin
        isAdult
        hashtag
        coverImage {
          extraLarge
          large
          medium
          color
        }
        bannerImage
        trailer {
          id
          site
          thumbnail
        }
        averageScore
        meanScore
        popularity
        favourites
        trending
        genres
        synonyms
        tags {
          id
          name
          description
          category
          rank
          isGeneralSpoiler
          isMediaSpoiler
          isAdult
        }
        relations {
          edges {
            node {
              id
              title {
                romaji
                english
              }
              format
              status
            }
            relationType
          }
        }
        studios {
          edges {
            node {
              id
              name
              isAnimationStudio
            }
            isMain
          }
        }
        externalLinks {
          id
          url
          site
          type
          language
          color
          icon
        }
        streamingEpisodes {
          title
          thumbnail
          url
          site
        }
        nextAiringEpisode {
          episode
          airingAt
          timeUntilAiring
        }
        rankings {
          id
          rank
          type
          format
          year
          season
          allTime
          context
        }
        stats {
          scoreDistribution {
            score
            amount
          }
          statusDistribution {
            status
            amount
          }
        }
        updatedAt
        """

    # def _build_query_by_mal_id(self) -> str:
    #     return f"query ($idMal: Int) {{ Media(idMal: $idMal, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    def _build_query_by_anilist_id(self) -> str:
        return f"query ($id: Int) {{ Media(id: $id, type: ANIME) {{ {self._get_media_query_fields()} }} }}"

    # async def fetch_anime_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
    #     query = self._build_query_by_mal_id()
    #     variables = {"idMal": mal_id}
    #     response = await self._make_request(query, variables)
    #     return response.get("Media")

    async def fetch_anime_by_anilist_id(
        self, anilist_id: int
    ) -> Optional[Dict[str, Any]]:
        query = self._build_query_by_anilist_id()
        variables = {"id": anilist_id}
        response = await self._make_request(query, variables)
        return response.get("Media")

    async def _fetch_paginated_data(
        self, anilist_id: int, query_template: str, data_key: str
    ) -> List[Dict[str, Any]]:
        all_items = []
        page = 1
        has_next_page = True
        while has_next_page:
            variables = {"id": anilist_id, "page": page}
            response = await self._make_request(query_template, variables)
            if (
                not response
                or not response.get("Media")
                or not response["Media"].get(data_key)
            ):
                break
            data = response["Media"][data_key]
            all_items.extend(data.get("edges", []))
            has_next_page = data.get("pageInfo", {}).get("hasNextPage", False)
            page += 1
            await asyncio.sleep(0.5)
        return all_items

    async def fetch_all_characters(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            characters(page: $page, perPage: 25, sort: ROLE) {
              pageInfo { hasNextPage }
              edges {
                node {
                  id
                  name {
                    full
                    native
                    alternative
                    alternativeSpoiler
                  }
                  image { large medium }
                  favourites
                  gender
                }
                role
                voiceActors(language: JAPANESE) {
                  id
                  name { full native }
                }
              }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "characters")

    async def fetch_all_staff(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            staff(page: $page, perPage: 25, sort: RELEVANCE) {
              pageInfo { hasNextPage }
              edges { node { id name { full native } } role }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "staff")

    async def fetch_all_episodes(self, anilist_id: int) -> List[Dict[str, Any]]:
        query = """
        query ($id: Int!, $page: Int!) {
          Media(id: $id, type: ANIME) {
            airingSchedule(page: $page, perPage: 50) {
              pageInfo { hasNextPage }
              edges { node { id episode airingAt } }
            }
          }
        }
        """
        return await self._fetch_paginated_data(anilist_id, query, "airingSchedule")

    async def _fetch_and_populate_details(
        self, anime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        anilist_id = anime_data.get("id")
        if not anilist_id:
            return anime_data

        details = {
            "characters": await self.fetch_all_characters(anilist_id),
            "staff": await self.fetch_all_staff(anilist_id),
            "airingSchedule": await self.fetch_all_episodes(anilist_id),
        }

        for key, data in details.items():
            if data:
                anime_data[key] = {"edges": data}
                logger.info(f"Total {key} fetched: {len(data)}")

        return anime_data

    # async def fetch_all_data_by_mal_id(self, mal_id: int) -> Optional[Dict[str, Any]]:
    #     anime_data = await self.fetch_anime_by_mal_id(mal_id)
    #     if not anime_data:
    #         logger.warning(f"No AniList data found for MAL ID: {mal_id}")
    #         return None
    #     return await self._fetch_and_populate_details(anime_data)

    async def fetch_all_data_by_anilist_id(
        self, anilist_id: int
    ) -> Optional[Dict[str, Any]]:
        anime_data = await self.fetch_anime_by_anilist_id(anilist_id)
        if not anime_data:
            logger.warning(f"No AniList data found for AniList ID: {anilist_id}")
            return None
        return await self._fetch_and_populate_details(anime_data)

    async def close(self) -> None:
        if self.session:
            await self.session.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test AniList data fetching")
    group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--mal-id", type=int, help="MyAnimeList ID to fetch")
    group.add_argument("--anilist-id", type=int, help="AniList ID to fetch")
    parser.add_argument(
        "--output",
        type=str,
        default="test_anilist_output.json",
        help="Output file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    helper = AniListEnrichmentHelper()
    anime_data = None
    try:
        if args.anilist_id:
            anime_data = await helper.fetch_all_data_by_anilist_id(args.anilist_id)
        # elif args.mal_id:
        #     anime_data = await helper.fetch_all_data_by_mal_id(args.mal_id)

        if anime_data:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(anime_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Data saved to {args.output}")
        else:
            logger.error("No data found for the given ID.")
    finally:
        await helper.close()


if __name__ == "__main__":
    asyncio.run(main())
