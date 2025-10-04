"""Anime-Planet scraping client implementation."""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin

from .base_scraper import BaseScraper


class AnimePlanetScraper(BaseScraper):
    """Anime-Planet scraping client."""

    def __init__(self, **kwargs):
        """Initialize Anime-Planet scraper."""
        super().__init__(service_name="animeplanet", **kwargs)
        self.base_url = "https://www.anime-planet.com"

    async def get_anime_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        """Get anime information by Anime-Planet slug."""
        # Check cache first
        cache_key = f"animeplanet_anime_{slug}"
        if self.cache_manager:
            try:
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result:
                    return cached_result
            except:
                pass

        try:
            url = f"{self.base_url}/anime/{slug}"
            response = await self._make_request(url, timeout=10)

            # Check if Cloudflare blocked us
            if (
                response.get("is_cloudflare_protected")
                and "checking your browser" in response["content"].lower()
            ):
                return None

            # Parse HTML
            soup = self._parse_html(response["content"])

            # Extract base data
            result = self._extract_base_data(soup, url)
            result["domain"] = "anime-planet"
            result["slug"] = slug

            # Extract anime-specific data
            anime_data = self._extract_anime_data(soup)
            result.update(anime_data)

            # Cache the result
            if self.cache_manager and result.get("title"):
                try:
                    await self.cache_manager.set(cache_key, result)
                except:
                    pass

            return result if result.get("json_ld", {}).get("name") else None

        except Exception as e:
            # Re-raise circuit breaker exceptions
            if "circuit breaker" in str(e).lower():
                raise
            return None

    async def search_anime(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search anime on Anime-Planet."""
        try:
            # Anime-Planet search URL
            search_url = f"{self.base_url}/anime/all?name={quote(query)}"
            response = await self._make_request(search_url, timeout=10)

            # Parse search results
            soup = self._parse_html(response["content"])
            results = self._parse_search_results(soup, limit)

            return results

        except Exception:
            return []

    async def get_anime_characters(
        self, slug: str, enrich_characters: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Get anime character information by Anime-Planet slug.

        Args:
            slug: Anime slug on Anime-Planet
            enrich_characters: If True (default), fetch detailed character data from individual pages
                              (adds gender, eye_color, birthday, description, anime_roles, etc.)
                              Warning: This makes N additional requests (one per character)
                              Set to False for faster extraction with basic data only

        Returns:
            Dict with 'characters' list and 'total_count', or None if no characters found
        """
        try:
            url = f"{self.base_url}/anime/{slug}/characters"
            response = await self._make_request(url, timeout=15)

            # Check if Cloudflare blocked us
            if (
                response.get("is_cloudflare_protected")
                and "checking your browser" in response["content"].lower()
            ):
                return None

            # Parse HTML
            soup = self._parse_html(response["content"])

            # Flat list of all characters
            all_characters = []

            # Find all role section headers
            sections = soup.find_all("h3", class_="sub")

            for section in sections:
                section_name = section.get_text(strip=True)

                # Map section name to role
                if "Main" in section_name:
                    role = "Main"
                elif "Secondary" in section_name:
                    role = "Secondary"
                elif "Minor" in section_name:
                    role = "Minor"
                else:
                    continue

                # Find the table following this header
                table = section.find_next("table")
                if not table:
                    continue

                # Parse each character row
                rows = table.find_all("tr")
                for row in rows:
                    char_data = self._parse_character_row(row, role)
                    if char_data:
                        all_characters.append(char_data)

            # Enrich characters with detailed data if requested
            if enrich_characters and all_characters:
                import asyncio

                for char in all_characters:
                    # Extract character slug from URL
                    char_slug = char.get("url", "").replace("/characters/", "")
                    if char_slug:
                        detailed_data = await self.get_character_details(char_slug)
                        if detailed_data:
                            # Merge detailed data, preserving role from anime page
                            char.update(detailed_data)
                        # Rate limiting between character requests
                        await asyncio.sleep(0.5)

            # Return flat structure with characters array
            if all_characters:
                return {
                    "characters": all_characters,
                    "total_count": len(all_characters),
                }
            return None

        except Exception as e:
            # Re-raise circuit breaker exceptions
            if "circuit breaker" in str(e).lower():
                raise
            return None

    async def get_character_details(
        self, character_slug: str
    ) -> Optional[Dict[str, Any]]:
        """Get detailed character information from individual character page.

        Args:
            character_slug: Character slug (e.g., 'monkey-d-luffy')

        Returns:
            Dict with detailed character data including gender, eye_color, birthday,
            description, anime_roles, manga_roles, etc.
        """
        try:
            url = f"{self.base_url}/characters/{character_slug}"
            response = await self._make_request(url, timeout=15)

            if (
                response.get("is_cloudflare_protected")
                and "checking your browser" in response["content"].lower()
            ):
                return None

            if response["status_code"] != 200:
                return None

            soup = self._parse_html(response["content"])
            char_data: Dict[str, Any] = {}

            # Basic info
            name_h1 = soup.find("h1", itemprop="name")
            if name_h1:
                char_data["name"] = name_h1.get_text(strip=True)

            img = soup.find("img", itemprop="image")
            if img:
                char_data["image"] = img.get("src")

            # Section 1: entryBar - Gender, Hair Color, Rankings
            entry_bar = soup.find("section", class_="pure-g entryBar")
            if entry_bar:
                # Find all divs with class containing "pure-1" (handles both "pure-1" and "pure-1 md-1-5")
                bar_divs = entry_bar.find_all(
                    "div", class_=lambda x: x and "pure-1" in x
                )

                for div in bar_divs:
                    text = div.get_text(strip=True)

                    if "Gender:" in text:
                        char_data["gender"] = text.replace("Gender:", "").strip()

                    if "Hair Color:" in text:
                        char_data["hair_color"] = text.replace(
                            "Hair Color:", ""
                        ).strip()

                    if "Rank" in text and "fa-heart" in str(div):
                        rank_link = div.find("a")
                        if rank_link:
                            rank_text = (
                                rank_link.get_text(strip=True)
                                .replace("#", "")
                                .replace(",", "")
                            )
                            try:
                                char_data["loved_rank"] = int(rank_text)
                            except ValueError:
                                pass

                    if "Rank" in text and "heartOff" in str(div):
                        rank_link = div.find("a")
                        if rank_link:
                            rank_text = (
                                rank_link.get_text(strip=True)
                                .replace("#", "")
                                .replace(",", "")
                            )
                            try:
                                char_data["hated_rank"] = int(rank_text)
                            except ValueError:
                                pass

            # Section 2: EntryMetadata - Eye color, Age, Birthday, Height, Weight, etc.
            entry_metadata = soup.find("div", class_="EntryMetadata")
            if entry_metadata:
                metadata_items = entry_metadata.find_all(
                    "div", class_="EntryMetadata__item"
                )

                for item in metadata_items:
                    title_elem = item.find("h3", class_="EntryMetadata__title")
                    value_elem = item.find("div", class_="EntryMetadata__value")

                    if title_elem and value_elem:
                        title = title_elem.get_text(strip=True)
                        value = value_elem.get_text(strip=True)

                        field_name = title.lower().replace(" ", "_")
                        char_data[field_name] = value

            # Section 3: Description
            synopsis_section = soup.find("div", class_="entrySynopsis")
            if synopsis_section:
                for elem in synopsis_section.find_all("p"):
                    text = elem.get_text(strip=True)
                    if text and len(text) > 50 and "Tags" not in text:
                        char_data["description"] = text
                        break

            # Section 4: Tags
            tags_section = soup.find("div", class_="tags")
            if tags_section:
                tags = [tag.get_text(strip=True) for tag in tags_section.find_all("a")]
                if tags:
                    char_data["tags"] = tags

            # Section 5: Alternative Names
            alt_names_section = soup.find("div", class_="entryAltNames")
            if alt_names_section:
                alt_names = []
                for name_elem in alt_names_section.find_all("li"):
                    alt_name = name_elem.get_text(strip=True)
                    if alt_name:
                        alt_names.append(alt_name)
                if alt_names:
                    char_data["alternative_names"] = alt_names

            # Section 6: Anime Roles
            anime_roles_h3 = soup.find("h3", string="Anime Roles")
            if anime_roles_h3:
                table = anime_roles_h3.find_next("table")
                if table:
                    rows = table.find_all("tr")
                    anime_roles = []

                    for row in rows:
                        anime_link = row.find("a", href=lambda x: x and "/anime/" in x)
                        if anime_link:
                            role_data = {
                                "anime_title": anime_link.get_text(strip=True),
                                "anime_url": anime_link.get("href"),
                            }

                            role_text = row.get_text()
                            if "Main" in role_text:
                                role_data["role"] = "Main"
                            elif "Supporting" in role_text:
                                role_data["role"] = "Supporting"

                            anime_roles.append(role_data)

                    if anime_roles:
                        char_data["anime_roles"] = anime_roles

            # Section 7: Manga Roles
            manga_roles_h3 = soup.find("h3", string="Manga Roles")
            if manga_roles_h3:
                table = manga_roles_h3.find_next("table")
                if table:
                    rows = table.find_all("tr")
                    manga_roles = []

                    for row in rows:
                        manga_link = row.find("a", href=lambda x: x and "/manga/" in x)
                        if manga_link:
                            role_data = {
                                "manga_title": manga_link.get_text(strip=True),
                                "manga_url": manga_link.get("href"),
                            }

                            role_text = row.get_text()
                            if "Main" in role_text:
                                role_data["role"] = "Main"
                            elif "Supporting" in role_text:
                                role_data["role"] = "Supporting"

                            manga_roles.append(role_data)

                    if manga_roles:
                        char_data["manga_roles"] = manga_roles

            # Section 8: Voice Actors (from individual page)
            va_h3 = soup.find("h3", string=lambda x: x and "Voice Actor" in str(x))
            if va_h3:
                table = va_h3.find_next("table")
                if table:
                    rows = table.find_all("tr")
                    voice_actors = {}

                    for row in rows:
                        va_link = row.find("a", href=lambda x: x and "/people/" in x)
                        if va_link:
                            va_name = va_link.get_text(strip=True)
                            va_url = va_link.get("href")

                            flag = row.find(
                                "div", class_=lambda x: x and "flag" in str(x)
                            )
                            lang = "unknown"
                            if flag:
                                classes = flag.get("class", [])
                                for cls in classes:
                                    if cls.startswith("flag") and len(cls) > 4:
                                        lang = cls[4:].lower()
                                        break

                            if lang not in voice_actors:
                                voice_actors[lang] = []
                            voice_actors[lang].append({"name": va_name, "url": va_url})

                    if voice_actors:
                        char_data["voice_actors"] = voice_actors

            return char_data if char_data else None

        except Exception as e:
            if "circuit breaker" in str(e).lower():
                raise
            return None

    def _parse_character_row(self, row, role: str) -> Optional[Dict[str, Any]]:
        """Parse a single character table row."""
        try:
            # Find character name link
            name_link = row.find(
                "a", class_="name", href=lambda x: x and "/characters/" in x
            )
            if not name_link:
                return None

            char_data = {
                "name": name_link.get_text(strip=True),
                "url": name_link.get("href"),
                "role": role,
            }

            # Find character image (in same row, before the name)
            img = row.find("img", alt=char_data["name"])
            if img:
                char_data["image"] = img.get("src") or img.get("data-src")

            # Find character tags
            tags_div = row.find("div", class_="tags")
            if tags_div:
                tag_links = tags_div.find_all(
                    "a", href=lambda x: x and "/characters/tags/" in x
                )
                char_data["tags"] = [tag.get_text(strip=True) for tag in tag_links]

            # Find voice actors by language
            actors_td = row.find("td", class_="tableActors")
            if actors_td:
                voice_actors = {}
                flag_divs = actors_td.find_all(
                    "div", class_=lambda x: x and "flag" in x
                )

                for flag_div in flag_divs:
                    # Extract language from flag class (e.g., "flagJP" -> "jp")
                    flag_classes = flag_div.get("class", [])
                    lang = None
                    for cls in flag_classes:
                        if cls.startswith("flag") and len(cls) > 4:
                            lang = cls[4:].lower()  # Remove "flag" prefix
                            break

                    if lang:
                        # Get all voice actor links in this language section
                        va_links = flag_div.find_all(
                            "a", href=lambda x: x and "/people/" in x
                        )
                        actors = [
                            {"name": va.get_text(strip=True), "url": va.get("href")}
                            for va in va_links
                        ]
                        if actors:
                            voice_actors[lang] = actors

                if voice_actors:
                    char_data["voice_actors"] = voice_actors

            return char_data

        except Exception:
            return None

    def _extract_json_ld(self, soup) -> Optional[Dict[str, Any]]:
        """Override base method to fix Anime-Planet's malformed image URLs."""
        json_ld = super()._extract_json_ld(soup)

        # Fix Anime-Planet's bug: malformed image URLs with double base_url
        if json_ld and "image" in json_ld and isinstance(json_ld["image"], str):
            image_url = json_ld["image"]
            if "anime-planet.comhttps://" in image_url:
                json_ld["image"] = image_url.replace(
                    "https://www.anime-planet.comhttps://", "https://"
                )

        return json_ld

    def _extract_anime_data(self, soup) -> Dict[str, Any]:
        """Extract anime-specific data from Anime-Planet page."""
        data = {}

        # Extract metadata from info table
        info_data = self._extract_info_table(soup)
        data.update(info_data)

        # Extract rating if available
        rating = self._extract_rating(soup)
        if rating:
            data["rating"] = rating

        # Extract enhanced data from JSON-LD and HTML
        enhanced_data = self._extract_enhanced_data(soup)
        data.update(enhanced_data)

        return data

    def _extract_info_table(self, soup) -> Dict[str, Any]:
        """Extract information from the anime info table."""
        info_data: Dict[str, Any] = {}

        # Look for info table in various locations
        info_containers = [
            soup.find("table"),
            soup.find(".entryBox table"),
            soup.find(".info-table"),
            soup.find(".anime-info"),
        ]

        for container in info_containers:
            if not container:
                continue

            # Extract key-value pairs from table rows
            rows = container.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = self._clean_text(cells[0].text).lower().replace(":", "")
                    value = self._clean_text(cells[1].text)

                    # Map common fields
                    if key in ["type"]:
                        info_data["type"] = value
                    elif key in ["episodes", "episode count"]:
                        try:
                            info_data["episodes"] = int(value)
                        except ValueError:
                            info_data["episodes"] = value
                    elif key in ["status"]:
                        info_data["status"] = value
                    elif key in ["aired", "air date"]:
                        info_data["aired"] = value
                    elif key in ["studio", "studios"]:
                        info_data["studio"] = value
                    elif key in ["year"]:
                        info_data["year"] = value

            if info_data:  # If we found data, break
                break

        return info_data

    def _extract_tags(self, soup) -> List[str]:
        """Extract tags/genres from the page."""
        tags = []

        # Look for tags in various locations
        tag_containers = [
            soup.find(".tags"),
            soup.find(".genres"),
            soup.find(".tags-list"),
            soup.find_all("a", href=re.compile(r"/tags/")),
            soup.find_all("a", href=re.compile(r"/genre/")),
        ]

        for container in tag_containers:
            if not container:
                continue

            if isinstance(container, list):
                # List of tag links
                for link in container:
                    tag_text = self._clean_text(link.text)
                    if tag_text and tag_text not in tags:
                        tags.append(tag_text)
            else:
                # Container with tag links
                tag_links = container.find_all("a")
                for link in tag_links:
                    tag_text = self._clean_text(link.text)
                    if tag_text and tag_text not in tags:
                        tags.append(tag_text)

        return tags[:10]  # Limit to 10 tags

    def _extract_rating(self, soup) -> Optional[str]:
        """Extract rating/score if available."""
        rating_selectors = [
            (".rating", {}),
            (".score", {}),
            (".avg-rating", {}),
            ("span", {"class": "rating"}),
        ]

        for selector, attrs in rating_selectors:
            if "." in selector:
                rating_elem = soup.select_one(selector)
            else:
                rating_elem = soup.find(selector, attrs)

            if rating_elem:
                rating_text = self._clean_text(rating_elem.text)
                # Extract numeric rating
                rating_match = re.search(r"(\d+\.?\d*)", rating_text)
                if rating_match:
                    return rating_match.group(1)

        return None

    def _parse_search_results(self, soup, limit: int) -> List[Dict[str, Any]]:
        """Parse search results from Anime-Planet search page."""
        results = []

        # Look for search result containers - Anime-Planet uses <li class="card"> in <ul class="cardDeck">
        result_containers = [
            soup.find_all("li", class_="card"),  # Actual structure used by Anime-Planet
            soup.find_all("div", class_="search-result"),
            soup.find_all("div", class_="card"),
            soup.find_all("div", class_="anime-card"),
            soup.find_all("li", class_=re.compile(r".*result.*")),
        ]

        for containers in result_containers:
            if not containers:
                continue

            for container in containers[:limit]:
                result = self._parse_single_search_result(container)
                if result:
                    results.append(result)

            if results:  # If we found results, break
                break

        return results[:limit]

    def _parse_single_search_result(self, container) -> Optional[Dict[str, Any]]:
        """Parse a single search result item with rich tooltip data."""
        try:
            # Extract title and link - Anime-Planet structure: <a><h3 class="cardName">Title</h3></a>
            title_link = container.find("a")
            if not title_link:
                return None

            href = title_link.get("href", "")

            # Extract title from h3.cardName inside the link
            title_elem = title_link.find("h3", class_="cardName")
            if title_elem:
                title = self._clean_text(title_elem.text)
            else:
                # Fallback to link text
                title = self._clean_text(title_link.text)

            # Extract slug from href
            slug_match = re.search(r"/anime/([^/?]+)", href)
            slug = slug_match.group(1) if slug_match else None

            if not title or not slug:
                return None

            result = {
                "title": title,
                "slug": slug,
                "url": urljoin(self.base_url, href),
                "domain": "anime-planet",
            }

            # Extract rich tooltip data if available
            tooltip_data = self._extract_tooltip_data(title_link)
            if tooltip_data:
                result.update(tooltip_data)

            # Extract data attributes from card container
            card_attributes = self._extract_card_attributes(container)
            if card_attributes:
                result.update(card_attributes)

            # Fallback: Extract basic info from card if tooltip data not available
            if not tooltip_data:
                # Extract additional info if available
                type_elem = container.find("div", class_="type") or container.find(
                    "div", class_="anime-type"
                )
                if type_elem:
                    result["type"] = self._clean_text(type_elem.text)

                synopsis_elem = container.find(
                    "p", class_="synopsis"
                ) or container.find("p")
                if synopsis_elem:
                    synopsis = self._clean_text(synopsis_elem.text)
                    if synopsis and len(synopsis) > 20:  # Only if substantial
                        result["synopsis"] = (
                            synopsis[:200] + "..." if len(synopsis) > 200 else synopsis
                        )

            return result

        except Exception:
            return None

    def _extract_tooltip_data(self, title_link) -> Optional[Dict[str, Any]]:
        """Extract rich data from tooltip hover information."""
        try:
            import html

            # Find tooltip link (has tooltip class)
            tooltip_link = title_link
            if not tooltip_link or not tooltip_link.get("class"):
                return None

            # Check if this is a tooltip link
            classes = tooltip_link.get("class", [])
            if not any("tooltip" in cls for cls in classes):
                return None

            # Extract tooltip HTML from title attribute
            title_html = tooltip_link.get("title", "")
            if not title_html:
                return None

            # Decode HTML entities
            decoded_html = html.unescape(title_html)

            # Parse the tooltip HTML
            tooltip_soup = self._parse_html(decoded_html)

            tooltip_data: Dict[str, Any] = {}

            # Extract title (should match main title, but might be more complete)
            title_elem = tooltip_soup.find("h5", class_="theme-font")
            if title_elem:
                tooltip_data["tooltip_title"] = self._clean_text(title_elem.text)

            # Extract alternative title
            alt_title_elem = tooltip_soup.find("h6", class_="tooltip-alt")
            if alt_title_elem:
                alt_title = self._clean_text(alt_title_elem.text)
                # Remove "Alt title:" prefix if present
                if alt_title.startswith("Alt title:"):
                    alt_title = alt_title[10:].strip()
                tooltip_data["alt_title"] = alt_title

            # Extract entry bar info (type, studio, year, rating)
            entry_bar = tooltip_soup.find("ul", class_="entryBar")
            if entry_bar and hasattr(entry_bar, "find_all"):
                li_elements = entry_bar.find_all("li")
                for li in li_elements:
                    li_classes = li.get("class", [])
                    li_text = self._clean_text(li.text)

                    if "type" in li_classes:
                        # Extract type and episodes: "TV (12 eps)" -> type="TV", episodes=12
                        type_match = re.match(
                            r"([^(]+)(?:\s*\((\d+)\s*eps?\))?", li_text
                        )
                        if type_match:
                            tooltip_data["type"] = type_match.group(1).strip()
                            if type_match.group(2):
                                tooltip_data["episodes"] = int(type_match.group(2))
                    elif "iconYear" in li_classes:
                        # Extract year
                        year_match = re.search(r"(\d{4})", li_text)
                        if year_match:
                            tooltip_data["year"] = int(year_match.group(1))
                    elif li.find("div", class_="ttRating"):
                        # Extract rating
                        rating_div = li.find("div", class_="ttRating")
                        if rating_div:
                            rating_text = self._clean_text(rating_div.text)
                            try:
                                tooltip_data["rating"] = float(rating_text)
                            except ValueError:
                                pass
                    elif (
                        li_text
                        and not any(cls in li_classes for cls in ["type", "iconYear"])
                        and li_text != "Add to list"
                    ):
                        # This is likely the studio (no specific class, just text)
                        if li_text not in tooltip_data.get("studios", []):
                            if "studios" not in tooltip_data:
                                tooltip_data["studios"] = []
                            tooltip_data["studios"].append(li_text)

            # Extract synopsis
            synopsis_elem = tooltip_soup.find("p")
            if synopsis_elem:
                # Get text content, removing any embedded links
                synopsis = synopsis_elem.get_text(strip=True)
                if synopsis and len(synopsis) > 20:  # Only if substantial
                    tooltip_data["synopsis"] = synopsis

            # Extract tags
            tags_section = tooltip_soup.find("div", class_="tags")
            if tags_section and hasattr(tags_section, "find_all"):
                tag_items = tags_section.find_all("li")
                if tag_items:
                    tags = [
                        self._clean_text(tag.text)
                        for tag in tag_items
                        if tag.text.strip()
                    ]
                    if tags:
                        tooltip_data["tags"] = tags

            return tooltip_data if tooltip_data else None

        except Exception:
            return None

    def _extract_card_attributes(self, container) -> Optional[Dict[str, Any]]:
        """Extract data attributes from card container."""
        try:
            card_data = {}

            # Extract data attributes
            if container.get("data-total-episodes"):
                try:
                    episodes = int(container.get("data-total-episodes"))
                    if episodes > 0:  # Only add if meaningful
                        card_data["total_episodes"] = episodes
                except ValueError:
                    pass

            if container.get("data-id"):
                card_data["animeplanet_id"] = container.get("data-id")

            if container.get("data-type"):
                card_data["content_type"] = container.get("data-type")

            return card_data if card_data else None

        except Exception:
            return None

    def _extract_enhanced_data(self, soup) -> Dict[str, Any]:
        """Extract enhanced data including ratings, rankings, alternative titles, studios, and characters."""
        enhanced_data = {}

        # Extract from JSON-LD structured data
        json_ld_data = self._extract_enhanced_json_ld(soup)
        enhanced_data.update(json_ld_data)

        # Extract ranking information
        rank_data = self._extract_ranking(soup)
        enhanced_data.update(rank_data)

        # Extract alternative titles
        alt_titles = self._extract_alternative_titles(soup)
        enhanced_data.update(alt_titles)

        # Extract studio information
        studio_data = self._extract_studios(soup)
        enhanced_data.update(studio_data)

        # Extract status with date logic
        status_data = self._extract_enhanced_status(soup)
        enhanced_data.update(status_data)

        # Extract related anime from same franchise
        related_data = self._extract_related_anime(soup)
        enhanced_data.update(related_data)

        return enhanced_data

    def _extract_enhanced_json_ld(self, soup) -> Dict[str, Any]:
        """Extract enhanced data from JSON-LD structured data."""
        data: Dict[str, Any] = {}
        json_ld = self._extract_json_ld(soup)

        if not json_ld:
            return data

        return data

    def _extract_staff_from_json_ld(self, json_ld) -> Dict[str, Any]:
        """Extract staff information from JSON-LD data."""
        staff_data = {}

        # Extract directors
        if "director" in json_ld:
            directors = json_ld["director"]
            if isinstance(directors, list):
                director_names = [
                    d.get("name", "")
                    for d in directors
                    if isinstance(d, dict) and "name" in d
                ]
                if director_names:
                    staff_data["directors"] = director_names

        # Extract music composers
        if "musicBy" in json_ld:
            composers = json_ld["musicBy"]
            if isinstance(composers, list):
                composer_names = [
                    c.get("name", "")
                    for c in composers
                    if isinstance(c, dict) and "name" in c
                ]
                if composer_names:
                    staff_data["music_composers"] = composer_names

        return staff_data

    def _extract_comprehensive_rating(self, soup) -> Dict[str, Any]:
        """Extract rating/score from multiple sources with fallbacks."""
        rating_data = {}

        # Priority 1: Look for structured rating displays
        rating_selectors = [
            (".avgRating", {}),
            (".avg-rating", {}),
            (".rating-value", {}),
            (".score-value", {}),
            (".rating .value", {}),
            ("span", {"class": "avgRating"}),
            ("div", {"class": "avgRating"}),
            (".pure-1 .md-1-5", {}),  # Anime-Planet specific
        ]

        for selector, attrs in rating_selectors:
            if "." in selector and " " in selector:
                # Complex CSS selector
                rating_elem = soup.select_one(selector)
            elif "." in selector:
                # Simple class selector
                rating_elem = soup.select_one(selector)
            else:
                rating_elem = soup.find(selector, attrs)

            if rating_elem:
                rating_text = self._clean_text(rating_elem.text)
                # Extract rating (decimal or integer)
                rating_match = re.search(r"(\d+\.?\d*)", rating_text)
                if rating_match:
                    try:
                        rating_data["score"] = float(rating_match.group(1))
                        break
                    except ValueError:
                        continue

        # Look for rating count
        count_selectors = [
            (".rating-count", {}),
            (".vote-count", {}),
            (".num-ratings", {}),
            ("span", {"class": "ratingCount"}),
        ]

        for selector, attrs in count_selectors:
            if "." in selector:
                count_elem = soup.select_one(selector)
            else:
                count_elem = soup.find(selector, attrs)

            if count_elem:
                count_text = self._clean_text(count_elem.text)
                # Extract count number
                count_match = re.search(r"(\d+)", count_text.replace(",", ""))
                if count_match:
                    try:
                        rating_data["score_count"] = int(count_match.group(1))
                        break
                    except ValueError:
                        continue

        return rating_data

    def _extract_ranking(self, soup) -> Dict[str, Any]:
        """Extract ranking information from the page."""
        rank_data = {}

        # Look for ranking displays - common patterns on Anime-Planet
        rank_selectors = [
            (".rank", {}),
            (".ranking", {}),
            (".anime-rank", {}),
            ("span", {"class": "rank"}),
            ("div", {"class": "rank"}),
            (".pure-1.md-1-5", {}),  # Anime-Planet specific rank container
        ]

        for selector, attrs in rank_selectors:
            if "." in selector:
                rank_elem = soup.select_one(selector)
            else:
                rank_elem = soup.find(selector, attrs)

            if rank_elem:
                rank_text = self._clean_text(rank_elem.text)
                # Look for "Rank #N" or "#N" patterns
                rank_matches = [
                    re.search(r"rank\s*#?(\d+)", rank_text, re.IGNORECASE),
                    re.search(r"#(\d+)", rank_text),
                    re.search(r"(\d+)", rank_text),
                ]

                for match in rank_matches:
                    if match:
                        try:
                            rank_data["rank"] = int(match.group(1))
                            return rank_data
                        except ValueError:
                            continue

        return rank_data

    def _extract_alternative_titles(self, soup) -> Dict[str, Any]:
        """Extract alternative titles including English, native, and synonyms."""
        title_data: Dict[str, Any] = {}

        # Look for alternative titles section
        alt_title_containers = [
            soup.find("div", class_="alt-titles"),
            soup.find("section", class_="alt-titles"),
            soup.find("div", class_="alternative-titles"),
            soup.find("div", class_="titles"),
            soup.find("ul", class_="titles"),
        ]

        # Also look for individual title elements
        title_selectors = [
            ("h2", {"class": "alt-title"}),
            ("span", {"class": "english-title"}),
            ("span", {"class": "native-title"}),
            ("span", {"class": "japanese-title"}),
            ("div", {"class": "english"}),
            ("div", {"class": "japanese"}),
        ]

        synonyms = []
        english_title = None
        native_title = None

        # Extract from containers
        for container in alt_title_containers:
            if not container:
                continue

            # Look for different title types
            title_elements = container.find_all(["h2", "h3", "h4", "span", "div", "li"])
            for elem in title_elements:
                title_text = self._clean_text(elem.text)
                if not title_text:
                    continue

                # Determine title type based on content or class
                elem_class = " ".join(elem.get("class", []))
                title_text.lower()

                if "english" in elem_class or self._looks_english(title_text):
                    if not english_title:
                        english_title = title_text
                elif (
                    "native" in elem_class
                    or "japanese" in elem_class
                    or self._looks_japanese(title_text)
                ):
                    if not native_title:
                        native_title = title_text
                else:
                    # Add to synonyms if it's different from main title
                    if title_text not in synonyms:
                        synonyms.append(title_text)

        # Extract from individual selectors
        for selector, attrs in title_selectors:
            elem = soup.find(selector, attrs)
            if elem:
                title_text = self._clean_text(elem.text)
                elem_class = " ".join(elem.get("class", []))

                if "english" in elem_class and not english_title:
                    english_title = title_text
                elif (
                    "native" in elem_class or "japanese" in elem_class
                ) and not native_title:
                    native_title = title_text

        # Set extracted titles
        if english_title:
            title_data["title_english"] = english_title
        if native_title:
            title_data["title_native"] = native_title
        if synonyms:
            title_data["synonyms"] = synonyms[:10]  # Limit to 10 synonyms

        return title_data

    def _looks_english(self, text: str) -> bool:
        """Check if text appears to be English."""
        if not text:
            return False
        # Simple heuristic: if mostly ASCII characters
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.8

    def _looks_japanese(self, text: str) -> bool:
        """Check if text appears to be Japanese."""
        if not text:
            return False
        # Check for Japanese Unicode ranges
        japanese_chars = sum(
            1
            for c in text
            if 0x3040 <= ord(c) <= 0x309F  # Hiragana
            or 0x30A0 <= ord(c) <= 0x30FF  # Katakana
            or 0x4E00 <= ord(c) <= 0x9FAF
        )  # Kanji
        return japanese_chars > 0

    def _extract_studios(self, soup) -> Dict[str, Any]:
        """Extract studio information from multiple sources."""
        studio_data = {}

        # Look for studio information in various locations
        studio_selectors = [
            ("a", {"href": re.compile(r"/studios/")}),
            ("span", {"class": "studio"}),
            ("div", {"class": "studio"}),
            (".studio-name", {}),
            (".production-studio", {}),
        ]

        studios = []
        for selector, attrs in studio_selectors:
            if "." in selector:
                studio_elems = soup.select(selector)
            else:
                studio_elems = soup.find_all(selector, attrs)

            for elem in studio_elems:
                studio_name = self._clean_text(elem.text)
                if studio_name and studio_name not in studios:
                    studios.append(studio_name)

        # Also check info table for studio information
        info_containers = soup.find_all("table")
        for container in info_containers:
            rows = container.find_all("tr")
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    key = self._clean_text(cells[0].text).lower()
                    if "studio" in key or "animation" in key:
                        studio_name = self._clean_text(cells[1].text)
                        if studio_name and studio_name not in studios:
                            studios.append(studio_name)

        if studios:
            studio_data["studios"] = studios[:5]  # Limit to 5 main studios

        return studio_data

    def _extract_enhanced_status(self, soup) -> Dict[str, Any]:
        """Extract status information with enhanced date logic."""
        status_data: Dict[str, Any] = {}

        # Get JSON-LD data for dates
        json_ld = self._extract_json_ld(soup)
        if json_ld:
            start_date = json_ld.get("startDate")
            end_date = json_ld.get("endDate")

            if start_date:
                # Extract year from start date
                year_match = re.search(r"(\d{4})", start_date)
                if year_match:
                    status_data["year"] = int(year_match.group(1))

                # Determine season from start date
                season = self._determine_season_from_date(start_date)
                if season:
                    status_data["season"] = season

            # Determine status from dates
            if start_date and end_date:
                status_data["status"] = "COMPLETED"
            elif start_date and not end_date:
                # Check if start date is in the future
                from datetime import datetime, timezone

                try:
                    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    if start_dt > now:
                        status_data["status"] = "UPCOMING"
                    else:
                        status_data["status"] = "AIRING"
                except:
                    status_data["status"] = "AIRING"
            else:
                status_data["status"] = "UNKNOWN"

        return status_data

    def _determine_season_from_date(self, date_str: str) -> Optional[str]:
        """Determine anime season from date string."""
        if not date_str:
            return None

        # Extract month from date
        month_match = re.search(r"-(\d{2})-", date_str)
        if not month_match:
            return None

        try:
            month = int(month_match.group(1))
            if month in [12, 1, 2]:
                return "WINTER"
            elif month in [3, 4, 5]:
                return "SPRING"
            elif month in [6, 7, 8]:
                return "SUMMER"
            elif month in [9, 10, 11]:
                return "FALL"
        except ValueError:
            pass

        return None

    def _extract_related_anime(self, soup) -> Dict[str, Any]:
        """Extract related anime from same franchise section."""
        related_data: Dict[str, Any] = {}

        # Look for the same franchise relations section
        same_franchise_section = soup.find(id="tabs--relations--anime--same_franchise")

        if not same_franchise_section:
            return related_data

        related_anime = []

        # Look for anime cards in the grid structure
        # Anime-Planet uses pure-u-* grid classes for the cards - direct children only
        card_containers = [
            child
            for child in same_franchise_section.children
            if hasattr(child, "name")
            and child.name == "div"
            and child.get("class")
            and any("pure-u-" in cls for cls in child.get("class", []))
        ]

        for container in card_containers:
            # Look for the anime link and title within each card
            anime_link = container.find("a", href=re.compile(r"/anime/"))

            if anime_link:
                # Extract title from h3.cardName or fallback to link text
                title_elem = anime_link.find("h3", class_="cardName")
                if title_elem:
                    title = self._clean_text(title_elem.text)
                else:
                    title = self._clean_text(anime_link.text)

                # Extract slug from href
                href = anime_link.get("href", "")
                slug_match = re.search(r"/anime/([^/?]+)", href)
                slug = slug_match.group(1) if slug_match else None

                # Extract metadata from the card's text content
                all_text = container.get_text(separator=" | ", strip=True)

                # Parse the structured text format: "Title | RelationType | StartDate | - | EndDate | MediaType: Episodes"
                parts = [part.strip() for part in all_text.split(" | ")]

                relation_subtype = None
                start_date = None
                end_date = None
                media_type = None
                episode_count = None

                if len(parts) >= 6:
                    # Extract relation subtype (Recap, Omake, Side Story, etc.)
                    if len(parts) > 1 and parts[1] not in ["-", ""]:
                        relation_subtype = parts[1]

                    # Extract start date
                    if len(parts) > 2 and re.match(r"\d{4}-\d{2}-\d{2}", parts[2]):
                        start_date = parts[2]

                    # Extract end date (usually after a "-")
                    if len(parts) > 4 and re.match(r"\d{4}-\d{2}-\d{2}", parts[4]):
                        end_date = parts[4]

                    # Extract media type and episode count from last part
                    if len(parts) > 5:
                        type_ep_text = parts[5]
                        # Pattern: "TV Special: 1 ep" or "OVA: 3 ep" or "Movie"
                        type_match = re.match(
                            r"([^:]+)(?::\s*(\d+)\s*ep)?", type_ep_text
                        )
                        if type_match:
                            media_type = type_match.group(1).strip()
                            if type_match.group(2):
                                episode_count = int(type_match.group(2))

                # Extract year from start date if available
                year = None
                if start_date:
                    year_match = re.search(r"(\d{4})", start_date)
                    if year_match:
                        year = int(year_match.group(1))

                if title and slug:
                    related_item = {
                        "title": title,
                        "slug": slug,
                        "url": f"https://www.anime-planet.com{href}",
                        "relation_type": "same_franchise",
                    }

                    # Add parsed metadata
                    if relation_subtype:
                        related_item["relation_subtype"] = relation_subtype
                    if media_type:
                        related_item["type"] = media_type
                    if year:
                        related_item["year"] = year
                    if episode_count:
                        related_item["episodes"] = episode_count
                    if start_date:
                        related_item["start_date"] = start_date
                    if end_date:
                        related_item["end_date"] = end_date

                    related_anime.append(related_item)

        if related_anime:
            related_data["related_anime"] = related_anime
            related_data["related_count"] = len(related_anime)

        return related_data
