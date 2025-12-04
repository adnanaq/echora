"""AniSearch scraper for anime metadata extraction.

Provides anime details, descriptions (multi-language), and metadata from AniSearch.
Uses cloudscraper for reliable access without Cloudflare blocking.
"""

import logging
import re
from typing import Any, Dict, Optional

from bs4 import BeautifulSoup

from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class AniSearchScraper(BaseScraper):
    """Scraper for AniSearch anime and character data."""

    def __init__(self):
        """Initialize AniSearch scraper."""
        super().__init__(service_name="anisearch")
        self.base_url = "https://www.anisearch.com"

    async def get_anime_by_id(
        self, anime_id: int, slug: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get anime details by AniSearch ID.

        Args:
            anime_id: AniSearch anime ID
            slug: Optional anime slug (e.g., 'dan-da-dan')

        Returns:
            Dict with anime data including:
            - Basic info (title, image, episodes, dates)
            - Descriptions (multiple languages)
            - Genres
            - Ratings
            - JSON-LD structured data
            None if anime not found
        """
        try:
            # Build URL (slug is optional)
            if slug:
                url = f"{self.base_url}/anime/{anime_id},{slug}"
            else:
                url = f"{self.base_url}/anime/{anime_id}"

            logger.debug(f"Fetching AniSearch anime: {url}")

            # Make request
            response = await self._make_request(url, timeout=15)

            # Check for errors
            if response.get("status_code") != 200:
                logger.warning(
                    f"AniSearch returned status {response.get('status_code')} for anime {anime_id}"
                )
                return None

            # Parse HTML
            soup = self._parse_html(response["content"])

            # Extract data
            anime_data = await self._parse_anime_page(soup, anime_id, url)

            if anime_data:
                logger.info(
                    f"Successfully scraped anime '{anime_data.get('title')}' from AniSearch"
                )
                return anime_data

            return None

        except Exception as e:
            logger.error(f"Error scraping AniSearch anime {anime_id}: {e}")
            return None

    async def _parse_anime_page(
        self, soup: BeautifulSoup, anime_id: int, url: str
    ) -> Optional[Dict[str, Any]]:
        """
        Extract structured anime metadata from an AniSearch anime page.
        
        Parameters:
            soup (BeautifulSoup): Parsed HTML of the anime page.
            anime_id (int): AniSearch numeric identifier for the anime.
            url (str): Original page URL for the anime.
        
        Returns:
            Dict[str, Any]: Dictionary of extracted anime fields (for example: `anisearch_id`, `url`, `title`, `title_japanese`, `description`, `cover`, `episodes`, `start_date`, `end_date`, `genres`, `rating`, `detailed_ratings`, `studios`, `regional_publishers`, `tags`, `relations`, `synonyms`, `external_links`, `staff`, `characters`, `screenshots`, `trailers`) if parsing succeeds; `None` if required data (such as `title`) is missing or parsing fails.
        """
        try:
            anime_data: Dict[str, Any] = {
                "anisearch_id": anime_id,
                "url": url,
            }

            # 1. Extract JSON-LD structured data (primary source)
            json_ld = self._extract_json_ld(soup)
            if json_ld:
                anime_data["title"] = json_ld.get("name")
                anime_data["cover"] = json_ld.get("image")
                anime_data["episodes"] = json_ld.get("numberOfEpisodes")
                anime_data["start_date"] = json_ld.get("startDate")
                anime_data["end_date"] = json_ld.get("endDate")
                anime_data["genres"] = json_ld.get("genre", [])

                # Extract rating data
                rating_data = json_ld.get("aggregateRating", {})
                if rating_data:
                    anime_data["rating"] = {
                        "score": rating_data.get("ratingValue"),
                        "count": rating_data.get("ratingCount"),
                        "best": rating_data.get("bestRating"),
                        "worst": rating_data.get("worstRating"),
                    }

            # 2. Extract title from H1 (fallback if JSON-LD missing)
            if not anime_data.get("title"):
                h1 = soup.find("h1")
                if h1:
                    anime_data["title"] = h1.get_text().strip()

            # 2a. Extract Japanese title from subheader
            japanese_title = self._extract_japanese_title(soup)
            if japanese_title:
                anime_data["title_japanese"] = japanese_title

            # 3. Extract English description only
            description = self._extract_english_description(soup)
            if description:
                anime_data["description"] = description

            # 4. Extract comprehensive metadata from header fields
            metadata = self._extract_header_metadata(soup)
            if metadata:
                anime_data.update(metadata)

            # 5. Extract studio/production info
            studios = self._extract_studios(soup)
            if studios:
                anime_data["studios"] = studios

            # 5a. Extract regional publishers
            publishers = self._extract_publishers(soup)
            if publishers:
                anime_data["regional_publishers"] = publishers

            # 5b. Extract tags (separate from genres)
            tags = self._extract_tags(soup)
            if tags:
                anime_data["tags"] = tags

            # 5c. Extract detailed ratings (calculated value, rankings)
            detailed_ratings = self._extract_detailed_ratings(soup)
            if detailed_ratings:
                anime_data["detailed_ratings"] = detailed_ratings

            # 6. Extract relations (sequels, prequels, etc.)
            relations = self._extract_relations(soup)
            if relations:
                anime_data["relations"] = relations

            # 7. Extract synonyms/alternative titles
            synonyms = self._extract_synonyms(soup)
            if synonyms:
                anime_data["synonyms"] = synonyms

            # 8. Extract external links (official website, streaming platforms)
            external_links = self._extract_external_links(soup)
            if external_links:
                anime_data["external_links"] = external_links

            # 8. Extract staff information
            staff = self._extract_staff(soup)
            if staff:
                anime_data["staff"] = staff

            # 9. Extract characters from /characters page
            characters = await self._extract_characters(anime_id)
            if characters:
                anime_data["characters"] = characters

            # 10. Extract all images (screenshots only, excluding cover to avoid duplication)
            screenshots = await self._extract_all_images(
                soup, anime_id, anime_data.get("cover")
            )
            if screenshots:
                anime_data["screenshots"] = screenshots

            # 11. Extract trailers
            trailers = self._extract_trailers(soup)
            if trailers:
                anime_data["trailers"] = trailers

            # Validate we have minimum required data
            if not anime_data.get("title"):
                logger.warning(f"Failed to extract title for anime {anime_id}")
                return None

            return anime_data

        except Exception as e:
            logger.error(f"Error parsing anime page: {e}")
            return None

    def _extract_japanese_title(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract the Japanese title from the page subheader.
        
        Scans the first span with class "subheader" and returns the text portion after a "/" delimiter if that portion contains Japanese characters (hiragana, katakana, or kanji).
        
        Returns:
            str: Japanese title if found, `None` otherwise.
        """
        import re

        # Find subheader span
        subheader = soup.find("span", class_="subheader")
        if subheader:
            text = subheader.get_text().strip()
            # Format is usually "English Title / 日本語タイトル"
            if "/" in text:
                parts = text.split("/")
                if len(parts) >= 2:
                    japanese_part = parts[-1].strip()
                    # Verify it contains Japanese characters
                    if re.search(
                        r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", japanese_part
                    ):
                        return japanese_part

        return None

    def _extract_english_description(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Selects and returns the English description text from a page's multi-language description blocks.
        
        Searches description blocks (elements with classes "textblock" and/or "details-text"), ignores empty or very short entries, and heuristically filters out German and Japanese content; returns the first block that appears to be English.
        
        Returns:
            The English description string if found, otherwise None.
        """
        # Find all description divs
        desc_divs = soup.find_all("div", class_=["textblock", "details-text"])

        for div in desc_divs:
            text = div.get_text().strip()

            # Skip empty or very short text
            if not text or len(text) < 50:
                continue

            # Check if this is English (not German/Japanese)
            # German indicators
            german_words = ["auf den ersten", "oberschule", "außerirdische", "dass"]
            is_german = any(word in text.lower() for word in german_words)

            # Japanese indicators
            is_japanese = bool(
                re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text)
            )

            # If not German or Japanese, assume English
            if not is_german and not is_japanese:
                return text

        return None

    def _extract_header_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract header metadata fields such as type, status, adapted source, and season.
        
        Only fields present in the page are returned; values are the extracted text as-is.
        
        Returns:
            metadata (Dict[str, Any]): Mapping of metadata keys to their string values. Possible keys include
            `type`, `status`, `adapted_from`, and `season`.
        """
        metadata = {}

        # Find all header spans and their values
        headers = soup.find_all("span", class_="header")

        for header in headers:
            label = header.get_text().strip().replace(":", "")

            # Get the value (next sibling or parent's text)
            value_elem = header.next_sibling
            value = None

            if value_elem:
                if isinstance(value_elem, str):
                    value = value_elem.strip()
                elif hasattr(value_elem, "get_text"):
                    value = value_elem.get_text().strip()
                elif hasattr(value_elem, "strip"):
                    value = str(value_elem).strip()

            if not value:
                # Try parent element
                parent = header.find_parent(["div", "li"])
                if parent:
                    value = parent.get_text().replace(label + ":", "").strip()

            if not value:
                continue

            # Map fields
            label_lower = label.lower()
            if "type" in label_lower:
                metadata["type"] = value
            elif "status" in label_lower:
                metadata["status"] = value
            elif "adapted from" in label_lower or "source" in label_lower:
                metadata["adapted_from"] = value
            elif "season" in label_lower:
                metadata["season"] = value

        return metadata

    def _extract_studios(self, soup: BeautifulSoup) -> list[str]:
        """
        Collect studio and production company names from the provided page soup.
        
        Returns:
            studios (list[str]): Unique studio or production company names in the order they appear.
        """
        studios = []

        # Find company divs with Studio header
        company_elements = soup.find_all(class_="company")

        for elem in company_elements:
            header = elem.find("span", class_="header")
            if header and "Studio" in header.get_text():
                # Extract all studio links (href contains "company" without leading slash)
                studio_links = elem.find_all(
                    "a", href=lambda x: x and "company" in str(x)
                )
                for link in studio_links:
                    studio_name = link.get_text().strip()
                    if studio_name and studio_name not in studios:
                        studios.append(studio_name)

        return studios

    def _extract_detailed_ratings(
        self, soup: BeautifulSoup
    ) -> Optional[Dict[str, Any]]:
        """
        Extract detailed rating metrics from the anime's ratings section.
        
        Scans the ratings block for a displayed calculated rating, the aggregated number of ratings derived from the star distribution, and ranking positions (toplist, popular, trending).
        
        Returns:
            ratings (dict): Mapping of extracted metrics. Possible keys:
                - `calculated_value` (str): Calculated rating value as shown (e.g., "4.17").
                - `total_count` (int): Sum of votes across the star distribution.
                - `toplist_rank` (int): Position in the toplist, if present.
                - `popular_rank` (int): Position in the popular list, if present.
                - `trending_rank` (int): Position in the trending list, if present.
            Returns `None` if no ratings information is found.
        """
        import re

        ratings_section = soup.find(id="ratings")
        if not ratings_section:
            return None

        text = ratings_section.get_text()
        ratings_data = {}

        # Extract calculated value (e.g., "4.17")
        calc_match = re.search(r"Calculated Value([\d.]+)", text)
        if calc_match:
            ratings_data["calculated_value"] = calc_match.group(1)

        # Calculate total count from star distribution
        stars = ratings_section.find_all("li")
        total_count = 0
        for star in stars:
            value_div = star.find("div", class_="value")
            if value_div:
                try:
                    total_count += int(value_div.get_text().strip())
                except ValueError:
                    pass

        if total_count > 0:
            ratings_data["total_count"] = total_count

        # Extract rankings
        toplist_match = re.search(r"Toplist#(\d+)", text)
        if toplist_match:
            ratings_data["toplist_rank"] = int(toplist_match.group(1))

        popular_match = re.search(r"Popular#(\d+)", text)
        if popular_match:
            ratings_data["popular_rank"] = int(popular_match.group(1))

        trending_match = re.search(r"Trending#(\d+)", text)
        if trending_match:
            ratings_data["trending_rank"] = int(trending_match.group(1))

        return ratings_data if ratings_data else None

    def _extract_tags(self, soup: BeautifulSoup) -> list[str]:
        """Extract tags (class='gt') from cloud, excluding genres.

        Genres are extracted from JSON-LD. Tags are in the cloud
        with class='gt' (while genres have class='gg' or 'gc').

        Returns:
            List of tag names
        """
        tags = []

        # Find cloud element containing genres and tags
        cloud = soup.find(class_="cloud")
        if cloud:
            # Extract only tags (class='gt'), not genres (class='gg' or 'gc')
            tag_links = cloud.find_all("a", class_="gt")
            for link in tag_links:
                tag_name = link.get_text().strip()
                if tag_name and tag_name not in tags:
                    tags.append(tag_name)

        return tags

    def _extract_publishers(self, soup: BeautifulSoup) -> list[list[str]]:
        """
        Extract regional publisher names grouped by region.
        
        Scans company sections labeled "Publisher" and collects publisher names from links within each region block.
        
        Returns:
            list[list[str]]: A list where each element is a list of publisher names for a specific region.
        """
        regional_publishers = []

        # Find company divs with Publisher header
        company_elements = soup.find_all(class_="company")

        for elem in company_elements:
            header = elem.find("span", class_="header")
            if header and "Publisher" in header.get_text():
                # Extract all publisher links (href contains "company" without leading slash)
                publisher_links = elem.find_all(
                    "a", href=lambda x: x and "company" in str(x)
                )
                publishers = [link.get_text().strip() for link in publisher_links]

                if publishers:
                    regional_publishers.append(publishers)

        return regional_publishers

    def _extract_relations(self, soup: BeautifulSoup) -> list[Dict[str, Any]]:
        """Extract related anime from relations section.

        Returns:
            List of dicts with relation type, title, url, and other info
        """
        relations = []

        # Find relations section
        relations_section = soup.find(id="relations")
        if not relations_section:
            return relations

        # Find all relation items (li elements with anime links)
        relation_items = relations_section.find_all("li", class_="swiper-slide")

        for item in relation_items:
            # Get relation type from header span
            header_span = item.find("span", class_="header")
            if not header_span:
                continue

            relation_type = header_span.get_text().strip()

            # Try anime link first
            anime_link = item.find("a", class_="anime-item")

            if anime_link:
                # Extract anime relation
                title_span = anime_link.find("span", class_="title")
                if not title_span:
                    continue

                title = title_span.get_text().strip()

                # Build relation data
                relation_data = {
                    "relation_type": relation_type,
                    "title": title,
                    "media_type": "anime",
                }

                # Get URL
                href = anime_link.get("href", "")
                if href:
                    if not href.startswith("http"):
                        href = (
                            f"{self.base_url}/{href}"
                            if not href.startswith("/")
                            else f"{self.base_url}{href}"
                        )
                    relation_data["url"] = href

                # Get date/type info
                date_span = anime_link.find("span", class_="date")
                if date_span:
                    relation_data["info"] = date_span.get_text().strip()

                # Get image
                image_url = anime_link.get("data-bg", "")
                if image_url:
                    if not image_url.startswith("http"):
                        image_url = f"https://cdn.anisearch.com/images/{image_url}"
                    relation_data["image"] = image_url

                relations.append(relation_data)
            else:
                # Try manga link
                manga_link = item.find("a", href=lambda x: x and "manga/" in str(x))

                if manga_link:
                    # Extract title from the link text (format: "Manga, info...TitlePublisher")
                    link_text = manga_link.get_text().strip()

                    # Try to parse: find title between info and publisher
                    # Format: "Manga, 20+/190+ (2021)DandadanShuueisha Inc."
                    import re

                    # Extract title - it's between the date and company
                    title_match = re.search(
                        r"\)\s*(.+?)\s*(?:[A-Z][a-z]+\s+Inc\.|$)", link_text
                    )
                    title = (
                        title_match.group(1)
                        if title_match
                        else link_text.split(")")[-1].strip()
                    )

                    relation_data = {
                        "relation_type": relation_type,
                        "title": title,
                        "media_type": "manga",
                    }

                    # Get URL
                    href = manga_link.get("href", "")
                    if href:
                        if not href.startswith("http"):
                            href = (
                                f"{self.base_url}/{href}"
                                if not href.startswith("/")
                                else f"{self.base_url}{href}"
                            )
                        relation_data["url"] = href

                    # Extract info (everything before title)
                    info_match = re.search(r"^(.+?)\)", link_text)
                    if info_match:
                        relation_data["info"] = info_match.group(0)

                    relations.append(relation_data)

        return relations

    def _extract_synonyms(self, soup: BeautifulSoup) -> list[str]:
        """
        Extract alternative titles (synonyms) from the page.
        
        Returns:
            synonyms (list[str]): List of synonym strings found on the page.
        """
        synonyms = []

        # Find Synonyms header
        headers = soup.find_all("span", class_="header")
        for header in headers:
            if "synonym" in header.get_text().lower():
                # Get the parent div and extract value
                parent = header.find_parent(["div", "li"])
                if parent:
                    value = parent.get_text().replace("Synonyms:", "").strip()
                    if value:
                        synonyms.append(value)

        return synonyms

    def _extract_external_links(self, soup: BeautifulSoup) -> Dict[str, str]:
        """
        Collects external (non-AniSearch) links from the parsed page and classifies common platforms.
        
        Searches anchor tags with http(s) hrefs that do not contain "anisearch" and maps recognized targets to short keys (e.g., `netflix`, `crunchyroll`, `twitter`, `facebook`, `youtube`, `discord`). If no specific platform is matched, the first substantial external link is assigned to `official_website`.
        
        Parameters:
            soup (BeautifulSoup): Parsed HTML of the anime page.
        
        Returns:
            Dict[str, str]: A mapping from link type to URL (keys include `netflix`, `crunchyroll`, `twitter`, `facebook`, `youtube`, `discord`, `official_website` when present).
        """
        links = {}

        # Find external links (non-anisearch URLs)
        external = soup.find_all(
            "a", href=lambda x: x and x.startswith("http") and "anisearch" not in x
        )

        for link in external:
            href = link.get("href", "")
            text = link.get_text().strip()

            # Skip AniSearch's own social media (these have empty text or contain "aniSearch")
            if not text or "anisearch" in text.lower():
                continue

            # Categorize links
            if "netflix.com" in href:
                links["netflix"] = href
            elif "crunchyroll.com" in href:
                links["crunchyroll"] = href
            elif "twitter.com" in href or "x.com" in href:
                links["twitter"] = href
            elif "facebook.com" in href:
                links["facebook"] = href
            elif "youtube.com" in href or "youtu.be" in href:
                if "youtube" not in links:
                    links["youtube"] = href
            elif "discord" in href:
                if "discord" not in links:
                    links["discord"] = href
            elif text and len(text) > 5 and "official_website" not in links:
                # First substantial link is likely official website
                links["official_website"] = href

        return links

    def _extract_staff(self, soup: BeautifulSoup) -> list[Dict[str, str]]:
        """
        Extract staff members referenced on the page.
        
        Parses links to person pages and returns up to 20 unique staff entries containing the person's name and their role (inferred from nearby parenthesis text, or "Unknown" when not found).
        
        Returns:
            list[dict[str, str]]: A list of dictionaries with keys `name` and `role`. Limited to the first 20 unique entries.
        """
        staff = []

        # Find person links
        person_links = soup.find_all("a", href=lambda x: x and "/person/" in str(x))

        for link in person_links:
            name = link.get_text().strip()
            if not name:
                continue

            # Try to find role nearby
            role = "Unknown"
            parent = link.find_parent(["div", "li"])
            if parent:
                parent_text = parent.get_text().strip()
                if "(" in parent_text and ")" in parent_text:
                    # Extract role from parentheses
                    role_match = re.search(r"\(([^)]+)\)", parent_text)
                    if role_match:
                        role = role_match.group(1)

            staff_entry = {"name": name, "role": role}
            if staff_entry not in staff:
                staff.append(staff_entry)

        return staff[:20]  # Limit to top 20

    async def _extract_characters(self, anime_id: int) -> list[Dict[str, Any]]:
        """
        Extract character entries from the anime's characters page on AniSearch.
        
        Parameters:
            anime_id (int): AniSearch numeric ID of the anime.
        
        Returns:
            characters (list[dict]): List of character objects. Each dict contains:
                - name (str): Character's display name.
                - url (str, optional): Absolute URL to the character page.
                - image (str, optional): Absolute URL to the character image.
                - role (str, optional): Character role or section label (e.g., "Main", "Supporting").
                - favorites (int, optional): Number of favorites for the character.
        """
        import re

        characters = []

        try:
            # Fetch characters page
            characters_url = f"{self.base_url}/anime/{anime_id}/characters"
            logger.debug(f"Fetching characters from: {characters_url}")

            response = await self._make_request(characters_url, timeout=15)

            if response.get("status_code") == 200:
                soup = self._parse_html(response["content"])

                # Find character links
                char_links = soup.find_all(
                    "a", href=lambda x: x and "character/" in str(x)
                )

                for link in char_links:
                    # Extract character name from title span
                    title_span = link.find("span", class_="title")
                    if not title_span:
                        continue

                    name = title_span.get_text().strip()
                    if not name:
                        continue

                    # Extract character data
                    char_data = {"name": name}

                    # Get character role/section by finding previous header
                    parent = link.parent
                    role = None
                    for _ in range(5):  # Check up to 5 levels up
                        if parent:
                            prev_sibling = parent.find_previous_sibling(
                                ["h2", "h3", "h4"]
                            )
                            if prev_sibling:
                                role = prev_sibling.get_text().strip()
                                # Remove "Character" text from role (e.g., "Main Character" -> "Main")
                                role = role.replace(" Character", "")
                                break
                            parent = parent.parent

                    if role:
                        char_data["role"] = role

                    # Get character URL
                    href = link.get("href", "")
                    if href:
                        if not href.startswith("http"):
                            href = (
                                f"{self.base_url}/{href}"
                                if not href.startswith("/")
                                else f"{self.base_url}{href}"
                            )
                        char_data["url"] = href

                    # Get character image from data-bg
                    image_url = link.get("data-bg", "")
                    if image_url:
                        if not image_url.startswith("http"):
                            # Convert to full CDN URL
                            image_url = f"https://cdn.anisearch.com/images/{image_url}"
                        char_data["image"] = image_url

                    # Get favorites count
                    favorites_span = link.find("span", class_="favorites")
                    if favorites_span:
                        fav_text = favorites_span.get_text().strip()
                        # Extract number from "55 ❤" format
                        fav_match = re.search(r"(\d+)", fav_text)
                        if fav_match:
                            char_data["favorites"] = int(fav_match.group(1))

                    characters.append(char_data)

                logger.debug(f"Found {len(characters)} characters")

        except Exception as e:
            logger.warning(f"Error fetching characters: {e}")

        return characters

    async def _extract_all_images(
        self, soup: BeautifulSoup, anime_id: int, cover_url: Optional[str] = None
    ) -> list[str]:
        """
        Retrieve screenshot image URLs for an anime, excluding the cover image.
        
        Parses the anime's screenshots page on AniSearch and collects full-size CDN image links.
        If `cover_url` is provided, that URL is excluded from the results; duplicate URLs are deduplicated.
        
        Parameters:
            cover_url (Optional[str]): Cover image URL to exclude from the returned screenshots (if present).
        
        Returns:
            list[str]: Screenshot image URLs (does not include the cover image when `cover_url` is provided).
        """
        images = []

        # Fetch screenshots page for all screenshot images
        try:
            screenshots_url = f"{self.base_url}/anime/{anime_id}/screenshots"
            logger.debug(f"Fetching screenshots from: {screenshots_url}")

            response = await self._make_request(screenshots_url, timeout=15)

            if response.get("status_code") == 200:
                screenshots_soup = self._parse_html(response["content"])

                # Find all links to full-size screenshot images
                screenshot_links = screenshots_soup.find_all(
                    "a",
                    href=lambda x: x
                    and "cdn.anisearch.com/images/anime/screen" in str(x),
                )

                for link in screenshot_links:
                    href = link.get("href", "")
                    if href and href.startswith("http"):
                        # Exclude cover image to avoid duplication
                        if cover_url and href == cover_url:
                            continue
                        if href not in images:
                            images.append(href)

                logger.debug(f"Found {len(images)} screenshot images")

        except Exception as e:
            logger.warning(f"Error fetching screenshots: {e}")

        return images

    def _extract_trailers(self, soup: BeautifulSoup) -> list[str]:
        """
        Collect YouTube trailer URLs from the provided BeautifulSoup document.
        
        Returns:
            trailers (list[str]): Trailer URLs found in anchor hrefs and iframe src attributes; duplicates removed.
        """
        trailers = []

        # Find YouTube embeds or links
        youtube_links = soup.find_all(
            "a", href=lambda x: x and ("youtube.com" in str(x) or "youtu.be" in str(x))
        )

        for link in youtube_links:
            href = link.get("href", "")
            if href and href not in trailers:
                trailers.append(href)

        # Find iframe embeds
        iframes = soup.find_all("iframe", src=lambda x: x and "youtube.com" in str(x))
        for iframe in iframes:
            src = iframe.get("src", "")
            if src and src not in trailers:
                trailers.append(src)

        return trailers

    async def search_anime(
        self, query: str, limit: int = 10
    ) -> Optional[list[Dict[str, Any]]]:
        """
        Attempt to search AniSearch for anime titles (not supported via scraping).
        
        This method is a placeholder because AniSearch blocks external AJAX search requests; use direct anime ID lookup instead.
        
        Parameters:
            query (str): Search query string.
            limit (int): Maximum number of results to return (ignored; method returns None).
        
        Returns:
            None: Always returns `None` because external search is not available.
        """
        logger.warning(
            "AniSearch search is not available - AJAX endpoint blocks external access. "
            "Use direct anime ID access instead."
        )
        return None

    def _parse_search_results(
        self, soup: BeautifulSoup, limit: int
    ) -> list[Dict[str, Any]]:
        """
        Extracts basic anime entries from a search results page.
        
        Parameters:
            soup (BeautifulSoup): Parsed HTML of an AniSearch search results page.
            limit (int): Maximum number of results to return.
        
        Returns:
            list[Dict[str, Any]]: A list of result dictionaries. Each dictionary contains:
                - `anisearch_id` (int): AniSearch numeric ID for the anime.
                - `title` (str): Display title of the anime.
                - `url` (str): Absolute or site-relative URL to the anime page.
                - `image` (str, optional): URL of an associated image if found.
        """
        results = []

        # Find anime links in search results
        anime_links = soup.find_all("a", href=lambda x: x and "/anime/" in str(x))

        seen_ids = set()

        for link in anime_links:
            if len(results) >= limit:
                break

            href = link.get("href", "")

            # Extract anime ID from URL
            match = re.search(r"/anime/(\d+)", href)
            if not match:
                continue

            anime_id = int(match.group(1))

            # Skip duplicates
            if anime_id in seen_ids:
                continue

            seen_ids.add(anime_id)

            # Extract title
            title = link.get_text().strip()
            if not title or len(title) < 2:
                continue

            # Build result
            result = {
                "anisearch_id": anime_id,
                "title": title,
                "url": (
                    f"{self.base_url}{href}" if not href.startswith("http") else href
                ),
            }

            # Try to find image nearby
            parent = link.find_parent(["div", "li"])
            if parent:
                img = parent.find("img")
                if img:
                    result["image"] = img.get("src")

            results.append(result)

        return results