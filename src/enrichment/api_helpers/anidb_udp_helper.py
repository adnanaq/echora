#!/usr/bin/env python3
"""
AniDB UDP API Helper for Character Data Retrieval

Helper function to fetch character data from AniDB UDP API.
"""

import argparse
import asyncio
import json
import logging
import os
import socket
import time
from typing import Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AniDBUDPHelper:
    """Helper for AniDB UDP API character data fetching."""

    def __init__(self) -> None:
        """Initialize AniDB UDP helper."""
        self.host = "api.anidb.net"
        self.port = 9000
        self.client_name = os.getenv("ANIDB_CLIENT")
        self.client_version = os.getenv("ANIDB_CLIENTVER")
        self.protocol_version = os.getenv("ANIDB_PROTOVER", "1")
        self.socket = None
        self.session_key = None
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests (0.5 req/sec)

    def _wait_for_rate_limit(self) -> None:
        """Ensure we don't exceed AniDB rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
            time.sleep(wait_time)

        self.last_request_time = time.time()

    def _send_command(self, command: str) -> str | None:
        """Send UDP command to AniDB API."""
        self._wait_for_rate_limit()

        try:
            if not self.socket:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.socket.settimeout(30)  # 30 second timeout

            # Add session key if authenticated
            if self.session_key:
                command += f"&s={self.session_key}"

            logger.info(f"Sending UDP command: {command}")

            # Send command
            self.socket.sendto(command.encode("utf-8"), (self.host, self.port))

            # Receive response
            response, _ = self.socket.recvfrom(8192)
            response_str = response.decode("utf-8").strip()

            logger.info(f"UDP response: {response_str[:200]}...")
            return response_str

        except Exception as e:
            logger.error(f"UDP command failed: {e}")
            return None

    def authenticate(self) -> bool:
        """Authenticate with AniDB UDP API."""
        # For public data, we can try without authentication first
        # If authentication is required, we would need username/password
        logger.info("Attempting to access AniDB UDP API without authentication")
        return True

    def get_anime_character_ids(self, anidb_id: int) -> list[int] | None:
        """Get character IDs for an anime using ANIME command with character amask."""
        try:
            # ANIME command with amask for character id list (Byte 6, Bit 7 = 0x80)
            # We need to request with the character amask bit set
            command = f"ANIME aid={anidb_id}&amask=80000000"  # Byte 6, Bit 7

            response = self._send_command(command)
            if not response:
                return None

            # Parse response
            if response.startswith("220"):  # ANIME response code
                parts = response.split("|")
                if len(parts) > 1:
                    # Character IDs should be in the response based on amask
                    # Format may vary, need to parse based on actual response
                    logger.info(f"ANIME response parts: {len(parts)}")
                    for i, part in enumerate(parts):
                        logger.info(f"Part {i}: {part}")

                    # Character IDs are typically comma-separated in the response
                    # The exact position depends on the amask configuration
                    return self._extract_character_ids_from_response(parts)

            return None

        except Exception as e:
            logger.error(f"Failed to get character IDs for anime {anidb_id}: {e}")
            return None

    def _extract_character_ids_from_response(self, parts: list[str]) -> list[int]:
        """Extract character IDs from ANIME response."""
        # This will need to be adjusted based on actual response format
        # For now, try to find comma-separated numbers
        for part in parts:
            if "," in part and part.replace(",", "").replace(" ", "").isdigit():
                try:
                    char_ids = [
                        int(x.strip()) for x in part.split(",") if x.strip().isdigit()
                    ]
                    if char_ids:
                        logger.info(f"Found character IDs: {char_ids}")
                        return char_ids
                except ValueError:
                    continue

        logger.warning("No character IDs found in response")
        return []

    def get_character_details(self, char_id: int) -> dict[str, Any] | None:
        """Get detailed character information by character ID."""
        try:
            command = f"CHARACTER charid={char_id}"
            response = self._send_command(command)

            if not response:
                return None

            if response.startswith("235"):  # CHARACTER response code
                # Parse character response
                parts = response.split("|")
                if len(parts) >= 9:
                    character_data = {
                        "anidb_character_id": char_id,
                        "name_kanji": parts[1] if len(parts) > 1 else None,
                        "name_transcription": parts[2] if len(parts) > 2 else None,
                        "picture": parts[3] if len(parts) > 3 else None,
                        "anime_blocks": parts[4] if len(parts) > 4 else None,
                        "episode_list": parts[5] if len(parts) > 5 else None,
                        "last_update": parts[6] if len(parts) > 6 else None,
                        "type": parts[7] if len(parts) > 7 else None,
                        "gender": parts[8] if len(parts) > 8 else None,
                    }
                    return character_data

            return None

        except Exception as e:
            logger.error(f"Failed to get character details for ID {char_id}: {e}")
            return None

    def get_all_characters_for_anime(
        self, anidb_id: int
    ) -> list[dict[str, Any]] | None:
        """Get all character data for an anime."""
        try:
            # First get character IDs
            char_ids = self.get_anime_character_ids(anidb_id)
            if not char_ids:
                logger.warning(f"No character IDs found for anime {anidb_id}")
                return []

            logger.info(f"Found {len(char_ids)} character IDs for anime {anidb_id}")

            # Then get details for each character
            characters = []
            for char_id in char_ids:
                logger.info(f"Fetching character details for ID: {char_id}")
                char_data = self.get_character_details(char_id)
                if char_data:
                    characters.append(char_data)
                else:
                    logger.warning(f"Failed to get data for character ID: {char_id}")

            logger.info(f"Successfully fetched {len(characters)} character details")
            return characters

        except Exception as e:
            logger.error(f"Failed to get characters for anime {anidb_id}: {e}")
            return None

    def close(self) -> None:
        """Close UDP socket."""
        if self.socket:
            self.socket.close()


async def main() -> None:
    """Main function for testing AniDB UDP character data fetching."""
    parser = argparse.ArgumentParser(
        description="Test AniDB UDP character data fetching"
    )
    parser.add_argument("--anidb-id", type=int, required=True, help="AniDB anime ID")
    parser.add_argument(
        "--output",
        type=str,
        default="test_anidb_udp_characters.json",
        help="Output file path",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    helper = AniDBUDPHelper()

    try:
        # Authenticate (if needed)
        if not helper.authenticate():
            logger.error("Authentication failed")
            return

        # Get character data
        characters = helper.get_all_characters_for_anime(args.anidb_id)

        if characters:
            # Save to file
            output_data = {
                "anidb_id": args.anidb_id,
                "character_count": len(characters),
                "characters": characters,
            }

            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(
                f"Successfully saved {len(characters)} characters to {args.output}"
            )
        else:
            logger.error("No character data retrieved")

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
    finally:
        helper.close()


if __name__ == "__main__":
    asyncio.run(main())
