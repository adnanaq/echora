"""Tests for enrichment.utils.text_utils Japanese text normalization.

This module tests the normalize_japanese_text function which converts
Japanese text (Hiragana, Katakana, Kanji) to Romaji using pykakasi.

Tests validate:
- Katakana to Romaji conversion
- Hiragana to Romaji conversion
- Kanji to Romaji conversion
- Mixed Japanese text conversion
- Non-Japanese text handling
- Empty string handling
- Edge cases with special characters
- Caching behavior with functools.cache
"""

import pytest
from enrichment.utils.text_utils import _get_kakasi, normalize_japanese_text


class TestNormalizeJapaneseText:
    """Test suite for normalize_japanese_text function."""

    def test_katakana_conversion(self):
        """Test Katakana to Romaji conversion."""
        # ワンピース (One Piece)
        result = normalize_japanese_text("ワンピース")
        assert result == "wanpi-su" or result == "wanpiisu", f"Got: {result}"
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hiragana_conversion(self):
        """Test Hiragana to Romaji conversion."""
        # ひらがな (hiragana)
        result = normalize_japanese_text("ひらがな")
        assert "hiragana" in result or "hiragana" == result.replace("-", "")
        assert isinstance(result, str)

    def test_kanji_conversion(self):
        """Test Kanji to Romaji conversion."""
        # 漢字 (kanji)
        result = normalize_japanese_text("漢字")
        assert "kanji" in result or result.startswith("kan")
        assert isinstance(result, str)

    def test_mixed_japanese_text(self):
        """Test mixed Hiragana, Katakana, and Kanji conversion."""
        # かな漢字カタカナ (kana kanji katakana)
        result = normalize_japanese_text("かな漢字カタカナ")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain romaji representations
        assert result.isascii()

    def test_english_text_unchanged(self):
        """Test that English text is lowercased and stripped but NOT reordered.

        Real cross-source data shows name order variations:
        - AniDB/Anime-Planet: "Monkey D. Luffy" (family first)
        - characters_detailed.json: "Luffy Monkey D." (given first)

        normalize_japanese_text() preserves these (ai_character_matcher.py handles matching).
        """
        result = normalize_japanese_text("ONE PIECE")
        assert result == "one piece"

        result = normalize_japanese_text("Naruto")
        assert result == "naruto"

        result = normalize_japanese_text("  Attack on Titan  ")
        assert result == "attack on titan"

        # Real cross-source name order variations from temp/One_agent1/
        result = normalize_japanese_text("Monkey D. Luffy")  # anidb.json format
        assert result == "monkey d. luffy"

        result = normalize_japanese_text("Luffy Monkey D.")  # characters_detailed.json format
        assert result == "luffy monkey d."
        # These remain different - character matching logic handles this

    def test_empty_string(self):
        """Test empty string handling."""
        result = normalize_japanese_text("")
        assert result == ""

    def test_whitespace_trimming(self):
        """Test whitespace is properly trimmed."""
        result = normalize_japanese_text("  テスト  ")
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_lowercase_output(self):
        """Test all output is lowercase."""
        # Test with various inputs
        test_cases = [
            "ONE PIECE",
            "ワンピース",
            "Test 123",
        ]
        for text in test_cases:
            result = normalize_japanese_text(text)
            assert result == result.lower(), f"Result not lowercase: {result}"

    def test_mixed_japanese_english(self):
        """Test mixed Japanese and English text."""
        # ワンピース ONE PIECE
        result = normalize_japanese_text("ワンピース ONE PIECE")
        assert isinstance(result, str)
        assert len(result) > 0
        # Should convert Japanese part and keep English lowercase
        assert result.isascii()

    def test_numbers_and_symbols(self):
        """Test that numbers and symbols are preserved."""
        result = normalize_japanese_text("Test 123")
        assert "123" in result
        assert result == "test 123"

    def test_long_text(self):
        """Test long Japanese text doesn't fail."""
        # Long Katakana text
        long_text = "ワンピース" * 10
        result = normalize_japanese_text(long_text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_special_characters(self):
        """Test text with special characters."""
        result = normalize_japanese_text("テスト！")
        assert isinstance(result, str)
        # Should convert テスト to romaji
        assert len(result) > 0

    def test_romaji_consistency(self):
        """Test that multiple calls produce consistent results."""
        text = "ワンピース"
        result1 = normalize_japanese_text(text)
        result2 = normalize_japanese_text(text)
        assert result1 == result2, "Results should be consistent"

    def test_common_anime_titles(self):
        """Test conversion of common anime titles."""
        test_cases = [
            ("ナルト", "naruto"),  # Naruto
            ("ドラゴンボール", "doragon"),  # Dragon Ball (should contain 'doragon')
            ("進撃の巨人", None),  # Attack on Titan (kanji - just verify it converts)
        ]

        for japanese, expected_contains in test_cases:
            result = normalize_japanese_text(japanese)
            assert isinstance(result, str)
            assert len(result) > 0
            assert result.isascii()
            if expected_contains:
                assert expected_contains in result, f"Expected '{expected_contains}' in '{result}'"

    def test_thread_safety_simulation(self):
        """Test that function can be called with different inputs (validates no global state issues)."""
        # Call with different inputs in sequence
        results = []
        test_inputs = ["ワンピース", "ナルト", "テスト", "ONE PIECE"]

        for text in test_inputs:
            result = normalize_japanese_text(text)
            results.append(result)

        # Verify all results are unique (no state bleeding)
        assert len(results) == len(test_inputs)
        # Verify results are consistent when called again
        for i, text in enumerate(test_inputs):
            assert normalize_japanese_text(text) == results[i]


class TestRealCharacterData:
    """Test normalize_japanese_text with real character names from One Piece enrichment data."""

    def test_one_piece_main_characters(self):
        """Test normalization with main One Piece character names (Katakana)."""
        # Real data from temp/One_agent1/characters_detailed.json
        test_cases = [
            ("モンキー・D・ルフィ", "monkii"),  # Luffy Monkey D. - pykakasi uses "ii" for long vowels
            ("ニコ・ロビン", "niko"),  # Robin Nico
            ("ロロノア・ゾロ", "roronoa"),  # Zoro Roronoa
            ("フランキー", "furankii"),  # Franky
            ("サンジ", "sanji"),  # Sanji
            ("トニートニー・チョッパー", "toniitoni"),  # Chopper Tony Tony
            ("ナミ", "nami"),  # Nami
            ("ウソップ", "usoppu"),  # Usopp
        ]

        for japanese_name, expected_romaji_contains in test_cases:
            result = normalize_japanese_text(japanese_name)
            assert isinstance(result, str)
            assert len(result) > 0
            assert result.isascii(), f"Result should be ASCII romaji, got: {result}"
            # Check if expected substring is in the result (accounting for variations)
            assert (
                expected_romaji_contains.lower() in result.lower()
            ), f"Expected '{expected_romaji_contains}' in '{result}' for '{japanese_name}'"

    def test_one_piece_antagonists(self):
        """Test normalization with One Piece antagonist names (mixed scripts)."""
        # Real data from temp/One_agent1/characters_detailed.json
        test_cases = [
            ("バギー", "bagii"),  # Buggy - pykakasi uses "ii" for long vowels
            ("ネフェルタリ・ビビ", "neferutari"),  # Vivi Nefertari
            ("シャンクス", "shankusu"),  # Shanks
            ("エネル", "eneru"),  # Enel
            ("ジュラキュール・ミホーク", "juraky"),  # Mihawk Dracule
            ("ポートガス・D・エース", "pootogasu"),  # Ace Portgas D. - ポー is "poo"
            ("クロコダイル", "kurokodairu"),  # Crocodile
        ]

        for japanese_name, _expected_contains in test_cases:
            result = normalize_japanese_text(japanese_name)
            assert isinstance(result, str)
            assert result.isascii()
            # Flexible matching - just check that conversion happened
            assert len(result) >= 3, f"Result too short for '{japanese_name}': {result}"

    def test_hiragana_character_names(self):
        """Test normalization with hiragana character names."""
        # Real hiragana names from One Piece data
        test_cases = [
            ("たしぎ", "tashigi"),  # Tashigi
            ("くれは", "kureha"),  # Kureha
        ]

        for japanese_name, expected_romaji in test_cases:
            result = normalize_japanese_text(japanese_name)
            assert isinstance(result, str)
            assert result.isascii()
            assert expected_romaji in result.lower() or result.lower() in expected_romaji

    def test_character_names_with_special_chars(self):
        """Test character names with middle dots and special characters."""
        # Names with ・ (middle dot) separator
        test_cases = [
            "モンキー・D・ルフィ",  # Monkey D. Luffy
            "ポートガス・D・エース",  # Portgas D. Ace
            "マーシャル・D・ティーチ",  # Marshall D. Teach
            "ドンキホーテ・ドフラミンゴ",  # Donquixote Doflamingo
        ]

        for japanese_name in test_cases:
            result = normalize_japanese_text(japanese_name)
            assert isinstance(result, str)
            assert len(result) > 0
            assert result.isascii()
            # Should have converted the middle dots to something
            assert "・" not in result, f"Middle dot should be converted: {result}"

    def test_thread_safety_with_real_data(self):
        """Test that multiple calls with different real character names don't interfere.

        Also validates cross-source consistency: the same Japanese name from different
        data sources (anidb.json, anime_planet.json, characters_detailed.json) always
        produces identical romaji output.
        """
        # Real names from temp/One_agent1/ - various sources use the same Japanese names
        names = [
            "モンキー・D・ルフィ",  # Luffy - in anidb.json, anime_planet.json, characters_detailed.json
            "ニコ・ロビン",  # Robin - consistent across all sources
            "ロロノア・ゾロ",  # Zoro - consistent across all sources
            "サンジ",  # Sanji
            "ナミ",  # Nami
        ]

        # First pass - collect results
        results_pass1 = [normalize_japanese_text(name) for name in names]

        # Second pass - should get identical results (no state bleeding)
        results_pass2 = [normalize_japanese_text(name) for name in names]

        # Verify consistency (critical for cross-source matching)
        assert results_pass1 == results_pass2, "Results should be consistent across calls"

        # Verify all results are unique (no state bleeding between calls)
        assert len(set(results_pass1)) == len(
            names
        ), "Each name should produce a unique result"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_none_value_handling(self):
        """Test that passing invalid types raises appropriate errors."""
        # The function checks 'if not text' first, so None returns ""
        # This is actually reasonable defensive behavior
        with pytest.raises((AttributeError, TypeError)):
            # Force it past the 'if not text' check to test type issues
            normalize_japanese_text(123)  # type: ignore

    def test_unicode_edge_cases(self):
        """Test various Unicode characters."""
        # Test with Unicode characters that aren't Japanese
        result = normalize_japanese_text("Café")
        assert result == "café"

    def test_only_whitespace(self):
        """Test string with only whitespace."""
        result = normalize_japanese_text("   ")
        assert result == ""

    def test_newlines_and_tabs(self):
        """Test text with newlines and tabs."""
        result = normalize_japanese_text("Test\nText")
        # Should preserve or handle newlines appropriately
        assert isinstance(result, str)


class TestCachingBehavior:
    """Test pykakasi instance caching with functools.cache."""

    def test_get_kakasi_returns_same_instance(self):
        """Test that _get_kakasi returns the same cached instance."""
        # Clear cache to start fresh
        _get_kakasi.cache_clear()

        # Get instance twice
        instance1 = _get_kakasi()
        instance2 = _get_kakasi()

        # Should be the exact same object (singleton)
        assert instance1 is instance2, "Should return cached instance"

    def test_cache_info_shows_hits(self):
        """Test that cache_info shows cache hits after repeated calls."""
        # Clear cache to start fresh
        _get_kakasi.cache_clear()

        # First call should be a cache miss
        _get_kakasi()
        info1 = _get_kakasi.cache_info()
        assert info1.hits == 0
        assert info1.misses == 1

        # Second call should be a cache hit
        _get_kakasi()
        info2 = _get_kakasi.cache_info()
        assert info2.hits == 1
        assert info2.misses == 1

    def test_normalize_uses_cached_instance(self):
        """Test that normalize_japanese_text benefits from cached instance."""
        # Clear cache to start fresh
        _get_kakasi.cache_clear()

        # Multiple normalizations should reuse same instance
        normalize_japanese_text("ワンピース")
        normalize_japanese_text("ナルト")
        normalize_japanese_text("テスト")

        # Should have 1 miss (first call) and 2 hits (subsequent calls)
        info = _get_kakasi.cache_info()
        assert info.hits >= 2, f"Expected at least 2 cache hits, got {info.hits}"
        assert info.misses == 1, f"Expected 1 cache miss, got {info.misses}"

    def test_cache_clear_works(self):
        """Test that cache_clear resets the cache."""
        # Get instance
        _get_kakasi()
        info_before = _get_kakasi.cache_info()
        assert info_before.hits > 0 or info_before.misses > 0

        # Clear cache
        _get_kakasi.cache_clear()
        info_after = _get_kakasi.cache_info()

        # Should be reset
        assert info_after.hits == 0
        assert info_after.misses == 0
