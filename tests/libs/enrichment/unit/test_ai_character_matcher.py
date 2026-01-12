"""Unit tests for ai_character_matcher, particularly CharacterNamePreprocessor romaji conversion with modern pykakasi API."""

import pytest
from enrichment.ai_character_matcher import CharacterNamePreprocessor

# Skip all tests in this module if pykakasi is not installed
pytest.importorskip("pykakasi")


class TestCharacterNamePreprocessorRomaji:
    """Test romaji conversion uses modern pykakasi API correctly."""

    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance."""
        return CharacterNamePreprocessor()

    def test_to_romaji_returns_string_not_object(self, preprocessor):
        """Verify _to_romaji returns a string (modern API), not conversion object (deprecated API)."""
        result = preprocessor._to_romaji("テスト")
        assert isinstance(result, str), "Result should be a string, not a conversion object"

    def test_romaji_hiragana_conversion(self, preprocessor):
        """Test hiragana to romaji conversion."""
        result = preprocessor._to_romaji("こんにちは")
        assert result.lower() == "konnichiha", f"Expected 'konnichiha', got '{result}'"

    def test_romaji_katakana_conversion(self, preprocessor):
        """Test katakana to romaji conversion."""
        result = preprocessor._to_romaji("ナルト")
        assert result.lower() == "naruto", f"Expected 'naruto', got '{result}'"

    def test_romaji_kanji_conversion(self, preprocessor):
        """Test kanji to romaji conversion."""
        result = preprocessor._to_romaji("桜")
        assert result.lower() == "sakura", f"Expected 'sakura', got '{result}'"

    def test_romaji_mixed_scripts(self, preprocessor):
        """Test mixed Japanese scripts to romaji."""
        result = preprocessor._to_romaji("ワンピース")
        # Should contain "wanpi" (One Piece in romaji)
        assert "wanpi" in result.lower(), f"Expected 'wanpi' in result, got '{result}'"

    def test_romaji_consistency(self, preprocessor):
        """Test romaji conversion is deterministic."""
        text = "サンジ"
        result1 = preprocessor._to_romaji(text)
        result2 = preprocessor._to_romaji(text)
        assert result1 == result2, "Romaji conversion should be deterministic"

    def test_romaji_fallback_on_error(self, preprocessor):
        """Test that romaji conversion handles errors gracefully."""
        # Non-Japanese text should either convert or return original
        result = preprocessor._to_romaji("English")
        assert result is not None
        assert isinstance(result, str)

    def test_process_japanese_name_includes_romaji(self, preprocessor):
        """Test that Japanese name processing includes romaji field."""
        result = preprocessor.preprocess_name("ルフィ", "japanese")
        assert "romaji" in result, "Result should include 'romaji' field"
        assert result["romaji"], "Romaji field should not be empty"
        assert isinstance(result["romaji"], str), "Romaji should be a string"

    def test_process_japanese_name_romaji_is_lowercase_ascii(self, preprocessor):
        """Test that romaji output is lowercase ASCII."""
        result = preprocessor.preprocess_name("モンキー", "japanese")
        romaji = result["romaji"]
        assert romaji.islower() or not romaji.isalpha(), "Romaji should be lowercase"
        assert romaji.isascii(), "Romaji should be ASCII characters"
