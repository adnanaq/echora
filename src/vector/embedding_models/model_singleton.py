import torch
from sentence_transformers import SentenceTransformer
import logging
from typing import Optional

import jaconv
import pykakasi

logger = logging.getLogger(__name__)

_global_embedding_model: Optional[SentenceTransformer] = None

# Setup pykakasi for Japanese text normalization
_kakasi = pykakasi.kakasi()
_kakasi.setMode("H", "a")  # Hiragana to ascii
_kakasi.setMode("K", "a")  # Katakana to ascii
_kakasi.setMode("J", "a")  # Kanji to ascii
_kakasi.setMode("r", "Hepburn") # Romanization system

_kakasi_converter = _kakasi.getConverter()

def normalize_japanese_text(text: str) -> str:
    """
    Converts Japanese text (Hiragana, Katakana, Kanji) to Romaji using pykakasi and jaconv.
    If the text does not contain Japanese characters, it returns the original text.
    """
    # Check if text contains Japanese characters (Hiragana, Katakana, Kanji)
    if any('\u3040' <= char <= '\u30ff' or '\u4e00' <= char <= '\u9faf' for char in text):
        # Convert Katakana to Hiragana first for consistent romaji conversion
        hiragana = jaconv.kata2hira(text)
        romaji = _kakasi_converter.do(hiragana)
        return romaji.lower().strip()
    return text.lower().strip()

def load_embedding_model(model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2') -> SentenceTransformer:
    """
    Loads the SentenceTransformer model as a global singleton.
    If the model is already loaded, it returns the existing instance.
    """
    global _global_embedding_model
    if _global_embedding_model is None:
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        try:
            _global_embedding_model = SentenceTransformer(model_name)
            if torch.cuda.is_available():
                _global_embedding_model.to('cuda')
                logger.info("SentenceTransformer model moved to GPU.")
            else:
                logger.info("SentenceTransformer model loaded on CPU.")
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
            _global_embedding_model = None
            raise RuntimeError(f"Failed to load embedding model: {e}")
    return _global_embedding_model

def get_embedding_model() -> Optional[SentenceTransformer]:
    """
    Returns the globally loaded SentenceTransformer model instance.
    If the model has not been loaded yet, it attempts to load it with default settings.
    """
    if _global_embedding_model is None:
        try:
            return load_embedding_model()
        except RuntimeError:
            return None # Model failed to load
    return _global_embedding_model

def get_text_embedding(text: str) -> Optional[list[float]]:
    """
    Generates a semantic embedding for the given text using the global model.
    """
    model = get_embedding_model()
    if model is None:
        logger.warning("Embedding model not loaded, cannot generate embedding.")
        return None
    if not text or not text.strip():
        return None
    try:
        # Apply normalization before embedding
        normalized_text = normalize_japanese_text(text)
        # Encode the text. convert_to_tensor=True returns a torch.Tensor
        embedding = model.encode(normalized_text, convert_to_tensor=True)
        return embedding.tolist() # Convert to list for JSON serialization
    except Exception as e:
        logger.error(f"Error generating embedding for text: '{text[:50]}...': {e}")
        return None
