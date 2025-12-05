import sys
import os
from pathlib import Path
from PIL import Image
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "vector_processing" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "libs" / "common" / "src"))

from common.config import get_settings
from vector_processing.embedding_models.factory import EmbeddingModelFactory
from vector_processing.utils.image_downloader import ImageDownloader
from vector_processing import TextProcessor, VisionProcessor

def verify():
    print("Starting verification...")
    
    try:
        settings = get_settings()
        print("Settings loaded.")

        # Factory
        print("Creating text model...")
        text_model = EmbeddingModelFactory.create_text_model(settings)
        print(f"Text model created: {text_model.model_name} (Provider: {settings.text_embedding_provider})")
        
        print("Creating vision model...")
        vision_model = EmbeddingModelFactory.create_vision_model(settings)
        print(f"Vision model created: {vision_model.model_name} (Provider: {settings.image_embedding_provider})")

        # Downloader
        print("Creating downloader...")
        downloader = ImageDownloader(cache_dir=settings.model_cache_dir)
        print("Downloader created.")

        # Processors
        print("Initializing TextProcessor...")
        text_processor = TextProcessor(text_model, settings)
        print("TextProcessor initialized.")
        
        print("Initializing VisionProcessor...")
        vision_processor = VisionProcessor(vision_model, downloader, settings)
        print("VisionProcessor initialized.")

        # Test Text
        print("Testing text encoding...")
        embedding = text_processor.encode_text("Test anime title")
        if embedding and len(embedding) == text_model.embedding_size:
            print(f"Text encoding successful. Size: {len(embedding)}")
        else:
            print(f"Text encoding failed. Got {type(embedding)}")

        # Test Vision
        print("Testing vision encoding...")
        img = Image.new('RGB', (224, 224), color = 'red')
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name
        
        print(f"Created dummy image at {tmp_path}")
        
        # Test direct encoding
        vision_embedding = vision_processor.encode_image(tmp_path)
        if vision_embedding and len(vision_embedding) == vision_model.embedding_size:
            print(f"Vision encoding successful. Size: {len(vision_embedding)}")
        else:
            print(f"Vision encoding failed. Got {type(vision_embedding)}")

        # Test CCIPS
        print("Testing CCIPS similarity...")
        from vector_processing.legacy_ccips import LegacyCCIPS
        ccips = LegacyCCIPS()
        # Compare image with itself (should be 1.0)
        similarity = ccips.calculate_character_similarity(tmp_path, tmp_path)
        print(f"Self-similarity score: {similarity}")
        if similarity > 0.99:
            print("CCIPS test passed.")
        else:
            print(f"CCIPS test failed. Expected ~1.0, got {similarity}")

        # Clean up
        os.remove(tmp_path)
        print("Verification complete!")
        
    except Exception as e:
        print(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
