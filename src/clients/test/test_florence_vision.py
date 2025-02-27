from src.clients.florence_vision_client import FlorenceVisionClient
from dotenv import load_dotenv
from loguru import logger
import os
load_dotenv()

def test_florence_vision():
    # Initialisation du client
    client = FlorenceVisionClient(
        hf_token=os.getenv('HF_TOKEN'),
        space_url=os.getenv('HF_SPACE_FLORENCE')
    )

    try:
        # Test avec différents prompts
        prompts = [
            "OCR_WITH_REGION",      # Pour la détection de texte
        ]
        
        for prompt in prompts:
            logger.info(f"Test avec prompt: {prompt}")
            try:
                result = client.analyze_image(
                    image_url=os.getenv("TEST_PHOTO_URL_1"),
                    prompt=prompt
                )
                logger.info(f"Résultat pour {prompt}:")
                logger.info(result)
            except Exception as e:
                logger.error(f"Erreur pour {prompt}: {e}")
            
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")

if __name__ == "__main__":
    test_florence_vision() 