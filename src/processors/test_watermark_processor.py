import os
from dotenv import load_dotenv
import requests
from PIL import Image
from io import BytesIO
from loguru import logger

from ..clients.florence_vision_client import FlorenceVisionClient
from ..clients.lama_inpainting_client import LamaInpaintingClient
from .watermark_processor import WatermarkRemovalProcessor

# Charger les variables d'environnement
load_dotenv()

def download_image(url: str) -> Image.Image:
    """Télécharge une image depuis une URL"""
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))

def save_and_log_result(image: Image.Image, output_path: str):
    """Sauvegarde l'image et log le chemin"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    logger.info(f"Image sauvegardée: {output_path}")

def test_watermark_removal():
    """Test simple du WatermarkRemovalProcessor"""
    
    # Configuration
    test_image_url = "https://cdn.shopify.com/s/files/1/0898/8344/3543/files/image_60bd0406_cleaned.jpg"
    output_path = "tests/output/watermark_removal_test.png"
    
    try:
        # Initialisation des clients
        florence_client = FlorenceVisionClient(
            hf_token=os.getenv("HUGGINGFACE_TOKEN"),
            space_url=os.getenv("FLORENCE_SPACE_URL")
        )
        
        lama_client = LamaInpaintingClient(
            space_url=os.getenv("LAMA_SPACE_URL"),
            hf_token=os.getenv("HUGGINGFACE_TOKEN")
        )
        
        # Initialisation du processeur
        processor = WatermarkRemovalProcessor(
            florence_client=florence_client,
            lama_client=lama_client,
            max_iterations=5,
            min_mask_pixels=50,
            max_bbox_percent=10.0
        )
        
        # Téléchargement de l'image de test
        logger.info(f"Téléchargement de l'image: {test_image_url}")
        image = download_image(test_image_url)
        
        # Traitement de l'image
        logger.info("Début du traitement...")
        result = processor.process_image(
            image=image,
            transparent=False
        )
        
        # Sauvegarde du résultat
        save_and_log_result(result, output_path)
        
        logger.info("Test terminé avec succès")
        
    except Exception as e:
        logger.error(f"Erreur pendant le test: {str(e)}")
        raise

if __name__ == "__main__":
    test_watermark_removal() 