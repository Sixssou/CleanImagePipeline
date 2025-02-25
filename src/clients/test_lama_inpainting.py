import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from loguru import logger

from lama_inpainting_client import LamaInpaintingClient

# Charger les variables d'environnement
load_dotenv()

def test_lama_inpainting():
    """Test simple du client LaMa"""
    
    # Initialisation du client
    client = LamaInpaintingClient(
        space_url=os.getenv("LAMA_SPACE_URL"),
        hf_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    # Charger l'image de test
    test_image_path = "tests/data/test_image.jpg"
    logger.info(f"Chargement de l'image: {test_image_path}")
    
    try:
        # Charger l'image
        image = np.array(Image.open(test_image_path))
        
        # Créer un masque simple (carré blanc)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[100:200, 100:200] = 255
        
        # Test de l'inpainting
        logger.info("Lancement de l'inpainting...")
        result = client.inpaint(image, mask)
        
        # Sauvegarder le résultat
        output_path = "tests/output/test_result.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(result).save(output_path)
        logger.info(f"Résultat sauvegardé: {output_path}")
        
    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        raise

if __name__ == "__main__":
    test_lama_inpainting() 