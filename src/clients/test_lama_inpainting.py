import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from loguru import logger

from lama_inpainting_client import LamaInpaintingClient

# Charger les variables d'environnement
load_dotenv()

def create_test_mask(image_size: tuple, rect_coords: list) -> np.ndarray:
    """
    Crée un masque de test avec des rectangles blancs
    
    Args:
        image_size (tuple): Taille de l'image (width, height)
        rect_coords (list): Liste de coordonnées [(x1,y1,x2,y2), ...]
    Returns:
        np.ndarray: Masque binaire
    """
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    
    for coords in rect_coords:
        draw.rectangle(coords, fill=255)
    
    return np.array(mask)

def display_results(original: np.ndarray, mask: np.ndarray, result: np.ndarray):
    """Affiche les images côte à côte"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Image originale")
    axes[0].axis("off")
    
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Masque")
    axes[1].axis("off")
    
    axes[2].imshow(result)
    axes[2].set_title("Résultat")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def test_lama_inpainting():
    """Test principal du client LaMa"""
    
    # Initialisation du client
    client = LamaInpaintingClient(
        space_url=os.getenv("LAMA_SPACE_URL"),
        hf_token=os.getenv("HUGGINGFACE_TOKEN")
    )
    
    # Charger une image de test
    test_image_path = "tests/data/test_image.jpg"  # À adapter selon votre structure
    logger.info(f"Chargement de l'image de test: {test_image_path}")
    
    try:
        # Charger et convertir l'image
        image = Image.open(test_image_path)
        image_np = np.array(image)
        
        # Créer un masque de test (rectangle au centre)
        width, height = image.size
        center_x, center_y = width // 2, height // 2
        rect_size = 100
        
        mask_np = create_test_mask(
            image.size,
            [(
                center_x - rect_size,
                center_y - rect_size,
                center_x + rect_size,
                center_y + rect_size
            )]
        )
        
        logger.info("Lancement de l'inpainting...")
        
        # Test de l'inpainting
        result = client.inpaint(image_np, mask_np)
        
        logger.info("Inpainting terminé")
        
        # Afficher les résultats
        display_results(image_np, mask_np, result)
        
        # Sauvegarder le résultat
        output_path = "tests/output/test_result.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        Image.fromarray(result).save(output_path)
        logger.info(f"Résultat sauvegardé: {output_path}")
        
    except Exception as e:
        logger.error(f"Erreur pendant le test: {str(e)}")
        raise

if __name__ == "__main__":
    test_lama_inpainting() 