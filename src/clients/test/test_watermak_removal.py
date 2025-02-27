from src.clients.watermak_removal_client import WatermakRemovalClient
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

def test_bg_removal(image_url):
    logger.info(f"Test de la suppression de watermarks, HF_TOKEN: {os.getenv('HF_TOKEN')} et HF_SPACE_WATERMAK_REMOVAL: {os.getenv('HF_SPACE_WATERMAK_REMOVAL')}")
    
    # Initialisation du client
    client = WatermakRemovalClient(
        hf_token=os.getenv('HF_TOKEN'),
        space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
    )

    try:
        result = client.remove_bg(image_url)
        logger.info(f"Résultat pour remove_bg: {result}")
            
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")

def test_wm_detection(image_url, threshold):
    logger.info(f"Test de la détection de watermarks, HF_TOKEN: {os.getenv('HF_TOKEN')} et HF_SPACE_WATERMAK_REMOVAL: {os.getenv('HF_SPACE_WATERMAK_REMOVAL')}")
    
    # Initialisation du client
    client = WatermakRemovalClient(
        hf_token=os.getenv('HF_TOKEN'),
        space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
    )

    try:
        result = client.wm_detection(image_url, threshold)
        logger.info(f"Résultat pour wm_detection: {result}")
            
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")

def test_remove_wm(image_url, threshold, max_bbox_percent, 
                      remove_background_option, add_watermark_option, watermark):
    logger.info(f"Test de la suppression de watermarks, HF_TOKEN: {os.getenv('HF_TOKEN')} et HF_SPACE_WATERMAK_REMOVAL: {os.getenv('HF_SPACE_WATERMAK_REMOVAL')}")
    
    # Initialisation du client
    client = WatermakRemovalClient(
        hf_token=os.getenv('HF_TOKEN'),
        space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
    )

    try:
        result = client.remove_wm(image_url, threshold, max_bbox_percent, 
                                     remove_background_option, add_watermark_option, 
                                     watermark)
        logger.info(f"Résultat pour remove_wm: {result}")
            
    except Exception as e:
        logger.error(f"Erreur lors du test: {e}")

if __name__ == "__main__":
    #test_bg_removal(os.getenv("TEST_PHOTO_URL_1")) 
    #test_wm_detection(os.getenv("TEST_PHOTO_URL_1"))
    test_remove_wm(
        os.getenv("TEST_PHOTO_URL_1"), 
        0.85, 
        5, 
        True, 
        True, 
        "www.inflatable-store.com"
    )