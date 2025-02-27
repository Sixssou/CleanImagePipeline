from typing import Optional
import numpy as np
from gradio_client import Client
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

class LamaInpaintingClient:
    """Client pour le modèle LaMa (Latent Matching) via l'API Gradio HuggingFace Spaces"""
    
    def __init__(self, space_url: str, hf_token: Optional[str] = None):
        """
        Initialise le client LaMa
        
        Args:
            space_url (str): URL du Space HuggingFace (ex: "username/space-name")
            hf_token (str, optional): Token HuggingFace pour les spaces privés
        """
        self.client = Client(
            src=space_url,
            hf_token=hf_token,
            verbose=False
        )
        logger.info(f"Connected to LaMa Space: {space_url}")
        
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Effectue l'inpainting sur les zones masquées de l'image via l'API Gradio
        
        Args:
            image (np.ndarray): Image source
            mask (np.ndarray): Masque binaire (255 pour zones à reconstruire)
        Returns:
            np.ndarray: Image avec zones reconstruites
        """
        try:
            # Vérification que le masque est en niveaux de gris
            if len(mask.shape) == 3:
                mask = mask.mean(axis=2).astype(np.uint8)
            
            # Normalisation du masque
            mask = (mask > 127).astype(np.uint8) * 255
            
            # Appel de l'API Gradio
            result = self.client.predict(
                image,          # Image originale
                mask,          # Masque
                api_name="/predict"  # Endpoint par défaut de Gradio
            )
            
            # Conversion du résultat en numpy array si nécessaire
            if not isinstance(result, np.ndarray):
                result = np.array(result)
                
            # Normalisation du résultat
            if result.dtype in [np.float64, np.float32]:
                result = np.clip(result, 0, 255).astype(np.uint8)
                
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'inpainting: {str(e)}")
            raise 