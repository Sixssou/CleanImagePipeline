import os
import uuid
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

from src.utils.image_utils import (
    create_empty_mask, 
    download_image, 
    visualize_mask, 
    save_temp_image,
    add_watermark_to_image,
    TEMP_DIR
)

class ImageProcessor:
    """
    Classe pour traiter les images, incluant la détection de watermarks,
    l'inpainting et l'ajout de filigranes.
    """
    
    def __init__(self, watermak_removal_client=None):
        """
        Initialise le processeur d'images.
        
        Args:
            watermak_removal_client: Client pour la suppression de watermarks
        """
        self.watermak_removal_client = watermak_removal_client
    
    def set_watermak_removal_client(self, client):
        """Définit le client de suppression de watermarks."""
        self.watermak_removal_client = client
    
    def detect_watermarks(self, image_url, max_bbox_percent=10.0, bbox_enlargement_factor=1.5, **kwargs):
        """
        Détecte les watermarks dans une image.
        
        Args:
            image_url: URL de l'image
            max_bbox_percent: Pourcentage maximal du masque par rapport à l'image
            bbox_enlargement_factor: Facteur d'agrandissement des bounding boxes
            **kwargs: Paramètres supplémentaires pour l'API de détection
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (image originale, image avec bounding boxes, masque)
        """
        if not self.watermak_removal_client:
            raise ValueError("Le client de suppression de watermarks n'est pas défini")
            
        params = {
            "image_url": image_url,
            "prompt": kwargs.get("prompt", ""),
            "max_new_tokens": kwargs.get("max_new_tokens", 1024),
            "early_stopping": kwargs.get("early_stopping", False),
            "do_sample": kwargs.get("do_sample", True),
            "num_beams": kwargs.get("num_beams", 5),
            "num_return_sequences": kwargs.get("num_return_sequences", 1),
            "temperature": kwargs.get("temperature", 0.75),
            "top_k": kwargs.get("top_k", 40),
            "top_p": kwargs.get("top_p", 0.85),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.2),
            "length_penalty": kwargs.get("length_penalty", 0.8),
            "max_bbox_percent": max_bbox_percent,
            "bbox_enlargement_factor": bbox_enlargement_factor
        }
        
        try:
            return self.watermak_removal_client.detect_wm(**params)
        except Exception as e:
            logger.error(f"Erreur lors de la détection des watermarks: {str(e)}")
            return None, None, None
    
    def inpaint_image(self, image, mask):
        """
        Effectue l'inpainting d'une image en utilisant un masque fourni.
        
        Args:
            image: Image à traiter (PIL.Image ou np.ndarray)
            mask: Masque binaire (PIL.Image ou np.ndarray)
            
        Returns:
            Tuple[bool, np.ndarray]: (succès, image traitée)
        """
        if not self.watermak_removal_client:
            raise ValueError("Le client de suppression de watermarks n'est pas défini")
        
        try:
            return self.watermak_removal_client.inpaint(image, mask)
        except Exception as e:
            logger.error(f"Erreur lors de l'inpainting: {str(e)}")
            return False, None
    
    def add_watermark(self, image, watermark_text):
        """
        Ajoute un filigrane à une image.
        
        Args:
            image: Image PIL ou tableau numpy
            watermark_text: Texte du filigrane
            
        Returns:
            PIL.Image: Image avec filigrane (pas un numpy array)
        """
        # Ajouter le texte avec une ombre légère pour la lisibilité
        try:
            watermark_image = self.watermak_removal_client.add_watermark(image, watermark_text, 0.25, 8, 0.03, 45)
            return watermark_image  
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du filigrane: {str(e)}")
            return image
    
    def remove_background(self, image_input):
        """
        Supprime l'arrière-plan d'une image et retourne un tuple (succès, image).
        
        Args:
            image_input: chemin local
            
        Returns:
            tuple: (bool, PIL.Image) Statut de succès et image avec l'arrière-plan supprimé
        """
        try:
            # Vérifier si le client watermak_removal est disponible
            if self.watermak_removal_client is None:
                logger.error("Le client WatermakRemoval n'est pas initialisé")
                return False, image_input
            
            # Log de l'appel à remove_bg
            logger.info(f"Appel de remove_bg avec l'image: {type(image_input)}")
            
            # Appeler l'API
            result = self.watermak_removal_client.remove_bg(image_input)
            
            # Si le résultat est None, retourner l'image d'origine
            if result is None:
                logger.error("La suppression de l'arrière-plan a échoué, API a retourné None")
                return False, image_input
            
            # Sinon, retourner le résultat
            return True, result
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'arrière-plan: {str(e)}")
            if isinstance(image_input, (Image.Image, np.ndarray)):
                if isinstance(image_input, np.ndarray):
                    return False, Image.fromarray(image_input.astype(np.uint8))
                return False, image_input
            return False, image_input
    
    def process_single_image(self, image_url, supprimer_background=False, 
                            add_watermark=False, watermark_text=None,
                            max_bbox_percent=10.0, bbox_enlargement_factor=1.5):
        """
        Traite une seule image: détection de watermarks, inpainting et optionnellement
        suppression de fond et ajout de filigrane.
        
        Args:
            image_url: URL de l'image à traiter
            supprimer_background: Supprimer l'arrière-plan après traitement
            add_watermark: Ajouter un filigrane après traitement
            watermark_text: Texte du filigrane
            max_bbox_percent: Pourcentage maximal du masque par rapport à l'image
            bbox_enlargement_factor: Facteur d'agrandissement des bounding boxes
            
        Returns:
            Tuple[str, str, str]: Chemins (image originale, masque, résultat)
        """
        try:
            # Générer un ID unique pour cette opération
            unique_id = uuid.uuid4()
            
            # Traitement complet avec la méthode remove_wm du client
            result = self.watermak_removal_client.remove_wm(
                image_url=image_url,
                threshold=0.85,
                max_bbox_percent=max_bbox_percent,
                remove_background_option=supprimer_background,
                add_watermark_option=add_watermark,
                watermark=watermark_text,
                bbox_enlargement_factor=bbox_enlargement_factor,
                remove_watermark_iterations=1
            )
            
            if not result or len(result) != 4:
                logger.error(f"Le traitement de l'image a échoué: {image_url}")
                return None, None, None
                
            # Extraire les résultats
            bg_removed_image, mask_image, inpainted_image, result_image = result
            
            # Sauvegarder les images
            input_path = save_temp_image(download_image(image_url), f"input_{unique_id}")
            mask_path = save_temp_image(mask_image, f"mask_{unique_id}")
            result_path = save_temp_image(result_image, f"result_{unique_id}")
            
            return input_path, mask_path, result_path
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image {image_url}: {str(e)}")
            logger.exception(e)
            return None, None, None
    
    def apply_manual_edits(self, image, edited_mask, supprimer_background=False, 
                          add_watermark=False, watermark_text=None):
        """
        Applique les modifications manuelles à une image: inpainting selon le masque édité,
        et optionnellement supprime l'arrière-plan et ajoute un watermark.
        
        Args:
            image: Image source (PIL.Image ou numpy array)
            edited_mask: Masque édité manuellement (PIL.Image ou numpy array)
            supprimer_background (bool): Si True, supprime l'arrière-plan
            add_watermark (bool): Si True, ajoute un watermark
            watermark_text (str): Texte du watermark à ajouter
            
        Returns:
            tuple: (chemin_image_entrée, chemin_masque, chemin_résultat)
        """
        try:
            # Appliquer l'inpainting
            inpainted_image = self.inpaint_image(image, edited_mask)
            
            # Supprimer l'arrière-plan si demandé
            if supprimer_background:
                success, processed_img = self.remove_background(inpainted_image)
                if not success:
                    logger.warning("La suppression de l'arrière-plan a échoué, utilisation de l'image inpainted")
                    processed_img = inpainted_image
            else:
                processed_img = inpainted_image
            
            # Ajouter un watermark si demandé
            if add_watermark and watermark_text:
                if isinstance(processed_img, np.ndarray):
                    img_pil = Image.fromarray(processed_img)
                else:
                    img_pil = processed_img
                    
                processed_img = self.add_watermark(img_pil, watermark_text)
                if processed_img is None:
                    logger.warning("L'ajout du watermark a échoué, utilisation de l'image sans cette étape")
                    processed_img = img_pil
            
            # Sauvegarder le résultat
            result_path = save_temp_image(processed_img, prefix="result")
            
            return inpainted_image, edited_mask, result_path
        
        except Exception as e:
            logger.error(f"Erreur lors de l'application des modifications manuelles: {str(e)}")
            return None, None, None 