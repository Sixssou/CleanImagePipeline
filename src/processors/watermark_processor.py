from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageDraw
from loguru import logger

from ..clients.florence_vision_client import FlorenceVisionClient
from ..clients.lama_inpainting_client import LamaInpaintingClient

class WatermarkRemovalProcessor:
    """Processeur principal pour la suppression de watermarks"""
    
    def __init__(
        self,
        florence_client: FlorenceVisionClient,
        lama_client: Optional[LamaInpaintingClient] = None,
        max_iterations: int = 5,
        min_mask_pixels: int = 50,
        max_bbox_percent: float = 10.0
    ):
        self.florence_client = florence_client
        self.lama_client = lama_client
        self.max_iterations = max_iterations
        self.min_mask_pixels = min_mask_pixels
        self.max_bbox_percent = max_bbox_percent

    def _create_mask_from_detection(
        self, 
        detection_result: dict, 
        image_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Crée un masque binaire à partir des détections Florence
        
        Args:
            detection_result (dict): Résultat de Florence Vision
            image_size (tuple): Taille de l'image (width, height)
        Returns:
            Image.Image: Masque binaire
        """
        mask = Image.new("L", image_size, 0)
        draw = ImageDraw.Draw(mask)
        
        detection_key = "<OCR_WITH_REGION>"
        if detection_key not in detection_result or "quad_boxes" not in detection_result[detection_key]:
            return mask
            
        image_area = image_size[0] * image_size[1]
        
        for quad, label in zip(
            detection_result[detection_key]["quad_boxes"],
            detection_result[detection_key]["labels"]
        ):
            # Conversion quad en bbox
            x_coords = [quad[i] for i in range(0, len(quad), 2)]
            y_coords = [quad[i+1] for i in range(0, len(quad), 2)]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            # Vérification taille bbox
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if (bbox_area / image_area) * 100 <= self.max_bbox_percent:
                draw.rectangle(bbox, fill=255)
            else:
                logger.warning(
                    f"Bbox ignorée car trop grande: {bbox_area/image_area:.2%} de l'image"
                )
                
        return mask

    def _make_transparent(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """Rend transparentes les zones masquées"""
        image = image.convert("RGBA")
        transparent = Image.new("RGBA", image.size)
        
        for x in range(image.width):
            for y in range(image.height):
                if mask.getpixel((x, y)) > 0:
                    transparent.putpixel((x, y), (0, 0, 0, 0))
                else:
                    transparent.putpixel((x, y), image.getpixel((x, y)))
        
        return transparent

    def process_image(
        self,
        image: Image.Image,
        transparent: bool = False
    ) -> Image.Image:
        """
        Traite une image pour supprimer les watermarks
        
        Args:
            image (Image.Image): Image source
            transparent (bool): Si True, rend les zones transparentes au lieu de les reconstruire
        Returns:
            Image.Image: Image traitée
        """
        if transparent and not self.lama_client:
            raise ValueError("LamaInpaintingClient requis pour le mode non-transparent")
            
        current_image = image.copy()
        
        for iteration in range(self.max_iterations):
            logger.info(f"Itération {iteration + 1}/{self.max_iterations}")
            
            # 1. Détection avec Florence
            result = self.florence_client.analyze_image(
                image=current_image,
                prompt="<OCR_WITH_REGION>"
            )
            
            # 2. Création du masque
            mask = self._create_mask_from_detection(result, current_image.size)
            mask_np = np.array(mask)
            
            # Critères d'arrêt
            if not np.any(mask_np):
                logger.info("Aucun watermark détecté")
                break
                
            if np.count_nonzero(mask_np) < self.min_mask_pixels:
                logger.info("Masque trop petit, possible faux positif")
                break
            
            # 3. Traitement
            if transparent:
                current_image = self._make_transparent(current_image, mask)
            else:
                result = self.lama_client.inpaint(
                    np.array(current_image),
                    mask_np
                )
                current_image = Image.fromarray(result)
                
        return current_image 