import os
import uuid
import shutil
import numpy as np
from PIL import Image
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
        Ajoute un filigrane (watermark) à une image.
        
        Args:
            image: L'image PIL sur laquelle ajouter le filigrane
            watermark_text: Le texte à utiliser comme filigrane
            
        Returns:
            Image: L'image PIL avec le filigrane
        """
        logger.info(f"Ajout d'un filigrane avec le texte: {watermark_text}")
        try:
            return add_watermark_to_image(image, watermark_text)
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du filigrane: {str(e)}")
            return image
    
    def remove_background(self, image_url):
        """
        Supprime l'arrière-plan d'une image.
        
        Args:
            image_url: URL de l'image
            
        Returns:
            Tuple[bool, np.ndarray]: (succès, image sans arrière-plan)
        """
        if not self.watermak_removal_client:
            raise ValueError("Le client de suppression de watermarks n'est pas défini")
            
        try:
            # Si l'entrée est un chemin ou une PIL Image, nous devons l'adapter
            if isinstance(image_url, str):
                return self.watermak_removal_client.remove_bg(image_url)
            elif isinstance(image_url, Image.Image):
                # Sauvegarder l'image temporairement
                temp_path = save_temp_image(image_url, "bg_input")
                result = self.watermak_removal_client.remove_bg(temp_path)
                
                # Supprimer le fichier temporaire
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
                return result
            elif isinstance(image_url, np.ndarray):
                # Convertir numpy array en PIL Image
                temp_path = save_temp_image(image_url, "bg_input")
                result = self.watermak_removal_client.remove_bg(temp_path)
                
                # Supprimer le fichier temporaire
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
                return result
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'arrière-plan: {str(e)}")
            return False, None
    
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
        Applique les modifications manuelles à une image.
        
        Args:
            image: Image originale (PIL.Image, np.ndarray ou chemin)
            edited_mask: Masque édité (PIL.Image, np.ndarray, dict ou chemin)
            supprimer_background: Supprimer l'arrière-plan après traitement
            add_watermark: Ajouter un filigrane après traitement
            watermark_text: Texte du filigrane
            
        Returns:
            Tuple[str, str, str]: Chemins (image originale, masque, résultat)
        """
        try:
            # Générer un ID unique pour cette opération
            unique_id = uuid.uuid4()
            
            # Créer des chemins pour les fichiers
            input_path = os.path.join(TEMP_DIR, f"input_{unique_id}.png")
            mask_path = os.path.join(TEMP_DIR, f"mask_{unique_id}.png")
            result_path = os.path.join(TEMP_DIR, f"result_{unique_id}.png")
            
            # Traiter l'image d'entrée
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
                pil_image.save(input_path)
            elif isinstance(image, Image.Image):
                pil_image = image
                pil_image.save(input_path)
            elif isinstance(image, str):
                if os.path.exists(image):
                    shutil.copy(image, input_path)
                    pil_image = Image.open(input_path)
                else:
                    raise ValueError(f"Le chemin d'image n'existe pas: {image}")
            else:
                raise ValueError(f"Type d'image non pris en charge: {type(image)}")
            
            # Traiter le masque édité
            mask_extracted = False
            mask_img = None
            
            # Extraire le masque en fonction de son type
            if isinstance(edited_mask, dict):
                # Format du nouveau ImageEditor (Gradio 5.x)
                logger.info(f"Traitement d'un masque sous forme de dictionnaire: {edited_mask.keys()}")
                
                if 'composite' in edited_mask:
                    # Le composite est l'image finale après édition
                    logger.info("Utilisation de la clé 'composite' du dictionnaire")
                    if isinstance(edited_mask['composite'], np.ndarray):
                        mask_img = Image.fromarray(edited_mask['composite'])
                    elif isinstance(edited_mask['composite'], Image.Image):
                        mask_img = edited_mask['composite']
                    mask_extracted = mask_img is not None
                    
                # Autres formats possibles de dictionnaire
                elif 'mask' in edited_mask:
                    logger.info("Utilisation de la clé 'mask' du dictionnaire")
                    if isinstance(edited_mask['mask'], np.ndarray):
                        mask_img = Image.fromarray(edited_mask['mask'])
                    elif isinstance(edited_mask['mask'], Image.Image):
                        mask_img = edited_mask['mask']
                    mask_extracted = mask_img is not None
                    
                elif 'layers' in edited_mask and edited_mask['layers'] and isinstance(edited_mask['layers'], list):
                    logger.info("Utilisation des couches du dictionnaire")
                    first_layer = edited_mask['layers'][0]
                    if isinstance(first_layer, Image.Image):
                        mask_img = first_layer
                        mask_extracted = True
                    elif isinstance(first_layer, np.ndarray):
                        mask_img = Image.fromarray(first_layer)
                        mask_extracted = True
                    elif isinstance(first_layer, dict) and 'content' in first_layer:
                        if isinstance(first_layer['content'], np.ndarray):
                            mask_img = Image.fromarray(first_layer['content'])
                            mask_extracted = True
                        
                elif 'background' in edited_mask:
                    logger.info("Utilisation de l'arrière-plan du dictionnaire")
                    if isinstance(edited_mask['background'], np.ndarray):
                        mask_img = Image.fromarray(edited_mask['background'])
                        mask_extracted = True
                    elif isinstance(edited_mask['background'], Image.Image):
                        mask_img = edited_mask['background']
                        mask_extracted = True
                
                if not mask_extracted:
                    logger.warning(f"Impossible d'extraire un masque du dictionnaire: {list(edited_mask.keys())}")
                    # Essayons de visualiser le contenu du dictionnaire pour mieux comprendre sa structure
                    for key, value in edited_mask.items():
                        logger.info(f"Clé: {key}, Type: {type(value)}")
                        if isinstance(value, list):
                            logger.info(f"  Liste de taille {len(value)}")
                            for i, item in enumerate(value):
                                logger.info(f"  Item {i}: Type {type(item)}")
                                if isinstance(item, dict):
                                    logger.info(f"    Sous-clés: {list(item.keys())}")
            
            elif isinstance(edited_mask, np.ndarray):
                logger.info("Traitement d'un masque sous forme de numpy array")
                mask_img = Image.fromarray(edited_mask)
                mask_extracted = True
            
            elif isinstance(edited_mask, Image.Image):
                logger.info("Traitement d'un masque sous forme d'image PIL")
                mask_img = edited_mask
                mask_extracted = True
            
            elif isinstance(edited_mask, str) and os.path.exists(edited_mask):
                logger.info("Traitement d'un masque sous forme de chemin")
                mask_img = Image.open(edited_mask)
                mask_extracted = True
            
            # Si nous n'avons pas réussi à extraire un masque, créer un masque vide
            if not mask_extracted or mask_img is None:
                logger.warning("Création d'un masque vide car aucun masque valide n'a été fourni")
                mask_img = create_empty_mask(pil_image)
            
            # Convertir le masque en mode L (niveaux de gris)
            if mask_img.mode != "L":
                logger.info(f"Conversion du masque du mode {mask_img.mode} au mode L")
                mask_img = mask_img.convert("L")
            
            # Sauvegarder le masque
            mask_img.save(mask_path)
            
            # Afficher plus d'infos sur le masque pour aider au débogage
            mask_array = np.array(mask_img)
            non_black_pixels = np.sum(mask_array > 0)
            total_pixels = mask_array.size
            logger.info(f"Masque: {non_black_pixels} pixels non noirs sur {total_pixels} ({non_black_pixels/total_pixels*100:.2f}%)")
            
            # Si le masque est entièrement vide (tous les pixels sont noirs), ne rien faire
            if non_black_pixels == 0:
                logger.warning("Masque entièrement vide ! Aucun pixel à traiter.")
                # Retourner l'image originale comme résultat
                shutil.copy(input_path, result_path)
                return input_path, mask_path, result_path
            
            # Appeler l'API d'inpainting
            logger.info("Appel de l'API d'inpainting")
            success, inpainted_result = self.watermak_removal_client.inpaint(pil_image, mask_img)
            
            if not success or inpainted_result is None:
                logger.error("L'inpainting a échoué")
                return input_path, mask_path, None
            
            # Sauvegarder le résultat intermédiaire
            inpainted_image = Image.fromarray(inpainted_result)
            inpainted_image.save(result_path)
            
            # Supprimer l'arrière-plan si demandé
            if supprimer_background:
                # Utiliser la nouvelle méthode de suppression d'arrière-plan avec l'image directement
                logger.info("Suppression de l'arrière-plan")
                local_image_np = np.array(inpainted_image)
                bg_removed_image = self.watermak_removal_client.remove_bg(local_image_np)
                
                if bg_removed_image is not None:
                    logger.info("Arrière-plan supprimé avec succès")
                    bg_removed_path = os.path.join(TEMP_DIR, f"bg_removed_{unique_id}.png")
                    
                    if isinstance(bg_removed_image, np.ndarray):
                        Image.fromarray(bg_removed_image).save(bg_removed_path)
                        result_path = bg_removed_path
                    elif isinstance(bg_removed_image, Image.Image):
                        bg_removed_image.save(bg_removed_path)
                        result_path = bg_removed_path
                    elif isinstance(bg_removed_image, str) and os.path.exists(bg_removed_image):
                        shutil.copy(bg_removed_image, bg_removed_path)
                        result_path = bg_removed_path
                else:
                    logger.warning("La suppression de l'arrière-plan a échoué")
            
            # Ajouter un filigrane si demandé
            if add_watermark and watermark_text:
                logger.info("Ajout d'un filigrane")
                result_img = Image.open(result_path)
                watermarked_img = add_watermark_to_image(result_img, watermark_text)
                wm_result_path = os.path.join(TEMP_DIR, f"wm_{unique_id}.png")
                watermarked_img.save(wm_result_path)
                result_path = wm_result_path
            
            return input_path, mask_path, result_path
        
        except Exception as e:
            logger.error(f"Erreur lors de l'application des modifications manuelles: {str(e)}")
            logger.exception(e)
            return None, None, None 