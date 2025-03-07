from gradio_client import Client
import httpx
from typing import Dict, Any, Tuple, Union
import numpy as np
from dotenv import load_dotenv
from loguru import logger
import os
import base64
from io import BytesIO
from PIL import Image
import cv2
from gradio_client import handle_file
import tempfile
import requests

load_dotenv()

class WatermakRemovalClient:
    """Client for interacting with Florence-2 Vision model on HuggingFace Spaces"""
    
    def __init__(self, hf_token: str, space_url: str, timeout: int = 120):
        self.token = hf_token
        self.space_url = space_url
        
        # Configure httpx client with timeout
        httpx_kwargs = {
            "timeout": httpx.Timeout(timeout, read=120.0)
        }
        
        # Initialize Gradio client
        self.client = Client(
            space_url,
            hf_token=hf_token,
            httpx_kwargs=httpx_kwargs
        )

    def remove_bg(self, image_input: Union[str, np.ndarray]):
        """
        Supprime l'arrière-plan d'une image.
        
        Args:
            image_input: URL de l'image ou tableau numpy
            
        Returns:
            Image sans arrière-plan
        """
        try:
            # Si l'entrée est un tableau numpy, convertir en base64
            if isinstance(image_input, np.ndarray):
                # Convertir l'image numpy en base64
                pil_img = Image.fromarray(image_input)
                buffered = BytesIO()
                pil_img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Créer une URL temporaire en ligne (si possible)
                try:
                    # Essayer d'utiliser un service comme ImgBB pour héberger l'image temporairement
                    # Ceci est juste un exemple, vous devrez peut-être utiliser un autre service
                    api_key = os.getenv("IMGBB_API_KEY")
                    if api_key:
                        url = "https://api.imgbb.com/1/upload"
                        payload = {
                            "key": api_key,
                            "image": img_str
                        }
                        response = requests.post(url, payload)
                        if response.status_code == 200:
                            data = response.json()
                            image_url = data["data"]["url"]
                            logger.info(f"Image téléchargée temporairement à {image_url}")
                            input_for_api = image_url
                        else:
                            # Si l'upload échoue, utiliser une URL d'image par défaut
                            logger.warning("Échec du téléchargement de l'image, utilisation de l'image d'origine")
                            return image_input
                    else:
                        # Si pas de clé API, utiliser une URL d'image par défaut
                        logger.warning("Pas de clé API pour ImgBB, utilisation de l'image d'origine")
                        return image_input
                except Exception as e:
                    logger.error(f"Erreur lors du téléchargement de l'image: {str(e)}")
                    return image_input
            else:
                # Utiliser l'URL directement
                input_for_api = image_input
                
            # Obtenir le nom de l'API à partir des variables d'environnement
            api_name = os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_REMOVE_BACKGROUND_FROM_URL")
            logger.info(f"Appel à l'API remove_bg avec api_name={api_name}")
            
            # Appel à l'API avec l'argument positionnel
            result = self.client.predict(
                input_for_api,
                api_name=api_name
            )
            
            # Traiter le résultat
            if isinstance(result, np.ndarray):
                return result
            elif isinstance(result, str) and os.path.exists(result):
                # Charger l'image résultante
                result_image = cv2.imread(result)
                # Convertir de BGR à RGB (OpenCV charge en BGR)
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                return result_image
            else:
                logger.warning(f"Type de résultat inattendu: {type(result)}")
                return image_input
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de l'arrière-plan: {str(e)}")
            # Retourner l'image d'origine en cas d'erreur
            return image_input
    
    def detect_wm(self, 
                  image_url: str, 
                  prompt: str = "",
                  max_new_tokens: int = 1024,
                  early_stopping: bool = False,
                  do_sample: bool = True,
                  num_beams: int = 5,
                  num_return_sequences: int = 1,
                  temperature: float = 0.75,
                  top_k: int = 40,
                  top_p: float = 0.85,
                  repetition_penalty: float = 1.2,
                  length_penalty: float = 0.8,
                  max_bbox_percent: float = 10.0,
                  bbox_enlargement_factor: float = 1.2):
        """
        Détecte les watermarks dans une image à partir d'une URL.
        
        Args:
            image_url (str): URL de l'image à analyser
            prompt (str, optional): Prompt pour guider la détection. Defaults to "".
            max_new_tokens (int, optional): Nombre maximum de tokens à générer. Defaults to 1024.
            early_stopping (bool, optional): Arrêt anticipé. Defaults to False.
            do_sample (bool, optional): Utiliser l'échantillonnage. Defaults to True.
            num_beams (int, optional): Nombre de beams pour la recherche. Defaults to 5.
            num_return_sequences (int, optional): Nombre de séquences à retourner. Defaults to 1.
            temperature (float, optional): Température pour l'échantillonnage. Defaults to 0.75.
            top_k (int, optional): Top K pour l'échantillonnage. Defaults to 40.
            top_p (float, optional): Top P pour l'échantillonnage. Defaults to 0.85.
            repetition_penalty (float, optional): Pénalité de répétition. Defaults to 1.2.
            length_penalty (float, optional): Pénalité de longueur. Defaults to 0.8.
            max_bbox_percent (float, optional): Pourcentage maximal de la taille de l'image pour les bounding boxes. Defaults to 10.0.
            bbox_enlargement_factor (float, optional): Facteur d'agrandissement des bounding boxes. Defaults to 1.2.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple contenant (image originale, image avec bounding boxes, masque)
        """
        try:
            result = self.client.predict(
                image_url,
                prompt,
                max_new_tokens,
                early_stopping,
                do_sample,
                num_beams,
                num_return_sequences,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                length_penalty,
                max_bbox_percent,
                bbox_enlargement_factor,
                api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_DETECT_WATERMARKS_FROM_URL")
            )
            
            # Vérifier que le résultat est bien un tuple de 3 éléments
            if isinstance(result, tuple) and len(result) == 3:
                return result
            else:
                logger.warning(f"Format de résultat inattendu: {type(result)}")
                # Si le résultat n'est pas au format attendu, essayer de l'adapter
                if isinstance(result, np.ndarray):
                    # Si c'est juste une image, supposer que c'est l'image avec bounding boxes
                    # et créer des images vides pour les autres
                    empty_image = np.zeros_like(result)
                    return empty_image, result, empty_image
                else:
                    raise ValueError(f"Format de résultat inattendu: {type(result)}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la détection des watermarks: {str(e)}")
            raise Exception(f"Erreur lors de la détection des watermarks: {str(e)}")

    def remove_wm(self, 
                  image_url: str, 
                  threshold: float = 0.85, 
                  max_bbox_percent: float = 10.0, 
                  remove_background_option: bool = False, 
                  add_watermark_option: bool = False, 
                  watermark: str = None,
                  bbox_enlargement_factor: float = 1.5,
                  remove_watermark_iterations: int = 1,
                  mask: np.ndarray = None
                  ):
        """
        Détecte et supprime les watermarks d'une image à partir d'une URL.
        
        Args:
            image_url (str): URL de l'image à traiter
            threshold (float, optional): Seuil de confiance pour la détection. Defaults to 0.85.
            max_bbox_percent (float, optional): Pourcentage maximal de la taille de l'image pour les bounding boxes. Defaults to 10.0.
            remove_background_option (bool, optional): Si True, l'arrière-plan sera supprimé. Defaults to False.
            add_watermark_option (bool, optional): Si True, un nouveau watermark sera ajouté. Defaults to False.
            watermark (str, optional): Texte du watermark à ajouter. Defaults to None.
            bbox_enlargement_factor (float, optional): Facteur d'agrandissement des bounding boxes. Defaults to 1.5.
            remove_watermark_iterations (int, optional): Nombre d'itérations pour la suppression. Defaults to 1.
            mask (np.ndarray, optional): Masque personnalisé pour la détection. Defaults to None.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple contenant (image sans fond, masque de détection, image inpainted, image finale)
        """
        try:
            # Appel à l'API avec les paramètres
            response = self.client.predict(
                image_url,
                threshold,
                max_bbox_percent,
                remove_background_option,
                add_watermark_option,
                watermark if watermark else "",  # Envoyer une chaîne vide si watermark est None
                bbox_enlargement_factor,
                remove_watermark_iterations,
                mask,
                api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_REMOVE_WATERMARKS_FROM_URL")
            )
            
            # Retourner la réponse telle quelle
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression des watermarks: {str(e)}")
            raise Exception(f"Erreur lors de la suppression des watermarks: {str(e)}")
        
    def _base64_to_numpy(self, base64_string):
        """
        Convertit une chaîne base64 en numpy array.
        
        Args:
            base64_string (str): Image encodée en base64
            
        Returns:
            np.ndarray: Image au format numpy array
        """
        if not base64_string:
            return None
        
        try:
            # Vérifier si la chaîne contient un préfixe data URI
            if base64_string.startswith('data:image'):
                # Extraire la partie base64 après la virgule
                base64_string = base64_string.split(',')[1]
            
            # Décoder la chaîne base64
            image_data = base64.b64decode(base64_string)
            
            # Convertir en image PIL
            image = Image.open(BytesIO(image_data))
            
            # Convertir en numpy array
            return np.array(image)
        except Exception as e:
            logger.error(f"Erreur lors de la conversion base64 en numpy: {str(e)}")
            return None
    
    def inpaint(self, 
                input_image: np.ndarray, 
                mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applique l'inpainting sur une image en utilisant un masque fourni.
        
        Args:
            input_image (np.ndarray): Image d'entrée au format numpy array
            mask (np.ndarray): Masque indiquant les zones à traiter (zones blanches)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple contenant (masque utilisé, image traitée)
        """
        try:
            # Vérifier si l'image a un canal alpha (4 canaux)
            if len(input_image.shape) == 3 and input_image.shape[2] == 4:
                # Supprimer le canal alpha
                input_image = input_image[:, :, :3]
                logger.info(f"Canal alpha supprimé de l'image, nouvelle forme: {input_image.shape}")
            
            # Vérifier si le masque a un canal alpha (4 canaux)
            if len(mask.shape) == 3 and mask.shape[2] == 4:
                # Utiliser le canal alpha comme masque ou convertir en niveaux de gris
                mask = mask[:, :, 3]  # Utiliser le canal alpha
                logger.info(f"Canal alpha utilisé comme masque, nouvelle forme: {mask.shape}")
            
            # S'assurer que le masque est en niveaux de gris
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
                logger.info(f"Masque converti en niveaux de gris, nouvelle forme: {mask.shape}")
            
            # S'assurer que le masque est au bon format
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            # Normaliser le masque (0-255)
            max_val = np.max(mask)
            if max_val > 0:
                mask = (mask / max_val * 255).astype(np.uint8)
            
            logger.info(f"Forme finale de l'image: {input_image.shape}, dtype: {input_image.dtype}")
            logger.info(f"Forme finale du masque: {mask.shape}, dtype: {mask.dtype}, min: {np.min(mask)}, max: {np.max(mask)}")
            
            # Utiliser la variable d'environnement pour le nom de l'API
            api_name = os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_INPAINT_WITH_MASK")
            logger.info(f"Appel à l'API avec api_name={api_name}")
            
            # Vérifier si api_name est None ou vide
            if api_name is None or api_name == "":
                logger.warning("HF_SPACE_WATERMAK_REMOVAL_ROUTE_INPAINT_WITH_MASK n'est pas défini, utilisation de la valeur par défaut 'inpaint_with_mask'")
                api_name = "inpaint_with_mask"
            
            # Convertir les images en base64
            # Convertir l'image en base64
            pil_img = Image.fromarray(input_image)
            buffered_img = BytesIO()
            pil_img.save(buffered_img, format="PNG")
            img_str = base64.b64encode(buffered_img.getvalue()).decode()
            
            # Convertir le masque en base64
            pil_mask = Image.fromarray(mask)
            buffered_mask = BytesIO()
            pil_mask.save(buffered_mask, format="PNG")
            mask_str = base64.b64encode(buffered_mask.getvalue()).decode()
            
            logger.info("Images converties en base64 pour l'API")
            
            # Tenter d'appeler l'API avec les chaînes base64
            try:
                logger.info("Tentative d'appel à l'API avec les chaînes base64")
                
                # Appel à l'API avec les arguments positionnels (sans noms de paramètres)
                result = self.client.predict(
                    img_str,  # Premier argument: image en base64
                    mask_str,  # Deuxième argument: masque en base64
                    api_name=api_name
                )
                
                logger.info(f"Appel à l'API réussi, type de résultat: {type(result)}")
                
                # Traiter le résultat
                if result is None:
                    logger.warning("L'API a retourné None, utilisation de l'image d'origine")
                    return mask, input_image
                elif isinstance(result, np.ndarray):
                    return mask, result
                elif isinstance(result, str):
                    # Si le résultat est une chaîne base64, la décoder
                    if result.startswith("data:image"):
                        # Extraire la partie base64 de la chaîne data URL
                        base64_data = result.split(",")[1]
                        img_data = base64.b64decode(base64_data)
                    else:
                        # Sinon, supposer que c'est directement une chaîne base64
                        try:
                            img_data = base64.b64decode(result)
                        except:
                            # Si ce n'est pas une chaîne base64 valide et que c'est un chemin de fichier
                            if os.path.exists(result):
                                result_image = cv2.imread(result)
                                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                                return mask, result_image
                            else:
                                logger.warning(f"Résultat non reconnu: {result[:100]}...")
                                return mask, input_image
                    
                    # Convertir les données décodées en image
                    img = Image.open(BytesIO(img_data))
                    result_image = np.array(img)
                    return mask, result_image
                elif isinstance(result, tuple) and len(result) == 2:
                    return result
                else:
                    logger.warning(f"Type de résultat inattendu: {type(result)}")
                    return mask, input_image
                
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à l'API: {str(e)}")
                
                # En cas d'échec, retourner l'image d'origine
                return mask, input_image
                
        except Exception as e:
            # Capturer l'exception et la relancer pour interrompre le traitement
            logger.error(f"Erreur lors de l'inpainting: {str(e)}")
            # Retourner l'image d'origine en cas d'erreur
            return mask, input_image
    
    def detect_and_inpaint(self, 
                          input_image: np.ndarray, 
                          threshold: float = 0.85, 
                          max_bbox_percent: float = 10.0, 
                          remove_watermark_iterations: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Détecte les watermarks dans une image et applique l'inpainting pour les supprimer.
        
        Args:
            input_image (np.ndarray): Image d'entrée au format numpy array
            threshold (float, optional): Seuil de confiance pour la détection. Defaults to 0.85.
            max_bbox_percent (float, optional): Pourcentage maximal de la taille de l'image pour les bounding boxes. Defaults to 10.0.
            remove_watermark_iterations (int, optional): Nombre d'itérations d'inpainting. Defaults to 1.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple contenant (masque de détection, image traitée)
        """
        try:
            # Appel à l'API d'inpainting standard avec détection automatique
            result = self.client.predict(
                input_image,
                threshold,
                max_bbox_percent,
                remove_watermark_iterations,
                api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_INPAINT_IMAGE")
            )
            
            # Vérifier le format du résultat
            if isinstance(result, tuple) and len(result) == 2:
                # Si le résultat est déjà un tuple (masque, image_traitée)
                return result
            elif isinstance(result, np.ndarray):
                # Si le résultat est seulement l'image traitée, on retourne un masque vide
                logger.warning("L'API n'a pas retourné de masque de détection")
                mask = np.zeros_like(input_image)
                if len(mask.shape) == 3 and mask.shape[2] == 3:
                    mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
                return mask, result
            else:
                # Format inattendu
                raise ValueError(f"Format de résultat inattendu: {type(result)}")
                
        except Exception as e:
            logger.error(f"Erreur lors de la détection et de l'inpainting: {str(e)}")
            raise Exception(f"Erreur lors de la détection et de l'inpainting: {str(e)}")
    