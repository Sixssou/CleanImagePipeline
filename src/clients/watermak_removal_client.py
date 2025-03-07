from gradio_client import Client, handle_file
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
    
    def inpaint(self, input_image: Image.Image, mask: Image.Image) -> Tuple[bool, np.ndarray]:
        """
        Applique l'inpainting sur une image en utilisant un masque.
        
        Args:
            input_image (Image.Image): Image d'entrée au format PIL
            mask (Image.Image): Masque binaire où les pixels blancs (255) indiquent les zones à inpainter
            
        Returns:
            Tuple[bool, np.ndarray]: (succès, image inpainted)
        """
        try:
            logger.info("Début de l'inpainting")
            
            # Vérifier que les entrées sont bien des images PIL
            if not isinstance(input_image, Image.Image):
                logger.error(f"L'image d'entrée n'est pas une image PIL: {type(input_image)}")
                return False, None
            
            if not isinstance(mask, Image.Image):
                logger.error(f"Le masque n'est pas une image PIL: {type(mask)}")
                return False, None
            
            logger.info(f"Taille de l'image d'entrée: {input_image.size}")
            logger.info(f"Taille du masque: {mask.size}")
            
            # Convertir en numpy pour le traitement
            input_image_np = np.array(input_image)
            mask_np = np.array(mask)
            
            # Vérifier le nombre de pixels blancs dans le masque
            white_pixels = np.sum(mask_np > 127)
            logger.info(f"Nombre de pixels blancs dans le masque: {white_pixels}")
            
            # Si le masque est complètement noir ou presque, retourner l'image d'origine
            if white_pixels < 10:
                logger.warning("Masque presque vide, retour de l'image d'origine sans inpainting")
                return True, input_image_np
            
            # Vérifier que le masque est bien un tableau numpy 2D
            if len(mask_np.shape) > 2:
                mask_np = mask_np[:, :, 0]  # Prendre le premier canal si c'est une image RGB/RGBA
            
            # S'assurer que le masque est binaire (0 ou 255)
            mask_np = np.where(mask_np > 127, 255, 0).astype(np.uint8)
            
            try:
                # Préparation des images pour l'API Gradio
                logger.info("Préparation des images pour l'API Gradio")
                
                # Utiliser directement les images PIL pour l'API
                input_image_path = self._prepare_image_for_api(input_image, prefix="input_")
                mask_path = self._prepare_image_for_api(mask, prefix="mask_", is_mask=True)
                
                logger.info(f"Fichiers préparés: {input_image_path}, {mask_path}")
                
                # Appel à l'API Gradio avec les chemins de fichiers
                result = self.client.predict(
                    input_image_path,
                    mask_path,
                    api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_INPAINT_WITH_MASK")
                )
                
                # Nettoyer les fichiers temporaires
                try:
                    if os.path.exists(input_image_path):
                        os.remove(input_image_path)
                    if os.path.exists(mask_path):
                        os.remove(mask_path)
                except Exception as cleanup_error:
                    logger.warning(f"Erreur lors du nettoyage des fichiers temporaires: {cleanup_error}")
                
                # Vérifier le résultat
                if result is None:
                    logger.error("L'API a retourné None")
                    return False, input_image_np
                
                # Si le résultat est une chaîne (chemin de fichier ou URL), charger l'image
                if isinstance(result, str):
                    logger.info(f"Résultat reçu sous forme de chaîne: {result}")
                    try:
                        if result.startswith(('http://', 'https://')):
                            # Charger depuis URL
                            response = requests.get(result)
                            if response.status_code == 200:
                                result_image = np.array(Image.open(BytesIO(response.content)))
                            else:
                                raise ValueError(f"Échec du téléchargement de l'image: {response.status_code}")
                        elif os.path.exists(result):
                            # Charger depuis fichier local
                            result_image = cv2.imread(result)
                            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                        else:
                            # Essayer de décoder comme base64
                            result_image = self._base64_to_numpy(result)
                            
                        if result_image is None:
                            raise ValueError("Impossible de décoder l'image résultante")
                            
                        return True, result_image
                    except Exception as load_error:
                        logger.error(f"Erreur lors du chargement du résultat: {load_error}")
                        return False, input_image_np
                
                # Si le résultat est déjà un array numpy
                if isinstance(result, np.ndarray):
                    logger.info(f"Résultat reçu sous forme de numpy array: {result.shape}")
                    return True, result
                    
                # Format non géré
                logger.error(f"Format de résultat non géré: {type(result)}")
                return False, input_image_np
                
            except Exception as api_error:
                logger.error(f"Erreur lors de l'appel à l'API: {str(api_error)}")
                logger.exception(api_error)  # Log la stack trace complète
                return False, input_image_np
            
        except Exception as e:
            logger.error(f"Erreur lors de l'inpainting: {str(e)}")
            logger.exception(e)  # Log la stack trace complète
            return False, np.array(input_image)

    def _prepare_image_for_api(self, image, prefix="img_", is_mask=False):
        """
        Prépare une image pour l'envoi à l'API en la convertissant 
        en fichier temporaire.
        
        Args:
            image: Image source (np.ndarray, PIL.Image, dict ou str)
            prefix: Préfixe pour le nom du fichier temporaire
            is_mask: True si l'image est un masque binaire
            
        Returns:
            str: Chemin vers le fichier temporaire créé
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=prefix).name
        
        # Cas 1: Image est déjà un chemin de fichier
        if isinstance(image, str) and (os.path.exists(image) or image.startswith('http')):
            if os.path.exists(image):
                # Copier le fichier local
                img = cv2.imread(image)
                cv2.imwrite(temp_file, img)
            else:
                # Télécharger l'URL
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
                img.save(temp_file)
            return temp_file
        
        # Cas 2: Image est un tableau numpy
        elif isinstance(image, np.ndarray):
            if is_mask:
                # Assurer que le masque est en niveaux de gris
                if len(image.shape) > 2 and image.shape[2] > 1:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # Normaliser entre 0 et 255
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
            
            # Sauvegarder l'image
            if is_mask and len(image.shape) == 2:
                cv2.imwrite(temp_file, image)
            else:
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Image RGB
                    cv2.imwrite(temp_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                else:
                    # Autres formats
                    Image.fromarray(image).save(temp_file)
            return temp_file
        
        # Cas 3: Image est un objet PIL
        elif isinstance(image, Image.Image):
            image.save(temp_file)
            return temp_file
        
        # Cas 4: Image est un dictionnaire (cas de l'éditeur Gradio)
        elif isinstance(image, dict):
            if 'composite' in image:
                img_array = image['composite']
                Image.fromarray(img_array).save(temp_file)
                return temp_file
            elif 'layers' in image and image['layers']:
                # Prendre la première couche ou la dernière selon le contexte
                layer = image['layers'][-1 if is_mask else 0]['content']
                Image.fromarray(layer).save(temp_file)
                return temp_file
            
        # Cas d'erreur
        raise ValueError(f"Format d'image non pris en charge: {type(image)}")
    
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
    