from gradio_client import Client, handle_file, file
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
import uuid

from src.utils.image_utils import TEMP_DIR

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
        Effectue l'inpainting d'une image en utilisant un masque.
        
        Args:
            input_image (Image.Image): Image d'entrée à traiter
            mask (Image.Image): Masque binaire où les zones blanches seront inpaintées
            
        Returns:
            Tuple[bool, np.ndarray]: Tuple contenant (succès, image résultante)
        """
        logger.info("=== DÉBUT inpaint ===")
        logger.info(f"Type de input_image: {type(input_image)}")
        if isinstance(input_image, Image.Image):
            logger.info(f"Mode de input_image: {input_image.mode}, taille: {input_image.size}")
            # Sauvegarder l'image d'entrée pour débogage
            try:
                debug_path = os.path.join(TEMP_DIR, f"debug_input_{uuid.uuid4()}.png")
                input_image.save(debug_path)
                logger.info(f"Image d'entrée sauvegardée pour débogage: {debug_path}")
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder l'image d'entrée pour débogage: {e}")
        
        logger.info(f"Type de mask: {type(mask)}")
        if isinstance(mask, Image.Image):
            logger.info(f"Mode de mask: {mask.mode}, taille: {mask.size}")
            # Sauvegarder le masque pour débogage
            try:
                debug_path = os.path.join(TEMP_DIR, f"debug_mask_{uuid.uuid4()}.png")
                mask.save(debug_path)
                logger.info(f"Masque sauvegardé pour débogage: {debug_path}")
            except Exception as e:
                logger.warning(f"Impossible de sauvegarder le masque pour débogage: {e}")
            
            # Analyser le contenu du masque pour détecter s'il y a des pixels non noirs (valeurs > 0)
            if mask.mode == "L":
                mask_array = np.array(mask)
                non_black_pixels = np.sum(mask_array > 0)
                total_pixels = mask_array.size
                logger.info(f"Masque: {non_black_pixels} pixels non noirs sur {total_pixels} ({non_black_pixels/total_pixels*100:.2f}%)")
                
                # Afficher les valeurs uniques et leur fréquence dans le masque
                unique_values, counts = np.unique(mask_array, return_counts=True)
                logger.info(f"Valeurs uniques dans le masque: {list(zip(unique_values, counts))}")
                
                # Vérifier s'il y a trop peu de pixels non noirs (< 0.01% de l'image)
                if non_black_pixels < total_pixels * 0.0001:
                    logger.warning(f"ATTENTION: Le masque contient très peu de pixels non noirs ({non_black_pixels} pixels, {non_black_pixels/total_pixels*100:.4f}% de l'image)")
                
                # Vérifier si le masque est entièrement noir ou blanc
                if len(unique_values) <= 2 and 0 in unique_values:
                    if len(unique_values) == 1:  # Uniquement des zéros
                        logger.warning("ATTENTION: Le masque est entièrement noir (tous les pixels sont à 0)")
                    else:  # Deux valeurs: 0 et une autre
                        non_zero_value = unique_values[1] if unique_values[0] == 0 else unique_values[0]
                        logger.info(f"Le masque est binaire avec des valeurs 0 et {non_zero_value}")
                else:
                    logger.info(f"Le masque contient {len(unique_values)} valeurs différentes")
        
        # Vérifier que les entrées sont des images PIL
        if not isinstance(input_image, Image.Image):
            logger.error(f"L'image d'entrée n'est pas une instance de PIL.Image: {type(input_image)}")
            return False, None
        
        if not isinstance(mask, Image.Image):
            logger.error(f"Le masque n'est pas une instance de PIL.Image: {type(mask)}")
            return False, None
        
        # Vérifier que les deux images ont la même taille
        if input_image.size != mask.size:
            logger.error(f"L'image et le masque n'ont pas la même taille: {input_image.size} vs {mask.size}")
            # Redimensionner le masque pour correspondre à l'image
            logger.info(f"Redimensionnement du masque pour correspondre à l'image...")
            mask = mask.resize(input_image.size, Image.LANCZOS)
            logger.info(f"Masque redimensionné: {mask.size}")
        
        try:
            # Préparer les images pour l'API
            logger.info("Préparation des images pour l'API...")
            try:
                input_image_path = self._prepare_image_for_api(input_image, prefix="input_")
                logger.info(f"Chemin de l'image d'entrée préparée: {input_image_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la préparation de l'image d'entrée: {e}")
                logger.exception(e)
                raise
            
            try:
                # S'assurer que le masque est en mode L (niveaux de gris)
                if mask.mode != "L":
                    logger.info(f"Conversion du masque du mode {mask.mode} au mode L")
                    mask = mask.convert("L")
                
                mask_path = self._prepare_image_for_api(mask, prefix="mask_", is_mask=True)
                logger.info(f"Chemin du masque préparé: {mask_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la préparation du masque: {e}")
                logger.exception(e)
                raise
            
            logger.info(f"Chemins préparés: input_image_path={input_image_path}, mask_path={mask_path}")
            
            # Importer file de gradio_client pour l'envoi de fichiers
            from gradio_client import file
            
            # Vérifier que les chemins existent
            if not os.path.exists(input_image_path):
                logger.error(f"Le chemin de l'image d'entrée n'existe pas: {input_image_path}")
                return False, None
            
            if not os.path.exists(mask_path):
                logger.error(f"Le chemin du masque n'existe pas: {mask_path}")
                return False, None
            
            # Appeler l'API d'inpainting avec le masque en utilisant file()
            logger.info("Appel de l'API d'inpainting...")
            try:
                # Obtenir le nom de l'API à partir des variables d'environnement
                api_name = os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_INPAINT_WITH_MASK", "/inpaint_with_mask")
                logger.info(f"Utilisation de l'API: {api_name}")
                
                result = self.client.predict(
                    file(input_image_path),  # Utiliser file() pour envoyer le fichier
                    file(mask_path),         # Utiliser file() pour envoyer le fichier
                    api_name=api_name
                )
                logger.info(f"Résultat de l'API: {result}")
            except Exception as e:
                logger.error(f"Erreur lors de l'appel à l'API d'inpainting: {e}")
                logger.exception(e)
                return False, None
            
            # Vérifier si le résultat est valide
            if result is not None and isinstance(result, str) and os.path.exists(result):
                # Charger l'image résultante
                logger.info(f"Chargement de l'image résultante: {result}")
                try:
                    result_image = np.array(Image.open(result))
                    logger.info(f"Image résultante chargée, forme: {result_image.shape}")
                    logger.info("=== FIN inpaint (succès) ===")
                    return True, result_image
                except Exception as e:
                    logger.error(f"Erreur lors du chargement de l'image résultante: {e}")
                    logger.exception(e)
                    return False, None
            else:
                if result is None:
                    logger.error("L'API a retourné None")
                elif not isinstance(result, str):
                    logger.error(f"L'API a retourné un type inattendu: {type(result)}")
                else:
                    logger.error(f"Le fichier de résultat n'existe pas: {result}")
                return False, None
                
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'inpainting: {e}")
            logger.exception(e)
            return False, None
        
        logger.info("=== FIN inpaint (échec) ===")
        return False, None

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
        logger.info(f"=== DÉBUT _prepare_image_for_api (prefix={prefix}, is_mask={is_mask}) ===")
        logger.info(f"Type de l'image à préparer: {type(image)}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png", prefix=prefix).name
        logger.info(f"Fichier temporaire créé: {temp_file}")
        
        # Cas 1: Image est déjà un chemin de fichier
        if isinstance(image, str) and (os.path.exists(image) or image.startswith('http')):
            logger.info(f"L'image est un chemin: {image}")
            if os.path.exists(image):
                # Copier le fichier local
                logger.info(f"Copie du fichier local: {image}")
                try:
                    img = cv2.imread(image)
                    if img is None:
                        logger.error(f"Impossible de lire l'image avec cv2.imread: {image}")
                        # Essayer avec PIL
                        pil_img = Image.open(image)
                        pil_img.save(temp_file)
                        logger.info(f"Image copiée avec PIL: {image} -> {temp_file}")
                    else:
                        cv2.imwrite(temp_file, img)
                        logger.info(f"Image copiée avec cv2: {image} -> {temp_file}")
                except Exception as e:
                    logger.error(f"Erreur lors de la copie du fichier local: {e}")
                    logger.exception(e)
                    raise
            else:
                # Télécharger l'URL
                logger.info(f"Téléchargement de l'URL: {image}")
                try:
                    response = requests.get(image)
                    img = Image.open(BytesIO(response.content))
                    img.save(temp_file)
                    logger.info(f"Image téléchargée: {image} -> {temp_file}")
                except Exception as e:
                    logger.error(f"Erreur lors du téléchargement de l'URL: {e}")
                    logger.exception(e)
                    raise
            logger.info(f"=== FIN _prepare_image_for_api (chemin) ===")
            return temp_file
        
        # Cas 2: Image est un tableau numpy
        elif isinstance(image, np.ndarray):
            logger.info(f"L'image est un tableau numpy de forme {image.shape}")
            try:
                if is_mask:
                    # Assurer que le masque est en niveaux de gris
                    if len(image.shape) > 2 and image.shape[2] > 1:
                        logger.info("Conversion du masque en niveaux de gris")
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    # Normaliser entre 0 et 255
                    if image.max() <= 1.0:
                        logger.info("Normalisation du masque entre 0 et 255")
                        image = (image * 255).astype(np.uint8)
                
                # Sauvegarder l'image
                if is_mask and len(image.shape) == 2:
                    logger.info("Sauvegarde du masque en niveaux de gris")
                    cv2.imwrite(temp_file, image)
                else:
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Image RGB
                        logger.info("Sauvegarde de l'image RGB")
                        cv2.imwrite(temp_file, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    else:
                        # Autres formats
                        logger.info("Sauvegarde de l'image avec PIL")
                        Image.fromarray(image).save(temp_file)
                logger.info(f"Image numpy sauvegardée: {temp_file}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du tableau numpy: {e}")
                logger.exception(e)
                raise
            logger.info(f"=== FIN _prepare_image_for_api (numpy) ===")
            return temp_file
        
        # Cas 3: Image est un objet PIL
        elif isinstance(image, Image.Image):
            logger.info(f"L'image est un objet PIL.Image de mode {image.mode} et taille {image.size}")
            try:
                image.save(temp_file)
                logger.info(f"Image PIL sauvegardée: {temp_file}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de l'image PIL: {e}")
                logger.exception(e)
                raise
            logger.info(f"=== FIN _prepare_image_for_api (PIL) ===")
            return temp_file
        
        # Cas 4: Image est un dictionnaire (cas de l'éditeur Gradio)
        elif isinstance(image, dict):
            logger.info(f"L'image est un dictionnaire avec clés {list(image.keys())}")
            try:
                if 'composite' in image and isinstance(image['composite'], np.ndarray):
                    logger.info("Utilisation de la clé 'composite' (numpy array)")
                    img_array = image['composite']
                    Image.fromarray(img_array).save(temp_file)
                    logger.info(f"Image composite sauvegardée: {temp_file}")
                    return temp_file
                elif 'composite' in image and isinstance(image['composite'], Image.Image):
                    logger.info("Utilisation de la clé 'composite' (PIL.Image)")
                    image['composite'].save(temp_file)
                    logger.info(f"Image composite sauvegardée: {temp_file}")
                    return temp_file
                elif 'background' in image and isinstance(image['background'], Image.Image):
                    logger.info("Utilisation de la clé 'background' (PIL.Image)")
                    image['background'].save(temp_file)
                    logger.info(f"Image background sauvegardée: {temp_file}")
                    return temp_file
                elif 'layers' in image and image['layers'] and isinstance(image['layers'], list):
                    # Prendre la première couche ou la dernière selon le contexte
                    layer_index = -1 if is_mask else 0
                    logger.info(f"Utilisation de la couche {layer_index} (is_mask={is_mask})")
                    
                    if isinstance(image['layers'][layer_index], dict) and 'content' in image['layers'][layer_index]:
                        layer_content = image['layers'][layer_index]['content']
                        if isinstance(layer_content, np.ndarray):
                            Image.fromarray(layer_content).save(temp_file)
                            logger.info(f"Image de la couche sauvegardée: {temp_file}")
                            return temp_file
                        elif isinstance(layer_content, Image.Image):
                            layer_content.save(temp_file)
                            logger.info(f"Image de la couche sauvegardée: {temp_file}")
                            return temp_file
                    else:
                        logger.error(f"Format de couche non reconnu: {type(image['layers'][layer_index])}")
                
                # Si aucune des options ci-dessus n'a fonctionné, essayer de trouver une image dans le dictionnaire
                for key, value in image.items():
                    if isinstance(value, np.ndarray):
                        logger.info(f"Utilisation de la clé '{key}' (numpy array)")
                        Image.fromarray(value).save(temp_file)
                        logger.info(f"Image sauvegardée: {temp_file}")
                        return temp_file
                    elif isinstance(value, Image.Image):
                        logger.info(f"Utilisation de la clé '{key}' (PIL.Image)")
                        value.save(temp_file)
                        logger.info(f"Image sauvegardée: {temp_file}")
                        return temp_file
                
                logger.error(f"Aucune image trouvée dans le dictionnaire")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde du dictionnaire: {e}")
                logger.exception(e)
                raise
        
        # Cas d'erreur
        error_msg = f"Format d'image non pris en charge: {type(image)}"
        logger.error(error_msg)
        logger.info(f"=== FIN _prepare_image_for_api (erreur) ===")
        raise ValueError(error_msg)
    
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
    
    def view_api(self) -> dict:
        """
        Récupère les informations sur l'API en appelant la méthode view_api du client Gradio.
        
        Returns:
            dict: Un dictionnaire contenant les informations sur l'API
        """
        logger.info("Récupération des informations sur l'API via client.view_api()")
        
        try:
            # Appeler directement la méthode view_api du client Gradio
            return self.client.view_api(return_format="dict")
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à client.view_api(): {e}")
            # Renvoyer un dictionnaire vide en cas d'erreur
            return {"error": f"Impossible d'accéder aux informations de l'API: {str(e)}"}
    