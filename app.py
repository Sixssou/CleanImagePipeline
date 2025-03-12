import gradio as gr
from loguru import logger
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import tempfile
import cv2
from datetime import datetime
import base64
import uuid
import shutil
from PIL import ImageDraw, ImageFont

from src.clients.gsheet_client import GSheetClient
from src.clients.shopify_client import ShopifyClient
from src.clients.watermak_removal_client import WatermakRemovalClient
from src.pipeline.pipeline import CleanImagePipeline
from src.utils.image_utils import download_image, visualize_mask, TEMP_DIR

# Définir TEMP_DIR au début du fichier, après les imports
TEMP_DIR = os.path.join(tempfile.gettempdir(), "cleanimage")
os.makedirs(TEMP_DIR, exist_ok=True)

load_dotenv()

# Variables globales pour les clients et le pipeline
pipeline = None
sheet_name = None  # Variable globale pour stocker le nom de l'onglet

# Définir les variables d'état pour le suivi des images
current_image_index = gr.State(value=0)  # Index de l'image actuelle
remaining_images = gr.State(value=[])    # Liste des images restantes à traiter

def initialize_clients():
    """
    Initialise et retourne les clients nécessaires pour l'application.
    """
    global pipeline
    
    # Initialisation du client GSheet
    credentials_file = os.getenv("GSHEET_CREDENTIALS_FILE")
    spreadsheet_id = os.getenv("GSHEET_ID")
    gsheet_client = GSheetClient(credentials_file, spreadsheet_id)

    # Initialisation du client Shopify
    shopify_api_key = os.getenv("SHOPIFY_API_VERSION")
    shopify_password = os.getenv("SHOPIFY_ACCESS_TOKEN")
    shopify_store_name = os.getenv("SHOPIFY_STORE_DOMAIN")
    shopify_client = ShopifyClient(shopify_api_key, 
                                   shopify_password, 
                                   shopify_store_name)

    # Initialisation du client WatermarkRemoval
    hf_token = os.getenv("HF_TOKEN")
    space_url = os.getenv("HF_SPACE_WATERMAK_REMOVAL")
    watermak_removal_client = WatermakRemovalClient(hf_token, space_url)
    
    # Initialisation du pipeline
    pipeline = CleanImagePipeline(gsheet_client, shopify_client, watermak_removal_client)
    
    return pipeline

def gsheet_test_connection(input_gheet_id: str, input_gheet_sheet: str):
    """Test de connexion à Google Sheets"""
    if pipeline is None:
        initialize_clients()
    return pipeline.test_connections(gsheet_id=input_gheet_id, gsheet_sheet=input_gheet_sheet).get("gsheet", False)

def shopify_test_connection(input_shopify_domain: str, input_shopify_api_version: str, input_shopify_api_key: str):
    """Test de connexion à Shopify"""
    if pipeline is None:
        initialize_clients()
    return pipeline.test_connections(
        shopify_domain=input_shopify_domain,
        shopify_api_version=input_shopify_api_version,
        shopify_api_key=input_shopify_api_key
    ).get("shopify", False)

def remove_background(input_image_url_remove_background: str):
    """Supprime l'arrière-plan d'une image"""
    if pipeline is None:
        initialize_clients()
    success, result = pipeline.image_processor.remove_background(input_image_url_remove_background)
    return result if success else None

def detect_wm(image_url: str, 
              threshold: float = 0.85,  # Ce paramètre n'est plus utilisé mais conservé pour compatibilité
              max_bbox_percent: float = 10.0, 
              bbox_enlargement_factor: float = 1.2):
    """Détecte les watermarks dans une image"""
    if pipeline is None:
        initialize_clients()
    return pipeline.image_processor.detect_watermarks(
        image_url=image_url,
        max_bbox_percent=max_bbox_percent,
        bbox_enlargement_factor=bbox_enlargement_factor
    )

def remove_wm(image_url, 
              threshold=0.85, 
              max_bbox_percent=10.0, 
              remove_background_option=False, 
              add_watermark_option=False, 
              watermark=None,
              bbox_enlargement_factor=1.5,
              remove_watermark_iterations=1):
    """Supprime les watermarks d'une image"""
    if pipeline is None:
        initialize_clients()
    return pipeline.image_processor.watermak_removal_client.remove_wm(
        image_url=image_url, 
        threshold=threshold, 
        max_bbox_percent=max_bbox_percent, 
        remove_background_option=remove_background_option, 
        add_watermark_option=add_watermark_option, 
        watermark=watermark, 
        bbox_enlargement_factor=bbox_enlargement_factor, 
        remove_watermark_iterations=remove_watermark_iterations
    )

def clean_image_pipeline(image_count: int, sheet_name: str):
    """Lance le pipeline de nettoyage des images"""
    if pipeline is None:
        initialize_clients()
    return pipeline.process_images_batch(
        image_count=image_count,
        sheet_name=sheet_name,
        remove_background=False,
        add_watermark=True,
        watermark_text="www.inflatable-store.com"
    )

def process_single_image(lien_image_source, supprimer_background=False):
    """Traite une seule image avec l'étape manuelle d'édition"""
    if pipeline is None:
        initialize_clients()
    try:
        # Télécharger l'image source pour l'édition manuelle
        image_source = download_image(lien_image_source)
        
        # Détection des watermarks
        original_image, image_with_bbox, detection_mask = pipeline.image_processor.detect_watermarks(
            image_url=lien_image_source
        )
        
        return image_source, detection_mask
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image {lien_image_source}: {str(e)}")
        return None, None

def clean_image_pipeline_manual(image_count: int, sheet_name: str):
    """
    Récupère les images à traiter pour l'édition manuelle.
    
    Args:
        image_count: Nombre d'images à traiter
        sheet_name: Nom de l'onglet dans Google Sheets
        
    Returns:
        list: Liste des images à traiter avec leurs informations
    """
    global pipeline
    if pipeline is None:
        initialize_clients()

    return pipeline.prepare_images_for_manual_edit(image_count, sheet_name)

def apply_manual_edits(index, image, edited_mask, supprimer_background=False, add_watermark=True, watermark="www.inflatable-store.com"):
    """Applique les modifications manuelles à une image"""
    global pipeline
    if pipeline is None:
        initialize_clients()
    
    return pipeline.image_processor.apply_manual_edits(
        image=image,
        edited_mask=edited_mask,
        supprimer_background=supprimer_background,
        add_watermark=add_watermark,
        watermark_text=watermark
    )

def inpaint_image(input_image, threshold, max_bbox_percent, remove_watermark_iterations):
    """Applique l'inpainting avec détection automatique des watermarks"""
    if pipeline is None:
        initialize_clients()
    
    return pipeline.image_processor.watermak_removal_client.detect_and_inpaint(
        input_image=input_image, 
        threshold=threshold, 
        max_bbox_percent=max_bbox_percent, 
        remove_watermark_iterations=remove_watermark_iterations
    )

def inpaint_with_mask(input_image, input_mask):
    """Effectue l'inpainting d'une image en utilisant un masque fourni"""
    if pipeline is None:
        initialize_clients()
    
    # Appeler l'API d'inpainting
    success, result = pipeline.image_processor.inpaint_image(input_image, input_mask)
    
    if not success or result is None:
        raise gr.Error("L'inpainting a échoué")
    
    # Sauvegarder le résultat
    output_path = os.path.join(TEMP_DIR, f"inpainted_{uuid.uuid4()}.png")
    Image.fromarray(result).save(output_path)
    
    return output_path

def detect_watermarks_from_url(url_input, 
                              input_prompt, 
                              input_max_new_tokens, 
                              input_early_stopping, 
                              input_do_sample, 
                              input_num_beams, 
                              input_num_return_sequences, 
                              input_temperature, 
                              input_top_k, 
                              input_top_p, 
                              input_repetition_penalty, 
                              input_length_penalty, 
                              max_bbox_percent, 
                              bbox_enlargement_factor):
    """Détecte les watermarks dans une image à partir d'une URL avec les paramètres avancés"""
    if pipeline is None:
        initialize_clients()
    
    try:
        return pipeline.image_processor.detect_watermarks(
            image_url=url_input,
            max_bbox_percent=max_bbox_percent,
            bbox_enlargement_factor=bbox_enlargement_factor,
            prompt=input_prompt,
            max_new_tokens=input_max_new_tokens,
            early_stopping=input_early_stopping,
            do_sample=input_do_sample,
            num_beams=input_num_beams,
            num_return_sequences=input_num_return_sequences,
            temperature=input_temperature,
            top_k=input_top_k,
            top_p=input_top_p,
            repetition_penalty=input_repetition_penalty,
            length_penalty=input_length_penalty
        )
    except Exception as e:
        logger.error(f"Erreur lors de la détection des watermarks: {str(e)}")
        return None, None, None

def test_inpaint_api():
    """Fonction de test pour vérifier l'API d'inpainting"""
    if pipeline is None:
        initialize_clients()
    
    try:
        # Créer une image de test et un masque en PIL
        image = Image.new('RGB', (100, 100), (0, 0, 0))
        # Dessiner un carré blanc au milieu
        for y in range(25, 75):
            for x in range(25, 75):
                image.putpixel((x, y), (255, 255, 255))
        
        # Créer un masque de test (un petit carré blanc au milieu)
        mask = Image.new('L', (100, 100), 0)
        # Dessiner un petit carré blanc au milieu
        for y in range(40, 60):
            for x in range(40, 60):
                mask.putpixel((x, y), 255)
        
        # Appel via la méthode inpaint
        success, inpainted_image = pipeline.image_processor.inpaint_image(image, mask)
        
        if success and inpainted_image is not None:
            logger.info(f"Test de la méthode inpaint réussi, type de résultat: {type(inpainted_image)}")
            return "Test réussi"
        else:
            return "Test échoué: l'inpainting n'a pas réussi"
    except Exception as e:
        logger.error(f"Erreur lors du test de l'API d'inpainting: {str(e)}")
        logger.exception(e)
        return f"Erreur: {str(e)}"

def visualize_mask(mask_img):
    """
    Crée une version colorée du masque pour une meilleure visualisation.
    
    Args:
        mask_img: Image PIL en mode L (niveaux de gris), tableau NumPy ou chemin de fichier
        
    Returns:
        np.ndarray: Image colorée du masque
    """
    if mask_img is None:
        return None
    
    # Convertir en tableau numpy si ce n'est pas déjà fait
    if isinstance(mask_img, str):  # Si c'est un chemin de fichier
        try:
            mask_img = Image.open(mask_img)
        except Exception as e:
            logger.error(f"Erreur lors du chargement du masque à partir du chemin: {str(e)}")
            return None
    
    if isinstance(mask_img, Image.Image):
        mask_data = np.array(mask_img)
    else:
        mask_data = mask_img
    
    # Créer une image colorée
    if len(mask_data.shape) == 2:  # Masque en niveaux de gris
        # Normaliser entre 0 et 255 si nécessaire
        if mask_data.max() > 0:
            mask_data = (mask_data / mask_data.max() * 255).astype(np.uint8)
        
        # Créer une image RGB
        colored_mask = np.zeros((mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)
        
        # Zones blanches en rouge vif
        colored_mask[:, :, 0] = mask_data  # Canal rouge
        
        return colored_mask
    elif len(mask_data.shape) == 3 and mask_data.shape[2] == 3:  # Déjà en RGB
        return mask_data
    else:
        return None

def process_edited_image(current_idx, images, index, edited_image, edited_mask=None, remove_bg_option=False, add_watermark_option=True, watermark_text="www.inflatable-store.com"):
    """
    Traite l'image éditée par l'utilisateur.
    
    Args:
        current_idx: Index actuel dans la liste des images
        images: Liste des images à traiter
        index: Index de sélection dans l'interface (pas utilisé directement)
        edited_image: Image éditée manuellement
        edited_mask: Masque édité manuellement (dessin sur l'image)
        remove_bg_option: Option pour supprimer l'arrière-plan
        add_watermark_option: Option pour ajouter un filigrane
        watermark_text: Texte du filigrane
        
    Returns:
        tuple: (current_idx, images, message, vis_mask, result_image, edit_panel_update)
    """
    if pipeline is None:
        initialize_clients()
        
    if 'gsheet_name' not in globals() or gsheet_name is None:
        sheet_name = "Auto-processed"
    else:
        sheet_name = gsheet_name
    
    try:
        if current_idx >= len(images) or current_idx < 0:
            return current_idx, images, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Extraire les informations de l'image
        idx, image_url, original_image, mask, inpainted_image, final_image, _ = images[current_idx]
        
        # Traitement du masque édité (extrait de l'ImageEditor)
        actual_mask = None
        
        if edited_mask is not None:
            logger.info(f"Type de données reçues du masque édité: {type(edited_mask)}")
            
            # Extraction du masque à partir de l'éditeur d'image
            if isinstance(edited_mask, dict):
                logger.info(f"Clés disponibles dans le masque: {edited_mask.keys()}")
                
                # Essayer d'extraire le masque à partir du premier calque
                if "layers" in edited_mask and edited_mask["layers"]:
                    logger.info(f"Nombre de calques: {len(edited_mask['layers'])}")
                    
                    # Extraire le premier calque (contenant le dessin utilisateur)
                    first_layer = edited_mask["layers"][0]
                    logger.info(f"Type du premier calque: {type(first_layer)}")
                    
                    try:
                        # Convertir directement le premier calque en masque
                        from PIL import Image
                        import numpy as np
                        from src.utils.image_utils import convert_to_pil_image
                        
                        # Tenter de convertir le calque en image PIL
                        mask_image = convert_to_pil_image(first_layer)
                        
                        # Convertir en niveau de gris si nécessaire
                        if mask_image.mode != 'L':
                            mask_image = mask_image.convert('L')
                        
                        # Vérifier que le masque contient des pixels blancs
                        mask_array = np.array(mask_image)
                        white_pixels = np.sum(mask_array > 128)
                        total_pixels = mask_array.size
                        white_percentage = (white_pixels / total_pixels) * 100
                        
                        logger.info(f"Masque extrait du premier calque: {white_pixels} pixels blancs ({white_percentage:.2f}%)")
                        
                        # Si le masque contient assez de pixels blancs, l'utiliser
                        if white_pixels > 0:
                            # Enregistrer le masque pour débogage
                            mask_debug_path = os.path.join(TEMP_DIR, "debug_edit_mask_1.png")
                            mask_image.save(mask_debug_path)
                            logger.info(f"Masque enregistré pour débogage: {mask_debug_path}")
                            actual_mask = mask_image
                        else:
                            logger.warning("Le masque ne contient pas de pixels blancs, ignoré")
                    except Exception as e:
                        logger.error(f"Erreur lors de l'extraction du masque: {str(e)}")
                else:
                    logger.warning("Aucun calque trouvé dans le masque")
            else:
                logger.warning(f"Format de masque non pris en charge: {type(edited_mask)}")
        
        # Si un masque édité a été fourni, utiliser apply_manual_edits sans fournir inpainted_result
        # pour forcer l'appel à l'API d'inpainting
        if actual_mask is not None:
            logger.info("Utilisation du masque édité pour appeler l'API d'inpainting")
            # MODIFICATION ICI: Nous passons l'image inpaintée comme paramètre inpainted_result
            # pour que la méthode process_edited_image utilise cette image comme base au lieu de l'image originale
            result = pipeline.process_edited_image(
                index=idx,
                image_url=image_url,
                edited_mask=actual_mask,
                inpainted_result=inpainted_image,  # Passer l'image inpaintée préalablement
                remove_background=remove_bg_option,
                add_watermark=add_watermark_option,
                watermark_text=watermark_text,
                sheet_name=sheet_name
            )
        # Sinon, utiliser le résultat de l'inpainting automatique
        else:
            logger.info("Utilisation du résultat d'inpainting automatique")
            result = pipeline.process_edited_image(
                index=idx,
                image_url=image_url,
                edited_mask=None,
                inpainted_result=edited_image,
                remove_background=remove_bg_option,
                add_watermark=add_watermark_option,
                watermark_text=watermark_text,
                sheet_name=sheet_name
            )
        
        # Mettre à jour l'image traitée dans notre liste
        images[current_idx] = (idx, image_url, original_image, mask, inpainted_image, result, True)
        
        # Afficher le masque extrait et l'image résultante
        vis_mask = visualize_mask(actual_mask) if actual_mask is not None else visualize_mask(mask)
        return current_idx, images, "Traitement terminé", vis_mask, result, gr.update(visible=True)
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image éditée: {str(e)}")
        import traceback
        traceback.print_exc()
        return current_idx, images, f"Erreur: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

def validate_automatic_processing(current_idx, images, remove_bg_option=False, add_watermark_option=True, watermark_text="www.inflatable-store.com"):
    """
    Valide le traitement automatique sans édition manuelle du masque.
    
    Args:
        current_idx: Index actuel dans la liste d'images
        images: Liste des images à traiter
        remove_bg_option: Supprimer l'arrière-plan de l'image traitée
        add_watermark_option: Ajouter un filigrane à l'image traitée
        watermark_text: Texte du filigrane
        
    Returns:
        tuple: (nouvel index, liste d'images mise à jour, résultat du traitement)
    """
    global pipeline
    global sheet_name  # Déclaration sur sa propre ligne
    if pipeline is None:
        initialize_clients()
    
    try:
        if current_idx >= len(images) or current_idx < 0:
            return current_idx, images, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Extraire les informations de l'image
        idx, image_url, original_image, mask, inpainted_image, final_image, _ = images[current_idx]
        
        # Utiliser le résultat de l'inpainting automatique
        result = pipeline.process_edited_image(
            index=idx,
            image_url=image_url,
            edited_mask=None,
            inpainted_result=inpainted_image,
            remove_background=remove_bg_option,
            add_watermark=add_watermark_option,
            watermark_text=watermark_text,
            sheet_name=sheet_name
        )
        
        if result:
            # Mise à jour de l'image traitée dans la liste avec l'URL de l'image uploadée
            images[current_idx] = (idx, image_url, original_image, mask, inpainted_image, final_image, remove_bg_option)
            return current_idx, images, result, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
        else:
            return current_idx, images, "Le traitement a échoué", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    except Exception as e:
        logger.error(f"Erreur lors de la validation du traitement automatique: {str(e)}")
        logger.exception(e)
        return current_idx, images, f"Erreur: {str(e)}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

def next_image(current_idx, images):
    """
    Passe à l'image suivante dans la liste des images à traiter.
    
    Args:
        current_idx: Index actuel dans la liste d'images
        images: Liste des images à traiter
        
    Returns:
        tuple: Informations sur l'image suivante
    """
    global pipeline
    global sheet_name
    try:
        # Vérifier s'il y a encore des images à traiter
        if not images or len(images) == 0:
            return (
                current_idx,
                images, 
                "Aucune image à traiter",
                None,
                None,
                None,
                None,
                False,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        # Passer à l'image suivante
        next_idx = current_idx + 1
        
        # Si nous avons atteint la fin de la liste, revenir au début
        if next_idx >= len(images):
            next_idx = 0
        
        # Récupérer les données de l'image suivante
        if next_idx < len(images):
            idx, image_url, original_image, mask_image, inpainted_image, final_image, bg_option = images[next_idx]
            
            # Vérifier si cette image a déjà été traitée ou s'il faut la traiter maintenant
            if mask_image is None or inpainted_image is None or final_image is None:
                try:
                    # Appeler remove_wm pour l'image actuelle
                    result = pipeline.image_processor.watermak_removal_client.remove_wm(
                        image_url=image_url,
                        threshold=0.85,
                        max_bbox_percent=10.0,
                        remove_background_option=False,
                        add_watermark_option=False,
                        watermark=None,
                        bbox_enlargement_factor=1.5,
                        remove_watermark_iterations=1
                    )
                    
                    if not result or len(result) != 4:
                        raise ValueError(f"Le traitement de l'image a échoué: {image_url}")
                    
                    # Extraire les résultats
                    bg_removed_image, mask_image, inpainted_image, final_image = result
                    
                    # Convertir les images pour l'interface Gradio
                    mask_pil = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image
                    inpainted_pil = Image.fromarray(inpainted_image) if isinstance(inpainted_image, np.ndarray) else inpainted_image
                    final_pil = Image.fromarray(final_image) if isinstance(final_image, np.ndarray) else final_image
                    
                    # Mettre à jour la liste des images avec l'image traitée
                    images[next_idx] = (idx, image_url, original_image, mask_pil, inpainted_pil, final_pil, bg_option)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de l'image {image_url}: {str(e)}")
                    logger.exception(e)
                    return (
                        current_idx,  # Rester sur l'image actuelle en cas d'erreur
                        images,
                        f"Erreur lors du traitement de l'image suivante: {str(e)}",
                        None,
                        None,
                        None,
                        None,
                        False,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            
            # Visualiser le masque en couleur pour une meilleure visibilité
            colored_mask = visualize_mask(mask_image)
            
            return (
                next_idx,
                images,
                f"Image {next_idx+1}/{len(images)} - Index {idx}",
                original_image,
                colored_mask,
                inpainted_pil,  # Utiliser les objets PIL déjà convertis
                final_pil,      # Utiliser les objets PIL déjà convertis
                inpainted_pil,  # Ajouter inpainted_pil pour inpainted_preview
                bg_option,
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True)
            )
        else:
            return (
                current_idx,
                images,
                "Toutes les images ont été traitées",
                None, 
                None,
                None,
                None,
                False,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    except Exception as e:
        logger.error(f"Erreur lors du passage à l'image suivante: {str(e)}")
        logger.exception(e)
        return (
            current_idx,
            images,
            f"Erreur: {str(e)}",
            None,
            None,
            None,
            None,
            False,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def start_manual_pipeline(image_count, sheet_name_param):
    """
    Démarre le pipeline de traitement manuel des images
    
    Args:
        image_count: Nombre d'images à traiter
        sheet_name_param: Nom de l'onglet dans Google Sheets
        
    Returns:
        tuple: Informations pour initialiser l'interface d'édition manuelle
    """
    global pipeline
    global sheet_name  # Déclaration sur sa propre ligne
    sheet_name = sheet_name_param  # Stocker dans une variable globale
    
    images_to_process = clean_image_pipeline_manual(image_count, sheet_name)
    if not images_to_process:
        # Retourner des valeurs par défaut pour tous les composants de sortie
        return (
            "Aucune image à traiter", 
            gr.update(visible=False), 
            0,                # current_idx
            [],               # images
            0,                # image_index (GSheet)
            "",               # image_source_url
            None,             # image_editor
            False,            # remove_bg_option
            True,             # add_watermark_option
            "www.inflatable-store.com",  # watermark_text
            "Aucune image à traiter",    # result_status
            None,             # original_image_display
            None,             # mask_display
            None,             # inpainted_image_display
            None,             # final_image_display
            None,             # processed_mask
            None,             # final_image
            gr.update(visible=False),  # results_row
            gr.update(visible=False),  # manual_edit_interface
            gr.update(visible=False)  # current_image_index
        )
    
    # Préparer la première image
    idx, url, original_image, bg_option = images_to_process[0]
    
    # Traiter la première image
    try:
        # Appeler remove_wm pour l'image actuelle
        result = pipeline.image_processor.watermak_removal_client.remove_wm(
            image_url=url,
            threshold=0.85,
            max_bbox_percent=10.0,
            remove_background_option=False,
            add_watermark_option=False,
            watermark=None,
            bbox_enlargement_factor=1.5,
            remove_watermark_iterations=1
        )
        
        if not result or len(result) != 4:
            raise ValueError(f"Le traitement de l'image a échoué: {url}")
        
        # Extraire les résultats
        bg_removed_image, mask_image, inpainted_image, final_image = result
        
        # Convertir les images pour l'interface Gradio
        mask_pil = Image.fromarray(mask_image) if isinstance(mask_image, np.ndarray) else mask_image
        inpainted_pil = Image.fromarray(inpainted_image) if isinstance(inpainted_image, np.ndarray) else inpainted_image
        final_pil = Image.fromarray(final_image) if isinstance(final_image, np.ndarray) else final_image
        
        # Créer la liste complète des images traitées
        processed_images = []
        # Ajouter l'image traitée à la première position
        processed_images.append((idx, url, original_image, mask_pil, inpainted_pil, final_pil, bg_option))
        # Ajouter les autres images à traiter
        for i in range(1, len(images_to_process)):
            image_idx, image_url, image_original, image_bg_option = images_to_process[i]
            processed_images.append((image_idx, image_url, image_original, None, None, None, image_bg_option))
        
        # Visualiser le masque en couleur pour une meilleure visibilité - utiliser mask_pil qui est déjà un objet PIL Image
        colored_mask = visualize_mask(mask_pil)
        
        return (
            f"Traitement de l'image {1}/{len(images_to_process)}",
            gr.update(visible=True),
            0,                # current_idx
            processed_images, # images
            idx,              # image_index (GSheet)
            url,              # image_source_url
            inpainted_pil,    # image_editor (initialiser avec l'image inpainted pour édition)
            bg_option,        # remove_bg_option
            True,             # add_watermark_option
            "www.inflatable-store.com",  # watermark_text
            f"Image {1}/{len(images_to_process)} - Index GSheet: {idx}",  # result_status
            original_image,   # original_image_display
            colored_mask,     # mask_display
            inpainted_pil,    # inpainted_image_display
            final_pil,        # final_image_display
            inpainted_pil,    # inpainted_preview (nouvel élément pour l'éditeur)
            None,             # processed_mask (sera affiché après traitement)
            None,             # final_image (sera affiché après traitement)
            gr.update(visible=False),   # results_row
            gr.update(visible=True),    # manual_edit_interface (répété)
            0                 # current_image_index (répété)
        )
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la première image: {str(e)}")
        logger.exception(e)
        return (
            f"Erreur: {str(e)}", 
            gr.update(visible=False), 
            0,                # current_idx
            [],               # images
            0,                # image_index (GSheet)
            "",               # image_source_url
            None,             # image_editor
            False,            # remove_bg_option
            True,             # add_watermark_option
            "www.inflatable-store.com",  # watermark_text
            f"Erreur: {str(e)}",    # result_status
            None,             # original_image_display
            None,             # mask_display
            None,             # inpainted_image_display
            None,             # final_image_display
            None,             # inpainted_preview
            None,             # processed_mask
            None,             # final_image
            gr.update(visible=False),  # results_row
            gr.update(visible=False),  # manual_edit_interface (répété)
            0                 # current_image_index (répété)
        )

# Interface Gradio
with gr.Blocks() as interfaces:
    gr.Markdown("# Détection et Suppression de Watermark avec watermak-removal")
    
    with gr.Tabs():
        with gr.Tab("Gsheet Client"):
            with gr.Row():
                with gr.Column():
                    input_gheet_id = gr.Textbox(label="ID du Gsheet", value=os.getenv("GSHEET_ID"))
                    input_gheet_sheet = gr.Textbox(label="Nom de le sheet", value=os.getenv("GSHEET_SHEET_SPLIT_TAB_NAME"))
                    test_connection = gr.Button("Test Connection")
                
                with gr.Column():
                    output_test_connection = gr.Textbox(label="Réponse du test", value="Réponse du test")
            
            test_connection.click(
                gsheet_test_connection,
                inputs=[input_gheet_id, input_gheet_sheet],
                outputs=[output_test_connection]
            )
        with gr.Tab("Shopify Client"):
            with gr.Row():
                with gr.Column():
                    input_shopify_domain = gr.Textbox(label="Domaine de la boutique Shopify", value=os.getenv("SHOPIFY_STORE_DOMAIN"))
                    input_shopify_api_version = gr.Textbox(label="Version de l'API Shopify", value=os.getenv("SHOPIFY_API_VERSION"))
                    input_shopify_api_key = gr.Textbox(label="Shopify Token API Key", value=os.getenv("SHOPIFY_ACCESS_TOKEN"))
                    test_connection = gr.Button("Test Connection")
                
                with gr.Column():
                    output_shopify_test_connection = gr.Textbox(label="Réponse du test", value="Réponse du test")
            
            test_connection.click(
                shopify_test_connection,
                inputs=[input_shopify_domain, 
                        input_shopify_api_version,
                        input_shopify_api_key],
                outputs=[output_shopify_test_connection]
            )
        with gr.Tab("Watermak Removal Client"):
            with gr.Accordion("Background Removal"):
                with gr.Row():
                    with gr.Column():
                        input_image_url_remove_background = gr.Textbox(label="Url de l'image", value=os.getenv("TEST_PHOTO_URL_1"))
                        test_remove_background = gr.Button("Remove background from image")
                    
                    with gr.Column():
                        output_remove_background = gr.Image(label="Image sans arrière-plan", type="numpy")
                
                test_remove_background.click(
                    remove_background,
                    inputs=[input_image_url_remove_background],
                    outputs=[output_remove_background]
                )
            
            with gr.Accordion("Watermark Detection"):
                with gr.Row():
                    with gr.Column():
                        input_image_url_watermark_detection = gr.Textbox(label="Url de l'image", value=os.getenv("TEST_PHOTO_URL_1"))
                        threshold_watermark_detection = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.05,label="Seuil de confiance")
                        max_bbox_percent_watermark_detection = gr.Slider(minimum=1, maximum=100, value=10, step=1,
                                              label="Maximal bbox percent")
                        bbox_enlargement_factor_watermark_detection = gr.Slider(minimum=1, maximum=100, value=10, step=1,
                                              label="Facteur d'agrandissement des bbox")
                        test_watermark_detection = gr.Button("Detect watermarks from image")
                    with gr.Column():
                        output_watermark_detection = gr.Image(label="Image avec détection", type="numpy")
                        output_watermark_detection_mask = gr.Image(label="Masque de détection", type="numpy")
                
                test_watermark_detection.click(
                    detect_wm,
                    inputs=[input_image_url_watermark_detection, 
                            threshold_watermark_detection,
                            max_bbox_percent_watermark_detection,
                            bbox_enlargement_factor_watermark_detection],
                    outputs=[output_watermark_detection, 
                             output_watermark_detection_mask])

            with gr.Accordion("Remove Watermark"):
                with gr.Row():
                    with gr.Column():
                        url_input = gr.Textbox(label="URL de l'image", value=os.getenv("TEST_PHOTO_URL_1"))
                        threshold_url = gr.Slider(minimum=0.0, maximum=1.0, value=0.85,
                                                label="Seuil de confiance")
                        max_bbox_percent = gr.Slider(minimum=1, maximum=100, value=10, step=1,
                                                label="Maximal bbox percent")
                        bbox_enlargement_factor = gr.Slider(minimum=1, maximum=10, value=1.5, step=0.1,
                                                label="Facteur d'agrandissement des bbox")
                        remove_background_option = gr.Checkbox(
                            label="Supprimer l'arrière-plan",
                            value=False,  # Décoché par défaut
                            info="Supprime l'arrière-plan de l'image après le traitement"
                        )
                        add_watermark_option = gr.Checkbox(
                            label="Ajouter un filigrane",
                            value=False,  # Décoché par défaut
                            info="Ajoute un filigrane sur l'image traitée"
                        )
                        watermark_text = gr.Textbox(label="Texte du watermark", value="www.inflatable-store.com", visible=True)
                        remove_watermark_iterations = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                                label="Nombre d'itérations d'inpainting")
                        remove_url_btn = gr.Button("Supprimer le watermark depuis l'URL")
                    with gr.Column():
                        preview_remove_bg = gr.Image(label="Image sans BG", type="numpy")
                        preview_detection = gr.Image(label="Detection Mask", type="numpy")
                        inpainted_image = gr.Image(label="Image nettoyée", type="numpy")
                        final_result = gr.Image(label="Image nettoyée avec WM", type="numpy")

            remove_url_btn.click(
                remove_wm,
                inputs=[url_input, 
                        threshold_url, 
                        max_bbox_percent,
                        remove_background_option, 
                        add_watermark_option, 
                        watermark_text, 
                        bbox_enlargement_factor,
                        remove_watermark_iterations],
                outputs=[
                    preview_remove_bg,
                    preview_detection,
                    inpainted_image,
                    final_result
                ]
            )
        with gr.Tab("Clean Image Pipeline (Manuel)"):
            with gr.Row():
                with gr.Column():
                    image_count_manual = gr.Slider(minimum=1, maximum=1000, value=2, label="Nombre d'images")           
                    sheet_name_manual = gr.Textbox(label="Onglet Source", value=os.getenv("GSHEET_SHEET_PHOTOS_MAC_TAB"))
                    launch_pipeline_manual = gr.Button("Lancer le traitement manuel")
                
                with gr.Column():
                    output_pipeline_manual = gr.Textbox(label="Statut du traitement", value="En attente de lancement")
            
            # Interface pour l'édition manuelle
            with gr.Row(visible=False) as manual_edit_interface:
                with gr.Column(scale=1):
                    gr.Markdown("## Traitement des watermarks")
                    gr.Markdown("""
                    ### Options de traitement:
                    1. Si la détection automatique est correcte, cliquez sur "Valider sans édition"
                    2. Sinon, dessinez en blanc sur les zones de watermark à supprimer dans l'éditeur
                    3. Choisissez vos options (background, filigrane) puis validez l'édition
                    """)
                    image_index = gr.Number(label="Index de l'image", visible=False)
                    image_source_url = gr.Textbox(label="URL de l'image source", visible=False)
                    remove_bg_option = gr.Checkbox(label="Supprimer le background", value=False)
                    add_watermark_option = gr.Checkbox(label="Ajouter un filigrane", value=True)
                    watermark_text = gr.Textbox(label="Texte du filigrane", value="www.inflatable-store.com")
                    
                    # Boutons d'action
                    with gr.Row():
                        validate_auto = gr.Button("Valider sans édition", variant="primary")
                        validate_edit = gr.Button("Valider édition manuelle", variant="primary")
                    
                    with gr.Row():
                        skip_image = gr.Button("Image suivante", variant="secondary")
                    
                    result_status = gr.Textbox(label="Résultat", value="")

                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.TabItem("Détection automatique"):
                            gr.Markdown("### Résultat de la détection automatique")
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("Image originale:")
                                    original_image_display = gr.Image(label="", type="pil")
                                with gr.Column():
                                    gr.Markdown("Masque détecté:")
                                    mask_display = gr.Image(label="", type="numpy")
                            
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("Inpainting automatique:")
                                    inpainted_image_display = gr.Image(label="", type="pil")
                                with gr.Column():
                                    gr.Markdown("Résultat final détecté:")
                                    final_image_display = gr.Image(label="", type="pil")
                        
                        with gr.TabItem("Édition manuelle"):
                            gr.Markdown("### Éditeur de masque")
                            gr.Markdown("""Pour effectuer une édition manuelle:
                            1. L'image ci-dessous est le résultat du traitement automatique
                            2. Dessinez en blanc sur les zones où des watermarks sont encore visibles
                            3. Cliquez sur "Valider édition manuelle" pour appliquer votre masque""")
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.Markdown("#### Image déjà traitée par l'IA:")
                                    inpainted_preview = gr.Image(label="", type="pil")  # Affichage de référence
                            
                                with gr.Column(scale=2):
                                    gr.Markdown("#### Dessinez votre masque sur cette image:")
                                    image_editor = gr.ImageEditor(
                                        label="Éditeur d'image"
                                    )
                                    
                                    gr.Markdown("""
                                    **Instructions pour créer un masque:**
                                    1. Dessinez en **BLANC** sur les zones du filigrane à supprimer
                                    2. Utilisez l'outil gomme pour corriger les erreurs
                                    3. Les zones blanches seront remplacées par l'inpainting
                                    """)
                                    
                                    with gr.Row():
                                        # Outil pour réinitialiser le masque
                                        reset_editor_btn = gr.Button("Réinitialiser le masque")
                                        # Bouton de validation de l'édition
                                        validate_edit = gr.Button("Valider édition manuelle", variant="primary")
                    
                    # Fonction pour réinitialiser l'éditeur
                    def reset_editor(current_idx, images):
                        if current_idx < 0 or current_idx >= len(images):
                            return None
                        
                        _, _, _, _, inpainted_image, _, _ = images[current_idx]
                        return inpainted_image
                    
                    # Connecter le bouton de réinitialisation
                    reset_editor_btn.click(
                        fn=reset_editor,
                        inputs=[current_image_index, remaining_images],
                        outputs=[image_editor]
                    )

                    # Résultats après traitement
                    with gr.Row(visible=False) as results_row:
                        processed_mask = gr.Image(
                            label="Masque appliqué",
                            type="numpy",
                            visible=True
                        )
                        
                        final_image = gr.Image(
                            label="Image traitée",
                            type="pil",
                            visible=True
                        )
            
            # Variables d'état pour gérer le flux de traitement
            current_image_index = gr.State(value=0)
            remaining_images = gr.State(value=[])
            
            # Connecter les événements
            launch_pipeline_manual.click(
                start_manual_pipeline,
                inputs=[image_count_manual, sheet_name_manual],
                outputs=[
                    output_pipeline_manual,
                    manual_edit_interface,
                    current_image_index,
                    remaining_images,
                    image_index,
                    image_source_url,
                    image_editor,
                    remove_bg_option,
                    add_watermark_option,
                    watermark_text,
                    result_status,
                    original_image_display,
                    mask_display,
                    inpainted_image_display,
                    final_image_display,
                    inpainted_preview,
                    processed_mask,
                    final_image,
                    results_row,
                    manual_edit_interface,
                    current_image_index
                ]
            )
            
            # Valider le traitement automatique sans édition
            validate_auto.click(
                validate_automatic_processing,
                inputs=[
                    current_image_index,
                    remaining_images,
                    remove_bg_option,
                    add_watermark_option,
                    watermark_text
                ],
                outputs=[
                    current_image_index,
                    remaining_images,
                    result_status,
                    processed_mask,
                    final_image,
                    results_row
                ]
            )
            
            # Valider l'édition manuelle
            validate_edit.click(
                process_edited_image,
                inputs=[
                    current_image_index,
                    remaining_images,
                    image_index,
                    image_editor,  # On passe l'éditeur d'image à la fois pour edited_image et edited_mask
                    image_editor,  # On passe l'éditeur d'image comme masque édité
                    remove_bg_option,
                    add_watermark_option,
                    watermark_text
                ],
                outputs=[
                    current_image_index,
                    remaining_images,
                    result_status,
                    processed_mask,
                    final_image,
                    results_row
                ]
            )
            
            # Passer à l'image suivante
            skip_image.click(
                next_image,
                inputs=[current_image_index, remaining_images],
                outputs=[
                    current_image_index,
                    remaining_images,
                    result_status,
                    original_image_display,
                    mask_display,
                    inpainted_image_display,
                    final_image_display,
                    inpainted_preview,  # Ajouter inpainted_preview
                    remove_bg_option,
                    manual_edit_interface,  # Utiliser le composant Gradio directement
                    results_row  # Utiliser le composant Gradio directement
                ]
            )
        with gr.Tab("Inpainting avec masque"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Image d'entrée", type="numpy")
                    input_mask = gr.Image(label="Masque (zones blanches = à traiter)", type="numpy")
                    inpaint_btn = gr.Button("Appliquer l'inpainting")
                
                with gr.Column():
                    output_image = gr.Image(label="Image traitée", type="numpy")
            
            inpaint_btn.click(
                inpaint_with_mask,
                inputs=[input_image, input_mask],
                outputs=[output_image]
            )
        
        # Conserver l'ancienne interface pour la détection automatique
        with gr.Tab("Inpainting automatique"):
            with gr.Row():
                with gr.Column():
                    auto_input_image = gr.Image(label="Image d'entrée", type="numpy")
                    threshold_remove = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.05,
                                          label="Seuil de confiance")
                    max_bbox_percent = gr.Slider(minimum=1, maximum=100, value=10,step=1,
                                          label="Maximal bbox percent")
                    remove_watermark_iterations = gr.Slider(minimum=1, maximum=10, value=1, step=1,
                                          label="Nombre d'itérations d'inpainting")
                    auto_inpaint_btn = gr.Button("Appliquer l'inpainting automatique")
                
                with gr.Column():
                    output_mask = gr.Image(label="Masque de détection", type="numpy")
                    auto_output_image = gr.Image(label="Image traitée", type="numpy")
            
            auto_inpaint_btn.click(
                inpaint_image,
                inputs=[auto_input_image, threshold_remove, max_bbox_percent, remove_watermark_iterations],
                outputs=[output_mask, auto_output_image]
            )

        with gr.Tab("Détection From URL"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(label="URL de l'image")
                    input_prompt = gr.Textbox(label="Prompt")
                    input_max_new_tokens = gr.Slider(minimum=256, maximum=10000, value=1024, step=256,
                                          label="Nombre de tokens maximum")
                    input_early_stopping = gr.Checkbox(label="Early Stopping", value=False)
                    input_do_sample = gr.Checkbox(label="Do Sample", value=True)
                    input_num_beams = gr.Slider(minimum=1, maximum=100, value=5, step=1,
                                          label="Nombre de beams")
                    input_num_return_sequences = gr.Slider(minimum=1, maximum=100, value=1, step=1,
                                          label="Nombre de séquences à retourner")
                    input_temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.75, step=0.05,
                                          label="Température")
                    input_top_k = gr.Slider(minimum=1, maximum=100, value=40, step=1,
                                          label="Top K")
                    input_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.05,
                                          label="Top P")
                    input_repetition_penalty = gr.Slider(minimum=0.0, maximum=10.0, value=1.2, step=0.1,
                                          label="Penalité de répétition")
                    input_length_penalty = gr.Slider(minimum=0.0, maximum=2.0, value=0.8, step=0.05,
                                          label="Penalité de longueur")
                    max_bbox_percent = gr.Slider(minimum=1, maximum=100, value=10, step=1,
                                          label="Maximal bbox percent")
                    bbox_enlargement_factor = gr.Slider(minimum=1, maximum=10, value=1.2, step=0.1,
                                          label="Facteur d'agrandissement des bbox")
                    detect_btn = gr.Button("Détecter")
                
                with gr.Column():
                    output_original_image = gr.Image(label="Detections", type="numpy")
                    output_detection = gr.Image(label="Détections extrapolées", type="numpy")
                    output_mask = gr.Image(label="Masque", type="numpy")
            
            detect_btn.click(
                detect_watermarks_from_url,
                inputs=[url_input, input_prompt, input_max_new_tokens, 
                        input_early_stopping, input_do_sample, input_num_beams, 
                        input_num_return_sequences, input_temperature, input_top_k, 
                        input_top_p, input_repetition_penalty, input_length_penalty, 
                        max_bbox_percent, bbox_enlargement_factor],
                outputs=[output_original_image, output_detection, output_mask]
            )

if __name__ == "__main__":
    initialize_clients()
    interfaces.launch()
