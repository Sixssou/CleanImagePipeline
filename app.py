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

from src.clients.gsheet_client import GSheetClient
from src.clients.shopify_client import ShopifyClient
from src.clients.watermak_removal_client import WatermakRemovalClient

load_dotenv()

# Variables globales pour les clients
gsheet_client = None
shopify_client = None
watermak_removal_client = None
sheet_name = None  # Variable globale pour stocker le nom de l'onglet

def initialize_clients():
    """
    Initialise et retourne les clients nécessaires pour l'application.
    """
    global gsheet_client, shopify_client, watermak_removal_client
    
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

def gsheet_test_connection(input_gheet_id: str, input_gheet_sheet: str):
    return gsheet_client.test_connection(input_gheet_id, input_gheet_sheet)

def shopify_test_connection(input_shopify_domain: str, input_shopify_api_version: str, input_shopify_api_key: str):
    return shopify_client.test_connection(input_shopify_domain, 
                                          input_shopify_api_version, 
                                          input_shopify_api_key, 
)

def remove_background(input_image_url_remove_background: str):
    return watermak_removal_client.remove_bg(input_image_url_remove_background)

def detect_wm(image_url: str, 
              threshold: float = 0.85,  # Ce paramètre n'est plus utilisé mais conservé pour compatibilité
              max_bbox_percent: float = 10.0, 
              bbox_enlargement_factor: float = 1.2):
    """
    Version simplifiée de la détection de watermarks pour la compatibilité avec le code existant.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple contenant (image originale, image avec bounding boxes, masque)
    """
    return watermak_removal_client.detect_wm(
        image_url=image_url,
        prompt="",  # Prompt vide par défaut
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=True,
        num_beams=5,
        num_return_sequences=1,
        temperature=0.75,
        top_k=40,
        top_p=0.85,
        repetition_penalty=1.2,
        length_penalty=0.8,
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
    """
    Fonction wrapper pour appeler la méthode remove_wm du client WatermakRemovalClient.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple contenant (image sans fond, masque de détection, image inpainted, image finale)
    """
    return watermak_removal_client.remove_wm(
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
    """
    Fonction pour lancer le pipeline de nettoyage des images.
    """
    data = gsheet_client.read_cells(sheet_name, f"A1:C{image_count}")
    
    for index, row in enumerate(data, start=1):
        lien_image_source = row[0]
        lien_image_traitee = row[1]
        supprimer_background = row[2].upper() == 'TRUE' if len(row) > 2 else False
        logger.info(f"Compteur : {index}")
        logger.info(f"Lien image source : {lien_image_source}")
        logger.info(f"Lien image traitee : {lien_image_traitee}")
        logger.info(f"Supprimer background : {supprimer_background}")
        if not lien_image_traitee:
            bg_removed_image, mask_image, inpainted_image, result_image = watermak_removal_client.remove_wm(
                image_url=lien_image_source, 
                threshold=0.85, 
                max_bbox_percent=10.0, 
                remove_background_option=supprimer_background, 
                add_watermark_option=True, 
                watermark="www.inflatable-store.com", 
                bbox_enlargement_factor=1.5, 
                remove_watermark_iterations=1
            )
            logger.info(f"Image nettoyée : {result_image}")
            lien_image_traitee = shopify_client.upload_file_to_shopify(result_image)
            logger.info(f"Image nettoyée et uploadée : {lien_image_traitee}")
            gsheet_client.write_cells(sheet_name, f"B{index}", [[lien_image_traitee]])
    return data

def download_image(url):
    """Télécharge une image depuis une URL et la convertit en numpy array."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def manual_watermark_edit(image, detection_mask):

    return image, detection_mask

def process_single_image(lien_image_source, supprimer_background=False):
    """
    Traite une seule image avec l'étape manuelle d'édition.
    Retourne le lien de l'image traitée uploadée sur Shopify.
    """
    try:
        # Étape 1: Détection automatique des watermarks
        original_image, image_with_bbox, detection_mask = watermak_removal_client.detect_wm(
            image_url=lien_image_source, 
            prompt="",  # Prompt vide par défaut
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=True,
            num_beams=5,
            num_return_sequences=1,
            temperature=0.75,
            top_k=40,
            top_p=0.85,
            repetition_penalty=1.2,
            length_penalty=0.8,
            max_bbox_percent=10.0,
            bbox_enlargement_factor=1.5
        )
        
        # Télécharger l'image source pour l'édition manuelle
        image_source = download_image(lien_image_source)
        
        return image_source, detection_mask
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image {lien_image_source}: {str(e)}")
        return None, None

def clean_image_pipeline_manual(image_count: int, sheet_name_value: str):
    """
    Version modifiée du pipeline de nettoyage des images avec étape manuelle.
    Étape 1: Détection et suppression automatique des watermarks avec remove_wm
    Étape 2: Affichage de l'image résultante pour édition manuelle des zones non détectées
    """
    global sheet_name
    sheet_name = sheet_name_value  # Stocker le nom de l'onglet dans la variable globale
    
    data = gsheet_client.read_cells(sheet_name, f"A1:C{image_count}")
    data = data[1:]
    results = []
    
    for index, row in enumerate(data, start=1):
        if len(row) < 1:
            continue
            
        lien_image_source = row[0]
        lien_image_traitee = row[1] if len(row) > 1 else ""
        supprimer_background = row[2].upper() == 'TRUE' if len(row) > 2 else False
        
        logger.info(f"Compteur : {index}")
        logger.info(f"Lien image source : {lien_image_source}")
        logger.info(f"Lien image traitee : {lien_image_traitee}")
        logger.info(f"Supprimer background : {supprimer_background}")
        
        if not lien_image_traitee and lien_image_source:
            try:
                # Étape 1: Détection et suppression automatique des watermarks avec remove_wm
                bg_removed_path, mask_path, inpainted_path, _ = watermak_removal_client.remove_wm(
                    image_url=lien_image_source,
                    threshold=0.85,
                    max_bbox_percent=10.0,
                    remove_background_option=False,  # On ne supprime pas le fond à cette étape
                    add_watermark_option=False,      # On n'ajoute pas de watermark à cette étape
                    watermark="",
                    bbox_enlargement_factor=1.5,
                    remove_watermark_iterations=1
                )
                logger.info(f"Sortie de remove_wm: {bg_removed_path}, {mask_path}, {inpainted_path}, None")
                
                # Charger l'image inpainted depuis le fichier temporaire
                inpainted_image = None
                if inpainted_path and os.path.exists(inpainted_path):
                    try:
                        inpainted_image = cv2.imread(inpainted_path)
                        # Convertir de BGR à RGB (OpenCV charge en BGR)
                        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
                        logger.info(f"Image inpainted chargée avec succès depuis {inpainted_path}")
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement de l'image inpainted: {str(e)}")
                
                # Charger le masque depuis le fichier temporaire
                mask_image = None
                if mask_path and os.path.exists(mask_path):
                    try:
                        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris
                        logger.info(f"Masque chargé avec succès depuis {mask_path}")
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement du masque: {str(e)}")
                
                # Si l'image inpainted n'est pas disponible, télécharger l'image source
                if inpainted_image is None:
                    logger.warning("Image inpainted non disponible, utilisation de l'image source")
                    inpainted_image = download_image(lien_image_source)
                
                # Si le masque n'est pas disponible, créer un masque vide
                if mask_image is None:
                    logger.warning("Masque non disponible, création d'un masque vide")
                    # Créer un masque vide de la même taille que l'image
                    h, w = inpainted_image.shape[:2]
                    mask_image = np.zeros((h, w), dtype=np.uint8)
                
                # Vérifier que l'image est valide
                if inpainted_image is not None and hasattr(inpainted_image, 'shape') and len(inpainted_image.shape) >= 2:
                    # Ajouter cette image à la liste des résultats pour l'édition manuelle
                    results.append((index, lien_image_source, inpainted_image, mask_image, supprimer_background))
                else:
                    logger.warning(f"Image inpainted invalide pour {lien_image_source}")
                    # Utiliser l'image source comme solution de repli
                    image_source = download_image(lien_image_source)
                    if image_source is not None:
                        h, w = image_source.shape[:2]
                        empty_mask = np.zeros((h, w), dtype=np.uint8)
                        results.append((index, lien_image_source, image_source, empty_mask, supprimer_background))
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement automatique de l'image {lien_image_source}: {str(e)}")
                # En cas d'erreur, utiliser l'image source
                try:
                    # Télécharger l'image source pour l'édition manuelle
                    image_source = download_image(lien_image_source)
                    
                    # Vérifier que l'image source est valide
                    if image_source is not None and hasattr(image_source, 'shape') and len(image_source.shape) >= 2:
                        # Créer un masque vide de la même taille que l'image
                        h, w = image_source.shape[:2]
                        empty_mask = np.zeros((h, w), dtype=np.uint8)
                        
                        # Ajouter cette image à la liste des résultats pour l'édition manuelle
                        results.append((index, lien_image_source, image_source, empty_mask, supprimer_background))
                    else:
                        logger.error(f"Image source invalide pour {lien_image_source}")
                except Exception as e:
                    logger.error(f"Erreur lors du téléchargement de l'image {lien_image_source}: {str(e)}")
    
    return results

def apply_manual_edits(index, image, edited_mask, supprimer_background, add_watermark=True, watermark="www.inflatable-store.com"):
    """
    Applique les modifications manuelles et finalise le traitement de l'image.
    """
    global sheet_name
    
    try:
        # Vérifier si edited_mask est un dictionnaire (cas où l'utilisateur a créé un nouveau layer)
        if isinstance(edited_mask, dict):
            logger.info(f"Masque reçu sous forme de dictionnaire: {edited_mask.keys()}")
            
            # Si le dictionnaire contient une clé 'composite', utiliser cette image
            if 'composite' in edited_mask:
                edited_mask = edited_mask['composite']
                logger.info(f"Utilisation de l'image composite du masque")
            # Sinon, essayer d'utiliser la dernière couche
            elif 'layers' in edited_mask and edited_mask['layers']:
                # Prendre la dernière couche (celle que l'utilisateur vient de dessiner)
                edited_mask = edited_mask['layers'][-1]['content']
                logger.info(f"Utilisation de la dernière couche du masque")
            else:
                # Si aucune option ne fonctionne, créer un masque vide
                logger.warning("Impossible d'extraire le masque du dictionnaire, création d'un masque vide")
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                    edited_mask = np.zeros((h, w), dtype=np.uint8)
                elif isinstance(image, dict) and 'composite' in image:
                    h, w = image['composite'].shape[:2]
                    edited_mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    raise ValueError("Format d'image invalide")
        
        # Vérifier si image est un dictionnaire (cas où l'image a aussi été modifiée)
        if isinstance(image, dict):
            logger.info(f"Image reçue sous forme de dictionnaire: {image.keys()}")
            
            # Si le dictionnaire contient une clé 'composite', utiliser cette image
            if 'composite' in image:
                image = image['composite']
                logger.info(f"Utilisation de l'image composite")
            # Sinon, essayer d'utiliser la première couche (l'image de base)
            elif 'layers' in image and image['layers']:
                image = image['layers'][0]['content']
                logger.info(f"Utilisation de la première couche de l'image")
            else:
                raise ValueError("Format d'image invalide")
        
        # Appliquer l'inpainting via l'API Gradio
        # Si l'appel échoue, une exception sera levée et le traitement s'arrêtera
        logger.info("Appel à la méthode inpaint de WatermakRemovalClient")
        _, inpainted_image = watermak_removal_client.inpaint(
            input_image=image,
            mask=edited_mask  # Utiliser le masque édité manuellement
        )
        
        # Vérifier que l'image inpainted est valide
        if inpainted_image is None:
            logger.error("L'inpainting a retourné None, arrêt du traitement")
            raise ValueError("L'inpainting a échoué")
        
        logger.info(f"Inpainting réussi, forme de l'image: {inpainted_image.shape}")
        
        # Étape 2: Suppression du fond si demandé
        if supprimer_background:
            # Utiliser le client de suppression de fond
            logger.info("Appel à la méthode remove_bg de WatermakRemovalClient")
            final_image = watermak_removal_client.remove_bg(inpainted_image)
            
            # Vérifier que l'image sans fond est valide
            if final_image is None:
                logger.error("La suppression de fond a retourné None, arrêt du traitement")
                raise ValueError("La suppression de fond a échoué")
            
            logger.info(f"Suppression de fond réussie, forme de l'image: {final_image.shape}")
        else:
            final_image = inpainted_image
            
        # Étape 3: Ajout d'un watermark si demandé
        if add_watermark and watermark:
            # Ajouter un watermark à l'image
            # Pour l'instant, on utilise l'image sans watermark
            # Cette fonctionnalité pourrait être implémentée ultérieurement
            pass
        
        # Étape 4: Sauvegarder l'image dans un fichier temporaire avant de l'uploader
        import tempfile
        import os
        from PIL import Image
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_path = temp_file.name
        
        # Sauvegarder l'image dans le fichier temporaire
        Image.fromarray(final_image).save(temp_path)
        logger.info(f"Image finale sauvegardée à {temp_path}")
        
        # Upload sur Shopify
        lien_image_traitee = shopify_client.upload_file_to_shopify(temp_path)
        
        # Nettoyer le fichier temporaire
        try:
            os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Erreur lors de la suppression du fichier temporaire: {str(e)}")
        
        # Étape 5: Mise à jour du GSheet
        gsheet_client.write_cells(sheet_name, f"B{index}", [[lien_image_traitee]])
        
        return lien_image_traitee
    except Exception as e:
        logger.error(f"Erreur lors du traitement manuel: {str(e)}")
        # Relancer l'exception pour interrompre le traitement
        raise e

def inpaint_image(input_image, threshold, max_bbox_percent, remove_watermark_iterations):
    """
    Applique l'inpainting avec détection automatique des watermarks.
    """
    # Utiliser la nouvelle méthode detect_and_inpaint au lieu de inpaint
    return watermak_removal_client.detect_and_inpaint(
        input_image=input_image, 
        threshold=threshold, 
        max_bbox_percent=max_bbox_percent, 
        remove_watermark_iterations=remove_watermark_iterations
    )

def inpaint_with_mask(input_image, input_mask):
    """
    Applique l'inpainting avec un masque fourni par l'utilisateur.
    """
    try:
        # Appel à la méthode inpaint avec le masque fourni
        _, result = watermak_removal_client.inpaint(
            input_image=input_image,
            mask=input_mask
        )
        return result
    except Exception as e:
        logger.error(f"Erreur lors de l'inpainting avec masque: {str(e)}")
        return input_image  # Retourner l'image d'origine en cas d'erreur

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
    """
    Détecte les watermarks dans une image à partir d'une URL avec les paramètres avancés.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple contenant (image originale, image avec bounding boxes, masque)
    """
    try:
        original_image, image_with_bbox, mask = watermak_removal_client.detect_wm(
            image_url=url_input,
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
            length_penalty=input_length_penalty,
            max_bbox_percent=max_bbox_percent,
            bbox_enlargement_factor=bbox_enlargement_factor
        )
        return original_image, image_with_bbox, mask
    except Exception as e:
        logger.error(f"Erreur lors de la détection des watermarks: {str(e)}")
        # Retourner des images vides en cas d'erreur
        return None, None, None

def test_inpaint_api():
    """
    Fonction de test pour vérifier l'API d'inpainting.
    """
    try:
        # Créer une image de test et un masque
        import numpy as np
        import cv2
        
        # Créer une image de test (un carré noir avec un carré blanc au milieu)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[25:75, 25:75] = 255
        
        # Créer un masque de test (un petit carré blanc au milieu)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255
        
        # Appeler l'API d'inpainting
        logger.info("Test de l'API d'inpainting")
        _, result = watermak_removal_client.inpaint(image, mask)
        
        logger.info(f"Test réussi, type de résultat: {type(result)}")
        return "Test réussi"
    except Exception as e:
        logger.error(f"Erreur lors du test de l'API d'inpainting: {str(e)}")
        return f"Erreur: {str(e)}"

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
                        remove_background_option = gr.Checkbox(label="Supprimer l'arrière-plan", value=False)
                        add_watermark_option = gr.Checkbox(label="Ajoute un watermark", value=False)
                        watermark = gr.Textbox(label="Watermark à ajouter", value="www.mybrand.com")
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
                        watermark, 
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
                    image_count_manual = gr.Slider(minimum=1, maximum=1000, value=1, label="Nombre d'images")           
                    sheet_name_manual = gr.Textbox(label="Onglet Source", value=os.getenv("GSHEET_SHEET_PHOTOS_MAC_TAB"))
                    launch_pipeline_manual = gr.Button("Lancer le traitement manuel")
                
                with gr.Column():
                    output_pipeline_manual = gr.Textbox(label="Statut du traitement", value="En attente de lancement")
            
            # Interface pour l'édition manuelle
            with gr.Row(visible=False) as manual_edit_interface:
                with gr.Column():
                    gr.Markdown("## Édition manuelle des watermarks")
                    gr.Markdown("Utilisez l'outil de dessin pour marquer les zones de watermark non détectées (dessinez en blanc)")
                    image_index = gr.Number(label="Index de l'image", visible=False)
                    image_source_url = gr.Textbox(label="URL de l'image source", visible=False)
                    remove_bg_option = gr.Checkbox(label="Supprimer l'arrière-plan", visible=False)
                
                with gr.Column(scale=2):  # Augmenter l'échelle pour agrandir l'éditeur
                    image_editor = gr.ImageEditor(
                        label="Éditez les zones de watermark (dessinez en blanc les watermarks non détectés)", 
                        type="numpy",
                        height=600,  # Augmenter la hauteur
                        width=800    # Augmenter la largeur
                    )
                    mask_editor = gr.ImageEditor(
                        label="Masque de détection (zones blanches = à traiter)", 
                        type="numpy",
                        height=600,  # Augmenter la hauteur
                        width=800    # Augmenter la largeur
                    )
                
                with gr.Column():
                    validate_edit = gr.Button("Valider et traiter l'image")
                    skip_image = gr.Button("Ignorer cette image")
                    result_status = gr.Textbox(label="Résultat", value="")
            
            # Variables d'état pour gérer le flux de traitement
            current_image_index = gr.State(0)
            remaining_images = gr.State([])
            
            # Fonction pour démarrer le pipeline manuel
            def start_manual_pipeline(image_count, sheet_name):
                results = clean_image_pipeline_manual(image_count, sheet_name)
                if not results:
                    # Retourner des valeurs par défaut pour tous les composants de sortie
                    return (
                        "Aucune image à traiter", 
                        gr.update(visible=False), 
                        0,  # index dans la liste
                        [],  # toutes les images à traiter
                        0,   # index dans le GSheet
                        "",  # URL de l'image source
                        None,  # Image à éditer
                        None,  # Masque à éditer
                        False  # Option de suppression de fond
                    )
                
                # Préparer la première image
                index, url, image, mask, remove_bg = results[0]
                
                return (
                    f"Traitement de l'image {1}/{len(results)}",
                    gr.update(visible=True),
                    0,  # index dans la liste
                    results,  # toutes les images à traiter
                    index,  # index dans le GSheet
                    url,
                    image,
                    mask,
                    remove_bg
                )
            
            # Fonction pour passer à l'image suivante
            def next_image(current_idx, images):
                if not images or current_idx >= len(images) - 1:
                    return (
                        "Toutes les images ont été traitées", 
                        gr.update(visible=False), 
                        current_idx, 
                        images, 
                        0,   # index dans le GSheet
                        "",  # URL de l'image source
                        None,  # Image à éditer
                        None,  # Masque à éditer
                        False  # Option de suppression de fond
                    )
                
                next_idx = current_idx + 1
                index, url, image, mask, remove_bg = images[next_idx]
                
                return (
                    f"Traitement de l'image {next_idx + 1}/{len(images)}",
                    gr.update(visible=True),
                    next_idx,
                    images,
                    index,
                    url,
                    image,
                    mask,
                    remove_bg
                )
            
            # Fonction pour traiter l'image après édition manuelle
            def process_edited_image(current_idx, images, gsheet_index, edited_image, edited_mask, remove_bg):
                if not images or current_idx >= len(images):
                    return (
                        "Erreur: aucune image à traiter", 
                        gr.update(visible=False), 
                        current_idx, 
                        images, 
                        0,   # index dans le GSheet
                        "",  # URL de l'image source
                        None,  # Image à éditer
                        None,  # Masque à éditer
                        False,  # Option de suppression de fond
                        "Erreur: aucune image à traiter"  # Statut du résultat
                    )
                
                # Appliquer les modifications et finaliser le traitement
                result_url = apply_manual_edits(gsheet_index, edited_image, edited_mask, remove_bg)
                
                # Vérifier s'il reste des images à traiter
                if current_idx >= len(images) - 1:
                    return (
                        "Toutes les images ont été traitées", 
                        gr.update(visible=False), 
                        current_idx, 
                        images, 
                        0,   # index dans le GSheet
                        "",  # URL de l'image source
                        None,  # Image à éditer
                        None,  # Masque à éditer
                        False,  # Option de suppression de fond
                        f"Image traitée et uploadée: {result_url}"  # Statut du résultat
                    )
                
                # Passer à l'image suivante
                next_idx = current_idx + 1
                next_index, next_url, next_image, next_mask, next_remove_bg = images[next_idx]
                
                return (
                    f"Traitement de l'image {next_idx + 1}/{len(images)}",
                    gr.update(visible=True),
                    next_idx,
                    images,
                    next_index,
                    next_url,
                    next_image,
                    next_mask,
                    next_remove_bg,
                    f"Image traitée et uploadée: {result_url}"
                )
            
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
                    mask_editor,
                    remove_bg_option
                ]
            )
            
            validate_edit.click(
                process_edited_image,
                inputs=[
                    current_image_index,
                    remaining_images,
                    image_index,
                    image_editor,
                    mask_editor,
                    remove_bg_option
                ],
                outputs=[
                    output_pipeline_manual,
                    manual_edit_interface,
                    current_image_index,
                    remaining_images,
                    image_index,
                    image_source_url,
                    image_editor,
                    mask_editor,
                    remove_bg_option,
                    result_status
                ]
            )
            
            skip_image.click(
                next_image,
                inputs=[current_image_index, remaining_images],
                outputs=[
                    output_pipeline_manual,
                    manual_edit_interface,
                    current_image_index,
                    remaining_images,
                    image_index,
                    image_source_url,
                    image_editor,
                    mask_editor,
                    remove_bg_option
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
