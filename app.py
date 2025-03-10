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
    Fonction wrapper pour appeler la méthode remove_wm du client WatermarkRemovalClient.
    
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
                # Charger l'image de test au lieu d'appeler remove_wm
                test_image_path = r"C:\Users\matrix\Downloads\TestRmWMimage.png"
                if os.path.exists(test_image_path):
                    try:
                        inpainted_image = cv2.imread(test_image_path)
                        # Convertir de BGR à RGB (OpenCV charge en BGR)
                        inpainted_image = cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB)
                        logger.info(f"Image de test chargée avec succès depuis {test_image_path}")
                    except Exception as e:
                        logger.error(f"Erreur lors du chargement de l'image de test: {str(e)}")
                        inpainted_image = None

                # Commenté temporairement l'appel à remove_wm
                """
                bg_removed_path, mask_path, inpainted_path, _ = watermak_removal_client.remove_wm(
                    image_url=lien_image_source,
                    threshold=0.85,
                    max_bbox_percent=10.0,
                    remove_background_option=False,
                    add_watermark_option=False,
                    watermark="",
                    bbox_enlargement_factor=1.5,
                    remove_watermark_iterations=1
                )
                logger.info(f"Sortie de remove_wm: {bg_removed_path}, {mask_path}, {inpainted_path}, None")
                """
                
                # Créer un masque vide
                if inpainted_image is not None:
                    h, w = inpainted_image.shape[:2]
                    mask_image = np.zeros((h, w), dtype=np.uint8)
                    logger.info("Masque vide créé")
                else:
                    logger.error("Impossible de créer le masque car l'image est None")
                    continue
                
                # Vérifier que l'image est valide
                if inpainted_image is not None and hasattr(inpainted_image, 'shape') and len(inpainted_image.shape) >= 2:
                    # Ajouter cette image à la liste des résultats pour l'édition manuelle
                    results.append((index, lien_image_source, inpainted_image, mask_image, supprimer_background))
                else:
                    logger.warning(f"Image invalide pour {lien_image_source}")
                    continue
            
            except Exception as e:
                logger.error(f"Erreur lors du traitement automatique de l'image {lien_image_source}: {str(e)}")
                continue
    
    return results

def apply_manual_edits(index, image, edited_mask, supprimer_background, add_watermark=True, watermark="www.inflatable-store.com"):
    try:
        # Extraction du masque à partir du dictionnaire si nécessaire
        processed_mask = None
        if isinstance(edited_mask, dict):
            logger.info(f"Masque reçu sous forme de dictionnaire: {edited_mask.keys()}")
            
            # Vérifier d'abord si l'image composite est utilisable directement
            if 'composite' in edited_mask and edited_mask['composite'] is not None:
                logger.info("Utilisation de l'image composite comme masque")
                
                # Si c'est une PIL Image, l'utiliser directement
                if isinstance(edited_mask['composite'], Image.Image):
                    processed_mask = edited_mask['composite']
                    logger.info(f"Image composite utilisée directement comme masque PIL: {processed_mask.size}")
                else:
                    # Convertir en PIL Image si c'est un numpy array
                    composite = edited_mask['composite']
                    if isinstance(composite, np.ndarray):
                        processed_mask = Image.fromarray(composite)
                        logger.info(f"Image composite convertie en PIL Image: {processed_mask.size}")
            
            # Si pas de masque valide à partir de composite, essayer les layers
            if processed_mask is None and 'layers' in edited_mask and edited_mask['layers']:
                logger.info(f"Nombre de layers: {len(edited_mask['layers'])}")
                
                # Prendre la dernière couche non vide 
                for i, layer in enumerate(reversed(edited_mask['layers'])):
                    logger.info(f"Analyse du layer {len(edited_mask['layers'])-i} (en partant de la fin)")
                    
                    if 'content' not in layer or layer['content'] is None:
                        logger.info("Layer sans contenu, ignoré")
                        continue
                    
                    # Traiter le contenu du layer
                    try:
                        # Si c'est une PIL Image, l'utiliser directement
                        if isinstance(layer['content'], Image.Image):
                            processed_mask = layer['content']
                            logger.info(f"Layer utilisé directement comme masque PIL: {processed_mask.size}")
                            break
                        else:
                            # Convertir en PIL Image si possible
                            user_layer = layer['content']
                            if isinstance(user_layer, np.ndarray):
                                processed_mask = Image.fromarray(user_layer)
                                logger.info(f"Layer converti en PIL Image: {processed_mask.size}")
                                break
                            else:
                                logger.warning(f"Contenu du layer non utilisable: {type(user_layer)}")
                    except Exception as layer_error:
                        logger.error(f"Erreur lors du traitement du layer: {str(layer_error)}")
                        continue
            
            if processed_mask is None:
                logger.warning("Aucun layer valide trouvé, création d'un masque vide")
                # Créer un masque PIL vide
                if isinstance(image, np.ndarray):
                    h, w = image.shape[:2]
                    processed_mask = Image.new('L', (w, h), 0)  # 'L' = 8-bit pixels, noir et blanc
                elif isinstance(image, Image.Image):
                    w, h = image.size
                    processed_mask = Image.new('L', (w, h), 0)
                else:
                    logger.error(f"Type d'image non pris en charge: {type(image)}")
                    raise ValueError(f"Type d'image non pris en charge: {type(image)}")
        
        # Vérifier que le masque est valide
        if processed_mask is None:
            raise ValueError("Impossible de créer un masque valide")
            
        # Préparation de l'image d'entrée (convertir en PIL.Image si nécessaire)
        if isinstance(image, np.ndarray):
            input_image = Image.fromarray(image)
            logger.info(f"Image numpy convertie en PIL Image: {input_image.size}")
        elif isinstance(image, Image.Image):
            input_image = image
            logger.info(f"Image déjà au format PIL: {input_image.size}")
        else:
            logger.error(f"Type d'image non pris en charge: {type(image)}")
            raise ValueError(f"Type d'image non pris en charge: {type(image)}")
            
        logger.info(f"Type de l'image avant inpainting: {type(input_image)}")
        logger.info(f"Taille de l'image avant inpainting: {input_image.size}")
        logger.info(f"Type du masque avant inpainting: {type(processed_mask)}")
        logger.info(f"Taille du masque avant inpainting: {processed_mask.size}")
        
        # Sauvegarder l'image et le masque pour visualisation
        debug_dir = os.path.join(os.getcwd(), "debug_images")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Générer un timestamp unique pour les fichiers
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sauvegarder l'image d'entrée
        input_image_path = os.path.join(debug_dir, f"input_image_{timestamp}.png")
        input_image.save(input_image_path)
        logger.info(f"Image d'entrée sauvegardée pour debug: {input_image_path}")
        
        # Sauvegarder le masque
        mask_path = os.path.join(debug_dir, f"mask_{timestamp}.png")
        processed_mask.save(mask_path)
        logger.info(f"Masque sauvegardé pour debug: {mask_path}")
        
        # Appel à inpainting
        logger.info("Appel à la méthode inpaint")
        success, inpainted_image = watermak_removal_client.inpaint(
            input_image=input_image,
            mask=processed_mask
        )
        
        if not success or inpainted_image is None:
            logger.warning("L'inpainting a échoué, utilisation de l'image d'origine")
            # Retourner l'image d'origine en cas d'échec
            return input_image_path, mask_path, np.array(input_image)
        
        # Sauvegarder l'image inpainted pour debug
        inpainted_image_path = os.path.join(debug_dir, f"inpainted_{timestamp}.png")
        if isinstance(inpainted_image, np.ndarray):
            Image.fromarray(inpainted_image).save(inpainted_image_path)
        else:
            inpainted_image.save(inpainted_image_path)
        logger.info(f"Image inpainted sauvegardée pour debug: {inpainted_image_path}")
        
        # Retourner les chemins des images sauvegardées et l'image inpainted
        return input_image_path, mask_path, inpainted_image
        
    except Exception as e:
        logger.error(f"Erreur pendant l'application des modifications manuelles: {str(e)}")
        logger.exception(e)  # Afficher la stacktrace complète
        return None, None, None

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
        # Créer une image de test et un masque en PIL
        from PIL import Image
        import numpy as np
        
        # Créer une image de test (un carré noir avec un carré blanc au milieu)
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
        
        # Convertir les images en base64
        buffered_img = BytesIO()
        image.save(buffered_img, format="PNG")
        img_base64 = base64.b64encode(buffered_img.getvalue()).decode('utf-8')
        
        buffered_mask = BytesIO()
        mask.save(buffered_mask, format="PNG")
        mask_base64 = base64.b64encode(buffered_mask.getvalue()).decode('utf-8')
        
        # Obtenir le nom de l'API
        api_name = os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_INPAINT_WITH_MASK")
        logger.info(f"Test de l'API d'inpainting avec api_name={api_name}")
        
        # Appel direct à l'API avec les chaînes base64
        try:
            # Essayer d'abord avec les arguments positionnels
            logger.info("Tentative d'appel à l'API avec les arguments positionnels")
            result = watermak_removal_client.client.predict(
                img_base64,
                mask_base64,
                api_name=api_name
            )
            logger.info(f"Test réussi avec arguments positionnels, type de résultat: {type(result)}")
        except Exception as e:
            logger.error(f"Erreur lors de l'appel à l'API avec les arguments positionnels: {str(e)}")
            # Si ça échoue, essayer avec les arguments nommés
            logger.info("Tentative d'appel à l'API avec les arguments nommés")
            result = watermak_removal_client.client.predict(
                input_image=img_base64,
                input_mask=mask_base64,
                api_name=api_name
            )
            logger.info(f"Test réussi avec arguments nommés, type de résultat: {type(result)}")
        
        # Appel via la méthode inpaint
        logger.info("Test de la méthode inpaint")
        success, inpainted_image = watermak_removal_client.inpaint(image, mask)
        
        if success and inpainted_image is not None:
            logger.info(f"Test de la méthode inpaint réussi, type de résultat: {type(inpainted_image)}")
            if isinstance(inpainted_image, np.ndarray):
                logger.info(f"Forme de l'image inpainted: {inpainted_image.shape}")
            return "Test réussi"
        else:
            return "Test échoué: l'inpainting n'a pas réussi"
    except Exception as e:
        logger.error(f"Erreur lors du test de l'API d'inpainting: {str(e)}")
        logger.exception(e)  # Afficher la stacktrace complète
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
                    image_count_manual = gr.Slider(minimum=1, maximum=1000, value=2, label="Nombre d'images")           
                    sheet_name_manual = gr.Textbox(label="Onglet Source", value=os.getenv("GSHEET_SHEET_PHOTOS_MAC_TAB"))
                    launch_pipeline_manual = gr.Button("Lancer le traitement manuel")
                
                with gr.Column():
                    output_pipeline_manual = gr.Textbox(label="Statut du traitement", value="En attente de lancement")
            
            # Interface pour l'édition manuelle
            with gr.Row(visible=False) as manual_edit_interface:
                with gr.Column():
                    gr.Markdown("## Édition manuelle des watermarks")
                    gr.Markdown("""
                    ### Instructions:
                    1. Un nouveau layer a été créé automatiquement pour vous
                    2. Dessinez en blanc sur les zones de watermark à supprimer
                    3. Seul ce nouveau layer sera utilisé comme masque pour l'inpainting
                    """)
                    image_index = gr.Number(label="Index de l'image", visible=False)
                    image_source_url = gr.Textbox(label="URL de l'image source", visible=False)
                    remove_bg_option = gr.Checkbox(label="Supprimer l'arrière-plan", visible=False)
                    
                    validate_edit = gr.Button("Valider et traiter l'image")
                    skip_image = gr.Button("Ignorer cette image")
                    result_status = gr.Textbox(label="Résultat", value="")
                    
                    # Composants pour afficher le masque et l'image finale
                    processed_mask = gr.Image(
                        label="Masque utilisé pour l'inpainting",
                        type="numpy",
                        visible=True
                    )
                    
                    final_image = gr.Image(
                        label="Image finale",
                        type="numpy",
                        visible=True
                    )
                
                with gr.Column():  # Augmenter l'échelle pour agrandir l'éditeur
                    image_editor = gr.ImageEditor(
                        label="Éditez les zones de watermark (dessinez en blanc les watermarks non détectés)", 
                        type="pil",  # Changé de "numpy" à "pil" pour assurer la compatibilité
                        show_download_button=True,
                        show_share_button=True,
                        brush=gr.Brush(colors=["#FFFFFF"], default_size=40)  # Taille de brosse 40px par défaut
                    )
            
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
                        False,  # Option de suppression de fond
                        "",    # Statut du résultat
                        gr.update(visible=False),  # Masque
                        gr.update(visible=False)   # Image finale
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
                    remove_bg,
                    "",  # Statut du résultat
                    gr.update(visible=False),  # Masque
                    gr.update(visible=False)   # Image finale
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
                        False,  # Option de suppression de fond
                        "",     # Statut du résultat
                        gr.update(visible=False),  # Masque
                        gr.update(visible=False)   # Image finale
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
                    remove_bg,
                    "",  # Statut du résultat
                    gr.update(visible=False),  # Masque
                    gr.update(visible=False)   # Image finale
                )
            
            # Fonction pour traiter l'image après édition manuelle
            def process_edited_image(current_idx, images, index, edited_image, remove_bg_option):
                """
                Traite l'image éditée manuellement et affiche les résultats.
                """
                try:
                    # Récupérer l'image originale et le masque édité
                    _, url, original_image, _, _ = images[current_idx]
                    
                    # Appliquer les modifications manuelles
                    input_image_path, mask_path, inpainted_image = apply_manual_edits(
                        index=index,
                        image=original_image,  # Utiliser l'image originale
                        edited_mask=edited_image,  # Utiliser l'image éditée comme masque
                        supprimer_background=remove_bg_option
                    )
                    
                    # Charger les images de debug pour affichage
                    debug_images = []
                    debug_labels = []
                    
                    if input_image_path and os.path.exists(input_image_path):
                        debug_images.append(input_image_path)
                        debug_labels.append("Image d'entrée")
                        
                    if mask_path and os.path.exists(mask_path):
                        debug_images.append(mask_path)
                        debug_labels.append("Masque")
                    
                    # Retourner les images de debug et leurs labels
                    return debug_images, debug_labels
                
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de l'image éditée: {str(e)}")
                    return [], []
            
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
                    result_status,
                    processed_mask,
                    final_image
                ]
            )
            
            validate_edit.click(
                process_edited_image,
                inputs=[
                    current_image_index,
                    remaining_images,
                    image_index,
                    image_editor,
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
                    remove_bg_option,
                    result_status,
                    processed_mask,
                    final_image,
                    processed_mask,
                    final_image
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
                    remove_bg_option,
                    result_status,
                    processed_mask,
                    final_image
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

    # Ajouter une galerie pour afficher les images de debug
    with gr.Row():
        debug_gallery = gr.Gallery(
            label="Images de debug",
            show_label=True,
            elem_id="debug_gallery",
            columns=2,
            rows=1,
            height="auto"
        )

if __name__ == "__main__":
    initialize_clients()
    interfaces.launch()
