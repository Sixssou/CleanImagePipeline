import gradio as gr
from loguru import logger
import os
from dotenv import load_dotenv

from src.clients.gsheet_client import GSheetClient
from src.clients.shopify_client import ShopifyClient
from src.clients.watermak_removal_client import WatermakRemovalClient  # Assumed import

load_dotenv()

# Variables globales pour les clients
gsheet_client = None
shopify_client = None
watermak_removal_client = None

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
    """hf_token = os.getenv("HF_TOKEN")
    space_url = os.getenv("HF_SPACE_WATERMAK_REMOVAL")
    watermak_removal_client = WatermakRemovalClient(hf_token, space_url)"""

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
              threshold: float, 
              max_bbox_percent: float, 
              bbox_enlargement_factor: float):
    return watermak_removal_client.detect_wm(image_url, 
                                             threshold, 
                                             max_bbox_percent, 
                                             bbox_enlargement_factor)

def remove_wm(  image_url, 
                threshold=0.85, 
                max_bbox_percent=10.0, 
                remove_background_option=False, 
                add_watermark_option=False, 
                watermark=None,
                bbox_enlargement_factor=1.5,
                remove_watermark_iterations=1):
    return watermak_removal_client.remove_wm(image_url, 
                                             threshold, 
                                             max_bbox_percent, 
                                             remove_background_option, 
                                             add_watermark_option, 
                                             watermark, 
                                             bbox_enlargement_factor, 
                                             remove_watermark_iterations)

def clean_image_pipeline(image_count: int, sheet_name: str):
    """
    Fonction pour lancer le pipeline de nettoyage des images.
    """
    data = gsheet_client.read_cells(sheet_name, f"A1:C{image_count}")
    
    for row in data:
        lien_image_source = row[0]
        lien_image_traitee = row[1]
        supprimer_background = row[2].upper() == 'TRUE'
        logger.info(f"Lien image source : {lien_image_source}")
        logger.info(f"Lien image traitee : {lien_image_traitee}")
        logger.info(f"Supprimer background : {supprimer_background}")
        if not lien_image_traitee:
            preview_remove_bg, preview_detection, inpainted_image, final_result = watermak_removal_client.remove_wm(lien_image_source, 
                                          threshold=0.85, 
                                          max_bbox_percent=10.0, 
                                          remove_background_option=supprimer_background, 
                                          add_watermark_option=True, 
                                          watermark="www.inflatable-store.com", 
                                          bbox_enlargement_factor=1.5, 
                                          remove_watermark_iterations=1)
            logger.info(f"Image nettoyée : {final_result}")
            lien_image_traitee = shopify_client.upload_file_to_shopify(final_result)
            logger.info(f"Image nettoyée et uploadée : {lien_image_traitee}")
    return data

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
        with gr.Tab("Clean Image Pipeline"):
            with gr.Row():
                with gr.Column():
                    image_count = gr.Slider(minimum=1, maximum=1000, value=1, label="Nombre d'images")           
                    sheet_name = gr.Textbox(label="Onglet Source", value=os.getenv("GSHEET_SHEET_PHOTOS_MAC_TAB"))
                    launch_pipeline = gr.Button("Lancer le traitement")
                
                with gr.Column():
                    output_pipeline = gr.Textbox(label="Réponse du test", value="Réponse du test")
            
            launch_pipeline.click(
                clean_image_pipeline,
                inputs=[image_count, 
                        sheet_name],
                outputs=[output_pipeline]
            )

if __name__ == "__main__":
    initialize_clients()
    interfaces.launch()
