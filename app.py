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
    global gsheet_client, shopify_client, watermark_removal_client
    
    # Initialisation du client GSheet
    credentials_file = os.getenv("GSHEET_CREDENTIALS_FILE")
    spreadsheet_id = os.getenv("GSHEET_ID")
    gsheet_client = GSheetClient(credentials_file, spreadsheet_id)

    # Initialisation du client Shopify
    shopify_api_key = os.getenv("SHOPIFY_API_KEY")
    shopify_password = os.getenv("SHOPIFY_PASSWORD")
    shopify_store_name = os.getenv("SHOPIFY_STORE_NAME")
    shopify_base_url = os.getenv("SHOPIFY_BASE_URL")
    shopify_client = ShopifyClient(shopify_api_key, 
                                   shopify_password, 
                                   shopify_store_name, 
                                   shopify_base_url)

    # Initialisation du client WatermarkRemoval
    hf_token = os.getenv("HF_TOKEN")
    space_url = os.getenv("HF_SPACE_BG_REMOVAL")
    watermak_removal_client = WatermakRemovalClient(hf_token, space_url)

def gsheet_test_connection(input_gheet_id: str, input_gheet_sheet: str):
    return gsheet_client.test_connection(input_gheet_id, input_gheet_sheet)

def shopify_test_connection(input_shopify_domain: str, input_shopify_api_version: str, input_shopify_api_key: str, input_shopify_base_url: str):
    return shopify_client.test_connection(input_shopify_domain, input_shopify_api_version, input_shopify_api_key, input_shopify_base_url)

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
                    input_shopify_base_url = gr.Textbox(label="Shopify Base URL", value=os.getenv("SHOPIFY_BASE_URL"))
                    test_connection = gr.Button("Test Connection")
                
                with gr.Column():
                    output_shopify_test_connection = gr.Textbox(label="Réponse du test", value="Réponse du test")
            
            test_connection.click(
                shopify_test_connection,
                inputs=[input_shopify_domain, 
                        input_shopify_api_version,
                        input_shopify_api_key,
                        input_shopify_base_url],
                outputs=[output_shopify_test_connection]
            )
        #gr.Image(label="Image sans arrière-plan", type="numpy")
        #threshold_detect = gr.Slider(minimum=0.0, maximum=1.0, value=0.85, step=0.05,label="Seuil de confiance")
        #remove_background_option = gr.Checkbox(label="Supprimer l'arrière-plan", value=False)

if __name__ == "__main__":
    initialize_clients()
    interfaces.launch()
