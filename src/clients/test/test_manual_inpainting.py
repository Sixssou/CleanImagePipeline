#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour l'édition manuelle et l'inpainting.
Ce script crée une interface Gradio permettant de tester interactivement le processus d'édition
et d'inpainting d'images.
"""

import os
import sys
import logging
from PIL import Image
import numpy as np
import gradio as gr
from dotenv import load_dotenv
import argparse
from pathlib import Path

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.clients.watermak_removal_client import WatermakRemovalClient
from src.pipeline.image_processor import ImageProcessor
from src.utils.image_utils import TEMP_DIR, download_image, convert_to_pil_image

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Créer le répertoire temporaire si nécessaire
os.makedirs(TEMP_DIR, exist_ok=True)

# Chemin vers l'image de test
TEST_IMAGE_PATH = "debug_images/TestRmWMimage.png"
# Nom de l'image pour l'affichage
TEST_IMAGE_NAME = "TestRmWMimage.png"

# Charger les variables d'environnement
load_dotenv()

def process_manual_edit(original_image, edited_mask, remove_bg=False):
    """
    Traite une image avec un masque édité manuellement.
    
    Args:
        original_image: Image originale (PIL.Image ou np.ndarray)
        edited_mask: Masque édité (PIL.Image ou np.ndarray)
        remove_bg: Si True, supprime l'arrière-plan après l'inpainting
        
    Returns:
        PIL.Image: Image résultante après inpainting
    """
    try:
        # Initialiser les clients nécessaires
        watermak_removal_client = WatermakRemovalClient(
            hf_token=os.getenv('HF_TOKEN'),
            space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
        )
        
        # Initialiser l'ImageProcessor avec le client
        image_processor = ImageProcessor(
            watermak_removal_client=watermak_removal_client
        )
        
        # Convertir en PIL.Image si nécessaire
        if not isinstance(original_image, Image.Image):
            original_image = convert_to_pil_image(original_image)
        
        if not isinstance(edited_mask, Image.Image):
            edited_mask = convert_to_pil_image(edited_mask)
        
        # Redimensionner le masque si nécessaire pour correspondre à l'image originale
        if original_image.size != edited_mask.size:
            edited_mask = edited_mask.resize(original_image.size, Image.LANCZOS)
        
        # Enregistrer le masque pour débogage
        mask_debug_path = os.path.join(TEMP_DIR, "debug_edited_mask.png")
        edited_mask.save(mask_debug_path)
        logger.info(f"Masque d'édition enregistré pour débogage: {mask_debug_path}")
        
        # Appliquer les éditions manuelles
        logger.info("Application des éditions manuelles avec inpainting...")
        input_path, mask_path, result_path = image_processor.apply_manual_edits(
            image=original_image,
            edited_mask=edited_mask,
            supprimer_background=remove_bg,
            add_watermark=False,
            watermark_text=None
        )
        
        # Charger l'image résultante
        if result_path and os.path.exists(result_path):
            result_image = Image.open(result_path)
            logger.info(f"Inpainting réussi. Résultat: {result_path}")
            return result_image
        else:
            logger.error("Échec de l'inpainting: pas de chemin de résultat")
            return original_image
            
    except Exception as e:
        logger.error(f"Erreur lors du traitement de l'image: {e}")
        return original_image

def test_direct_inpainting(original_image, mask_image):
    """
    Teste directement la fonction d'inpainting du WatermakRemovalClient.
    
    Args:
        original_image: Image originale (PIL.Image ou np.ndarray)
        mask_image: Masque (PIL.Image ou np.ndarray)
        
    Returns:
        PIL.Image: Image résultante après inpainting
    """
    try:
        # Initialiser le client
        watermak_removal_client = WatermakRemovalClient(
            hf_token=os.getenv('HF_TOKEN'),
            space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
        )
        
        # Convertir en PIL.Image si nécessaire
        if not isinstance(original_image, Image.Image):
            original_image = convert_to_pil_image(original_image)
        
        if not isinstance(mask_image, Image.Image):
            mask_image = convert_to_pil_image(mask_image)
        
        # Redimensionner le masque si nécessaire
        if original_image.size != mask_image.size:
            mask_image = mask_image.resize(original_image.size, Image.LANCZOS)
        
        # Enregistrer le masque pour débogage
        mask_debug_path = os.path.join(TEMP_DIR, "debug_inpaint_mask.png")
        mask_image.save(mask_debug_path)
        logger.info(f"Masque d'inpainting enregistré pour débogage: {mask_debug_path}")
        
        # Appeler directement la fonction inpaint
        logger.info("Appel direct de la fonction inpaint...")
        result = watermak_removal_client.inpaint(original_image, mask_image)
        
        # Convertir le résultat en PIL.Image si nécessaire
        if isinstance(result, np.ndarray):
            result = Image.fromarray(result)
        
        # Enregistrer le résultat pour débogage
        result_debug_path = os.path.join(TEMP_DIR, "debug_inpaint_result.png")
        if result is not None:
            result.save(result_debug_path)
            logger.info(f"Résultat d'inpainting enregistré pour débogage: {result_debug_path}")
        
        logger.info("Inpainting direct réussi")
        return result
        
    except Exception as e:
        logger.error(f"Erreur lors de l'inpainting direct: {e}")
        return original_image

def load_test_image():
    """Charge l'image de test par défaut."""
    if os.path.exists(TEST_IMAGE_PATH):
        try:
            img = Image.open(TEST_IMAGE_PATH)
            logger.info(f"Image de test chargée: {TEST_IMAGE_PATH}")
            return img
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image de test: {e}")
            return None
    else:
        logger.error(f"Image de test non trouvée: {TEST_IMAGE_PATH}")
        # Vérifier si le dossier debug_images existe
        debug_dir = os.path.dirname(TEST_IMAGE_PATH)
        if not os.path.exists(debug_dir):
            logger.error(f"Le dossier {debug_dir} n'existe pas. Création du dossier.")
            try:
                os.makedirs(debug_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Impossible de créer le dossier {debug_dir}: {e}")
        
        logger.error(f"Assurez-vous que le fichier {TEST_IMAGE_NAME} est bien présent dans le dossier {debug_dir}")
        return None

def create_interface():
    """Crée l'interface Gradio pour tester l'édition et l'inpainting."""
    
    # Introduction de l'application
    with gr.Blocks(title="Test d'édition et d'inpainting") as demo:
        gr.Markdown("# Test d'édition et d'inpainting d'images")
        gr.Markdown(f"Cette interface permet d'éditer directement l'image: **{TEST_IMAGE_NAME}**")
        
        # Charger l'image de test
        test_image = load_test_image()
        
        if test_image is None:
            gr.Markdown(f"## ERREUR: Impossible de charger l'image de test {TEST_IMAGE_PATH}")
            return demo
            
        try:
            # Section principale: édition directe sur l'image
            with gr.Tab("Édition directe et inpainting"):
                gr.Markdown("## Créez le masque directement sur l'image")
                gr.Markdown("""
                ### Instructions:
                1. Utilisez les outils de dessin pour marquer les zones du filigrane que vous voulez supprimer
                2. Dessinez avec une couleur claire (blanc de préférence) sur les zones à effacer
                3. Cliquez sur 'Extraire le masque et appliquer l'inpainting'
                
                **Note**: L'outil va extraire un masque où les zones modifiées seront considérées comme à remplacer
                """)
                
                with gr.Row():
                    # Colonne gauche - Éditeur d'image
                    with gr.Column(scale=2):
                        # Utiliser ImageEditor au lieu de Image
                        image_editor = gr.ImageEditor(
                            label=f"Éditez l'image {TEST_IMAGE_NAME} - Marquez les zones à supprimer",
                            value=test_image,
                            type="pil",
                            brush=gr.Brush(colors=["#FFFFFF"], default_size=10, default_color="#FFFFFF")
                        )
                        
                        with gr.Row():
                            # Option pour supprimer l'arrière-plan
                            remove_bg_checkbox = gr.Checkbox(label="Supprimer l'arrière-plan après inpainting", value=False)
                            
                            # Bouton pour réinitialiser l'éditeur
                            reset_editor_btn = gr.Button("Réinitialiser l'image")
                        
                        # Bouton pour appliquer l'inpainting
                        apply_inpaint_btn = gr.Button("Extraire le masque et appliquer l'inpainting", variant="primary")
                    
                    # Colonne droite - Résultat et informations
                    with gr.Column(scale=2):
                        # Résultat de l'inpainting
                        result_image = gr.Image(label="Résultat de l'inpainting", type="pil")
                        
                        # Image du masque extrait (pour visualisation)
                        extracted_mask_image = gr.Image(label="Masque extrait", type="pil")
                        
                        # Informations sur le processus
                        process_info = gr.Textbox(label="Informations sur le processus", lines=4)
                
                # Fonction pour réinitialiser l'éditeur
                def reset_editor():
                    return test_image
                
                # Reset editor when the button is clicked
                reset_editor_btn.click(
                    fn=reset_editor,
                    inputs=[],
                    outputs=[image_editor]
                )
                
                # Fonction pour extraire le masque et appliquer l'inpainting
                def process_with_editor_data(edited_image_data, remove_bg):
                    info_text = f"Traitement de l'image {TEST_IMAGE_NAME}\n"
                    try:
                        # Vérifier que les données d'édition sont valides
                        if edited_image_data is None:
                            raise ValueError("Aucune donnée d'édition fournie")
                        
                        # Afficher pour débogage la structure des données reçues
                        logger.info(f"Type de données reçues: {type(edited_image_data)}")
                        if isinstance(edited_image_data, dict):
                            logger.info(f"Clés disponibles: {edited_image_data.keys()}")
                        
                        # Vérifier que les calques existent
                        if not isinstance(edited_image_data, dict) or "layers" not in edited_image_data or not edited_image_data["layers"]:
                            raise ValueError("Aucun calque trouvé dans les données d'édition")
                        
                        # Extraire le premier calque uniquement
                        logger.info(f"Nombre de calques: {len(edited_image_data['layers'])}")
                        first_layer = edited_image_data["layers"][0]
                        logger.info(f"Type du premier calque: {type(first_layer)}")
                        
                        # Convertir directement le premier calque en masque
                        mask = convert_to_pil_image(first_layer)
                        
                        # Convertir en niveau de gris si nécessaire
                        if mask.mode != 'L':
                            mask = mask.convert('L')
                        
                        # Enregistrer le masque pour débogage
                        mask_debug_path = os.path.join(TEMP_DIR, "editor_mask.png")
                        mask.save(mask_debug_path)
                        logger.info(f"Masque extrait du premier calque enregistré: {mask_debug_path}")
                        
                        # Vérifier que le masque contient des pixels blancs
                        mask_array = np.array(mask)
                        white_pixels = np.sum(mask_array > 128)
                        total_pixels = mask_array.size
                        white_percentage = (white_pixels / total_pixels) * 100
                        
                        if white_pixels == 0:
                            raise ValueError("Aucune modification détectée. Veuillez marquer les zones à modifier.")
                        
                        info_text += f"Masque créé: {white_pixels} pixels modifiés détectés ({white_percentage:.2f}% de l'image)\n"
                        
                        # IMPORTANT: Utiliser l'image originale pour l'inpainting
                        logger.info("Appel à process_manual_edit avec l'image originale et le masque extrait")
                        result = process_manual_edit(test_image, mask, remove_bg)
                        
                        info_text += "Inpainting terminé avec succès!"
                        return result, mask, info_text
                    except Exception as e:
                        info_text += f"Erreur: {str(e)}"
                        logger.error(f"Erreur lors de l'extraction du masque: {e}")
                        logger.exception("Exception détaillée:")
                        return test_image, None, info_text
                
                # Appliquer l'inpainting
                apply_inpaint_btn.click(
                    fn=process_with_editor_data,
                    inputs=[image_editor, remove_bg_checkbox],
                    outputs=[result_image, extracted_mask_image, process_info]
                )
                
            # Test direct d'inpainting (conserver cette option)
            with gr.Tab("Test direct d'inpainting"):
                gr.Markdown("## Test direct de l'API d'inpainting")
                gr.Markdown(f"Cette section permet de tester directement l'API d'inpainting sur l'image **{TEST_IMAGE_NAME}**.")
                
                with gr.Row():
                    with gr.Column():
                        direct_input_image = gr.Image(label="Image originale", type="pil", value=test_image)
                        direct_mask_image = gr.Image(label="Masque (blanc = zones à remplacer)", type="pil")
                        direct_apply_btn = gr.Button("Appliquer l'inpainting")
                    
                    with gr.Column():
                        direct_result_image = gr.Image(label="Résultat de l'inpainting", type="pil")
                
                # Appliquer l'inpainting direct
                direct_apply_btn.click(
                    fn=test_direct_inpainting,
                    inputs=[direct_input_image, direct_mask_image],
                    outputs=direct_result_image
                )
            
            # Aide pour créer un masque
            with gr.Tab("Aide - Création de masque"):
                gr.Markdown("## Comment créer un masque pour l'inpainting")
                gr.Markdown("""
                ### Comprendre le masque:
                
                Un masque est une image en noir et blanc où:
                - **Zones NOIRES** = zones à préserver (ne pas modifier)
                - **Zones BLANCHES** = zones à remplacer par l'inpainting (le filigrane)
                
                ### Comment créer un bon masque:
                
                1. Commencez avec un masque entièrement noir
                2. Peignez en BLANC les zones contenant le filigrane que vous voulez supprimer
                3. Soyez précis dans votre sélection - couvrez uniquement les zones du filigrane
                4. Pour les filigranes textuels, assurez-vous de couvrir tout le texte
                
                ### Conseils pour de meilleurs résultats:
                
                - Évitez de sélectionner trop de zones (limitez-vous au filigrane)
                - Si le résultat n'est pas satisfaisant, essayez d'ajuster votre masque
                - Pour les filigranes complexes, vous pouvez faire plusieurs passes d'inpainting
                
                ### Exemple:
                
                ![Exemple de masque](https://i.imgur.com/example.jpg)
                """)
        
        except Exception as e:
            logger.error(f"Erreur lors de la création de l'interface: {e}")
            gr.Markdown(f"## Erreur: Impossible de créer l'interface\n\n{str(e)}")
        
    return demo

def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(description="Test d'édition et d'inpainting d'images")
    parser.add_argument("--share", action="store_true", help="Partager l'interface publiquement")
    args = parser.parse_args()
    
    logger.info("Démarrage de l'interface de test d'édition et d'inpainting...")
    demo = create_interface()
    demo.launch(share=args.share)
    
def extract_mask_from_edit_data(edit_data):
    """
    Extrait le masque d'édition du dictionnaire de données d'édition retourné par Gradio.
    Compatible avec différentes versions de Gradio.
    
    Args:
        edit_data: Données d'édition de Gradio
        
    Returns:
        PIL.Image ou None: Masque extrait
    """
    if edit_data is None:
        return None
        
    # Format A: {'mask': image}
    if isinstance(edit_data, dict) and "mask" in edit_data:
        return edit_data["mask"]
        
    # Format B: masque directement
    elif isinstance(edit_data, (Image.Image, np.ndarray)):
        return edit_data
        
    # Format C: d'autres structures possibles selon la version
    elif isinstance(edit_data, dict):
        # Essayer de trouver d'autres clés possibles
        for key in ["edits", "layers", "image"]:
            if key in edit_data:
                if key == "edits":
                    # Dans certaines versions, le masque peut être dans une liste d'éditions
                    if isinstance(edit_data[key], list) and len(edit_data[key]) > 0:
                        last_edit = edit_data[key][-1]
                        if isinstance(last_edit, dict) and "mask" in last_edit:
                            return last_edit["mask"]
                else:
                    return edit_data[key]
    
    # Si on arrive ici, on n'a pas pu extraire le masque
    logger.warning(f"Impossible d'extraire le masque des données d'édition: {type(edit_data)}")
    return None
    
if __name__ == "__main__":
    main() 