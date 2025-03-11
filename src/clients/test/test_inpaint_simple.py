#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script simple pour tester l'inpainting avec des images et des masques spécifiques.
Usage: python test_inpaint_simple.py <chemin_image> <chemin_masque>
"""

import os
import sys
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from src.clients.watermak_removal_client import WatermakRemovalClient
import logging

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

def main():
    # Vérifier les arguments
    if len(sys.argv) < 3:
        logger.error("Usage: python -m src.clients.test.test_inpaint_simple <image_path> <mask_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    
    # Charger l'image et le masque
    try:
        input_image = Image.open(image_path)
        logger.info(f"Image chargée: {image_path}, taille: {input_image.size}, mode: {input_image.mode}")
        
        mask = Image.open(mask_path)
        logger.info(f"Masque chargé: {mask_path}, taille: {mask.size}, mode: {mask.mode}")
        
        # Convertir le masque en mode L si nécessaire
        if mask.mode != "L":
            mask = mask.convert("L")
            logger.info(f"Masque converti en mode L")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des images: {e}")
        sys.exit(1)
    
    # Initialiser le client
    space_url = os.getenv("WATERMAK_REMOVAL_SPACE_URL", "https://cyrilar-watermak-removal.hf.space")
    logger.info(f"Initialisation du client avec space_url={space_url}")
    client = WatermakRemovalClient(hf_token=None, space_url=space_url)
    
    # Afficher les informations sur l'API
    print("=== Visualisation de l'API d'inpainting ===")
    api_info = client.view_api()
    
    # Effectuer l'inpainting
    success, result = client.inpaint(input_image, mask)
    
    # Vérifier le résultat
    if not success:
        logger.error("L'inpainting a échoué")
        sys.exit(1)
        
    if result is None:
        logger.error("Le résultat de l'inpainting est None")
        sys.exit(1)
    
    # Sauvegarder le résultat
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../output'))
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(output_dir, f"{name}_inpainted{ext}")
    
    if isinstance(result, np.ndarray):
        # Convertir le résultat en PIL Image
        result_image = Image.fromarray(result)
        result_image.save(output_path)
        logger.info(f"Résultat sauvegardé: {output_path}")
    else:
        logger.error(f"Type de résultat inattendu: {type(result)}")
        sys.exit(1)
    
    logger.info("Test d'inpainting terminé avec succès")
    
if __name__ == "__main__":
    main() 