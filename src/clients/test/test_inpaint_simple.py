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
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <chemin_image> <chemin_masque>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    
    # Vérifier que les fichiers existent
    if not os.path.exists(image_path):
        logger.error(f"Le fichier image {image_path} n'existe pas")
        sys.exit(1)
        
    if not os.path.exists(mask_path):
        logger.error(f"Le fichier masque {mask_path} n'existe pas")
        sys.exit(1)
    
    # Importer le client
    try:
        from src.clients.watermak_removal_client import WatermakRemovalClient
    except ImportError:
        logger.error("Impossible d'importer WatermakRemovalClient. Vérifiez que vous êtes dans le bon répertoire.")
        sys.exit(1)
    
    # Charger les images
    try:
        input_image = Image.open(image_path)
        logger.info(f"Image chargée: {image_path}, taille: {input_image.size}, mode: {input_image.mode}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'image: {str(e)}")
        sys.exit(1)
        
    try:
        mask = Image.open(mask_path)
        logger.info(f"Masque chargé: {mask_path}, taille: {mask.size}, mode: {mask.mode}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du masque: {str(e)}")
        sys.exit(1)
    
    # Convertir le masque en mode L (niveaux de gris) si nécessaire
    if mask.mode != 'L':
        logger.info(f"Conversion du masque du mode {mask.mode} au mode L (niveaux de gris)")
        mask = mask.convert('L')
    
    # Initialiser le client
    hf_token = os.getenv("HF_TOKEN")
    space_url = os.getenv("HF_SPACE_WATERMAK_REMOVAL")
    
    if not hf_token or not space_url:
        logger.error("Les variables d'environnement HF_TOKEN et/ou HF_SPACE_WATERMAK_REMOVAL ne sont pas définies")
        sys.exit(1)
        
    logger.info(f"Initialisation du client avec space_url={space_url}")
    client = WatermakRemovalClient(hf_token, space_url)
    
    # Appeler la fonction inpaint
    logger.info("Appel de la fonction inpaint")
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
        logger.error(f"Type de résultat non pris en charge: {type(result)}")
        sys.exit(1)
        
    logger.info("Test réussi!")
    
if __name__ == "__main__":
    main() 