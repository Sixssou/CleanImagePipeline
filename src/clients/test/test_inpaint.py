#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging
from PIL import Image
import numpy as np
from dotenv import load_dotenv

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

def test_inpaint_with_files(image_path, mask_path):
    """
    Teste la fonction d'inpainting avec des fichiers d'image et de masque.
    
    Args:
        image_path (str): Chemin vers l'image originale
        mask_path (str): Chemin vers le masque
        
    Returns:
        bool: True si le test est réussi, False sinon
    """
    try:
        # Importer le client après avoir chargé les variables d'environnement
        from src.clients.watermak_removal_client import WatermakRemovalClient
        
        # Vérifier que les fichiers existent
        if not os.path.exists(image_path):
            logger.error(f"Le fichier image {image_path} n'existe pas")
            return False
            
        if not os.path.exists(mask_path):
            logger.error(f"Le fichier masque {mask_path} n'existe pas")
            return False
            
        # Charger les images
        try:
            input_image = Image.open(image_path)
            logger.info(f"Image chargée: {image_path}, taille: {input_image.size}, mode: {input_image.mode}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image: {str(e)}")
            return False
            
        try:
            mask = Image.open(mask_path)
            logger.info(f"Masque chargé: {mask_path}, taille: {mask.size}, mode: {mask.mode}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du masque: {str(e)}")
            return False
            
        # Convertir le masque en mode L (niveaux de gris) si nécessaire
        if mask.mode != 'L':
            logger.info(f"Conversion du masque du mode {mask.mode} au mode L (niveaux de gris)")
            mask = mask.convert('L')
            
        # Initialiser le client
        hf_token = os.getenv("HF_TOKEN")
        space_url = os.getenv("HF_SPACE_WATERMAK_REMOVAL")
        
        if not hf_token or not space_url:
            logger.error("Les variables d'environnement HF_TOKEN et/ou HF_SPACE_WATERMAK_REMOVAL ne sont pas définies")
            return False
            
        logger.info(f"Initialisation du client avec space_url={space_url}")
        client = WatermakRemovalClient(hf_token, space_url)
        
        # Appeler la fonction inpaint
        logger.info("Appel de la fonction inpaint")
        success, result = client.inpaint(input_image, mask)
        
        # Vérifier le résultat
        if not success:
            logger.error("L'inpainting a échoué")
            return False
            
        if result is None:
            logger.error("Le résultat de l'inpainting est None")
            return False
            
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
            return False
            
        logger.info("Test réussi!")
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Fonction principale pour exécuter le test depuis la ligne de commande.
    """
    parser = argparse.ArgumentParser(description="Test de la fonction d'inpainting avec des fichiers")
    parser.add_argument("image_path", help="Chemin vers l'image originale")
    parser.add_argument("mask_path", help="Chemin vers le masque")
    
    args = parser.parse_args()
    
    success = test_inpaint_with_files(args.image_path, args.mask_path)
    
    if success:
        logger.info("Test réussi!")
        sys.exit(0)
    else:
        logger.error("Test échoué!")
        sys.exit(1)

if __name__ == "__main__":
    main() 