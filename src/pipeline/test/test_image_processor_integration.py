#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests d'intégration pour la classe ImageProcessor.
Ce script teste unitairement chaque fonction du processeur d'images pour s'assurer qu'elles fonctionnent correctement.
"""

import os
import sys
import unittest
from PIL import Image
import numpy as np
from dotenv import load_dotenv
import tempfile
import shutil
import logging
from pathlib import Path

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.pipeline.image_processor import ImageProcessor
from src.clients.watermak_removal_client import WatermakRemovalClient
from src.utils.image_utils import TEMP_DIR, download_image

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Charger les variables d'environnement
load_dotenv()

# URL d'image de test (filigrane visible)
TEST_IMAGE_URL = "https://cdn.shopify.com/s/files/1/0503/4324/8066/files/0001_975c348c-b6f1-4c40-b503-f40358b442f0.png?v=1739784393"

class TestImageProcessor(unittest.TestCase):
    """Test d'intégration pour la classe ImageProcessor."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Initialiser les clients nécessaires
        cls.watermak_removal_client = WatermakRemovalClient(
            hf_token=os.getenv('HF_TOKEN'),
            space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
        )
        
        # Initialiser l'ImageProcessor avec le client
        cls.image_processor = ImageProcessor(
            watermak_removal_client=cls.watermak_removal_client
        )
        
        # Créer un répertoire temporaire pour les tests si nécessaire
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Télécharger l'image de test
        cls.test_image_path = os.path.join(TEMP_DIR, "test_image_processor.png")
        cls.download_image(TEST_IMAGE_URL, cls.test_image_path)
        
        # Créer un masque de test simple (rectangle blanc au centre)
        cls.test_mask_path = os.path.join(TEMP_DIR, "test_mask_processor.png")
        cls.create_test_mask(cls.test_image_path, cls.test_mask_path)
        
        logger.info(f"Configuration terminée : image={cls.test_image_path}, masque={cls.test_mask_path}")
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après tous les tests."""
        # Supprimer les fichiers de test
        for path in [cls.test_image_path, cls.test_mask_path]:
            if os.path.exists(path):
                os.remove(path)
        logger.info("Nettoyage terminé")
    
    @staticmethod
    def download_image(url, save_path):
        """Télécharge une image depuis une URL."""
        import requests
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"Image téléchargée : {save_path}")
        else:
            logger.error(f"Erreur lors du téléchargement de l'image : {response.status_code}")
    
    @staticmethod
    def create_test_mask(image_path, mask_path, rect_percent=0.2):
        """Crée un masque de test simple (rectangle blanc au centre)."""
        try:
            # Ouvrir l'image source pour obtenir ses dimensions
            img = Image.open(image_path)
            width, height = img.size
            
            # Créer un masque noir (tout zéros)
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Calculer les dimensions du rectangle blanc au centre
            rect_width = int(width * rect_percent)
            rect_height = int(height * rect_percent)
            x1 = (width - rect_width) // 2
            y1 = (height - rect_height) // 2
            x2 = x1 + rect_width
            y2 = y1 + rect_height
            
            # Ajouter un rectangle blanc
            mask[y1:y2, x1:x2] = 255
            
            # Enregistrer le masque
            Image.fromarray(mask).save(mask_path)
            logger.info(f"Masque de test créé : {mask_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du masque : {e}")
    
    def test_detect_watermarks(self):
        """Teste la fonction detect_watermarks."""
        logger.info("Test de la fonction detect_watermarks")
        try:
            # Appeler la fonction detect_watermarks avec l'URL de l'image de test
            result = self.image_processor.detect_watermarks(
                image_url=TEST_IMAGE_URL,
                max_bbox_percent=10.0, 
                bbox_enlargement_factor=1.5
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Le résultat devrait être un tuple de 3 éléments (image, mask, inpainted)
            self.assertIsInstance(result, tuple, "Le résultat n'est pas un tuple")
            self.assertEqual(len(result), 3, "Le résultat n'a pas 3 éléments")
            
            # Enregistrer les résultats pour inspection visuelle
            original, bbox_image, mask = result
            
            if original is not None:
                original_path = os.path.join(TEMP_DIR, "test_ip_detect_original.png")
                if isinstance(original, np.ndarray):
                    Image.fromarray(original).save(original_path)
                elif isinstance(original, Image.Image):
                    original.save(original_path)
                else:
                    shutil.copy(original, original_path)
                logger.info(f"Image originale enregistrée : {original_path}")
            
            if bbox_image is not None:
                bbox_path = os.path.join(TEMP_DIR, "test_ip_detect_bbox.png")
                if isinstance(bbox_image, np.ndarray):
                    Image.fromarray(bbox_image).save(bbox_path)
                elif isinstance(bbox_image, Image.Image):
                    bbox_image.save(bbox_path)
                else:
                    shutil.copy(bbox_image, bbox_path)
                logger.info(f"Image avec bboxes enregistrée : {bbox_path}")
            
            if mask is not None:
                mask_path = os.path.join(TEMP_DIR, "test_ip_detect_mask.png")
                if isinstance(mask, np.ndarray):
                    Image.fromarray(mask).save(mask_path)
                elif isinstance(mask, Image.Image):
                    mask.save(mask_path)
                else:
                    shutil.copy(mask, mask_path)
                logger.info(f"Masque détecté enregistré : {mask_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de detect_watermarks : {e}")
    
    def test_inpaint_image(self):
        """Teste la fonction inpaint_image."""
        logger.info("Test de la fonction inpaint_image")
        try:
            # Charger l'image et le masque
            image = Image.open(self.test_image_path)
            mask = Image.open(self.test_mask_path)
            
            # Appeler la fonction inpaint_image
            result = self.image_processor.inpaint_image(image, mask)
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Enregistrer le résultat pour inspection visuelle
            result_path = os.path.join(TEMP_DIR, "test_ip_inpaint_result.png")
            if isinstance(result, np.ndarray):
                Image.fromarray(result).save(result_path)
            elif isinstance(result, Image.Image):
                result.save(result_path)
            else:
                shutil.copy(result, result_path)
            logger.info(f"Résultat d'inpainting enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test d'inpainting_image : {e}")
    
    def test_remove_background(self):
        """Teste la fonction remove_background."""
        logger.info("Test de la fonction remove_background")
        try:
            # Charger l'image de test
            image = Image.open(self.test_image_path)
            image_array = np.array(image)
            
            # Appeler la fonction remove_background
            result = self.image_processor.remove_background(image_array)
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Enregistrer le résultat pour inspection visuelle
            result_path = os.path.join(TEMP_DIR, "test_ip_remove_bg_result.png")
            if isinstance(result, np.ndarray):
                Image.fromarray(result).save(result_path)
            elif isinstance(result, Image.Image):
                result.save(result_path)
            else:
                shutil.copy(result, result_path)
            logger.info(f"Résultat de remove_background enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de remove_background : {e}")
    
    def test_apply_manual_edits(self):
        """Teste la fonction apply_manual_edits."""
        logger.info("Test de la fonction apply_manual_edits")
        try:
            # Charger l'image et le masque de test
            image = Image.open(self.test_image_path)
            mask = Image.open(self.test_mask_path)
            
            # Appeler la fonction apply_manual_edits
            input_path, mask_path, result_path = self.image_processor.apply_manual_edits(
                image=image,
                edited_mask=mask,
                supprimer_background=False,
                add_watermark=False,
                watermark_text=None
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(input_path, "Le chemin d'entrée est None")
            self.assertIsNotNone(mask_path, "Le chemin du masque est None")
            self.assertIsNotNone(result_path, "Le chemin du résultat est None")
            
            # Copier le résultat pour inspection visuelle
            if result_path is not None and os.path.exists(result_path):
                target_path = os.path.join(TEMP_DIR, "test_ip_manual_edit_result.png")
                shutil.copy(result_path, target_path)
                logger.info(f"Résultat de apply_manual_edits enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de apply_manual_edits : {e}")
    
    def test_process_single_image(self):
        """Teste la fonction process_single_image."""
        logger.info("Test de la fonction process_single_image")
        try:
            # Appeler la fonction process_single_image
            result = self.image_processor.process_single_image(
                image_url=TEST_IMAGE_URL,
                supprimer_background=False,
                add_watermark=False,
                watermark_text=None,
                max_bbox_percent=10.0,
                bbox_enlargement_factor=1.5
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Le résultat devrait être un tuple
            self.assertIsInstance(result, tuple, "Le résultat n'est pas un tuple")
            
            # Enregistrer le résultat pour inspection visuelle
            if len(result) >= 1 and result[0] is not None:
                final_image = result[0]
                result_path = os.path.join(TEMP_DIR, "test_ip_process_single_result.png")
                if isinstance(final_image, np.ndarray):
                    Image.fromarray(final_image).save(result_path)
                elif isinstance(final_image, Image.Image):
                    final_image.save(result_path)
                else:
                    shutil.copy(final_image, result_path)
                logger.info(f"Résultat de process_single_image enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de process_single_image : {e}")

if __name__ == "__main__":
    unittest.main() 