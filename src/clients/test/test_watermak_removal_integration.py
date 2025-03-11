#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests d'intégration pour la classe WatermakRemovalClient.
Ce script teste unitairement chaque fonction du client de suppression de filigrane 
pour s'assurer qu'elles fonctionnent correctement.
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
import time

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

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

class TestWatermakRemovalClient(unittest.TestCase):
    """Test d'intégration pour la classe WatermakRemovalClient."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Initialiser le client
        cls.client = WatermakRemovalClient(
            hf_token=os.getenv('HF_TOKEN'),
            space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
        )
        
        # Créer un répertoire temporaire pour les tests si nécessaire
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Télécharger l'image de test
        cls.test_image_path = os.path.join(TEMP_DIR, "test_wm_client_image.png")
        cls.download_image(TEST_IMAGE_URL, cls.test_image_path)
        
        # Créer un masque de test simple (rectangle blanc au centre)
        cls.test_mask_path = os.path.join(TEMP_DIR, "test_wm_client_mask.png")
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
    
    def test_inpaint(self):
        """Teste la fonction inpaint."""
        logger.info("Test de la fonction inpaint")
        try:
            # Charger l'image et le masque
            image = Image.open(self.test_image_path)
            mask = Image.open(self.test_mask_path)
            
            # Vérifier les dimensions
            logger.info(f"Dimensions de l'image: {image.size}")
            logger.info(f"Dimensions du masque: {mask.size}")
            
            # Convertir les images en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            if mask.mode != 'L':
                mask = mask.convert('L')
            
            # Appeler la fonction inpaint
            result = self.client.inpaint(image, mask)
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Enregistrer le résultat pour inspection visuelle
            result_path = os.path.join(TEMP_DIR, "test_wm_client_inpaint_result.png")
            if isinstance(result, np.ndarray):
                Image.fromarray(result).save(result_path)
            elif isinstance(result, Image.Image):
                result.save(result_path)
            else:
                shutil.copy(result, result_path)
            logger.info(f"Résultat d'inpainting enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test d'inpainting : {e}")
    
    def test_remove_bg(self):
        """Teste la fonction remove_bg."""
        logger.info("Test de la fonction remove_bg")
        try:
            # Charger l'image de test
            image = Image.open(self.test_image_path)
            
            # Convertir l'image en RGB si nécessaire
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Appeler la fonction remove_bg
            result = self.client.remove_bg(image)
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Enregistrer le résultat pour inspection visuelle
            result_path = os.path.join(TEMP_DIR, "test_wm_client_remove_bg_result.png")
            if isinstance(result, np.ndarray):
                Image.fromarray(result).save(result_path)
            elif isinstance(result, Image.Image):
                result.save(result_path)
            else:
                shutil.copy(result, result_path)
            logger.info(f"Résultat de remove_bg enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de remove_bg : {e}")
    
    def test_detect_wm(self):
        """Teste la fonction detect_wm."""
        logger.info("Test de la fonction detect_wm")
        try:
            # Appeler la fonction detect_wm avec l'URL de l'image de test
            result = self.client.detect_wm(
                image_url=TEST_IMAGE_URL,
                threshold=0.8
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Vérifier le format du résultat
            if result and isinstance(result, tuple) and len(result) >= 2:
                success, data = result[:2]
                self.assertIsInstance(success, bool, "Le premier élément n'est pas un booléen")
                if success:
                    self.assertIsInstance(data, list, "Les données ne sont pas une liste")
                    logger.info(f"Filigranes détectés : {len(data)}")
                    for i, watermark in enumerate(data):
                        self.assertIn('bbox', watermark, f"Filigrane {i} : pas de bbox")
                        self.assertIn('confidence', watermark, f"Filigrane {i} : pas de confidence")
                        logger.info(f"Filigrane {i}: bbox={watermark['bbox']}, confidence={watermark['confidence']}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de detect_wm : {e}")
    
    def test_remove_wm(self):
        """Teste la fonction remove_wm."""
        logger.info("Test de la fonction remove_wm")
        try:
            # Appeler la fonction remove_wm avec l'URL de l'image de test
            result = self.client.remove_wm(
                image_url=TEST_IMAGE_URL,
                max_bbox_percent=10.0,
                bbox_enlargement_factor=1.5
            )
            
            # Attendre un peu car cette fonction peut prendre du temps
            time.sleep(2)
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Vérifier le format du résultat - devrait être un tuple (success, data, error)
            self.assertIsInstance(result, tuple, "Le résultat n'est pas un tuple")
            if len(result) >= 3:
                success, data, error = result
                self.assertIsInstance(success, bool, "Le premier élément n'est pas un booléen")
                logger.info(f"Résultat de remove_wm : success={success}, error={error}")
                
                # Si succès, vérifier les données
                if success and data:
                    self.assertIsInstance(data, dict, "Les données ne sont pas un dictionnaire")
                    self.assertIn('original_image', data, "Pas d'image originale dans les données")
                    self.assertIn('bbox_image', data, "Pas d'image avec bbox dans les données")
                    self.assertIn('mask_image', data, "Pas de masque dans les données")
                    self.assertIn('inpainted_image', data, "Pas d'image inpainted dans les données")
                    
                    # Enregistrer les résultats pour inspection visuelle
                    for key, image_data in data.items():
                        if image_data:
                            result_path = os.path.join(TEMP_DIR, f"test_wm_client_remove_wm_{key}.png")
                            if isinstance(image_data, np.ndarray):
                                Image.fromarray(image_data).save(result_path)
                            elif isinstance(image_data, Image.Image):
                                image_data.save(result_path)
                            elif isinstance(image_data, str) and os.path.exists(image_data):
                                shutil.copy(image_data, result_path)
                            logger.info(f"Résultat {key} enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de remove_wm : {e}")
    
    def test_detect_and_generate_mask(self):
        """Teste la fonction detect_and_generate_mask."""
        logger.info("Test de la fonction detect_and_generate_mask")
        try:
            # Charger l'image de test
            image = Image.open(self.test_image_path)
            
            # Appeler la fonction detect_and_generate_mask
            result = self.client.detect_and_generate_mask(
                input_image=image,
                max_bbox_percent=10.0,
                bbox_enlargement_factor=1.5
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Vérifier le format du résultat
            self.assertIsInstance(result, tuple, "Le résultat n'est pas un tuple")
            self.assertEqual(len(result), 3, "Le résultat n'a pas 3 éléments")
            
            bbox_image, mask_image, all_bboxes = result
            
            # Vérifier les composants du résultat
            self.assertIsNotNone(bbox_image, "L'image avec bboxes est None")
            self.assertIsNotNone(mask_image, "Le masque est None")
            self.assertIsInstance(all_bboxes, list, "all_bboxes n'est pas une liste")
            
            # Enregistrer les résultats pour inspection visuelle
            bbox_path = os.path.join(TEMP_DIR, "test_wm_client_bbox_image.png")
            if isinstance(bbox_image, np.ndarray):
                Image.fromarray(bbox_image).save(bbox_path)
            elif isinstance(bbox_image, Image.Image):
                bbox_image.save(bbox_path)
            logger.info(f"Image avec bboxes enregistrée : {bbox_path}")
            
            mask_path = os.path.join(TEMP_DIR, "test_wm_client_mask_image.png")
            if isinstance(mask_image, np.ndarray):
                Image.fromarray(mask_image).save(mask_path)
            elif isinstance(mask_image, Image.Image):
                mask_image.save(mask_path)
            logger.info(f"Masque enregistré : {mask_path}")
            
            # Afficher les bbox détectées
            for i, bbox in enumerate(all_bboxes):
                logger.info(f"BBox {i}: {bbox}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de detect_and_generate_mask : {e}")
    
    def test_add_watermark(self):
        """Teste la fonction add_watermark."""
        logger.info("Test de la fonction add_watermark")
        try:
            # Charger l'image de test
            image = Image.open(self.test_image_path)
            
            # Appeler la fonction add_watermark
            result = self.client.add_watermark(
                image=image,
                text="TEST WATERMARK"
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            
            # Enregistrer le résultat pour inspection visuelle
            result_path = os.path.join(TEMP_DIR, "test_wm_client_add_watermark_result.png")
            if isinstance(result, np.ndarray):
                Image.fromarray(result).save(result_path)
            elif isinstance(result, Image.Image):
                result.save(result_path)
            else:
                shutil.copy(result, result_path)
            logger.info(f"Résultat d'add_watermark enregistré : {result_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test d'add_watermark : {e}")
            
    def test_predict(self):
        """Teste la fonction predict de base."""
        logger.info("Test de la fonction predict")
        try:
            # Préparer les données pour l'appel API
            api_name = "/detect_wm"
            data = {"image_url": TEST_IMAGE_URL, "threshold": 0.8}
            
            # Appeler la fonction predict
            result = self.client.predict(api_name, data)
            
            # Vérifier les résultats
            self.assertIsNotNone(result, "Le résultat est None")
            self.assertIsInstance(result, tuple, "Le résultat n'est pas un tuple")
            
            # Le résultat devrait être un tuple (success, data, error)
            if len(result) >= 3:
                success, data, error = result
                logger.info(f"Résultat de predict : success={success}, error={error}")
                if success:
                    self.assertIsNotNone(data, "Les données sont None malgré le succès")
            
        except Exception as e:
            self.fail(f"Exception lors du test de predict : {e}")

if __name__ == "__main__":
    unittest.main() 