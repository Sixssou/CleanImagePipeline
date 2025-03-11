#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests d'intégration pour la classe CleanImagePipeline.
Ce script teste unitairement chaque fonction du pipeline pour s'assurer qu'elles fonctionnent correctement,
et teste également l'ensemble du flux de traitement des images.
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

from src.pipeline.pipeline import CleanImagePipeline
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

class TestCleanImagePipeline(unittest.TestCase):
    """Test d'intégration pour la classe CleanImagePipeline."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Initialiser les clients nécessaires
        cls.watermak_removal_client = WatermakRemovalClient(
            hf_token=os.getenv('HF_TOKEN'),
            space_url=os.getenv('HF_SPACE_WATERMAK_REMOVAL')
        )
        
        # Initialiser le CleanImagePipeline avec les clients
        cls.pipeline = CleanImagePipeline(
            watermak_removal_client=cls.watermak_removal_client
        )
        
        # Créer un répertoire temporaire pour les tests si nécessaire
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        # Télécharger l'image de test
        cls.test_image_path = os.path.join(TEMP_DIR, "test_pipeline_image.png")
        cls.download_image(TEST_IMAGE_URL, cls.test_image_path)
        
        # Créer un masque de test simple (rectangle blanc au centre)
        cls.test_mask_path = os.path.join(TEMP_DIR, "test_pipeline_mask.png")
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
    
    def test_process_url(self):
        """Teste la fonction process_url du pipeline."""
        logger.info("Test de la fonction process_url")
        try:
            # Appeler la fonction process_url avec l'URL de l'image de test
            result_path, source_image, metadata = self.pipeline.process_url(
                url=TEST_IMAGE_URL,
                is_watermarked=True,
                bg_removed=False,
                add_watermark=False,
                watermark_text=None
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(source_image, "L'image source est None")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_process_url_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat de process_url enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de process_url : {e}")
    
    def test_manual_edit(self):
        """Teste le workflow de modification manuelle."""
        logger.info("Test du workflow de modification manuelle")
        try:
            # Charger l'image et le masque de test
            image = Image.open(self.test_image_path)
            mask = Image.open(self.test_mask_path)
            
            # Appeler la fonction process_manual_edit
            result_path, metadata = self.pipeline.process_manual_edit(
                image=image,
                edited_mask=mask,
                bg_removed=False,
                add_watermark=False,
                watermark_text=None
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_manual_edit_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat de process_manual_edit enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de process_manual_edit : {e}")
    
    def test_process_image_path(self):
        """Teste la fonction process_image_path."""
        logger.info("Test de la fonction process_image_path")
        try:
            # Appeler la fonction process_image_path avec le chemin de l'image de test
            result_path, source_image, metadata = self.pipeline.process_image_path(
                image_path=self.test_image_path,
                is_watermarked=True,
                bg_removed=False,
                add_watermark=False,
                watermark_text=None
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(source_image, "L'image source est None")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_process_image_path_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat de process_image_path enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de process_image_path : {e}")
    
    def test_process_uploaded_file(self):
        """Teste la fonction process_uploaded_file."""
        logger.info("Test de la fonction process_uploaded_file")
        try:
            # Préparer un dictionnaire simulant un fichier uploadé
            # (À adapter selon la structure attendue par process_uploaded_file)
            uploaded_file = {
                "path": self.test_image_path,
                "type": "image/png",
                "name": "test_image.png"
            }
            
            # Appeler la fonction process_uploaded_file
            result_path, source_image, metadata = self.pipeline.process_uploaded_file(
                uploaded_file=uploaded_file,
                is_watermarked=True,
                bg_removed=False,
                add_watermark=False,
                watermark_text=None
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(source_image, "L'image source est None")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_process_uploaded_file_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat de process_uploaded_file enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de process_uploaded_file : {e}")
    
    def test_remove_background(self):
        """Teste la fonction process_url avec suppression d'arrière-plan."""
        logger.info("Test de la fonction process_url avec suppression d'arrière-plan")
        try:
            # Appeler la fonction process_url avec suppression d'arrière-plan
            result_path, source_image, metadata = self.pipeline.process_url(
                url=TEST_IMAGE_URL,
                is_watermarked=True,
                bg_removed=True,  # Activer la suppression d'arrière-plan
                add_watermark=False,
                watermark_text=None
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(source_image, "L'image source est None")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Vérifier que le résultat est au format PNG (pour la transparence)
            self.assertTrue(result_path.endswith('.png'), "Le résultat n'est pas au format PNG")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_bg_removed_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat avec suppression d'arrière-plan enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test de suppression d'arrière-plan : {e}")
    
    def test_add_watermark(self):
        """Teste la fonction process_url avec ajout de filigrane."""
        logger.info("Test de la fonction process_url avec ajout de filigrane")
        try:
            # Appeler la fonction process_url avec ajout de filigrane
            result_path, source_image, metadata = self.pipeline.process_url(
                url=TEST_IMAGE_URL,
                is_watermarked=True,
                bg_removed=False,
                add_watermark=True,  # Activer l'ajout de filigrane
                watermark_text="TEST WATERMARK"
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(source_image, "L'image source est None")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_watermark_added_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat avec ajout de filigrane enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test d'ajout de filigrane : {e}")
    
    def test_complete_workflow(self):
        """Teste le workflow complet avec suppression de filigrane, suppression d'arrière-plan et ajout de filigrane."""
        logger.info("Test du workflow complet")
        try:
            # Appeler la fonction process_url avec toutes les options activées
            result_path, source_image, metadata = self.pipeline.process_url(
                url=TEST_IMAGE_URL,
                is_watermarked=True,  # Suppression de filigrane
                bg_removed=True,      # Suppression d'arrière-plan
                add_watermark=True,   # Ajout de filigrane
                watermark_text="CLEAN IMAGE"
            )
            
            # Vérifier les résultats
            self.assertIsNotNone(result_path, "Le chemin de résultat est None")
            self.assertTrue(os.path.exists(result_path), f"Le fichier de résultat n'existe pas : {result_path}")
            self.assertIsNotNone(source_image, "L'image source est None")
            self.assertIsNotNone(metadata, "Les métadonnées sont None")
            
            # Vérifier que le résultat est au format PNG (pour la transparence)
            self.assertTrue(result_path.endswith('.png'), "Le résultat n'est pas au format PNG")
            
            # Vérifier les métadonnées
            self.assertIn('is_watermarked', metadata, "Les métadonnées ne contiennent pas is_watermarked")
            self.assertIn('bg_removed', metadata, "Les métadonnées ne contiennent pas bg_removed")
            self.assertIn('add_watermark', metadata, "Les métadonnées ne contiennent pas add_watermark")
            
            # Copier le résultat pour inspection visuelle
            target_path = os.path.join(TEMP_DIR, "test_pipeline_complete_workflow_result.png")
            shutil.copy(result_path, target_path)
            logger.info(f"Résultat du workflow complet enregistré : {target_path}")
            
        except Exception as e:
            self.fail(f"Exception lors du test du workflow complet : {e}")

if __name__ == "__main__":
    unittest.main() 