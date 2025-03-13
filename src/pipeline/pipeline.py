import os
from loguru import logger
import tempfile
from typing import List, Tuple, Dict, Any, Optional
import uuid
import numpy as np
from PIL import Image
import shutil

from src.pipeline.image_processor import ImageProcessor
from src.utils.image_utils import download_image, TEMP_DIR, add_watermark_to_image

class CleanImagePipeline:
    """
    Classe pour gérer le pipeline complet de traitement des images.
    """
    
    def __init__(self, gsheet_client=None, shopify_client=None, watermak_removal_client=None):
        """
        Initialise le pipeline de traitement des images.
        
        Args:
            gsheet_client: Client pour Google Sheets
            shopify_client: Client pour Shopify
            watermak_removal_client: Client pour la suppression de watermarks
        """
        self.gsheet_client = gsheet_client
        self.shopify_client = shopify_client
        self.image_processor = ImageProcessor(watermak_removal_client)
        
        # S'assurer que le répertoire temporaire existe
        os.makedirs(TEMP_DIR, exist_ok=True)
    
    def set_clients(self, gsheet_client=None, shopify_client=None, watermak_removal_client=None):
        """
        Définit les clients pour le pipeline.
        
        Args:
            gsheet_client: Client pour Google Sheets
            shopify_client: Client pour Shopify
            watermak_removal_client: Client pour la suppression de watermarks
        """
        if gsheet_client:
            self.gsheet_client = gsheet_client
        if shopify_client:
            self.shopify_client = shopify_client
        if watermak_removal_client:
            self.watermak_removal_client = watermak_removal_client
            self.image_processor.set_watermak_removal_client(watermak_removal_client)
    
    def test_connections(self, gsheet_id=None, gsheet_sheet=None, 
                         shopify_domain=None, shopify_api_version=None, shopify_api_key=None):
        """
        Teste les connexions aux différentes APIs.
        
        Returns:
            Dict[str, bool]: Résultats des tests
        """
        results = {}
        
        # Test de la connexion à Google Sheets
        if self.gsheet_client and gsheet_id and gsheet_sheet:
            try:
                results["gsheet"] = self.gsheet_client.test_connection(gsheet_id, gsheet_sheet)
            except Exception as e:
                logger.error(f"Erreur lors du test de connexion à Google Sheets: {str(e)}")
                results["gsheet"] = False
        
        # Test de la connexion à Shopify
        if self.shopify_client and shopify_domain and shopify_api_version and shopify_api_key:
            try:
                results["shopify"] = self.shopify_client.test_connection(
                    shopify_domain, shopify_api_version, shopify_api_key
                )
            except Exception as e:
                logger.error(f"Erreur lors du test de connexion à Shopify: {str(e)}")
                results["shopify"] = False
        
        # Test de la connexion au service de suppression de watermarks
        if self.image_processor.watermak_removal_client:
            try:
                # Tester avec une petite image de test
                test_url = "https://picsum.photos/200"
                _, _, mask = self.image_processor.detect_watermarks(test_url)
                results["watermark_removal"] = mask is not None
            except Exception as e:
                logger.error(f"Erreur lors du test de connexion au service de suppression de watermarks: {str(e)}")
                results["watermark_removal"] = False
        
        return results
    
    def process_images_batch(self, image_count: int, sheet_name: str,
                          remove_background: bool = False,
                          add_watermark: bool = False,
                          watermark_text: str = None):
        """
        Traite un lot d'images depuis Google Sheets.
        
        Args:
            image_count: Nombre d'images à traiter
            sheet_name: Nom de la feuille dans Google Sheets
            remove_background: Supprimer l'arrière-plan des images traitées
            add_watermark: Ajouter un filigrane aux images traitées
            watermark_text: Texte du filigrane
            
        Returns:
            List: Données des images traitées
        """
        if not self.gsheet_client:
            raise ValueError("Le client Google Sheets n'est pas défini")
        if not self.shopify_client:
            raise ValueError("Le client Shopify n'est pas défini")
        if not self.image_processor.watermak_removal_client:
            raise ValueError("Le client de suppression de watermarks n'est pas défini")
        
        # Récupérer les données depuis Google Sheets
        data = self.gsheet_client.read_cells(sheet_name, f"A2:C{image_count}")
        
        # Traiter chaque image
        for index, row in enumerate(data, start=1):
            lien_image_source = row[0]
            lien_image_traitee = row[1] if len(row) > 1 else ""
            supprimer_background = row[2].upper() == 'TRUE' if len(row) > 2 else remove_background
            
            logger.info(f"Compteur : {index}")
            logger.info(f"Lien image source : {lien_image_source}")
            logger.info(f"Lien image traitee : {lien_image_traitee}")
            logger.info(f"Supprimer background : {supprimer_background}")
            
            # Si l'image n'a pas déjà été traitée
            if not lien_image_traitee:
                try:
                    # Traiter l'image
                    result = self.image_processor.watermak_removal_client.remove_wm(
                        image_url=lien_image_source,
                        threshold=0.85,
                        max_bbox_percent=10.0,
                        remove_background_option=supprimer_background,
                        add_watermark_option=add_watermark,
                        watermark=watermark_text,
                        bbox_enlargement_factor=1.5,
                        remove_watermark_iterations=1
                    )
                    
                    if not result or len(result) != 4:
                        logger.error(f"Le traitement de l'image a échoué: {lien_image_source}")
                        continue
                    
                    # Extraire les résultats
                    _, _, _, result_image = result
                    
                    # Sauvegarder l'image temporairement
                    result_path = os.path.join(TEMP_DIR, f"result_{index}.png")
                    with open(result_path, "wb") as f:
                        f.write(result_image)
                    
                    # Télécharger l'image sur Shopify
                    lien_image_traitee = self.shopify_client.upload_file_to_shopify(result_path)
                    logger.info(f"Image nettoyée et uploadée : {lien_image_traitee}")
                    
                    # Mettre à jour la feuille Google Sheets
                    self.gsheet_client.write_cells(sheet_name, f"B{index}", [[lien_image_traitee]])
                    
                    # Supprimer le fichier temporaire
                    os.remove(result_path)
                    
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de l'image {lien_image_source}: {str(e)}")
                    logger.exception(e)
        
        return data
    
    def prepare_images_for_manual_edit(self, image_count: int, sheet_name: str):
        """
        Prépare les images pour l'édition manuelle.
        
        Args:
            image_count: Nombre d'images à préparer
            sheet_name: Nom de la feuille dans Google Sheets
            
        Returns:
            List: Liste des images à traiter avec leurs informations
        """
        if not self.gsheet_client:
            raise ValueError("Le client Google Sheets n'est pas défini")
        if not self.image_processor.watermak_removal_client:
            raise ValueError("Le client de suppression de watermarks n'est pas défini")
        
        try:
            # Récupérer les données depuis Google Sheets
            data = self.gsheet_client.read_cells(sheet_name, f"A1:C{image_count+1}")
            data = data[1:]
            
            results = []
            for idx, row in enumerate(data, start=2):
                lien_image_source = row[0]
                lien_image_traitee = row[1] if len(row) > 1 else ""
                supprimer_background = row[2].upper() == 'TRUE' if len(row) > 2 else False
                logger.info(f"Lien image source : {lien_image_source}")
                logger.info(f"Lien image traitee : {lien_image_traitee}")
                logger.info(f"Supprimer background : {supprimer_background}")
                
                # Si l'image n'a pas déjà été traitée
                if not lien_image_traitee:
                    try:
                        # Télécharger l'image source
                        image_source = download_image(lien_image_source)
                        
                        # Convertir l'image pour l'interface Gradio
                        pil_image = Image.fromarray(image_source) if isinstance(image_source, np.ndarray) else image_source
                        
                        # Ajouter à la liste des résultats
                        # Format: (index, lien_image_source, image originale, supprimer_background)
                        results.append((idx, lien_image_source, pil_image, supprimer_background))
                        
                    except Exception as e:
                        logger.error(f"Erreur lors de la préparation de l'image {lien_image_source}: {str(e)}")
                        logger.exception(e)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des images: {str(e)}")
            logger.exception(e)
            return []
    
    def process_edited_image(self, index: int, image_url: str, edited_mask=None, inpainted_result=None,
                           remove_background: bool = False,
                           add_watermark: bool = False,
                           watermark_text: str = None,
                           sheet_name: str = None):
        """
        Traite une image après édition manuelle ou inpainting automatique.
        
        Args:
            index (int): Index de l'image dans la liste
            image_url (str): URL de l'image originale
            edited_mask (Image.Image, optional): Masque édité manuellement
            inpainted_result (np.ndarray, optional): Résultat de l'inpainting automatique
            remove_background (bool, optional): Si True, supprime l'arrière-plan. Par défaut False.
            add_watermark (bool, optional): Si True, ajoute un watermark. Par défaut False.
            watermark_text (str, optional): Texte du watermark à ajouter
            sheet_name (str, optional): Nom de la feuille de calcul pour l'enregistrement
            
        Returns:
            str: Chemin vers l'image traitée ou None en cas d'échec
        """
        logger.info("=== DÉBUT process_edited_image ===")
        logger.info(f"edited_mask est None: {edited_mask is None}")
        logger.info(f"inpainted_result est None: {inpainted_result is None}")
        logger.info(f"Type de edited_mask: {type(edited_mask)}")
        logger.info(f"Type de inpainted_result: {type(inpainted_result)}")
        
        if self.shopify_client is None:
            logger.warning("Shopify client n'est pas initialisé, les résultats ne seront pas ajoutés au site.")
        
        # Créer un répertoire temporaire si nécessaire
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR, exist_ok=True)
        
        try:
            result_path = None
            
            # Si l'inpainting a déjà été effectué et qu'on utilise ce résultat
            if inpainted_result is not None and edited_mask is None:
                # Mode automatique - utiliser directement le résultat de l'inpainting
                logger.info("Utilisation du résultat d'inpainting existant")
                # Convertir en image PIL si nécessaire
                if isinstance(inpainted_result, np.ndarray):
                    inpainted_image = Image.fromarray(inpainted_result)
                else:
                    inpainted_image = inpainted_result
                
                # Sauvegarder temporairement
                unique_id = uuid.uuid4()
                result_path = os.path.join(TEMP_DIR, f"result_{unique_id}.png")
                
                # Vérifier si inpainted_image est une chaîne (chemin de fichier) ou un objet Image
                if isinstance(inpainted_image, str):
                    # Si c'est un chemin de fichier, on copie simplement le fichier
                    if os.path.exists(inpainted_image):
                        shutil.copy(inpainted_image, result_path)
                    else:
                        # Si le chemin n'existe pas, essayer de charger depuis une URL
                        try:
                            downloaded_image = download_image(inpainted_image)
                            if isinstance(downloaded_image, np.ndarray):
                                Image.fromarray(downloaded_image).save(result_path)
                            else:
                                downloaded_image.save(result_path)
                        except Exception as e:
                            logger.error(f"Impossible de traiter l'image depuis le chemin {inpainted_image}: {str(e)}")
                            raise ValueError(f"Chemin d'image invalide: {inpainted_image}")
                else:
                    # Si c'est un objet Image, on l'enregistre normalement
                    inpainted_image.save(result_path)
                
                logger.info(f"Résultat d'inpainting sauvegardé: {result_path}")
            
            # Si un masque édité a été fourni, appliquer l'inpainting manuellement
            elif edited_mask is not None:
                logger.info("Application de l'inpainting avec le masque édité manuellement")
                
                # Déterminer l'image source à utiliser
                if inpainted_result is not None:
                    # Utiliser le résultat de l'inpainting automatique comme base
                    logger.info("Utilisation du résultat d'inpainting automatique comme base pour l'édition manuelle")
                    
                    # Convertir inpainted_result en image si nécessaire
                    if isinstance(inpainted_result, np.ndarray):
                        image_source = Image.fromarray(inpainted_result)
                    elif isinstance(inpainted_result, Image.Image):
                        image_source = inpainted_result
                    elif isinstance(inpainted_result, str):
                        # Si c'est un chemin ou une URL
                        if os.path.exists(inpainted_result):
                            # C'est un chemin de fichier local
                            logger.info(f"Chargement de l'image depuis le chemin local: {inpainted_result}")
                            try:
                                image_source = Image.open(inpainted_result)
                            except Exception as e:
                                logger.error(f"Erreur lors du chargement de l'image locale: {str(e)}")
                                # Fallback sur l'image originale
                                logger.info("Utilisation de l'image originale comme solution de repli")
                                image_source = download_image(image_url)
                        else:
                            # C'est une URL
                            try:
                                image_source = download_image(inpainted_result)
                            except Exception as e:
                                logger.error(f"Erreur lors du téléchargement de l'image: {str(e)}")
                                # Fallback sur l'image originale
                                logger.info("Utilisation de l'image originale comme solution de repli")
                                image_source = download_image(image_url)
                    else:
                        logger.warning(f"Format non reconnu pour inpainted_result: {type(inpainted_result)}, utilisation de l'image originale")
                        image_source = download_image(image_url)
                else:
                    # MODIFICATION ICI: au lieu d'utiliser l'image originale,
                    # On va d'abord essayer de récupérer l'image inpaintée préalablement
                    logger.info("Tentative de récupération de l'image préalablement traitée (inpainted_preview)")
                    try:
                        # Vérifier si on a déjà traité cette image et qu'elle est disponible dans le cache
                        if hasattr(self, 'image_cache') and image_url in self.image_cache and 'inpainted' in self.image_cache[image_url]:
                            logger.info("Utilisation de l'image inpaintée depuis le cache")
                            image_source = self.image_cache[image_url]['inpainted']
                        else:
                            # Essayer de détecter les watermarks et appliquer l'inpainting automatique
                            logger.info("Application de l'inpainting automatique avant édition manuelle")
                            success, detected_mask, inpainted_auto = self.detect_and_remove_watermark(image_url)
                            if success and inpainted_auto is not None:
                                logger.info("Inpainting automatique réussi, utilisation comme base")
                                image_source = inpainted_auto
                            else:
                                logger.warning("Inpainting automatique échoué, utilisation de l'image originale")
                                image_source = download_image(image_url)
                    except Exception as e:
                        logger.error(f"Erreur lors de la récupération de l'image prétraitée: {str(e)}")
                        logger.info("Utilisation de l'image originale comme solution de repli")
                        image_source = download_image(image_url)
                
                logger.info(f"Image source téléchargée, type: {type(image_source)}")
                
                # Appliquer les modifications manuelles (inpainting)
                logger.info("Appel de apply_manual_edits...")
                input_path, mask_path, result_path = self.image_processor.apply_manual_edits(
                    image=image_source,
                    edited_mask=edited_mask,
                    supprimer_background=False,  # On appliquera remove_bg séparément
                    add_watermark=False,         # On appliquera watermark séparément
                    watermark_text=None
                )
                
                if not result_path:
                    logger.error(f"L'inpainting manuel a échoué: {image_url}")
                    return None
                
                logger.info(f"Inpainting manuel appliqué: {result_path}")
            
            # Supprimer l'arrière-plan si demandé
            if remove_background and result_path:
                try:
                    logger.info(f"Suppression de l'arrière-plan demandée pour {result_path}")
                    
                    # Charger l'image résultante
                    result_image = Image.open(result_path)
                    
                    # Supprimer l'arrière-plan
                    success, no_bg_image = self.image_processor.remove_background(result_image)
                    
                    if success and no_bg_image is not None:
                        # Sauvegarder la nouvelle image sans arrière-plan
                        no_bg_path = os.path.join(TEMP_DIR, f"nobg_{os.path.basename(result_path)}")
                        
                        if isinstance(no_bg_image, np.ndarray):
                            Image.fromarray(no_bg_image).save(no_bg_path)
                        else:
                            no_bg_image.save(no_bg_path)
                        
                        logger.info(f"Image sans arrière-plan sauvegardée: {no_bg_path}")
                        result_path = no_bg_path
                    else:
                        logger.warning("La suppression de l'arrière-plan a échoué")
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression de l'arrière-plan: {str(e)}")
            
            # Ajouter un watermark si demandé
            if add_watermark and result_path and watermark_text:
                try:
                    logger.info(f"Ajout d'un watermark demandé pour {result_path} avec le texte: {watermark_text}")
                    
                    # Charger l'image résultante
                    result_image = Image.open(result_path)
                    
                    # Ajouter le watermark
                    watermarked_image = self.image_processor.add_watermark(result_image, watermark_text)
                    
                    if watermarked_image is not None:
                        # Sauvegarder la nouvelle image avec watermark
                        watermarked_path = os.path.join(TEMP_DIR, f"wm_{os.path.basename(result_path)}")
                        watermarked_image.save(watermarked_path)
                        
                        logger.info(f"Image avec watermark sauvegardée: {watermarked_path}")
                        result_path = watermarked_path
                    else:
                        logger.warning("L'ajout du watermark a échoué")
                except Exception as e:
                    logger.error(f"Erreur lors de l'ajout du watermark: {str(e)}")
            
            # Enregistrer dans Google Sheet si demandé
            if sheet_name and self.gsheet_client:
                try:
                    logger.info(f"Enregistrement dans Google Sheet demandé pour {result_path} dans la feuille: {sheet_name}")
                    logger.info(f"Type de sheet_name: {type(sheet_name)}")
                    logger.info(f"Valeur de index: {index}, type: {type(index)}")
                    
                    # Construire le lien Shopify si disponible
                    shopify_url = ""
                    if self.shopify_client:
                        try:
                            # Construire un nom de fichier unique basé sur l'URL de l'image
                            base_name = os.path.basename(image_url)
                            name_parts = os.path.splitext(base_name)
                            unique_name = f"{name_parts[0]}_processed_{uuid.uuid4().hex[:8]}{name_parts[1]}"
                            
                            # Uploader l'image sur Shopify
                            shopify_url = self.shopify_client.upload_file(result_path, unique_name)
                            logger.info(f"Image uploadée sur Shopify: {shopify_url}")
                        except Exception as e:
                            logger.error(f"Erreur lors de l'upload sur Shopify: {str(e)}")
                    
                    # Mettre à jour la cellule dans Google Sheet avec l'URL Shopify
                    if shopify_url:
                        logger.info(f"Tentative d'écriture dans Google Sheet: feuille={sheet_name}, cellule=B{index}, valeur={shopify_url}")
                        # Vérifier si la feuille existe
                        try:
                            # Tester si on peut accéder à la feuille
                            test_read = self.gsheet_client.read_cells(sheet_name, "A1")
                            logger.info(f"Test de lecture de la feuille {sheet_name} réussi: {test_read}")
                        except Exception as e:
                            logger.error(f"Erreur lors du test de lecture de la feuille {sheet_name}: {str(e)}")
                        
                        # Écrire dans la cellule
                        self.gsheet_client.write_cells(sheet_name, f"B{index}", [[shopify_url]])
                        logger.info(f"URL mise à jour dans Google Sheet: {sheet_name}, cellule B{index}")
                except Exception as e:
                    logger.error(f"Erreur lors de l'enregistrement dans Google Sheet: {sheet_name}")
                    logger.error(f"Détails de l'erreur: {str(e)}")
                    logger.exception(e)
            
            logger.info(f"=== FIN process_edited_image (result_path={result_path}) ===")
            return result_path
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image éditée: {str(e)}")
            logger.exception(e)
            return None 