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
            for idx, row in enumerate(data, start=1):
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
        Traite une image éditée manuellement ou avec le masque original et la met à jour dans Google Sheets.
        
        Args:
            index: Index de l'image dans Google Sheets
            image_url: URL de l'image source
            edited_mask: Masque édité manuellement (None si on utilise le résultat de l'inpainting automatique)
            inpainted_result: Résultat de l'inpainting automatique (None si le masque a été édité manuellement)
            remove_background: Supprimer l'arrière-plan de l'image traitée
            add_watermark: Ajouter un filigrane à l'image traitée
            watermark_text: Texte du filigrane
            sheet_name: Nom de la feuille dans Google Sheets
            
        Returns:
            str: URL de l'image traitée
        """
        logger.info(f"=== DÉBUT process_edited_image ===")
        logger.info(f"edited_mask est None: {edited_mask is None}")
        logger.info(f"inpainted_result est None: {inpainted_result is None}")
        logger.info(f"Type de edited_mask: {type(edited_mask) if edited_mask is not None else 'None'}")
        logger.info(f"Type de inpainted_result: {type(inpainted_result) if inpainted_result is not None else 'None'}")
        
        if not self.shopify_client:
            raise ValueError("Le client Shopify n'est pas défini")
        if not self.gsheet_client and sheet_name:
            raise ValueError("Le client Google Sheets n'est pas défini")
        
        try:
            result_path = None
            
            # Si l'inpainting a déjà été effectué et qu'on utilise ce résultat
            if inpainted_result is not None:
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
                # Télécharger l'image source
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
            else:
                logger.error("Ni masque édité ni résultat d'inpainting fourni")
                return None
                
            # Supprimer l'arrière-plan si demandé
            if remove_background and result_path:
                try:
                    # Ouvrir l'image depuis le chemin local au lieu de passer le chemin comme URL
                    local_image = Image.open(result_path)
                    # Convertir en numpy array si nécessaire
                    local_image_np = np.array(local_image)
                    # Appeler remove_bg avec l'image, pas le chemin
                    bg_removed_image = self.image_processor.watermak_removal_client.remove_bg(local_image_np)
                    if bg_removed_image is not None:
                        bg_removed_path = os.path.join(TEMP_DIR, f"bg_removed_{uuid.uuid4()}.png")
                        # Vérifier si bg_removed_image est une chaîne (chemin de fichier) ou une image
                        result_path_updated = False
                        
                        if isinstance(bg_removed_image, np.ndarray):
                            Image.fromarray(bg_removed_image).save(bg_removed_path)
                            result_path = bg_removed_path
                            result_path_updated = True
                        elif isinstance(bg_removed_image, Image.Image):
                            bg_removed_image.save(bg_removed_path)
                            result_path = bg_removed_path
                            result_path_updated = True
                        elif isinstance(bg_removed_image, str) and os.path.exists(bg_removed_image):
                            shutil.copy(bg_removed_image, bg_removed_path)
                            result_path = bg_removed_path
                            result_path_updated = True
                        else:
                            # Si le format n'est pas reconnu, passer cette étape
                            logger.warning(f"Format non reconnu pour bg_removed_image: {type(bg_removed_image)}")
                        
                        if result_path_updated:
                            logger.info(f"Arrière-plan supprimé: {result_path}")
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression de l'arrière-plan: {str(e)}")
                    logger.exception(e)
            
            # Ajouter un filigrane si demandé
            if add_watermark and watermark_text and result_path:
                result_img = Image.open(result_path)
                watermarked_img = add_watermark_to_image(result_img, watermark_text)
                wm_result_path = os.path.join(TEMP_DIR, f"wm_{uuid.uuid4()}.png")
                watermarked_img.save(wm_result_path)
                result_path = wm_result_path
                logger.info(f"Filigrane ajouté: {result_path}")
            
            # Télécharger l'image sur Shopify
            if result_path:
                image_url_processed = self.shopify_client.upload_file_to_shopify(result_path)
                logger.info(f"Image traitée et uploadée: {image_url_processed}")
                
                # Mettre à jour la feuille Google Sheets si nécessaire
                if self.gsheet_client and sheet_name and index:
                    self.gsheet_client.write_cells(sheet_name, f"B{index}", [[image_url_processed]])
                    logger.info(f"Google Sheets mis à jour pour l'index {index}")
                
                # Supprimer le fichier temporaire
                try:
                    os.remove(result_path)
                except:
                    pass
                
                return image_url_processed
            else:
                return None
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image éditée {image_url}: {str(e)}")
            logger.exception(e)
            return None 