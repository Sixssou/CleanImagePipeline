import os
import uuid
import numpy as np
import tempfile
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from loguru import logger

# Définir TEMP_DIR
TEMP_DIR = os.path.join(tempfile.gettempdir(), "cleanimage")
os.makedirs(TEMP_DIR, exist_ok=True)

def download_image(url):
    """Télécharge une image depuis une URL et la convertit en numpy array."""
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return np.array(img)

def convert_to_pil_image(image):
    """
    Convertit différents formats d'image en PIL.Image.
    
    Args:
        image: Image au format PIL.Image, numpy.ndarray, chemin de fichier (str) ou dictionnaire Gradio
        
    Returns:
        PIL.Image.Image: Image convertie au format PIL
    """
    if image is None:
        return None
        
    # Si c'est déjà une image PIL, la retourner directement
    if isinstance(image, Image.Image):
        return image
        
    # Si c'est un tableau numpy
    elif isinstance(image, np.ndarray):
        # Vérifier s'il s'agit d'un masque binaire ou d'une image en niveaux de gris
        if len(image.shape) == 2:
            return Image.fromarray(image.astype(np.uint8), mode='L')
        # Images RGB/RGBA
        else:
            return Image.fromarray(image.astype(np.uint8))
            
    # Si c'est un chemin de fichier
    elif isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
        
    # Si c'est un dictionnaire (format possible depuis Gradio)
    elif isinstance(image, dict) and "path" in image:
        return Image.open(image["path"])
        
    # Si c'est une liste (parfois le cas avec certains modèles)
    elif isinstance(image, list) and len(image) > 0:
        # Tenter de convertir le premier élément
        return convert_to_pil_image(image[0])
        
    else:
        # En cas d'échec, lever une exception
        raise ValueError(f"Impossible de convertir l'image de type {type(image)} en PIL.Image")

def create_empty_mask(image_size):
    """
    Crée un masque vide (noir) de la taille spécifiée.
    
    Args:
        image_size: Tuple (largeur, hauteur) ou objet PIL.Image.Image
        
    Returns:
        PIL.Image.Image: Masque vide en mode 'L'
    """
    if isinstance(image_size, Image.Image):
        width, height = image_size.size
    elif isinstance(image_size, tuple) and len(image_size) == 2:
        width, height = image_size
    else:
        raise ValueError("image_size doit être un tuple (largeur, hauteur) ou un objet PIL.Image.Image")
    
    # Créer un masque noir (valeur 0)
    mask = Image.new('L', (width, height), 0)
    return mask

def visualize_mask(mask_img):
    """
    Crée une version colorée du masque pour une meilleure visualisation.
    
    Args:
        mask_img: Image PIL en mode L (niveaux de gris)
        
    Returns:
        np.ndarray: Image colorée du masque
    """
    if mask_img is None:
        return None
    
    # Convertir en tableau numpy si ce n'est pas déjà fait
    if isinstance(mask_img, Image.Image):
        mask_data = np.array(mask_img)
    else:
        mask_data = mask_img
    
    # Créer une image colorée
    if len(mask_data.shape) == 2:  # Masque en niveaux de gris
        # Normaliser entre 0 et 255 si nécessaire
        if mask_data.max() > 0:
            mask_data = (mask_data / mask_data.max() * 255).astype(np.uint8)
        
        # Créer une image RGB
        colored_mask = np.zeros((mask_data.shape[0], mask_data.shape[1], 3), dtype=np.uint8)
        
        # Zones blanches en rouge vif
        colored_mask[:, :, 0] = mask_data  # Canal rouge
        
        return colored_mask
    elif len(mask_data.shape) == 3 and mask_data.shape[2] == 3:  # Déjà en RGB
        return mask_data
    else:
        return None

def save_temp_image(image, prefix="img"):
    """
    Sauvegarde une image (PIL ou numpy) dans un fichier temporaire et retourne le chemin.
    
    Args:
        image: Image PIL ou tableau numpy
        prefix: Préfixe pour le nom du fichier
        
    Returns:
        str: Chemin du fichier temporaire
    """
    if image is None:
        return None
        
    # Générer un nom de fichier unique
    unique_id = uuid.uuid4()
    output_path = os.path.join(TEMP_DIR, f"{prefix}_{unique_id}.png")
    
    # Sauvegarder l'image
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(output_path)
    elif isinstance(image, Image.Image):
        image.save(output_path)
    else:
        raise ValueError(f"Type d'image non pris en charge: {type(image)}")
        
    return output_path

def add_watermark_to_image(image, watermark_text):
    # Créer une copie de l'image
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    # Définir la police et la taille
    font_size = int(min(img.size) * 0.05)  # 5% de la plus petite dimension
    font = ImageFont.truetype("arial.ttf", font_size)
    
    # Obtenir les dimensions du texte (utiliser textbbox au lieu de textsize)
    bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculer la position du texte (centré et en bas)
    x = (img.size[0] - text_width) // 2
    y = img.size[1] - text_height - 10  # 10 pixels du bas
    
    # Ajouter le texte avec une ombre légère pour la lisibilité
    # Ombre
    draw.text((x+2, y+2), watermark_text, font=font, fill=(0, 0, 0, 128))
    # Texte principal
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 192))
    
    return img 