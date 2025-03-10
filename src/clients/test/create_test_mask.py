#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script pour générer un masque simple à partir d'une image.
Usage: python create_test_mask.py <chemin_image> [--rectangle x1 y1 x2 y2] [--circle x y radius]
"""

import os
import sys
import argparse
from PIL import Image, ImageDraw
import numpy as np

# Ajouter le répertoire parent au chemin de recherche des modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

def create_rectangle_mask(image_path, x1, y1, x2, y2, output_path=None):
    """
    Crée un masque rectangulaire pour une image.
    
    Args:
        image_path (str): Chemin vers l'image originale
        x1, y1 (int): Coordonnées du coin supérieur gauche du rectangle
        x2, y2 (int): Coordonnées du coin inférieur droit du rectangle
        output_path (str, optional): Chemin de sortie pour le masque. Si None, 
                                    utilise le nom de l'image avec "_mask" ajouté.
    
    Returns:
        str: Chemin vers le masque créé
    """
    # Charger l'image
    try:
        image = Image.open(image_path)
        width, height = image.size
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {str(e)}")
        sys.exit(1)
    
    # Créer un masque vide (noir)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Dessiner un rectangle blanc
    draw.rectangle([(x1, y1), (x2, y2)], fill=255)
    
    # Déterminer le chemin de sortie
    if output_path is None:
        # Créer le dossier de sortie
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../output'))
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_mask{ext}")
    
    # Sauvegarder le masque
    mask.save(output_path)
    print(f"Masque rectangulaire créé: {output_path}")
    
    return output_path

def create_circle_mask(image_path, x, y, radius, output_path=None):
    """
    Crée un masque circulaire pour une image.
    
    Args:
        image_path (str): Chemin vers l'image originale
        x, y (int): Coordonnées du centre du cercle
        radius (int): Rayon du cercle
        output_path (str, optional): Chemin de sortie pour le masque. Si None, 
                                    utilise le nom de l'image avec "_mask" ajouté.
    
    Returns:
        str: Chemin vers le masque créé
    """
    # Charger l'image
    try:
        image = Image.open(image_path)
        width, height = image.size
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {str(e)}")
        sys.exit(1)
    
    # Créer un masque vide (noir)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Dessiner un cercle blanc
    draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill=255)
    
    # Déterminer le chemin de sortie
    if output_path is None:
        # Créer le dossier de sortie
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../output'))
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_mask{ext}")
    
    # Sauvegarder le masque
    mask.save(output_path)
    print(f"Masque circulaire créé: {output_path}")
    
    return output_path

def create_auto_mask(image_path, output_path=None):
    """
    Crée un masque automatique au centre de l'image.
    
    Args:
        image_path (str): Chemin vers l'image originale
        output_path (str, optional): Chemin de sortie pour le masque. Si None, 
                                    utilise le nom de l'image avec "_mask" ajouté.
    
    Returns:
        str: Chemin vers le masque créé
    """
    # Charger l'image
    try:
        image = Image.open(image_path)
        width, height = image.size
    except Exception as e:
        print(f"Erreur lors du chargement de l'image: {str(e)}")
        sys.exit(1)
    
    # Calculer les dimensions du masque (25% de l'image au centre)
    center_x = width // 2
    center_y = height // 2
    mask_width = width // 4
    mask_height = height // 4
    
    x1 = center_x - mask_width // 2
    y1 = center_y - mask_height // 2
    x2 = center_x + mask_width // 2
    y2 = center_y + mask_height // 2
    
    # Créer un masque vide (noir)
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Dessiner un rectangle blanc
    draw.rectangle([(x1, y1), (x2, y2)], fill=255)
    
    # Déterminer le chemin de sortie
    if output_path is None:
        # Créer le dossier de sortie
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../output'))
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join(output_dir, f"{name}_mask{ext}")
    
    # Sauvegarder le masque
    mask.save(output_path)
    print(f"Masque automatique créé: {output_path}")
    print(f"Rectangle: ({x1}, {y1}) - ({x2}, {y2})")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Crée un masque simple pour une image")
    parser.add_argument("image_path", help="Chemin vers l'image originale")
    parser.add_argument("--output", help="Chemin de sortie pour le masque")
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rectangle", nargs=4, type=int, metavar=('X1', 'Y1', 'X2', 'Y2'),
                      help="Coordonnées du rectangle (x1 y1 x2 y2)")
    group.add_argument("--circle", nargs=3, type=int, metavar=('X', 'Y', 'RADIUS'),
                      help="Coordonnées du cercle (x y radius)")
    group.add_argument("--auto", action="store_true",
                      help="Crée un masque automatique au centre de l'image")
    
    args = parser.parse_args()
    
    # Vérifier que l'image existe
    if not os.path.exists(args.image_path):
        print(f"Le fichier image {args.image_path} n'existe pas")
        sys.exit(1)
    
    # Créer le masque selon les arguments
    if args.rectangle:
        x1, y1, x2, y2 = args.rectangle
        create_rectangle_mask(args.image_path, x1, y1, x2, y2, args.output)
    elif args.circle:
        x, y, radius = args.circle
        create_circle_mask(args.image_path, x, y, radius, args.output)
    elif args.auto:
        create_auto_mask(args.image_path, args.output)
    else:
        # Par défaut, créer un masque automatique
        create_auto_mask(args.image_path, args.output)

if __name__ == "__main__":
    main() 