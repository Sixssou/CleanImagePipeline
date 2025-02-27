#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script : update_split_urls.py

Objectif :
- Lire dans PhotosMacProcess et PhotosMatrixProcess (col. A=old_url, col. B=new_url).
- Pour chaque old_url, chercher la ligne correspondante dans SPLIT (col. Y).
- Écraser la même colonne Y de cette ligne par new_url.

Notes :
- On fait une recherche linéaire par boucle pour chaque URL (simple, pas optimisé).
- Adapter la plage Y2:Y si besoin (ex. Y2:Y1000 ou Y2:Y si on ignore la fin).
- Si la colonne Y est la 25e colonne, on peut faire un update de la cellule
  "Y{row_number}" pour y mettre new_url.
"""

import sys
from src.clients.gsheet_client import GSheetClient

# Noms d'onglets
PHOTOS_MAC_TAB = "PhotosMacProcess"
PHOTOS_MATRIX_TAB = "PhotosMatrixProcess"
SPLIT_TAB = "SPLIT"

def log_message(msg: str, level="INFO"):
    """Simple fonction de log."""
    print(f"[{level}] {msg}")

def update_split_urls():
    # 1) Initialiser client GSheet
    credentials_file = "/Users/6ssou/Dev/Ecom/WatermarkRemover-AI/credentialsGapi.json"
    spreadsheet_id = "1U31mI849aYXrW4caG2G_ovmmubk6_ey4WeZbm-t7dk4"
    sheet_client = GSheetClient(credentials_file, spreadsheet_id)

    # 2) Lire toutes les lignes Y2:Y (pour localiser la ligne d'old_url)
    #    => renvoie un tableau list[list], ex: [["https://..."], ["https://..."], ...]
    #    Adaptez la plage à vos besoins (ex. "Y2:Y3000" si vous savez la taille).
    split_rows = sheet_client.read_cells(SPLIT_TAB, "Y2:Y")
    # On ne sait pas combien de lignes on obtient, on en déduira l'index quand on cherche.
    log_message(f"Dans l'onglet {SPLIT_TAB}, on a récupéré {len(split_rows)} lignes pour la colonne Y.", "INFO")

    # 3) On va lire PhotosMacProcess et PhotosMatrixProcess
    #    => on suppose la plage A2:B car A=old_url, B=new_url
    mac_rows = sheet_client.read_cells(PHOTOS_MAC_TAB, "A2:B")
    matrix_rows = sheet_client.read_cells(PHOTOS_MATRIX_TAB, "A2:B")

    log_message(f"Lignes Mac = {len(mac_rows)}, Lignes Matrix = {len(matrix_rows)}", "INFO")

    # 4) Fusionner ces deux listes
    #    => On obtiendra un tableau du style [ [old_url, new_url], ... ]
    all_rows = []
    # D'abord Mac
    for row in mac_rows:
        if row and len(row) >= 2:
            old_url = row[0].strip() if row[0] else ""
            new_url = row[1].strip() if row[1] else ""
            if old_url and new_url:
                all_rows.append((old_url, new_url))
    # Puis Matrix
    for row in matrix_rows:
        if row and len(row) >= 2:
            old_url = row[0].strip() if row[0] else ""
            new_url = row[1].strip() if row[1] else ""
            if old_url and new_url:
                all_rows.append((old_url, new_url))

    log_message(f"Nombre total de paires old/new URLs à traiter = {len(all_rows)}", "INFO")

    if not all_rows:
        log_message("Aucune paire à traiter, on s'arrête.", "WARNING")
        return

    # 5) Pour chaque (old_url, new_url), on cherche la ligne correspondante dans split_rows
    #    => On effectue une recherche linéaire. 
    #       index  0 => row[0] = "https://..."
    #       On veut old_url == row[0]
    #       Ensuite on fait "Y{index+2}" pour cibler la bonne ligne
    #       et on fait un update
    cpt = 0
    for (old_url, new_url) in all_rows:
        if not old_url:
            continue

        found_idx = None
        for i, splitted in enumerate(split_rows):
            # splitted est du style ["https://..."] 
            if splitted and len(splitted) > 0 and splitted[0].strip() == old_url:
                found_idx = i
                break

        if found_idx is not None:
            # On calcule la ligne Google Sheets => i + 2 (car i=0 correspond à A2)
            row_number = found_idx + 2  
            # ex: "Y{row_number}"
            cell_ref = f"Y{row_number}"
            # On met new_url dedans
            sheet_client.write_cells(SPLIT_TAB, cell_ref, [[new_url]])
            log_message(f"[OK] {old_url} => {new_url} (mis à jour dans {SPLIT_TAB}:{cell_ref})", "INFO")
            cpt += 1
        else:
            log_message(f"[SKIP] {old_url} non trouvé dans la colonne Y de {SPLIT_TAB}", "WARNING")

    log_message(f"Script terminé, {cpt} lignes mises à jour.", "INFO")


if __name__ == "__main__":
    update_split_urls()
