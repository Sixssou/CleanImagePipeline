from src.clients.gsheet_client import GSheetClient
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

def update_split_urls():
    # 1) Initialiser client GSheet
    credentials_file = os.getenv("GSHEET_CREDENTIALS_FILE")
    spreadsheet_id = os.getenv("GSHEET_ID")
    sheet_client = GSheetClient(credentials_file, spreadsheet_id)

    # 2) Lire toutes les lignes Y2:Y (pour localiser la ligne d'old_url)
    #    => renvoie un tableau list[list], ex: [["https://..."], ["https://..."], ...]
    #    Adaptez la plage à vos besoins (ex. "Y2:Y3000" si vous savez la taille).
    split_rows = sheet_client.read_cells(os.getenv("GSHEET_SHEET_SPLIT_TAB_NAME"), "Y2:Y")
    # On ne sait pas combien de lignes on obtient, on en déduira l'index quand on cherche.
    logger.info(f"Dans l'onglet {os.getenv('GSHEET_SHEET_SPLIT_TAB_NAME')}, on a récupéré {len(split_rows)} lignes pour la colonne Y.")

    # 3) On va lire PhotosMacProcess et PhotosMatrixProcess
    #    => on suppose la plage A2:B car A=old_url, B=new_url
    mac_rows = sheet_client.read_cells(os.getenv("GSHEET_SHEET_PHOTOS_MAC_TAB_NAME"), "A2:B")
    matrix_rows = sheet_client.read_cells(os.getenv("GSHEET_SHEET_PHOTOS_MATRIX_TAB_NAME"), "A2:B")

    logger.info(f"Lignes Mac = {len(mac_rows)}, Lignes Matrix = {len(matrix_rows)}")

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

    logger.info(f"Nombre total de paires old/new URLs à traiter = {len(all_rows)}")

    if not all_rows:
        logger.warning("Aucune paire à traiter, on s'arrête.")
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
            sheet_client.write_cells(os.getenv("GSHEET_SHEET_SPLIT_TAB_NAME"), cell_ref, [[new_url]])
            logger.info(f"[OK] {old_url} => {new_url} (mis à jour dans {os.getenv('GSHEET_SHEET_SPLIT_TAB_NAME')}:{cell_ref})")
            cpt += 1
        else:
            logger.warning(f"[SKIP] {old_url} non trouvé dans la colonne Y de {os.getenv('GSHEET_SHEET_SPLIT_TAB_NAME')}")

    logger.info(f"Script terminé, {cpt} lignes mises à jour.")


if __name__ == "__main__":
    update_split_urls()
