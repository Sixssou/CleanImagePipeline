#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 08:02:02 2025

@author: 6ssou
"""

# gsheets_lib.py

import gspread
from google.oauth2.service_account import Credentials

class GSheetClient:
    """
    Classe permettant de se connecter à une feuille Google Sheets 
    et d'effectuer des lectures/écritures.
    """

    def __init__(self, credentials_path: str, spreadsheet_name_or_id: str):
        """
        Initialise la connexion à l'API Google Sheets,
        et ouvre le document (soit par nom, soit par ID).
        
        :param credentials_path: Chemin du fichier JSON contenant les credentials
                                du compte de service.
        :param spreadsheet_name_or_id: Nom exact de la feuille (si elle est
                                       partagée) ou son ID unique.
        """
        # 1) Charger les credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = Credentials.from_service_account_file(credentials_path, scopes=scopes)

        # 2) Autoriser gspread
        self.client = gspread.authorize(creds)

        # 3) Ouvrir la feuille (vous pouvez utiliser open_by_key(...) si vous avez l'ID)
        #    ou open(...) si vous connaissez le nom exact
        #    => En général, si vous avez l'ID, on utilise open_by_key
        try:
            # Tenter d'ouvrir comme un ID
            self.spreadsheet = self.client.open_by_key(spreadsheet_name_or_id)
        except gspread.exceptions.APIError:
            # Si la feuille n'a pas été trouvée par ID, on tente par nom
            self.spreadsheet = self.client.open(spreadsheet_name_or_id)

    def test_connection(self, input_gheet_id: str, input_gheet_sheet: str) -> str:
        """
        Teste la connexion à la feuille Google Sheets et retourne un message
        indiquant si la connexion a réussi ou échoué.
        
        :param input_gheet_id: ID de la feuille Google Sheets
        :param input_gheet_sheet: Nom de l'onglet à tester
        :return: Message de succès ou d'échec de la connexion
        """
        try:
            self.spreadsheet = self.client.open_by_key(input_gheet_id)
            self.spreadsheet.worksheet(input_gheet_sheet)
            return f"Connexion réussie à la feuille '{input_gheet_sheet}' avec l'ID '{input_gheet_id}'."
        except gspread.exceptions.APIError as e:
            return f"Échec de la connexion à la feuille '{input_gheet_sheet}' avec l'ID '{input_gheet_id}': {e}"
        except Exception as e:
            return f"Une erreur inattendue est survenue lors de la connexion: {e}"

    def write_cells(self, worksheet_name: str, cell_range: str, values: list[list]):
        """
        Écrit les valeurs dans l'onglet / worksheet spécifié, 
        dans la plage cell_range ('A1', 'B2:C5', etc.).
        
        :param worksheet_name: Nom de l'onglet
        :param cell_range: Ex: 'A1', 'B2:D2', 'Feuille1!A1:B5', etc.
        :param values: Liste de listes, ex: [["val1", "val2"], ["val3", "val4"]]
        """
        ws = self.spreadsheet.worksheet(worksheet_name)
        # GSpread attend un format "update(range, [[...], [...]] )"
        ws.update(cell_range, values)

    def read_cells(self, worksheet_name: str, cell_range: str) -> list[list]:
        """
        Lit les valeurs d'une plage d'un onglet et renvoie
        une liste de listes de chaînes de caractères.
        
        :param worksheet_name: Nom de l'onglet
        :param cell_range: Ex: 'A1', 'B2:D2', etc.
        :return: Liste de listes (chaque sous-liste = une ligne)
        """
        ws = self.spreadsheet.worksheet(worksheet_name)
        return ws.get(cell_range)
    
    def append_row(self, worksheet_name: str, row_values: list):
        """
        Ajoute une ligne à la fin d'un onglet, dans la première 
        ligne vide disponible.
        
        :param worksheet_name: Nom de l'onglet
        :param row_values: ex: ["val1", "val2", "val3"]
        """
        ws = self.spreadsheet.worksheet(worksheet_name)
        ws.append_row(row_values)

    # Vous pouvez rajouter d'autres méthodes utilitaires si besoin.
