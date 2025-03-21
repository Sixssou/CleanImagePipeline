import requests
import mimetypes
import uuid
import time
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

class ShopifyClient:
    """Client for interacting with Shopify API"""

    def __init__(self, shopify_api_version, shopify_access_token, shopify_store_name):
        """
        Initialize the Shopify client.

        Args:
            shop_name (str): The name of the Shopify shop
            access_token (str): The access token for the Shopify API
        """
        self.shopify_api_version = shopify_api_version
        self.shopify_access_token = shopify_access_token
        self.shopify_store_name = shopify_store_name
        self.graphql_url = f"https://{shopify_store_name}/admin/api/{shopify_api_version}/graphql.json"

        logger.info(f"Shopify API version: {self.shopify_api_version}")
        logger.info(f"Shopify Access Token: {self.shopify_access_token}")
        logger.info(f"Shopify Store Name: {self.shopify_store_name}")
        logger.info(f"GraphQL URL: {self.graphql_url}")


    def get_mime_type(self, file_path: str) -> str:
        """Détermine le MIME type (image/jpeg, image/png...) pour l'upload S3."""
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

    def staged_uploads_create(self, file_paths):
        """
        Appelle la mutation GraphQL `stagedUploadsCreate` pour obtenir les URL
        de pré-upload (S3) associées à chaque fichier.
        Retourne la liste des cibles de staging (url + paramètres),
        ou None en cas d'erreur.
        """
        inputs = []
        for fp in file_paths:
            mime_type = self.get_mime_type(fp)
            inputs.append({
                "resource": "FILE",
                "filename": os.path.basename(fp),
                "mimeType": mime_type,
                "httpMethod": "POST",
            })

        query = """
        mutation stagedUploadsCreate($input: [StagedUploadInput!]!) {
        stagedUploadsCreate(input: $input) {
            stagedTargets {
            url
            resourceUrl
            parameters {
                name
                value
            }
            }
            userErrors {
            field
            message
            }
        }
        }
        """

        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": self.shopify_access_token
        }

        try:
            resp = requests.post(
                self.graphql_url,
                headers=headers,
                json={"query": query, "variables": {"input": inputs}},
                timeout=30
            )
            data = resp.json()
            if "errors" in data:
                logger.error(f"Erreur GraphQL stagedUploadsCreate: {data['errors']}")
                return None

            user_errors = data["data"]["stagedUploadsCreate"]["userErrors"]
            if user_errors:
                logger.error(f"UserErrors (stagedUploadsCreate): {user_errors}")
                return None

            staged_targets = data["data"]["stagedUploadsCreate"]["stagedTargets"]
            return staged_targets

        except Exception as e:
            logger.error(f"Exception lors de stagedUploadsCreate: {e}")
            return None

    ### AJOUT / MODIF ###
    def file_create_no_preview(self, staged_target: dict, alt_text: str) -> bool:
        """
        Appelle la mutation `fileCreate` pour déclarer le fichier
        (sans demander tout de suite la preview).
        Retourne True si OK, False sinon.
        """
        query = """
        mutation fileCreate($files: [FileCreateInput!]!) {
        fileCreate(files: $files) {
            files {
            alt
            createdAt
            }
            userErrors {
            field
            message
            }
        }
        }
        """
        files_input = [{
            "originalSource": staged_target["resourceUrl"],
            "alt": alt_text
        }]

        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": self.shopify_access_token
        }

        try:
            resp = requests.post(
                self.graphql_url,
                headers=headers,
                json={"query": query, "variables": {"files": files_input}},
                timeout=30
            )
            data = resp.json()

            if "errors" in data:
                logger.error(f"Erreur GraphQL fileCreate: {data['errors']}")
                return False

            user_errors = data["data"]["fileCreate"]["userErrors"]
            if user_errors:
                logger.error(f"UserErrors (fileCreate): {user_errors}")
                return False

            created_files = data["data"]["fileCreate"]["files"]
            if not created_files:
                logger.error("fileCreate: aucun fichier créé.")
                return False

            logger.info(f"[INFO] fileCreate OK, alt={alt_text}")
            return True

        except Exception as e:
            logger.error(f"Exception lors de fileCreate: {e}")
            return False

    def retrieve_file_by_alt(self, alt_value: str) -> str:
        """
        Recherche dans les fichiers Shopify (jusqu'à 100 résultats), 
        triés par date de création (les plus récents en premier),
        celui dont le champ alt = alt_value.
        Retourne l'URL preview.image.url si disponible, sinon "".
        Ajoute des logs détaillés pour inspecter la requête et la réponse.
        """

        # --------------------
        # 1) Définition de la requête et des variables
        # --------------------
        query = """
        query getFiles($first: Int!, $sortKey: FileSortKeys!, $reverse: Boolean!) {
        files(
            first: $first,
            sortKey: $sortKey,
            reverse: $reverse
        ) {
            nodes {
            alt
            createdAt
            preview {
                image {
                url
                }
            }
            }
        }
        }
        """
        variables = {
            "first": 1,
            "sortKey": "CREATED_AT",
            "reverse": True
        }

        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": self.shopify_access_token
        }

        try:
            # --------------------
            # 2) Exécution de la requête GraphQL
            # --------------------
            resp = requests.post(
                self.graphql_url,
                headers=headers,
                json={"query": query, "variables": variables},
                timeout=30
            )

            # Parse le JSON
            data = resp.json()

            # --------------------
            # 3) Vérification d'éventuelles erreurs
            # --------------------
            if "errors" in data:
                logger.error(f"[ERROR] retrieve_file_by_alt: {data['errors']}")
                return ""

            # Log de la structure JSON complète
            logger.debug(f"[DEBUG] Data parsée: {data}")

            # --------------------
            # 4) Lecture des noeuds "files"
            # --------------------
            nodes = data["data"]["files"]["nodes"]
            logger.debug(f"[DEBUG] nodes récupérés: {nodes}")

            # On parcourt chaque node
            for file_node in nodes:
                logger.debug(f"[DEBUG] file_node: {file_node}")

                # Vérifie si l'alt correspond
                if file_node["alt"] == alt_value:
                    preview = file_node.get("preview")
                    if preview and preview.get("image"):
                        url_val = preview["image"].get("url", "")
                        if url_val:
                            logger.info(f"[INFO] Fichier correspondant trouvé, url = {url_val}")
                            return url_val
                        else:
                            logger.warning("[WARNING] preview.image.url est vide.")
                    else:
                        logger.warning("[WARNING] Pas de preview/image pour ce file_node.")

            # Si on n'a pas trouvé de correspondance
            logger.info(f"[INFO] Aucun fichier ne correspond à alt={alt_value} ou pas d'URL de preview.")
            return ""

        except Exception as e:
            logger.error(f"[ERROR] retrieve_file_by_alt Exception: {e}")
            return ""

    def upload_file_to_shopify(self, cleaned_image_path: str) -> str:
        """
        Effectue l'upload du fichier `cleaned_image_path` dans l'onglet "Fichiers" Shopify,
        en deux étapes :
        1) stagedUploadsCreate + POST S3
        2) fileCreate (sans preview)
        3) retrieve_file_by_alt (plusieurs tentatives)
        Retourne l'URL finale ou "" si échec.
        """
        logger.info(f"Uploading file to Shopify: {cleaned_image_path}")
        staged_targets = self.staged_uploads_create([cleaned_image_path])
        if not staged_targets or len(staged_targets) < 1:
            return ""

        staged_target = staged_targets[0]

        # Paramètres à passer dans le POST (S3)
        form_data = {}
        for p in staged_target["parameters"]:
            form_data[p["name"]] = p["value"]

        # On ouvre le fichier en binaire
        try:
            with open(cleaned_image_path, "rb") as f:
                files_to_upload = {
                    'file': (os.path.basename(cleaned_image_path), f)
                }
                resp = requests.post(
                    staged_target["url"],
                    data=form_data,
                    files=files_to_upload,
                    timeout=60
                )
            # On accepte 201 ou 204 comme code de succès
            if resp.status_code not in (201, 204):
                logger.error(f"Erreur lors de l'upload S3 (status={resp.status_code}): {resp.text}")
                return ""
        except Exception as e:
            logger.error(f"Exception pendant l'upload S3: {e}")
            return ""

        ### AJOUT / MODIF ###
        # 2) fileCreate (sans preview) + alt unique
        unique_alt = f"image_{uuid.uuid4()}"
        success = self.file_create_no_preview(staged_target, unique_alt)
        if not success:
            # Echec de fileCreate => on arrête
            return ""

        # 3) Faire plusieurs tentatives pour récupérer l'URL (preview.image.url)
        final_url = ""
        for attempt in range(10):
            logger.info(f"Tentative de récupération de l'URL (alt={unique_alt}), essai {attempt+1}/5")
            final_url = self.retrieve_file_by_alt(unique_alt)
            if final_url:
                break
            time.sleep(2)  # Attendre 2s avant de réessayer

        if final_url:
            logger.info(f"Image envoyée et créée avec succès : {final_url}")
        else:
            logger.warning(f"Preview introuvable après 5 essais (alt={unique_alt})")

        return final_url

    def test_connection(self, 
                        input_shopify_domain: str, 
                        input_shopify_api_version: str, 
                        input_shopify_api_key: str) -> str:
        """
        Test the connection to the Shopify API and return a message
        indicating whether the connection was successful or not.

        Returns:
            str: Success or error message.
        """
        url = f"https://{input_shopify_domain}/admin/api/{input_shopify_api_version}/shop.json"
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": input_shopify_api_key
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return "Connexion réussie à l'API Shopify."
            else:
                return f"Échec de la connexion à l'API Shopify: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Erreur lors de la tentative de connexion à l'API Shopify: {e}"
    
    def upload_file(self, local_file_path: str, file_name: str = None) -> str:
        """
        Télécharge un fichier sur Shopify et retourne l'URL
        
        Args:
            local_file_path: Chemin local du fichier à télécharger
            file_name: Nom du fichier à utiliser (facultatif, utilise le nom original si non fourni)
            
        Returns:
            str: URL du fichier téléchargé sur Shopify ou chaîne vide en cas d'erreur
        """
        try:
            from loguru import logger
            logger.info(f"Téléchargement du fichier {local_file_path} sur Shopify")
            
            # Utiliser la méthode existante pour télécharger le fichier
            shopify_url = self.upload_file_to_shopify(local_file_path)
            
            if shopify_url:
                logger.info(f"Fichier téléchargé avec succès: {shopify_url}")
                return shopify_url
            else:
                logger.error(f"Échec du téléchargement du fichier {local_file_path}")
                return ""
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement du fichier sur Shopify: {str(e)}")
            return ""