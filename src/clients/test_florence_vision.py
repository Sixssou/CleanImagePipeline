from src.clients.florence_vision_client import FlorenceVisionClient
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def test_florence_vision():
    # Initialisation du client
    client = FlorenceVisionClient(
        hf_token=os.getenv('HF_TOKEN'),
        space_url=os.getenv('HF_SPACE_FLORENCE')
    )
    
    # URL de test
    test_image_url = "https://cdn.shopify.com/s/files/1/0898/8344/3543/files/image_60bd0406_cleaned.jpg?v=1739955469"
    
    try:
        # Test avec différents prompts
        prompts = [
            "OCR_WITH_REGION",      # Pour la détection de texte
        ]
        
        for prompt in prompts:
            print(f"\nTest avec prompt: {prompt}")
            try:
                result = client.analyze_image(
                    image_url=test_image_url,
                    prompt=prompt
                )
                print(f"Résultat pour {prompt}:")
                print(result)
            except Exception as e:
                print(f"Erreur pour {prompt}: {e}")
            
    except Exception as e:
        print(f"Erreur lors du test: {e}")

if __name__ == "__main__":
    test_florence_vision() 