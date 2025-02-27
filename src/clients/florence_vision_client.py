from gradio_client import Client
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

class FlorenceVisionClient:
    """Client for interacting with Florence-2 Vision model on HuggingFace Spaces"""
    
    def __init__(self, hf_token: str, space_url: str, timeout: int = 120):
        """
        Initialize the Florence Vision client.
        
        Args:
            hf_token (str): HuggingFace API token
            space_url (str): URL of the HuggingFace space
            timeout (int): Request timeout in seconds
        """
        self.token = hf_token
        self.space_url = space_url
        
        # Configure httpx client with timeout
        httpx_kwargs = {
            "timeout": httpx.Timeout(timeout, read=30.0)
        }
        
        # Initialize Gradio client
        self.client = Client(
            space_url,
            hf_token=hf_token,
            httpx_kwargs=httpx_kwargs
        )

    def analyze_image(self, image_url: str, prompt: str) -> Dict[str, Any]:
        """
        Analyze an image using Florence-2 model.
        
        Args:
            image_url (str): URL of the image to analyze
            prompt (str): Type of analysis (OCR_WITH_REGION, OBJECT_DETECTION, IMAGE_CAPTION)
        Returns:
            Dict[str, Any]: Model predictions and analysis results
        """
        try:
            result = self.client.predict(
                prompt,          # Type d'analyse (OCR_WITH_REGION, etc.)
                image_url,       # URL de l'image
                "",             # Prompt additionnel (vide pour l'instant)
                api_name="/process_image_from_url"
            )
            return result
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}") 