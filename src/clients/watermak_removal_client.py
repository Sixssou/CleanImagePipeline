from gradio_client import Client
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()

class WatermakRemovalClient:
    """Client for interacting with Florence-2 Vision model on HuggingFace Spaces"""
    
    def __init__(self, hf_token: str, space_url: str, timeout: int = 120):
        self.token = hf_token
        self.space_url = space_url
        
        # Configure httpx client with timeout
        httpx_kwargs = {
            "timeout": httpx.Timeout(timeout, read=120.0)
        }
        
        # Initialize Gradio client
        self.client = Client(
            space_url,
            hf_token=hf_token,
            httpx_kwargs=httpx_kwargs
        )

    def remove_bg(self, image_url: str):
        try:
            result = self.client.predict(
                image_url,
                api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_REMOVE_BACKGROUND_FROM_URL")
            )
            return result
        except Exception as e:
            raise Exception(f"Error analyzing image: {str(e)}")
    
    def detect_wm(self, 
                  image_url: str, 
                  threshold: float, 
                  max_bbox_percent: float, 
                  bbox_enlargement_factor: float):
        try:
            result = self.client.predict(
                image_url,
                threshold,
                max_bbox_percent,
                bbox_enlargement_factor,
                api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_DETECT_WATERMARKS_FROM_URL")
            )
            return result
        except Exception as e:
            raise Exception(f"Error detecting watermarks: {e}")

    def remove_wm(self, image_url, threshold, max_bbox_percent, 
                  remove_background_option, add_watermark_option, watermark):
        try:
            result = self.client.predict(
                image_url,
                threshold,
                max_bbox_percent,
                remove_background_option,
                add_watermark_option,
                watermark,
                api_name=os.getenv("HF_SPACE_WATERMAK_REMOVAL_ROUTE_REMOVE_WATERMARKS_FROM_URL")
            )
            return result
        except Exception as e:
            raise Exception(f"Error detecting watermarks: {str(e)}")
    