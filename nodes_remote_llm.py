import base64
import io
import json
import requests
from PIL import Image
import torch
import numpy as np


def tensor_to_pil(image_tensor, batch_index=0) -> Image.Image:
    """Convert ComfyUI tensor to PIL Image."""
    # Extract single image from batch
    image_tensor = image_tensor[batch_index]
    
    # Perform scaling and clamping on GPU before CPU transfer
    img_data = (image_tensor * 255.0).clamp(0, 255).byte()
    
    # Single CPU transfer
    img_np = img_data.cpu().numpy()
    
    # Convert to PIL
    img = Image.fromarray(img_np)
    return img


def pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


class SCGRemoteLLMVLM_OAI:
    """
    Remote LLM/VLM Node using OpenAI-compatible API standard.
    
    Compatible with:
    - OpenAI API
    - LM Studio local server
    - Any OpenAI-compatible endpoint
    
    Supports:
    - Up to 4 optional input images for vision models
    - Text-only mode for LLM inference
    - Full control over generation parameters
    - Custom API endpoints and models
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": (
                    "STRING",
                    {
                        "default": "http://localhost:1234/v1/chat/completions",
                        "multiline": False,
                    },
                ),
                "model_name": (
                    "STRING",
                    {
                        "default": "qwen2-vl-7b-instruct",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "lm-studio",
                        "multiline": False,
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "max_tokens": (
                    "INT",
                    {"default": 512, "min": 1, "max": 8000, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": 42},
                ),
                "bypass": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "scg-utils/llm"
    
    def generate(
        self,
        api_url,
        model_name,
        api_key,
        system_prompt,
        user_prompt,
        temperature,
        top_p,
        max_tokens,
        seed,
        bypass,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
    ):
        """
        Generate text using OpenAI-compatible API.
        
        Args:
            api_url: API endpoint URL
            model_name: Model identifier
            api_key: API key for authentication
            system_prompt: System message for context
            user_prompt: User message/query
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
            seed: Seed value for ComfyUI caching control (not sent to API)
            bypass: If True, return user_prompt directly
            image1-4: Optional image inputs
        
        Returns:
            Tuple containing the generated text response
        """
        # Bypass mode: return user prompt without API call
        if bypass:
            print("[SCG Remote LLM/VLM] Bypass mode enabled, returning user_prompt")
            return (user_prompt,)
        
        # Validate inputs
        if not api_url.strip():
            return ("Error: api_url is required.",)
        
        if not model_name.strip():
            return ("Error: model_name is required.",)
        
        try:
            # Build user content - process images first, then add text
            user_content = []
            
            # Process images in order
            images = [image1, image2, image3, image4]
            image_count = 0
            for idx, img in enumerate(images, start=1):
                if img is not None:
                    print(f"[SCG Remote LLM/VLM] Processing image{idx}")
                    pil_image = tensor_to_pil(img)
                    base64_image = pil_to_base64(pil_image)
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    })
                    image_count += 1
            
            if image_count > 0:
                print(f"[SCG Remote LLM/VLM] Total images added: {image_count}")
            
            # Add text prompt
            user_content.append({
                "type": "text",
                "text": user_prompt
            })
            
            # Build messages array
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            # Build request payload
            payload = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            
            # Build headers
            headers = {
                "Content-Type": "application/json",
            }
            
            # Add API key if provided (some local servers don't require it)
            if api_key and api_key.strip() and api_key.strip() != "lm-studio":
                headers["Authorization"] = f"Bearer {api_key}"
            
            print(f"[SCG Remote LLM/VLM] Sending request to {api_url}")
            print(f"[SCG Remote LLM/VLM] Model: {model_name}")
            print(f"[SCG Remote LLM/VLM] Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")
            
            # Make API request
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=120  # 2 minute timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract text from OpenAI-format response
            if "choices" in response_data and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                
                if "message" in choice and "content" in choice["message"]:
                    result_text = choice["message"]["content"]
                    
                    # Log usage info if available
                    if "usage" in response_data:
                        usage = response_data["usage"]
                        print(f"[SCG Remote LLM/VLM] Token usage: {usage}")
                    
                    print(f"[SCG Remote LLM/VLM] Response received successfully")
                    return (result_text,)
                else:
                    return (f"Error: Unexpected response format - no message content. Response: {json.dumps(response_data, indent=2)}",)
            else:
                return (f"Error: Unexpected response format - no choices. Response: {json.dumps(response_data, indent=2)}",)
        
        except requests.exceptions.Timeout:
            return ("Error: Request timed out after 120 seconds.",)
        
        except requests.exceptions.ConnectionError as e:
            return (f"Error: Connection failed. Is the server running? Details: {str(e)}",)
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"Error: HTTP {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f" - {json.dumps(error_detail, indent=2)}"
            except:
                error_msg += f" - {response.text}"
            return (error_msg,)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[SCG Remote LLM/VLM] Error during API call:\n{error_details}")
            return (f"Error during API call: {str(e)}",)
