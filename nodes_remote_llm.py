import base64
import json
import os
import time
from io import BytesIO
from urllib.parse import urlparse
import requests
from PIL import Image
from comfy_api.latest import io


MAX_VISION_EDGE = 2048


def tensor_to_pil(image_tensor, batch_index=0) -> Image.Image:
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[batch_index]
    elif image_tensor.ndim != 3:
        raise ValueError(f"expected image tensor HWC or BHWC, got {tuple(image_tensor.shape)}")
    img_data = (image_tensor * 255.0).clamp(0, 255).byte()
    img_np = img_data.cpu().numpy()
    if img_np.ndim == 3 and img_np.shape[-1] == 1:
        img_np = img_np[..., 0]
    return Image.fromarray(img_np)


def letterbox_square(pil_image: Image.Image) -> Image.Image:
    w, h = pil_image.size
    if w == h:
        return pil_image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(pil_image, ((side - w) // 2, (side - h) // 2))
    return canvas


def fit_vision_image(pil_image: Image.Image, pad_input: bool) -> Image.Image:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    if pad_input:
        pil_image = letterbox_square(pil_image)
    w, h = pil_image.size
    edge = max(w, h)
    if edge > MAX_VISION_EDGE:
        scale = MAX_VISION_EDGE / float(edge)
        pil_image = pil_image.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))),
            Image.LANCZOS,
        )
    return pil_image


def pil_to_base64(pil_image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    pil_image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


DEFAULT_API_URL = "http://localhost:1234/api/v1/chat"
_API_KEY_ENV_NAMES = ("SCG_LMSTUDIO_API_KEY", "LM_API_TOKEN")
_API_KEY_UNSET = {"", "lm-studio"}


def api_origin(url):
    parsed = urlparse(url if "://" in url else "http://" + url)
    return f"{parsed.scheme}://{parsed.netloc}"


def _dotenv_values(path):
    values = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                if key:
                    values[key] = value
    except OSError:
        return {}
    return values


def resolve_api_key(api_key):
    typed = (api_key or "").strip()
    if typed.lower() not in _API_KEY_UNSET:
        return typed, "node input"
    for name in _API_KEY_ENV_NAMES:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value, f"environment {name}"
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    file_values = _dotenv_values(env_path)
    for name in _API_KEY_ENV_NAMES:
        value = (file_values.get(name) or "").strip()
        if value:
            return value, f"{env_path} {name}"
    return "", ""


def request_headers(api_key):
    headers = {"Content-Type": "application/json"}
    key, source = resolve_api_key(api_key)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers, source


def loaded_lmstudio_model(origin, headers):
    response = requests.get(f"{origin}/api/v1/models", headers=headers, timeout=10)
    if response.status_code == 401 and "Authorization" not in headers:
        raise requests.HTTPError(
            "401 Unauthorized from LM Studio. Set SCG_LMSTUDIO_API_KEY in custom_nodes/scg-utils/.env "
            "or export SCG_LMSTUDIO_API_KEY / LM_API_TOKEN. The api_key input overrides that for a one-off test.",
            response=response,
        )
    response.raise_for_status()
    for model in response.json().get("models") or []:
        instances = model.get("loaded_instances") or []
        if instances:
            return instances[0].get("id") or model.get("key")
    return None


def resolve_api_url(api_url):
    """LM Studio uses /api/v1/chat with store=false. OpenAI uses chat/completions."""
    origin = api_origin(api_url.strip())
    if "openai.com" in api_url.lower():
        return f"{origin}/v1/chat/completions", "chat"
    return f"{origin}/api/v1/chat", "lmstudio_chat"


def extract_choice_text(response_data):
    choices = response_data.get("choices") or []
    if choices:
        choice = choices[0]
        if choice.get("text") is not None:
            return choice["text"]
        message = choice.get("message")
        if isinstance(message, dict) and message.get("content") is not None:
            return message["content"]

    output = response_data.get("output")
    if isinstance(output, list):
        messages = []
        reasoning = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not content:
                continue
            if item.get("type") == "message":
                messages.append(content)
            elif item.get("type") == "reasoning":
                reasoning.append(content)
        if messages:
            return "\n".join(messages)
        if reasoning:
            return "\n".join(reasoning)
    return None


MAX_IMAGES = 10


def image_slot_index(name):
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return int(digits) if digits else 0


def collect_image_urls(images, extra=None, pad_input=False):
    slots = dict(images or {})
    for name, batch in (extra or {}).items():
        if batch is None or not str(name).startswith("image"):
            continue
        slots[name] = batch
    urls = []
    fit_label = "padded" if pad_input else "as-is"
    for name in sorted(slots, key=image_slot_index):
        batch = slots[name]
        if batch is None:
            continue
        pil = tensor_to_pil(batch)
        orig_w, orig_h = pil.size
        sent = fit_vision_image(pil, pad_input)
        print(
            f"[SCG Remote LLM/VLM] {name}: {orig_w}x{orig_h} -> {sent.size[0]}x{sent.size[1]} ({fit_label})"
        )
        urls.append(pil_to_base64(sent))
    return urls


class SCGRemoteLLMVLM_OAI(io.ComfyNode):
    """Remote OpenAI-compatible LLM/VLM. Image sockets autogrow up to 10, same as Luma API LLM/VLM."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCGRemoteLLMVLM_OAI",
            display_name="SCG Remote LLM/VLM - OAI Standard",
            category="scg-utils/llm",
            description="Chat with an OpenAI-compatible LLM/VLM. Image sockets autogrow up to 10.",
            inputs=[
                io.String.Input("api_url", default=DEFAULT_API_URL),
                io.String.Input("model_name", default="qwen2-vl-7b-instruct"),
                io.String.Input(
                    "api_key",
                    default="",
                    tooltip="Leave empty to use SCG_LMSTUDIO_API_KEY or LM_API_TOKEN from the environment or custom_nodes/scg-utils/.env. A value here overrides that and is saved in the workflow.",
                ),
                io.String.Input(
                    "system_prompt",
                    default="You are a helpful assistant.",
                    multiline=True,
                ),
                io.String.Input("user_prompt", default="", multiline=True),
                io.Autogrow.Input(
                    "images",
                    optional=True,
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image{i}" for i in range(1, MAX_IMAGES + 1)],
                        min=0,
                    ),
                    tooltip=f"Images sent in socket order. Extra slots appear as you connect them, up to {MAX_IMAGES}.",
                ),
                io.Float.Input("temperature", default=0.7, min=0.0, max=2.0, step=0.05),
                io.Float.Input("top_p", default=0.9, min=0.0, max=1.0, step=0.05),
                io.Int.Input("max_tokens", default=512, min=1, max=8000, step=1),
                io.Int.Input("seed", default=42, control_after_generate=True),
                io.Boolean.Input("bypass", default=False),
                io.Int.Input("max_retries", default=3, min=0, max=10, step=1),
                io.Int.Input(
                    "timeout",
                    default=600,
                    min=1,
                    max=36000,
                    step=1,
                    tooltip="HTTP timeout in seconds for each API attempt",
                ),
                io.Combo.Input(
                    "reasoning",
                    options=["default", "off", "on", "low", "medium", "high"],
                    default="default",
                    tooltip="LM Studio reasoning. default uses the model's automatic setting. off disables thinking. low/medium/high only work on models that advertise them.",
                ),
                io.Boolean.Input(
                    "pad_input",
                    default=False,
                    tooltip="Pad tall/wide images to a square before sending so CLIP-style VLMs do not center-crop the edges. Off by default; neither setting crops.",
                ),
            ],
            outputs=[io.String.Output(display_name="response")],
        )

    @classmethod
    def execute(
        cls,
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
        max_retries=3,
        timeout=600,
        reasoning="default",
        pad_input=False,
        images=None,
        **kwargs,
    ):
        if bypass:
            print("[SCG Remote LLM/VLM] Bypass mode enabled, returning user_prompt")
            return io.NodeOutput(user_prompt)

        if not api_url.strip():
            return io.NodeOutput("Error: api_url is required.")

        try:
            try:
                max_retries = int(max_retries)
            except (TypeError, ValueError):
                max_retries = 3
            max_retries = max(0, max_retries)

            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = 600
            timeout = max(1, timeout)

            image_urls = collect_image_urls(images, kwargs, pad_input=pad_input)
            if image_urls:
                print(f"[SCG Remote LLM/VLM] Total images added: {len(image_urls)}")

            api_url, api_mode = resolve_api_url(api_url)
            headers, key_source = request_headers(api_key)
            if key_source:
                print(f"[SCG Remote LLM/VLM] Using API key from {key_source}")

            payload = {
                "temperature": temperature,
                "top_p": top_p,
            }
            model_name = model_name.strip()

            if api_mode == "lmstudio_chat":
                if image_urls:
                    native_input = [{"type": "image", "data_url": url} for url in image_urls]
                    native_input.append({"type": "text", "content": user_prompt})
                    payload["input"] = native_input
                else:
                    payload["input"] = user_prompt
                payload["store"] = False
                payload["max_output_tokens"] = max_tokens
                if system_prompt and system_prompt.strip():
                    payload["system_prompt"] = system_prompt.strip()
                if reasoning and reasoning != "default":
                    payload["reasoning"] = reasoning
                if not model_name:
                    loaded = loaded_lmstudio_model(api_origin(api_url), headers)
                    if loaded:
                        model_name = loaded
                        print(f"[SCG Remote LLM/VLM] Using loaded model {model_name}")
                print("[SCG Remote LLM/VLM] Fresh request (store=false)")
            elif api_mode == "chat":
                user_content = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]
                user_content.append({"type": "text", "text": user_prompt})
                payload["messages"] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]
                payload["max_tokens"] = max_tokens
                payload["store"] = False

            if model_name:
                payload["model"] = model_name

            print(f"[SCG Remote LLM/VLM] Sending request to {api_url}")
            print(f"[SCG Remote LLM/VLM] Model: {model_name or 'server default'} ({api_mode})")
            print(f"[SCG Remote LLM/VLM] Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}, Timeout: {timeout}s, Reasoning: {reasoning}")
            
            total_attempts = max_retries + 1
            for attempt in range(total_attempts):
                attempt_label = f"{attempt + 1}/{total_attempts}"
                if total_attempts > 1:
                    print(f"[SCG Remote LLM/VLM] API attempt {attempt_label}")
                
                try:
                    response = requests.post(
                        api_url,
                        headers=headers,
                        json=payload,
                        timeout=timeout,
                    )
                    
                    # Check for HTTP errors
                    response.raise_for_status()
                    
                    # Parse response
                    response_data = response.json()
                    result_text = extract_choice_text(response_data)
                    if result_text is not None:
                        if "usage" in response_data:
                            print(f"[SCG Remote LLM/VLM] Token usage: {response_data['usage']}")
                        print(f"[SCG Remote LLM/VLM] Response received successfully")
                        return io.NodeOutput(result_text)
                    return io.NodeOutput(f"Error: Unexpected response format. Response: {json.dumps(response_data, indent=2)}")
                
                except requests.exceptions.Timeout:
                    error_msg = f"Error: Request timed out after {timeout} seconds."
                    should_retry = True
                
                except requests.exceptions.ConnectionError as e:
                    error_msg = f"Error: Connection failed. Is the server running? Details: {str(e)}"
                    should_retry = True
                
                except requests.exceptions.HTTPError as e:
                    response = e.response
                    status_code = response.status_code if response is not None else "unknown"
                    error_msg = f"Error: HTTP {status_code}"
                    if response is not None:
                        try:
                            error_detail = response.json()
                            error_msg += f" - {json.dumps(error_detail, indent=2)}"
                        except Exception:
                            error_msg += f" - {response.text}"
                    else:
                        error_msg += f" - {str(e)}"
                    error_text = error_msg.lower()
                    should_retry = (
                        isinstance(status_code, int)
                        and (
                            status_code >= 500
                            or (
                                status_code == 400
                                and (
                                    "model has crashed" in error_text
                                    or "model reloaded" in error_text
                                )
                            )
                        )
                    )
                
                except requests.exceptions.RequestException as e:
                    error_msg = f"Error: Request failed. Details: {str(e)}"
                    should_retry = True
                
                if should_retry and attempt < max_retries:
                    retry_delay = min(2 ** attempt, 8)
                    print(f"[SCG Remote LLM/VLM] {error_msg}")
                    print(f"[SCG Remote LLM/VLM] Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                
                if should_retry and max_retries > 0:
                    error_msg = f"{error_msg} (failed after {total_attempts} attempts)"
                return io.NodeOutput(error_msg)
        
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[SCG Remote LLM/VLM] Error during API call:\n{error_details}")
            return io.NodeOutput(f"Error during API call: {str(e)}")
