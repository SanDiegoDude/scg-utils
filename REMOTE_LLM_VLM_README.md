# SCG Remote LLM/VLM - OAI Standard

A multipurpose ComfyUI node for connecting to OpenAI-compatible API endpoints, including local servers like LM Studio.

## Features

- **Multi-Modal Support**: Up to 4 optional image inputs for Vision-Language Models (VLMs)
- **Text-Only Mode**: Works with standard LLMs when no images are provided
- **OpenAI Compatible**: Works with any OpenAI API-compatible endpoint
- **Local Server Support**: Perfect for LM Studio, Ollama (with OpenAI compatibility), and other local servers
- **Full Parameter Control**: Temperature, top_p, max_tokens, and more

## Use Cases

### 1. LM Studio (Local VLM Server)
Connect to your local LM Studio server running vision models like Qwen2-VL:

```
API URL: http://localhost:1234/v1/chat/completions
Model Name: qwen2-vl-7b-instruct
API Key: lm-studio (or leave as default)
```

### 2. OpenAI API
Use OpenAI's GPT-4 Vision or other models:

```
API URL: https://api.openai.com/v1/chat/completions
Model Name: gpt-4-vision-preview
API Key: sk-your-actual-api-key-here
```

### 3. Other OpenAI-Compatible Services
Works with:
- Together AI
- Groq
- OpenRouter
- Local Ollama servers
- Any service implementing the OpenAI chat completions API

## Parameters

### Required Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_url` | STRING | `http://localhost:1234/v1/chat/completions` | API endpoint URL |
| `model_name` | STRING | `qwen2-vl-7b-instruct` | Model identifier |
| `api_key` | STRING | `lm-studio` | API key for authentication |
| `system_prompt` | STRING (multiline) | `You are a helpful assistant.` | System context message |
| `user_prompt` | STRING (multiline) | _(empty)_ | Your query/instruction |
| `temperature` | FLOAT | `0.7` | Sampling temperature (0.0-2.0) |
| `top_p` | FLOAT | `0.9` | Nucleus sampling parameter (0.0-1.0) |
| `max_tokens` | INT | `512` | Maximum tokens to generate (1-8000) |
| `seed` | INT | `42` | Seed value for ComfyUI caching control |
| `bypass` | BOOLEAN | `False` | If True, returns user_prompt without API call |

### Optional Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `image1` | IMAGE | First optional image input |
| `image2` | IMAGE | Second optional image input |
| `image3` | IMAGE | Third optional image input |
| `image4` | IMAGE | Fourth optional image input |

## Output

| Output | Type | Description |
|--------|------|-------------|
| `response` | STRING | Generated text response from the model |

## Usage Examples

### Example 1: Image Captioning with LM Studio

1. Load an image in ComfyUI
2. Connect it to `image1` input
3. Set `user_prompt` to: "Describe this image in detail."
4. Set `api_url` to your LM Studio server
5. Run the workflow

### Example 2: Multi-Image Comparison

1. Load 2-4 images in ComfyUI
2. Connect them to `image1`, `image2`, etc.
3. Set `user_prompt` to: "Compare these images and describe their differences."
4. Run the workflow

### Example 3: Text-Only LLM Query

1. Don't connect any images (leave all image inputs empty)
2. Set `user_prompt` to your question
3. Run the workflow - works as a standard LLM node

### Example 4: Complex Vision Task

```
system_prompt: "You are an expert image analyst."
user_prompt: "Analyze the composition, lighting, and subject matter of this photograph. Provide specific technical details."
temperature: 0.7
max_tokens: 1000
```

## LM Studio Configuration

To use with LM Studio:

1. **Download a VLM model** in LM Studio (e.g., Qwen2-VL-7B-Instruct)
2. **Start the server** in LM Studio (default port: 1234)
3. **Load the model** in LM Studio's server tab
4. **Use these settings** in the node:
   - API URL: `http://localhost:1234/v1/chat/completions`
   - Model Name: _(exact name from LM Studio)_
   - API Key: `lm-studio` (LM Studio doesn't require real API keys)

## Tips

- **Image Format**: Images are automatically converted to base64 PNG format
- **Large Images**: Consider resizing images before the node to reduce processing time
- **Timeout**: Requests timeout after 120 seconds
- **Error Messages**: Check the console output for detailed error information
- **Bypass Mode**: Use bypass mode to test workflows without making API calls
- **Seed Parameter**: The seed doesn't affect the API call itself, but changing it forces ComfyUI to re-execute the node. Set to "fixed" in ComfyUI to prevent automatic re-runs when upstream nodes change, or use "randomize" to force new generations each time.

## Troubleshooting

### "Connection failed" Error
- Verify the server is running (e.g., LM Studio server is started)
- Check the API URL is correct
- Ensure no firewall is blocking the connection

### "HTTP 401/403" Error
- Check your API key is correct
- For LM Studio, the API key can be anything (it's not validated)

### "HTTP 404" Error
- Verify the model name matches exactly
- Check the API endpoint path is correct

### Slow Response
- Try reducing `max_tokens`
- Use a smaller/faster model
- For local servers, check GPU/CPU utilization

## Technical Details

- **Image Encoding**: Base64 PNG format
- **Protocol**: OpenAI Chat Completions API v1
- **Timeout**: 120 seconds per request
- **Image Processing**: Converts ComfyUI tensors to PIL Images automatically

## Category

`scg-utils/llm`

## Version

1.0.0

## Author

scg-utils contributors
