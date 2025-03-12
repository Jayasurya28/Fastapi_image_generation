from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

token = os.environ.get("HF_TOKEN")
client = InferenceClient(
    provider="together",
    api_key=token
)

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        logging.info(f'Generating image for prompt: {request.prompt}')
        image = client.text_to_image(
            request.prompt,
            model="stabilityai/stable-diffusion-3.5"
        )
        
        image_path = "generated_image.jpg"
        image.save(image_path, format='JPEG')
        
        return FileResponse(image_path, media_type='image/jpeg')
    except Exception as e:
        logging.error(f'Error generating image: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def hello_world():
    return {'message': 'Hello World'}

@app.get("/test")
async def test(prompt: str):
    request = ImageRequest(prompt=prompt)
    return await generate_image(request)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from io import BytesIO
from PIL import Image
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Configure the API key
genai.configure(api_key="GEMINI_API_KEY")

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        logging.info(f'Generating image for prompt: {request.prompt}')
        model = genai.GenerativeModel('imagen-3.0-generate-002')
        response = model.generate_images(
            prompt=request.prompt,
            config=genai.GenerateImagesConfig(
                number_of_images=1,
            )
        )
        
        image_path = "generated_image.jpg"
        response.images[0].save(image_path, format='JPEG')
        
        return FileResponse(image_path, media_type='image/jpeg')
    except Exception as e:
        logging.error(f'Error generating image: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def hello_world():
    return {'message': 'Hello World'}

@app.get("/test")
async def test(prompt: str):
    request = ImageRequest(prompt=prompt)
    return await generate_image(request)