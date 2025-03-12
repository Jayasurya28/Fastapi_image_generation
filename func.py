from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image
from fastapi.responses import FileResponse
import logging

logging.basicConfig(level=logging.INFO)

token = os.getenv("HUGGING_FACE_KEY")

app = FastAPI()

client = InferenceClient(
    provider="hf-inference",
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
            model="codermert/gamzekocc_fluxx"
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