from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import os
from io import BytesIO
from PIL import Image
from fastapi.responses import FileResponse

token = os.getenv("HUGGING_FACE_KEY")

app = FastAPI()

client = InferenceClient(
    provider="together",
    api_key=token
)

class ImageRequest(BaseModel):
    prompt: str

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        image = client.text_to_image(
            request.prompt,
            model="black-forest-labs/FLUX.1-dev"
        )
        
        image_path = "generated_image.jpg"
        image.save(image_path, format='JPEG')
        
        return FileResponse(image_path, media_type='image/jpeg')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/')
def hello_world():
    return {'message': 'Hello World'}
