from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

token = os.getenv("HUGGING_FACE_KEY")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/predict")
async def predict(request: PromptRequest):
    if not token:
        raise HTTPException(status_code=500, detail="Hugging Face API key is missing")

    headers = {"Authorization": f"Bearer {token}"}
    url = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

    response = requests.post(url, headers=headers, json={"inputs": request.prompt})

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return {"result": response.json()}
