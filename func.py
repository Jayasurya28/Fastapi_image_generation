from huggingface_hub import InferenceClient
import os
from io import BytesIO

token = os.getenv("HUGGING_FACE_KEY")

client = InferenceClient(
	provider="together",
	api_key=token
)

# output is a PIL.Image object
image = client.text_to_image(
	"Astronaut riding a horse",
	model="black-forest-labs/FLUX.1-dev"
)

with open("image.jpg", "wb") as f:
    image.save(f, format='JPEG')
