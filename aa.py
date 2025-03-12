import google.generativeai as genai

# Configure the API key
genai.configure(api_key="AIzaSyDHm6WB50DdfBmj9rcXNx0W9i8RjyvEiOY")

# Initialize the model
model = genai.GenerativeModel('gemini-1.0-pro')

# Generate text content
response = model.generate_content("Hello, world!")
print(response.text)