import google.generativeai as genai

# Configure the API key
genai.configure(api_key="GEMINI_API_KEY")

# Initialize the model
model = genai.GenerativeModel('gemini-pro')

# Generate text content
response = model.generate_content("Hello, world!")
print(response.text)