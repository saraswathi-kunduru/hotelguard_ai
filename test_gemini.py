import os
from dotenv import load_dotenv
from google import genai

print("1. Starting HotelGuard Gemini test...")

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("2. ERROR: API key not found")
    raise SystemExit

print("2. API key found")

client = genai.Client(
    api_key=api_key,
    http_options={
        "timeout": 60000
    }
)

print("3. Sending request to Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents="Say hello to HotelGuard AI in one short sentence."
    )

    print("4. Gemini responded!")
    print(response.text)

except Exception as e:
    print("❌ Gemini request failed:")
    print(type(e).__name__)
    print(str(e))