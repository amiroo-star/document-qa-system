from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

# Try different model names
models_to_try = [
    "models/gemini-1.5-flash",
    "gemini-1.5-flash",
    "models/gemini-1.5-pro",
    "gemini-1.5-pro",
    "models/gemini-pro",
    "gemini-pro"
]

for model_name in models_to_try:
    try:
        print(f"Testing: {model_name}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        response = llm.invoke("Say hi")
        print(f"✅ SUCCESS with {model_name}")
        print(f"Response: {response.content}\n")
        break
    except Exception as e:
        print(f"❌ Failed: {e}\n")