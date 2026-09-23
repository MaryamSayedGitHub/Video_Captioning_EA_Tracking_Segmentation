import os, requests

api_key = os.environ.get("GROQ_API_KEY", "PUT_YOUR_KEY_HERE")
url = "https://api.groq.com/openai/v1/models"
headers = {"Authorization": f"Bearer {api_key}"}
response = requests.get(url, headers=headers)
for m in response.json().get("data", []):
    print(m["id"])