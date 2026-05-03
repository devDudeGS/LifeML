import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from dotenv import load_dotenv
load_dotenv()

CLIENT_ID     = os.environ["GOOGLE_CLIENT_ID"]
CLIENT_SECRET = os.environ["GOOGLE_CLIENT_SECRET"]

SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly"
]

client_config = {
    "web": {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uris": ["http://localhost:8080"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
    }
}

flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
creds = flow.run_local_server(port=8080)

# Save tokens for future use
tokens = {
    "access_token":  creds.token,
    "refresh_token": creds.refresh_token,
    "client_id":     CLIENT_ID,
    "client_secret": CLIENT_SECRET,
}
with open("google_health_tokens.json", "w") as f:
    json.dump(tokens, f, indent=2)

print("Tokens saved to google_health_tokens.json")
print(f"Access token:  {creds.token[:20]}...")
print(f"Refresh token: {creds.refresh_token[:20]}...")