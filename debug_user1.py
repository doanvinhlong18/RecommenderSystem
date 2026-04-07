import sys
import os
import requests

try:
    res = requests.get("http://localhost:8000/api/user/1/history")
    print(res.json())
except Exception as e:
    print("Error calling api:", e)
