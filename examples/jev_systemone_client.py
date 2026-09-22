import httpx

payload = {"state": "Paris is France's capital.", "questions": {"claim": {"type": "noul", "instructions": "Paris is France's capital."}}}
print(httpx.post("http://127.0.0.1:8000/v1/systemone", json=payload).json())
