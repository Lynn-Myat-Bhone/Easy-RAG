import os
from typing import Any

import requests
from dotenv import load_dotenv


class AIVOOVClient:
  def __init__(self, api_key: str | None = None, base_url: str = "https://aivoov.com"):
    load_dotenv()
    self.api_key = api_key or os.getenv("AIVOOV_API")
    if not self.api_key:
      raise ValueError("AIVOOV_API is missing in environment variables")
    self.url = f"{base_url.rstrip('/')}/api/v8/create"

  def create_tts(
    self,
    text: str,
    voice_id: str,
    pitch_rate: str = "default",
    speed_rate: str = "default",
    timeout: int = 30,
  ) -> dict[str, Any]:
    payload = {
      "voice_id[]": voice_id,
      "transcribe_text[]": text,
      "transcribe_ssml_pitch_rate[]": pitch_rate,
      "transcribe_ssml_spk_rate[]": speed_rate,
    }
    headers = {
      "X-API-KEY": self.api_key,
      "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(self.url, headers=headers, data=payload, timeout=timeout)
    response.raise_for_status()

    try:
      return response.json()
    except ValueError:
      return {"raw_response": response.text}


if __name__ == "__main__":
  # Quick manual test
  client = AIVOOVClient()
  result = client.create_tts(
    text="hello world",
    voice_id="cffc1d81-07cc-494f-a03a-0c0eebe99c8c",
  )
  print(result)