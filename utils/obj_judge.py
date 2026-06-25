"""
Lightweight API wrapper for GPT-4o image analysis (used by SDM for scene parsing).
"""

import os
import requests
import base64


class ImageAnalyzer:
    """GPT-4o image analysis wrapper."""

    def __init__(self, api_key: str, api_url: str, detect_prompt: str):
        self.api_key = api_key
        self.api_url = api_url
        self.detect_prompt = detect_prompt
        self.api_headers = {"Authorization": f"Bearer {api_key}"}
        self.completion_url = f"{api_url}/v1/chat/completions"

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def get_completion(self, messages, filename):
        data = {"model": "gpt-4o", "max_tokens": 300, "messages": messages}
        try:
            resp = requests.post(
                self.completion_url, json=data, headers=self.api_headers, timeout=60
            ).json()
            return (
                resp.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception as e:
            print(f"[GPT-4o Error] {filename}: {e}")
            return ""

    def get_img_txt(self, filename, folder_path):
        image_path = os.path.join(folder_path, filename)
        if not os.path.exists(image_path):
            return ""
        base64_img = self.encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.detect_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}"
                        },
                    },
                ],
            }
        ]
        return self.get_completion(messages, filename)

    def count_files(self, folder_path):
        from tqdm import tqdm
        results = {}
        for filename in tqdm(os.listdir(folder_path), desc="GPT-4o analyzing"):
            if "csv" not in filename:
                results[filename] = self.get_img_txt(filename, folder_path)
        return results

    def object_judge(self, folder_path):
        return self.count_files(folder_path)
