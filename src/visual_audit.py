import os
from pathlib import Path

from google import genai
from google.genai import types


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    image_path = Path("data/audit_target.png")

    if not image_path.exists():
        raise FileNotFoundError("data/audit_target.png not found")

    client = genai.Client(api_key=api_key)

    image_bytes = image_path.read_bytes()

    prompt = """
You are a Visual Detective inspecting a dynamic wave signal plot.

Tasks:
1. Identify whether there is a visible anomaly.
2. Estimate the approximate X-axis region where the malfunction occurs.
3. Describe what kind of artifact it looks like.
4. Write a short funny poem mocking the engineering team that allowed this bug to pass.
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            ),
            prompt,
        ],
    )

    print(response.text)


if __name__ == "__main__":
    main()