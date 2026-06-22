import os

from google import genai


def main():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Explain the difference between stateful NumPy random generation "
            "and stateless JAX PRNG splitting in exactly one highly sarcastic sentence."
        ),
    )

    print(response.text)


if __name__ == "__main__":
    main()