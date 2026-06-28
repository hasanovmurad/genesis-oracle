from google import genai

client = genai.Client()

metrics = {
    "entropy": 1.7408490925712425,
    "boundary_complexity": 0.499552,
    "center_real": -0.745,
    "center_imag": 0.105,
    "zoom": 250.0,
    "max_iterations": 200,
}

prompt = f"""
You are exploring the Mandelbrot set.

Current simulation metrics:

{metrics}

Suggest better coordinates to move toward Seahorse Valley.

Return only:

center_real=
center_imag=
zoom=
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

print(response.text)