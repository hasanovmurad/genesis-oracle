import re

from google import genai

from mandelbrot import run_simulation


def simulate_mandelbrot(
    center_real: float,
    center_imag: float,
    zoom: float,
    max_iterations: int = 200,
) -> dict:
    _, metrics = run_simulation(
        center_real=center_real,
        center_imag=center_imag,
        zoom=zoom,
        max_iterations=max_iterations,
        output_path=f"data/mandelbrot_auto_{int(zoom)}.png",
    )
    return metrics


def parse_numbers(text):
    real = float(re.search(r"center_real=([-0-9.]+)", text).group(1))
    imag = float(re.search(r"center_imag=([-0-9.]+)", text).group(1))
    zoom = float(re.search(r"zoom=([0-9.]+)", text).group(1))
    return real, imag, zoom


def main():
    client = genai.Client()

    metrics = simulate_mandelbrot(-0.5, 0.0, 1.5)

    for step in range(1, 6):
        prompt = f"""
You are an autonomous Mandelbrot explorer.

Current metrics:
{metrics}

Target:
Navigate toward Seahorse Valley around center_real=-0.745, center_imag=0.105.
Increase zoom until zoom >= 15000.

Return only:
center_real=
center_imag=
zoom=
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text.strip()
        print(f"\nSTEP {step} MODEL DECISION")
        print(text)

        center_real, center_imag, zoom = parse_numbers(text)

        metrics = simulate_mandelbrot(
            center_real=center_real,
            center_imag=center_imag,
            zoom=zoom,
        )

        print("OBSERVATION:")
        print(metrics)

        if zoom >= 15000:
            print("\nConverged: zoom >= 15000")
            break


if __name__ == "__main__":
    main()