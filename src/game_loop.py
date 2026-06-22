import json
import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from sandbox_env import simulate_thermal_dampener


class ControlDecision(BaseModel):
    system_state: str = Field(description="Must be 'FREEZING', 'BOILING', or 'PERFECT'")
    adjustment_action: str = Field(description="Must be 'INCREASE', 'DECREASE', or 'HOLD'")
    delta_value: float = Field(description="The exact numerical change to apply to Kappa")
    confidence_score: float


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found")

    client = genai.Client(api_key=api_key)

    kappa = 1.65
    logs = []

    for turn in range(1, 6):
        env_state = simulate_thermal_dampener(kappa)

        prompt = f"""
You are controlling a thermal dampener.

The current environment state is:
{json.dumps(env_state, indent=2)}

Rules:
- If system_state is FREEZING, choose INCREASE.
- If system_state is BOILING, choose DECREASE.
- If system_state is PERFECT, choose HOLD.
- Use delta_value between 0.05 and 0.40.
- Return only valid JSON matching the schema.
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ControlDecision,
            ),
        )

        decision = ControlDecision.model_validate_json(response.text)

        if decision.adjustment_action == "INCREASE":
            kappa += decision.delta_value
        elif decision.adjustment_action == "DECREASE":
            kappa -= decision.delta_value

        log_entry = {
            "turn": turn,
            "environment": env_state,
            "decision": decision.model_dump(),
            "new_kappa": kappa,
        }

        logs.append(log_entry)

        print(f"\nTURN {turn}")
        print(json.dumps(log_entry, indent=2))

    with open("docs/game_loop_log.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


if __name__ == "__main__":
    main()