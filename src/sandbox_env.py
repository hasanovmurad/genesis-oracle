def simulate_thermal_dampener(kappa: float) -> dict:
    base_temperature = 300.0
    instability = (kappa - 1.0) * 120.0
    temperature = base_temperature + instability

    if temperature < 260:
        state = "FREEZING"
    elif temperature > 340:
        state = "BOILING"
    else:
        state = "PERFECT"

    return {
        "kappa": kappa,
        "temperature": temperature,
        "system_state": state,
    }