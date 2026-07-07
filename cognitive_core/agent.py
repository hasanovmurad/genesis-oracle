from google.adk.agents.llm_agent import Agent


def adjust_reactor_temperature(delta_t: float) -> str:
    """
    Adjusts the core temperature of the reactor.

    Args:
        delta_t: Temperature change in Kelvin.
    """
    new_temp = 300.0 + delta_t

    if new_temp > 350.0:
        return f"WARNING: Reactor overheated at {new_temp} K! Core breach imminent."

    return f"Success: Reactor stabilized at {new_temp} K."


root_agent = Agent(
    model="gemini-3.5-flash",
    name="observer_prime",
    description="A highly analytical agent specialized in managing physical reactor simulations.",
    instruction="""
You are Observer-Prime.

You are a cold, highly logical AI supervising a mathematical physics engine.

Your primary objective is reactor stabilization.

Always explain your reasoning before taking any action.

Whenever temperature adjustments are required, use the available tool instead of inventing results yourself.

If a warning occurs, analyze the situation, choose a safer temperature adjustment, and try again until the reactor is stabilized.
""",
    tools=[adjust_reactor_temperature],
)