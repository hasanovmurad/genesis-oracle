# Project Genesis – Cognitive Core (Week 10)

## Exercise 3

The ADK Web UI successfully maintained conversation state across multiple user interactions.

The agent remembered the stored thermal friction coefficient after an unrelated discussion about matrix multiplication, demonstrating native state tracking without manually managing chat history.

---

## Exercise 4

The reactor temperature adjustment tool was registered directly inside the ADK agent.

When instructed to increase the temperature by 80 K, the agent first triggered an overheating warning. It analyzed the result, selected a safer adjustment of 40 K, called the tool again, and successfully stabilized the reactor.

---

## Reflection

Compared with the manual while-loops from Week 9, the ADK automatically manages conversation state and tool execution. There is no need to manually parse JSON responses or implement custom orchestration logic. This results in cleaner code, simpler agent development, and more reliable autonomous behavior.