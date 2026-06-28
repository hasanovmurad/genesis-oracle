# Project Genesis – Autonomous Engineer

## Exercise 1 – Manual Cartographer

A Mandelbrot simulator was executed manually and the resulting metrics were passed to Gemini for coordinate suggestions.

Manual exploration steps:

| Step | Center Real | Center Imag | Zoom | Entropy | Boundary Complexity |
|---|---:|---:|---:|---:|---:|
| 1 | -0.500 | 0.000 | 1.5 | 1.1147 | 0.6220 |
| 2 | -0.745 | 0.105 | 25.0 | 1.0278 | 0.2460 |
| 3 | -0.745 | 0.105 | 250.0 | 1.7408 | 0.4996 |

### Reflection

Manual coordination introduces latency because the human must copy simulation metrics into the model prompt, wait for a suggestion, edit parameters, and rerun the simulation. This process is also error-prone because coordinates and zoom values can be copied incorrectly. Automated tool-calling loops reduce this friction by allowing the model to invoke the simulator directly.

---

## Exercise 2 – Closed-Loop Tool Calling

An autonomous cartographer loop was implemented in `src/autonomous_cartographer.py`.

The loop performs:

1. Model decision
2. Parameter parsing
3. Local Mandelbrot simulation
4. Observation feedback
5. Iterative zoom refinement

Observed autonomous steps reached zoom level `1620.0` within five iterations.

---

## Exercise 3 – Gemma-Skill Capsule

A reusable Mandelbrot exploration skill was packaged under:

`skills/mandelbrot-explorer/`

It contains:

- `SKILL.md`
- `tools/mandelbrot_schema.json`
- `scripts/mandelbrot_solver.py`

The dynamic loader `src/skill_loader.py` successfully parsed:

- skill metadata
- system instructions
- tool schemas

### Maintainability Answer

Packaging capabilities into self-contained Gemma-Skills improves maintainability because each skill bundles instructions, tool schemas, and implementation scripts in one modular unit. Multi-agent systems can load, reuse, version, and replace skills without rewriting the entire agent architecture. This makes complex autonomous systems easier to extend, debug, and share across projects.