# Project Genesis – Scholar Prime (Week 11)

## Exercise 1 – OpenAlex Setup

The Google DeepMind Science Skills repository was cloned successfully.

The OpenAlex literature search tool was initialized and tested.

Resolved author:

- Geoffrey E. Hinton
- OpenAlex ID: https://openalex.org/A5108093963

---

## Exercise 2 – Scholar-Prime Agent

A new ADK agent named `scholar_prime` was created.

Configuration:

- Model: gemini-3.5-flash
- Scientific research persona
- Academic literature analysis
- DOI reporting
- Material parameter extraction

---

## Exercise 3 – Literature Search

The `search_arxiv` tool was exposed to the agent.

Query:

> thermodynamic simulation parameters for advanced fission reactors

The agent successfully:

- searched arXiv,
- selected the most relevant publication,
- summarized the abstract,
- reported the DOI whenever available.

---

## Exercise 4 – Parameter Extraction

A simple extraction function was implemented.

The generated JSON file:

`data/simulation_parameters.json`

Example output:

```json
{
    "thermal_conductivity": "mentioned",
    "doi": "10.1016/j.nucengdes.2019.04.023"
}
```

---

## Reflection

Science Skills extend ADK agents with direct access to scientific databases such as OpenAlex and arXiv. This enables autonomous literature retrieval before simulation, making engineering agents capable of combining numerical models with current scientific knowledge. Structured parameter extraction further supports reproducible simulation workflows by converting research literature into machine-readable data.