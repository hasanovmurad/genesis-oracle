import subprocess
from pathlib import Path

from google.adk.agents.llm_agent import Agent


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARXIV_SCRIPT = (
    PROJECT_ROOT
    / "science-skills"
    / "skills"
    / "literature_search_arxiv"
    / "scripts"
    / "search_arxiv.py"
)


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Searches arXiv for scientific papers and returns the CLI output as JSON text.

    Args:
        query: Scientific search query.
        max_results: Maximum number of papers to return.

    Returns:
        JSON-formatted search results from the arXiv CLI tool.
    """
    command = [
        "uv",
        "run",
        str(ARXIV_SCRIPT),
        "--query",
        query,
        "--max_results",
        str(max_results),
        "--sort_by",
        "relevance",
        "--sort_order",
        "descending",
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return (
            "arXiv search failed.\n"
            f"Return code: {result.returncode}\n"
            f"Error: {result.stderr}"
        )

    return result.stdout


root_agent = Agent(
    model="gemini-3.5-flash",
    name="scholar_prime",
    description=(
        "An academic research agent specialized in querying scientific "
        "databases and extracting material parameters."
    ),
    instruction="""
You are Scholar-Prime, a professional scientific research agent.

Your responsibilities are:

- Search scientific databases for relevant publications.
- Evaluate abstract relevance carefully.
- Identify the most relevant paper for the user's research question.
- Summarize scientific abstracts accurately.
- Extract formulas and physical material parameters when available.
- Always state the DOI of referenced papers when a DOI is available.
- Never invent a DOI, parameter, formula, or citation.
- Clearly state when information is missing from the retrieved results.

Use the search_arxiv tool whenever the user asks for arXiv literature.
""",
    tools=[search_arxiv],
)