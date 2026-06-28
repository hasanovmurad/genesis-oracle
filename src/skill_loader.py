import json
from pathlib import Path


class GemmaSkillLoader:
    def __init__(self, skill_dir):
        self.skill_dir = Path(skill_dir)
        self.instructions = ""
        self.metadata = {}
        self.schemas = []
        self.load_skill()

    def load_skill(self):
        skill_file = self.skill_dir / "SKILL.md"
        text = skill_file.read_text(encoding="utf-8")

        if text.startswith("---"):
            _, yaml_part, markdown_body = text.split("---", 2)

            for line in yaml_part.strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    self.metadata[key.strip()] = value.strip()

            self.instructions = markdown_body.strip()
        else:
            self.instructions = text.strip()

        tools_dir = self.skill_dir / "tools"

        for schema_file in tools_dir.glob("*.json"):
            with open(schema_file, "r", encoding="utf-8") as f:
                self.schemas.append(json.load(f))

    def summary(self):
        return {
            "metadata": self.metadata,
            "instructions": self.instructions,
            "schemas": self.schemas,
        }


if __name__ == "__main__":
    loader = GemmaSkillLoader("skills/mandelbrot-explorer")

    print("Loading Gemma-Skill...")
    print(json.dumps(loader.summary(), indent=2))