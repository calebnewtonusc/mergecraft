"""
agents/contribution_agent.py — Generates convention-compliant code contributions.

Given a repository URL, task description, and extracted conventions,
generates complete code changes that will pass maintainer review.

Usage:
    agent = ContributionAgent(model_path="checkpoints/final")
    changes = agent.generate(
        repo_url="https://github.com/fastapi/fastapi",
        task="Add response model exclude_unset parameter to OpenAPI docs",
        conventions=conventions,
    )
"""

import os
import re

from loguru import logger

from core.project_conventions import ProjectConventions
from synthesis.prompts import CONTRIBUTION_SYSTEM, CONTRIBUTION_USER


class ContributionAgent:
    """Generates code contributions following project conventions."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self._model = None
        self._tokenizer = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None:
            return

        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading MergeCraft model from {self.model_path}")
        base = "Qwen/Qwen2.5-7B-Coder-Instruct"
        self._tokenizer = AutoTokenizer.from_pretrained(base)  # nosec B615
        import torch

        model = AutoModelForCausalLM.from_pretrained(  # nosec B615
            base, device_map="auto", torch_dtype=torch.bfloat16
        )
        self._model = PeftModel.from_pretrained(model, self.model_path)  # nosec B615

    def _call_api(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=CONTRIBUTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    def _generate(self, prompt: str) -> str:
        self._load_model()
        if self._model is None:
            return self._call_api(prompt)

        import torch

        text = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": CONTRIBUTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            # MC-14: pass input_ids and attention_mask explicitly so padding tokens are
            # correctly masked out, preventing spurious attention to pad positions
            out = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=4096,
                temperature=0.3,
                do_sample=True,
            )
        return self._tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def generate(
        self,
        repo_url: str,
        task: str | None,
        conventions: ProjectConventions,
    ) -> dict:
        """
        Generate code changes for a contribution task.

        Returns a dict with code_changes, commit_message, and pr_description.
        """
        prompt = CONTRIBUTION_USER.format(
            repo_url=repo_url,
            task_description=task or "General contribution",
            conventions_summary=conventions.to_summary(),
            commit_style=conventions.commit_style,
            test_requirement=(
                "Required — include tests with all changes"
                if conventions.test_required
                else "Optional but encouraged"
            ),
            max_pr_size=f"{conventions.max_pr_size_soft} lines (soft limit)",
            special_requirements=(
                "DCO Signed-off-by required"
                if conventions.commit_requires_dco
                else "none"
            ),
        )

        response = self._generate(prompt)

        # Parse response sections
        def extract_section(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
            return m.group(1).strip() if m else ""

        return {
            "analysis": extract_section("analysis"),
            "code_changes": extract_section("code_changes"),
            "commit_message": extract_section("commit_message"),
            "pr_description": extract_section("pr_description"),
            "scope_rationale": extract_section("scope_rationale"),
            "raw_response": response,
        }
