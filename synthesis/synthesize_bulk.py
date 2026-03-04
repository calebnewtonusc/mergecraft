"""
synthesis/synthesize_bulk.py — Bulk synthesis of contribution training pairs.

Generates synthetic contribution scenarios for underrepresented project types
and edge cases not well-covered by the PR outcome corpus.

Dispatches requests across up to 4 vLLM endpoints for throughput.
Falls back to Anthropic Claude API if vLLM is unavailable.

Usage:
    python synthesis/synthesize_bulk.py \
        --count 50000 \
        --output-dir data/synthesized/synthetic \
        --vllm_urls http://localhost:8001,http://localhost:8002,http://localhost:8003,http://localhost:8004
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import asyncio
import itertools
import json
import os
import random

# MC-9: module-level counter for collision-free sequential synthetic IDs
_synth_id_counter = itertools.count()

import aiohttp
from loguru import logger

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from synthesis.prompts import CONTRIBUTION_SYSTEM, CONTRIBUTION_USER

SYNTHETIC_SCENARIOS = [
    {
        "repo_type": "python_web_framework",
        "convention_profile": {
            "commit_style": "imperative with issue ref",
            "test_required": True,
            "max_pr_size": 200,
            "issue_first": False,
            "dco_required": False,
        },
        "task_templates": [
            "Add response header middleware support",
            "Fix query parameter encoding for Unicode characters",
            "Add support for streaming responses",
            "Implement request timeout handling",
        ],
    },
    {
        "repo_type": "ml_library",
        "convention_profile": {
            "commit_style": "conventional commits",
            "test_required": True,
            "max_pr_size": 400,
            "issue_first": True,
            "dco_required": False,
        },
        "task_templates": [
            "Add F1 score metric with macro/micro averaging",
            "Fix numerical stability in softmax for large values",
            "Add learning rate warmup scheduler",
            "Implement gradient checkpointing for memory efficiency",
        ],
    },
    {
        "repo_type": "devops_tool",
        "convention_profile": {
            "commit_style": "conventional commits with DCO",
            "test_required": True,
            "max_pr_size": 300,
            "issue_first": True,
            "dco_required": True,
        },
        "task_templates": [
            "Add AWS S3 backend for remote state",
            "Fix race condition in concurrent plan execution",
            "Add support for workspace variable files",
            "Implement provider plugin caching",
        ],
    },
]


# ─────────────────────────────────────────────────────────────
# Top-level async API helpers (spec-compliant)
# ─────────────────────────────────────────────────────────────

async def call_vllm(
    client: aiohttp.ClientSession,
    base_url: str,
    messages: list,
    model: str = "Qwen/Qwen2.5-72B-Instruct",
    api_key: str = "synthesis",
) -> str:
    """Call a vLLM OpenAI-compatible endpoint."""
    resp = await client.post(
        f"{base_url}/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": messages, "max_tokens": 4096, "temperature": 0.8},
        timeout=aiohttp.ClientTimeout(total=120.0),
    )
    resp.raise_for_status()
    return (await resp.json())["choices"][0]["message"]["content"]


async def call_claude(
    client: aiohttp.ClientSession,
    messages: list,
    api_key: str,
    system_prompt: str,
) -> str:
    """Call the Anthropic Claude API as a fallback."""
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
        },
        timeout=aiohttp.ClientTimeout(total=120.0),
    )
    resp.raise_for_status()
    return (await resp.json())["content"][0]["text"]


# ─────────────────────────────────────────────────────────────
# Synthesizer class
# ─────────────────────────────────────────────────────────────

class ContributionSynthesizer:
    def __init__(
        self,
        vllm_endpoints: list[str] | None = None,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        anthropic_api_key: str | None = None,
    ):
        if vllm_endpoints:
            self.endpoints = vllm_endpoints
        else:
            urls_env = os.environ.get("VLLM_URLS", "")
            if urls_env:
                self.endpoints = [u.strip() for u in urls_env.split(",") if u.strip()]
            else:
                self.endpoints = []

        self.model_name = model_name
        self.anthropic_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.vllm_api_key = os.environ.get("VLLM_API_KEY", "synthesis")
        self._endpoint_idx = 0

    def _next_endpoint(self) -> str | None:
        """Round-robin across available endpoints."""
        if not self.endpoints:
            return None
        idx = self._endpoint_idx % len(self.endpoints)
        self._endpoint_idx += 1
        return self.endpoints[idx]

    async def _generate(
        self,
        session: aiohttp.ClientSession,
        system: str,
        user: str,
    ) -> str | None:
        """Generate text: try vLLM first, fall back to Claude."""
        endpoint = self._next_endpoint()
        if endpoint:
            try:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
                return await call_vllm(session, endpoint, messages, self.model_name, self.vllm_api_key)
            except Exception as e:
                logger.debug(f"vLLM call failed at {endpoint}: {e}")

        if self.anthropic_key:
            try:
                messages = [{"role": "user", "content": user}]
                return await call_claude(session, messages, self.anthropic_key, system)
            except Exception as e:
                logger.error(f"Claude API error: {e}")

        return None

    async def _synthesize_single(
        self,
        session: aiohttp.ClientSession,
        scenario: dict,
        task: str,
    ) -> dict | None:
        """Generate one synthetic (task, contribution) training pair."""
        conventions = scenario["convention_profile"]
        user = CONTRIBUTION_USER.format(
            repo_url=f"github.com/example/{scenario['repo_type']}",
            task_description=task,
            conventions_summary=json.dumps(conventions, indent=2),
            commit_style=conventions["commit_style"],
            test_requirement="required" if conventions["test_required"] else "optional",
            max_pr_size=f"{conventions['max_pr_size']} lines",
            special_requirements=(
                "DCO Signed-off-by required" if conventions["dco_required"] else "none"
            ),
        )

        try:
            response = await self._generate(session, CONTRIBUTION_SYSTEM, user)
            if not response:
                return None

            return {
                "id": f"synth_{next(_synth_id_counter):08d}",  # MC-9: sequential, no collisions
                "repo_type": scenario["repo_type"],
                "task": task,
                "contribution": response,
                "conventions": conventions,
                "outcome": "merged",
                "merge_probability_label": 0.9,
                "source": "synthetic",
            }

        except Exception as e:
            logger.debug(f"Synthesis failed for {task[:50]}: {e}")
            return None

    async def synthesize_bulk(
        self,
        output_dir: Path,
        count: int,
        batch_size: int = 32,
    ) -> int:
        """Generate `count` synthetic contribution training pairs."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "synthetic_contributions.jsonl"

        # MC-32: use ceiling division so total generated >= requested count
        per_scenario = max(1, -(-count // len(SYNTHETIC_SCENARIOS)))
        tasks_list = []
        for scenario in SYNTHETIC_SCENARIOS:
            templates = scenario["task_templates"]
            for _ in range(per_scenario):
                tasks_list.append((scenario, random.choice(templates)))

        logger.info(
            f"Generating {len(tasks_list)} synthetic contribution pairs via "
            f"{len(self.endpoints)} vLLM endpoint(s)..."
        )

        completed = 0
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(tasks_list), batch_size):
                batch = tasks_list[i:i + batch_size]
                coros = [self._synthesize_single(session, s, t) for s, t in batch]
                results = await asyncio.gather(*coros, return_exceptions=True)

                with out_file.open("a") as fh:
                    for result in results:
                        if isinstance(result, BaseException):
                            logger.warning(f"Synthesis task raised an exception: {result}")
                        elif isinstance(result, dict):
                            fh.write(json.dumps(result) + "\n")
                            completed += 1

                if completed % 500 == 0 and completed > 0:
                    logger.info(f"Synthetic pairs: {completed}/{len(tasks_list)}")

        logger.success(f"Synthetic synthesis complete: {completed} pairs → {out_file}")
        return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="MergeCraft bulk synthesis")
    parser.add_argument("--count", type=int, default=50000)
    parser.add_argument("--output-dir", default="data/synthesized/synthetic")
    parser.add_argument(
        "--vllm_urls",
        default=os.environ.get("VLLM_URLS", ""),
        help="Comma-separated vLLM endpoints",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-72B-Instruct")
    args = parser.parse_args()

    vllm_endpoints = None
    if args.vllm_urls:
        vllm_endpoints = [u.strip() for u in args.vllm_urls.split(",") if u.strip()]

    synthesizer = ContributionSynthesizer(
        vllm_endpoints=vllm_endpoints,
        model_name=args.model_name,
    )

    asyncio.run(
        synthesizer.synthesize_bulk(
            output_dir=Path(args.output_dir),
            count=args.count,
        )
    )


if __name__ == "__main__":
    main()
