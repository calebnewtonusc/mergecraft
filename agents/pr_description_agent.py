"""
agents/pr_description_agent.py — Writes compelling PR titles and descriptions.

A good PR description is not documentation — it's a pitch. It answers:
  - WHY does this change need to exist?
  - WHAT did you change (summary, not the diff itself)?
  - HOW did you test it?
  - What tradeoffs did you make?

MergeCraft is trained on the correlation between description quality and merge rate.
"""

import os


from core.project_conventions import ProjectConventions


PR_DESCRIPTION_SYSTEM = """\
You are an expert at writing open source pull request descriptions that get merged.

A great PR description:
1. Has a clear, concise title (imperative mood, <70 chars)
2. Opens with WHY this change is needed (the problem)
3. Briefly describes WHAT changed (not the full diff)
4. Lists what was tested and how
5. Links the relevant issue (Fixes #XXX or Closes #XXX)
6. Acknowledges any tradeoffs or limitations
7. Is written for the reviewer's time — they have 50 other PRs to review

Avoid:
- Repeating the diff in the description
- Long walls of unstructured text
- Missing issue references when the project uses them
- Defensive language ("sorry if this is wrong", "please let me know if...")
"""

PR_DESCRIPTION_USER = """\
Write a pull request description for the following contribution.

Repository: {repo}
Task: {task}
Code summary: {code_summary}

Project conventions:
- Commit style: {commit_style}
- Issue linking: {issue_linking}
- DCO: {dco_requirement}

Format:
## Title (one line, imperative, <70 chars)

## Body
[WHY this change / problem being solved]
[WHAT changed — summary, not the diff]
[HOW tested]
[Fixes #ISSUE if applicable]
[Any tradeoffs or known limitations]
"""


class PRDescriptionAgent:
    """Writes compelling PR titles and descriptions that improve merge probability."""

    def __init__(self, model_path: str | None = None) -> None:
        # MC-12: model_path is stored but the agent currently uses the Anthropic API only.
        # A local-model inference path would read from self.model_path in _generate().
        self.model_path = model_path

    def write(
        self,
        code_changes: dict,
        conventions: ProjectConventions,
        task: str | None = None,
    ) -> dict:
        """
        Write a PR title and description.

        Returns dict with 'title' and 'body'.
        """
        code_summary = self._summarize_changes(code_changes.get("code_changes", ""))

        prompt = PR_DESCRIPTION_USER.format(
            repo=conventions.repo,
            task=task or "Contribution",
            code_summary=code_summary,
            commit_style=conventions.commit_style,
            issue_linking="required"
            if conventions.commit_requires_issue_ref
            else "optional",
            dco_requirement="required — add Signed-off-by"
            if conventions.commit_requires_dco
            else "not required",
        )

        response = self._generate(prompt)

        # MC-13: previous parser appended body_lines for every non-"## Title" line once
        # body_lines was non-empty, capturing duplicate content.
        # Use an explicit section state machine instead.
        lines = response.strip().splitlines()
        title = ""
        body_lines: list[str] = []
        section = "preamble"  # preamble | title_next | body
        for line in lines:
            if line.startswith("## Title"):
                section = "title_next"
            elif line.startswith("## Body"):
                section = "body"
            elif section == "title_next":
                if not title:
                    title = line.strip()
                section = "preamble"  # only grab the first line after ## Title
            elif section == "body":
                body_lines.append(line)

        if not title and lines:
            title = lines[0].strip()

        body = "\n".join(body_lines).strip()

        # Add DCO if required
        if conventions.commit_requires_dco:
            if "Signed-off-by:" not in body:
                body += "\n\nSigned-off-by: MergeCraft Contributor <contributor@example.com>"

        return {"title": title, "body": body}

    def _summarize_changes(self, code_changes: str) -> str:
        """Create a brief summary of code changes for the description prompt."""
        if not code_changes:
            return "Implementation details not available"
        lines = code_changes.splitlines()
        added = [line for line in lines if line.startswith("+") and not line.startswith("+++")]
        removed = [line for line in lines if line.startswith("-") and not line.startswith("---")]
        return f"{len(added)} lines added, {len(removed)} lines removed"

    def _generate(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=PR_DESCRIPTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
