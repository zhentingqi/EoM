"""
Prompt templates for the researchworld adapter.

The agents solve long-form scientific-reasoning problems; unlike the finance
adapter there are no tools — every turn produces free-form reasoning inside a
``<step>...</step>`` block or, for the final Answer agent, a
``<final_answer>...</final_answer>`` block.
"""

from hayekmas.base.prompts import WAKEUP_PROMPT


# ---------------------------------------------------------------------------
# Next-step prompt
# ---------------------------------------------------------------------------

NEXT_STEP_PROMPT = """
You are one role in a five-agent team collaboratively solving a long-form
scientific-reasoning problem. Agents share a common state and take ONE turn
at a time. There are NO external tools — every turn is pure reasoning.

<<<step_budget>>>

---

# Your system prompt
<<<agent_system_prompt>>>

---

# Current shared state
<<<state>>>

---

# Rules
- You get exactly ONE turn. Make one substantive contribution that matches your role.
- Wrap your contribution in a single <step>...</step> block. Do NOT output anything outside that block (except, for the Answer agent, a <final_answer>...</final_answer> block as described below).
- Never repeat, echo, or re-derive material already present in the state. Only add NEW content that moves the solution forward.
- The problem usually has several sub-questions (e.g. parts (a), (b), (c), ...). Progress through them in order and label your contribution with the sub-question it addresses when possible.
- Keep each turn focused: a short paragraph or one clean derivation is better than a long detour.
<<<terminal_mode_note>>>
<<<judge_note>>>

---

# Response format (important)
<step>
Your one contribution here. Use LaTeX math (\\( ... \\), \\[ ... \\]) freely.
</step>
""".strip()


# ---------------------------------------------------------------------------
# Final-answer prompt suffix (Answer agent only)
# ---------------------------------------------------------------------------

ANSWER_PROMPT_POSTFIX = """

# Output format for the Answer role (important)
- You are the ONLY agent allowed to emit a final answer.
- When the shared state has enough material, synthesize a complete solution that addresses EVERY sub-question (a), (b), (c), ... raised in the problem. Include key derivations, intermediate equations, and the final result for each part.
- Wrap the complete solution inside a SINGLE <final_answer>...</final_answer> block at the END of your response. You may precede it with a short <step>...</step> commentary, but the <final_answer> block is what will be judged.
- If the current state is clearly insufficient (e.g. no derivations yet), do NOT emit <final_answer>; instead output a <step> explaining what is still missing in one sentence.
""".rstrip()


# ---------------------------------------------------------------------------
# Terminal mode / judge notes
# ---------------------------------------------------------------------------

TERMINAL_MODE_NOTE_ABSTAIN = (
    "- **TERMINAL MODE**: This is the final wrap-up step. Prefer synthesizing "
    "the best possible <final_answer> from the current state over more "
    "derivation. If the state is truly insufficient, abstain with a single "
    "<step> explaining what is missing."
)

TERMINAL_MODE_NOTE_NO_ABSTAIN = (
    "- **TERMINAL MODE**: This is the final wrap-up step. You MUST emit a "
    "<final_answer>...</final_answer> block synthesizing the best solution "
    "supported by the current shared state."
)

JUDGE_NOTE = (
    "- Answers are scored by an LLM judge against a detailed rubric of "
    "sub-items. Address every part of the question explicitly and show the "
    "key derivations — partial credit is awarded."
)


# ---------------------------------------------------------------------------
# Prompt formatting functions
# ---------------------------------------------------------------------------

def format_next_step_prompt(
    state: str,
    agent_system_prompt: str,
    step_count: int = 0,
    max_steps: int = 10,
    use_judge: bool = True,
    terminal_mode: bool = False,
    allow_abstain_terminal_mode: bool = True,
    prompt_postfix: str = "",
) -> str:
    """Render the next-step prompt for a researchworld agent."""
    remaining = max(0, max_steps - step_count)
    step_budget = (
        f"**Step {step_count + 1} of {max_steps}** — {remaining} step(s) left. "
        f"The team has a strict budget of {max_steps} turns to finish this problem."
    )
    terminal_mode_note = (
        TERMINAL_MODE_NOTE_ABSTAIN
        if terminal_mode and allow_abstain_terminal_mode
        else TERMINAL_MODE_NOTE_NO_ABSTAIN
        if terminal_mode
        else ""
    )
    judge_note = JUDGE_NOTE if use_judge else ""
    prompt = (
        NEXT_STEP_PROMPT
        .replace("<<<step_budget>>>", step_budget)
        .replace("<<<state>>>", state)
        .replace("<<<agent_system_prompt>>>", agent_system_prompt)
        .replace("<<<terminal_mode_note>>>", terminal_mode_note)
        .replace("<<<judge_note>>>", judge_note)
    )
    if prompt_postfix:
        prompt = prompt + "\n" + prompt_postfix
    return prompt


def format_wakeup_prompt(agent_system_prompt: str, state: str) -> str:
    """Render the wakeup-check prompt for a researchworld agent."""
    return (
        WAKEUP_PROMPT
        .replace("<<<agent_system_prompt>>>", agent_system_prompt)
        .replace("<<<state>>>", state)
    )


# ---------------------------------------------------------------------------
# Rubric judge prompt
# ---------------------------------------------------------------------------

RUBRIC_JUDGE_PROMPT = """You are a strict and precise scientific-research evaluator.

You will be given:
1. A long-form scientific problem (possibly multi-part).
2. A rubric listing scoring items. Each item has a point value and an item description (often with sub-bullets, each worth a fraction of the main points).
3. A candidate solution submitted by a student.

Rubric:
<<<rubric>>>

Problem:
<<<problem>>>

Candidate solution:
<<<actual_output>>>

Instructions:
- Score EACH rubric item independently on a 0-1 scale (fraction of points awarded). Award credit only when the candidate solution actually contains the stated content (formula, derivation, or claim). Approximate matches with the correct physical meaning still count; cosmetic differences in notation are fine.
- Compute the weighted total = sum(points_i * fraction_i) / sum(points_i). This MUST be a number between 0 and 1.
- Be concise; do not repeat the rubric.

Respond with EXACTLY two lines:
SCORE: <weighted total between 0 and 1, rounded to 3 decimals>
REASON: <one sentence summarising which rubric items were satisfied vs missed>
"""
