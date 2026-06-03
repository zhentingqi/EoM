# ═══════════════════════════════════════════════════════════════════════════
# MUTATE AGENT PROMPT
# ═══════════════════════════════════════════════════════════════════════════

GOOD_AGENT_BIRTH_PROMPT = """
You are an expert prompt engineer. You will modify an agent's system prompt to improve its general problem-solving ability.

## The agent's system prompt
The agent's system prompt has two parts:

Frozen system prompt (immutable):
<frozen_system_prompt>
<<<frozen_system_prompt>>>
</frozen_system_prompt>

Trainable system prompt (this is what you will modify):
<trainable_system_prompt>
<<<trainable_system_prompt>>>
</trainable_system_prompt>

## Your task
Modify the trainable system prompt to add a general-purpose reasoning skill or strategy that helps across many different problems. The new prompt must contain ZERO references to any specific problem, topic, number, equation, or answer.

Good modifications add transferable skills such as:
- A general reasoning heuristic (e.g. "try small examples first", "check edge cases")
- A problem-solving strategy (e.g. "identify what type of problem this is before solving")
- A self-checking habit (e.g. "verify by substituting back", "sanity-check the magnitude")
- A communication discipline (e.g. "state your key insight before the calculation")

Bad modifications (NEVER do these):
- Mentioning any specific numbers, formulas, or problem details
- Adding instructions that only help on one type of problem
- Making the prompt significantly longer — keep it concise

OUTPUT FORMAT:
Return ONLY the new trainable system prompt text, nothing else. No explanations, no meta-commentary, no labels.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# SPAWN FROM DEATH PROMPT
# ═══════════════════════════════════════════════════════════════════════════

BAD_AGENT_BIRTH_PROMPT = """
You are an expert prompt engineer. You will modify an agent's system prompt to fix a general weakness revealed by a failure.

## The agent's system prompt
The agent's system prompt has two parts:

Frozen system prompt (immutable):
<frozen_system_prompt>
<<<frozen_system_prompt>>>
</frozen_system_prompt>

Trainable system prompt (this is what you will modify):
<trainable_system_prompt>
<<<trainable_system_prompt>>>
</trainable_system_prompt>

## Failure context
The agent failed on a task. Study the failure to identify the GENERAL skill or habit that was missing — NOT the specific problem details.

<task>
<<<task_description>>>
</task>

<correct_answer>
<<<correct_answer>>>
</correct_answer>

Lines marked with >>> are the source agent's turns. All other lines are from other agents.
<trajectory>
<<<agent_trace>>>
</trajectory>

## Analysis instructions
Identify which general reasoning failure occurred:
- Arithmetic/calculation carelessness?
- Misreading the problem statement?
- Jumping to an answer without verifying?
- Not considering edge cases or special conditions?
- Ignoring relevant information from other agents?
- Applying the wrong method for the problem type?

## Your task
By analyzing the failure context, modify the trainable system prompt to help the agent avoid similar mistakes in the future. 

You can choose to:
- Add a new skill
- Edit an existing skill
- Delete an existing skill

Provide the skills in a do's and don'ts manner. Use itemized list to show the skills. 

CRITICAL RULES:
- The new prompt must contain ZERO references to any specific problem, topic, number, equation, formula, or answer from the failure context above.
- Keep the prompt concise. Do not let it grow beyond a short paragraph.

OUTPUT FORMAT:
Return ONLY the new trainable system prompt text, nothing else. No explanations, no meta-commentary, no labels.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# WAKEUP PROMPT
# ═══════════════════════════════════════════════════════════════════════════

WAKEUP_PROMPT = """
You are in a collaborative multi-agent task. You might or might not be the next agent to act at this moment.

## Your system prompt
<<<agent_system_prompt>>>

## Current state
<<<state>>>

## Instructions
Reason briefly and answer: Should you take action given the current state?

End your response with: \\boxed{yes} or \\boxed{no}.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATE ANSWER PROMPT
# ═══════════════════════════════════════════════════════════════════════════

EVALUATE_ANSWER_PROMPT = """You are an evaluator. Score whether the "Actual output" matches the "Expected output" for the given question.

Question: <<<input_question>>>

Expected output: <<<expected_output>>>

Actual output: <<<actual_output>>>

Reply with exactly two lines:
SCORE: <number between 0 and 1>
REASON: <one sentence explaining whether and why the actual output matches the expected output>"""


# ═══════════════════════════════════════════════════════════════════════════
# Format helpers
# ═══════════════════════════════════════════════════════════════════════════

def format_mutate_agent_prompt(
    frozen_system_prompt: str,
    trainable_system_prompt: str,
) -> str:
    """Return a filled-in good-agent-birth meta-prompt."""
    return (
        GOOD_AGENT_BIRTH_PROMPT
        .replace("<<<frozen_system_prompt>>>", frozen_system_prompt or "(empty)")
        .replace("<<<trainable_system_prompt>>>", trainable_system_prompt)
    )


def format_spawn_from_bankruptcy_prompt(
    frozen_system_prompt: str,
    trainable_system_prompt: str,
    agent_trace: str = "",
    *,
    task_description: str = "",
    correct_answer: str = "",
) -> str:
    """Return a filled-in bad-agent-birth meta-prompt."""
    if not agent_trace:
        agent_trace = "(No actions recorded - agent never successfully acted)"
    if not task_description:
        task_description = "(Task description not available)"
    if not correct_answer:
        correct_answer = "(Correct answer not available)"
    return (
        BAD_AGENT_BIRTH_PROMPT
        .replace("<<<frozen_system_prompt>>>", frozen_system_prompt or "(empty)")
        .replace("<<<trainable_system_prompt>>>", trainable_system_prompt)
        .replace("<<<agent_trace>>>", agent_trace)
        .replace("<<<task_description>>>", task_description)
        .replace("<<<correct_answer>>>", correct_answer)
    )


import re as _re

_PROMPT_TAG_RE = _re.compile(
    r"</?(?:trainable_system_prompt|frozen_system_prompt|section_to_modify)>",
)


def clean_trainable_prompt_output(raw: str, frozen_system_prompt: str = "") -> str:
    """Strip XML-style tags that the LLM may echo from the birth prompts."""
    if frozen_system_prompt in raw:
        raw = raw.replace(frozen_system_prompt, "")
    return _PROMPT_TAG_RE.sub("", raw).strip().strip("'\"").strip()
