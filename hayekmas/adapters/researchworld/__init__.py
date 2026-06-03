"""Researchworld adapter for HayekMAS.

This adapter targets long-form scientific-reasoning benchmarks such as
FrontierScience. Each task is a multi-part physics/quantum/chemistry problem
scored against a rubric of ``Points: X, Item: ...`` entries by an LLM judge.
No external tools — the five agents collaborate through the shared state only.
"""

from hayekmas.adapters.researchworld.agent import (
    ResearchAction,
    ResearchAgent,
    LiteratureResearchAgent,
    PlannerResearchAgent,
    DeriverResearchAgent,
    VerifierResearchAgent,
    AnswerResearchAgent,
    RESEARCH_AGENT_CLASSES,
)
from hayekmas.adapters.researchworld.env import ResearchTask, ResearchEnv
from hayekmas.adapters.researchworld.runtime import (
    ResearchRuntimeConfig,
    ResearchTrainer,
    ResearchEvaluator,
    run,
    main,
)
