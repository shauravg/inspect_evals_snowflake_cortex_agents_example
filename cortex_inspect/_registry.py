from inspect_ai.model import ModelAPI
from inspect_ai.model._registry import modelapi


@modelapi(name="cortex-agents")
def cortex_agents() -> type[ModelAPI]:
    from ._cortex_agent import CortexAgentModelAPI

    return CortexAgentModelAPI
