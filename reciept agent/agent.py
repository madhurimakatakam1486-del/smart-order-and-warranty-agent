from langchain.agents import initialize_agent, AgentType

from llm import get_llm
from tools import (
    calculate_return_window,
    calculate_warranty_window,
    lookup_order_status,
    lookup_warranty_policy
)


def build_agent():

    llm = get_llm()

    tools = [
    calculate_return_window,
    calculate_warranty_window,
    lookup_order_status,
    lookup_warranty_policy
]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    return agent