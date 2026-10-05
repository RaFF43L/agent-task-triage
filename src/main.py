from ai.agent import triage_agent
from config.envs import envs
from config.logging import setup_logging, get_logger

setup_logging(level=envs.log_level)
logger = get_logger(__name__)


def run(message: str) -> dict:
    """Executes the triage graph for a message."""
    result = triage_agent.invoke(
        {"message": message},
        config={"recursion_limit": envs.recursion_limit},
    )
    return result


if __name__ == "__main__":
    examples = [
        "My system has been down for 2 hours and I can't work!",
        "I was charged twice on this month's invoice.",
    ]

    for msg in examples:
        print("\n" + "=" * 70)
        print(f"Message    : {msg}")
        out = run(msg)
        print(f"Category   : {out.get('category')}")
        print(f"Priority   : {out.get('priority')}")
        print(f"Attempts   : {out.get('attempts')}")
        print(f"Approved   : {out.get('quality_approved')}")
        print(f"Response   : {out.get('response')}")
