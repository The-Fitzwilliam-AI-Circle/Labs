"""Reference agent: request one action, observe it, and repeat."""

import json

from mathlab.runtime import AgentContext
from mathlab.types import Task

INSTRUCTIONS = """Solve the maths problem. Use submit_answer to return one scalar answer:
an integer, fraction, or decimal, e.g. -3, 7/2, or 0.75. No units, LaTeX, or prose.
You may use Python 3.12, the standard library, SymPy 1.14.0 and mpmath 1.3.0.
Each Python call starts a fresh container: imports, variables and files do not persist.
Write self-contained programs and print useful output. Network access is disabled.
Execution errors are observations; repair your code if budget remains.
All conversations share these episode limits: {limits}
Submit your answer before the request budget or time runs out.
"""


async def solve(task: Task, ctx: AgentContext) -> str:
    history = [{"role": "user", "content": task.problem}]
    instructions = INSTRUCTIONS.format(limits=ctx.limits)

    while True:
        turn = await ctx.model(history, instructions)
        # Keep every provider item, including any reasoning items, in order.
        history.extend(turn.output_items)
        if turn.kind == "answer":
            # Only this return submits the episode's final answer.
            return turn.value

        observation = await ctx.python(turn.value)
        history.append(
            {
                "type": "function_call_output",
                "call_id": turn.call_id,
                "output": json.dumps(observation.to_dict()),
            }
        )
