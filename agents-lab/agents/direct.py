"""Direct baseline: exactly one model request, with no Python tool."""

from mathlab.runtime import AgentContext
from mathlab.types import Task

INSTRUCTIONS = """Solve the maths problem. Use submit_answer to return one scalar answer:
an integer, fraction, or decimal, e.g. -3, 7/2, or 0.75. No units, LaTeX, or prose.
You have one model request and no Python access in this baseline.
Configured episode limits: {limits}
"""


async def solve(task: Task, ctx: AgentContext) -> str:
    history = [{"role": "user", "content": task.problem}]
    turn = await ctx.model(history, INSTRUCTIONS.format(limits=ctx.limits), allow_python=False)
    return turn.value
