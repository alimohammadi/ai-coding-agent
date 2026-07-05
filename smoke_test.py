import asyncio
from pathlib import Path

from tools.base import ToolInvocation
from tools.builtin.read_file import ReadFileTool


async def main():
    tool = ReadFileTool()

    inv = ToolInvocation(
        parameters={"path": "tools/builtin/read_file.py", "offset": 1, "limit": 5},
        cwd=Path.cwd(),
    )
    r = await tool.execute(inv)
    print("=== normal read ===")
    print(r.success, r.truncated)
    print(r.output)
    print(r.metadata)

    inv2 = ToolInvocation(
        parameters={"path": "tools/builtin/read_file.py", "offset": 100000},
        cwd=Path.cwd(),
    )
    r2 = await tool.execute(inv2)
    print("=== out-of-range offset ===")
    print(r2.success, r2.error)

    inv3 = ToolInvocation(parameters={"path": "nope.txt"}, cwd=Path.cwd())
    r3 = await tool.execute(inv3)
    print("=== missing file ===")
    print(r3.success, r3.error)


asyncio.run(main())
