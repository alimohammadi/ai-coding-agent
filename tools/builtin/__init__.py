from .read_file import ReadFileTool

# from .write_file import WriteFileTool


__all__ = [
    "ReadFileTool",
    # "WriteFileTool",
]


def get_all_builtin_tools() -> list[type]:
    return [ReadFileTool]
