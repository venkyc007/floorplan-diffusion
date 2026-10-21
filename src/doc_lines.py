"""Fill in line ranges of functions / classes in a markdown file, so docs never point at stale lines.

    python -m src.doc_lines docs/code_walkthrough.md

In the markdown, write {{path::name}} (name = a function, a class, or Class.method). The tool replaces it with
<!--path::name-->L<first>–<last> (the comment is invisible when rendered and lets the tool update the range on
the next run). Unknown names are reported and left as they are.
"""

import ast
import re
import sys
from pathlib import Path

from src.utils import PROJECT_ROOT

TOKEN = re.compile(r"\{\{([^:{}]+)::([\w.]+)\}\}|<!--([^:<>]+)::([\w.]+)-->L\d+–\d+")


def spans(path):
    """{qualified name: (first line, last line)} of every def / class in a Python file (decorators included)."""
    tree = ast.parse((PROJECT_ROOT / path).read_text(encoding="utf-8"))
    out = {}

    def walk(node, prefix=""):
        for n in ast.iter_child_nodes(node):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + n.name
                first = min([n.lineno] + [d.lineno for d in n.decorator_list])
                out.setdefault(name, (first, n.end_lineno))
                walk(n, name + ".")
    walk(tree)
    return out


def main(md):
    p = Path(md)
    text = p.read_text(encoding="utf-8")
    cache, missing = {}, []

    def sub(m):
        path, name = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        try:
            s = cache.setdefault(path, spans(path)).get(name)
        except (FileNotFoundError, SyntaxError):
            s = None
        if s is None:
            missing.append(f"{path}::{name}")
            return m.group(0)
        return f"<!--{path}::{name}-->L{s[0]}–{s[1]}"

    new = TOKEN.sub(sub, text)
    p.write_text(new, encoding="utf-8")
    print(f"{md}: {len(TOKEN.findall(text))} references, {len(missing)} not found")
    for x in missing:
        print("  not found:", x)


if __name__ == "__main__":
    main(sys.argv[1])
