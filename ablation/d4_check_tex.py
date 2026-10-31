"""Static checks for report_v2/main.tex (no LaTeX engine is available in this environment): matched \\begin/\\end, brace balance, every \\ref has a \\label, every \\cite key has a \\bibitem, duplicate labels,
included figure files exist, every old label survives, and no non-ASCII character in a new or changed line. Exit 1 on any problem. This does NOT replace compiling: overfull boxes and font problems are invisible to it."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tex = (ROOT / "report_v2" / "main.tex").read_text(encoding="utf8")
old = (ROOT / "report" / "Capstone_report_Sem5.tex").read_text(encoding="utf8")
COMMENT = re.compile(r"(?<!\\)%.*")


def strip(t):
    return "\n".join(COMMENT.sub("", ln) for ln in t.splitlines())


body = strip(tex)
problems = []
stack = []
for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
    if kind == "begin":
        stack.append(name)
    elif not stack or stack.pop() != name:
        problems.append(f"environment mismatch at \\end{{{name}}}")
if stack:
    problems.append(f"unclosed environments: {stack}")
depth = 0
clean = body.replace("\\{", "").replace("\\}", "")
for ch in clean:
    depth += (ch == "{") - (ch == "}")
    if depth < 0:
        problems.append("negative brace depth")
        break
if depth:
    problems.append(f"brace imbalance {depth}")
labels = re.findall(r"\\label\{([^}]+)\}", body)
dup = {x for x in labels if labels.count(x) > 1}
if dup:
    problems.append(f"duplicate labels {sorted(dup)}")
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", body))
if refs - set(labels):
    problems.append(f"undefined references {sorted(refs - set(labels))}")
bib = set(re.findall(r"\\bibitem\{([^}]+)\}", body))
cites = {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", body) for k in g.split(",")}
if cites - bib:
    problems.append(f"undefined citations {sorted(cites - bib)}")
for f in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)):
    if not (ROOT / "report_v2" / f).exists():
        problems.append(f"missing figure file {f}")
lost = set(re.findall(r"\\label\{([^}]+)\}", strip(old))) - set(labels)
if lost:
    problems.append(f"old labels lost: {sorted(lost)}")
oldlines = set(old.splitlines())
for n, ln in enumerate(tex.splitlines(), 1):
    if ln not in oldlines and any(ord(c) > 127 for c in ln):
        problems.append(f"line {n}: non-ASCII character in a new or changed line: {ln[:90]}")
print(f"{len(tex.splitlines())} lines; {len(labels)} labels, {len(refs)} refs, {len(cites)} cited keys, {len(bib)} bibitems ({len(bib - cites)} not cited: {sorted(bib - cites)}); {len(problems)} problems")
for p in problems:
    print("  PROBLEM:", p)
sys.exit(1 if problems else 0)
