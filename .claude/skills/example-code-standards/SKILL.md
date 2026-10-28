---
name: example-code-standards
description: Pre-PR quality gate for the ai-agents-frameworks repo. Checks every changed example file, README, OUTPUTS.md and lockfile against the repo's teaching-code standards (file layout, tutorial docstring, comments, simplicity, public APIs, output quality, docs links) and fixes violations before the PR goes out. Use it before opening a pull request, before pushing new commits to an existing PR branch, after a framework upgrade or new example, and whenever someone asks to "check the standards", "review before the PR" or "make sure this follows the repo conventions". Run it even when the change looks small; version bumps routinely leave stale comments and OUTPUTS notes behind.
---

# Example code standards (pre-PR gate)

Every framework folder in this repo is a hands-on tutorial. The reader is a **Python developer meeting this framework for the first time**: they open one numbered file, read it top to bottom, run it, and compare the terminal with `OUTPUTS.md`. Every rule below serves that reader. When a rule and the reader's understanding conflict, the reader wins. Say so in the report.

## When to run

- Before `gh pr create`.
- Before pushing new commits to a branch that already has an open PR.
- After a framework version bump, even with no new examples.

## Workflow

1. **Scope the diff.** List what the PR changes (`git diff --stat origin/main...HEAD`, or `HEAD~N` for the new commits on an existing PR). Only lines the branch adds or changes are in scope. Note any violations already present in untouched lines as `LEFT (pre-existing)`, but don't fix them in an upgrade PR.
2. **Calibrate.** Read 2-3 already-merged examples in the same folder. New code must look like it belongs there: same env-setup pattern, section header style, print style, OUTPUTS heading format.
3. **Check** each changed file against the standards below.
4. **Fix** clear violations directly, keeping each fix surgical and leaving the example's teaching intent unchanged. Re-run every `.py` file you touch with `uv run python <file>`; it must exit 0.
5. **Verify** with the commands in [Mechanical checks](#mechanical-checks).
6. **Commit** the fixes as a new commit (`style(<folder>): align examples with repo conventions`). Never amend commits that are already pushed.
7. **Report** the findings in the format under [Report](#report).

## Standards

### 1. File layout, in this order

1. **Imports.** Stdlib, then third-party, then local (`from settings import settings`, `utils`), with a blank line between groups and no unused imports. Import from the shortest public path (`from mcp.server import MCPServer`, not `mcp.server.mcpserver`).
2. **Env setup.** Use the pattern this folder already uses, e.g. `os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()` or `load_dotenv()`. An example that never calls an LLM skips `settings` entirely, and its docstring says no API key is needed.
3. **Tutorial string block.** A string literal placed after the env setup:

   ```python
   """
   -------------------------------------------------------
   In this example, we explore <Framework> with the following features:
   - <feature>
   - <feature>

   <One paragraph, at most ~6 lines: what it shows and when you would use it.>

   For more details, visit:
   <URL>
   -------------------------------------------------------
   """
   ```

   - Use the opener sentence exactly as written.
   - Bullets start with `- `, and a blank line comes before the paragraph.
   - The URL goes on its own line.
4. **Numbered sections.** Mark them `# --- 1. Description ---`, `# --- 2. ... ---`, numbered in order with no gaps. Numbering continues across helpers and `main()`. A file with several scenarios uses the separator block plus `print("=== Example N: Title ===")`.
5. **Async files.** Use `async def main() -> None:` plus `if __name__ == "__main__": asyncio.run(main())`.
6. **File name.** Use a zero-padded `NN_snake_case.py` that continues the folder's sequence.

### 2. Code

- **Settings.** The model and API key always come from `settings` (`settings.OPENAI_MODEL_NAME`, `.get_secret_value()`). The only exception is a feature locked to one model. In that case, the docstring and the README both name the model and say that `OPENAI_MODEL_NAME` is not used. If a feature needs a different kind of model (e.g. reasoning), add a settings field with a default rather than hardcoding a string.
- **No interactive input.** Never use `input()` or stdin; demo inputs are hardcoded.
- **Tools.** They return mock data, have type hints, and have a one-line docstring, since the LLM reads it.
- **Minimum code to teach the feature:**
  - no speculative options;
  - no single-use classes or abstractions;
  - small helpers only for formatting output;
  - every argument either *is* the feature or is needed to run; don't copy default-off flags from a neighbouring file.
- **Guards only for what can happen:**
  - `.get()` is fine for keys typed `Optional`/`NotRequired`.
  - A loop or fallback for real model variability (e.g. sequential vs. parallel tool calls) is fine, with a short comment naming the case.
  - Don't guard values the framework already guarantees.
- **Public APIs only.** Don't use underscore names, private state keys (even through a public constant whose value is private), or constants imported from implementation modules. Look for a public route first: hooks, middleware, callbacks, message tags, documented fields. If none exists, add one comment saying why.
- **Type hints** on parameters and returns, including wrap/middleware handlers; type `next`/`call_next` with the framework's public alias.
- **Python style.** No semicolons, one statement per line.
- **Length.** Most examples are 60-160 lines; anything longer needs a reason. In a multi-scenario file, a scenario that adds only one fact (one field, one `isinstance`) gets folded into the scenario that covers the general case.

### 3. Comments and prose

- A comment either explains *why* or labels a step. It never restates the code.
- Single-line comments; two lines at most when the point really needs it.
- **No changelog narration** in `.py` files or in new OUTPUTS notes: no "new in vX", "since 1.60", "previously…", "ported to…", PR numbers, or version tags in bullets. Version history belongs in the README version line and the PR description.
- No comments addressed to reviewers, about test runs, or about what the model did on some run.
- Helper docstrings are one line.
- **Upgrades go stale silently.** For each behaviour the upgrade changed, grep the folder's comments, docstrings and OUTPUTS notes for claims about it (e.g. "X raises without Y"), and fix the ones in files the PR touches.

### 4. Terminal output

- Output is grouped under `=== ... ===` headers with short, descriptive prefixes.
- **It proves the feature fired:** tool calls and results, hook or middleware firing, interrupt ids, blocked vs. allowed, before/after.
- **Never print a raw `repr()` of an SDK object.** Extract the fields that matter instead.
- Agent replies are constrained with an explicit length in the instructions ("answer in 1-2 sentences", "3 bullets max"). "Be concise" alone is not enough.
- Shutdown tracebacks the framework writes to stderr are acceptable when the exit code is 0. Leave them out of OUTPUTS.md.

### 5. Docs links

- **Link a page that actually documents the feature.** Prefer the official docs page with an anchor. A changelog or release page is not documentation.
- **When no docs page exists, link GitHub source or a sample pinned to a release tag** (`/blob/v1.2.3/...#L120`), never `main`.
- **Check every new URL** with `curl -sIL -o /dev/null -w '%{http_code}' <url>`; it must end in 200.

### 6. README, OUTPUTS.md and generated files

- **Framework `README.md`:**
  - the version line;
  - one examples-table row per new file, in the table's existing columns;
  - a run command if the README lists one per file;
  - a dependency version if the README has that section.
- **Root `README.md`:** only this framework's Version cell changes.
- **`OUTPUTS.md` header:** version line updated. If older sections were captured on an older version, say so honestly.
- **`OUTPUTS.md` sections:** one section per new or changed file, using **the heading format this file already uses** (folders differ: `## 00_hello_world.py` vs `## 00. Hello World (\`00_hello_world.py\`)`). Each section has the output in a fenced block and an optional note of at most 3 sentences that doesn't repeat the docstring. Sections for unchanged files stay as they are.
- **When to re-capture.** Re-capture when the output's *structure* changes, not when only the LLM's wording or token counts change. Run it under the conditions the header describes (e.g. an empty config dir). Run-specific ids are fine.
- **Notes that cite example numbers** are updated whenever examples are renumbered or merged.
- **Generated artifacts** go in the folder's `res/`. Side effects a run leaves on tracked files (`res/*`, `*.db`, logs) are reverted, not committed.

## Mechanical checks

Run these from the repo root for each changed framework folder `$F`. They mirror CI (`.github/workflows/smoke-test.yml`) plus the checks CI can't do.

```bash
# Scope: only $F/ and the root README version cell should change
git diff --stat origin/main...HEAD

# Lockfile and compile, as CI does
(cd "$F" && uv lock --check && uv sync --frozen && uv run --no-sync python -m py_compile *.py)

# Red flags in changed example files (review every hit by hand)
changed() { git diff --name-only --diff-filter=AM origin/main...HEAD -- "$F/[0-9]*.py"; }
changed | xargs grep -nE 'input\(|;\s*$|"gpt-[0-9]|"gemini-|"claude-'
changed | xargs grep -nEi '(new in|added in|since) v?[0-9]+\.[0-9]+|\(v[0-9]+\.[0-9]+'
changed | xargs grep -nE 'from [a-z_.]+\._[a-z]|\._[a-z]'
changed | xargs grep -L 'For more details, visit:'

# The working tree is clean after the runs (no res/ or db side effects)
git status --short
```

Treat a grep hit as a lead to check, not a confirmed violation. A hardcoded model is fine in the one-model exception above, for example.

## Report

List the findings per framework, most severe first. Each finding gets one P0-P3 label, a one-line reason, `file:line`, and a status.

| Level | Meaning |
|---|---|
| P0 | Example crashes, leaks a secret, or teaches wrong behaviour |
| P1 | Misleading output or docstring, a stale claim about the API, a broken docs link, a missing OUTPUTS section |
| P2 | Standards violation that keeps the meaning intact: changelog narration, a private API, a hardcoded model, a raw repr, over-engineering |
| P3 | Polish: wording, type hints, small duplication |

The status is `FIXED`, `LEFT (pre-existing)` or `LEFT (<reason>)`. End with the mechanical-check results, one line each, and say explicitly if no finding is P0-P1.
