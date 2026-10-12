#!/usr/bin/env bash
# Label a PR by review size (size:XXS..size:XL) and by every framework folder it
# touches (framework:<folder>), removing size/framework labels that no longer apply.
# Run by .github/workflows/pr-labels.yml on every PR update; run it by hand to
# (re)label an existing PR: bash .github/scripts/label_pr.sh <pr-number>
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pr=$1
api="repos/{owner}/{repo}"

# Size counts reviewable lines: the generated lockfile and captured example output
# would otherwise put every version bump in the largest tier.
SIZE_IGNORE='(^|/)(uv[.]lock|OUTPUTS[.]md)$'
# name|first line count of the next tier|color; the last tier has no upper bound
SIZE_TIERS="size:XXS|10|3cbf00
size:XS|50|5d9801
size:S|150|7f7203
size:M|300|a14c05
size:L|600|c32607
size:XL||e50009"
FRAMEWORK_COLOR=1d76db
# Top-level folders with a pyproject.toml that are not agent frameworks
NOT_FRAMEWORKS=" study-agents-differences "

files=$(gh api --paginate "$api/pulls/$pr/files?per_page=100" \
  --jq '.[] | "\(.filename)\t\(.additions + .deletions)"')
lines=$(awk -F'\t' -v ignore="$SIZE_IGNORE" '$1 !~ ignore {s += $2} END {print s + 0}' <<< "$files")

# --- Desired labels, one "name|color|description" per line ---
desired=""
low=0
while IFS='|' read -r name max color; do
  range="${low}+"
  [ -n "$max" ] && range="${low}-$((max - 1))"
  if [ -z "$desired" ] && { [ -z "$max" ] || [ "$lines" -lt "$max" ]; }; then
    desired="$name|$color|Reviewable diff of $range lines (excludes uv.lock, OUTPUTS.md)"
  fi
  low=${max:-$low}
done <<< "$SIZE_TIERS"

# A framework is a top-level folder with a pyproject.toml, either on the checked-out
# branch or added by this PR (a new framework)
for dir in $(awk -F'\t' '$1 ~ /\// {sub(/\/.*/, "", $1); print $1}' <<< "$files" | sort -u); do
  case "$NOT_FRAMEWORKS" in *" $dir "*) continue ;; esac
  if [ -f "$dir/pyproject.toml" ] \
    || awk -F'\t' -v f="$dir/pyproject.toml" '$1 == f {found = 1} END {exit !found}' <<< "$files"; then
    desired+=$'\n'"framework:$dir|$FRAMEWORK_COLOR|Touches the $dir/ framework folder"
  fi
done

# --- Create missing labels, then sync the PR's size/framework labels ---
existing=$(gh api --paginate "$api/labels?per_page=100" --jq '.[].name')
while IFS='|' read -r name color description; do
  if ! awk -v n="$name" '$0 == n {found = 1} END {exit !found}' <<< "$existing"; then
    gh api -X POST "$api/labels" -f name="$name" -f color="$color" -f description="$description" >/dev/null
  fi
done <<< "$desired"

wanted=$(cut -d'|' -f1 <<< "$desired")
current=$(gh api "$api/issues/$pr/labels" --jq '.[].name | select(test("^(size|framework):"))')
for name in $current; do
  if ! awk -v n="$name" '$0 == n {found = 1} END {exit !found}' <<< "$wanted"; then
    gh api -X DELETE "$api/issues/$pr/labels/$name" >/dev/null
  fi
done
add_args=()
for name in $wanted; do
  add_args+=(-f "labels[]=$name")
done
gh api -X POST "$api/issues/$pr/labels" "${add_args[@]}" >/dev/null

echo "#$pr: $(tr '\n' ' ' <<< "$wanted")($lines reviewable lines)"
