#!/usr/bin/env sh
# Build the documentation site (GitHub Pages) into _site/.
#   tools/site.sh [OUT_DIR]
# Requires python3 + jsonschema, pandoc, and Graphviz dot (for --view).
#
#   _site/index.html           README
#   _site/spec/*.html          docs/*.md, hand-written documents
#   _site/catalog/index.html   catalog reference, generated from data/catalog
#   _site/status/index.html    generated catalog, corpus and verification state
#   _site/branch-ledger/       generated work remaining per model and generator
#   _site/models/*.html        one --view page per data/models/*.json
#   _site/schemas, _site/data  copies, so links to them resolve
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(dirname "$here")
out=${1:-"$repo_dir/_site"}
style="$repo_dir/docs/style"
repo_url=${TENSORSPINE_REPO_URL:-$(git -C "$repo_dir" remote get-url origin 2>/dev/null | sed -E 's#^git@github\.com:#https://github.com/#; s#\.git$##')}

rm -rf "$out"
mkdir -p "$out/spec" "$out/catalog" "$out/status" "$out/branch-ledger" \
  "$out/models" "$out/style" "$out/assets"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
touch "$out/.nojekyll"
cp "$style/catalog.css" "$style/doc.css" "$out/style/"
cp -r "$repo_dir/schemas" "$out/schemas"
mkdir -p "$out/data" && cp -r "$repo_dir/data/models" "$out/data/models"
for asset in "$repo_dir"/docs/*.svg; do
  [ -e "$asset" ] || continue
  cp "$asset" "$out/assets/"
done

# The site navigation for one page, from the single source docs/style/nav.html:
# %ROOT% becomes the page's own path to the site root, and the entry whose
# `data-nav` is CURRENT is marked as the page being read.
#   navbar OUT.html ROOT CURRENT
navbar() {
  sed -e "s|%ROOT%|$2|g" -e "s|data-nav=\"$3\"|data-nav=\"$3\" class=\"current\"|" \
    "$style/nav.html" > "$1"
}

# A hand-written document: doc IN.md OUT.html ROOT SOURCE_DIR NAV_ID [extra pandoc args...]
doc() {
  in=$1; dst=$2; root=$3; src=$4; nav_id=$5; shift 5
  navbar "$work/nav.html" "$root" "$nav_id"
  pandoc "$in" --from markdown --to html5 --standalone --section-divs \
    --toc --toc-depth=3 \
    --template "$style/doc.html" --lua-filter "$style/doc.lua" \
    --include-before-body "$work/nav.html" \
    --metadata root="$root" --metadata source-dir="$src" \
    ${repo_url:+--metadata repo="$repo_url"} \
    "$@" --output "$dst"
  echo "wrote ${dst#$out/}"
}

doc "$repo_dir/README.md" "$out/index.html" "" "" index \
  --metadata pagetitle="Overview"

for f in "$repo_dir"/docs/*.md; do
  name=$(basename "$f" .md)
  case "$name" in CATALOG-REFERENCE|PLAN-DOCUMENTATION|TENSORSPINE_MODEL_TSPL) continue ;; esac
  slug=$(printf %s "$name" | tr '[:upper:]' '[:lower:]')
  title=$(sed -n 's/^# //p' "$f" | head -1)
  doc "$f" "$out/spec/$slug.html" "../" "docs" "$slug" \
    --metadata pagetitle="$title"
done

# Generated pages. Their Markdown exists only in the build's temporary directory.
python3 "$here/tensorspine" --document catalog -o "$work/CATALOG-REFERENCE.md" \
  --link-base "$repo_dir/docs"
navbar "$work/nav-catalog.html" "../" catalog
"$style/catalog.sh" "$work/CATALOG-REFERENCE.md" "$out/catalog/index.html" \
  "$work/nav-catalog.html" >/dev/null
sed -i -e 's#<link rel="stylesheet" href="[^"]*catalog\.css">#<link rel="stylesheet" href="../style/catalog.css">#' \
  -e 's#href="\(SPECIFICATION\|TENSORSPINE-MODEL_JSON\|GLOSSARY\|ARCHITECTURE\|CATALOG-DOCUMENTATION\)\.md#href="../spec/\L\1\E.html#g' \
  "$out/catalog/index.html"
echo "wrote catalog/index.html"

python3 "$here/tensorspine" --document status \
  --capabilities "$repo_dir/generators/reference/capabilities.json" \
  --capabilities "$repo_dir/generators/zml/capabilities.json" \
  -o "$work/status.md"
doc "$work/status.md" "$out/status/index.html" "../" "" status \
  --metadata pagetitle="Status"

python3 "$here/tensorspine" --document branch-ledger \
  --capabilities "$repo_dir/generators/reference/capabilities.json" \
  -o "$work/branch-ledger.md"
doc "$work/branch-ledger.md" "$out/branch-ledger/index.html" "../" "" branch-ledger \
  --metadata pagetitle="Branch ledger"

# Model views. They are self-contained pages, so they are handed the same
# navigation as everything else rather than linking to a site they may not be in.
navbar "$work/nav-models.html" "../" models
python3 "$here/tensorspine" --view "$repo_dir"/data/models/*.json -o "$out/models" \
  --site-nav "$work/nav-models.html" >/dev/null
index="$out/models/index.md"
{
  echo "# Model views"
  echo
  echo "Each page is the self-contained inspector produced by \`tensorspine --view\` for one model of"
  echo "\`data/models/\`: its quantities, occurrences, compositions and bindings, with the value graph."
  echo
  echo '<ul class="models">'
  for m in "$repo_dir"/data/models/*.json; do
    n=$(basename "$m" .json)
    echo "<li><a href=\"$n.html\">$n</a> — <a href=\"../data/models/$n.json\">JSON</a></li>"
  done
  echo '</ul>'
} > "$index"
doc "$index" "$out/models/index.html" "../" "" models \
  --metadata pagetitle="Model views"
rm "$index"

# The explainer is an illustration of the goal, not evidence of a deployed
# three-engine system. Its own drawing and animation remain untouched.
if [ -d "$repo_dir/docs/explainer" ]; then
  mkdir -p "$out/explainer"
  cp -r "$repo_dir/docs/explainer/." "$out/explainer/"
  if [ -f "$out/explainer/index.html" ]; then
    sed -i '0,/<header/{s#<header#<p class="aim-label"><strong>Illustration of the aim.</strong> This is not a deployment produced by this repository. See <a href="../index.html#with-tensorspine">how TensorSpine connects a model lab to serving applications</a>.</p>\n<header#}' \
      "$out/explainer/index.html"
  fi
fi

python3 "$here/checklinks.py" "$out" --repo "$repo_dir"

echo "site: $out"
