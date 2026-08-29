#!/usr/bin/env sh
# Build the documentation site (GitHub Pages) into _site/.
#   tools/site.sh [OUT_DIR]
# Requires python3 + jsonschema, pandoc, and Graphviz dot (for --view).
#
#   _site/index.html           README
#   _site/spec/*.html          docs/*.md, hand-written documents
#   _site/catalog/index.html   catalog reference, generated from data/catalog
#   _site/models/*.html        one --view page per data/models/*.json
#   _site/schemas, _site/data  copies, so links to them resolve
set -eu
here=$(cd "$(dirname "$0")" && pwd)
repo_dir=$(dirname "$here")
out=${1:-"$repo_dir/_site"}
style="$repo_dir/docs/style"
repo_url=${ARMATURE_REPO_URL:-$(git -C "$repo_dir" remote get-url origin 2>/dev/null | sed -E 's#^git@github\.com:#https://github.com/#; s#\.git$##')}

rm -rf "$out"
mkdir -p "$out/spec" "$out/catalog" "$out/models" "$out/style"
touch "$out/.nojekyll"
cp "$style/catalog.css" "$style/doc.css" "$out/style/"
cp -r "$repo_dir/schemas" "$out/schemas"
mkdir -p "$out/data" && cp -r "$repo_dir/data/models" "$out/data/models"

# A hand-written document: doc IN.md OUT.html ROOT SOURCE_DIR [extra pandoc args...]
doc() {
  in=$1; dst=$2; root=$3; src=$4; shift 4
  pandoc "$in" --from markdown --to html5 --standalone --section-divs \
    --toc --toc-depth=3 \
    --template "$style/doc.html" --lua-filter "$style/doc.lua" \
    --metadata root="$root" --metadata source-dir="$src" \
    ${repo_url:+--metadata repo="$repo_url"} \
    "$@" --output "$dst"
  echo "wrote ${dst#$out/}"
}

doc "$repo_dir/README.md" "$out/index.html" "" "" \
  --metadata pagetitle="Overview" --metadata kind="Overview" --metadata current-index=true

for f in "$repo_dir"/docs/*.md; do
  name=$(basename "$f" .md)
  case "$name" in CATALOG-REFERENCE|PLAN-DOCUMENTATION) continue ;; esac
  slug=$(printf %s "$name" | tr '[:upper:]' '[:lower:]')
  title=$(sed -n 's/^# //p' "$f" | head -1)
  doc "$f" "$out/spec/$slug.html" "../" "docs" \
    --metadata pagetitle="$title" --metadata kind="Documentation" --metadata "current-$slug=true"
done

# Catalog reference: regenerate the Markdown, then render it with the catalog style.
python3 "$here/armature" --document catalog -o "$repo_dir/docs/CATALOG-REFERENCE.md"
"$style/catalog.sh" "$repo_dir/docs/CATALOG-REFERENCE.md" "$out/catalog/index.html" >/dev/null
sed -i -e 's#<link rel="stylesheet" href="[^"]*catalog\.css">#<link rel="stylesheet" href="../style/catalog.css">#' \
  -e 's#href="\(SPECIFICATION\|ARMATURE-MODEL_JSON\|GLOSSARY\|ARCHITECTURE\|CATALOG-DOCUMENTATION\)\.md#href="../spec/\L\1\E.html#g' \
  "$out/catalog/index.html"
echo "wrote catalog/index.html"

# Model views.
python3 "$here/armature" --view "$repo_dir"/data/models/*.json -o "$out/models" >/dev/null
index="$out/models/index.md"
{
  echo "# Model views"
  echo
  echo "Each page is the self-contained inspector produced by \`armature --view\` for one model of"
  echo "\`data/models/\`: its quantities, occurrences, compositions and bindings, with the value graph."
  echo
  echo '<ul class="models">'
  for m in "$repo_dir"/data/models/*.json; do
    n=$(basename "$m" .json)
    echo "<li><a href=\"$n.html\">$n</a> — <a href=\"../data/models/$n.json\">JSON</a></li>"
  done
  echo '</ul>'
} > "$index"
doc "$index" "$out/models/index.html" "../" "" \
  --metadata pagetitle="Model views" --metadata kind="Model views" --metadata current-models=true
rm "$index"

echo "site: $out"
