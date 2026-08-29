#!/usr/bin/env sh
# Render the generated catalog Markdown as a styled HTML page.
#   docs/style/catalog.sh [IN.md] [OUT.html] [NAV.html]
# Defaults: docs/CATALOG-REFERENCE.md -> docs/CATALOG-REFERENCE.html
# NAV.html is the site navigation for this page (tools/site.sh resolves
# nav.html for it); without it the page keeps the wordmark and drops the links,
# which would lead nowhere outside the site.
# Regenerate the Markdown first with: python3 tools/tensorspine --document catalog -o docs/CATALOG-REFERENCE.md
set -eu
here=$(cd "$(dirname "$0")" && pwd)
docs=$(dirname "$here")
in=${1:-"$docs/CATALOG-REFERENCE.md"}
out=${2:-"$docs/CATALOG-REFERENCE.html"}
nav=${3:-}
css=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))" "$here/catalog.css" "$(dirname "$out")")

if [ -z "$nav" ]; then
  nav=$(mktemp)
  trap 'rm -f "$nav"' EXIT
  cat > "$nav" <<'BAR'
<nav class="sitebar">
  <span class="wordmark">Tensorspine</span>
  <span class="sitekind">Reference catalog</span>
</nav>
BAR
fi
pandoc "$in" \
  --from markdown \
  --to html5 \
  --standalone \
  --section-divs \
  --toc --toc-depth=4 \
  --template "$here/catalog.html" \
  --lua-filter "$here/catalog.lua" \
  --css "$css" \
  --include-before-body "$nav" \
  --metadata pagetitle="Tensorspine reference catalog" \
  --output "$out"
echo "wrote $out"
