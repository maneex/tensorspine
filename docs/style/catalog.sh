#!/usr/bin/env sh
# Render the generated catalog Markdown as a styled HTML page.
#   docs/style/catalog.sh [IN.md] [OUT.html]
# Defaults: docs/CATALOG-REFERENCE.md -> docs/CATALOG-REFERENCE.html
# Regenerate the Markdown first with: python3 tools/tensorspine --document catalog -o docs/CATALOG-REFERENCE.md
set -eu
here=$(cd "$(dirname "$0")" && pwd)
docs=$(dirname "$here")
in=${1:-"$docs/CATALOG-REFERENCE.md"}
out=${2:-"$docs/CATALOG-REFERENCE.html"}
css=$(python3 -c "import os,sys;print(os.path.relpath(sys.argv[1],sys.argv[2]))" "$here/catalog.css" "$(dirname "$out")")
pandoc "$in" \
  --from markdown \
  --to html5 \
  --standalone \
  --section-divs \
  --toc --toc-depth=4 \
  --template "$here/catalog.html" \
  --lua-filter "$here/catalog.lua" \
  --css "$css" \
  --metadata pagetitle="Tensorspine reference catalog" \
  --output "$out"
echo "wrote $out"
