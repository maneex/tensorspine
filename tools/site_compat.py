#!/usr/bin/env python3
"""Historical site entry points; current pages use the canonical vocabulary."""
import html
import json
from pathlib import Path
import re
import sys


def build(site):
    site = Path(site)
    reference = site / 'primitive-library/index.html'
    text = reference.read_text()
    text = re.sub(r'(<section id="primitive-([^" ]+)-1\.0\.0")',
                  lambda m: f'<a id="contract-{m[2]}-1.0.0"></a>' + m[1], text)
    reference.write_text(text)
    routes = {
        'catalog/index.html': '../primitive-library/index.html',
        'spec/catalog-reference.html': '../primitive-library/index.html',
        'spec/catalog-documentation.html': 'primitive-library-documentation.html',
        'spec/tensorspine-catalog-unit.html': 'tensorspine-primitive-library-unit.html',
    }
    for old, current in routes.items():
        page = site / old
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
                        '<title>Primitive Library</title>'
                        f'<link rel="canonical" href="{html.escape(current)}">'
                        f'<script>location.replace({json.dumps(current)} + location.hash)</script>'
                        f'<p>Continue to the <a href="{html.escape(current)}">Primitive Library documentation</a>.</p></html>')


if __name__ == '__main__':
    build(sys.argv[1])
