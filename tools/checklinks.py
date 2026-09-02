#!/usr/bin/env python3
"""Check links in a built TensorSpine documentation site and its Markdown sources."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import os
import posixpath
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INLINE = re.compile(r'!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))')
REFERENCE = re.compile(r'^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))', re.MULTILINE)
HTML_LINK = re.compile(r'\b(?:href|src)\s*=\s*["\']([^"\']+)["\']', re.IGNORECASE)


def _relative(target):
    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(('/', '#')):
        return None
    return unquote(parsed.path)


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.targets = []

    def handle_starttag(self, _tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src') and value is not None:
                self.targets.append(value)


def _site_target(page, target, site):
    rel = _relative(target)
    if rel is None or rel == '':
        return None
    path = os.path.normpath(os.path.join(os.path.dirname(page), rel))
    if os.path.isdir(path) or rel.endswith('/'):
        path = os.path.join(path, 'index.html')
    try:
        if os.path.commonpath((site, path)) != site:
            return path
    except ValueError:
        return path
    return path


def check_site(site):
    findings = []
    site = os.path.abspath(site)
    for page in sorted(_walk(site, '.html')):
        with open(page, encoding='utf-8') as stream:
            text = stream.read()
        if '{{' in text:
            findings.append(f"{os.path.relpath(page, site)}: unexpanded '{{{{' template marker")
        parser = Links()
        parser.feed(text)
        for target in parser.targets:
            resolved = _site_target(page, target, site)
            if resolved and not os.path.isfile(resolved):
                findings.append(
                    f"{os.path.relpath(page, site)}: {target!r} -> "
                    f"{os.path.relpath(resolved, site)} does not exist")
    return findings


def _walk(root, suffix):
    for directory, _subdirs, files in os.walk(root):
        for name in files:
            if name.endswith(suffix):
                yield os.path.join(directory, name)


def _tracked_markdown(repo):
    result = subprocess.run(
        ['git', '-C', repo, 'ls-files', '-z', '*.md'],
        check=True,
        stdout=subprocess.PIPE,
    )
    return [path.decode() for path in result.stdout.split(b'\0') if path]


def _targets(text):
    for pattern in (INLINE, REFERENCE):
        for match in pattern.finditer(text):
            yield match.group(1) or match.group(2), text.count('\n', 0, match.start()) + 1
    for match in HTML_LINK.finditer(text):
        yield match.group(1), text.count('\n', 0, match.start()) + 1


def check_markdown(repo):
    findings = []
    tracked = set(_tracked_markdown(repo))
    tracked_all = set(subprocess.run(
        ['git', '-C', repo, 'ls-files', '-z'], check=True, stdout=subprocess.PIPE
    ).stdout.decode().split('\0'))
    for source in sorted(tracked):
        absolute = os.path.join(repo, source)
        with open(absolute, encoding='utf-8') as stream:
            text = stream.read()
        for target, line in _targets(text):
            rel = _relative(target)
            if rel is None or rel == '':
                continue
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(source), rel))
            if resolved == 'docs' or resolved.startswith('docs/'):
                if resolved not in tracked_all:
                    findings.append(f"{source}:{line}: {target!r} points to untracked {resolved}")
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('site', help='built site directory')
    parser.add_argument('--repo', default=ROOT, help='repository root')
    args = parser.parse_args(argv)
    site = os.path.abspath(args.site)
    repo = os.path.abspath(args.repo)
    if not os.path.isdir(site):
        parser.error(f'not a directory: {site}')
    findings = check_site(site) + check_markdown(repo)
    for finding in findings:
        print(finding)
    if findings:
        print(f'links: {len(findings)} finding(s)', file=sys.stderr)
        return 1
    print('links: all relative site links resolve; tracked Markdown names tracked docs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
