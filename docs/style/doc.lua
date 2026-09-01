-- Pandoc filter for the TensorSpine site (tools/site.sh): hand-written documents.
--
--   * Relative links between repository files are rewritten to their location
--     on the site: docs/*.md become spec/*.html, CATALOG-REFERENCE.md the
--     catalog page, README.md the index, schemas/ and data/ their copies. A
--     relative path with no counterpart on the site links to the file on
--     GitHub (metadata `repo`). `root` is the page's relative path to the
--     site root ("" or "../").
--   * Tables get default column widths and a scrolling wrapper, as in
--     catalog.lua.

local root, repo, source_dir = '', nil, ''

function Meta(m)
  if m.root then root = pandoc.utils.stringify(m.root) end
  if m.repo then repo = pandoc.utils.stringify(m.repo) end
  if m['source-dir'] then source_dir = pandoc.utils.stringify(m['source-dir']) end
end

-- Normalise `source_dir .. '/' .. rel` against the repository root.
local function repo_path(rel)
  local parts = {}
  local joined = (source_dir ~= '' and source_dir .. '/' or '') .. rel
  for seg in joined:gmatch('[^/]+') do
    if seg == '..' then table.remove(parts)
    elseif seg ~= '.' then table.insert(parts, seg) end
  end
  return table.concat(parts, '/')
end

local function site_target(path)
  local dir, file = path:match('^(.-)([^/]*)$')
  if dir == 'docs/' and file:match('%.md$') then
    if file == 'CATALOG-REFERENCE.md' then return 'catalog/index.html' end
    return 'spec/' .. file:gsub('%.md$', '.html'):lower()
  end
  if dir == '' and file == 'README.md' then return 'index.html' end
  if dir:match('^schemas/') or dir:match('^data/') then return path end
  return nil
end

function Link(l)
  local target = l.target
  if target:match('^%a[%w+.-]*:') or target:match('^#') or target:match('^/') then return nil end
  local path, frag = target:match('^([^#]*)(#?.*)$')
  local site = site_target(repo_path(path))
  if site then
    l.target = root .. site .. frag
  elseif repo then
    l.target = repo .. '/blob/main/' .. repo_path(path) .. frag
  end
  return l
end

function Table(tbl)
  for i, spec in ipairs(tbl.colspecs) do
    tbl.colspecs[i] = { spec[1], pandoc.ColWidthDefault }
  end
  return pandoc.Div({ tbl }, pandoc.Attr('', { 'table-wrap' }))
end

return { { Meta = Meta }, { Link = Link, Table = Table } }
