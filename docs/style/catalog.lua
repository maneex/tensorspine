-- Pandoc filter for the Armature catalog rendering (docs/style/catalog.sh).
--
--   * The generator puts an explicit anchor `<a id="X" name="X"></a>` on the
--     line before headings it cross-references. Pandoc keeps the anchor and
--     invents its own id for the heading; here the anchor's id is moved onto
--     the heading (so --section-divs sections and the TOC use the catalog's
--     identities) and the anchor is dropped. A heading without an anchor whose
--     pandoc id collides with an anchor id is suffixed.
--   * The "Contents" section is removed: the template's sidebar replaces it.
--   * Every table is wrapped in <div class="table-wrap"> so wide tables scroll
--     inside the content column; the key/value strip after a contract lede is
--     tagged class="facts".
--   * "Tags: …" paragraphs become <p class="tags">.

local function anchor_id(block)
  -- pandoc reads `<a id="X" name="X"></a>` as one or two raw HTML inlines.
  if block.t ~= 'Para' and block.t ~= 'Plain' then return nil end
  local text = ''
  for _, inl in ipairs(block.content) do
    if inl.t == 'RawInline' then text = text .. inl.text
    elseif inl.t ~= 'Space' and inl.t ~= 'SoftBreak' then return nil end
  end
  return text:match('^<a%s+id="([^"]+)"[^>]*>%s*</a>$')
end

function Pandoc(doc)
  local blocks = doc.blocks
  local anchored = {}
  -- pass 1: which ids are claimed by anchors
  for i, b in ipairs(blocks) do
    local id = anchor_id(b)
    if id and blocks[i + 1] and blocks[i + 1].t == 'Header' then anchored[id] = true end
  end
  -- pass 2: rebuild
  local out = {}
  local promoted = {}
  local skip_contents = false
  local i = 1
  while i <= #blocks do
    local b = blocks[i]
    local nxt = blocks[i + 1]
    local id = anchor_id(b)
    if id and nxt and nxt.t == 'Header' then
      nxt.identifier = id
      promoted[nxt] = true
      i = i + 1                      -- drop the anchor; the header is handled next round
    elseif b.t == 'Header' and b.level == 2 and b.identifier == 'contents' then
      skip_contents = true           -- drop the heading and its list
      i = i + 1
    elseif skip_contents and b.t == 'BulletList' then
      skip_contents = false
      i = i + 1
    else
      if b.t == 'Header' and anchored[b.identifier] and not promoted[b] then
        -- an unanchored heading that took the id an anchor needs
        local parent = ''
        for j = i - 1, 1, -1 do
          if blocks[j].t == 'Header' and blocks[j].level < b.level then
            parent = blocks[j].identifier .. '-'; break
          end
        end
        b.identifier = parent .. b.identifier
      end
      table.insert(out, b)
      i = i + 1
    end
  end
  doc.blocks = out
  return doc
end

local function is_facts(tbl)
  local head = tbl.head and tbl.head.rows or {}
  if #head ~= 1 then return false end
  local cell = head[1].cells[1]
  return cell and pandoc.utils.stringify(cell.contents) == 'Arguments'
end

local CHIP_COLUMNS = { Required = true, Template = true }

function Table(tbl)
  if is_facts(tbl) then tbl.classes:insert('facts') end
  -- Pipe tables get relative column widths from their dash rows; drop them so
  -- the browser sizes columns by content (description columns get the room).
  for i, spec in ipairs(tbl.colspecs) do
    tbl.colspecs[i] = { spec[1], pandoc.ColWidthDefault }
  end
  -- yes/no cells under Required / Template become chips.
  local chip_cols = {}
  local head = tbl.head and tbl.head.rows[1]
  if head then
    for i, cell in ipairs(head.cells) do
      if CHIP_COLUMNS[pandoc.utils.stringify(cell.contents)] then chip_cols[i] = true end
    end
  end
  if next(chip_cols) then
    for _, body in ipairs(tbl.bodies) do
      for _, row in ipairs(body.body) do
        for i, cell in ipairs(row.cells) do
          if chip_cols[i] then
            local v = pandoc.utils.stringify(cell.contents)
            if v == 'yes' or v == 'no' then
              cell.contents = { pandoc.Plain({ pandoc.Span({ pandoc.Str(v) }, pandoc.Attr('', { 'chip', 'chip-' .. v })) }) }
            end
          end
        end
      end
    end
  end
  return pandoc.Div({ tbl }, pandoc.Attr('', { 'table-wrap' }))
end

function Para(p)
  local first = p.content[1]
  if first and first.t == 'Str' and first.text == 'Tags:' then
    local inner = pandoc.write(pandoc.Pandoc({ pandoc.Plain(p.content) }), 'html')
    return pandoc.RawBlock('html', '<p class="tags">' .. inner .. '</p>')
  end
end
