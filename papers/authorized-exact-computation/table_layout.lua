-- Fixed semantic column widths for the paper's exact comparison tables.
function Table(t)
  local widths = {
    ["Study"] = {0.12, 0.44, 0.44},
    ["NOR-59 endpoint"] = {0.45, 0.20, 0.35},
    ["Arm"] = {0.08, 0.48, 0.44},
    ["Frozen gate"] = {0.50, 0.30, 0.20},
    ["Full signed verifier arm"] = {0.53, 0.27, 0.20},
    ["Label"] = {0.20, 0.80},
  }
  if #t.head.rows == 0 then return t end
  local key = pandoc.utils.stringify(t.head.rows[1].cells[1].contents)
  local row = widths[key]
  if row then
    for i, width in ipairs(row) do
      t.colspecs[i] = {t.colspecs[i][1], width}
    end
  end
  return t
end

-- Flush the NOR-60 figures before the gym section begins in the PDF.
function Header(h)
  if FORMAT:match("latex") and h.level == 2
      and pandoc.utils.stringify(h.content):match("^7%. Authorized Exact Gym") then
    return {pandoc.RawBlock("latex", "\\clearpage"), h}
  end
  return h
end
