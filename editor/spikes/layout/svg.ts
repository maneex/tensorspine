import type { Layout, PlacedNode, RoutedEdge } from '@tensorspine/ui/layout';

import type { Box, Diagram } from './graphs.ts';

/**
 * The layout spike's drawing: one SVG per diagram, so that a human can judge the readability the
 * feature asks about — against `--view`'s, which is Graphviz's `dot -Tsvg` of the same document.
 *
 * The ink is the design pass's, from `plans/graph-editor-design/_ts.css` and the token table of
 * the component inventory §1: the dark ground `--bg`, `--bg-raised` cards, `--struct` for
 * compositions, `--derived` for a range or a carry, `--edge` for a value edge. The type is the
 * same two families. Nothing here is a component of the application: it is a picture of a layout.
 */

/** The tokens the drawing uses, dark theme, from the component inventory's §1 table. */
const INK = {
  bg: '#101317',
  bgPanel: '#14181d',
  bgRaised: '#1b2026',
  bgTint: '#222932',
  ink: '#e3e8ef',
  ink3: '#9aa3b0',
  muted: '#78828f',
  faint: '#58616d',
  rule2: '#2b323b',
  chip: '#333b45',
  accent: '#2ac3c1',
  struct: '#4d9fd4',
  derived: '#d9a441',
  edge: '#4c5765',
} as const;

const MONO = "'IBM Plex Mono', ui-monospace, monospace";
const SANS = "'IBM Plex Sans', system-ui, sans-serif";

function escape(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

const round = (value: number): string => (Math.round(value * 10) / 10).toString();

/**
 * What fits on one line of a box, cut with an ellipsis — the browser's `text-overflow` by hand.
 * A box is sized from its content, but a compound node is sized from what it holds, so its
 * header can still be the longer of the two.
 */
function fit(text: string, room: number, perCharacter: number): string {
  const characters = Math.floor(room / perCharacter);
  if (characters >= text.length) return text;
  return characters <= 1 ? '…' : `${text.slice(0, characters - 1)}…`;
}

interface Palette {
  fill: string;
  stroke: string;
  title: string;
  dashed: boolean;
}

function palette(kind: Box['kind']): Palette {
  switch (kind) {
    case 'composition':
      return { fill: '#16222c', stroke: INK.struct, title: INK.struct, dashed: false };
    case 'input':
    case 'output':
      return { fill: 'none', stroke: INK.chip, title: INK.ink3, dashed: true };
    case 'expanded':
      return { fill: INK.bgRaised, stroke: INK.rule2, title: INK.ink, dashed: false };
    case 'site':
      return { fill: INK.bgRaised, stroke: INK.rule2, title: INK.ink, dashed: false };
    case 'instance':
      return { fill: INK.bgRaised, stroke: INK.chip, title: INK.ink, dashed: false };
  }
}

function drawNode(node: PlacedNode, box: Box, compound: boolean): string {
  const skin = palette(box.kind);
  const parts: string[] = [];
  parts.push(
    `<rect x="${round(node.x)}" y="${round(node.y)}" width="${round(node.width)}" height="${round(node.height)}" rx="3"` +
      ` fill="${skin.fill}" stroke="${skin.stroke}" stroke-width="1"${skin.dashed ? ' stroke-dasharray="4 3"' : ''}/>`,
  );
  const left = node.x + (box.kind === 'composition' && compound ? 12 : 11);
  const room = node.width - (left - node.x) * 2;
  let baseline = node.y + (box.kind === 'expanded' ? 18 : 23);
  const size = box.kind === 'expanded' ? 11 : box.kind === 'composition' ? 14 : 13.5;
  // The steps are the row heights `graphs.ts` reserved, so a box is the size of what it says.
  const step = box.kind === 'composition' ? { subtitle: 21, note: 24 } : { subtitle: 20, note: 19 };
  parts.push(
    `<text x="${round(left)}" y="${round(baseline)}" font-family="${MONO}" font-size="${size}" fill="${skin.title}">${escape(fit(box.title, room, size * 0.6))}</text>`,
  );
  if (box.kind === 'expanded') {
    parts.push(
      `<text x="${round(node.x + node.width - 10)}" y="${round(baseline)}" text-anchor="end" font-family="${SANS}" font-size="9.5" fill="${INK.muted}">${escape(box.subtitle)}</text>`,
    );
    return parts.join('\n');
  }
  baseline += step.subtitle;
  parts.push(
    `<text x="${round(left)}" y="${round(baseline)}" font-family="${SANS}" font-size="10.5" fill="${box.kind === 'composition' ? INK.derived : INK.muted}">${escape(fit(box.subtitle, room, 10.5 * 0.55))}</text>`,
  );
  for (const line of box.notes) {
    baseline += step.note;
    const colour = line.startsWith('⚑') || line.startsWith('↺') ? INK.derived : INK.ink3;
    parts.push(
      `<text x="${round(left)}" y="${round(baseline)}" font-family="${MONO}" font-size="9.5" fill="${colour}">${escape(fit(line, room, 9.5 * 0.6))}</text>`,
    );
  }
  return parts.join('\n');
}

function drawEdge(edge: RoutedEdge, withLabels: boolean): string {
  if (edge.points.length < 2) return '';
  const path = edge.points
    .map((point, index) => `${index === 0 ? 'M' : 'L'}${round(point.x)} ${round(point.y)}`)
    .join(' ');
  const parts = [
    `<path d="${path}" fill="none" stroke="${INK.edge}" stroke-width="1.2" marker-end="url(#arrow)"/>`,
  ];
  if (withLabels && edge.label !== undefined) {
    parts.push(
      `<text x="${round(edge.label.x)}" y="${round(edge.label.y + 9)}" font-family="${MONO}" font-size="9.5" fill="${INK.faint}">${escape(edge.label.text)}</text>`,
    );
  }
  return parts.join('\n');
}

/** One diagram, drawn. `layout` must be the layout of `diagram.graph`. */
export function toSvg(diagram: Diagram, layout: Layout): string {
  const margin = 24;
  const headerHeight = 54;
  // Wide enough for the caption as well as the drawing: a header the viewBox cuts off is a lie.
  const width = Math.max(layout.width + margin * 2, diagram.caption.length * 7.2 + margin * 2);
  const height = layout.height + margin * 2 + headerHeight;
  const compound = new Set(layout.nodes.filter((node) => node.depth > 0).map((node) => node.parent));
  const withLabels = diagram.reading !== 'expanded';
  const draw = (node: PlacedNode): string[] => {
    const box = diagram.boxes.get(node.id);
    return box === undefined ? [] : [drawNode(node, box, compound.has(node.id))];
  };
  // A compound box is a ground, so it is painted first — an edge that crosses it and the cards
  // inside it are drawn over it; a shallower box is painted before a deeper one.
  const grounds = layout.nodes
    .filter((node) => compound.has(node.id))
    .sort((a, b) => a.depth - b.depth);
  const body = [
    ...grounds.flatMap(draw),
    ...layout.edges.map((edge) => drawEdge(edge, withLabels)),
    ...layout.nodes.filter((node) => !compound.has(node.id)).flatMap(draw),
  ].join('\n');
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="${round(width)}" height="${round(height)}" viewBox="0 0 ${round(width)} ${round(height)}" font-family="${SANS}">
<defs>
  <marker id="arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0 0 L8 4 L0 8 z" fill="${INK.edge}"/>
  </marker>
</defs>
<rect width="100%" height="100%" fill="${INK.bg}"/>
<rect x="0" y="0" width="100%" height="${headerHeight}" fill="${INK.bgPanel}"/>
<text x="${margin}" y="24" font-size="13" fill="${INK.ink}">${escape(diagram.caption)}</text>
<text x="${margin}" y="41" font-size="10.5" fill="${INK.muted}">ELK layered, top to bottom · ${layout.nodes.length} boxes · ${layout.edges.length} edges · ${round(layout.width)} × ${round(layout.height)} px · ${Math.round(layout.milliseconds)} ms</text>
<g transform="translate(${margin}, ${margin + headerHeight})">
${body}
</g>
</svg>
`;
}
