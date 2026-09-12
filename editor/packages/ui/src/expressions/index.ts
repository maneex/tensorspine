/**
 * Expressions and conditions as text and as a tree — plan §4.13, artboard S7.
 *
 * The documents store tagged unions and the editor offers two generated views of them, on both
 * sides of the language. `language.ts` derives the text form's grammar from the schemas and
 * `presentation.json` once; `print.ts` writes it, `parse.ts` reads it, `tree.ts` is the
 * structural editor over the same productions, and `Expression.tsx` draws both views. There is
 * one table and one reading of it, which is what makes text → JSON → text the identity over every
 * expression of the corpus and of the reference base.
 */
export { lex, quoted, runsTogether, splitSign, type Lexed, type Token } from './lex.js';
export {
  grammarAt,
  editsExpression,
  unionAnchorOf,
  APPLICATION,
  CONDITION,
  EXPRESSION,
  FUNCTION,
  GENERIC,
  INFIX,
  KEYWORD,
  LOOSEST,
  PAYLOAD,
  PREFIX,
  type Grammar,
  type LanguageContext,
  type Operand,
  type OperandList,
  type Payload,
  type Production,
} from './language.js';
export { ExpressionEditor, type ExpressionEditorProps } from './Expression.js';
export { callName, parseAt, type ExpressionProblem, type Parsed } from './parse.js';
export { documentResolver, type Resolver } from './resolve.js';
export { printAt, printValue, UNPRINTABLE, type PrintContext } from './print.js';
export {
  blankAt,
  blankFor,
  nodeAt,
  operandAnchor,
  operandFor,
  productionLabel,
  replaceAt,
  replacementsAt,
  rowsOf,
  UNNAMED,
  unwrapsTo,
  unwrapTo,
  withMoreOperands,
  withOperator,
  withoutOperand,
  withPayload,
  wrapIn,
  wrappersAt,
  type TreePath,
  type TreeRow,
} from './tree.js';
