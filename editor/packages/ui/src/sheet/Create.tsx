/**
 * The `create` half of a binding, drawn — plan Appendix B, feature 2.21.
 *
 * `presentation.json` has carried `create` since feature 2.2 ("the label of the action offered
 * beside that picker when what is wanted does not exist yet") and nothing drew it: `New family…`
 * and `New axis…` were bound and invisible. This is the drawing, once, for every picker that has
 * one — so the primitive's `New primitive…` and the two that were already bound appear together.
 *
 * **What it does belongs to the place, not here.** A family is declared by being written, so its
 * action writes the word beside the chips; a primitive is declared by a unit in a base of the
 * model (Q6), so its action opens the confirm; an axis is the primitive editor's (3.4), so its
 * action says so in the Log rather than being hidden — §4.4's own rule for a command nobody has
 * wired yet, and the reason a plan the reader can see is better than a menu that lies.
 */
import type { JSX } from 'react';

/** The action beside a picker, labelled by the binding's own `create`. */
export function CreateAction({
  label,
  picker,
  onCreate,
}: {
  /** The binding's `create`, printed as written — it is the file's text, not a dictionary key. */
  readonly label: string | undefined;
  /** The picker it stands beside, so a suite and a stylesheet can name it. */
  readonly picker: string | undefined;
  readonly onCreate: () => void;
}): JSX.Element | null {
  if (label === undefined || label === '') return null;
  return (
    <button
      type="button"
      className="link create"
      data-create={picker ?? ''}
      title={label}
      onClick={onCreate}
    >
      {label}
    </button>
  );
}
