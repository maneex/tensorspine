/**
 * The browser's `Shell` — feature 2.4, §5.2: "dialogs, open-external, clipboard, native menu
 * hooks".
 *
 * The static application has no shell around it, so each of these is the smallest honest thing a
 * page can do. The one that matters is {@link browserShell}'s download: it is how Save works on
 * every workspace the editor cannot write, which is what two of the three engines get (feature
 * 0.6) and what the Examples workspace always is.
 */
import type { AcceleratorModifier, ColourScheme, MenuCommand, Shell, Unsubscribe } from '@tensorspine/store/platform';

/**
 * How long an object URL is kept alive after it is handed over.
 *
 * The browser's own fetch of the URL has to outlive the click; a minute is generous, and the URL
 * is revoked afterwards so that a session of saves does not hold every document it ever wrote.
 */
const URL_LIFETIME_MS = 60_000;

/** What the last download handed over, for the chrome that says what it did. */
export interface LastDownload {
  readonly name: string;
  readonly bytes: number;
  readonly url: string;
}

/** The browser shell, and what it last did. */
export interface BrowserShell extends Shell {
  /** The last file handed to the user — what the log line and the suites read. */
  readonly lastDownload: LastDownload | null;
  /** The commands a native menu was offered; a browser has none, so nothing was done with them. */
  readonly menu: readonly MenuCommand[];
}

/**
 * Which modifier this machine writes an accelerator with — §4.4's "⌘ on macOS, Ctrl elsewhere".
 *
 * `userAgentData.platform` where the engine has it (Chromium), the deprecated `navigator.platform`
 * elsewhere; both answer a string naming the operating system, and both are the only thing a page
 * is told about the machine's keyboard. `iPhone` and `iPad` join `Mac` because the conventions are
 * the same and a page served to one is served the same menus.
 */
export function modifierOf(): AcceleratorModifier {
  return appleConventions() ? 'command' : 'control';
}

/** Whether this machine writes accelerators the way Apple's conventions do. */
export function appleConventions(): boolean {
  const agent = navigator as Navigator & { userAgentData?: { platform?: string } };
  const named = agent.userAgentData?.platform ?? navigator.platform;
  return /mac|iphone|ipad|ipod/i.test(named);
}

/**
 * The machine's colour-scheme preference, where the engine can be asked.
 *
 * `prefers-color-scheme` has two values and answers `light` where no preference is expressed, so
 * that is what a missing `matchMedia` answers too: the same reading, not a different default.
 */
function prefersDark(): MediaQueryList | null {
  return typeof window.matchMedia === 'function' ? window.matchMedia('(prefers-color-scheme: dark)') : null;
}

/** What {@link browserShell} is given, for the runs where a download must not be delivered. */
export interface ShellOptions {
  /**
   * `false` builds the download without handing it to the browser.
   *
   * The one caller is an unattended cross-engine run, which opens browsers nobody is watching and
   * must not litter the machine with files. Every other path delivers, which is the point of Save.
   */
  readonly deliver?: boolean;
}

export function browserShell(options: ShellOptions = {}): BrowserShell {
  const deliver = options.deliver ?? true;
  const shell = {
    lastDownload: null as LastDownload | null,
    menu: [] as readonly MenuCommand[],

    confirm: (question: string): Promise<boolean> => Promise.resolve(window.confirm(question)),

    openExternal: (url: string): Promise<void> => {
      // `noopener` so the opened page cannot reach back into the editor's window.
      window.open(url, '_blank', 'noopener,noreferrer');
      return Promise.resolve();
    },

    readClipboard: async (): Promise<string | null> => {
      try {
        return await navigator.clipboard.readText();
      } catch {
        // Reading the clipboard needs a permission the user may never have been asked for.
        return null;
      }
    },

    writeClipboard: async (text: string): Promise<void> => {
      await navigator.clipboard.writeText(text);
    },

    download: (name: string, text: string): Promise<void> => {
      const blob = new Blob([text], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = name;
      anchor.rel = 'noopener';
      // `hidden` and not `style.display`, whose value would be the string `none` — a value of the
      // model schema's `mask` and of `partition_options`, which catching rule (b) of §1 forbids the
      // interface to write as a literal. The property says the same thing and says it better.
      anchor.hidden = true;
      if (deliver) {
        document.body.append(anchor);
        anchor.click();
        anchor.remove();
      }
      setTimeout(() => {
        URL.revokeObjectURL(url);
      }, URL_LIFETIME_MS);
      shell.lastDownload = { name, bytes: blob.size, url };
      return Promise.resolve();
    },

    /**
     * A browser has no native menu: the commands are taken and nothing is done with them, which
     * is what lets the interface register them unconditionally (§4.4) and the desktop build one.
     */
    setMenu: (commands: readonly MenuCommand[]): void => {
      shell.menu = commands;
    },

    modifier: modifierOf(),

    colourScheme: (): ColourScheme => (prefersDark()?.matches === true ? 'dark' : 'light'),

    onColourSchemeChange: (callback: (scheme: ColourScheme) => void): Unsubscribe => {
      const query = prefersDark();
      if (query === null) return () => undefined;
      const listener = (event: MediaQueryListEvent): void => {
        callback(event.matches ? 'dark' : 'light');
      };
      query.addEventListener('change', listener);
      return () => {
        query.removeEventListener('change', listener);
      };
    },
  };
  return shell;
}
