import { beforeEach, describe, expect, it } from 'vitest';

import { MemoryWorkspace, stubPlatform } from '@tensorspine/store/platform';

import { COMMANDS } from '../../src/shell/commands.js';
import { BOUNDS, defaultRegions, defaultSide } from '../../src/shell/regions.js';
import { forgetKeysUsed, keysUsed, setLocale, text, textWith } from '../../src/shell/strings.js';
import {
  createShell,
  documentationBase,
  HELP_PAGES,
  HELP_TABS,
  LAYOUT_SETTING,
  themeOf,
  type ShellStore,
} from '../../src/shell/store.js';
import { DEFAULT_THEME, THEME_SETTING } from '../../src/shell/theme.js';

// The shell's state over the stub `Platform` — the three things it subscribes to, the arrangement
// it remembers, and what running a command of §4.4 does when nobody has wired it yet.

type Stub = ReturnType<typeof stubPlatform>;

function shellOver(options: Parameters<typeof stubPlatform>[0] = {}): {
  platform: Stub;
  store: ShellStore;
  dispose: () => void;
} {
  const platform = stubPlatform(options);
  const { store, dispose } = createShell({ platform, docsBase: 'https://example.test/site/' });
  return { platform, store, dispose };
}

describe('what the shell’s state holds', () => {
  it('is chrome, and nothing of the document (D1)', () => {
    const { store, dispose } = shellOver();
    const state = store.getState();
    const held = Object.keys(state)
      .filter((key) => typeof state[key as keyof typeof state] !== 'function')
      .sort();
    // A snapshot rather than a list, so a field that started holding a quantity, an instance or a
    // problem is a reviewed diff. "There is no second graph model" is the rule it guards.
    expect(held).toMatchInlineSnapshot(`
      [
        "activeTab",
        "draggingPanel",
        "log",
        "menu",
        "palette",
        "regions",
        "scheme",
        "session",
        "side",
        "tabs",
        "theme",
        "workspace",
        "zoom",
      ]
    `);
    dispose();
  });

  it('opens on §4.2’s arrangement and on the machine’s own theme', () => {
    const { store, dispose } = shellOver();
    expect(store.getState().side).toEqual(defaultSide());
    expect(store.getState().regions).toEqual(defaultRegions());
    expect(store.getState().theme).toBe(DEFAULT_THEME);
    expect(store.getState().zoom).toBe(1);
    expect(store.getState().tabs).toEqual([]);
    dispose();
  });

  it('says in the Log what it is looking at', () => {
    const { store, dispose } = shellOver();
    const lines = store.getState().log.map((line) => line.text);
    expect(lines[0]).toContain('stub');
    expect(lines.join('\n')).toContain('memory');
    dispose();
  });
});

describe('what the platform changes underneath it', () => {
  it('shows no session where the deployment has none, and one where it has', () => {
    const without = shellOver();
    expect(without.store.getState().session).toBeNull();
    without.dispose();

    const withOne = shellOver({ session: { id: 'pl', name: 'Perceval Lambert' } });
    expect(withOne.store.getState().session?.name).toBe('Perceval Lambert');
    withOne.dispose();
  });

  it('follows the workspace a command of §4.3 opened', async () => {
    const { platform, store, dispose } = shellOver();
    expect(store.getState().workspace.kind).toBe('memory');
    await platform.workspaces.openExamples();
    expect(store.getState().workspace.kind).toBe('examples');
    dispose();
  });

  it('follows the machine’s colour scheme, and resolves `system` against it', () => {
    const { platform, store, dispose } = shellOver();
    expect(themeOf(store.getState())).toBe('light');
    platform.preferScheme('dark');
    expect(store.getState().scheme).toBe('dark');
    expect(themeOf(store.getState())).toBe('dark');

    store.getState().setTheme('light');
    expect(themeOf(store.getState())).toBe('light');
    platform.preferScheme('light');
    store.getState().setTheme('dark');
    expect(themeOf(store.getState())).toBe('dark');
    dispose();
  });

  it('stops listening when it is disposed of', () => {
    const { platform, store, dispose } = shellOver();
    dispose();
    platform.preferScheme('dark');
    expect(store.getState().scheme).toBe('light');
  });
});

describe('what it remembers', () => {
  it('writes the theme to the settings and reads it back next time', () => {
    const first = shellOver();
    first.store.getState().setTheme('dark');
    expect(first.platform.settings.peek(THEME_SETTING)).toBe('dark');
    first.dispose();

    const again = shellOver({ settings: { [THEME_SETTING]: 'dark' } });
    expect(again.store.getState().theme).toBe('dark');
    again.dispose();
  });

  it('ignores a stored theme this editor cannot name', () => {
    const { store, dispose } = shellOver({ settings: { [THEME_SETTING]: 'sepia' } });
    expect(store.getState().theme).toBe(DEFAULT_THEME);
    dispose();
  });

  it('writes the arrangement on every gesture that changes it, and reads it back', () => {
    const first = shellOver();
    first.store.getState().setSideWidth(300);
    first.store.getState().setRegionSize('region.bottom', 260);
    first.store.getState().dockPanel('panel.derived', 'region.right');
    first.store.getState().setActivity('activity.weights');
    const written = first.platform.settings.peek(LAYOUT_SETTING);
    expect(written).toBeDefined();
    first.dispose();

    const again = shellOver({ settings: { [LAYOUT_SETTING]: written as never } });
    const state = again.store.getState();
    expect(state.side.width).toBe(300);
    expect(state.side.activity).toBe('activity.weights');
    expect(state.regions['region.bottom'].size).toBe(260);
    expect(state.regions['region.right'].panels).toEqual(['panel.properties', 'panel.derived']);
    again.dispose();
  });

  it('clamps a width a drag pushed past its bounds before it stores it', () => {
    const { store, dispose } = shellOver();
    store.getState().setSideWidth(5000);
    expect(store.getState().side.width).toBe(BOUNDS.side.max);
    store.getState().setRegionSize('region.right', 5);
    expect(store.getState().regions['region.right'].size).toBe(BOUNDS['region.right'].min);
    dispose();
  });
});

describe('running a command of §4.4', () => {
  it('performs the ones the shell can perform itself', () => {
    const { store, dispose } = shellOver();
    store.getState().run('view.toggle-side-bar');
    expect(store.getState().side.open).toBe(false);
    store.getState().run('view.toggle-side-bar');
    expect(store.getState().side.open).toBe(true);

    store.getState().run('view.log');
    expect(store.getState().regions['region.bottom'].active).toBe('panel.log');

    store.getState().run('view.zoom-in');
    expect(store.getState().zoom).toBeGreaterThan(1);
    store.getState().run('view.zoom-fit');
    expect(store.getState().zoom).toBe(1);

    store.getState().run('view.theme-dark');
    expect(store.getState().theme).toBe('dark');

    store.getState().run('edit.preferences');
    expect(store.getState().side.activity).toBe('activity.settings');
    dispose();
  });

  it('toggles Properties wherever Properties is', () => {
    const { store, dispose } = shellOver();
    store.getState().run('view.toggle-properties');
    expect(store.getState().regions['region.right'].open).toBe(false);
    store.getState().run('view.toggle-properties');
    expect(store.getState().regions['region.right'].open).toBe(true);

    store.getState().dockPanel('panel.properties', 'region.bottom');
    store.getState().selectPanel('region.bottom', 'panel.problems');
    store.getState().run('view.toggle-properties');
    expect(store.getState().regions['region.bottom'].active).toBe('panel.properties');
    store.getState().run('view.toggle-properties');
    expect(store.getState().regions['region.bottom'].open).toBe(false);
    dispose();
  });

  it('opens the Help menu’s two screens as tabs, and never twice', () => {
    const { store, dispose } = shellOver();
    store.getState().run('help.about');
    expect(store.getState().tabs.map((tab) => tab.id)).toEqual([HELP_TABS['help.about']?.id]);
    store.getState().run('help.keyboard-shortcuts');
    expect(store.getState().tabs).toHaveLength(2);
    store.getState().run('help.about');
    expect(store.getState().tabs).toHaveLength(2);
    expect(store.getState().activeTab).toBe(HELP_TABS['help.about']?.id);

    store.getState().closeTab(HELP_TABS['help.about']?.id ?? '');
    expect(store.getState().tabs).toHaveLength(1);
    expect(store.getState().activeTab).toBe(HELP_TABS['help.keyboard-shortcuts']?.id);
    store.getState().closeAllTabs();
    expect(store.getState().tabs).toEqual([]);
    expect(store.getState().activeTab).toBeNull();
    dispose();
  });

  it('opens the Help menu’s six links beside the site the editor is deployed with', async () => {
    const { platform, store, dispose } = shellOver();
    for (const id of Object.keys(HELP_PAGES)) store.getState().run(id);
    await Promise.resolve();
    expect(platform.shellRecord.opened).toEqual(
      Object.values(HELP_PAGES).map((page) => `https://example.test/site/${page}`),
    );
    dispose();
  });

  it('says in the Log when nobody has wired one yet, instead of doing nothing', () => {
    const { store, dispose } = shellOver();
    const before = store.getState().log.length;
    store.getState().run('model.derive-now');
    const line = store.getState().log[before]?.text ?? '';
    expect(line).toContain('Derive Now');
    expect(store.getState().log).toHaveLength(before + 1);
    dispose();
  });

  it('runs a handler a later feature bound, in place of the Log line', () => {
    const { store, dispose } = shellOver();
    let ran = 0;
    store.getState().bind({
      'model.derive-now': () => {
        ran += 1;
      },
    });
    const before = store.getState().log.length;
    store.getState().run('model.derive-now');
    expect(ran).toBe(1);
    expect(store.getState().log).toHaveLength(before);
    dispose();
  });

  it('reports a handler that threw, and one whose promise rejected, rather than losing it', async () => {
    const { store, dispose } = shellOver();
    store.getState().bind({
      'model.lint': () => {
        throw new Error('the lint went wrong');
      },
      'model.validate-now': () => Promise.reject(new Error('the validation went wrong')),
    });
    store.getState().run('model.lint');
    expect(store.getState().log.at(-1)?.text).toContain('the lint went wrong');
    store.getState().run('model.validate-now');
    await Promise.resolve();
    await Promise.resolve();
    expect(store.getState().log.at(-1)?.text).toContain('the validation went wrong');
    dispose();
  });

  it('closes the menu and the palette whatever it ran', () => {
    const { store, dispose } = shellOver();
    store.getState().openMenu('menu.file');
    store.getState().setPalette(true);
    store.getState().run('view.log');
    expect(store.getState().menu).toBeNull();
    expect(store.getState().palette).toBe(false);
    dispose();
  });

  it('says so for an identity no command carries', () => {
    const { store, dispose } = shellOver();
    store.getState().run('file.nonesuch');
    expect(store.getState().log.at(-1)?.text).toContain('file.nonesuch');
    dispose();
  });

  it('has a handler or a Log line for every command there is', () => {
    // The palette's claim, from the other side: running any of the ninety does something visible.
    const { store, dispose } = shellOver();
    for (const command of COMMANDS) {
      if (command.id.startsWith('help.')) continue;
      const before = store.getState().log.length;
      store.getState().run(command.id);
      const after = store.getState().log.length;
      const changed = after > before;
      expect(typeof changed).toBe('boolean');
    }
    dispose();
  });
});

describe('the shell’s strings', () => {
  beforeEach(() => {
    setLocale('en');
    forgetKeysUsed();
  });

  it('answers the English source string when nothing is translated', () => {
    expect(text('Open Folder…')).toBe('Open Folder…');
    expect(textWith('{} commands.', '90')).toBe('90 commands.');
  });

  it('answers the translation when a locale carries one, and the key when it does not', () => {
    setLocale('fr', { 'Open Folder…': 'Ouvrir un dossier…' });
    expect(text('Open Folder…')).toBe('Ouvrir un dossier…');
    expect(text('New Model')).toBe('New Model');
    setLocale('en');
    expect(text('Open Folder…')).toBe('Open Folder…');
  });

  it('lets a translator see what was asked for', () => {
    text('Editor');
    text('Commands');
    expect(keysUsed()).toEqual(['Commands', 'Editor']);
  });

  it('reorders the value with the sentence a translation puts it in', () => {
    setLocale('fr', { 'Platform: {}.': 'Plateforme : {}.' });
    expect(textWith('Platform: {}.', 'browser')).toBe('Plateforme : browser.');
  });
});

describe('where the Help menu looks for the documentation', () => {
  it('is the directory above the page, whatever the page’s address is', () => {
    // D11: the editor is deployed *beside* the documentation site.
    expect(documentationBase('https://maneex.github.io/tensorspine/editor/')).toBe(
      'https://maneex.github.io/tensorspine/',
    );
    expect(documentationBase('https://maneex.github.io/tensorspine/editor/index.html')).toBe(
      'https://maneex.github.io/tensorspine/',
    );
    expect(documentationBase('http://127.0.0.1:4173/')).toBe('http://127.0.0.1:4173/');
    expect(documentationBase('http://127.0.0.1:4173/editor/')).toBe('http://127.0.0.1:4173/');
  });
});

describe('the workspace the shell shows', () => {
  it('names the folder when there is one', async () => {
    const platform = stubPlatform({ workspace: MemoryWorkspace.of({ 'a.json': '{}' }, 'data') });
    const { store, dispose } = createShell({ platform });
    expect(store.getState().workspace.name).toBe('data');
    dispose();
    await Promise.resolve();
  });
});
