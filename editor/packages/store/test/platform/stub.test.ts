import { describe, expect, it } from 'vitest';

import {
  MemoryWorkspace,
  PlatformError,
  draftKey,
  memoryDrafts,
  memorySettings,
  stubPlatform,
  summaryOf,
  type Draft,
  type Platform,
} from '../../src/platform/index.js';

// The stub `Platform` of D11 — what CI builds the application against, and what every suite in
// this workspace uses. Every member of §5.2 is here, so a component that reaches for something
// the interface does not carry fails against it before it fails against a browser.

describe('the stub platform', () => {
  it('carries every member of §5.2, and says which platform it is', () => {
    const platform: Platform = stubPlatform();
    expect(Object.keys(platform).sort()).toEqual([
      'auth',
      'checkpoints',
      'describe',
      'drafts',
      'settings',
      'shell',
      'shellRecord',
      'workspace',
      'workspaces',
    ]);
    expect(platform.describe()).toBe('stub');
    expect(platform.checkpoints).toEqual([]);
  });

  it('has no picker and says so instead of inventing a folder nobody chose', async () => {
    const platform = stubPlatform();
    expect(platform.workspaces.writablePicker).toBe(false);
    await expect(platform.workspaces.open()).rejects.toBeInstanceOf(PlatformError);
    expect(await platform.workspaces.recent()).toEqual([]);
    expect(await platform.workspaces.reopen('anything')).toBeNull();
  });

  it('replaces the open workspace and tells whoever asked to be told', async () => {
    const platform = stubPlatform({ examples: { 'models/llama3-8b.json': '{}\n' } });
    const seen: string[] = [];
    const stop = platform.workspaces.onChange((workspace) => seen.push(workspace.root().kind));
    expect(platform.workspace.root().kind).toBe('memory');
    const examples = await platform.workspaces.openExamples();
    // `platform.workspace` is the open one, not the one a component happened to be holding.
    expect(platform.workspace).toBe(examples);
    expect(platform.workspaces.current()).toBe(examples);
    expect(seen).toEqual(['examples']);
    stop();
    await platform.workspaces.openExamples();
    expect(seen).toEqual(['examples']);
  });

  it('sends a read-only Save to the shell, which is where a download comes from', async () => {
    const platform = stubPlatform({ examples: { 'models/llama3-8b.json': '{"a": 1}\n' } });
    const examples = await platform.workspaces.openExamples();
    await examples.write('models/llama3-8b.json', 'edited');
    expect(platform.shellRecord.downloads.map((one) => one.name)).toEqual(['llama3-8b.json']);
  });

  it('has no accounts, and refuses a sign-in in the deployment’s own words', async () => {
    const platform = stubPlatform();
    expect(platform.auth.current()).toBeNull();
    const error = await platform.auth.signIn().catch((one: unknown) => one);
    expect(error).toBeInstanceOf(PlatformError);
    expect((error as PlatformError).reason).toBe('unsupported');
    // Signing out of nothing is done rather than refused.
    await expect(platform.auth.signOut()).resolves.toBeUndefined();
    platform.auth.onChange(() => undefined)();
  });

  it('records what the shell was asked to do, and takes the safe branch of a question', async () => {
    const platform = stubPlatform();
    expect(await platform.shell.confirm('discard the changes?')).toBe(false);
    platform.shellRecord.answer = true;
    expect(await platform.shell.confirm('discard the changes?')).toBe(true);
    await platform.shell.openExternal('https://tensorspine.dev/');
    await platform.shell.writeClipboard('attention.dense');
    platform.shell.setMenu([{ id: 'file.save', label: 'Save', enabled: true }]);
    expect(platform.shellRecord.asked).toHaveLength(2);
    expect(platform.shellRecord.opened).toEqual(['https://tensorspine.dev/']);
    expect(await platform.shell.readClipboard()).toBe('attention.dense');
    expect(platform.shellRecord.menu.map((one) => one.id)).toEqual(['file.save']);
  });
});

describe('settings in memory', () => {
  it('answers the fallback for a key it does not hold, and remembers what it is given', () => {
    const settings = memorySettings({ theme: 'dark' });
    expect(settings.get('theme', 'light')).toBe('dark');
    expect(settings.get('panel.size', 320)).toBe(320);
    expect(settings.peek('panel.size')).toBeUndefined();
    settings.set('panel.size', 280);
    expect(settings.get('panel.size', 320)).toBe(280);
    expect(settings.keys()).toEqual(['panel.size', 'theme']);
    settings.remove('theme');
    expect(settings.keys()).toEqual(['panel.size']);
  });

  it('tells whoever asked which key changed, and says it does not persist', () => {
    const settings = memorySettings();
    const seen: string[] = [];
    const stop = settings.onChange((key) => seen.push(key));
    settings.set('unlocked.bases', ['data/primitive-library']);
    settings.remove('unlocked.bases');
    stop();
    settings.set('theme', 'dark');
    expect(seen).toEqual(['unlocked.bases', 'unlocked.bases']);
    expect(settings.persistent).toBe(false);
  });
});

describe('drafts in memory', () => {
  const draft = (path: string, text: string, savedAt: string): Draft => ({
    workspace: 'memory:memory',
    path,
    text,
    revision: 'm1',
    savedAt,
  });

  it('holds one draft per workspace and path', async () => {
    const drafts = memoryDrafts();
    await drafts.put(draft('models/llama3-8b.json', 'first', '2026-09-11T10:00:00.000Z'));
    await drafts.put(draft('models/llama3-8b.json', 'second', '2026-09-11T10:00:30.000Z'));
    const found = await drafts.get('memory:memory', 'models/llama3-8b.json');
    expect(found?.text).toBe('second');
    expect(await drafts.list()).toHaveLength(1);
  });

  it('lists them newest first, without their texts', async () => {
    const drafts = memoryDrafts();
    await drafts.put(draft('a.json', 'a', '2026-09-11T10:00:00.000Z'));
    await drafts.put(draft('b.json', 'bb', '2026-09-11T10:01:00.000Z'));
    const listed = await drafts.list();
    expect(listed.map((one) => one.path)).toEqual(['b.json', 'a.json']);
    expect(listed[0]).not.toHaveProperty('text');
    expect(listed[0]?.bytes).toBe(2);
  });

  it('forgets one, and forgets a whole workspace', async () => {
    const drafts = memoryDrafts();
    await drafts.put(draft('a.json', 'a', '2026-09-11T10:00:00.000Z'));
    await drafts.put({ ...draft('b.json', 'b', '2026-09-11T10:00:00.000Z'), workspace: 'other' });
    await drafts.remove('memory:memory', 'a.json');
    expect(await drafts.list()).toHaveLength(1);
    await drafts.clear('other');
    expect(await drafts.list()).toEqual([]);
  });

  it('keys a draft by its workspace and its path, injectively', () => {
    expect(draftKey('memory:memory', 'models/llama3-8b.json')).not.toBe(
      draftKey('memory:memory/models', 'llama3-8b.json'),
    );
    expect(draftKey('w', './models/x.json')).toBe(draftKey('w', 'models/x.json'));
    expect(summaryOf(draft('a.json', 'héllo', '2026-09-11T10:00:00.000Z')).bytes).toBe(6);
  });
});

describe('the stub’s workspace', () => {
  it('is a memory workspace, which is a real implementation of the interface', async () => {
    const platform = stubPlatform({ workspace: MemoryWorkspace.of({ 'a.json': '{}\n' }, 'seeded') });
    expect(platform.workspace.root().name).toBe('seeded');
    expect(await platform.workspace.list('')).toEqual([
      { name: 'a.json', path: 'a.json', kind: 'file' },
    ]);
  });
});
