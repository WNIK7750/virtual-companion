import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

const css = readFileSync(new URL('../src/style.css', import.meta.url), 'utf8');
const html = readFileSync(new URL('../src/index.html', import.meta.url), 'utf8');

function declarations(selector, source = css) {
  const matches = [...source.matchAll(/([^{}]+)\{([^{}]*)\}/g)]
    .filter((match) => match[1].split(',').map((part) => part.trim()).includes(selector));
  assert.ok(matches.length, `missing CSS rule for ${selector}`);
  return matches.reduce((result, match) => {
    for (const declaration of match[2].split(';').map((part) => part.trim()).filter(Boolean)) {
      const separator = declaration.indexOf(':');
      result[declaration.slice(0, separator).trim()] = declaration.slice(separator + 1).trim();
    }
    return result;
  }, {});
}

function resolveColor(value) {
  const variable = value.match(/^var\((--[\w-]+)\)$/);
  if (variable) {
    const token = css.match(new RegExp(`${variable[1]}\\s*:\\s*(#[0-9a-f]{6})`, 'i'));
    assert.ok(token, `unresolved color token ${variable[1]}`);
    return token[1];
  }
  assert.match(value, /^#[0-9a-f]{6}$/i, `expected a six-digit color, got ${value}`);
  return value;
}

function luminance(hex) {
  const [r, g, b] = hex.match(/[0-9a-f]{2}/gi).map((channel) => parseInt(channel, 16) / 255)
    .map((channel) => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrast(foreground, background) {
  const [a, b] = [luminance(foreground), luminance(background)].sort((x, y) => y - x);
  return (a + 0.05) / (b + 0.05);
}

function mediaBlock(maxWidth) {
  const start = css.search(new RegExp(`@media\\s*\\(max-width:\\s*${maxWidth}px\\)\\s*\\{`));
  assert.notEqual(start, -1, `missing ${maxWidth}px responsive rules`);
  const open = css.indexOf('{', start);
  let depth = 0;
  for (let index = open; index < css.length; index += 1) {
    if (css[index] === '{') depth += 1;
    if (css[index] === '}' && --depth === 0) return css.slice(open + 1, index);
  }
  assert.fail(`unterminated ${maxWidth}px responsive rules`);
}

test('secondary copy and placeholders meet AA contrast on the app surfaces', () => {
  for (const tokenName of ['--muted', '--subtle']) {
    const token = css.match(new RegExp(`${tokenName}\\s*:\\s*(#[0-9a-f]{6})`, 'i'));
    assert.ok(token, `missing ${tokenName}`);
    for (const background of ['#ffffff', '#f5f6f8']) {
      assert.ok(contrast(token[1], background) >= 4.5, `${tokenName} is too faint on ${background}`);
    }
  }

  for (const selector of ['textarea::placeholder', '.keyboard-hint', '.footnote']) {
    const color = resolveColor(declarations(selector).color);
    assert.ok(contrast(color, '#ffffff') >= 4.5, `${selector} is too faint on white`);
  }
});

test('keyboard focus stays visible on buttons and the composer', () => {
  const focus = css.match(/--focus:\s*(#[0-9a-f]{6})/i);
  assert.ok(focus, 'missing --focus token');
  assert.ok(contrast(focus[1], '#ffffff') >= 3, 'focus color lacks 3:1 contrast on white');

  const composer = declarations('#composer:focus-within');
  assert.equal(composer.outline, '2px solid var(--focus)');
  assert.equal(composer['outline-offset'], '3px');
  assert.match(declarations('button:focus-visible').outline, /2px solid var\(--focus\)/);
});

test('feedback dialog has a programmatic name', () => {
  const dialog = html.match(/<dialog\b[^>]*\bid="feedback-dialog"[^>]*>/)?.[0];
  assert.ok(dialog, 'feedback dialog is missing');
  assert.match(dialog, /aria-labelledby="feedback-title"/);
  assert.match(html, /<h2\b[^>]*\bid="feedback-title"[^>]*>/);
});

test('the minimum-width header can shrink its title and status without clipping controls', () => {
  const narrow = mediaBlock(560);
  const title = declarations('h1', narrow);
  assert.equal(title.flex, '1');
  assert.equal(title['min-width'], '0');
  assert.equal(title['max-width'], 'none');
  assert.equal(declarations('.runtime', narrow)['min-width'], '0');
  assert.equal(declarations('#status', narrow)['min-width'], '0');
});

test('small icon actions provide a 44px target and the privacy note separates model processing from local history', () => {
  for (const selector of ['.icon-button', '.composer-tools button', '#send']) {
    const rule = declarations(selector);
    assert.equal(rule.width, '44px', `${selector} width`);
    assert.equal(rule.height, '44px', `${selector} height`);
  }

  const note = html.match(/<p class="footnote">([^<]+)<\/p>/)?.[1] ?? '';
  assert.match(note, /当前所选模型/);
  assert.match(note, /对话记录/);
  assert.match(note, /本机/);
  assert.doesNotMatch(note, /仅保存在本机|不会外发|不会上传/);
});
