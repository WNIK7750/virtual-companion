import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { ProviderPanel } from '../src/provider-panel.mjs';

function harness() {
  const elements = new Map();
  class Element {
    constructor() { this.children = []; this.dataset = {}; this.textContent = ''; }
    append(...values) { this.children.push(...values); }
    replaceChildren(...values) { this.children = values; }
    close() {} focus() {} showModal() { this.open = true; }
  }
  const document = { createElement: () => new Element(), getElementById(id) {
    if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id);
  } };
  return { panel: new ProviderPanel({ api: {}, document }), element: id => elements.get(id) };
}

function state(status, configured = true) {
  return { default_provider: 'deepseek-api', selected_provider: 'deepseek-api',
    active: { provider_id: 'deepseek-api', label: 'DeepSeek Flash', model_label: 'DeepSeek V4.1 Flash',
      status, capabilities: ['流式文本'] },
    options: [{ provider_id: 'deepseek-api', label: 'DeepSeek Flash', model_label: 'DeepSeek V4.1 Flash',
      status, selectable: configured, requires_confirmation: status !== 'active', message: '合成范围' }] };
}

test('provider panel shows the bound Flash route and no selection prompt', () => {
  const h = harness(), value = state('active');
  h.panel.setState(value); h.panel.render(value);
  assert.equal(h.element('provider-label').textContent, 'DeepSeek Flash · 当前');
  const row = h.element('provider-options').children[0];
  assert.equal(row.children.at(-1).textContent, '正在使用');
  assert.equal(row.children.at(-1).disabled, true);
  assert.equal(row.children.some(node => node.children.some(child => child.type === 'checkbox')), false);
});

test('selected but unconfigured Flash is shown unavailable and cannot look active', () => {
  const h = harness(), value = state('unconfigured', false);
  h.panel.setState(value); h.panel.render(value);
  assert.equal(h.element('provider-label').textContent, 'DeepSeek Flash · 未配置');
  const row = h.element('provider-options').children[0];
  assert.equal(row.children.at(-1).textContent, '当前不可用');
  assert.equal(row.children.at(-1).disabled, true);
});

test('failed binding can retry the same selected route after scope confirmation', () => {
  const h = harness(), value = state('unavailable');
  h.panel.setState(value); h.panel.render(value);
  assert.equal(h.element('provider-label').textContent, 'DeepSeek Flash · 未就绪');
  const row = h.element('provider-options').children[0];
  const consent = row.children.find(node => node.children.some(child => child.type === 'checkbox'))
    .children[0];
  assert.equal(row.children.at(-1).textContent, '使用此路由');
  assert.equal(row.children.at(-1).disabled, true);
  consent.checked = true; consent.onchange();
  assert.equal(row.children.at(-1).disabled, false);
});

test('initial shell identifies Flash as the product default', async () => {
  const source = await readFile(new URL('../src/index.html', import.meta.url), 'utf8');
  assert.match(source, /id="provider-label"[^>]*>DeepSeek Flash 默认</u);
  assert.match(source, /DeepSeek Flash 是当前默认模型/u);
});
