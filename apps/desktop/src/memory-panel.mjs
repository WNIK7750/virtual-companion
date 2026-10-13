const text = (document, name, value, className) => {
  const node = document.createElement(name);
  if (className) node.className = className;
  node.textContent = value;
  return node;
};

const formatDate = value => {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? '更新时间待确认' :
    new Intl.DateTimeFormat('zh-CN', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
};

const eventTimeText = item => item.event_time
  ? `发生时间：${item.event_time_label || item.event_time}` : '发生时间不详';

export function memoryStatsText(stats) {
  if (!stats || !Number.isInteger(stats.effective_count)
      || !Number.isInteger(stats.current_record_count)
      || !Number.isInteger(stats.history_version_count)) return '记忆概况待确认';
  const unknown = Number.isInteger(stats.unknown_event_time_count)
    ? ` · 发生时间不详 ${stats.unknown_event_time_count} 条` : '';
  return `当前有效 ${stats.effective_count} 条 · 当前记录 ${stats.current_record_count} 条 · 历史版本 ${stats.history_version_count} 条${unknown}`;
}

export function mergeCurrentMemoryPage(items, current) {
  if (!current || current.status !== 'active') return items;
  return [current, ...items.filter(item => item.record_id !== current.record_id)];
}

export const attentionAllowsSelection = item => item?.resolution_mode === 'record_selection';

export class MemoryPanel {
  constructor({ api, document, randomUUID }) {
    this.api = api; this.document = document; this.randomUUID = randomUUID;
    this.openButton = document.getElementById('memory-open');
    this.dialog = document.getElementById('memory-dialog');
    this.closeButton = document.getElementById('memory-close');
    this.feedback = document.getElementById('memory-feedback');
    this.stats = document.getElementById('memory-stats');
    this.attention = document.getElementById('memory-attention');
    this.state = document.getElementById('memory-state');
    this.list = document.getElementById('memory-list');
    this.more = document.getElementById('memory-more');
    this.items = []; this.nextOffset = null; this.loading = false;
    this.operations = new Map(); this.resolvesJob = null;
    this.openButton.onclick = () => this.open();
    this.closeButton.onclick = () => this.dialog.close();
    this.more.onclick = () => this.load(false);
  }

  setAvailable(available) {
    this.openButton.hidden = !available;
    this.openButton.disabled = !available;
    if (!available && this.dialog.open) this.dialog.close();
  }

  async open() {
    if (this.openButton.disabled) return;
    if (!this.dialog.open) this.dialog.showModal();
    this.closeButton.focus();
    await this.load(true);
  }

  showState(message, kind = '') {
    this.state.textContent = message;
    this.state.dataset.kind = kind;
    this.state.hidden = !message;
  }

  showFeedback(message, kind = 'success') {
    this.feedback.textContent = message;
    this.feedback.dataset.kind = kind;
    this.feedback.hidden = !message;
  }

  async load(reset) {
    if (this.loading || (!reset && this.nextOffset == null)) return;
    this.loading = true; this.more.disabled = true;
    if (reset) { this.items = []; this.nextOffset = 0; this.list.replaceChildren(); }
    this.showState('正在读取本机记忆…', 'loading');
    try {
      const page = await this.api.memories({ offset: this.nextOffset || 0, limit: 50 });
      const merged = [...this.items, ...page.items];
      this.items = [...new Map(merged.map(item => [item.record_id, item])).values()];
      this.nextOffset = page.next_offset;
      this.renderStats(page.stats);
      this.renderAttention(page.attention);
      this.renderItems();
      this.showState(this.items.length ? '' : '还没有可管理的有效记忆。', 'empty');
      this.more.hidden = this.nextOffset == null;
    } catch {
      this.showState('记忆暂时无法读取。请重试；已有记忆没有被修改。', 'error');
      this.more.hidden = true;
    } finally {
      this.loading = false; this.more.disabled = false;
    }
  }

  renderStats(stats) {
    this.stats.textContent = memoryStatsText(stats);
    this.stats.hidden = !stats;
  }

  renderAttention(items) {
    this.attention.replaceChildren(...items.map(item => {
      const article = this.document.createElement('article');
      article.className = 'memory-attention-item';
      article.append(text(this.document, 'strong', '需要确认'),
        text(this.document, 'p', item.source), text(this.document, 'small', item.message));
      const actions = this.document.createElement('div'); actions.className = 'memory-actions';
      if (attentionAllowsSelection(item)) {
        const select = text(this.document, 'button', '选择一条记录继续'); select.type = 'button';
        select.onclick = () => {
          this.resolvesJob = item.job;
          this.showFeedback('请选择下方一条记忆，再进行更正或忘记。', 'notice');
          this.list.querySelector('button')?.focus();
        };
        actions.append(select);
      }
      const abandon = text(this.document, 'button', '放弃本次处理'); abandon.type = 'button';
      abandon.onclick = () => this.abandon(item.job, abandon);
      actions.append(abandon); article.append(actions); return article;
    }));
    this.attention.hidden = items.length === 0;
  }

  renderItems() {
    this.list.replaceChildren(...this.items.map(item => this.renderItem(item)));
  }

  renderItem(item) {
    const article = this.document.createElement('article');
    article.className = 'memory-item'; article.dataset.recordId = item.record_id;
    const body = text(this.document, 'p', item.summary, 'memory-content');
    const meta = this.document.createElement('div'); meta.className = 'memory-meta';
    meta.append(text(this.document, 'span', item.status_label),
      text(this.document, 'span', eventTimeText(item)),
      text(this.document, 'span', `${item.updated_at_label || '录入/修订时间'}：${formatDate(item.updated_at)}`),
      text(this.document, 'span', item.source_label));
    const actions = this.document.createElement('div'); actions.className = 'memory-actions';
    const details = text(this.document, 'button', '来源与历史'); details.type = 'button';
    const detailBox = this.document.createElement('div'); detailBox.className = 'memory-details'; detailBox.hidden = true;
    details.onclick = () => this.loadDetails(item, detailBox, details);
    actions.append(details);
    if (item.actions.correct) {
      const correct = text(this.document, 'button', '更正'); correct.type = 'button';
      correct.onclick = () => this.showCorrection(item, detailBox, correct);
      actions.append(correct);
    }
    if (item.actions.forget) {
      const forget = text(this.document, 'button', '忘记'); forget.type = 'button';
      forget.onclick = () => this.operate(item, 'forget', '', forget);
      actions.append(forget);
    }
    article.append(body, meta, actions, detailBox); return article;
  }

  async loadDetails(item, box, button) {
    box.hidden = false; box.replaceChildren(text(this.document, 'p', '正在读取来源与历史…'));
    button.disabled = true;
    try {
      const value = await this.api.memoryDetails(item.record_id);
      const source = this.document.createElement('section');
      source.append(text(this.document, 'h3', '来源'),
        text(this.document, 'p', value.source.text || `${value.source.label}（无对话原文）`));
      const history = this.document.createElement('section'); history.append(text(this.document, 'h3', '修订历史'));
      const list = this.document.createElement('ol');
      for (const version of value.history) {
        const row = this.document.createElement('li');
        row.append(text(this.document, 'span', version.content),
          text(this.document, 'small', `${version.status_label} · ${eventTimeText(version)} · ${version.updated_at_label || '录入/修订时间'}：${formatDate(version.updated_at)}`));
        list.append(row);
      }
      history.append(list); box.replaceChildren(source, history);
    } catch { box.replaceChildren(text(this.document, 'p', '详情读取失败，记忆没有被修改。')); }
    finally { button.disabled = false; }
  }

  showCorrection(item, box, button) {
    box.hidden = false;
    const form = this.document.createElement('form'); form.className = 'memory-correction';
    const label = text(this.document, 'label', '更正后的完整事实');
    const input = this.document.createElement('textarea'); input.rows = 3; input.maxLength = 8000;
    input.placeholder = '请完整写出更正后的内容'; label.append(input);
    const submit = text(this.document, 'button', '提交更正'); submit.type = 'submit';
    form.append(label, submit); box.replaceChildren(form); button.disabled = true;
    form.onsubmit = event => {
      event.preventDefault();
      if (!input.value.trim()) { this.showFeedback('请先写出更正后的完整内容。', 'error'); input.focus(); return; }
      this.operate(item, 'correct', input.value, submit).finally(() => { button.disabled = false; });
    };
    input.focus();
  }

  async operate(item, action, replacement, button) {
    const key = JSON.stringify([item.record_id, item.version, action, replacement, this.resolvesJob]);
    const operation = this.operations.get(key) || this.randomUUID(); this.operations.set(key, operation);
    button.disabled = true;
    try {
      const result = await this.api.memoryAction({ operation_id: operation, action,
        record_id: item.record_id, expected_version: item.version, replacement,
        ...(this.resolvesJob ? { resolves_job: this.resolvesJob } : {}) });
      this.operations.delete(key); this.resolvesJob = null;
      this.showFeedback(result.message, result.status === 'version_changed' ? 'notice' : 'success');
      await this.load(true);
      if (result.status === 'version_changed') {
        this.items = mergeCurrentMemoryPage(this.items, result.current);
        this.renderItems();
      }
      return result;
    } catch (error) {
      this.showFeedback('操作没有提交。再次尝试会沿用同一操作标识。', 'error');
      return null;
    } finally { button.disabled = false; }
  }

  async abandon(job, button) {
    const key = `abandon:${job}`;
    const operation = this.operations.get(key) || this.randomUUID(); this.operations.set(key, operation);
    button.disabled = true;
    try {
      const result = await this.api.memoryAbandon({ job, operation_id: operation });
      this.operations.delete(key); if (this.resolvesJob === job) this.resolvesJob = null;
      this.showFeedback(result.message); await this.load(true);
    } catch { this.showFeedback('放弃操作尚未提交，请重试。', 'error'); }
    finally { button.disabled = false; }
  }
}
