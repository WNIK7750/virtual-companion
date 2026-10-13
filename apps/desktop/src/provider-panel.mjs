const text = (document, name, value, className) => {
  const node = document.createElement(name);
  if (className) node.className = className;
  node.textContent = value;
  return node;
};

export class ProviderPanel {
  constructor({ api, document }) {
    this.api = api;
    this.document = document;
    this.openButton = document.getElementById('provider-open');
    this.label = document.getElementById('provider-label');
    this.dialog = document.getElementById('provider-dialog');
    this.closeButton = document.getElementById('provider-close');
    this.active = document.getElementById('provider-active');
    this.options = document.getElementById('provider-options');
    this.feedback = document.getElementById('provider-feedback');
    this.value = null;
    this.openButton.onclick = () => this.open();
    this.closeButton.onclick = () => this.dialog.close();
  }

  setState(value) {
    this.value = value || null;
    const active = value?.active;
    const status = active?.status === 'active' ? '当前'
      : active?.status === 'unconfigured' ? '未配置' : '未就绪';
    this.label.textContent = active?.label ? `${active.label} · ${status}` : '正在读取模型状态';
  }

  open() {
    if (!this.dialog.open) this.dialog.showModal();
    this.render(this.value);
    this.closeButton.focus();
  }

  render(value) {
    const active = value?.active;
    this.active.replaceChildren(
      text(this.document, 'strong', active ? `当前路由：${active.label}` : '正在读取模型状态'),
      text(this.document, 'p', active ? `${active.model_label} · ${active.capabilities.join('、')}` : '正在读取模型状态。'),
    );
    if (active?.config_revision) this.active.append(text(this.document, 'small', `生效配置：修订 ${active.config_revision} · ${active.model_profile}`));
    const options = value?.options || [];
    this.options.replaceChildren(...options.map(option => {
      const article = this.document.createElement('article');
      article.className = 'provider-option';
      const heading = this.document.createElement('div'); heading.className = 'provider-option-heading';
      const statusLabel = { available: '可用', active: '当前', requires_confirmation: '需要确认', unconfigured: '未配置', unavailable: '未就绪' }[option.status] || option.status;
      heading.append(text(this.document, 'strong', `${option.label} · ${option.model_label}`), text(this.document, 'small', statusLabel));
      article.append(heading, text(this.document, 'p', option.message));
      if (option.missing?.length) article.append(text(this.document, 'small', `待补：${option.missing.join('、')}`));
      let consent;
      if (option.requires_confirmation && option.selectable) {
        const label = this.document.createElement('label');
        consent = this.document.createElement('input'); consent.type = 'checkbox';
        consent.dataset.providerConsent = option.provider_id;
        label.append(consent, text(this.document, 'span', '我确认本伴侣上述内容可发送给 DeepSeek，包括后台整理。'));
        article.append(label);
      }
      const current = option.provider_id === active?.provider_id
        && active?.status === 'active' && !value?.recovery_required;
      const button = text(this.document, 'button', current ? '正在使用' : option.selectable ? '使用此路由' : '当前不可用');
      button.type = 'button';
      button.disabled = current || !option.selectable || Boolean(consent);
      if (consent) consent.onchange = () => { button.disabled = !consent.checked; };
      button.onclick = () => this.select(option.provider_id, consent?.checked === true);
      article.append(button); return article;
    }));
  }

  async select(providerId, confirmed = false) {
    try {
      const result = await this.api.selectProvider(providerId, confirmed);
      this.setState(result.provider);
      this.render(result.provider);
      this.feedback.textContent = result.message;
      this.feedback.dataset.kind = result.status === 'unavailable' ? 'notice' : 'success';
    } catch (error) {
      this.feedback.textContent = error.message;
      this.feedback.dataset.kind = 'error';
    }
  }
}
