const paths = {
  flower: '<path d="M11 11C2 13 0 4 5 2c5-2 7 3 6 9Zm2 0c-2-9 7-11 9-6 2 5-3 7-9 6Zm0 2c9-2 11 7 6 9-5 2-7-3-6-9Zm-2 0c2 9-7 11-9 6-2-5 3-7 9-6Z"/>',
  plus: '<path d="M12 4v16M4 12h16"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  chat: '<path d="M4 4h16v13H8l-4 4V4Z"/>',
  database: '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0"/>',
  sidebar: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M9 4v16"/>',
  panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16m-5-11 2 3-2 3"/>',
  heart: '<path d="M12 21 3.5 12.5C-3 5 8-1 12 7c4-8 15-2 8.5 5.5Z"/>',
  send: '<path d="m4 3 17 9-17 9 3-9-3-9Zm3 9h14"/>',
  mic: '<rect x="9" y="2" width="6" height="13" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3m-3 0h6"/>',
  eye: '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="m3 17 5-5 4 4 4-6 5 6"/><circle cx="8" cy="8" r="1"/>',
  headphones: '<path d="M3 14v-3a9 9 0 0 1 18 0v3"/><rect x="3" y="12" width="4" height="9" rx="2"/><rect x="17" y="12" width="4" height="9" rx="2"/>',
  video: '<rect x="2" y="5" width="13" height="14" rx="1"/><path d="m15 10 7-4v12l-7-4"/>',
  settings: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.8 1.8 0 0 0 .3 2l.1.1-1.8 1.8-.1-.1a1.8 1.8 0 0 0-2-.3 1.8 1.8 0 0 0-1.1 1.7v.2h-2.6v-.2a1.8 1.8 0 0 0-1.1-1.7 1.8 1.8 0 0 0-2 .3l-.1.1-1.8-1.8.1-.1a1.8 1.8 0 0 0 .3-2 1.8 1.8 0 0 0-1.7-1.1h-.2v-2.6h.2a1.8 1.8 0 0 0 1.7-1.1 1.8 1.8 0 0 0-.3-2l-.1-.1 1.8-1.8.1.1a1.8 1.8 0 0 0 2 .3 1.8 1.8 0 0 0 1.1-1.7v-.2h2.6v.2a1.8 1.8 0 0 0 1.1 1.7 1.8 1.8 0 0 0 2-.3l.1-.1 1.8 1.8-.1.1a1.8 1.8 0 0 0-.3 2 1.8 1.8 0 0 0 1.7 1.1h.2v2.6h-.2a1.8 1.8 0 0 0-1.7 1.1Z"/>',
  chevron: '<path d="m9 5 7 7-7 7"/>',
};
export function icon(name) {
  const span = document.createElement('span');
  span.setAttribute('aria-hidden', 'true');
  // Static code-owned geometry only; never inserts model/user content.
  span.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[name] || ''}</svg>`;
  return span;
}
export function populateIcons(root = document) {
  for (const element of root.querySelectorAll('[data-icon]')) element.replaceChildren(icon(element.dataset.icon));
}
