export function initializePanels() {
  const body = document.body;
  const toggle = document.getElementById('companion-toggle');
  const history = document.getElementById('history-toggle');
  const narrow = matchMedia('(max-width: 1100px)');
  let collapsed = localStorage.getItem('companion-collapsed') === 'true';
  let overlay = false;
  function update() {
    body.classList.toggle('companion-collapsed', collapsed);
    body.classList.toggle('companion-overlay', narrow.matches && overlay);
    toggle.setAttribute('aria-expanded', String(narrow.matches ? overlay : !collapsed));
  }
  toggle.onclick = () => {
    if (narrow.matches) overlay = !overlay;
    else { collapsed = !collapsed; localStorage.setItem('companion-collapsed', String(collapsed)); }
    update();
  };
  document.getElementById('companion-close').onclick = () => {
    if (narrow.matches) overlay = false;
    else { collapsed = true; localStorage.setItem('companion-collapsed', 'true'); }
    update(); toggle.focus();
  };
  history.onclick = () => {
    const open = body.classList.toggle('history-open');
    history.setAttribute('aria-expanded', String(open));
  };
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      overlay = false; body.classList.remove('history-open'); history.setAttribute('aria-expanded', 'false'); update();
    }
  });
  narrow.addEventListener('change', () => { overlay = false; update(); });
  update();
}
