// Only the newly approved suffix counts. Old visible text cannot acknowledge a
// later unit that is still clipped by the scroll container or the window.
function textNodes(body) {
  const nodes = [];
  const visit = node => {
    if (node.nodeType === 3) nodes.push(node);
    else for (const child of node.childNodes || []) visit(child);
  };
  for (const child of body?.childNodes || (body?.firstChild ? [body.firstChild] : [])) visit(child);
  return nodes;
}

function setTextRange(range, nodes, start, end) {
  const point = (offset, ending) => {
    for (let index = 0; index < nodes.length; index += 1) {
      const node = nodes[index], length = node.textContent.length;
      if (offset < length || ending && offset === length || index === nodes.length - 1) return [node, offset];
      offset -= length;
    }
  };
  range.setStart(...point(start, false));
  range.setEnd(...point(end, true));
}

export function approvedTextIsVisible({ element, elements, container, payload, document, view }) {
  if (!element?.isConnected || !container?.isConnected || document.visibilityState !== 'visible'
      || element.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) === false
      || typeof payload.text !== 'string' || !payload.text.trim()
      || typeof payload.display_text !== 'string' || !payload.display_text.endsWith(payload.text)) return false;
  const roots = elements?.length ? elements : [element];
  const bodies = roots.map(root => root.querySelector('div'));
  const nodes = bodies.flatMap(textNodes);
  if (bodies.map(body => body?.textContent || '').join('') !== payload.display_text || !nodes.length
      || nodes.map(node => node.textContent).join('') !== payload.display_text
      || roots.some(root => !root.isConnected)
      || bodies.some(body => body?.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) === false)) return false;
  const bounds = container.getBoundingClientRect();
  const left = Math.max(0, bounds.left + container.clientLeft);
  const top = Math.max(0, bounds.top + container.clientTop);
  const right = Math.min(view.innerWidth, bounds.left + container.clientLeft + container.clientWidth);
  const bottom = Math.min(view.innerHeight, bounds.top + container.clientTop + container.clientHeight);
  if (!(right > left && bottom > top)) return false;
  const range = document.createRange();
  setTextRange(range, nodes, payload.display_text.length - payload.text.length, payload.display_text.length);
  const lines = [...range.getClientRects()].filter(rect => rect.right > rect.left && rect.bottom > rect.top);
  // Full short-unit coverage only. Long units that cannot fit stay unknown; a
  // pixel or one visible character is not proof of the whole approved unit.
  return lines.length > 0 && lines.every(rect => {
    if (rect.left < left || rect.right > right || rect.top < top || rect.bottom > bottom) return false;
    if (typeof document.elementFromPoint !== 'function') return true;
    const hit = document.elementFromPoint((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2);
    return bodies.some(body => hit === body || body.contains?.(hit) === true);
  });
}

export function isReadingNavigation(event) {
  return event.type !== 'keydown' || (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
    'PageUp', 'PageDown', 'Home', 'End', ' '].includes(event.key)
    && !event.target?.isContentEditable && !event.target?.closest?.('input, textarea, select'));
}

export function waitForApprovedText(options) {
  const { element, elements, container, document, view, payload, deadlineMs, recovery = false,
    cancelled = () => false, presentUnseen = false, onUserNavigation } = options;
  const remaining = Math.min(recovery === true ? 120000 : 10000, deadlineMs - Date.now());
  const text = payload?.text, displayText = payload?.display_text;
  if (!Number.isFinite(remaining) || remaining <= 0 || cancelled()
      || typeof text !== 'string' || !text.trim() || typeof displayText !== 'string'
      || !displayText.endsWith(text) || new TextEncoder().encode(displayText).length > 65536) {
    return Promise.resolve(false);
  }
  const roots = elements?.length ? elements : [element];
  const bodies = roots.map(root => root?.querySelector('div')), nodes = bodies.flatMap(textNodes);
  if (!nodes.length || bodies.map(body => body?.textContent || '').join('') !== displayText
      || nodes.map(node => node.textContent).join('') !== displayText) return Promise.resolve(false);
  // DOM Range offsets use UTF-16. A grapheme's combining/ZWJ code points share
  // its visible glyph; neither a grapheme nor a surrogate pair may be split.
  for (const point of text) {
    const code = point.codePointAt(0);
    if (code >= 0xD800 && code <= 0xDFFF) return Promise.resolve(false);
  }
  const characters = typeof Intl.Segmenter === 'function'
    ? Array.from(new Intl.Segmenter('und', { granularity: 'grapheme' }).segment(text), item => item.segment)
    : Array.from(text);
  const offsets = [displayText.length - text.length];
  for (const character of characters) {
    offsets.push(offsets.at(-1) + character.length);
  }
  const root = { start: 0, end: characters.length, complete: false };
  return new Promise(resolve => {
    let settled = false, checking = false, firstFrame, secondFrame, painted = false;
    let timer, cancellationTimer, observer;
    let following = presentUnseen === true, yielded = false, moved = false, advancedCoverage = -1;
    const originalLeft = container.scrollLeft;
    const targets = [[container, 'scroll'], [view, 'resize'], [document, 'visibilitychange'],
      [document, 'scroll', true]];
    const inputTargets = [[container, 'wheel'], [container, 'pointerdown'], [document, 'keydown', true]];
    const yieldToUser = event => {
      if (!isReadingNavigation(event)) return;
      following = false;
      if (!yielded) { yielded = true; onUserNavigation?.(); }
    };
    const finish = visible => {
      if (settled) return;
      settled = true; clearTimeout(timer); clearInterval(cancellationTimer);
      observer?.disconnect();
      view.cancelAnimationFrame(firstFrame); view.cancelAnimationFrame(secondFrame);
      for (const [target, event, capture = false] of targets) target.removeEventListener(event, check, { capture });
      for (const [target, event, capture = false] of inputTargets) target.removeEventListener(event, yieldToUser, { capture });
      resolve(visible);
    };
    const sameNodes = () => {
      const current = bodies.flatMap(textNodes);
      return current.length === nodes.length && current.every((node, index) => node === nodes[index]);
    };
    const invalid = () => cancelled() || roots.some(root => !root?.isConnected) || !container.isConnected
      || payload.text !== text || payload.display_text !== displayText
      || roots.some((root, index) => root.querySelector('div') !== bodies[index]) || !sameNodes()
      || bodies.map(body => body.textContent).join('') !== displayText || (observer?.takeRecords().length ?? 0) > 0;
    const covered = branch => branch.complete ? branch.end - branch.start
      : (branch.children?.reduce((total, child) => total + covered(child), 0) ?? 0);
    const unseen = branch => branch.complete ? null
      : (branch.children ? branch.children.map(unseen).find(Boolean) : branch);
    const advance = (left, top, right, bottom, coverage) => {
      if (!following || coverage <= advancedCoverage) return false;
      const branch = unseen(root), range = document.createRange();
      let rects = [];
      for (let index = branch.start; index < branch.end; index += 1) {
        setTextRange(range, nodes, offsets[index], offsets[index + 1]);
        rects = [...range.getClientRects()].filter(rect => rect.right > rect.left && rect.bottom > rect.top);
        if (rects.length || !/^\s+$/u.test(characters[index])) break;
      }
      if (!rects.length) return false;
      const box = { left: Math.min(...rects.map(rect => rect.left)), right: Math.max(...rects.map(rect => rect.right)),
        top: Math.min(...rects.map(rect => rect.top)), bottom: Math.max(...rects.map(rect => rect.bottom)) };
      // A covered glyph or a glyph that cannot fit offers no scrolling progress.
      // Occlusion cannot be converted to coverage by moving on to later text.
      if (box.right - box.left > right - left || box.bottom - box.top > bottom - top
        || (box.left >= left && box.right <= right && box.top >= top && box.bottom <= bottom)) return false;
      const oldTop = container.scrollTop, oldLeft = container.scrollLeft;
      const padY = Math.min(1, (bottom - top - (box.bottom - box.top)) / 2);
      const padX = Math.min(1, (right - left - (box.right - box.left)) / 2);
      const nextTop = box.top < top || box.bottom > bottom
        ? Math.max(0, Math.min(container.scrollHeight - container.clientHeight, oldTop + box.top - top - padY)) : oldTop;
      const nextLeft = box.left < left || box.right > right
        ? Math.max(0, Math.min(container.scrollWidth - container.clientWidth, oldLeft + box.left - left - padX)) : oldLeft;
      container.scrollTop = nextTop; container.scrollLeft = nextLeft;
      if (container.scrollTop === oldTop && container.scrollLeft === oldLeft) return false;
      advancedCoverage = coverage; moved = true;
      // Movement schedules another two-painted-frame sample; it is not evidence.
      return true;
    };
    const sample = () => {
      if (invalid()) { finish(false); return false; }
      if (document.visibilityState !== 'visible'
          || roots.some(root => root.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) === false)
          || bodies.some(body => body.checkVisibility?.({ checkOpacity: true, checkVisibilityCSS: true }) === false)) return false;
      const bounds = container.getBoundingClientRect();
      let left = Math.max(0, bounds.left + container.clientLeft);
      let top = Math.max(0, bounds.top + container.clientTop);
      let right = Math.min(view.innerWidth, bounds.left + container.clientLeft + container.clientWidth);
      let bottom = Math.min(view.innerHeight, bounds.top + container.clientTop + container.clientHeight);
      // A nested text scroller can itself be clipped by its dialog. Checking a
      // glyph's center alone would miss partially cropped ascenders/descenders.
      for (let ancestor = container.parentElement; ancestor; ancestor = ancestor.parentElement) {
        const style = view.getComputedStyle?.(ancestor);
        const clipsX = ['auto', 'scroll', 'hidden', 'clip'].includes(style?.overflowX);
        const clipsY = ['auto', 'scroll', 'hidden', 'clip'].includes(style?.overflowY);
        if (!clipsX && !clipsY) continue;
        const box = ancestor.getBoundingClientRect();
        if (clipsX) {
          left = Math.max(left, box.left + ancestor.clientLeft);
          right = Math.min(right, box.left + ancestor.clientLeft + ancestor.clientWidth);
        }
        if (clipsY) {
          top = Math.max(top, box.top + ancestor.clientTop);
          bottom = Math.min(bottom, box.top + ancestor.clientTop + ancestor.clientHeight);
        }
      }
      if (!(right > left && bottom > top)) return false;
      const range = document.createRange();
      let probes = 0, exhausted = false;
      const beforeCoverage = covered(root);
      const scan = branch => {
        if (branch.complete) return;
        if (probes >= 4096) { exhausted = true; return; }
        probes += 1;
        setTextRange(range, nodes, offsets[branch.start], offsets[branch.end]);
        const rects = [...range.getClientRects()].filter(rect => rect.right > rect.left && rect.bottom > rect.top);
        if (branch.end - branch.start === 1) {
          // A newline/collapsed space has no ink to hit-test. It is only accepted
          // as part of a unit whose every painted character is separately seen;
          // an invisible non-whitespace character never gains this exemption.
          if (!rects.length) { branch.complete = /^\s+$/u.test(characters[branch.start]); return; }
          branch.complete = rects.every(rect => {
            if (rect.left < left || rect.right > right || rect.top < top || rect.bottom > bottom) return false;
            if (typeof document.elementFromPoint !== 'function') return true;
            const hit = document.elementFromPoint((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2);
            return bodies.some(body => hit === body || body.contains?.(hit) === true);
          });
          if (branch.complete && !/^\s+$/u.test(characters[branch.start])) painted = true;
          return;
        }
        // Prune wholly offscreen ranges. Only potentially visible characters
        // require individual hit tests, so a long offscreen answer is cheap.
        if (rects.length && !rects.some(rect => rect.right > left && rect.left < right
            && rect.bottom > top && rect.top < bottom)) return;
        if (!branch.children) {
          const middle = Math.floor((branch.start + branch.end) / 2);
          branch.children = [{ start: branch.start, end: middle, complete: false },
            { start: middle, end: branch.end, complete: false }];
        }
        for (const child of branch.children) scan(child);
        branch.complete = branch.children.every(child => child.complete);
      };
      scan(root);
      if (invalid()) finish(false);
      else if (root.complete && painted) {
        // Keep ordinary latest-message following after all glyphs were proved.
        // This final position contributes no coverage and never runs after yield.
        if (following && moved) { container.scrollTop = container.scrollHeight; container.scrollLeft = originalLeft; }
        finish(true);
      } else {
        const coverage = covered(root);
        if (exhausted) return coverage > beforeCoverage;
        return advance(left, top, right, bottom, coverage);
      }
      return false;
    };
    const check = () => {
      if (settled || checking) return;
      if (invalid()) { finish(false); return; }
      checking = true;
      firstFrame = view.requestAnimationFrame(() => {
        secondFrame = view.requestAnimationFrame(() => {
          checking = false;
          if (Date.now() >= deadlineMs) finish(false);
          else if (sample()) check();
        });
      });
    };
    // A change-and-restore between paints also invalidates the accumulated
    // observation. Coverage is never transferred to a replacement text node.
    if (typeof view.MutationObserver === 'function') {
      observer = new view.MutationObserver(() => finish(false));
      for (const body of bodies) observer.observe(body, { subtree: true, childList: true, characterData: true });
    }
    timer = setTimeout(() => finish(false), remaining);
    cancellationTimer = setInterval(() => { if (invalid()) finish(false); }, 100);
    for (const [target, event, capture = false] of targets) target.addEventListener(event, check, { passive: true, capture });
    if (presentUnseen === true || onUserNavigation) {
      for (const [target, event, capture = false] of inputTargets) target.addEventListener(event, yieldToUser, { passive: true, capture });
    }
    check();
  });
}
