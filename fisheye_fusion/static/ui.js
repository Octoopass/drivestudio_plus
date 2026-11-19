// Shell polish: "previous / next step" footer at the end of each step (step-by-step view), and the
// sidebar badge counting missing libraries (fed by deps.js through window.__fflab.setDepsBadge).

const $ = (sel, root = document) => root.querySelector(sel);
const STEPS = [
  ['overview', 'Tổng quan'],
  ['step-convert', '1. Dữ liệu'],
  ['step-train', '2. Huấn luyện / Dựng'],
  ['step-results', '3. Bảng xếp hạng & Video'],
];

STEPS.forEach(([id], i) => {
  const sec = document.getElementById(id);
  if (!sec || $('.step-foot', sec)) return;
  const prev = STEPS[i - 1];
  const next = STEPS[i + 1];
  const foot = document.createElement('nav');
  foot.className = 'step-foot';
  foot.setAttribute('aria-label', 'Chuyển bước');
  foot.innerHTML = `${prev ? `<button type="button" class="prev" data-step-go="${prev[0]}">← ${prev[1]}</button>` : ''}
    ${next ? `<button type="button" class="next" data-step-go="${next[0]}">Tiếp: ${next[1]} →</button>` : ''}`;
  sec.append(foot);
});

document.addEventListener('click', (e) => {
  const b = e.target.closest('[data-step-go]');
  if (!b) return;
  window.__fflab.goto?.(b.dataset.stepGo);
  window.scrollTo({ top: 0, behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
});

window.__fflab.setDepsBadge = (missing, total) => {
  const el = $('#navDepsBadge');
  if (!el) return;
  el.hidden = false;
  el.classList.toggle('ok', !missing);
  el.textContent = missing ? String(missing) : 'đủ';
  el.title = missing ? `Còn thiếu ${missing}/${total} thư viện` : 'Đã cài đủ thư viện';
  el.setAttribute('aria-label', el.title);
};
