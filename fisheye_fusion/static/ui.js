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

// Deep link to a job: /?job=<32-hex id> shows that pipeline job (progress board + log) at the top of the step it
// belongs to, so a job started elsewhere (another tab, the API, a script) can be followed from any browser.
const JOB_STEP = { prepare: 'step-convert', preprocess: 'step-convert', drive_sync: 'step-convert', convert: 'step-convert',
  'nurec-setup': 'step-train', evaluate: 'step-results' };
const JOB_LABEL = { prepare: 'Chuẩn bị dữ liệu DriveStudio+', preprocess: 'Tiền xử lý dataset', drive_sync: 'Đồng bộ Google Drive',
  convert: 'Chuyển KITTI-360 sang NCore', 'nurec-setup': 'Cài Instant NuRec', evaluate: 'Chấm điểm' };
(async () => {
  const id = new URLSearchParams(location.search).get('job') || '';
  if (!/^[0-9a-f]{32}$/.test(id)) return;
  let s;
  try { s = await window.__fflab.api(`/api/pipeline/${id}`); } catch (err) { window.__fflab.toast(`Không mở được job: ${err.message}`, true); return; }
  const stepId = JOB_STEP[s.kind] || (s.kind.startsWith('deps-') ? 'overview' : 'overview');
  const sec = document.getElementById(stepId);
  const panel = document.createElement('section');
  panel.className = 'panel linked-job';
  panel.setAttribute('aria-label', 'Job đang theo dõi');
  panel.innerHTML = `<h3>Job đang theo dõi: ${window.__fflab.esc(JOB_LABEL[s.kind] || s.kind)}</h3><div class="job"></div>`;
  ($('.sec-head', sec) || sec.firstElementChild).after(panel);
  window.__fflab.goto?.(stepId);
  window.__fflab.watchJob($('.job', panel), `/api/pipeline/${id}`, `/api/pipeline/${id}/cancel`, null, () => {
    window.__fflab.reloadData?.()?.catch?.(() => {});
    window.__fflab.fetchSetup?.(true)?.catch?.(() => {});
  });
})();
