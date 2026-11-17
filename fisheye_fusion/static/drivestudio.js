// Page shell (tabs, active section) and the DriveStudio+ panels: data preparation, training, run comparison.
// Talks to /api/setup, /api/drivestudio*, /api/pipeline/{id} (fisheye_fusion/server.py, drivestudio.py).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const enc = (p) => p.split('/').map(encodeURIComponent).join('/');
const STATE = { running: ['Đang chạy', 'run'], succeeded: ['Xong', 'ok'], failed: ['Lỗi', 'bad'], stopped: ['Đã dừng', 'warn'] };
const POLL_MS = 10000;

let overview = { options: {}, runs: [] };
let setup = null;
let selected = null;
let sort = { key: 'updated', dir: -1 };
let pollTimer = null;

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const d = data.detail;
    throw new Error(Array.isArray(d) ? d.map((e) => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('\n') : d || res.statusText);
  }
  return data;
}

function toast(msg, error = false) {
  const el = $('#toast');
  el.textContent = msg;
  el.className = `toast${error ? ' error' : ''}`;
  el.hidden = false;
  clearTimeout(toast.t);
  toast.t = setTimeout(() => { el.hidden = true; }, error ? 9000 : 4000);
}

function fill(select, items, keep = true) {
  const prev = select.value;
  select.innerHTML = items.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('');
  if (keep && items.some(([v]) => v === prev)) select.value = prev;
}

// ------------------------------------------------------------------ shell: tabs + active nav item
function initTabs() {
  for (const list of $$('[role=tablist]')) {
    const tabs = $$('[role=tab]', list);
    const show = (tab, focus = false) => {
      for (const t of tabs) {
        const on = t === tab;
        t.setAttribute('aria-selected', String(on));
        t.tabIndex = on ? 0 : -1;
        document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
      }
      if (focus) tab.focus();
    };
    list.addEventListener('click', (e) => { const t = e.target.closest('[role=tab]'); if (t) show(t); });
    list.addEventListener('keydown', (e) => {
      const i = tabs.indexOf(document.activeElement);
      if (i < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      e.preventDefault();
      const next = e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
      show(tabs[next], true);
    });
    list.show = show;
  }
}

function showTab(id) {
  const tab = document.getElementById(id);
  tab?.closest('[role=tablist]')?.show(tab);
}

function initNav() {
  const links = $$('#steps a');
  const byId = new Map(links.map((a) => [a.getAttribute('href').slice(1), a]));
  const mark = (id) => links.forEach((a) => a.setAttribute('aria-current', String(byId.get(id) === a)));
  const io = new IntersectionObserver((entries) => {
    const visible = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
    if (visible.length) mark(visible[0].target.id);
  }, { rootMargin: '-10% 0px -60% 0px' });
  byId.forEach((_, id) => { const s = document.getElementById(id); if (s) io.observe(s); });
  mark('overview');
  // pipeline.js jumps to #step-results after a NuRec evaluation: show that tab
  window.addEventListener('hashchange', () => { if (location.hash === '#step-results' && !$('#evalJob').hidden) showTab('t-res-nurec'); });
}

// ------------------------------------------------------------------ data
async function loadSetup() {
  setup = await api('/api/setup');
  const f = $('#dsPrepareForm');
  fill(f.elements.dataset, setup.datasets.map((d) => [d.path, d.path]));
  updateSequences();
  const gpus = setup.gpus.map((g) => [String(g.index), `GPU ${g.index} · ${g.name} · ${g.utilization}% · ${(g.memory_used_mib / 1024).toFixed(0)}/${(g.memory_total_mib / 1024).toFixed(0)} GB`]);
  fill($('#dsTrainForm').elements.gpu, gpus.length ? gpus : [['', 'mặc định']]);
  // default to the least loaded GPU
  const free = [...setup.gpus].sort((a, b) => a.utilization - b.utilization || a.memory_used_mib - b.memory_used_mib)[0];
  if (free && !loadSetup.done) $('#dsTrainForm').elements.gpu.value = String(free.index);
  loadSetup.done = true;
}

function updateSequences() {
  const f = $('#dsPrepareForm');
  const ds = setup?.datasets.find((d) => d.path === f.elements.dataset.value);
  fill(f.elements.sequence, Object.keys(ds?.sequences || {}).map((s) => [s, s]));
}

async function loadRuns() {
  overview = await api('/api/drivestudio');
  const o = overview.options;
  const t = $('#dsTrainForm');
  fill(t.elements.method, (o.methods || []).map((m) => [m, m.replace(/^configs\//, '')]));
  if (!loadRuns.done && (o.methods || []).includes('configs/streetgs_ut.yaml')) t.elements.method.value = 'configs/streetgs_ut.yaml';
  fill(t.elements.dataset, (o.datasets || []).map((d) => [d, d]));
  if (!loadRuns.done && (o.datasets || []).includes('kitti360/ncore_3cams_objects')) t.elements.dataset.value = 'kitti360/ncore_3cams_objects';
  fill(t.elements.data_root, [['', 'theo dataset config'], ...(o.data_roots || []).map((d) => [d, d])]);
  loadRuns.done = true;
  $('#dsTrainHint').innerHTML = o.venv_ok ? `Môi trường <code>${esc(o.venv)}</code>` : `<span class="tag bad">thiếu môi trường DriveStudio+</span> <code>${esc(o.venv)}</code>`;
  $('#dsEnv').innerHTML = `
    <dt>DriveStudio+</dt><dd>${o.venv_ok ? '<span class="tag ok">có</span>' : '<span class="tag bad">chưa có</span>'} <code>${esc(o.venv)}</code></dd>
    <dt>Dữ liệu đã chuẩn bị</dt><dd>${(o.data_roots || []).map((d) => `<code>${esc(d)}</code>`).join(' ') || '<span class="muted">chưa có, xem bước 1</span>'}</dd>`;
  renderKpis();
  renderRuns();
  if (selected) renderViewer();
  $('#dsUpdated').textContent = `Cập nhật ${new Date().toLocaleTimeString('vi-VN')}`;
  clearTimeout(pollTimer);
  if (overview.runs.some((r) => r.state === 'running')) pollTimer = setTimeout(() => loadRuns().catch(() => {}), POLL_MS);
}

// ------------------------------------------------------------------ overview tiles
function renderKpis() {
  const runs = overview.runs;
  const done = runs.filter((r) => r.state === 'succeeded' && r.metrics?.['Full Image PSNR'] != null);
  const best = done.sort((a, b) => b.metrics['Full Image PSNR'] - a.metrics['Full Image PSNR'])[0];
  const running = runs.filter((r) => r.state === 'running');
  const gpuBusy = (setup?.gpus || []).filter((g) => g.utilization > 20).length;
  const tile = (label, value, sub, ok = false) => `<div class="kpi"><div class="label">${label}</div><div class="value${ok ? ' ok' : ''}">${value}</div><div class="sub">${sub}</div></div>`;
  $('#dsKpis').innerHTML = [
    tile('Lần chạy DriveStudio+', runs.length, `${done.length} xong · ${runs.filter((r) => r.state === 'failed').length} lỗi`),
    tile('Đang chạy', running.length, running.map((r) => `${esc(r.name)} ${r.step ?? 0}/${r.total ?? '?'}`).join(' · ') || 'không có', running.length > 0),
    tile('PSNR test tốt nhất', best ? `${best.metrics['Full Image PSNR'].toFixed(2)} dB` : '—', best ? esc(best.name) : 'chưa có lần chạy xong'),
    tile('GPU đang bận', `${gpuBusy}/${setup?.gpus.length ?? 0}`, 'dùng > 20%'),
  ].join('');
}

// ------------------------------------------------------------------ comparison table
const COLS = [
  ['name', 'Lần chạy'], ['method', 'Phương pháp'], ['render', 'Renderer'], ['fastgs', 'FastGS'], ['state', 'Trạng thái'],
  ['progress', 'Tiến độ'], ['psnr', 'PSNR', 'num', 1], ['nonsky', 'PSNR (bỏ trời)', 'num', 1], ['ssim', 'SSIM', 'num', 1],
  ['lpips', 'LPIPS ↓', 'num', -1], ['elapsed', 'Thời gian', 'num'],
];
const metric = (r, k) => r.metrics?.[k] ?? null;
const VALUE = {
  name: (r) => r.name, method: (r) => (r.method || '').replace(/^configs\//, '').replace(/\.yaml$/, ''),
  render: (r) => r.render_mode || (/_geer/.test(r.method || '') ? 'geer' : /_ut/.test(r.method || '') ? 'ut' : 'config'),
  fastgs: (r) => r.fastgs ?? (/fastgs/.test(r.method || '') ? 'true' : '—'), state: (r) => r.state,
  progress: (r) => (r.total ? r.step / r.total : 0), psnr: (r) => metric(r, 'Full Image PSNR'), nonsky: (r) => metric(r, 'Non-Sky PSNR'),
  ssim: (r) => metric(r, 'Full Image SSIM'), lpips: (r) => metric(r, 'Full Image LPIPS'), elapsed: (r) => r.elapsed || '', updated: (r) => r.updated || 0,
};

function renderRuns() {
  const table = $('#dsRuns');
  const runs = [...overview.runs];
  if (!runs.length) {
    table.innerHTML = `<tbody><tr><td class="empty" colspan="${COLS.length}"><b>Chưa có lần chạy nào</b>Chuẩn bị dữ liệu ở bước 1, rồi bắt đầu train ở bước 2. Lần chạy từ dòng lệnh (<code>fflab train --backend drivestudio</code>) cũng hiện ở đây.</td></tr></tbody>`;
    return;
  }
  const finished = runs.filter((r) => r.state === 'succeeded');
  const best = {};
  for (const [key, , kind, dir] of COLS) {
    if (kind !== 'num' || !dir) continue;
    const vals = finished.map(VALUE[key]).filter((v) => v != null);
    if (vals.length > 1) best[key] = dir > 0 ? Math.max(...vals) : Math.min(...vals);
  }
  runs.sort((a, b) => {
    const va = VALUE[sort.key](a), vb = VALUE[sort.key](b);
    if (va == null) return 1; if (vb == null) return -1;
    return (va > vb ? 1 : va < vb ? -1 : 0) * sort.dir;
  });
  const head = COLS.map(([key, label, kind]) => `<th scope="col" class="${kind || ''}" aria-sort="${sort.key === key ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-sort="${key}">${label}</button></th>`).join('');
  const fmt = (v, d) => (v == null ? '<span class="muted">—</span>' : v.toFixed(d));
  const body = runs.map((r) => {
    const [stateText, stateCls] = STATE[r.state] || [r.state, ''];
    const pct = r.total ? Math.min(100, (100 * r.step) / r.total) : 0;
    const prog = r.step == null ? '<span class="muted">đang khởi tạo</span>'
      : `<span class="mono">${r.step.toLocaleString('vi-VN')}/${r.total.toLocaleString('vi-VN')}</span>${r.state === 'running' && r.eta ? ` <span class="muted">· còn ${esc(r.eta)}</span>` : ''}<div class="bar${r.state === 'running' ? ' run' : ''}"><i style="width:${pct}%"></i></div>`;
    // only the evaluation after the last step is a test result; earlier blocks are periodic checks on a few images
    const final = r.state === 'succeeded';
    const cell = (key, d) => {
      const v = VALUE[key](r);
      if (!final && v != null) return `<td class="num prov" title="Đánh giá tạm trên ${r.eval_images ?? '?'} ảnh, chưa phải kết quả test">${fmt(v, d)}*</td>`;
      return `<td class="num${best[key] != null && v === best[key] ? ' best' : ''}">${fmt(v, d)}</td>`;
    };
    return `<tr tabindex="0" data-run="${esc(r.name)}" aria-selected="${r.name === selected}">
      <td><b>${esc(r.name)}</b></td><td>${esc(VALUE.method(r))}</td><td>${esc(VALUE.render(r))}</td><td>${esc(VALUE.fastgs(r))}</td>
      <td><span class="tag ${stateCls}" title="${esc(r.error || '')}">${stateText}</span></td>
      <td class="prog">${prog}</td>${cell('psnr', 2)}${cell('nonsky', 2)}${cell('ssim', 3)}${cell('lpips', 3)}
      <td class="num mono">${esc(r.elapsed || '—')}</td></tr>`;
  }).join('');
  const foot = runs.some((r) => r.state !== 'succeeded' && r.metrics)
    ? `<tfoot><tr><td colspan="${COLS.length}" class="muted">* đánh giá tạm trong lúc train (vài ảnh), không so sánh được với kết quả test cuối.</td></tr></tfoot>` : '';
  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>${foot}`;
}

// ------------------------------------------------------------------ viewer
function stepsOf(run) {
  const steps = new Map();
  for (const p of run.images) {
    const m = p.match(/step_(\d+)_(.+)\.png$/);
    if (m) (steps.get(+m[1]) || steps.set(+m[1], new Map()).get(+m[1])).set(m[2], p);
  }
  return [...steps.entries()].sort((a, b) => b[0] - a[0]);
}

function renderViewer() {
  const run = overview.runs.find((r) => r.name === selected);
  const box = $('#dsViewer');
  if (!run) { box.hidden = true; return; }
  box.hidden = false;
  const steps = stepsOf(run);
  const cur = $('#dsStep')?.value;
  const step = steps.find(([s]) => String(s) === cur) || steps[0];
  const kindNames = { rgbs: 'Render', Background_rgbs: 'Chỉ nền', Dynamic_rgbs: 'Chỉ vật thể động' };
  const kindSel = $('#dsKind')?.value || 'rgbs';
  const img = (p, cap) => `<figure><img loading="lazy" src="/api/drivestudio/runs/${enc(run.name)}/files/${enc(p)}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>`;
  const files = step ? step[1] : new Map();
  const kinds = [...files.keys()].filter((k) => k !== 'gt_rgbs' && kindNames[k]);
  const shown = files.get(kindSel) ? kindSel : 'rgbs';
  box.innerHTML = `
    <div class="vhead">
      <h3>${esc(run.name)} <span class="tag ${STATE[run.state]?.[1] || ''}">${STATE[run.state]?.[0] || run.state}</span></h3>
      <div class="actions">
        <label>Bước <select id="dsStep">${steps.map(([s]) => `<option value="${s}" ${step && s === step[0] ? 'selected' : ''}>${s.toLocaleString('vi-VN')}</option>`).join('')}</select></label>
        <label>Hiển thị <select id="dsKind">${kinds.map((k) => `<option value="${k}" ${k === shown ? 'selected' : ''}>${kindNames[k]}</option>`).join('')}</select></label>
      </div>
    </div>
    ${run.error ? `<pre class="log">${esc(run.error)}</pre>` : ''}
    ${step ? `<div class="compare">${files.get('gt_rgbs') ? img(files.get('gt_rgbs'), 'Ảnh gốc (GT)') : ''}${files.get(shown) ? img(files.get(shown), `${kindNames[shown]} · bước ${step[0]}`) : ''}</div>`
      : '<p class="empty"><b>Chưa có ảnh</b>DriveStudio lưu ảnh render định kỳ trong lúc train.</p>'}
    ${run.videos.length ? `<h3 style="margin-top:16px">Video tập test</h3><div class="list">${run.videos.map((v) => `<a href="/api/drivestudio/runs/${enc(run.name)}/files/${enc(v)}" target="_blank" rel="noopener">${esc(v.split('/').pop())}</a>`).join('')}</div>` : ''}`;
}

// ------------------------------------------------------------------ jobs
async function pollJob(id, box, onDone) {
  const st = await api(`/api/pipeline/${id}`);
  const label = { running: 'Đang chạy', succeeded: 'Xong', failed: 'Lỗi', cancelled: 'Đã huỷ', cancelling: 'Đang huỷ' }[st.state] || st.state;
  box.hidden = false;
  box.innerHTML = `<div class="head">${st.state === 'running' ? '<span class="spinner" aria-hidden="true"></span>' : ''}
    <span class="state ${st.state}">${label}</span>${st.step ? `<span class="muted">bước: ${esc(st.step)}</span>` : ''}
    ${st.info?.data_root ? `<code>${esc(st.info.data_root)}</code>` : ''}</div>
    ${st.error ? `<pre>${esc(st.error)}</pre>` : ''}<pre>${esc(st.log.split('\n').slice(-30).join('\n'))}</pre>`;
  if (st.state === 'running' || st.state === 'cancelling') setTimeout(() => pollJob(id, box, onDone).catch((e) => toast(e.message, true)), 2000);
  else onDone?.(st);
}

function initForms() {
  $('#dsPrepareForm').addEventListener('change', (e) => { if (e.target.name === 'dataset') updateSequences(); });
  $('#dsPrepareForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    const btn = f.querySelector('button[type=submit]');
    btn.disabled = true;
    try {
      const st = await api('/api/drivestudio/prepare', {
        dataset: f.elements.dataset.value, sequence: f.elements.sequence.value, start: +f.elements.start.value, frames: +f.elements.frames.value,
        name: f.elements.name.value.trim(), objects: f.elements.objects.checked,
      });
      toast('Đã bắt đầu chuẩn bị dữ liệu.');
      await pollJob(st.job_id, $('#dsPrepareJob'), async (done) => {
        btn.disabled = false;
        if (done.state === 'succeeded') {
          toast(`Dữ liệu sẵn sàng: ${done.info.data_root}`);
          await loadRuns();
          const t = $('#dsTrainForm');
          t.elements.data_root.value = done.info.data_root;
          if ([...t.elements.dataset.options].some((o) => o.value === done.info.dataset_config)) t.elements.dataset.value = done.info.dataset_config;
        }
      });
    } catch (err) { btn.disabled = false; toast(err.message, true); }
  });

  $('#dsTrainForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const f = e.target;
    const btn = f.querySelector('button[type=submit]');
    btn.disabled = true;
    try {
      const body = {
        name: f.elements.name.value.trim(), method: f.elements.method.value, dataset: f.elements.dataset.value, data_root: f.elements.data_root.value,
        render_mode: f.elements.render_mode.value, steps: +f.elements.steps.value, gpu: f.elements.gpu.value === '' ? null : +f.elements.gpu.value,
      };
      if (f.elements.fastgs.value) body.fastgs = f.elements.fastgs.value === 'true';
      await api('/api/drivestudio/train', body);
      toast(`Đã bắt đầu ${body.name}. Theo dõi ở bước 3.`);
      selected = body.name;
      setTimeout(() => loadRuns().catch(() => {}), 1500);
      showTab('t-res-ds');
      location.hash = '#step-results';
    } catch (err) { toast(err.message, true); } finally { btn.disabled = false; }
  });

  const table = $('#dsRuns');
  table.addEventListener('click', (e) => {
    const s = e.target.closest('button[data-sort]');
    if (s) {
      sort = { key: s.dataset.sort, dir: sort.key === s.dataset.sort ? -sort.dir : (COLS.find((c) => c[0] === s.dataset.sort)?.[3] === -1 ? 1 : -1) };
      renderRuns();
      return;
    }
    const row = e.target.closest('tr[data-run]');
    if (row) { selected = row.dataset.run; renderRuns(); renderViewer(); $('#dsViewer').scrollIntoView({ block: 'nearest' }); }
  });
  table.addEventListener('keydown', (e) => {
    const row = e.target.closest('tr[data-run]');
    if (row && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); row.click(); }
  });
  $('#dsViewer').addEventListener('change', (e) => { if (e.target.id === 'dsStep' || e.target.id === 'dsKind') renderViewer(); });
  $('#dsViewer').addEventListener('click', (e) => {
    const img = e.target.closest('img');
    if (!img) return;
    const box = document.createElement('div');
    box.className = 'lightbox';
    box.innerHTML = `<img src="${img.src}" alt="${esc(img.alt)}">`;
    box.addEventListener('click', () => box.remove());
    document.addEventListener('keydown', function close(k) { if (k.key === 'Escape') { box.remove(); document.removeEventListener('keydown', close); } });
    document.body.append(box);
  });
  $('#dsRefresh').addEventListener('click', () => Promise.all([loadSetup(), loadRuns()]).catch((err) => toast(err.message, true)));
}

initTabs();
initNav();
initForms();
Promise.all([loadSetup(), loadRuns()]).then(renderKpis).catch((err) => toast(err.message, true));
