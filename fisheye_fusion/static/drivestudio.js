// Page shell (tabs, active section) and the DriveStudio+ panels: data preparation, training, run comparison.
// Talks to /api/setup, /api/drivestudio*, /api/pipeline/{id} (fisheye_fusion/server.py, drivestudio.py).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const enc = (p) => p.split('/').map(encodeURIComponent).join('/');
const STATE = { running: ['Đang chạy', 'run'], succeeded: ['Xong', 'ok'], failed: ['Lỗi', 'bad'], stopped: ['Đã dừng', 'warn'] };
const POLL_MS = 10000;

let overview = { options: {}, runs: [], videos: [] };
let setup = null;
let selected = null;
let sort = { key: 'updated', dir: -1 };
let pollTimer = null;
let viewer = { state: 'stopped' };
let viewerTimer = null;

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
  const activate = () => {
    const id = byId.has(location.hash.slice(1)) ? location.hash.slice(1) : 'overview';
    links.forEach(a => a.setAttribute('aria-current', String(byId.get(id) === a)));
    $$('.sec').forEach(s => s.classList.toggle('is-active', s.id === id));
    if (location.hash === '#' + id) document.getElementById(id).scrollIntoView();
  };
  window.__fflab.goto = (id, tab) => {
    if (!byId.has(id)) return;
    location.hash = id;
    activate();
    if (tab) showTab(tab);
  };
  $$('[data-goto]').forEach(b => b.onclick = () => window.__fflab.goto(b.dataset.goto));
  $('#toggleViewMode').onclick = e => {
    const focus = document.body.dataset.view !== 'focus';
    document.body.dataset.view = focus ? 'focus' : 'scroll';
    e.currentTarget.textContent = focus ? 'Chế độ: Từng bước' : 'Chế độ: Cuộn';
    activate();
  };
  window.addEventListener('hashchange', activate);
  activate();
}

// ------------------------------------------------------------------ data
async function loadSetup(preferred) {
  setup = await window.__fflab.fetchSetup();
  const f = $('#dsPrepareForm');
  fill(f.elements.dataset, setup.datasets.map((d) => [d.path, d.path]));
  updateSequences();
  const picker = $('#localSequence');
  const entries = setup.datasets.flatMap(d => Object.entries(d.sequences || {}).map(([seq, info]) =>
    ({ value: JSON.stringify([d.path, seq]), path: d.path, seq, info, meta: d.calibration_pickles })));
  const prev = picker.value;
  fill(picker, [['', 'Chọn sequence trên máy'], ...entries.map(e => [e.value, `${e.seq} · ${e.path}`])]);
  if (preferred) {
    const matches = entries.filter(e => e.seq === preferred.seq &&
      (!preferred.path || e.path === preferred.path || preferred.path.endsWith('/' + e.path)));
    picker.value = matches.length === 1 ? matches[0].value : '';
  } else if (!entries.some(e => e.value === prev) && entries.length === 1) {
    picker.value = entries[0].value;
  }
  picker.disabled = !entries.length;
  const cards = $('#seqCards');
  cards.innerHTML = entries.length ? entries.map(e => {
    const cams = e.info.cameras?.length || 0;
    const ok = cams === 4 && e.info.lidar && e.meta;
    return `<button type="button" role="option" class="seq-card" data-value="${esc(e.value)}" aria-selected="${picker.value === e.value}">
      <span class="seq-name">${esc(e.seq.replace('2013_05_28_drive_', 'drive_').replace('_sync', ''))}</span>
      <code>${esc(e.path)}/</code>
      <span class="tags"><span class="tag ${cams === 4 ? 'ok' : 'warn'}">${cams}/4 camera</span>
        <span class="tag ${e.info.lidar ? 'ok' : 'warn'}">${e.info.lidar ? 'LiDAR' : 'thiếu LiDAR'}</span>
        ${e.meta ? '' : '<span class="tag warn">thiếu file .pkl</span>'}
        ${ok ? '<span class="tag ok">đủ dữ liệu</span>' : ''}</span>
    </button>`;
  }).join('') : `<div class="empty-inline">Chưa có sequence KITTI-360 nào trên máy ảo.
      <div class="actions"><button type="button" class="primary small" data-src="src-upload">Tải lên từ máy tính</button>
      <button type="button" class="small" data-src="src-drive">Lấy từ Google Drive</button></div></div>`;
  for (const card of $$('.seq-card', cards)) card.onclick = () => { picker.value = card.dataset.value; picker.onchange(); };
  for (const btn of $$('[data-src]', cards)) btn.onclick = () => showTab(btn.dataset.src);
  picker.onchange = () => {
    for (const card of $$('.seq-card', cards)) card.setAttribute('aria-selected', String(card.dataset.value === picker.value));
    const entry = entries.find(e => e.value === picker.value);
    $('#prepareSection').hidden = !entry;
    $('#selectedBar').hidden = !entry;
    if (entry) {
      $('#selectedSeq').textContent = entry.seq;
      $('#selectedTags').innerHTML = `<span class="tag">${esc(entry.path)}/</span><span class="tag">${entry.info.cameras?.length || 0} camera</span>${entry.info.lidar ? '<span class="tag ok">LiDAR</span>' : ''}`;
    }
    for (const form of [f, $('#convertForm')]) {
      form.querySelector('button[type=submit]').disabled = !picker.value;
      if (!picker.value) continue;
      const [path, seq] = JSON.parse(picker.value);
      const ds = form.elements.dataset || form.elements.ds;
      ds.value = path;
      ds.dispatchEvent(new Event('change', {bubbles:true}));
      form.elements.sequence.value = seq;
      form.elements.sequence.dispatchEvent(new Event('change', {bubbles:true}));
    }
  };
  picker.onchange();
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

let runsRequest = 0;
async function loadRuns() {
  const request = ++runsRequest;
  const next = await api('/api/drivestudio');
  if (request !== runsRequest) return;
  overview = next;
  const o = overview.options;
  const t = $('#dsTrainForm');
  fill(t.elements.method, [['', 'Chọn phương pháp'], ...(o.methods || []).map((m) => [m, m.replace(/^configs\//, '')])]);
  fill(t.elements.dataset, [['', 'Chọn cấu hình dữ liệu'], ...(o.datasets || []).map((d) => [d, d])]);
  fill(t.elements.data_root, [['', 'theo dataset config'], ...(o.data_roots || []).map((d) => [d, d])]);
  $('#dsTrainHint').innerHTML = o.venv_ok ? `Môi trường <code>${esc(o.venv)}</code>` : `<span class="tag bad">thiếu môi trường DriveStudio+</span> <code>${esc(o.venv)}</code>`;
  $('#dsEnv').innerHTML = `
    <dt>DriveStudio+</dt><dd>${o.venv_ok ? '<span class="tag ok">có</span>' : '<span class="tag bad">chưa có</span>'} <code>${esc(o.venv)}</code></dd>
    <dt>Dữ liệu đã chuẩn bị</dt><dd>${(o.data_roots || []).map((d) => `<code>${esc(d)}</code>`).join(' ') || '<span class="muted">chưa có, xem bước 1</span>'}</dd>`;
  renderKpis();
  renderRuns();
  renderViewerRuns();
  renderVideoCompare();
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
  ['progress', 'Tiến độ'], ['psnr', 'PSNR', 'num', 1], ['nonsky', 'Non-Sky', 'num', 1], ['dyn', 'Dynamic PSNR', 'num', 1],
  ['veh', 'Vehicle PSNR', 'num', 1], ['ssim', 'SSIM', 'num', 1], ['lpips', 'LPIPS ↓', 'num', -1],
  ['drmse', 'Depth RMSE ↓', 'num', -1], ['absrel', 'AbsRel ↓', 'num', -1], ['delta1', 'δ<1.25', 'num', 1],
  ['fps', 'FPS', 'num', 1], ['vram', 'VRAM (GiB) ↓', 'num', -1], ['elapsed', 'Thời gian', 'num'],
];
const metric = (r, k) => r.metrics?.[k] ?? null;
const VALUE = {
  name: (r) => r.name, method: (r) => (r.method || '').replace(/^configs\//, '').replace(/\.yaml$/, ''),
  render: (r) => r.render_mode || (/_geer/.test(r.method || '') ? 'geer' : /_ut/.test(r.method || '') ? 'ut' : 'config'),
  fastgs: (r) => r.fastgs ?? (/fastgs/.test(r.method || '') ? 'true' : '—'), state: (r) => r.state,
  progress: (r) => (r.total ? r.step / r.total : 0), psnr: (r) => metric(r, 'Full Image PSNR'), nonsky: (r) => metric(r, 'Non-Sky PSNR'),
  dyn: (r) => metric(r, 'Dynamic-Only PSNR'), veh: (r) => metric(r, 'Vehicle-Only PSNR'),
  ssim: (r) => metric(r, 'Full Image SSIM'), lpips: (r) => metric(r, 'Full Image LPIPS'),
  drmse: (r) => metric(r, 'Depth RMSE'), absrel: (r) => metric(r, 'Depth AbsRel'), delta1: (r) => metric(r, 'Depth Delta1'),
  fps: (r) => metric(r, 'Render FPS'), vram: (r) => metric(r, 'Peak VRAM GiB'),
  elapsed: (r) => r.elapsed || '', updated: (r) => r.updated || 0,
};

const groups = { core: ['psnr','ssim','lpips','fps'], image: ['psnr','nonsky','dyn','veh','ssim','lpips'], geom: ['drmse','absrel','delta1'], cost: ['fps','vram','elapsed'] };
let group = 'core';
function renderRuns() {
  const table = $('#dsRuns');
  const cols = COLS.filter(([key], i) => i < 6 || group === 'all' || groups[group].includes(key));
  const head = cols.map(([key, label, kind]) => `<th scope="col" class="${kind || ''}" aria-sort="${sort.key === key ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none'}"><button type="button" data-sort="${key}">${label}</button></th>`).join('');
  const runs = [...overview.runs];
  if (!runs.length) {
    table.innerHTML = `<thead><tr>${head}</tr></thead><tbody><tr><td class="empty" colspan="${cols.length}"><b>Chưa có lần chạy nào</b>Chuẩn bị dữ liệu ở bước 1, rồi bắt đầu train ở bước 2.</td></tr></tbody>`;
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
      <td class="prog">${prog}</td>${cols.slice(6).map(([key]) => key === 'elapsed' ? `<td class="num mono">${esc(r.elapsed || '—')}</td>` : cell(key, ({ssim:3,lpips:3,absrel:3,delta1:3,fps:1})[key] ?? 2)).join('')}</tr>`;
  }).join('');
  const foot = runs.some((r) => r.state !== 'succeeded' && r.metrics)
    ? `<tfoot><tr><td colspan="${cols.length}" class="muted">* đánh giá tạm trong lúc train (vài ảnh), không so sánh được với kết quả test cuối.</td></tr></tfoot>` : '';
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
    ${run.metrics ? `<div class="mgrid">${Object.entries(run.metrics).map(([k, val]) => `<div class="mchip"><span>${esc(k)}</span><b>${k === 'Gaussians' ? `${(val / 1e6).toFixed(2)}M` : Number(val).toFixed(/SSIM|LPIPS|AbsRel|Delta1/.test(k) ? 3 : 2)}</b></div>`).join('')}</div>` : ''}
    ${step ? `<div class="compare">${files.get('gt_rgbs') ? img(files.get('gt_rgbs'), 'Ảnh gốc (GT)') : ''}${files.get(shown) ? img(files.get(shown), `${kindNames[shown]} · bước ${step[0]}`) : ''}</div>`
      : (run.images.length ? `<div class="gallery wide">${run.images.map((p) => img(p, p)).join('')}</div>`
        : '<p class="empty"><b>Chưa có ảnh</b>DriveStudio lưu ảnh render định kỳ trong lúc train.</p>')}
    ${run.videos.length ? `<h3 style="margin-top:16px">Video Output (${run.videos.length})</h3><div class="vcomp-arena side">${run.videos.map((v) => `<figure class="vcomp-card"><video controls preload="metadata" src="/api/drivestudio/runs/${enc(run.name)}/files/${enc(v)}"></video><figcaption><span>${esc(v.split('/').pop())}</span><a href="/api/drivestudio/runs/${enc(run.name)}/files/${enc(v)}" download>Tải</a></figcaption></figure>`).join('')}</div>` : ''}`;
}

// ------------------------------------------------------------------ 3D viewer (DriveStudio tools/viewer.py, served on its own port)
const viewerUrl = () => `${location.protocol}//${location.hostname}:${viewer.port}/`;

function renderViewerRuns() {
  const sel = $('#dsViewRun');
  const ready = overview.runs.filter((r) => r.checkpoint);
  fill(sel, ready.map((r) => [r.name, `${r.name}${r.metrics?.['Full Image PSNR'] != null && r.state === 'succeeded' ? ` · ${r.metrics['Full Image PSNR'].toFixed(2)} dB` : ''}`]));
  if (viewer.run && ready.some((r) => r.name === viewer.run)) sel.value = viewer.run;
  else if (selected && ready.some((r) => r.name === selected)) sel.value = selected;
  $('#dsViewStart').disabled = !ready.length;
}

function renderStage() {
  const body = $('#dsStageBody');
  body.hidden = !['loading', 'running', 'ready', 'failed'].includes(viewer.state);
  $('#dsStage').classList.toggle('collapsed', body.hidden);
  const st = viewer.state;
  $('#dsViewStop').hidden = !(st === 'ready' || st === 'loading');
  $('#dsViewOpen').hidden = st !== 'ready';
  if (st === 'ready') $('#dsViewOpen').href = viewerUrl();
  $('#dsViewState').innerHTML = {
    ready: `<span class="tag ok">đang mở</span> <code>${esc(viewer.run)}</code>`,
    loading: `<span class="tag run">đang nạp checkpoint…</span> <code>${esc(viewer.run)}</code>`,
    failed: '<span class="tag bad">viewer lỗi</span>',
    stopped: '',
  }[st] || '';
  const frame = body.querySelector('iframe');
  if (st === 'ready') {
    if (!frame || frame.dataset.run !== viewer.run) {
      body.innerHTML = `<iframe title="DriveStudio viewer: ${esc(viewer.run)}" src="${viewerUrl()}" data-run="${esc(viewer.run)}" allow="fullscreen"></iframe>`;
    }
    return;
  }
  if (st === 'loading') {
    body.innerHTML = `<div class="stage-empty"><span class="spinner" aria-hidden="true"></span><b>Đang khởi động viewer trên GPU của kit</b>Nạp checkpoint và dữ liệu mất khoảng 30–60 giây.<pre class="log">${esc(viewer.log.split('\n').slice(-8).join('\n'))}</pre></div>`;
  } else if (st === 'failed') {
    body.innerHTML = `<div class="stage-empty"><b>Viewer dừng vì lỗi</b><pre class="log">${esc(viewer.log.split('\n').slice(-14).join('\n'))}</pre></div>`;
  } else {
    body.innerHTML = `<div class="stage-empty"><b>Xem cảnh đã dựng trong không gian 3D</b>Chọn một lần chạy có checkpoint rồi bấm <b>Xem 3D</b>: viewer của DriveStudio chạy trên GPU của kit, bạn xoay, bay trong cảnh, đổi camera, chọn khung hình, tắt bật nền và vật thể.</div>`;
  }
}

async function loadViewer() {
  viewer = await api('/api/drivestudio/viewer');
  renderStage();
  clearTimeout(viewerTimer);
  if (viewer.state === 'loading') viewerTimer = setTimeout(() => loadViewer().catch(() => {}), 3000);
}

function initViewer() {
  $('#dsViewStart').addEventListener('click', async () => {
    const name = $('#dsViewRun').value;
    if (!name) return;
    const free = [...(setup?.gpus || [])].sort((a, b) => (a.memory_used_mib - b.memory_used_mib))[0];
    $('#dsViewStart').disabled = true;
    try {
      viewer = await api('/api/drivestudio/viewer', { name, gpu: free ? free.index : null });
      renderStage();
      loadViewer();
    } catch (err) { toast(err.message, true); } finally { $('#dsViewStart').disabled = false; }
  });
  $('#dsViewStop').addEventListener('click', async () => {
    try { viewer = await api('/api/drivestudio/viewer/stop', {}); renderStage(); } catch (err) { toast(err.message, true); }
  });
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
    window.__fflab.openLightbox(img.src, img.alt);
  });
  $('#dsRefresh').addEventListener('click', () => Promise.all([loadSetup(), loadRuns()]).catch((err) => toast(err.message, true)));
}


// ------------------------------------------------------------------ synchronized multi-method video comparison
async function renderVideoCompare() {
  const snapshot = overview;
  let vids;
  try { ({ videos: vids } = await api('/api/videos')); }
  catch (err) { if (overview === snapshot) toast(err.message, true); return; }
  if (overview !== snapshot) return;
  overview.videos = vids;
  const selA = $('#vcSelectA'), selB = $('#vcSelectB'), arena = $('#vcArena');
  if (!selA || !selB || !arena) return;
  $('#vcCountHint').textContent = `${vids.length} video`;
  $('#videoCompareBox').classList.toggle('is-empty', !vids.length);
  if (!vids.length) {
    selA.innerHTML = selB.innerHTML = '<option value="">Chưa có video (.mp4)</option>';
    selA.disabled = selB.disabled = true;
    delete arena.dataset.pair;
    $('#vcDownloadLink').hidden = true;
    $('#vcPlayBtn').disabled = $('#vcRestartBtn').disabled = $('#vcExportBtn').disabled = true;
    arena.innerHTML = '<div class="empty" style="grid-column:1/-1"><b>Chưa có video output để so sánh</b>Sau khi train DriveStudio+, render <code>fflab render</code> (3DGUT) hoặc đồng bộ runs từ Google Drive, video sẽ tự xuất hiện ở đây.</div>';
    return;
  }
  selA.disabled = selB.disabled = false;
  const opts = vids.map((v) => [v.id, v.label]);
  const prevA = selA.value, prevB = selB.value;
  fill(selA, opts, true);
  fill(selB, opts, true);
  if (!prevA || !vids.some((v) => v.id === selA.value)) selA.value = vids[0].id;
  if (!vids.some((v) => v.id === prevB)) {
    selB.value = (vids.find((v) => v.id !== selA.value) || vids[0]).id;
  }
  updateVideoArena();
}

function updateVideoArena() {
  const vids = overview.videos || [];
  const va = vids.find((v) => v.id === $('#vcSelectA')?.value);
  const vb = vids.find((v) => v.id === $('#vcSelectB')?.value);
  const arena = $('#vcArena');
  if (!arena || !va || !vb) return;
  const layout = $('#vcLayout').value, wipe = layout === 'wipe';
  const pair = JSON.stringify([va.id, va.url, vb.id, vb.url]);
  const changed = arena.dataset.pair !== pair;
  const btn = $('#vcExportBtn');
  btn.disabled = !!btn.dataset.busy || va.id === vb.id;
  btn.textContent = wipe ? 'Xuất MP4 cạnh nhau' : 'Xuất MP4 ghép nhãn';
  if (!changed && arena.dataset.layout === layout) return;
  arena.dataset.layout = layout;
  arena.dataset.pair = pair;
  arena.className = `vcomp-arena ${layout === 'hstack' ? 'side' : 'stack'}`;
  const card = (v, id) => `<figure class="vcomp-card">
    <video id="${id}" preload="metadata" muted playsinline src="${esc(v.url)}"></video>
    <figcaption><span><b>${esc(v.run)}</b> · ${esc(v.method)}${v.psnr != null ? ` · ${v.psnr.toFixed(2)} dB` : ''}</span>
      <a href="${esc(v.url)}" download>MP4</a></figcaption></figure>`;
  if (changed) arena.innerHTML = card(va, 'vcVidA') + card(vb, 'vcVidB') +
    '<label>Rèm A/B <input id="vcWipe" type="range" min="0" max="100" value="50"></label>';
  $('#vcDownloadLink').hidden = true;
  const elA = $('#vcVidA'), elB = $('#vcVidB');
  const slider = $('#vcWipe');
  slider.parentElement.hidden = !wipe;
  for (const v of [elA, elB]) {
    v.parentElement.style.display = wipe ? 'contents' : '';
    v.style.gridArea = wipe ? '1 / 1' : '';
    v.style.height = wipe ? '100%' : '';
    v.style.objectFit = 'contain';
  }
  slider.oninput = () => { elB.style.clipPath = wipe ? `inset(0 0 0 ${slider.value}%)` : ''; };
  slider.oninput();
  if (!changed) return;
  elA.playbackRate = elB.playbackRate = Number($('#vcRate').value);
  const tick = () => {
    if (!elA.isConnected) return;
    const end = Math.min(elA.duration, elB.duration);
    const ready = Number.isFinite(end) && end > 0;
    for (const id of ['vcPlayBtn', 'vcRestartBtn', 'vcPrevFrame', 'vcNextFrame', 'vcScrub']) {
      $('#' + id).disabled = !ready;
    }
    $('#vcScrub').value = ready ? 1000 * elA.currentTime / end : 0;
    $('#vcTime').textContent = `${elA.currentTime.toFixed(2)}s`;
    if (!ready || elA.seeking || elB.seeking) return;
    if ((elA.currentTime >= end || elB.ended) && (!elA.paused || !elB.paused)) {
      elA.pause(); elB.pause(); elA.currentTime = elB.currentTime = end;
    } else if (!elA.paused && Math.abs(elA.currentTime - elB.currentTime) > 0.12) {
      elB.currentTime = elA.currentTime;
    }
  };
  for (const v of [elA, elB]) {
    v.ondurationchange = tick;
    v.onended = () => { if (v.isConnected) { elA.pause(); elB.pause(); elA.currentTime = elB.currentTime = Math.min(elA.duration, elB.duration); } };
  }
  elA.ontimeupdate = tick;
  tick();
}

function initVideoCompare() {
  if (!$('#vcSelectA')) return;
  for (const id of ['#vcSelectA', '#vcSelectB', '#vcLayout']) {
    $(id).addEventListener('change', updateVideoArena);
  }
  $('#vcRate').addEventListener('change', () => {
    const r = Number($('#vcRate').value || 1);
    for (const v of [$('#vcVidA'), $('#vcVidB')]) if (v) v.playbackRate = r;
  });
  $('#vcPlayBtn').addEventListener('click', () => {
    const a = $('#vcVidA'), b = $('#vcVidB');
    if (!a || !b) return;
    if (a.paused) {
      if (a.currentTime >= Math.min(a.duration, b.duration)) a.currentTime = 0;
      b.currentTime = a.currentTime;
      Promise.all([a.play(), b.play()]).catch((err) => {
        if (!a.isConnected || err.name === 'AbortError') return;
        a.pause(); b.pause(); toast(err.message, true);
      });
    }
    else { a.pause(); b.pause(); }
  });
  const seek = (fraction, frames = 0) => {
    const a = $('#vcVidA'), b = $('#vcVidB');
    const end = Math.min(a?.duration, b?.duration);
    if (!Number.isFinite(end)) return;
    const fps = Number(overview.videos?.find(v => v.id === $('#vcSelectA').value)?.fps);
    // ponytail: approximate frame steps at 30 fps when the catalog omits media FPS.
    if (frames) { a.pause(); b.pause(); }
    const time = frames ? a.currentTime + frames / (fps > 0 ? fps : 30) : end * fraction;
    a.currentTime = b.currentTime = Math.max(0, Math.min(end, time));
  };
  $('#vcRestartBtn').onclick = () => seek(0);
  $('#vcScrub').oninput = e => seek(e.target.value / 1000);
  $('#vcPrevFrame').onclick = () => seek(0, -1);
  $('#vcNextFrame').onclick = () => seek(0, 1);
  $('#vcExportBtn').addEventListener('click', async () => {
    const video_a = $('#vcSelectA').value, video_b = $('#vcSelectB').value, layout = $('#vcLayout').value;
    if (!video_a || !video_b || video_a === video_b) return toast('Chọn 2 video khác nhau để ghép.', true);
    const btn = $('#vcExportBtn');
    const arena = $('#vcArena'), pair = arena.dataset.pair;
    btn.dataset.busy = '1';
    btn.disabled = true;
    try {
      const res = await api('/api/videos/compare', { video_a, video_b, layout: layout === 'vstack' ? 'vstack' : 'hstack' });
      if (arena.dataset.pair !== pair || arena.dataset.layout !== layout) return;
      const dl = $('#vcDownloadLink');
      dl.href = res.url;
      dl.hidden = false;
      toast('Đã xuất video so sánh MP4.');
    } catch (err) { toast(err.message, true); } finally {
      delete btn.dataset.busy;
      btn.disabled = $('#vcSelectA').value === $('#vcSelectB').value;
    }
  });
}

initTabs();
initNav();
$('#changeSeq').onclick = () => {
  showTab('src-local');
  ($('#seqCards .seq-card[aria-selected=true]') || $('#seqCards .seq-card'))?.focus();
};
$('#rescanData').onclick = async (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  try { await loadSetup(); toast('Đã quét lại dữ liệu trên máy ảo'); }
  catch (err) { toast(err.message, true); }
  finally { btn.disabled = false; }
};
window.__fflab.reloadData = loadSetup;
window.__fflab.showTab = showTab;
window.__fflab.selectImported = async (path, seq) => {
  await loadSetup({path: String(path || '').replace(/^\.\//, '').replace(/\/$/, ''), seq});
  showTab('src-local');
  if ($('#localSequence').value) {
    showTab('t-data-ds');
    $('#prepareSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
};
$$('[data-mgroup]').forEach(b => {
  b.setAttribute('aria-pressed', String(b.dataset.mgroup === group));
  b.onclick = () => {
    group = b.dataset.mgroup;
    $$('[data-mgroup]').forEach(p => p.setAttribute('aria-pressed', String(p.classList.toggle('active', p === b))));
    renderRuns();
  };
});
initForms();
initViewer();
initVideoCompare();
Promise.all([loadSetup(), loadRuns()]).then(() => { renderKpis(); return loadViewer(); }).catch((err) => toast(err.message, true));
