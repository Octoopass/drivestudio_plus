// Fisheye Fusion Lab: shared UI runtime + KITTI-360 / NCore / Instant NuRec / Google Drive sync.
import { progressBoard, fmtDur } from './progress.js?v=20261119n';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const url = (path) => String(path ?? '').split('/').map(encodeURIComponent).join('/');

const MODEL_SLOTS = {
  'pa-multiview': [['Trước', 'camera_front_left_00'], ['Chéo trái', 'camera_fisheye_left_02'], ['Chéo phải', 'camera_fisheye_right_03']],
  'pa-front': [['Trước', 'camera_front_left_00']],
  'pq-front': [],
};
const STATE_TEXT = { running: 'Đang chạy', cancelling: 'Đang huỷ', succeeded: 'Xong', failed: 'Lỗi', cancelled: 'Đã huỷ' };

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

let toastTimer = null;
function toast(msg, error = false) {
  const el = $('#toast');
  if (!el) return;
  el.textContent = String(msg ?? '');
  el.className = `toast${error ? ' error' : ''}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, error ? 8500 : 4000);
}

let activeLightbox = null;
function openLightbox(src, alt = '', trigger = document.activeElement) {
  if (activeLightbox) activeLightbox.close();
  const box = document.createElement('div');
  box.className = 'lightbox';
  box.tabIndex = -1;
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', alt || 'Xem ảnh phóng to');
  const img = document.createElement('img');
  img.src = src;
  img.alt = alt;
  box.appendChild(img);
  const close = () => {
    document.removeEventListener('keydown', onKey);
    box.remove();
    activeLightbox = null;
    if (trigger && typeof trigger.focus === 'function') trigger.focus();
  };
  const onKey = (k) => { if (k.key === 'Escape') { k.preventDefault(); close(); } };
  box.addEventListener('click', close);
  document.addEventListener('keydown', onKey);
  document.body.appendChild(box);
  box.focus();
  activeLightbox = { close };
}

document.addEventListener('click', (e) => {
  const img = e.target.closest('.gallery img, .compare img');
  if (img) openLightbox(img.src, img.alt || '', img);
});

let setup = null;
let setupInFlight = null;
const setupListeners = new Set();

async function fetchSetup(force = false) {
  if (!force && setupInFlight) return setupInFlight;
  setupInFlight = api('/api/setup').then((data) => {
    setup = data;
    renderPipelineSetup(data);
    for (const fn of setupListeners) fn(data);
    return data;
  }).finally(() => { setupInFlight = null; });
  return setupInFlight;
}

window.__fflab = { api, esc, url, toast, openLightbox, fetchSetup, onSetup: (fn) => setupListeners.add(fn), getSetup: () => setup, progressBoard, fmtDur };

function fillSelect(select, options, keep = true) {
  if (!select) return;
  const old = select.value;
  select.innerHTML = options.map(([v, label]) => `<option value="${esc(v)}">${esc(label)}</option>`).join('');
  if (keep && options.some(([v]) => v === old)) select.value = old;
  select.disabled = !options.length;
}

function gpuOptions() {
  const gpus = setup?.gpus || [];
  return gpus.map((g) => [String(g.index), `GPU ${g.index}: ${g.name} · ${(g.memory_used_mib / 1024).toFixed(1)}/${(g.memory_total_mib / 1024).toFixed(0)} GB · ${g.utilization}%`]);
}

function freestGpu() {
  const gpus = setup?.gpus || [];
  return gpus.length ? String(gpus.reduce((a, b) => (b.memory_used_mib < a.memory_used_mib ? b : a)).index) : '';
}

function gallery(files, src, wide = false) {
  if (!files.length) return '';
  return `<div class="gallery${wide ? ' wide' : ''}">${files.map((f) => `
    <figure><img loading="lazy" tabindex="0" src="${esc(src(f))}" alt="${esc(f)}"><figcaption>${esc(f)}</figcaption></figure>`).join('')}</div>`;
}

// Generation-guarded job watcher: cancels old pollers on the same box and always calls onFinish
const boxGen = new WeakMap();
function watchJob(box, statusUrl, cancelUrl, render, onFinish) {
  const gen = (boxGen.get(box) || 0) + 1;
  boxGen.set(box, gen);
  box.hidden = false;
  let timer = null;
  const tick = async () => {
    if (boxGen.get(box) !== gen) return;
    let s;
    try {
      s = await api(statusUrl);
    } catch (err) {
      if (boxGen.get(box) !== gen) return;
      box.innerHTML = `<div class="head"><span class="state failed">Lỗi kết nối</span></div><p class="state failed">${esc(err.message)}</p>`;
      onFinish?.({ state: 'failed', error: err.message });
      return;
    }
    if (boxGen.get(box) !== gen) return;
    const active = s.state === 'running' || s.state === 'cancelling';
    const safeState = ['running', 'cancelling', 'succeeded', 'failed', 'cancelled'].includes(s.state) ? s.state : 'failed';
    box.innerHTML = `
      <div class="head">
        ${active ? '<span class="spinner" aria-hidden="true"></span>' : ''}
        <span class="state ${safeState}">${esc(STATE_TEXT[s.state] || s.state)}${s.step ? ` · ${esc(s.step)}` : ''}</span>
        ${s.output ? `<code>${esc(s.output)}</code>` : ''}
        ${active && cancelUrl ? '<button type="button" class="small" data-cancel>Huỷ job</button>' : ''}
      </div>
      ${s.plan?.length ? progressBoard(s.plan, { now: s.now, state: s.state, started: s.started, ended: s.ended }) : ''}
      ${s.error ? `<p class="state failed">${esc(s.error)}</p>` : ''}
      ${render ? render(s) : ''}
      <details ${active || s.state === 'failed' ? 'open' : ''}><summary>Log chi tiết</summary><pre>${esc((s.log || '').slice(-12000))}</pre></details>`;
    const pre = $('pre', box);
    if (pre) pre.scrollTop = pre.scrollHeight;
    const cancel = $('[data-cancel]', box);
    if (cancel) cancel.onclick = async () => { cancel.disabled = true; await api(cancelUrl, {}).catch((e) => toast(e.message, true)); };
    if (active) timer = setTimeout(tick, 2000);
    else onFinish?.(s);
  };
  tick();
  return () => { if (boxGen.get(box) === gen) boxGen.set(box, gen + 1); clearTimeout(timer); };
}
window.__fflab.watchJob = watchJob;

function renderPipelineSetup(data) {
  const n = data.nurec || {};
  const ckpts = Object.keys(n.checkpoints || {});
  $('#status').innerHTML = [
    n.ok ? `<span class="good">NuRec (${ckpts.length})</span>` : '<button type="button" class="link bad" data-dep-goto="nurec">Chưa cài NuRec</button>',
    data.gsplat?.ok ? '<span class="good">gsplat</span>' : '<button type="button" class="link bad" data-dep-goto="gsplat-eval">chưa cài gsplat</button>',
    `${(data.gpus || []).length} GPU`,
  ].join(' · ');

  $('#env').innerHTML = `
    <dt>Instant NuRec</dt><dd>${n.ok ? `<span class="tag ok">sẵn sàng</span> <code>${esc(n.repo)}</code>` : '<span class="tag warn">tùy chọn</span> <code>backends/instant-nurec</code>'}</dd>
    <dt>Checkpoint</dt><dd>${ckpts.map((m) => `<span class="tag ok">${esc(m)}</span>`).join('') || '<span class="muted">chưa có</span>'}</dd>
    <dt>gsplat (PLY eval)</dt><dd>${data.gsplat?.ok ? '<span class="tag ok">có</span>' : '<span class="tag warn">chưa cài</span> <code>fflab setup</code>'}</dd>
    <dt>Colab CLI (Drive)</dt><dd>${data.colab?.ok ? `<span class="tag ok">sẵn sàng</span> <code>${esc(data.colab.bin)}</code>` : '<span class="tag bad">thiếu</span>'}</dd>`;

  $('#gpus').innerHTML = (data.gpus || []).map((g) => {
    const pct = g.memory_total_mib ? Math.round((100 * g.memory_used_mib) / g.memory_total_mib) : 0;
    return `<div class="item"><div class="row"><span class="name">GPU ${Number(g.index)} · ${esc(g.name)}</span>
      <span class="muted">${(g.memory_used_mib / 1024).toFixed(1)}/${(g.memory_total_mib / 1024).toFixed(0)} GB · ${Number(g.utilization)}%</span></div>
      <div class="bar"><i style="width:${pct}%"></i></div></div>`;
  }).join('') || '<div class="empty-inline">Không thấy GPU NVIDIA (chế độ CPU / Colab).</div>';

  $('#datasets').innerHTML = (data.datasets || []).map((d) => `
    <div class="item"><div class="row"><span class="name">${esc(d.path)}</span>
      ${d.calibration_pickles ? '<span class="tag ok">calib đủ</span>' : '<span class="tag bad">thiếu pkl</span>'}</div>
      ${Object.entries(d.sequences || {}).map(([seq, info]) => `<div><code>${esc(seq)}</code>
        ${(info.cameras || []).map((c) => `<span class="tag">${esc(c)}</span>`).join('')}
        ${info.lidar ? '<span class="tag ok">LiDAR</span>' : ''}</div>`).join('') || '<p class="muted">Chưa có sequence đủ file.</p>'}
    </div>`).join('') || `<div class="empty-inline">Chưa có <code>./kitti360</code> trên máy ảo. Vào <b>1. Dữ liệu</b> để tải lên từ máy tính hoặc lấy từ Google Drive.</div>`;

  fillSelect($('#convertForm [name=ds]'), (data.datasets || []).map((d) => [d.path, d.path]));
  updateSequences();
  const clips = (data.clips || []).map((c) => [c.path, `${c.sequence_id} · ${c.frames} frame · ${c.cameras.length} cam`]);
  fillSelect($('#nurecForm [name=clip]'), clips);
  fillSelect($('#evalForm [name=clip]'), clips);
  for (const sel of [$('#nurecForm [name=gpu]'), $('#evalForm [name=gpu]')]) {
    const had = sel.value;
    fillSelect(sel, gpuOptions());
    if (!had) sel.value = freestGpu();
  }
  updateSlots();
  renderNurecSetup(data);
  renderResults();
}

// ---------------------------------------------------------------- Instant NuRec setup checklist
const NUREC_MODELS = [['pa-multiview', '3 camera (khuyên dùng)'], ['pa-front', 'camera trước'], ['pq-front', 'camera trước, gọn']];
let nurecSetupWatching = false;
function renderNurecSetup(data) {
  const box = $('#nurecSetup');
  if (!box) return;
  const n = data.nurec || {};
  const have = n.available || n.checkpoints || {};
  const sizes = n.weights_bytes || {};
  const clips = data.clips || [];
  const envOk = n.source && n.venv;
  const missing = NUREC_MODELS.filter(([m]) => !have[m]);
  const ckptOk = Object.keys(have).length > 0;
  const ready = envOk && ckptOk && clips.length;
  const row = (ok, title, body) => `<li class="${ok ? 'ok' : 'todo'}"><span class="nr-ic" aria-hidden="true">${ok ? '✓' : '○'}</span><div><b>${title}</b>${body}</div></li>`;
  const gb = (b) => `${(b / 2 ** 30).toFixed(1)} GB`;
  const models = NUREC_MODELS.map(([m, label]) => have[m]
    ? `<span class="tag ok">${esc(m)}</span>`
    : `<label class="check"><input type="checkbox" name="nrModel" value="${esc(m)}" ${m === 'pa-multiview' && !ckptOk ? 'checked' : ''}> ${esc(m)} <small>${esc(label)} · ${gb(sizes[m] || 0)}</small></label>`).join(' ');
  box.dataset.ready = ready ? 'yes' : 'no';
  box.innerHTML = `
    <div class="nr-head"><b>${ready ? 'Instant NuRec sẵn sàng' : 'Cần chuẩn bị trước khi chạy Instant NuRec'}</b>
      <span class="hint">Mô hình feed-forward của NVIDIA: dựng cảnh 3D Gaussian từ clip NCore trong vài phút, không cần train.</span></div>
    <ol class="nr-list">
      ${row(envOk, 'Mã nguồn + môi trường Python', envOk ? ` <code>${esc(n.repo)}</code>` : ' <small>Tải NVIDIA/instant-nurec (commit cố định) và cài môi trường riêng bằng uv (~6 GB, khoảng 10 phút).</small>')}
      ${row(ckptOk, 'Checkpoint (Hugging Face nvidia/instant-nurec)', `<div class="nr-models">${models}</div>${ckptOk && missing.length ? '<small>Tick để tải thêm model khác (không bắt buộc).</small>' : ''}`)}
      ${row(clips.length > 0, 'Clip NCore từ KITTI-360', clips.length
        ? ` <small>${clips.length} clip trong <code>outputs/ncore/</code></small>`
        : ` <small>${envOk ? 'Chưa có clip.' : 'Làm sau khi cài xong môi trường.'}</small> <button type="button" class="small" id="nrMakeClip" ${envOk ? '' : 'disabled'}>Tạo clip NCore ở bước 1 →</button>`)}
    </ol>
    ${!envOk || missing.length ? `<div class="actions"><button type="button" class="${envOk && ckptOk ? 'small' : 'primary'}" id="nrInstall" ${nurecSetupWatching ? 'disabled' : ''}>${envOk && ckptOk ? 'Tải checkpoint đã chọn' : 'Cài Instant NuRec'}</button>
      <span class="hint" id="nrInstallHint"></span></div>
      <p class="hint">Bằng việc cài, bạn đồng ý với <a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license" target="_blank" rel="noopener">NVIDIA Open Model License</a> của checkpoint.</p>` : ''}`;
  const hint = () => {
    const picked = $$('[name=nrModel]:checked', box).map((x) => x.value);
    const bytes = picked.reduce((a, m) => a + (sizes[m] || 0), 0) + (envOk ? 0 : 6 * 2 ** 30);
    const secs = (envOk ? 0 : 645) + bytes / (25 * 2 ** 20);
    const h = $('#nrInstallHint');
    if (h) h.textContent = bytes ? `Tải khoảng ${gb(bytes)} · dự kiến ${fmtDur(secs)}` : 'Chưa chọn model nào để tải';
    const b = $('#nrInstall');
    if (b && !nurecSetupWatching) b.disabled = !bytes;
  };
  box.onchange = hint;
  hint();
  const install = $('#nrInstall');
  if (install) install.onclick = async () => {
    const models = $$('[name=nrModel]:checked', box).map((x) => x.value);
    if (envOk && !models.length) return toast('Chọn ít nhất một checkpoint', true);
    install.disabled = true;
    try {
      const job = await api('/api/nurec/setup', { models: models.length ? models : ['pa-multiview'] });
      try { localStorage.setItem('fflab.nurecSetupJob', job.job_id); } catch { /* private mode */ }
      watchNurecSetup(job.job_id);
    } catch (err) { install.disabled = false; toast(err.message, true); }
  };
  if (n.setup_job && !nurecSetupWatching) watchNurecSetup(n.setup_job);
  const make = $('#nrMakeClip');
  if (make) make.onclick = () => window.__fflab.goto?.('step-convert', 't-data-nurec');
  const modelSel = $('#nurecForm [name=model]');
  if (modelSel) {
    [...modelSel.options].forEach((o) => { o.disabled = !have[o.value]; o.textContent = o.textContent.replace(/ \(chưa tải\)$/, '') + (have[o.value] ? '' : ' (chưa tải)'); });
    if (!have[modelSel.value] && ckptOk) { modelSel.value = Object.keys(have)[0]; updateSlots(); }
  }
  const run = $('#nurecForm button[type=submit]');
  if (run) {
    run.disabled = !ready;
    run.title = ready ? '' : 'Hoàn tất các mục ở khung trên trước';
  }
}
function watchNurecSetup(id) {
  nurecSetupWatching = true;
  watchJob($('#nurecSetupJob'), `/api/pipeline/${id}`, `/api/pipeline/${id}/cancel`, null, async (s) => {
    nurecSetupWatching = false;
    if (s.state === 'succeeded') toast('Đã cài xong Instant NuRec');
    await fetchSetup(true).catch(() => {});
  });
}
(async () => {
  let id = '';
  try { id = localStorage.getItem('fflab.nurecSetupJob') || ''; } catch { /* private mode */ }
  if (!/^[0-9a-f]{32}$/.test(id)) return;
  try {
    const s = await api(`/api/pipeline/${id}`);
    if (s.state === 'running') watchNurecSetup(id);
  } catch { /* job gone after a server restart */ }
})();

function updateSequences() {
  const form = $('#convertForm');
  if (!form || !setup) return;
  const ds = (setup.datasets || []).find((d) => d.path === form.ds.value);
  const seqs = ds ? Object.keys(ds.sequences) : [];
  fillSelect(form.sequence, seqs.map((s) => [s, s]));
  const cams = ds?.sequences[form.sequence.value]?.cameras || [];
  const names = setup.cameras || {};
  $('#convertCams').innerHTML = Object.keys(names).map((c) => `
    <label><input type="checkbox" name="cam" value="${esc(c)}" ${cams.includes(c) ? 'checked' : 'disabled'}>
      ${esc(c)} <span class="muted">${esc(names[c])}</span></label>`).join('');
}

$('#convertForm').addEventListener('change', (e) => {
  if (e.target.name === 'ds' || e.target.name === 'sequence') updateSequences();
});

$('#convertForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = {
    dataset: f.ds.value, sequence: f.sequence.value, start: Number(f.start.value), frames: Number(f.frames.value),
    cameras: [...f.querySelectorAll('[name=cam]:checked')].map((x) => x.value),
    lidar: f.lidar.checked, name: f.clipname.value.trim(), fisheye_model: f.fisheye_model.value,
  };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/pipeline/convert', body);
    watchJob($('#convertJob'), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`,
      (s) => gallery((s.artifacts || []).filter((a) => a.endsWith('.png')), (a) => `/api/pipeline/${url(s.job_id)}/artifacts/${url(a)}`),
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await fetchSetup(true);
          $('#nurecForm [name=clip]').value = s.info.clip;
          $('#evalForm [name=clip]').value = s.info.clip;
          updateSlots();
          toast('Đã tạo clip NCore.');
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

function updateSlots() {
  const form = $('#nurecForm');
  if (!form) return;
  const clip = (setup?.clips || []).find((c) => c.path === form.clip.value);
  const cams = clip?.cameras || [];
  const slots = MODEL_SLOTS[form.model.value] || [];
  $('#camHint').textContent = slots.length ? `${slots.length} camera theo thứ tự` : 'camera cố định';
  $('#nurecCams').innerHTML = slots.map(([label, preferred], i) => `
    <label>${esc(label)}<select name="slot${i}">${cams.map((c) => `<option value="${esc(c)}" ${c === preferred ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></label>`).join('')
    || '<p class="muted">pq-front chỉ chạy với camera <code>camera_front_wide_120fov</code>.</p>';
}

$('#nurecForm').addEventListener('change', (e) => {
  if (e.target.name === 'clip' || e.target.name === 'model') updateSlots();
});

$('#nurecForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const model = f.model.value;
  const checkpoint = setup?.nurec?.checkpoints?.[model];
  if (!checkpoint) return toast(`Chưa có checkpoint cho ${model}`, true);
  const repo = setup?.nurec?.repo || 'backends/instant-nurec';
  const python = setup?.nurec?.python || `${repo}/.venv/bin/python`;
  const camera_ids = (MODEL_SLOTS[model] || []).map((_, i) => f[`slot${i}`].value);
  if (new Set(camera_ids).size !== camera_ids.length) return toast('Mỗi vị trí phải là một camera khác nhau', true);
  const body = {
    repo, python, checkpoint, ncore_path: f.clip.value, model, camera_ids,
    gpu: f.gpu.value === '' ? null : Number(f.gpu.value),
    n_gaussians: Number(f.n_gaussians.value), max_chunks: Number(f.max_chunks.value),
  };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/reconstruction', body);
    const clip = f.clip.value;
    watchJob($('#nurecJob'), `/api/reconstruction/${job.job_id}`, `/api/reconstruction/${job.job_id}/cancel`,
      (s) => {
        const ply = (s.artifacts || []).filter((a) => a.endsWith('.ply'));
        return ply.map((p) => `<p>PLY: <a href="/api/reconstruction/${url(s.job_id)}/artifacts/${url(p)}" download>${esc(p)}</a></p>`).join('')
          + gallery((s.artifacts || []).filter((a) => a.endsWith('.png')), (a) => `/api/reconstruction/${url(s.job_id)}/artifacts/${url(a)}`);
      },
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await fetchSetup(true);
          const ply = (setup.reconstructions || []).find((r) => r.job_id === s.job_id);
          if (ply) $('#evalForm [name=ply]').value = ply.path;
          $('#evalForm [name=clip]').value = clip;
          toast('NuRec xong. Chuyển sang bước 3 để chấm điểm.');
          window.__fflab.goto?.('step-results', 't-res-nurec');
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

function metricsTable(m) {
  const rows = Object.entries(m || {}).filter(([k]) => !k.startsWith('_')).sort(([a], [b]) => a.localeCompare(b));
  const mean = rows.reduce((s, [, v]) => s + (Number(v.psnr_mean) || 0), 0) / (rows.length || 1);
  const sv = rows.map(([, v]) => v.ssim_mean).filter((x) => x != null && !Number.isNaN(Number(x)));
  const sMean = sv.length ? (sv.reduce((a, b) => a + Number(b), 0) / sv.length).toFixed(3) : '—';
  const sum = m?._summary || {};
  const chips = (sum.gaussians || sum.render_fps) ? `<div class="mgrid">
    ${sum.gaussians ? `<div class="mchip"><span>Gaussians</span><b>${(Number(sum.gaussians) / 1e6).toFixed(2)}M</b></div>` : ''}
    ${sum.render_fps ? `<div class="mchip"><span>Render FPS</span><b>${Number(sum.render_fps).toFixed(2)}</b></div>` : ''}
    ${sum.peak_vram_gib ? `<div class="mchip"><span>Peak VRAM</span><b>${Number(sum.peak_vram_gib).toFixed(2)} GiB</b></div>` : ''}
  </div>` : '';
  return `${chips}<div class="table-wrap"><table class="metrics"><tr><th>Camera</th><th class="num">PSNR (dB)</th><th class="num">SSIM</th><th class="num">Số view</th></tr>
    ${rows.map(([c, v]) => `<tr><td>${esc(c)}</td><td class="num">${(Number(v.psnr_mean) || 0).toFixed(2)}</td><td class="num">${v.ssim_mean != null ? Number(v.ssim_mean).toFixed(3) : '—'}</td><td class="num">${Number(v.views) || 0}</td></tr>`).join('')}
    <tr><th>Trung bình</th><th class="num">${mean.toFixed(2)}</th><th class="num">${sMean}</th><th></th></tr></table></div>`;
}

function renderResults() {
  const recons = setup?.reconstructions || [];
  fillSelect($('#evalForm [name=ply]'), recons.map((r) => [r.path, `${r.path.split('/').at(-1)} · ${r.size_mib} MB · job ${String(r.job_id).slice(0, 8)}`]));
  $('#recons').innerHTML = recons.map((r) => `
    <div class="item"><div class="row"><span class="name">${esc(r.path.split('/').at(-1))}</span><span class="muted">${Number(r.size_mib)} MB</span></div>
      <code>${esc(r.path)}</code><br>
      <a href="/api/outputs/${url(r.path.replace(/^outputs\//, ''))}" download>Tải PLY</a></div>`).join('')
    || '<div class="empty-inline">Chưa có file PLY từ NuRec.</div>';

  $('#evals').innerHTML = (setup?.evaluations || []).map((ev, i) => {
    const base = String(ev.path || '').replace(/^outputs\//, '');
    return `<details class="eval" ${i === 0 ? 'open' : ''}>
      <summary>${esc(ev.ply ? ev.ply.split('/').at(-1) : ev.path)} <span class="muted">vs ${esc(ev.clip ? ev.clip.split('/').at(-1) : '?')}</span></summary>
      ${metricsTable(ev.metrics)}
      ${(ev.videos || []).map((v) => `<video controls preload="metadata" style="width:100%;max-height:360px;margin:8px 0;border-radius:6px;background:#000" src="/api/outputs/${url(`${base}/${v}`)}"></video>`).join('')}
      ${gallery(ev.images || [], (f) => `/api/outputs/${url(`${base}/${f}`)}`, true)}
    </details>`;
  }).join('');
}

$('#evalForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = e.target;
  const body = { ply: f.ply.value, clip: f.clip.value, every: Number(f.every.value), gpu: f.gpu.value === '' ? null : Number(f.gpu.value) };
  const btn = $('button[type=submit]', f);
  btn.disabled = true;
  try {
    const job = await api('/api/pipeline/evaluate', body);
    watchJob($('#evalJob'), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`,
      (s) => (s.metrics ? metricsTable(s.metrics) : ''),
      async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') {
          await fetchSetup(true);
          $('#evalJob').hidden = true;
          document.getElementById('dsRefresh')?.click();
          toast('Đánh giá xong.');
        }
      });
  } catch (err) {
    btn.disabled = false;
    toast(err.message, true);
  }
});

fetchSetup().catch((err) => {
  $('#status').innerHTML = `<span class="bad">${esc(err.message)}</span>`;
});
