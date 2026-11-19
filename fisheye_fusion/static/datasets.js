// Dataset registry in the UI: profile pickers, raw datasets (+ DriveStudio preprocess form), preprocessed scenes.
// Talks to /api/datasets, /api/datasets/preprocess (fisheye_fusion/dataset_profiles.py, pipeline.py).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const { api, esc, toast } = window.__fflab;

let state = { profiles: [], local: { raw: [], processed: [] }, split_files: [] };
const byId = (id) => state.profiles.find((p) => p.id === id);

async function load() {
  state = await api('/api/datasets');
  fillPickers();
  render();
  return state;
}
const ready = load().catch((err) => { toast(err.message, true); return state; });
window.__fflab.datasets = { ready, load, byId, get: () => state };

function fillPickers() {
  const opts = state.profiles.map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');
  const up = $('#uploadDataset');
  if (up && up.options.length < 2) up.insertAdjacentHTML('beforeend', opts);
  const dr = $('#driveDataset');
  if (dr && !dr.options.length) dr.innerHTML = `<option value="">Chưa nhận ra, hãy chọn…</option>${opts}`;
}

function render() {
  const { raw, processed } = state.local;
  $('#rawSection').hidden = !raw.length;
  $('#processedSection').hidden = !processed.length;
  $('#rawCards').innerHTML = raw.map((r) => {
    const prof = byId(r.dataset);
    return `<div class="raw-card" data-ds="${esc(r.dataset)}">
      <div class="row-between"><b>${esc(r.label)}</b>
        <span class="tag ${r.detected ? 'ok' : 'warn'}">${r.detected ? 'đúng cấu trúc' : 'chưa nhận ra cấu trúc'}</span></div>
      <code>${esc(r.path)}/</code>
      <small>${r.entries} mục: ${esc(r.preview.join(', '))}${r.entries > r.preview.length ? '…' : ''}${r.partial ? ` · <span class="bad">${r.partial} file tải dở</span>` : ''}</small>
      ${r.detected ? '' : `<small>Cần có: ${esc(prof?.hint || '')}</small>`}
      <div class="actions"><button type="button" class="primary small" data-pre="${esc(r.dataset)}">Tiền xử lý…</button>
        <small>Hướng dẫn: <code>${esc(prof?.doc || '')}</code></small></div>
      <div class="pre-form" hidden></div>
    </div>`;
  }).join('');
  $$('[data-pre]').forEach((b) => b.onclick = () => openPreprocess(b.closest('.raw-card')));
  $('#processedCards').innerHTML = processed.map((p) => `
    <div class="seq-card static">
      <span class="seq-name">${esc(p.label)} · scene ${esc(p.scene)}</span>
      <code>${esc(p.data_root)}/</code>
      <span class="tags"><span class="tag ${p.sky_masks ? 'ok' : 'warn'}">${p.sky_masks ? 'sky mask' : 'thiếu sky mask'}</span>
        <span class="tag ${p.objects ? 'ok' : ''}">${p.objects ? 'có object' : 'không object'}</span></span>
      <button type="button" class="small" data-train="${esc(p.data_root)}" data-ds="${esc(p.dataset)}">Dùng để train →</button>
    </div>`).join('');
  $$('[data-train]').forEach((b) => b.onclick = () => useForTraining(b.dataset.train, b.dataset.ds));
  updateCount();
}

function updateCount() {
  const setup = window.__fflab.getSetup?.();
  const k360 = (setup?.datasets || []).reduce((n, d) => n + Object.keys(d.sequences || {}).length, 0);
  const parts = [];
  if (k360) parts.push(`${k360} KITTI-360`);
  if (state.local.raw.length) parts.push(`${state.local.raw.length} bộ gốc`);
  if (state.local.processed.length) parts.push(`${state.local.processed.length} scene sẵn sàng`);
  const el = $('#srcLocalCount');
  if (el && setup) el.textContent = parts.join(' · ') || 'Chưa có dữ liệu';
}
window.__fflab.onSetup?.(() => updateCount());

function openPreprocess(card) {
  const ds = card.dataset.ds;
  const prof = byId(ds);
  const pre = prof?.preprocess || {};
  const box = $('.pre-form', card);
  if (!box.hidden) { box.hidden = true; return; }
  const files = state.split_files.filter((f) => f.includes(ds));
  box.innerHTML = `
    <form class="form compact-grid">
      <label>Split <input name="split" value="${esc(pre.split || '')}" placeholder="theo mặc định" pattern="[A-Za-z0-9_.\\-]{0,40}"></label>
      <label>Chọn scene
        <select name="mode">
          <option value="ids"${pre.scene_ids ? ' selected' : ''}>Theo số thứ tự scene</option>
          <option value="file"${pre.split_file ? ' selected' : ''}${files.length ? '' : ' disabled'}>Theo danh sách có sẵn</option>
          <option value="range"${pre.num_scenes ? ' selected' : ''}>Một dải scene</option>
        </select></label>
      <label data-for="ids">Số scene (cách nhau bởi dấu cách) <input name="scene_ids" value="${esc(pre.scene_ids || '0')}" pattern="[0-9 ,]{1,400}"></label>
      <label data-for="file">Danh sách scene <select name="split_file">${files.map((f) => `<option${f === pre.split_file ? ' selected' : ''}>${esc(f)}</option>`).join('')}</select></label>
      <label data-for="range">Từ scene <input name="start_idx" type="number" min="0" value="${pre.start_idx || 0}"></label>
      <label data-for="range">Số lượng <input name="num_scenes" type="number" min="1" max="2000" value="${pre.num_scenes || 1}"></label>
      <label>Số luồng CPU <input name="workers" type="number" min="1" max="64" value="8"></label>
      <label class="inline"><input name="sky_masks" type="checkbox" checked> Tạo sky mask (SegFormer, cần GPU)</label>
      <p class="hint full">Kết quả: <code>${esc(pre.target || '')}/NNN</code>. Chạy <code>datasets/preprocess.py</code> của DriveStudio.</p>
      <div class="actions"><button type="submit" class="primary small">Bắt đầu tiền xử lý</button></div>
    </form>
    <div class="job" hidden></div>`;
  box.hidden = false;
  const form = $('form', box);
  const sync = () => $$('[data-for]', form).forEach((l) => { l.hidden = l.dataset.for !== form.elements.mode.value; });
  form.elements.mode.onchange = sync;
  sync();
  form.onsubmit = async (e) => {
    e.preventDefault();
    const f = form.elements, btn = $('button[type=submit]', form);
    const mode = f.mode.value;
    btn.disabled = true;
    try {
      const job = await api('/api/datasets/preprocess', {
        dataset: ds, split: f.split.value.trim(), workers: Number(f.workers.value), sky_masks: f.sky_masks.checked,
        scene_ids: mode === 'ids' ? f.scene_ids.value.trim() : '',
        split_file: mode === 'file' ? f.split_file.value : '',
        start_idx: mode === 'range' ? Number(f.start_idx.value) : 0,
        num_scenes: mode === 'range' ? Number(f.num_scenes.value) : 0,
      });
      window.__fflab.watchJob($('.job', box), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`, null, async (s) => {
        btn.disabled = false;
        if (s.state === 'succeeded') { toast(`Tiền xử lý xong: ${s.info?.data_root || ''}`); await load(); }
      });
    } catch (err) { btn.disabled = false; toast(err.message, true); }
  };
}

function useForTraining(dataRoot, ds) {
  const form = $('#dsTrainForm');
  window.__fflab.goto?.('step-train', 't-train-ds');
  if (!form) return;
  const root = form.elements.data_root;
  if (![...root.options].some((o) => o.value === dataRoot)) root.add(new Option(dataRoot, dataRoot));
  root.value = dataRoot;
  const cfg = form.elements.dataset;
  const match = [...cfg.options].find((o) => o.value.startsWith(`${ds}/`));
  if (match && !cfg.value.startsWith(`${ds}/`)) cfg.value = match.value;
  form.dispatchEvent(new Event('change'));
  toast(`Đã chọn ${dataRoot} cho bước huấn luyện`);
}

$('#rescanData')?.addEventListener('click', () => load().catch((err) => toast(err.message, true)));
