// Step 1 "Tải lên từ máy tính": resumable chunked upload of KITTI-360 files into the VM workspace.
// Talks to /api/upload, /api/upload/plan, /api/upload/chunk, /api/upload/cancel (fisheye_fusion/uploads.py).

const $ = (sel, root = document) => root.querySelector(sel);
const { api, esc, toast } = window.__fflab;
const CHUNK = 8 * 2 ** 20;
const LABELS = {
  'data_poses.zip': 'Poses (data_poses.zip)',
  'calibration.zip': 'Calibration (calibration.zip)',
  'ImageSets-rosa-6class/kitti360_rosa_infos_train.pkl': 'Metadata train (.pkl)',
  'ImageSets-rosa-6class/kitti360_rosa_infos_val.pkl': 'Metadata val (.pkl)',
};

const queue = [];          // {file, target, size, offset, state: wait|up|done|err|skip, error}
let running = false, paused = false, server = null;
let speed = { t: 0, bytes: 0, rate: 0 };

const fmt = (n) => n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(1)} MB` : `${Math.max(0, Math.round(n / 1024))} KB`;
const label = (target) => LABELS[target] || target.split('/').pop();

const MAX_FILES = 20000;
const relOf = (f) => f.__rel || f.name;

async function refreshServer() {
  server = await api('/api/upload');
  updateTarget();
  renderChecklist();
}

function updateTarget() {
  const ds = $('#uploadDataset').value;
  const prof = window.__fflab.datasets?.byId(ds);
  const dest = ds === 'kitti360' ? server?.absolute : prof ? `${server?.workspace}/${prof.raw}` : `${server?.workspace || '~'}/data/<dataset>/raw`;
  $('#uploadFolder').textContent = ds === 'kitti360' ? 'kitti360/' : prof ? `${prof.raw}/` : 'data/<dataset>/raw/';
  $('#uploadDatasetHint').textContent = prof ? `Cần có: ${prof.hint}` : 'Hệ thống tự nhận ra loại dataset theo tên file và thư mục.';
  if (server) $('#scpCmd').textContent = `scp -r "D:\\dataset\\*" ${server.user || 'research03'}@${location.hostname}:${dest}/`;
  $('#uploadChecklist').hidden = !(ds === 'auto' || ds === 'kitti360');
}

function renderChecklist() {
  const box = $('#uploadChecklist');
  const seqs = Object.entries(server?.sequences || {});
  if (!seqs.length) {
    box.innerHTML = `<p class="hint">Cần cho mỗi sequence: <code>data_poses.zip</code>, <code>calibration.zip</code>, 2 file
      <code>kitti360_rosa_infos_train/val.pkl</code>, 4 file <code>&lt;sequence&gt;_image_00…03.tar</code> và <code>&lt;sequence&gt;.tar</code> (LiDAR).</p>`;
    return;
  }
  box.innerHTML = seqs.map(([seq, need]) => {
    const have = need.filter((f) => server.files[f]).length;
    return `<div class="check-seq"><div class="row-between"><b>${esc(seq)}</b>
      <span class="tag ${have === need.length ? 'ok' : 'warn'}">${have}/${need.length} file</span></div>
      <ul>${need.map((f) => {
        const ok = server.files[f], part = server.partial[f];
        return `<li class="${ok ? 'ok' : part ? 'part' : 'miss'}"><span aria-hidden="true">${ok ? '✓' : part ? '◐' : '○'}</span>
          ${esc(label(f))} <small>${ok ? fmt(ok) : part ? `đã tải ${fmt(part)}, chọn lại file để tải tiếp` : 'chưa có'}</small></li>`;
      }).join('')}</ul></div>`;
  }).join('');
}

function render() {
  const list = $('#uploadList');
  $('#uploadQueue').hidden = !queue.length;
  if (!queue.length) return;
  const total = queue.filter((q) => q.state !== 'skip').reduce((a, q) => a + q.size, 0);
  const sent = queue.filter((q) => q.state !== 'skip').reduce((a, q) => a + q.offset, 0);
  const done = queue.filter((q) => q.state === 'done').length;
  const active = queue.filter((q) => q.state !== 'skip').length;
  $('#uploadSummary').textContent = `${done}/${active} file · ${fmt(sent)} / ${fmt(total)}`;
  $('#uploadSpeed').textContent = running && !paused && speed.rate
    ? `· ${fmt(speed.rate)}/s · còn ~${eta((total - sent) / speed.rate)}` : paused ? '· đã tạm dừng' : '';
  $('#uploadTotalBar').style.width = `${total ? (100 * sent / total).toFixed(1) : 0}%`;
  $('#uploadPause').hidden = !running;
  $('#uploadPause').textContent = paused ? 'Tiếp tục' : 'Tạm dừng';
  $('#uploadClear').disabled = running && !paused;
  list.innerHTML = queue.map((q, i) => {
    const pct = q.size ? (100 * q.offset / q.size).toFixed(1) : 0;
    const state = { wait: 'Chờ', up: `${pct}%`, done: 'Xong', err: 'Lỗi', skip: 'Bỏ qua' }[q.state];
    return `<li class="up-item ${q.state}">
      <div class="row-between"><span class="up-name">${esc(relOf(q.file))}</span>
        <span class="up-state">${esc(state)} ${q.state === 'up' || q.state === 'wait' ? `<button type="button" class="link" data-cancel="${i}">Huỷ</button>` : ''}</span></div>
      ${q.state === 'skip' ? `<small>${esc(q.error || 'Không thuộc bộ dữ liệu này')}</small>` : `
      <small>→ <code>${esc(q.target)}</code> · ${fmt(q.size)}${q.error ? ` · <span class="bad">${esc(q.error)}</span>` : ''}</small>
      <div class="bar ${q.state === 'done' ? '' : 'run'}"><i style="width:${pct}%"></i></div>`}
    </li>`;
  }).join('');
}

function eta(sec) {
  if (!isFinite(sec) || sec <= 0) return '—';
  if (sec < 90) return `${Math.round(sec)} giây`;
  if (sec < 5400) return `${Math.round(sec / 60)} phút`;
  return `${(sec / 3600).toFixed(1)} giờ`;
}

async function addFiles(files) {
  files = [...files].filter((f) => f.size > 0);
  if (!files.length) return;
  if (files.length > MAX_FILES) {
    toast(`Thư mục có ${files.length} file, quá nhiều để tải qua trình duyệt. Hãy nén thành .zip/.tar, hoặc dùng lệnh scp ở mục bên dưới.`, true);
    return;
  }
  if (!server) await refreshServer().catch(() => {});
  let plan;
  try {
    plan = await api('/api/upload/plan', {
      dataset: $('#uploadDataset').value,
      files: files.map((f) => ({ name: f.name, path: relOf(f), size: f.size })),
    });
  } catch (err) { toast(err.message, true); return; }
  const byPath = new Map(plan.files.map((p) => [p.path, p]));
  if (plan.dataset && $('#uploadDataset').value === 'auto') toast(`Nhận diện: ${plan.label}`);
  let added = 0;
  for (const file of files) {
    const p = byPath.get(relOf(file));
    if (!p) {
      if (files.length <= 40) queue.push({ file, target: '', size: file.size, offset: 0, state: 'skip' });
      continue;
    }
    byPath.delete(relOf(file));
    const existing = queue.find((q) => q.target === p.target && q.state !== 'skip');
    if (existing && existing.state !== 'err') continue;
    if (existing) queue.splice(queue.indexOf(existing), 1);
    queue.push({ file, target: p.target, size: p.size, offset: p.offset, state: p.done ? 'done' : 'wait',
                 error: p.done ? '' : p.offset ? `tải tiếp từ ${fmt(p.offset)}` : '' });
    added++;
  }
  const skipped = plan.skipped.length;
  if (!added) toast(skipped ? `Không có file ${plan.label} hợp lệ nào trong ${skipped} file đã chọn` : 'Các file này đã có trên máy ảo', !!skipped);
  else if (skipped) toast(`Đã thêm ${added} file; bỏ qua ${skipped} file không liên quan`);
  render();
  run();
}

async function sendChunk(q) {
  const end = Math.min(q.offset + CHUNK, q.size);
  const params = new URLSearchParams({ target: q.target, offset: String(q.offset), total: String(q.size) });
  const res = await fetch(`/api/upload/chunk?${params}`, {
    method: 'PUT', headers: { 'Content-Type': 'application/octet-stream' }, body: q.file.slice(q.offset, end),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(typeof data.detail === 'string' ? data.detail : res.statusText);
    err.fatal = res.status >= 400 && res.status < 500;
    throw err;
  }
  const delta = data.offset - q.offset;
  q.offset = data.offset;
  if (data.done) q.state = 'done';
  return Math.max(0, delta);
}

async function run() {
  if (running) return;
  running = true;
  speed = { t: performance.now(), bytes: 0, rate: 0 };
  const finished = new Set();
  try {
    for (;;) {
      if (paused) break;
      const q = queue.find((x) => x.state === 'wait' || x.state === 'up');
      if (!q) break;
      q.state = 'up';
      q.error = '';
      render();
      let retries = 0;
      while (q.state === 'up' && !paused) {
        try {
          speed.bytes += await sendChunk(q);
          retries = 0;
          const dt = (performance.now() - speed.t) / 1000;
          if (dt > 1.5) { speed.rate = speed.rate ? 0.6 * speed.rate + 0.4 * speed.bytes / dt : speed.bytes / dt; speed = { ...speed, t: performance.now(), bytes: 0 }; }
        } catch (err) {
          if (err.fatal || ++retries > 8) { q.state = 'err'; q.error = err.message; break; }
          q.error = `mất kết nối, thử lại (${retries}/8)…`;
          render();
          await new Promise((r) => setTimeout(r, Math.min(30000, 1000 * 2 ** retries)));
        }
        render();
      }
      if (q.state === 'done') finished.add(q.target);
    }
  } finally {
    running = false;
    render();
  }
  if (finished.size) await afterUpload([...finished]);
}

async function afterUpload(targets) {
  await refreshServer().catch(() => {});
  if (!targets.some((t) => t.startsWith('kitti360/'))) {
    if (queue.some((q) => q.state === 'wait' || q.state === 'up') || paused) return;
    toast('Đã tải xong lên máy ảo. Bước tiếp theo: bấm "Tiền xử lý…" ở thẻ Có sẵn trên máy ảo.');
    await window.__fflab.datasets?.load();
    window.__fflab.showTab?.('src-local');
    return;
  }
  const seq = targets.map((t) => t.match(/2013_05_28_drive_\d{4}_sync/)?.[0]).find(Boolean)
    || Object.keys(server?.sequences || {})[0];
  const pending = queue.some((q) => q.state === 'wait' || q.state === 'up');
  if (pending || paused) return;
  const failed = queue.filter((q) => q.state === 'err').length;
  toast(failed ? `Tải xong, ${failed} file lỗi; chọn lại các file đó để thử tiếp` : 'Đã tải xong lên máy ảo', !!failed);
  if (seq && window.__fflab.selectImported) {
    await window.__fflab.selectImported(server?.folder || 'kitti360', seq);
  } else {
    await window.__fflab.reloadData?.();
  }
}

// ---- drop zone and pickers
async function entriesToFiles(items) {
  const files = [];
  const walk = async (entry) => {
    if (!entry) return;
    if (entry.isFile) {
      const file = await new Promise((ok, bad) => entry.file(ok, bad));
      file.__rel = entry.fullPath.replace(/^\/+/, '');
      files.push(file);
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (;;) {
        const batch = await new Promise((ok, bad) => reader.readEntries(ok, bad));
        if (!batch.length) break;
        for (const child of batch) await walk(child);
      }
    }
  };
  for (const entry of items.map((i) => i.webkitGetAsEntry?.()).filter(Boolean)) await walk(entry);
  return files;
}

const zone = $('#dropzone');
if (zone) {
  ['dragenter', 'dragover'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.classList.remove('over'); }));
  zone.addEventListener('drop', async (e) => {
    // read both lists before any await: the DataTransfer is emptied once the handler yields
    const plain = [...(e.dataTransfer?.files || [])];
    const items = [...(e.dataTransfer?.items || [])];
    const walked = items.some((i) => i.webkitGetAsEntry?.()) ? await entriesToFiles(items) : [];
    addFiles(walked.length ? walked : plain);
  });
  $('#pickFiles').onclick = () => $('#fileInput').click();
  $('#pickFolder').onclick = () => $('#folderInput').click();
  for (const id of ['#fileInput', '#folderInput']) {
    $(id).onchange = (e) => {
      // a picked folder arrives as "<folder>/<sub>/<file>": its own name is not part of the dataset layout
      const files = [...e.target.files];
      for (const f of files) if (f.webkitRelativePath) f.__rel = f.webkitRelativePath.split('/').slice(1).join('/') || f.name;
      addFiles(files);
      e.target.value = '';
    };
  }
  $('#uploadPause').onclick = () => {
    paused = !paused;
    render();
    if (!paused) run();
  };
  $('#uploadClear').onclick = () => {
    for (let i = queue.length - 1; i >= 0; i--) if (queue[i].state !== 'up') queue.splice(i, 1);
    render();
  };
  $('#uploadList').addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-cancel]');
    if (!btn) return;
    const q = queue[Number(btn.dataset.cancel)];
    if (!q) return;
    q.state = 'err';
    q.error = 'đã huỷ';
    await api('/api/upload/cancel', { target: q.target }).catch((err) => toast(err.message, true));
    q.offset = 0;
    render();
    refreshServer().catch(() => {});
  });
  $('#copyScp').onclick = async () => {
    try { await navigator.clipboard.writeText($('#scpCmd').textContent); toast('Đã copy lệnh'); }
    catch { toast('Không copy được; hãy bôi đen lệnh rồi Ctrl+C', true); }
  };
  window.addEventListener('beforeunload', (e) => {
    if (running && !paused) { e.preventDefault(); e.returnValue = ''; }
  });
  // stop the browser from opening a file dropped just outside the zone
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => { if (!zone.contains(e.target)) e.preventDefault(); });
  $('#uploadDataset').addEventListener('change', updateTarget);
  window.__fflab.datasets?.ready.then(updateTarget);
  refreshServer().catch(() => {});
}
