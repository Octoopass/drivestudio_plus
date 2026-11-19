// Step 1 "Google Drive": guided Colab session -> Drive mount (consent button) -> folder pick -> download.
// Talks to /api/colab/* (fisheye_fusion/colab.py) and /api/colab/sync (tools/colab_sync_drive.py).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const { api, esc, toast } = window.__fflab;
const MY = '/content/drive/MyDrive';
const store = {
  get: (k) => { try { return localStorage.getItem(`fflab.${k}`) || ''; } catch { return ''; } },
  set: (k, v) => { try { localStorage.setItem(`fflab.${k}`, v); } catch { /* private mode */ } },
};
const fmt = (n) => n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(n / 2 ** 20))} MB`;
const st = { session: '', mounted: false, path: '', kitti: null, seq: '', ds: '', folder: null, items: new Set(), started: false, mountTimer: null };
const q = (params) => new URLSearchParams(params).toString();

function step(id, state, text = '') {
  const el = $(`#${id}`);
  el.dataset.state = state;          // todo | active | busy | done | error
  const label = $('.wz-state', el);
  if (label) label.textContent = text;
}

async function busy(btn, fn) {
  btn.disabled = true;
  btn.classList.add('is-busy');
  try { return await fn(); }
  catch (err) { toast(err.message, true); return undefined; }
  finally { btn.disabled = false; btn.classList.remove('is-busy'); }
}

// ---- 1. session
async function loadSessions(prefer) {
  step('wzSession', 'busy', 'Đang tải danh sách phiên…');
  let data;
  try { data = await api('/api/colab/sessions'); }
  catch (err) {
    step('wzSession', 'error', 'Không đọc được phiên');
    $('#colabSession').innerHTML = '<option value="">—</option>';
    toast(err.message, true);
    return;
  }
  const names = data.sessions.map((s) => s.name);
  const sel = $('#colabSession');
  sel.innerHTML = data.sessions.length
    ? data.sessions.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${esc(s.hardware || '?')}</option>`).join('')
    : '<option value="">Chưa có phiên nào, hãy tạo mới bên dưới</option>';
  const pick = [prefer, store.get('colabSession'), names[0]].find((n) => n && names.includes(n)) || '';
  sel.value = pick;
  $('#colabNewBox').open = !names.length;
  if (!names.includes($('#colabNewName').value)) { /* keep default */ } else {
    let i = 2; while (names.includes(`fisheye-lab-${i}`)) i++;
    $('#colabNewName').value = `fisheye-lab-${i}`;
  }
  await selectSession(pick);
}

async function selectSession(name) {
  st.session = name;
  st.mounted = false;
  $('#colabOpen').hidden = true;
  resetFolder();
  if (!name) {
    step('wzSession', 'active', 'Cần một phiên');
    step('wzDrive', 'todo', '');
    return;
  }
  store.set('colabSession', name);
  step('wzSession', 'done', `Đang dùng: ${name}`);
  api(`/api/colab/url?${q({ session: name })}`).then((r) => {
    if (st.session === name && /^https:\/\//.test(r.url)) { $('#colabOpen').href = r.url; $('#colabOpen').hidden = false; }
  }).catch(() => {});
  await checkDrive();
}

// ---- 2. Drive
async function checkDrive() {
  const name = st.session;
  step('wzDrive', 'busy', 'Đang kiểm tra Drive…');
  let data;
  try { data = await api(`/api/colab/drive?${q({ session: name })}`); }
  catch (err) { if (st.session === name) step('wzDrive', 'error', 'Không kiểm tra được'); toast(err.message, true); return; }
  if (st.session !== name) return;
  st.mounted = data.mounted;
  $('#driveConsent').hidden = true;
  if (data.mounted) {
    step('wzDrive', 'done', 'Đã kết nối');
    $('#driveMount').hidden = true;
    $('#driveMountHint').textContent = 'Drive đã sẵn sàng trên phiên này.';
    await openFolder(store.get(`drivePath.${name}`) || MY, true);
    autoFind();
  } else {
    $('#driveMount').hidden = false;
    $('#driveMountHint').textContent = 'Chỉ cần làm một lần cho mỗi phiên Colab.';
    step('wzDrive', 'active', 'Chưa kết nối');
    if (data.mount && ['running', 'needs_consent'].includes(data.mount.state)) showMount(data.mount);
  }
}

function showMount(view) {
  clearTimeout(st.mountTimer);
  const name = st.session;
  $('#driveMountLogBox').hidden = !view.log;
  $('#driveMountLog').textContent = view.log || '';
  const consent = view.state === 'needs_consent' && /^https:\/\//.test(view.consent_url || '');
  $('#driveConsent').hidden = !consent;
  $('#driveMount').hidden = view.state === 'running' || consent;
  if (consent) $('#driveConsentLink').href = view.consent_url;
  if (view.state === 'succeeded') { toast('Đã kết nối Google Drive'); checkDrive(); return; }
  if (view.state === 'failed') {
    step('wzDrive', 'error', 'Kết nối thất bại, thử lại');
    $('#driveMountLogBox').open = true;
    $('#driveMount').hidden = false;
    return;
  }
  step('wzDrive', consent ? 'active' : 'busy', consent ? 'Đang chờ bạn cho phép' : 'Đang kết nối…');
  st.mountTimer = setTimeout(async () => {
    if (st.session !== name) return;
    try { showMount(await api(`/api/colab/drive/mount?${q({ session: name })}`)); }
    catch (err) { toast(err.message, true); }
  }, 2000);
}

// ---- 3. folders
function resetFolder() {
  st.path = ''; st.kitti = null; st.seq = ''; st.ds = ''; st.folder = null; st.items = new Set();
  $('#driveFolders').innerHTML = '';
  $('#driveCrumbs').innerHTML = '';
  $('#driveFound').innerHTML = '';
  $('#driveFindHint').textContent = '';
  step('wzFolder', 'todo', '');
  renderSeqs();
}

async function openFolder(path, quiet = false) {
  const name = st.session;
  step('wzFolder', 'busy', 'Đang mở thư mục…');
  let data;
  try { data = await api(`/api/colab/browse?${q({ session: name, path })}`); }
  catch (err) { if (!quiet) toast(err.message, true); if (path !== MY) return openFolder(MY, true); step('wzFolder', 'error', ''); return; }
  if (st.session !== name) return;
  if (!data.exists) { if (path !== MY) return openFolder(MY, true); step('wzFolder', 'error', 'Không thấy MyDrive'); return; }
  st.path = data.path;
  store.set(`drivePath.${name}`, data.path);
  const rel = data.path.replace(/^\/content\/drive\/?/, '').split('/').filter(Boolean);
  $('#driveCrumbs').innerHTML = rel.map((part, i) => {
    const target = `/content/drive/${rel.slice(0, i + 1).join('/')}`;
    return i === rel.length - 1 ? `<b>${esc(part === 'MyDrive' ? 'Drive của tôi' : part)}</b>`
      : `<button type="button" class="link" data-path="${esc(target)}">${esc(part === 'MyDrive' ? 'Drive của tôi' : part)}</button>`;
  }).join('<span aria-hidden="true">›</span>');
  const up = rel.length > 1 ? `/content/drive/${rel.slice(0, -1).join('/')}` : '';
  $('#driveFolders').innerHTML = [
    up ? `<li><button type="button" class="folder up" data-path="${esc(up)}">↩ Lên một cấp</button></li>` : '',
    ...data.dirs.map((d) => `<li><button type="button" class="folder" data-path="${esc(`${data.path}/${d}`)}">📁 ${esc(d)}</button></li>`),
  ].join('') || '<li class="hint">Thư mục trống</li>';
  setFolder(data);
}

function setFolder(data) {
  st.path = data.path;
  st.folder = data;
  st.kitti = data.kitti && Object.keys(data.kitti.sequences || {}).length ? data.kitti : null;
  st.ds = data.dataset || '';
  st.items = new Set([...(data.dirs || []), ...(data.files || []).map((f) => f[0])]);
  const seqs = st.kitti ? Object.keys(st.kitti.sequences) : [];
  if (!seqs.includes(st.seq)) st.seq = seqs.length === 1 ? seqs[0] : '';
  const prof = window.__fflab.datasets?.byId(st.ds);
  const isRoot = data.path === MY;
  step('wzFolder', prof ? 'done' : 'active',
    prof ? `Nhận diện: ${prof.label}${st.kitti ? ` · ${seqs.length} sequence` : ''}`
      : isRoot ? 'Mở thư mục chứa dataset' : 'Chưa nhận ra dataset ở thư mục này');
  $('#driveDataset').value = st.ds;
  renderSeqs();
}

async function autoFind(btn) {
  const name = st.session;
  if (!st.mounted) return;
  $('#driveFindHint').textContent = 'Đang quét Drive (vài giây)…';
  const run = async () => {
    const data = await api(`/api/colab/find?${q({ session: name })}`);
    if (st.session !== name) return;
    const roots = data.roots;
    $('#driveFindHint').textContent = roots.length ? `Tìm thấy ${roots.length} thư mục dataset. Bấm để chọn.`
      : 'Không tự tìm thấy. Hãy mở thư mục chứa dataset ở danh sách bên dưới.';
    $('#driveFound').innerHTML = roots.map((r) => {
      const n = r.kitti ? Object.keys(r.kitti.sequences || {}).length : 0;
      return `<button type="button" class="found" data-path="${esc(r.path)}">
        <b>${esc(r.path.replace(MY, 'Drive của tôi'))}</b><small>${esc(r.label)}${n ? ` · ${n} sequence` : ''}</small></button>`;
    }).join('');
    $$('#driveFound .found').forEach((b) => { b.onclick = () => openFolder(b.dataset.path, true); });
    if (roots.length === 1 && !st.ds) await openFolder(roots[0].path, true);
  };
  if (btn) await busy(btn, run); else await run().catch((err) => { $('#driveFindHint').textContent = err.message; });
}

// ---- 4. what to download
function renderSeqs() {
  const box = $('#driveSeqs'), items = $('#driveItems');
  const ds = $('#driveDataset').value || st.ds || '';
  const prof = window.__fflab.datasets?.byId(ds);
  $('#driveChosen').textContent = st.path ? st.path.replace(MY, 'Drive của tôi') : '—';
  $('#driveDatasetHint').textContent = prof ? `Lưu vào ${prof.raw}/` : '';
  $('#driveCommon').innerHTML = '';
  box.innerHTML = '';
  items.hidden = true;
  const btn = $('#driveDownload');
  if (!st.path || st.path === MY) {
    box.innerHTML = '<p class="hint">Chọn thư mục dataset ở bước 3.</p>';
    btn.disabled = true;
    step('wzSeq', 'todo', '');
    return;
  }
  if (ds === 'kitti360') {
    if (!st.kitti) {
      box.innerHTML = '<p class="hint">Thư mục này không có data_2d_raw / data_poses.zip của KITTI-360.</p>';
      btn.disabled = true;
      step('wzSeq', 'active', 'Không thấy sequence');
      return;
    }
    const c = st.kitti.common || {};
    $('#driveCommon').innerHTML = [['data_poses.zip', 'poses'], ['calibration', 'calibration'], ['metadata', 'metadata .pkl']]
      .map(([k, label]) => `<span class="tag ${c[k] ? 'ok' : 'warn'}">${c[k] ? '✓' : 'thiếu'} ${label}</span>`).join('');
    box.innerHTML = Object.entries(st.kitti.sequences).sort(([a], [b]) => a.localeCompare(b)).map(([seq, info]) => `
      <button type="button" role="option" class="seq-card" data-seq="${esc(seq)}" aria-selected="${seq === st.seq}">
        <span class="seq-name">${esc(seq.replace('2013_05_28_drive_', 'drive_').replace('_sync', ''))}</span>
        <span class="tags"><span class="tag ${info.cameras === 4 ? 'ok' : 'warn'}">${info.cameras}/4 camera</span>
          <span class="tag ${info.lidar ? 'ok' : 'warn'}">${info.lidar ? 'LiDAR' : 'thiếu LiDAR'}</span>
          <span class="tag">${fmt(info.bytes || 0)}</span></span>
      </button>`).join('');
    $$('.seq-card', box).forEach((b) => b.onclick = () => { st.seq = b.dataset.seq; renderSeqs(); });
    btn.disabled = !st.seq;
    btn.lastChild.textContent = ' Tải sequence về máy ảo';
    step('wzSeq', 'active', st.seq ? `Sẵn sàng tải ${st.seq}` : 'Bấm chọn một sequence');
    return;
  }
  if (!prof) {
    box.innerHTML = '<p class="hint">Chưa nhận ra loại dataset. Hãy chọn đúng loại ở ô "Loại dữ liệu" phía trên.</p>';
    btn.disabled = true;
    step('wzSeq', 'active', 'Chọn loại dữ liệu');
    return;
  }
  const dirs = st.folder?.dirs || [], files = st.folder?.files || [];
  items.hidden = false;
  items.innerHTML = `<div class="row-between"><span class="hint">Chọn mục cần tải (thư mục được tải cả bên trong):</span>
      <span><button type="button" class="link" data-all="1">Chọn hết</button><button type="button" class="link" data-all="0">Bỏ hết</button></span></div>
    <ul>${[...dirs.map((d) => [d, null]), ...files].map(([n, size]) => `
      <li><label><input type="checkbox" value="${esc(n)}" ${st.items.has(n) ? 'checked' : ''}>
        ${size === null ? '📁' : '📄'} ${esc(n)} ${size === null ? '' : `<small>${fmt(size)}</small>`}</label></li>`).join('')}</ul>`;
  $$('input[type=checkbox]', items).forEach((c) => c.onchange = () => {
    c.checked ? st.items.add(c.value) : st.items.delete(c.value);
    renderSeqs();
  });
  $$('[data-all]', items).forEach((b) => b.onclick = () => {
    st.items = new Set(b.dataset.all === '1' ? [...dirs, ...files.map((f) => f[0])] : []);
    renderSeqs();
  });
  btn.disabled = !st.items.size;
  btn.lastChild.textContent = ` Tải ${st.items.size} mục về ${prof.raw}/`;
  step('wzSeq', 'active', st.items.size ? `Sẵn sàng tải ${prof.label}` : 'Chọn ít nhất một mục');
}

async function startSync(mode, btn) {
  if (!st.session || !st.path) return;
  const ds = $('#driveDataset').value || st.ds;
  if (mode === 'dataset' && ds !== 'kitti360') mode = 'folder';
  await busy(btn, async () => {
    const body = { mode, drive_path: st.path, sequence: mode === 'dataset' ? st.seq : '', session: st.session, dest: '' };
    if (mode === 'folder') {
      const all = (st.folder?.dirs || []).length + (st.folder?.files || []).length;
      Object.assign(body, { dataset: ds, items: st.items.size === all && all > 200 ? [] : [...st.items] });
    }
    const job = await api('/api/colab/sync', body);
    step('wzSeq', 'busy', mode === 'runs' ? 'Đang đồng bộ kết quả…' : 'Đang tải về máy ảo…');
    btn.disabled = true;
    window.__fflab.watchJob($('#driveSyncJob'), `/api/pipeline/${job.job_id}`, `/api/pipeline/${job.job_id}/cancel`, null, async (s) => {
      btn.disabled = false;
      if (s.state !== 'succeeded') { step('wzSeq', 'error', 'Tải thất bại, xem log bên dưới'); return; }
      step('wzSeq', 'done', 'Đã tải xong');
      toast(`Đã tải về máy ảo: ${s.info?.dest || ''}`);
      if (mode === 'dataset') await window.__fflab.selectImported?.(s.info?.dest || 'kitti360', st.seq);
      else if (mode === 'folder') { await window.__fflab.datasets?.load(); window.__fflab.showTab?.('src-local'); }
      else document.getElementById('dsRefresh')?.click();
    });
  });
}

// ---- wiring
$('#colabSession').onchange = (e) => selectSession(e.target.value);
$('#colabRefresh').onclick = (e) => busy(e.currentTarget, () => loadSessions(st.session));
$('#colabCreate').onclick = (e) => busy(e.currentTarget, async () => {
  const name = $('#colabNewName').value.trim();
  step('wzSession', 'busy', 'Đang tạo phiên (có thể mất 1 phút)…');
  try {
    await api('/api/colab/sessions', { session: name, gpu: $('#colabNewGpu').value });
  } catch (err) { step('wzSession', 'error', 'Tạo phiên thất bại'); throw err; }
  toast(`Đã tạo phiên ${name}`);
  $('#colabNewBox').open = false;
  await loadSessions(name);
});
$('#driveMount').onclick = (e) => busy(e.currentTarget, async () => {
  if (!st.session) throw new Error('Chọn hoặc tạo phiên Colab ở bước 1 trước');
  step('wzDrive', 'busy', 'Đang kết nối…');
  showMount(await api('/api/colab/drive/mount', { session: st.session }));
});
$('#driveConsentLink').addEventListener('click', () => { $('#driveConfirm').classList.add('pulse'); });
$('#driveConfirm').onclick = (e) => busy(e.currentTarget, async () => {
  $('#driveConfirm').classList.remove('pulse');
  $('#driveConsent').hidden = true;
  step('wzDrive', 'busy', 'Đang hoàn tất kết nối…');
  showMount(await api('/api/colab/drive/confirm', { session: st.session }));
});
$('#driveFind').onclick = (e) => autoFind(e.currentTarget);
for (const sel of ['#driveFolders', '#driveCrumbs']) {
  $(sel).addEventListener('click', (e) => {
    const b = e.target.closest('[data-path]');
    if (b) openFolder(b.dataset.path);
  });
}
$('#driveDownload').onclick = (e) => startSync('dataset', e.currentTarget);
$('#driveDataset').onchange = () => renderSeqs();
$('#driveRuns').onclick = (e) => startSync('runs', e.currentTarget);

function start() {
  if (st.started) return;
  st.started = true;
  loadSessions();
}
// lazy: only talk to Colab once the Drive source is shown (click, keyboard or a shortcut button)
const pane = $('#p-data-drive');
new MutationObserver(() => { if (!pane.hidden) start(); }).observe(pane, { attributes: true, attributeFilter: ['hidden'] });
if (!pane.hidden) start();
renderSeqs();
