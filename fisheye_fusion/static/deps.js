// "Thư viện & môi trường" (overview): every library / environment the app needs, with its source links,
// status and an install button (jobs shown with the shared progress board). Other steps put
// <div class="deps-need" data-dep-need="ID[,ID]"> where something is required; decorate() fills those with a
// warning and a "Cài ngay" button while the dependency is missing. Talks to /api/deps (fisheye_fusion/deps.py).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const { api, esc, toast, watchJob, fetchSetup } = window.__fflab;
const st = { items: [], byId: new Map(), watching: new Set(), loaded: false };

const isUrl = (u) => /^https:\/\//.test(u);
const minutes = (m) => (m ? (m < 60 ? `~${m} phút` : `~${Math.round(m / 6) / 10} giờ`) : '');

function row(d) {
  const reqMissing = (d.requires || []).filter((r) => !st.byId.get(r)?.ok);
  const running = d.job || st.watching.has(d.id);
  let action = '';
  if (d.ok) action = '<span class="tag ok">Đã cài</span>';
  else if (!d.installable) action = `<a class="btn small" href="${esc(d.links[0][1])}" target="_blank" rel="noopener">Hướng dẫn cài <svg class="ic sm"><use href="#i-ext"/></svg></a>`;
  else if (running) action = '<span class="tag run">Đang cài…</span>';
  else if (reqMissing.length) action = `<button type="button" class="small" disabled title="Cài trước: ${esc(reqMissing.map((r) => st.byId.get(r)?.name).join(', '))}">Cài đặt</button>
    <small>cần cài trước: ${reqMissing.map((r) => `<button type="button" class="link" data-dep-goto="${esc(r)}">${esc(st.byId.get(r)?.name || r)}</button>`).join(', ')}</small>`;
  else action = `<button type="button" class="primary small" data-dep-install="${esc(d.id)}">Cài đặt</button>`;
  const links = (d.links || []).map(([label, url]) => isUrl(url)
    ? `<a href="${esc(url)}" target="_blank" rel="noopener">${esc(label)} <svg class="ic sm"><use href="#i-ext"/></svg></a>`
    : `<code>${esc(url)}</code>`).join(' · ');
  return `<article class="dep" id="dep-${esc(d.id)}" data-ok="${d.ok ? 'yes' : 'no'}">
    <span class="dep-ic" aria-hidden="true">${d.ok ? '✓' : '○'}</span>
    <div class="dep-main">
      <div class="dep-title"><b>${esc(d.name)}</b>${d.size ? `<span class="hint">${esc(d.size)}${d.minutes ? ` · ${minutes(d.minutes)}` : ''}</span>` : ''}</div>
      <p>${esc(d.desc)}</p>
      <p class="dep-detail ${d.ok ? '' : 'warn'}">${esc(d.detail || '')}</p>
      <div class="dep-links">${links}</div>
    </div>
    <div class="dep-act">${action}</div>
    <div class="dep-job job" hidden></div>
  </article>`;
}

function render() {
  const box = $('#depsList');
  if (!box) return;
  const groups = [...new Set(st.items.map((d) => d.group))];
  box.innerHTML = groups.map((g) => {
    const items = st.items.filter((d) => d.group === g);
    const ok = items.filter((d) => d.ok).length;
    return `<section class="dep-group"><h4>${esc(g)} <small>${ok}/${items.length}</small></h4>${items.map(row).join('')}</section>`;
  }).join('');
  const missing = st.items.filter((d) => !d.ok);
  $('#depsSummary').textContent = missing.length ? `Thiếu ${missing.length}/${st.items.length} mục` : 'Đã cài đủ';
  // re-attach progress boards of running installs
  st.items.filter((d) => d.job).forEach((d) => watch(d.id, d.job));
  st.watching.forEach((id) => { const d = st.byId.get(id); if (d?.jobId) watch(id, d.jobId, true); });
  decorate(document);
}

function watch(id, jobId, force = false) {
  const box = $(`#dep-${CSS.escape(id)} .dep-job`);
  if (!box) return;
  if (box.dataset.job === jobId && !force) return;
  box.dataset.job = jobId;
  st.watching.add(id);
  const d = st.byId.get(id);
  if (d) d.jobId = jobId;
  watchJob(box, `/api/pipeline/${jobId}`, `/api/pipeline/${jobId}/cancel`, null, async (s) => {
    st.watching.delete(id);
    if (s.state === 'succeeded') toast(`Đã cài xong: ${d?.name || id}`);
    else if (s.state === 'failed') toast(`Cài ${d?.name || id} lỗi, xem log trong bảng Thư viện`, true);
    await load();
    fetchSetup?.(true).catch(() => {});
    window.__fflab.reloadData?.().catch?.(() => {});
  });
}

async function install(id, btn) {
  const d = st.byId.get(id);
  if (btn) btn.disabled = true;
  try {
    const job = await api(`/api/deps/${encodeURIComponent(id)}/install`, {});
    st.watching.add(id);
    if (d) d.jobId = job.job_id;
    render();
    watch(id, job.job_id, true);
    toast(`Bắt đầu cài ${d?.name || id}`);
  } catch (err) { toast(err.message, true); if (btn) btn.disabled = false; }
}

function gotoDep(id) {
  window.__fflab.goto?.('overview');
  const el = $(`#dep-${CSS.escape(id)}`);
  if (!el) return;
  el.scrollIntoView({ block: 'center', behavior: 'smooth' });
  el.classList.remove('flash');
  void el.offsetWidth;
  el.classList.add('flash');
}

/** Fill every [data-dep-need] placeholder under root with a warning + button for missing dependencies. */
function decorate(root = document) {
  if (!st.loaded) return;
  $$('[data-dep-need]', root).forEach((el) => {
    const ids = (el.dataset.depNeed || '').split(',').map((x) => x.trim()).filter(Boolean);
    const missing = ids.map((id) => st.byId.get(id)).filter((d) => d && !d.ok);
    el.hidden = !missing.length;
    el.innerHTML = missing.length ? `<span>Cần cài trước:</span>
      ${missing.map((d) => (d.job || st.watching.has(d.id))
        ? `<button type="button" class="small" data-dep-goto="${esc(d.id)}">${esc(d.name)} · đang cài…</button>`
        : `<button type="button" class="small primary" data-dep-goto="${esc(d.id)}">${esc(d.name)} →</button>`).join(' ')}` : '';
  });
}

async function load() {
  try {
    const data = await api('/api/deps');
    st.items = data.items;
    st.byId = new Map(data.items.map((d) => [d.id, { ...st.byId.get(d.id), ...d }]));
    st.items = st.items.map((d) => st.byId.get(d.id));
    st.loaded = true;
    render();
  } catch (err) {
    const box = $('#depsList');
    if (box) box.innerHTML = `<p class="bad">Không đọc được danh sách thư viện: ${esc(err.message)}</p>`;
  }
}

document.addEventListener('click', (e) => {
  const go = e.target.closest('[data-dep-goto]');
  if (go) { e.preventDefault(); gotoDep(go.dataset.depGoto); return; }
  const ins = e.target.closest('[data-dep-install]');
  if (ins) install(ins.dataset.depInstall, ins);
});
$('#depsRefresh')?.addEventListener('click', (e) => {
  const btn = e.currentTarget;
  btn.disabled = true;
  load().finally(() => { btn.disabled = false; });
});

window.__fflab.deps = { load, decorate, gotoDep, get: (id) => st.byId.get(id) };
load();
