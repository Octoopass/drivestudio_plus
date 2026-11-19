// Step 2 "Huấn luyện": where to run (this VM's GPUs or a Colab GPU), the Colab session picker and the
// Colab job cards. The form itself is submitted by drivestudio.js, which calls trainColab.start() when
// the Colab target is selected. Talks to /api/colab/train* (fisheye_fusion/colab_train.py).

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const { api, esc, toast } = window.__fflab;
const form = $('#dsTrainForm');
const store = {
  get: (k) => { try { return localStorage.getItem(`fflab.${k}`) || ''; } catch { return ''; } },
  set: (k, v) => { try { localStorage.setItem(`fflab.${k}`, v); } catch { /* private mode */ } },
};
const NEW = '__new__';
const st = { target: store.get('trainTarget') === 'colab' ? 'colab' : 'local', sessions: [], loaded: false, jobsTimer: null, stages: [] };

async function busy(btn, fn) {
  btn.disabled = true;
  try { return await fn(); } catch (err) { toast(err.message, true); return undefined; } finally { btn.disabled = false; }
}
const isGpu = (s) => s.hardware && s.hardware.toUpperCase() !== 'CPU';
const gpuList = () => window.__fflab.config?.colab?.gpus || [];
const defaultGpu = () => window.__fflab.config?.colab?.default_gpu || gpuList()[0]?.id || '';
window.__fflab.configReady?.then((cfg) => {
  const c = cfg.colab || {};
  $('#tcGpu').innerHTML = gpuList().map((g) => `<option value="${esc(g.id)}">${esc(g.label)} · ${esc(g.memory_gb)} GB${g.tier === 'free' ? ' (miễn phí)' : ' (trả phí)'}</option>`).join('');
  $('#tcGpu').value = defaultGpu();
  if (!$('#tcName').value) $('#tcName').value = c.gpu_session || 'fflab-gpu';
  summary();
}).catch(() => {});

// ---------------------------------------------------------------- target cards
function setTarget(target) {
  st.target = target;
  store.set('trainTarget', target);
  for (const [id, pane, on] of [['tgt-local', 'tgtPaneLocal', target === 'local'], ['tgt-colab', 'tgtPaneColab', target === 'colab']]) {
    $(`#${id}`).setAttribute('aria-selected', String(on));
    $(`#${id}`).tabIndex = on ? 0 : -1;
    $(`#${pane}`).hidden = !on;
  }
  $('#tfSubmit span').textContent = target === 'colab' ? 'Bắt đầu train trên Colab GPU' : 'Bắt đầu train trên máy ảo';
  const warn = $('#trainDepsLocal');
  if (warn) {
    warn.dataset.depNeed = target === 'colab' ? '' : 'ds-env,ds-cuda';
    window.__fflab.deps?.decorate(warn.parentElement);
  }
  if (target === 'colab' && !st.loaded) loadColab();
  summary();
}
$('#tgt-local').onclick = () => setTarget('local');
$('#tgt-colab').onclick = () => setTarget('colab');
$('[aria-label="Nơi chạy"]').addEventListener('keydown', (e) => {
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  const next = st.target === 'local' ? 'colab' : 'local';
  setTarget(next);
  $(`#tgt-${next}`).focus();
});

function localInfo() {
  const setup = window.__fflab.getSetup?.();
  const gpus = setup?.gpus || [];
  if (!gpus.length) { $('#tgtLocalInfo').textContent = 'Không thấy GPU'; return; }
  const names = [...new Set(gpus.map((g) => g.name.replace(/^NVIDIA\s+/, '')))].join(', ');
  const idle = gpus.filter((g) => g.utilization <= 20).length;
  $('#tgtLocalInfo').textContent = `${gpus.length}× ${names} · ${idle} rảnh`;
  setTimeout(summary, 0);
}
window.__fflab.onSetup?.(localInfo);

// ---------------------------------------------------------------- Colab session picker
function renderSessions() {
  const sel = $('#tcSession');
  const keep = sel.value || store.get('trainSession');
  const gpu = st.sessions.filter(isGpu);
  const cpu = st.sessions.filter((s) => !isGpu(s));
  sel.innerHTML = [
    ...gpu.map((s) => `<option value="${esc(s.name)}">${esc(s.name)} · ${esc(s.hardware)}</option>`),
    ...cpu.map((s) => `<option value="${esc(s.name)}" disabled>${esc(s.name)} · CPU (không train được)</option>`),
    `<option value="${NEW}">+ Tạo phiên GPU mới</option>`,
  ].join('');
  sel.value = gpu.some((s) => s.name === keep) ? keep : gpu[0]?.name || NEW;
  onSession();
}
async function onSession() {
  const name = $('#tcSession').value;
  const isNew = name === NEW;
  $('#tcNewWrap').hidden = !isNew;
  $('#tcNameWrap').hidden = !isNew;
  $('#tcStop').hidden = isNew;
  $('#tcOpen').hidden = true;
  if (!isNew) {
    store.set('trainSession', name);
    api(`/api/colab/url?session=${encodeURIComponent(name)}`).then((r) => {
      if ($('#tcSession').value === name && /^https:\/\//.test(r.url)) { $('#tcOpen').href = r.url; $('#tcOpen').hidden = false; }
    }).catch(() => {});
  }
  summary();
}
$('#tcSession').onchange = onSession;
$('#tcGpu').onchange = summary;

async function loadColab() {
  st.loaded = true;
  $('#tcSession').innerHTML = '<option value="">Đang tải…</option>';
  const [acct, sessions, usage] = await Promise.allSettled([api('/api/colab/accounts'), api('/api/colab/sessions'), api('/api/colab/usage')]);
  if (acct.status === 'fulfilled') {
    $('#tcAccount').innerHTML = acct.value.active ? `Tài khoản Google: <b>${esc(acct.value.active)}</b>` : '<span class="tag bad">Chưa đăng nhập Google</span>';
  } else $('#tcAccount').textContent = `Tài khoản Google: ${acct.reason.message}`;
  if (usage.status === 'fulfilled' && usage.value.balance !== null) {
    const u = usage.value;
    const free = gpuList().filter((g) => g.tier === 'free').map((g) => g.label);
    $('#tcUsage').innerHTML = `· ${u.balance.toFixed(2)} compute unit${u.balance < 1 ? ` <span class="tag warn">chỉ GPU miễn phí${free.length ? `: ${esc(free.join(', '))}` : ''}</span>` : ''}`;
    $('#tcGpu').querySelectorAll('option').forEach((o) => { o.disabled = u.balance < 1 && gpuList().find((g) => g.id === o.value)?.tier !== 'free'; });
  }
  if (sessions.status === 'fulfilled') {
    st.sessions = sessions.value.sessions || [];
    renderSessions();
    const gpus = st.sessions.filter(isGpu).length;
    $('#tgtColabInfo').textContent = gpus ? `${gpus} phiên GPU đang mở` : 'Chưa có phiên GPU, sẽ tạo mới';
  } else {
    $('#tcSession').innerHTML = `<option value="${NEW}">+ Tạo phiên GPU mới</option>`;
    onSession();
    toast(sessions.reason.message, true);
  }
}
$('#tcReload').onclick = (e) => busy(e.currentTarget, loadColab);
$('#tcStop').onclick = (e) => busy(e.currentTarget, async () => {
  const name = $('#tcSession').value;
  if (!confirm(`Dừng phiên Colab "${name}"? Mọi thứ trên máy Colab đó (môi trường đã cài, file chưa tải về) sẽ mất.`)) return;
  const r = await api('/api/colab/sessions/stop', { session: name });
  st.sessions = r.sessions || [];
  renderSessions();
  toast(`Đã dừng phiên ${name}`);
});
$('#tcAccountChange').onclick = () => {
  window.__fflab.goto('step-convert');
  $('#src-drive')?.click();
  $('#wzAccount')?.scrollIntoView({ block: 'center' });
};

// ---------------------------------------------------------------- name, steps, summary
const radio = (n) => form.elements[n].value;
function autoName() {
  const m = (form.elements.method.value || 'run').replace(/^configs\//, '').replace(/\.yaml$/, '');
  const r = radio('render_mode');
  const steps = +form.elements.steps.value || 30000;
  const k = steps >= 1000 ? `${Math.round(steps / 1000)}k` : String(steps);
  const d = new Date();
  const stamp = `${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
  const where = st.target === 'colab' ? '_colab' : '';
  form.elements.name.value = `${m}${r && !m.includes(r) ? `_${r}` : ''}_${k}${where}_${stamp}`.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 80);
  summary();
}
$('#tfAutoName').onclick = autoName;
$$('[data-steps]', form).forEach((b) => { b.onclick = () => { form.elements.steps.value = b.dataset.steps; summary(); }; });

function summary() {
  const parts = [];
  const m = form.elements.method.value;
  const ds = form.elements.dataset.value;
  if (ds) parts.push(`<code>${esc(form.elements.data_root.value || ds)}</code>`);
  if (m) parts.push(`<code>${esc(m.replace(/^configs\//, ''))}</code>`);
  const r = radio('render_mode');
  parts.push(r ? `renderer <b>${esc(r)}</b>` : 'renderer theo config');
  const f = radio('fastgs');
  if (f) parts.push(`FastGS ${f === 'true' ? 'bật' : 'tắt'}`);
  parts.push(`${(+form.elements.steps.value || 0).toLocaleString('vi-VN')} bước`);
  if (st.target === 'colab') {
    const s = $('#tcSession').value;
    parts.push(s === NEW ? `Colab: phiên mới <b>${esc($('#tcName').value)}</b> (${esc($('#tcGpu').value)})` : `Colab: <b>${esc(s || '—')}</b>`);
  } else {
    const g = form.elements.gpu;
    parts.push(`máy ảo: ${esc(g.selectedOptions[0]?.textContent.split(' · ').slice(0, 2).join(' · ') || 'GPU mặc định')}`);
  }
  $('#tfSummary').innerHTML = `Sẽ chạy: ${parts.join(' · ')}`;
}
form.addEventListener('change', summary);
form.addEventListener('input', (e) => { if (e.target.name === 'steps' || e.target.id === 'tcName') summary(); });

// ---------------------------------------------------------------- Colab start + job cards
async function start(body) {
  let session = $('#tcSession').value;
  if (!session) throw new Error('Chọn phiên Colab');
  let gpu = $('#tcGpu').value;
  if (session === NEW) {
    session = $('#tcName').value.trim();
    if (!/^[A-Za-z0-9_.-]{1,64}$/.test(session)) throw new Error('Tên phiên mới chỉ gồm chữ, số, _ . -');
    if (st.sessions.some((s) => s.name === session)) throw new Error(`Đã có phiên "${session}", chọn nó trong danh sách hoặc đặt tên khác`);
  } else {
    const hw = (st.sessions.find((s) => s.name === session)?.hardware || '').toUpperCase();
    gpu = gpuList().find((g) => g.id.toUpperCase() === hw)?.id || defaultGpu();
  }
  const payload = { ...body, gpu: null, session, colab_gpu: gpu, with_ckpt: $('#tcCkpt').checked };
  await api('/api/colab/train', payload);
  store.set('trainSession', session);
  toast(`Đã gửi ${body.name} lên Colab. Theo dõi tiến trình ngay bên dưới.`);
  await loadJobs();
  $('#colabJobsWrap').scrollIntoView({ block: 'start', behavior: 'smooth' });
}

const ago = (t) => {
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? `${Math.round(s)} giây trước` : s < 3600 ? `${Math.round(s / 60)} phút trước` : `${(s / 3600).toFixed(1)} giờ trước`;
};
const STATE = { running: ['run', 'đang chạy'], succeeded: ['ok', 'xong'], failed: ['bad', 'lỗi'], cancelled: ['warn', 'đã huỷ'] };
const openLogs = new Set();

function renderJobs(jobs) {
  $('#colabJobsWrap').hidden = !jobs.length;
  $('#colabJobs').innerHTML = jobs.slice(0, 8).map((j) => {
    const [cls, label] = STATE[j.state] || ['', j.state];
    const steps = st.stages.map((s) => ({ label: s.label, ...(j.stages[s.key] || { state: 'todo' }) }));
    const board = window.__fflab.progressBoard(steps, { now: j.now, state: j.state, started: j.started });
    return `<article class="cjob" data-state="${esc(j.state)}" data-name="${esc(j.name)}">
      <div class="cjob-head"><b>${esc(j.name)}</b><span class="tag ${cls}">${label}</span>
        <span class="hint grow">Colab <code>${esc(j.session)}</code> · ${esc(j.cfg?.method?.replace(/^configs\//, '') || '')} · ${esc(j.data_root)}/${String(j.scene).padStart(3, '0')} · cập nhật ${ago(j.updated || j.started)}</span>
        <button type="button" class="small" data-act="log">${openLogs.has(j.name) ? 'Ẩn log' : 'Xem log'}</button>
        ${j.state === 'running' ? '<button type="button" class="small" data-act="cancel">Huỷ</button>' : ''}
        ${j.stages.train?.state !== 'todo' ? '<button type="button" class="small" data-act="results">Kết quả →</button>' : ''}</div>
      ${board}
      ${j.error ? `<p class="err">${esc(j.error)}</p>` : ''}
      <pre data-log ${openLogs.has(j.name) ? '' : 'hidden'}></pre>
    </article>`;
  }).join('');
  openLogs.forEach((n) => showLog(n));
}
async function showLog(name) {
  const pre = $(`.cjob[data-name="${CSS.escape(name)}"] pre`);
  if (!pre) return;
  try { const s = await api(`/api/colab/train/${encodeURIComponent(name)}`); pre.textContent = s.log || '(chưa có log)'; pre.scrollTop = pre.scrollHeight; } catch { /* keep */ }
}
$('#colabJobs').addEventListener('click', async (e) => {
  const btn = e.target.closest('button[data-act]');
  if (!btn) return;
  const name = btn.closest('.cjob').dataset.name;
  if (btn.dataset.act === 'log') {
    if (openLogs.has(name)) openLogs.delete(name); else openLogs.add(name);
    loadJobs();
  } else if (btn.dataset.act === 'cancel') {
    if (!confirm(`Huỷ ${name}? Job trên Colab sẽ bị dừng.`)) return;
    await busy(btn, () => api(`/api/colab/train/${encodeURIComponent(name)}/cancel`, {}));
    loadJobs();
  } else if (btn.dataset.act === 'results') {
    window.__fflab.goto('step-results', 't-res-ds');
  }
});

async function loadJobs() {
  clearTimeout(st.jobsTimer);
  try {
    const data = await api('/api/colab/train');
    st.stages = data.stages;
    renderJobs(data.jobs);
    if (data.jobs.some((j) => j.state === 'running')) st.jobsTimer = setTimeout(loadJobs, 8000);
  } catch { st.jobsTimer = setTimeout(loadJobs, 30000); }
}

window.__fflab.trainColab = { active: () => st.target === 'colab', start, summary };
setTarget(st.target);
loadJobs();
localInfo();
