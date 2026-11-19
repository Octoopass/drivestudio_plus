// In-app assistant (fisheye_fusion/assistant.py): floating chat panel; each question is answered with the app's
// real state (prepared data, configs, runs, GPUs, libraries) and may end in a proposed training form that the
// "Áp dụng vào form" button fills in. The API key is typed here once and kept on the server, never sent back.

const $ = (sel, root = document) => root.querySelector(sel);
const { api, esc, toast } = window.__fflab;
const KEY = 'fflab.assistantChat';
const st = { open: false, busy: false, cfg: null, msgs: load() };

function load() { try { return JSON.parse(sessionStorage.getItem(KEY) || '[]').slice(-30); } catch { return []; } }
function save() { try { sessionStorage.setItem(KEY, JSON.stringify(st.msgs.slice(-30))); } catch { /* private mode */ } }

const root = document.createElement('div');
root.innerHTML = `
  <button type="button" class="as-fab" id="asFab" aria-expanded="false" aria-controls="asPanel">
    <svg class="ic" aria-hidden="true"><use href="#i-chat"/></svg><span>Trợ lý</span></button>
  <aside class="as-panel" id="asPanel" role="dialog" aria-label="Trợ lý" hidden>
    <header class="as-head">
      <div><b>Trợ lý</b><small id="asModel">chưa cấu hình</small></div>
      <button type="button" class="as-icon" id="asSettingsBtn" aria-label="Cài đặt trợ lý" title="Cài đặt"><svg class="ic"><use href="#i-gear"/></svg></button>
      <button type="button" class="as-icon" id="asClear" aria-label="Xoá hội thoại" title="Xoá hội thoại"><svg class="ic"><use href="#i-trash"/></svg></button>
      <button type="button" class="as-icon" id="asClose" aria-label="Đóng"><svg class="ic"><use href="#i-x"/></svg></button>
    </header>
    <form class="as-settings" id="asSettings" hidden>
      <label>Nhà cung cấp <select name="provider"></select></label>
      <label id="asUrlWrap">URL endpoint (OpenAI-compatible) <input name="base_url" placeholder="https://host/v1" spellcheck="false" autocomplete="off"></label>
      <p class="hint" id="asUrlFixed" hidden></p>
      <label>API key <input name="api_key" type="password" autocomplete="new-password" spellcheck="false" placeholder="dán key vào đây">
        <small id="asKeyHint">Key được lưu trên máy chủ (file chỉ chủ sở hữu đọc được), không gửi lại trình duyệt.</small></label>
      <label>Model <span class="row-inline nowrap"><select name="model" class="grow"><option value="">— bấm "Tải danh sách" sau khi nhập key —</option></select>
        <button type="button" class="small" id="asLoadModels">Tải danh sách</button></span></label>
      <div class="row-inline"><button type="submit" class="primary small">Lưu</button>
        <button type="button" class="link" id="asForgetKey">Xoá key đã lưu</button></div>
    </form>
    <div class="as-log" id="asLog" aria-live="polite"></div>
    <div class="as-chips" id="asChips">
      <button type="button" class="chip" data-q="Với dữ liệu và form hiện tại, nên train cấu hình nào? Đề xuất cụ thể.">Nên train cấu hình nào?</button>
      <button type="button" class="chip" data-q="Lần chạy gần nhất thế nào? Có vấn đề gì cần sửa không?">Xem lần chạy gần nhất</button>
      <button type="button" class="chip" data-q="Giải thích ngắn sự khác nhau giữa các method config và renderer có ở đây.">Giải thích các method</button>
      <button type="button" class="chip" data-q="Máy còn thiếu thư viện nào và có ảnh hưởng gì không?">Thiếu thư viện gì?</button>
    </div>
    <form class="as-input" id="asForm">
      <textarea id="asText" rows="2" placeholder="Hỏi về dữ liệu, cấu hình, kết quả… (Enter để gửi, Shift+Enter xuống dòng)" aria-label="Câu hỏi"></textarea>
      <button type="submit" class="primary" id="asSend" aria-label="Gửi"><svg class="ic"><use href="#i-send"/></svg></button>
    </form>
  </aside>`;
document.body.append(root);

// ---------------------------------------------------------------- rendering (escaped, light markdown)
function md(text) {
  const blocks = String(text).replace(/```fflab-apply[\s\S]*?```/g, '').trim().split(/```/);
  return blocks.map((b, i) => {
    if (i % 2) return `<pre><code>${esc(b.replace(/^[a-z0-9-]*\n/i, ''))}</code></pre>`;
    return esc(b)
      .replace(/`([^`\n]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
      .replace(/^\s*[-*] (.*)$/gm, '<span class="as-li">• $1</span>')
      .replace(/^#{1,4} (.*)$/gm, '<b>$1</b>')
      .replace(/\n/g, '<br>');
  }).join('');
}
const APPLY_LABEL = { dataset: 'dataset', data_root: 'dữ liệu', method: 'method', render_mode: 'renderer', fastgs: 'FastGS',
  steps: 'số bước', gpu: 'GPU', target: 'nơi chạy' };
function render() {
  $('#asLog').innerHTML = st.msgs.length ? st.msgs.map((m, i) => `
    <div class="as-msg ${m.role === 'user' ? 'me' : 'bot'}">
      <div class="as-bubble">${m.role === 'user' ? esc(m.content).replace(/\n/g, '<br>') : md(m.content)}</div>
      ${m.apply ? `<div class="as-apply"><span>${Object.entries(m.apply).filter(([k]) => APPLY_LABEL[k])
        .map(([k, v]) => `${APPLY_LABEL[k]}: <code>${esc(v === '' ? 'theo config' : v)}</code>`).join(' · ')}</span>
        <button type="button" class="small primary" data-apply="${i}">Áp dụng vào form</button></div>` : ''}
    </div>`).join('') + (st.busy ? '<div class="as-msg bot"><div class="as-bubble as-typing"><span></span><span></span><span></span></div></div>' : '')
    : `<p class="as-empty">Hỏi trợ lý về dữ liệu, cấu hình train hay kết quả. Mỗi câu hỏi được gửi kèm trạng thái hiện tại của app (dữ liệu đã chuẩn bị, config, các lần chạy, GPU) để trả lời sát thực tế.</p>`;
  const log = $('#asLog');
  log.scrollTop = log.scrollHeight;
  $('#asChips').hidden = st.msgs.length > 0;
}

// ---------------------------------------------------------------- config
function showProvider(c) {
  const f = $('#asSettings');
  const prov = (c.providers || []).find((p) => p.id === f.elements.provider.value) || {};
  $('#asUrlWrap').hidden = !prov.needs_url;
  $('#asUrlFixed').hidden = !!prov.needs_url;
  $('#asUrlFixed').innerHTML = prov.base_url ? `Endpoint chính thức: <code>${esc(prov.base_url)}</code>`
    : prov.id === 'anthropic' ? 'Gọi qua SDK chính thức của Anthropic (api.anthropic.com).' : '';
}
async function loadConfig() {
  try { st.cfg = await api('/api/assistant/config'); } catch { st.cfg = null; }
  const c = st.cfg || {};
  const label = (c.providers || []).find((p) => p.id === c.provider)?.label || '';
  $('#asModel').textContent = c.model && c.has_key ? `${c.model}${label ? ` · ${label}` : ''}` : 'chưa cấu hình';
  const f = $('#asSettings');
  f.elements.provider.innerHTML = (c.providers || []).map((p) => `<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');
  f.elements.provider.value = c.provider || 'custom';
  f.elements.base_url.value = c.base_url && c.provider === 'custom' ? c.base_url : '';
  f.elements.api_key.value = '';
  f.elements.api_key.placeholder = c.has_key ? `đã lưu (${c.key_hint}); để trống nếu không đổi` : 'dán key vào đây';
  f.elements.model.innerHTML = '<option value="">— bấm "Tải danh sách" sau khi nhập key —</option>';
  if (c.model) f.elements.model.add(new Option(c.model, c.model));
  f.elements.model.value = c.model || '';
  showProvider(c);
  if (!(c.base_url !== undefined && c.has_key && c.model)) $('#asSettings').hidden = false;
}
// switching provider saves the choice at once, so each provider keeps its own key and model on the server
async function onProvider() {
  try {
    st.cfg = await api('/api/assistant/config', { provider: $('#asSettings').elements.provider.value });
    await loadConfig();
  } catch (err) { toast(err.message, true); }
}
async function loadModels(btn) {
  btn.disabled = true;
  try {
    const f = $('#asSettings');
    await saveSettings(false);
    const { models } = await api('/api/assistant/models');
    const keep = f.elements.model.value;
    f.elements.model.innerHTML = models.map((m) => `<option value="${esc(m)}">${esc(m)}</option>`).join('') || '<option value="">(API không trả model nào)</option>';
    const luna = models.find((m) => /luna/i.test(m));
    f.elements.model.value = models.includes(keep) ? keep : (luna || models[0] || '');
    toast(`${models.length} model`);
  } catch (err) { toast(err.message, true); } finally { btn.disabled = false; }
}
async function saveSettings(close = true) {
  const f = $('#asSettings');
  const body = { provider: f.elements.provider.value, model: f.elements.model.value };
  if (f.elements.provider.value === 'custom') body.base_url = f.elements.base_url.value.trim();
  if (f.elements.api_key.value.trim()) body.api_key = f.elements.api_key.value.trim();
  st.cfg = await api('/api/assistant/config', body);
  f.elements.api_key.value = '';
  const keepModel = f.elements.model.value;
  await loadConfig();
  if (keepModel && ![...f.elements.model.options].some((o) => o.value === keepModel)) f.elements.model.add(new Option(keepModel, keepModel));
  if (close && st.cfg.model && st.cfg.has_key && st.cfg.base_url !== undefined) { f.hidden = true; toast('Đã lưu cài đặt trợ lý'); }
}
$('#asSettings').addEventListener('submit', (e) => { e.preventDefault(); saveSettings().catch((err) => toast(err.message, true)); });
$('#asLoadModels').onclick = (e) => loadModels(e.currentTarget);
$('#asSettings').elements.provider.addEventListener('change', onProvider);
$('#asForgetKey').onclick = async () => {
  if (!confirm('Xoá API key đã lưu trên máy chủ?')) return;
  st.cfg = await api('/api/assistant/config', { clear_key: true }).catch((err) => toast(err.message, true));
  loadConfig();
};

// ---------------------------------------------------------------- chat
function formState() {
  const f = $('#dsTrainForm');
  if (!f) return {};
  const v = (n) => f.elements[n]?.value ?? '';
  return { dataset: v('dataset'), data_root: v('data_root'), method: v('method'), render_mode: v('render_mode'),
    fastgs: v('fastgs'), steps: v('steps'), gpu: v('gpu'), name: v('name'),
    target: window.__fflab.trainColab?.active() ? 'colab' : 'local' };
}
async function ask(text) {
  text = text.trim();
  if (!text || st.busy) return;
  const c = st.cfg;
  if (!(c && c.has_key && c.model && (c.base_url || c.provider === 'anthropic'))) { $('#asSettings').hidden = false; toast('Chọn nhà cung cấp, nhập key và chọn model trước', true); return; }
  st.msgs.push({ role: 'user', content: text });
  st.busy = true;
  render();
  try {
    const page = document.querySelector('.sec.is-active')?.id || '';
    const r = await api('/api/assistant/chat', { messages: st.msgs.map(({ role, content }) => ({ role, content })), form: formState(), page });
    st.msgs.push({ role: 'assistant', content: r.reply, apply: r.apply || null });
  } catch (err) {
    st.msgs.push({ role: 'assistant', content: `Lỗi: ${err.message}` });
  } finally { st.busy = false; save(); render(); }
}
$('#asForm').addEventListener('submit', (e) => { e.preventDefault(); const t = $('#asText'); const q = t.value; t.value = ''; ask(q); });
$('#asText').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#asForm').requestSubmit(); } });
$('#asChips').addEventListener('click', (e) => { const b = e.target.closest('[data-q]'); if (b) ask(b.dataset.q); });
$('#asClear').onclick = () => { st.msgs = []; save(); render(); };

// ---------------------------------------------------------------- apply a proposal to the training form
function applyProposal(p) {
  window.__fflab.goto?.('step-train', 't-train-ds');
  const f = $('#dsTrainForm');
  const skipped = [];
  const setSelect = (name, val) => {
    if (val == null || val === undefined) return;
    const el = f.elements[name];
    if (!el) return;
    if (el instanceof RadioNodeList || el.type === 'radio') {
      const r = [...f.querySelectorAll(`input[name="${name}"]`)].find((x) => x.value === String(val));
      if (r) { r.checked = true; } else skipped.push(`${APPLY_LABEL[name] || name}=${val}`);
      return;
    }
    if (el.tagName === 'SELECT' && ![...el.options].some((o) => o.value === String(val))) { skipped.push(`${APPLY_LABEL[name] || name}=${val}`); return; }
    el.value = String(val);
  };
  if (p.target === 'colab' || p.target === 'local') $(`#tgt-${p.target}`)?.click();
  setSelect('data_root', p.data_root);
  f.elements.data_root.dispatchEvent(new Event('change', { bubbles: true }));
  setSelect('dataset', p.dataset);
  setSelect('method', p.method);
  setSelect('render_mode', p.render_mode ?? '');
  setSelect('fastgs', p.fastgs === true ? 'true' : p.fastgs === false ? 'false' : (p.fastgs ?? ''));
  if (p.steps) f.elements.steps.value = Math.max(100, Math.min(100000, parseInt(p.steps, 10) || 30000));
  if (p.target !== 'colab') setSelect('gpu', p.gpu);
  if (p.name && /^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$/.test(p.name)) f.elements.name.value = p.name;
  else if (!f.elements.name.value) $('#tfAutoName')?.click();
  f.dispatchEvent(new Event('change', { bubbles: true }));
  toast(skipped.length ? `Đã điền form; bỏ qua giá trị không có: ${skipped.join(', ')}` : 'Đã điền form. Kiểm tra lại rồi bấm Bắt đầu.', !!skipped.length);
  $('#tfSubmit')?.scrollIntoView({ block: 'center', behavior: 'smooth' });
}
$('#asLog').addEventListener('click', (e) => {
  const b = e.target.closest('[data-apply]');
  if (b) applyProposal(st.msgs[+b.dataset.apply].apply || {});
});

// ---------------------------------------------------------------- open / close
function setOpen(open) {
  st.open = open;
  $('#asPanel').hidden = !open;
  $('#asFab').setAttribute('aria-expanded', String(open));
  if (open) { loadConfig().then(() => $('#asSettings').hidden || $('#asText').focus()); render(); }
}
$('#asFab').onclick = () => setOpen(!st.open);
$('#asClose').onclick = () => { setOpen(false); $('#asFab').focus(); };
$('#asSettingsBtn').onclick = () => { $('#asSettings').hidden = !$('#asSettings').hidden; };
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && st.open) { setOpen(false); $('#asFab').focus(); } });
window.__fflab.assistant = { open: () => setOpen(true), ask };
