// Progress board shared by long jobs (pipeline jobs with a `plan`, Colab training stages): overall bar,
// time left, expected finish time and per-step durations. Pure rendering: callers pass server data.
//
// steps: [{label, state: todo|busy|done|error, est (s) | null, started, ended (epoch s), frac 0..1, eta (s), note}]
// opts:  {now: server epoch s, state: running|succeeded|failed|cancelled, started, ended}

export function fmtDur(sec) {
  if (sec == null || !Number.isFinite(sec)) return '?';
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return sec < 15 ? 'vài giây' : 'dưới 1 phút';
  const m = Math.round(sec / 60);
  if (m < 60) return `${m} phút`;
  const h = Math.floor(m / 60);
  return m % 60 ? `${h} giờ ${m % 60} phút` : `${h} giờ`;
}

const clock = (ms) => new Date(ms).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function remainingOf(step, now) {
  if (step.state === 'done' || step.state === 'error') return 0;
  if (step.state === 'todo') return step.est ?? null;
  const elapsed = Math.max(0, now - (step.started || now));
  if (step.eta != null) return step.eta;
  if (step.frac > 0.03 && elapsed > 5) return (elapsed * (1 - step.frac)) / step.frac;
  if (step.est != null) return Math.max(step.est - elapsed, Math.min(60, step.est * 0.1));
  return null;
}

export function progressBoard(steps, opts = {}) {
  if (!steps?.length) return '';
  const now = opts.now || Date.now() / 1000;
  const skew = Date.now() / 1000 - now;               // client clock minus server clock
  const running = opts.state === 'running' || opts.state === 'cancelling';
  let left = 0;
  let unknown = false;
  const rows = steps.map((s) => {
    const rem = running ? remainingOf(s, now) : 0;
    if (rem == null) unknown = true; else left += rem;
    let time = '';
    if (s.state === 'done' || (s.state === 'error' && s.started)) time = fmtDur((s.ended || now) - (s.started || now));
    else if (s.state === 'busy') time = rem == null ? `đã ${fmtDur(now - s.started)}` : `còn ~${fmtDur(rem)}`;
    else if (s.state === 'todo' && running) time = s.est != null ? `~${fmtDur(s.est)}` : '';
    const over = s.state === 'busy' && s.est != null && s.eta == null && !(s.frac > 0.03) && now - s.started > s.est;
    const bar = s.state === 'busy' && s.frac != null ? `<span class="pb-mini"><i style="width:${Math.round(s.frac * 100)}%"></i></span>` : '';
    return `<li data-state="${esc(s.state)}"><span class="pb-dot" aria-hidden="true"></span>
      <span class="pb-label">${esc(s.label)}${s.note ? `<small>${esc(s.note)}</small>` : ''}${bar}</span>
      <span class="pb-time${over ? ' over' : ''}">${esc(time)}${over ? ' · lâu hơn dự kiến' : ''}</span></li>`;
  }).join('');
  const startedAt = opts.started || steps.find((s) => s.started)?.started || now;
  const elapsed = Math.max(0, (opts.ended || now) - startedAt);
  let head;
  let pct;
  if (running) {
    const done = steps.filter((s) => s.state === 'done').length;
    const where = `bước ${Math.min(done + 1, steps.length)}/${steps.length}`;
    if (unknown && left < 60) {
      // no estimate for what is left: say so instead of "a few seconds"; progress by finished steps
      const busy = steps.find((s) => s.state === 'busy');
      pct = Math.round((100 * (done + (busy?.frac || 0))) / steps.length);
      head = `<b>Đang chạy ${where}${busy ? `: ${esc(busy.label)}` : ''}</b>
        <small>đã chạy ${fmtDur(elapsed)} · chưa đủ dữ liệu để ước tính thời gian còn lại</small>`;
    } else {
      pct = left > 0 ? Math.min(99, Math.round((100 * elapsed) / (elapsed + left))) : 99;
      head = `<b>Còn khoảng ${unknown ? 'ít nhất ' : ''}${fmtDur(left)}</b>
        <small>đã chạy ${fmtDur(elapsed)} · ${where}${left > 0 ? ` · dự kiến xong lúc ${clock((now + left + skew) * 1000)}` : ''}</small>`;
    }
  } else if (opts.state === 'succeeded') {
    pct = 100;
    head = `<b>Hoàn tất</b><small>tổng thời gian ${fmtDur(elapsed)}</small>`;
  } else {
    pct = Math.round((100 * steps.filter((s) => s.state === 'done').length) / steps.length);
    const at = steps.find((s) => s.state === 'error' || s.state === 'busy');
    head = `<b>${opts.state === 'cancelled' ? 'Đã huỷ' : 'Dừng vì lỗi'}${at ? ` ở bước "${esc(at.label)}"` : ''}</b><small>sau ${fmtDur(elapsed)}</small>`;
  }
  return `<div class="pboard" data-state="${esc(opts.state || '')}">
    <div class="pb-top"><div class="pb-head">${head}</div><div class="pb-pct">${pct}%</div></div>
    <div class="pb-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct}"><i style="width:${pct}%"></i></div>
    <ol class="pb-steps">${rows}</ol>
    ${running ? '<p class="pb-foot">Thời gian là ước tính, tự cập nhật theo tốc độ thực tế. Có thể đóng trang, job vẫn chạy.</p>' : ''}
  </div>`;
}

window.__fflab = window.__fflab || {};
window.__fflab.progressBoard = progressBoard;
window.__fflab.fmtDur = fmtDur;
