// 스캔 → 결과까지 진행 표시 (스캔 콘솔·분석 대시보드 공용).
// 서버 작업 상태(job.phases, job.eta_s)를 단계 표시와 남은 시간으로 그린다.
const ScanProgress = (() => {
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function dur(s) {
    s = Math.max(0, Math.round(s));
    return s >= 60 ? `${Math.floor(s / 60)}분 ${s % 60}초` : `${s}초`;
  }

  // job: /api/scan/status 의 job. opts.resultHref: 완료 뒤 결과 링크 (없으면 표시 안 함)
  function html(job, opts = {}) {
    const ph = job.phases || [];
    if (!ph.length) return "";
    const done = !job.busy;
    const failed = !!job.error || ph.some((p) => p.state === "failed");
    const took = ph.reduce((a, p) => a + p.took_s, 0);
    const total = done ? took : took + (job.eta_s || 0);
    const pct = done ? 100 : Math.min(99, (took / Math.max(total, 1)) * 100);

    let head;
    if (job.busy) head = `<b>결과까지 약 ${dur(job.eta_s || 0)} 남음</b><span>경과 ${dur(took)}</span>`;
    else if (failed) head = `<b>중단됨</b><span>${esc(job.error || job.message)}</span>`;
    else head = `<b>결과 준비 완료</b><span>총 ${dur(took)}</span>`;

    const steps = ph.map((p, i) => {
      const cnt = p.key === "capture" && job.total ? ` ${job.progress}/${job.total}장` : "";
      const time = p.state === "done" ? dur(p.took_s) : p.state === "active" ? `${dur(p.took_s)} 경과` : `약 ${dur(p.expected_s)}`;
      const icon = p.state === "done" ? "✓" : p.state === "failed" ? "!" : p.state === "active" ? "" : i + 1;
      // 다시 그려도 회전이 끊기지 않게 현재 시각만큼 애니메이션을 앞당긴다
      const spin = p.state === "active" ? ` style="animation-delay:-${Date.now() % 900}ms"` : "";
      return `<li class="sp-${p.state}"><i${spin}>${icon}</i><span>${esc(p.name)}${cnt}</span><small>${time}</small></li>`;
    }).join("");

    const detail = job.busy && job.message ? `<p class="sp-detail">${esc(job.message)}</p>` : "";
    const link = done && !failed && opts.resultHref ? `<a class="sp-go" href="${opts.resultHref}">결과 보기 →</a>` : "";
    return `<div class="sp ${done ? (failed ? "sp-failed" : "sp-finished") : "sp-running"}" role="status" aria-live="polite">
      <div class="sp-head">${head}</div>
      <div class="sp-bar"><span style="width:${pct}%"></span></div>
      <ol class="sp-steps">${steps}</ol>${detail}${link}
    </div>`;
  }

  return { html, dur };
})();
