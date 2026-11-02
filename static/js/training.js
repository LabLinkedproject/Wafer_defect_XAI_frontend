/* 학습 상태 화면. 3초마다 다시 읽고, 서버 응답이 끊기면 그 사실을 그대로 표시한다. */
(() => {
  const $ = (id) => document.getElementById(id);
  const TITLES = {
    running: "지금 학습 중입니다", stalled: "멈춘 것 같습니다", stopped: "학습이 중단되었습니다",
    idle: "지금은 학습하지 않습니다", offline: "상태를 읽을 수 없습니다",
  };

  function duration(sec) {
    if (sec == null) return "-";
    if (sec < 60) return `${sec}초`;
    const m = Math.round(sec / 60);
    return m < 60 ? `${m}분` : `${Math.floor(m / 60)}시간 ${m % 60}분`;
  }

  function lineChart(host, series, opts) {
    const W = 440, H = 190, L = 40, R = 14, T = 10, B = 24;
    const xs = series[0].values.map((v) => v.x);
    const all = series.flatMap((s) => s.values.map((v) => v.y));
    const xMax = Math.max(opts.xMax || 1, ...xs);
    let lo = opts.min ?? Math.min(...all), hi = opts.max ?? Math.max(...all);
    if (hi === lo) hi = lo + 1;
    const X = (x) => L + (x / xMax) * (W - L - R), Y = (y) => H - B - ((y - lo) / (hi - lo)) * (H - T - B);
    const ticks = [0, 0.5, 1].map((t) => lo + (hi - lo) * t);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.label}">`;
    for (const t of ticks) {
      svg += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--grid)"/>`;
      svg += `<text x="${L - 6}" y="${Y(t) + 4}" text-anchor="end">${t.toFixed(2)}</text>`;
    }
    svg += `<text x="${W - R}" y="${H - 6}" text-anchor="end">${xMax} 에폭</text><text x="${L}" y="${H - 6}">0</text>`;
    for (const s of series) {
      const d = s.values.map((v, i) => `${i ? "L" : "M"}${X(v.x).toFixed(1)} ${Y(v.y).toFixed(1)}`).join("");
      svg += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
      const e = s.values[s.values.length - 1];
      svg += `<circle cx="${X(e.x)}" cy="${Y(e.y)}" r="4" fill="${s.color}" stroke="#fff" stroke-width="2"/>`;
    }
    svg += `</svg><div class="tip" hidden></div>`;
    if (series.length > 1) {
      svg += `<div class="legend">${series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("")}</div>`;
    }
    host.innerHTML = svg;
    const el = host.querySelector("svg"), tip = host.querySelector(".tip");
    el.onmousemove = (ev) => {
      const box = el.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * W;
      let i = 0;
      xs.forEach((x, k) => { if (Math.abs(X(x) - px) < Math.abs(X(xs[i]) - px)) i = k; });
      tip.hidden = false;
      tip.textContent = `에폭 ${xs[i]}: ` + series.map((s) => `${s.name} ${s.values[i].y.toFixed(3)}`).join(" · ");
      tip.style.left = `${(X(xs[i]) / W) * box.width}px`;
      tip.style.top = `${(Y(series[0].values[i].y) / H) * box.height}px`;
    };
    el.onmouseleave = () => { tip.hidden = true; };
  }

  function renderCharts(s) {
    const h = s.history || [];
    $("charts").hidden = h.length === 0;
    if (!h.length) return;
    const xMax = s.epochs_total || h[h.length - 1].epoch;
    lineChart($("chartLoss"), [{ name: "손실", color: "var(--series-1)", values: h.map((r) => ({ x: r.epoch, y: r.loss })) }],
      { label: "에폭별 학습 손실", xMax });
    const keys = Object.keys(h[0]).filter((k) => k.startsWith("val_"));
    const NAMES = { val_full_f1: "전면 맵", val_die_f1: "44다이 2×2 구역", val_f1: "검증" };
    const colors = ["var(--series-1)", "var(--series-2)"];
    lineChart($("chartF1"), keys.map((k, i) => ({ name: NAMES[k] || k, color: colors[i % 2], values: h.map((r) => ({ x: r.epoch, y: r[k] })) })),
      { label: "에폭별 검증 Macro-F1", xMax, min: 0, max: 1 });
  }

  const fmt = (v) => (v == null ? "-" : v.toFixed(3));
  const range = (r) => (r ? ` <span class="ci">${r[0].toFixed(3)}~${r[1].toFixed(3)}</span>` : "");

  function renderMetrics(m) {
    $("metricsPanel").hidden = !m;
    if (!m) return;
    const tests = m.tests || { [m.dataset]: { size: m.test_size, conditions: m.conditions } };
    $("metricsNote").textContent = "공개 데이터 기준 값이며 실물 아크릴 웨이퍼 정확도가 아닙니다. 범위는 95 % 신뢰구간입니다.";
    $("metricsBody").innerHTML = Object.entries(tests).map(([name, t]) => {
      const rows = Object.entries(t.conditions).map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v.macro_f1)}${range(v.macro_ci)}</td>
        <td>${fmt(v.exact_match)}${range(v.exact_ci)}</td><td>${fmt(v.normal_correct)}</td></tr>`).join("");
      return `<h3>${name} · ${t.size.toLocaleString()}장</h3><div class="scroll"><table>
        <tr><th>입력</th><th>Macro-F1</th><th>완전 일치</th><th>정상 정답률</th></tr>${rows}</table></div>`;
    }).join("");
  }

  function jobCard(j) {
    const pct = j.total ? (j.done / j.total) * 100 : 0;
    return `<section class="panel">
      <div class="row-between"><b>${j.job_name}</b><span class="num">${j.total ? `${j.done} / ${j.total} ${j.unit || ""}` : ""}</span></div>
      <div class="track"><div class="fill" style="width:${pct}%"></div></div>
      <dl class="facts">
        <div><dt>경과</dt><dd>${duration(j.elapsed_seconds)}</dd></div>
        <div><dt>남은 시간</dt><dd>${j.eta_seconds != null ? `약 ${duration(j.eta_seconds)}` : "계산 중"}</dd></div>
        <div><dt>CPU 사용</dt><dd>${j.cpu_percent != null ? `${j.cpu_percent}%` : "-"}</dd></div>
        <div><dt>마지막 갱신</dt><dd>${j.last_update_age != null ? `${duration(j.last_update_age)} 전` : "-"}</dd></div>
      </dl>
      ${j.detail ? `<p class="muted detail">${j.detail}</p>` : ""}
    </section>`;
  }

  function headline(s) {
    if (s.state === "running") return s.training ? TITLES.running : `지금 ${s.jobs[0].job_name} 중입니다`;
    return TITLES[s.state];
  }

  function render(s) {
    $("lamp").dataset.state = s.state;
    $("lampTitle").textContent = headline(s);
    document.title = `${s.jobs.length ? `● ${s.jobs[0].job_name} 중` : "○ 작업 없음"} · 학습 상태`;
    $("lampSub").textContent = s.jobs.length
      ? s.jobs.map((j) => `${j.job_name}${j.total ? ` ${j.done}/${j.total} ${j.unit || ""}` : ""}`).join(" · ")
      : s.message || "";
    $("jobs").innerHTML = s.jobs.map(jobCard).join("");
    renderCharts(s);
    renderMetrics(s.metrics);
    $("clock").textContent = `${new Date().toLocaleTimeString("ko-KR")} 확인`;
  }

  async function tick() {
    try {
      render(await (await fetch("/api/training/status", { cache: "no-store" })).json());
    } catch (e) {
      $("lamp").dataset.state = "offline";
      $("lampTitle").textContent = TITLES.offline;
      $("lampSub").textContent = "상태 서버와 연결이 끊겼습니다. 학습 자체가 멈췄다는 뜻은 아닙니다.";
    }
  }
  tick();
  setInterval(tick, 3000);
})();
