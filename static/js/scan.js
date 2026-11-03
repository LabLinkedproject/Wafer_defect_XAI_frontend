// 웨이퍼 스캔 콘솔
(() => {
  const $ = (id) => document.getElementById(id);
  let plan = null;
  let status = null;
  let pollTimer = null;

  async function api(url, body) {
    const opt = body === undefined ? {} : {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    };
    const r = await fetch(url, opt);
    const data = await r.json().catch(() => ({}));
    if (!r.ok || data.success === false) {
      showBanner(data.error || `요청 실패 (${r.status})`);
      throw new Error(data.error || r.status);
    }
    return data;
  }

  function showBanner(msg, info = false) {
    const b = $("banner");
    b.textContent = msg;
    b.className = "banner" + (info ? " info" : "");
    b.hidden = !msg;
  }

  function chip(el, state, text) {
    el.className = "chip " + state;
    el.textContent = text;
  }

  // ── 모자이크 (44다이 배치) ───────────────────────────────────
  function buildMosaic() {
    const m = $("mosaic");
    const n = Math.max(...plan.order.map((d) => Math.max(d.row, d.col))) + 1;
    m.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
    m.innerHTML = "";
    const byPos = new Map(plan.order.map((d) => [`${d.row},${d.col}`, d]));
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const d = byPos.get(`${r},${c}`);
        const cell = document.createElement("div");
        cell.className = d ? "die" : "die off";
        if (d) {
          cell.id = `die-${d.id}`;
          cell.title = `다이 ${d.id} · 행 ${r + 1}, 열 ${c + 1} · 갠트리 (${d.gx}, ${d.gy}) mm`;
          cell.innerHTML = `<span class="id">${d.id}</span>`;
        }
        m.appendChild(cell);
      }
    }
  }

  function fillMosaic(results, currentId) {
    document.querySelectorAll(".die.current").forEach((e) => e.classList.remove("current"));
    for (const r of results) {
      const cell = $(`die-${r.id}`);
      if (!cell || cell.dataset.file === r.file) continue;
      cell.dataset.file = r.file;
      cell.classList.add("done");
      cell.innerHTML = `<a href="/scans/${r.file}" target="_blank" rel="noopener">
        <img src="/scans/${r.thumb}" alt="다이 ${r.id} 촬영 이미지" loading="lazy"></a>
        <span class="id">${r.id}</span>`;
    }
    if (currentId) $(`die-${currentId}`)?.classList.add("current");
  }

  function clearMosaic() {
    document.querySelectorAll(".die.done").forEach((cell) => {
      cell.classList.remove("done");
      delete cell.dataset.file;
      cell.innerHTML = `<span class="id">${cell.id.replace("die-", "")}</span>`;
    });
  }

  // ── 미니맵 (갠트리 범위 · 웨이퍼 · 현재 위치) ────────────────
  function drawMinimap() {
    if (!status || !plan) return;
    const svg = $("minimap");
    const [xl, xh] = status.limits.x, [yl, yh] = status.limits.y;
    const W = 220, H = 190, pad = 10;
    const s = Math.min((W - 2 * pad) / (xh - xl), (H - 2 * pad) / (yh - yl));
    const X = (x) => pad + (x - xl) * s;
    const Y = (y) => H - pad - (y - yl) * s;      // 위쪽이 +y
    const cfg = status.config;
    const cx = cfg.wafer_center_x, cy = cfg.wafer_center_y;
    const R = plan.spec.diameter_mm / 2;
    const b = plan.bounds_mm;
    const p = status.position;
    let html = `<rect x="${X(xl)}" y="${Y(yh)}" width="${(xh - xl) * s}" height="${(yh - yl) * s}"
                 fill="none" stroke="#c9d1da" stroke-dasharray="3 3"/>
      <circle cx="${X(cx)}" cy="${Y(cy)}" r="${R * s}" fill="#eef4fc" stroke="#9db8dc"/>
      <rect x="${X(cx - b)}" y="${Y(cy + b)}" width="${2 * b * s}" height="${2 * b * s}"
            fill="none" stroke="#2c3e50" stroke-width="1.5"/>`;
    for (const d of plan.order) {
      html += `<circle cx="${X(d.gx)}" cy="${Y(d.gy)}" r="1.4" fill="${d.in_limits ? "#7f8c8d" : "#e74c3c"}"/>`;
    }
    html += `<line x1="${X(0)}" y1="${Y(yl)}" x2="${X(0)}" y2="${Y(yh)}" stroke="#e1e5ea"/>
             <line x1="${X(xl)}" y1="${Y(0)}" x2="${X(xh)}" y2="${Y(0)}" stroke="#e1e5ea"/>`;
    if (p) {
      html += `<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="4.5" fill="#f39c12" stroke="#fff" stroke-width="1.5">
                 <title>카메라 (${p.x.toFixed(1)}, ${p.y.toFixed(1)}) mm</title></circle>`;
    }
    svg.innerHTML = html;
  }

  // ── 상태 반영 ────────────────────────────────────────────────
  function render() {
    const s = status;
    chip($("chipGpio"), s.gpio_access ? "ok" : "bad", s.gpio_access ? "GPIO 권한" : "GPIO 권한 없음");
    chip($("chipOrigin"), s.origin_set ? "ok" : "bad", s.origin_set ? "원점 지정됨" : "원점 미지정");
    const cal = s.calibrated && s.calibrated.x && s.calibrated.y;
    chip($("chipCal"), cal ? "ok" : "warn", cal ? "거리 보정됨" : "거리 미보정");
    $("chipDry").hidden = !s.dry_run;

    if (!s.gpio_access) {
      showBanner("모터를 움직이려면 GPIO 권한이 필요합니다. 터미널에서 한 번만 실행하세요: " +
                 "sudo /home/orangepi/motor_test/install_gpio_access.sh  (그 뒤 이 서버를 재시작)");
    } else if (!s.origin_set) {
      showBanner("원점이 없습니다. 갠트리를 손으로 중앙에 두고 터미널에서 'gantry.sh set-center --micro 8' 을 실행하세요.");
    } else if (s.job.error) {
      showBanner(s.job.error);
    } else if (!plan?.all_in_limits) {
      showBanner("일부 다이 위치가 갠트리 범위 밖입니다. 웨이퍼 중심 설정을 확인하세요.");
    } else {
      showBanner("");
    }

    const p = s.position;
    $("posLine").textContent = p ? `X ${p.x.toFixed(2)}  Y ${p.y.toFixed(2)} mm` : "X —  Y —";

    const j = s.job;
    const busy = j.busy;
    const pct = j.total ? Math.round((100 * j.progress) / j.total) : (busy ? 0 : 0);
    $("bar").style.width = `${pct}%`;
    $("jobMsg").textContent = j.message || "";
    $("jobCount").textContent = j.total ? `${j.progress}/${j.total}` : "";

    const ready = s.gpio_access && s.origin_set && !busy;
    for (const id of ["btnScan", "btnBounds", "btnOrigin", "btnWaferCenter", "btnSetCenter", "btnSave"]) {
      $(id).disabled = !ready;
    }
    $("btnScan").disabled = !ready || !plan?.all_in_limits;
    document.querySelectorAll(".jog").forEach((b) => (b.disabled = !ready));
    $("btnStop").disabled = !busy;

    // 결과
    let currentId = null;
    if (busy && j.name === "scan") {
      const m = /다이 (\d+)/.exec(j.message || "");
      if (m) currentId = Number(m[1]);
    }
    fillMosaic(s.scan.results, currentId);
    renderMap(s, ready);

    const c = $("corners");
    if (s.scan.preview.length && c.dataset.n !== String(s.scan.preview.length)) {
      c.dataset.n = s.scan.preview.length;
      c.innerHTML = s.scan.preview.map((p) => `<figure>
        <a href="/scans/${p.file}" target="_blank" rel="noopener"><img src="/scans/${p.thumb}" alt="${p.name} 모서리"></a>
        <figcaption>${p.name} (${p.gx}, ${p.gy})</figcaption></figure>`).join("");
    }
    drawMinimap();
  }

  // ── 웨이퍼 맵 ────────────────────────────────────────────────
  let mapKey = null;
  function renderMap(s, ready) {
    const m = s.wafer_map;
    $("btnMap").disabled = !(s.origin_set !== undefined && !s.job.busy);
    $("btnAlign").disabled = !(ready && m?.wafer);
    if (!m) return;
    const key = `${m.dir}|${m.seconds}|${$("mapGrid").checked}`;
    if (key === mapKey) return;
    mapKey = key;
    $("mapEmpty").hidden = true;
    $("mapMeta").textContent = `${m.dir} · ${m.images}장 합성 · ${m.seconds}초`;
    $("btnAnalyze").href = `/analysis?scan=${encodeURIComponent(m.dir)}`;
    $("btnAnalyze").hidden = !m.wafer;
    if (m.wafer) WaferView.draw($("mapView"), m, { grid: $("mapGrid").checked });
    else $("mapView").innerHTML = `<img src="${m.url}" alt="합성 웨이퍼 맵" style="width:100%">`;
    const w = m.wafer, g = m.grid, c = m.camera;
    const rows = [["카메라 축척", `${c.px_per_mm_full} px/mm`], ["카메라 회전", `${c.rotation_deg}°`], ["시야", `${c.fov_mm[0]} × ${c.fov_mm[1]} mm`],
                  ["겹침 일치도", `${m.overlap_ncc.before} → ${m.overlap_ncc.after}`]];
    if (w) {
      rows.push(["웨이퍼 커버리지", `${m.coverage_pct}%`], ["웨이퍼 중심", `${w.center_gantry_mm[0]}, ${w.center_gantry_mm[1]} mm`],
                ["웨이퍼 반지름", `${w.radius_gantry_mm} mm`], ["원 맞춤 오차", `${w.fit_residual_px} px`]);
    }
    if (g) rows.push(["격자 피치", `${g.pitch_mm ?? "-"} mm`], ["격자 기울기", `${g.angle_deg}°`]);
    $("mapFacts").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    $("mapHint").textContent = w && m.coverage_pct < 98
      ? "웨이퍼 일부만 담겼습니다. '측정값으로 정렬'을 누른 뒤 다시 스캔하면 전체를 담습니다."
      : "이음새 없이 합성됐습니다.";
  }

  function renderConfig() {
    const cfg = status.config;
    $("cfgRpm").value = cfg.rpm;
    $("cfgDwell").value = cfg.dwell_s;
    $("cfgPitch").value = cfg.pitch_mm ?? plan.spec.pitch_mm;
    $("cfgFlipX").checked = cfg.wafer_sign_x < 0;
    $("cfgFlipY").checked = cfg.wafer_sign_y < 0;
    const auto = cfg.exposure === null;
    $("expoAuto").checked = auto;
    $("expo").disabled = auto;
    if (!auto) { $("expo").value = cfg.exposure; $("expoVal").textContent = cfg.exposure; }
  }

  function renderPlan() {
    const sp = plan.spec;
    $("specLine").textContent =
      `${sp.diameter_mm} mm · 다이 ${sp.die_count}개 · 피치 ${sp.pitch_mm} mm · 스캔 ${plan.order.length}지점`;
    $("pitchHint").textContent = sp.count_ok ? "" :
      `현재 피치로는 다이가 ${sp.die_count}개입니다 (기대 ${sp.expected_dies}). 실측 피치를 확인하세요.`;
    $("boundsInfo").textContent = `웨이퍼 중심 ±${plan.bounds_mm} mm`;
    // 예상 시간: 이동(대략) + 대기
    const dwell = status ? status.config.dwell_s : 0.6;
    const est = Math.round(plan.order.length * (dwell + 0.9) + 60);   // 이동 + 대기 + 맵 합성
    $("eta").textContent = `예상 약 ${Math.floor(est / 60)}분 ${est % 60}초`;
  }

  async function refreshPlan() {
    plan = await api("/api/scan/plan");
    renderPlan();
    buildMosaic();
  }

  async function poll() {
    try {
      status = await api("/api/scan/status");
      render();
    } catch (e) { /* 배너로 표시됨 */ }
    clearTimeout(pollTimer);
    pollTimer = setTimeout(poll, status?.job.busy ? 400 : 1500);
  }

  async function act(url, body, confirmMsg) {
    if (confirmMsg && !confirm(confirmMsg)) return;
    try {
      await api(url, body ?? {});
      setTimeout(poll, 150);
    } catch (e) { /* 배너 */ }
  }

  // ── 이벤트 ───────────────────────────────────────────────────
  $("btnScan").onclick = () => {
    clearMosaic();
    act("/api/scan/start", {}, `${plan.order.length}지점 스캔을 시작합니다. 갠트리 주변에 손이나 물건이 없는지 확인하세요.`);
  };
  $("btnStop").onclick = () => act("/api/gantry/stop", {});
  $("btnMap").onclick = () => act("/api/scan/map", {});
  $("mapGrid").onchange = () => { mapKey = null; if (status) renderMap(status, !status.job.busy); };
  $("btnAlign").onclick = async () => {
    if (!confirm("맵에서 측정한 웨이퍼 중심과 크기로 스캔 경로를 다시 만듭니다.")) return;
    try {
      await api("/api/scan/align", {});
      await refreshPlan();
      poll();
      showBanner(`정렬했습니다. 다음 스캔은 ${plan.order.length}장으로 웨이퍼 전체를 담습니다.`, true);
    } catch (e) { /* 배너 */ }
  };
  $("btnBounds").onclick = () => { $("corners").innerHTML = ""; $("corners").dataset.n = "0"; act("/api/scan/bounds", {}); };
  $("btnOrigin").onclick = () => act("/api/gantry/origin", {});
  $("btnWaferCenter").onclick = () => act("/api/gantry/wafer-center", {});
  $("btnSetCenter").onclick = async () => {
    if (!confirm("지금 카메라 위치를 웨이퍼 중심으로 지정합니다. 십자선이 웨이퍼 중심에 있나요?")) return;
    try {
      await api("/api/scan/set-center", {});
      await refreshPlan();
      poll();
      showBanner("웨이퍼 중심을 저장했습니다. '네 모서리 돌며 촬영'으로 범위를 확인해 보세요.", true);
    } catch (e) { /* 배너 */ }
  };
  document.querySelectorAll(".jog").forEach((b) => {
    b.onclick = () => act("/api/gantry/jog", { axis: b.dataset.axis, mm: Number(b.dataset.dir) * Number($("stepSel").value) });
  });
  $("btnSave").onclick = async () => {
    try {
      await api("/api/scan/config", {
        rpm: Number($("cfgRpm").value),
        dwell_s: Number($("cfgDwell").value),
        pitch_mm: $("cfgPitch").value,
        wafer_sign_x: $("cfgFlipX").checked ? -1 : 1,
        wafer_sign_y: $("cfgFlipY").checked ? -1 : 1,
      });
      await refreshPlan();
      poll();
      showBanner("설정을 저장했습니다.", true);
    } catch (e) { /* 배너 */ }
  };
  let expoTimer = null;
  $("expo").oninput = () => {
    $("expoVal").textContent = $("expo").value;
    clearTimeout(expoTimer);
    expoTimer = setTimeout(() => api("/api/scan/config", { exposure: Number($("expo").value) }).catch(() => {}), 250);
  };
  $("expoAuto").onchange = () => {
    $("expo").disabled = $("expoAuto").checked;
    api("/api/scan/config", { exposure: $("expoAuto").checked ? "auto" : Number($("expo").value) }).catch(() => {});
  };

  // ── 시작 ─────────────────────────────────────────────────────
  (async () => {
    status = await api("/api/scan/status");
    await refreshPlan();
    renderConfig();
    render();
    poll();
  })();
})();
