// 합성 웨이퍼 맵 표시 (대시보드·스캔 콘솔 공용).
// 측정한 웨이퍼 원과 다이 격자를 맵 좌표에 그대로 겹쳐 그린다.
const WaferView = (() => {
  const GRID = 8;

  function geometry(m) {
    const [cx, cy] = m.wafer.center_px;
    const r = m.wafer.radius_px;
    const g = m.grid || null;
    const pitch = g ? g.pitch_px : null;
    // measure_grid 의 각도는 cv2 기준(반시계 양수)으로 격자를 곧게 펴는 회전이다.
    // 화면 좌표(y 아래)에서 격자축은 그 각도만큼 시계 방향으로 돌아간 벡터가 된다.
    const a = g ? (-g.angle_deg * Math.PI) / 180 : 0;
    return { cx, cy, r, pitch, cos: Math.cos(a), sin: Math.sin(a) };
  }

  // 다이 (row, col) 좌표계 u,v (다이 칸 단위, 웨이퍼 중심 0) → 맵 픽셀
  function toPx(geo, u, v) {
    const x = u * geo.pitch, y = v * geo.pitch;
    return [geo.cx + x * geo.cos + y * geo.sin, geo.cy - x * geo.sin + y * geo.cos];
  }

  function draw(host, m, opts = {}) {
    const geo = geometry(m);
    const [W, H] = m.size_px;
    const margin = geo.r * 1.05;
    const vb = `${geo.cx - margin} ${geo.cy - margin} ${margin * 2} ${margin * 2}`;
    const sw = geo.r / 360;
    let lines = "";
    if (opts.grid && geo.pitch) {
      const n = Math.ceil(geo.r / geo.pitch) + 1;
      for (let i = -n; i <= n; i++) {
        const a = toPx(geo, i, -n), b = toPx(geo, i, n), c = toPx(geo, -n, i), d = toPx(geo, n, i);
        lines += `<path d="M${a}L${b}M${c}L${d}"/>`;
      }
    }
    const id = `wv${Math.random().toString(36).slice(2, 8)}`;
    host.classList.add("wafer-view");
    host.innerHTML = `<svg viewBox="${vb}" role="img" aria-label="합성 웨이퍼 맵">
      <defs><clipPath id="${id}"><circle cx="${geo.cx}" cy="${geo.cy}" r="${geo.r}"/></clipPath></defs>
      <rect x="${geo.cx - margin}" y="${geo.cy - margin}" width="${margin * 2}" height="${margin * 2}" fill="#101114"/>
      <image href="${m.url}" x="0" y="0" width="${W}" height="${H}" opacity="${opts.outer === false ? 0 : 0.35}"/>
      <image href="${m.url}" x="0" y="0" width="${W}" height="${H}" clip-path="url(#${id})"/>
      <g clip-path="url(#${id})" stroke="#4da3ff" stroke-opacity=".5" stroke-width="${sw}" fill="none">${lines}</g>
      <circle cx="${geo.cx}" cy="${geo.cy}" r="${geo.r}" fill="none" stroke="#fff" stroke-width="${sw * 1.6}"/>
    </svg>`;
  }

  function cropDie(canvas, m, row, col, grid = GRID) {
    const geo = geometry(m);
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      const [px, py] = toPx(geo, col - grid / 2 + 0.5, row - grid / 2 + 0.5);
      const k = canvas.width / (geo.pitch * 1.25);
      ctx.fillStyle = "#101114";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.scale(k, k);
      ctx.rotate(Math.atan2(geo.sin, geo.cos));
      ctx.translate(-px, -py);
      ctx.drawImage(img, 0, 0);
      ctx.restore();
    };
    img.src = m.url;
  }

  return { draw, cropDie };
})();
