/* 모든 화면 오른쪽 아래의 작업 상태 표시. 누르면 학습 상태 화면으로 간다. */
(() => {
  const a = document.createElement("a");
  a.className = "train-badge";
  a.href = "/training";
  a.textContent = "학습 상태 확인 중";
  document.body.appendChild(a);
  async function tick() {
    try {
      const s = await (await fetch("/api/training/status", { cache: "no-store" })).json();
      const j = s.jobs[0];
      a.dataset.state = s.state;
      a.textContent = j
        ? `● ${j.job_name} 중${j.total ? ` ${j.done}/${j.total}` : ""}`
        : { stopped: "✖ 학습 중단됨", paused: "⏸ 학습 멈춤", idle: "○ 학습 안 함" }[s.state];
    } catch (e) { a.textContent = "학습 상태 알 수 없음"; a.dataset.state = "idle"; }
  }
  tick();
  setInterval(tick, 5000);
})();
