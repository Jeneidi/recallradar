(() => {
  const select = document.querySelector("#quarter-select");
  const list = document.querySelector("#flag-list");
  const status = document.querySelector("#demo-status");
  const chart = document.querySelector("#backtest-chart");

  const formatScore = (value) => Number(value).toFixed(3);
  const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;

  function renderFlags(results, quarter) {
    list.replaceChildren();
    for (const item of results.demo_quarters[quarter] || []) {
      const row = document.createElement("div");
      row.className = "flag-row";
      const label = item.label ? "had a recall within 12 months" : "no recall within 12 months";
      row.innerHTML = `
<span class="flag-code">${item.code}</span>
<div class="flag-copy"><div class="flag-title"><b>Flagged</b><span class="flag-label">${label}</span></div>
<p class="flag-reason">${item.reason}</p>
<p class="data-note">${item.report_count.toLocaleString()} problem reports, ${pct(item.death_share)} involved a death</p></div>
<span class="flag-score">${formatScore(item.score)}<span>risk score</span></span>`;
      list.append(row);
    }
    status.textContent = `The 10 device types ranked riskiest in ${quarter}. Whether a recall followed is shown for checking only. The model never saw it.`;
  }

  function renderChart(results) {
    chart.replaceChildren();
    for (const row of results.backtest) {
      const wrapper = document.createElement("div");
      wrapper.className = "chart-row";
      wrapper.innerHTML = `
<span class="chart-label">${row.quarter}</span><div class="chart-bars"><div class="bar" title="RecallRadar: ${pct(row.model_precision_at_10)} of top 10 picks had a recall"><i style="width:${row.model_precision_at_10 * 100}%"></i></div><div class="bar volume" title="Number of reports: ${pct(row.volume_precision_at_10)} of top 10 picks had a recall"><i style="width:${row.volume_precision_at_10 * 100}%"></i></div></div>`;
      chart.append(wrapper);
    }
  }

  fetch("results.json").then((response) => {
    if (!response.ok) throw new Error(`results.json ${response.status}`);
    return response.json();
  }).then((results) => {
    for (const quarter of Object.keys(results.demo_quarters)) {
      const option = document.createElement("option");
      option.value = quarter;
      option.textContent = quarter;
      select.append(option);
    }
    select.addEventListener("change", () => renderFlags(results, select.value));
    renderFlags(results, select.value);
    renderChart(results);
  }).catch((error) => {
    status.textContent = "The saved results could not be loaded.";
    console.error(error);
  });
})();
