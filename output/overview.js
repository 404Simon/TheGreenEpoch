// ── State ─────────────────────────────────────────────────────────
let rawData = [];
let filteredData = [];
let charts = {};
let filters = {};
let filterTimer = null;

// ── Field config ──────────────────────────────────────────────────
const FIELDS = [
  {key:'scenario_description', label:'Scenario', fmt:'text'},
  {key:'model', label:'Model', fmt:'text'},
  {key:'region', label:'Region', fmt:'text'},
  {key:'threshold', label:'Threshold', fmt:'num', dec:0},
  {key:'hysteresis_margin', label:'Hysteresis', fmt:'num', dec:0},
  {key:'num_pauses', label:'Pauses', fmt:'num', dec:0},
  {key:'actual_overhead_pct', label:'Overhead %', fmt:'pct', dec:1},
  {key:'co2_savings_pct', label:'CO\u2082 Save %', fmt:'kpct', dec:2},
  {key:'score', label:'Score', fmt:'score', dec:4},
  {key:'total_emissions_kgco2', label:'Emissions (kg)', fmt:'num', dec:0},
  {key:'baseline_emissions_kgco2', label:'Base Em. (kg)', fmt:'num', dec:0},
  {key:'total_wall_time_h', label:'Wall (h)', fmt:'num', dec:1},
  {key:'baseline_time_h', label:'Base Time (h)', fmt:'num', dec:1},
  {key:'training_time_h', label:'Train (h)', fmt:'num', dec:1},
  {key:'paused_time_h', label:'Paused (h)', fmt:'num', dec:1},
  {key:'total_energy_kwh', label:'Energy (kWh)', fmt:'num', dec:0},
  {key:'completed', label:'Done', fmt:'bool'},
  {key:'within_overhead_budget', label:'Budget', fmt:'bool'},
  {key:'stop_reason', label:'Stop', fmt:'text'},
  {key:'issues', label:'Issues', fmt:'text'},
];

// ── CSV parsing ───────────────────────────────────────────────────
function parseCSV(text) {
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true, dynamicTyping: true });
  return parsed.data.map(row => {
    row.threshold = +row.threshold;
    row.hysteresis_margin = +row.hysteresis_margin;
    row.num_pauses = +row.num_pauses;
    row.actual_overhead_pct = +row.actual_overhead_pct;
    row.co2_savings_pct = +row.co2_savings_pct;
    row.score = +row.score;
    row.total_emissions_kgco2 = +row.total_emissions_kgco2;
    row.baseline_emissions_kgco2 = +row.baseline_emissions_kgco2;
    row.total_wall_time_h = +row.total_wall_time_h;
    row.baseline_time_h = +row.baseline_time_h;
    row.training_time_h = +row.training_time_h;
    row.paused_time_h = +row.paused_time_h;
    row.total_energy_kwh = +row.total_energy_kwh;
    row.tokens_processed = +row.tokens_processed;
    row.tokens_total = +row.tokens_total;
    row.completed = row.completed === true || row.completed === 'True' || row.completed === 'TRUE';
    row.within_overhead_budget = row.within_overhead_budget === true || row.within_overhead_budget === 'True' || row.within_overhead_budget === 'TRUE';
    if (!row.issues) row.issues = '';
    return row;
  });
}

// ── Format helpers ────────────────────────────────────────────────
function fmtNum(v, d=0) {
  if (v == null || isNaN(v)) return '-';
  return Number(v).toLocaleString(undefined, {minimumFractionDigits:d, maximumFractionDigits:d});
}
function fmtPct(v, d=1) {
  if (v == null || isNaN(v)) return '-';
  return (v >= 0 ? '+' : '') + v.toFixed(d) + '%';
}
function valClass(v) {
  if (v == null || isNaN(v)) return 'zero';
  if (v > 0.01) return 'pos';
  if (v < -0.01) return 'neg';
  return 'zero';
}

// ── Summary cards ────────────────────────────────────────────────
function renderCards(data) {
  const n = data.length;
  const avgSave = d3(data, r => r.co2_savings_pct);
  const avgScore = d3(data, r => r.score);
  const best = data.reduce((a,b) => (a.score > b.score ? a : b), data[0]);
  const worst = data.reduce((a,b) => (a.co2_savings_pct < b.co2_savings_pct ? a : b), data[0]);
  const totalPauses = data.reduce((s,r) => s + r.num_pauses, 0);
  const totalEm = data.reduce((s,r) => s + r.total_emissions_kgco2, 0);
  const totalBaseEm = data.reduce((s,r) => s + r.baseline_emissions_kgco2, 0);
  const totalSave = totalBaseEm > 0 ? (totalBaseEm - totalEm) / totalBaseEm * 100 : 0;

  document.getElementById('summaryCards').innerHTML = `
    <div class="card">
      <div class="accent-bar bar-blue"></div>
      <div class="label">Total Runs</div>
      <div class="value">${n}</div>
    </div>
    <div class="card">
      <div class="accent-bar bar-green"></div>
      <div class="label">Avg CO\u2082 Savings</div>
      <div class="value ${valClass(avgSave)}">${fmtPct(avgSave, 2)}</div>
      <div class="sub">Aggregate: ${fmtPct(totalSave, 2)}</div>
    </div>
    <div class="card">
      <div class="accent-bar bar-purple"></div>
      <div class="label">Avg Score</div>
      <div class="value ${valClass(avgScore)}">${avgScore.toFixed(4)}</div>
      <div class="sub">Best: ${best.scenario_description} (${best.score.toFixed(4)})</div>
    </div>
    <div class="card">
      <div class="accent-bar ${totalSave >= 0 ? 'bar-green' : 'bar-red'}"></div>
      <div class="label">Total CO\u2082 Saved</div>
      <div class="value ${valClass(totalSave)}">${fmtPct(totalSave, 2)}</div>
      <div class="sub">${fmtNum(totalEm,0)} vs ${fmtNum(totalBaseEm,0)} kg (policy vs base)</div>
    </div>
    <div class="card">
      <div class="accent-bar bar-orange"></div>
      <div class="label">Total Pauses</div>
      <div class="value">${fmtNum(totalPauses,0)}</div>
    </div>
    <div class="card">
      <div class="accent-bar ${worst.co2_savings_pct >= 0 ? 'bar-green' : 'bar-red'}"></div>
      <div class="label">Worst CO\u2082 Savings</div>
      <div class="value ${valClass(worst.co2_savings_pct)}">${fmtPct(worst.co2_savings_pct, 2)}</div>
      <div class="sub">${worst.scenario_description} (\u03B8=${worst.threshold})</div>
    </div>
  `;
}

// ── Filtering ─────────────────────────────────────────────────────
function fmtCell(v, fmt, dec) {
  if (v == null || v === '' || (fmt !== 'text' && isNaN(v))) return '\u2014';
  switch (fmt) {
    case 'num': return fmtNum(v, dec ?? 0);
    case 'pct': return fmtPct(v, dec ?? 1);
    case 'kpct': return fmtPct(v, 2);
    case 'score': return Number(v).toFixed(dec ?? 4);
    case 'bool': return v ? '\u2713 Yes' : '\u2717 No';
    default: return String(v);
  }
}

function uniqVals(data, key, fmt, dec) {
  const map = new Map();
  data.forEach(r => {
    const f = fmtCell(r[key], fmt, dec);
    if (!map.has(f)) map.set(f, r[key]);
  });
  return [...map.entries()].sort((a, b) => {
    const va = a[1], vb = b[1];
    if (va == null) return 1; if (vb == null) return -1;
    if (!isNaN(va) && !isNaN(vb)) return va - vb;
    return String(va).localeCompare(String(vb));
  }).map(e => e[0]);
}

function applyFilters(data) {
  return data.filter(row => FIELDS.every(f => {
    const fv = filters[f.key];
    if (!fv) return true;
    return fmtCell(row[f.key], f.fmt, f.dec) === fv;
  }));
}

function onFilterChange() {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(() => {
    document.querySelectorAll('.filter-select').forEach(sel => {
      filters[sel.dataset.key] = sel.value;
    });
    filteredData = applyFilters(rawData);
    renderCards(filteredData);
    renderCharts(filteredData);
    renderTable(filteredData);
    document.getElementById('rowCount').textContent =
      `${filteredData.length} / ${rawData.length} run${rawData.length !== 1 ? 's' : ''}`;
  }, 100);
}

function d3(arr, fn) {
  const vals = arr.map(fn).filter(v => v != null && !isNaN(v));
  return vals.reduce((s,v) => s+v, 0) / (vals.length || 1);
}

// ── Charts ────────────────────────────────────────────────────────
function destroyCharts() {
  Object.values(charts).forEach(c => { try { c.destroy(); } catch(e) {} });
  charts = {};
}

function colorForSave(v) {
  if (v > 5) return '#00c9a7';
  if (v > 0) return '#7ddbc6';
  if (v > -5) return '#ff9f43';
  return '#ff5e7a';
}

function renderCharts(data) {
  destroyCharts();
  const labels = data.map((r,i) => `${r.region} \u03B8=${r.threshold} (#${i+1})`);
  const saves = data.map(r => +r.co2_savings_pct);
  const scores = data.map(r => +r.score);
  const overheads = data.map(r => +r.actual_overhead_pct);

  const saveColors = saves.map(v => colorForSave(v));
  const scoreColors = scores.map(v => v > 0 ? '#00c9a7' : '#ff5e7a');

  const common = { responsive: true, maintainAspectRatio: true };

  charts.savings = new Chart(document.getElementById('chartSavings'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'CO\u2082 Savings %',
        data: saves,
        backgroundColor: saveColors,
        borderRadius: 4,
      }]
    },
    options: {
      ...common,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: ctx => {
              const r = data[ctx.dataIndex];
              return `Region: ${r.region}  |  Threshold: ${r.threshold}  |  Overhead: ${r.actual_overhead_pct.toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        y: {
          title: { display: true, text: 'CO\u2082 Savings %' },
        },
        x: { ticks: { display: false } }
      }
    }
  });

  charts.score = new Chart(document.getElementById('chartScore'), {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Score',
        data: scores,
        backgroundColor: scoreColors,
        borderRadius: 4,
      }]
    },
    options: {
      ...common,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: ctx => {
              const r = data[ctx.dataIndex];
              return `Savings: ${r.co2_savings_pct.toFixed(2)}%  |  Overhead: ${r.actual_overhead_pct.toFixed(1)}%`;
            }
          }
        }
      },
      scales: {
        y: {
          title: { display: true, text: 'Score' },
        },
        x: { ticks: { display: false } }
      }
    }
  });

  const done = data.filter(r => r.completed);
  charts.scatter = new Chart(document.getElementById('chartScatter'), {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Run',
        data: done.map(r => ({ x: r.actual_overhead_pct, y: r.co2_savings_pct })),
        backgroundColor: done.map(r => colorForSave(r.co2_savings_pct)),
        pointRadius: 7,
        pointHoverRadius: 10,
      }]
    },
    options: {
      ...common,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const r = done[ctx.dataIndex];
              return [
                `${r.scenario_description}  (\u03B8=${r.threshold}, ${r.region})`,
                `  Savings: ${r.co2_savings_pct.toFixed(2)}%`,
                `  Overhead: ${r.actual_overhead_pct.toFixed(1)}%`,
                `  Score: ${r.score.toFixed(4)}`,
              ];
            }
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Overhead %' },
          type: 'linear',
        },
        y: {
          title: { display: true, text: 'CO\u2082 Savings %' },
        }
      }
    }
  });
}

// ── Table ─────────────────────────────────────────────────────────
let sortKey = 'co2_savings_pct';
let sortAsc = false;

function renderTable(data) {
  const sorted = [...data].sort((a,b) => {
    const va = a[sortKey], vb = b[sortKey];
    if (va == null) return 1; if (vb == null) return -1;
    return sortAsc ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  });
  document.getElementById('rowCount').textContent =
    `${data.length} / ${rawData.length} run${rawData.length !== 1 ? 's' : ''}`;

  const uniques = {};
  FIELDS.forEach(f => { uniques[f.key] = uniqVals(rawData, f.key, f.fmt, f.dec); });

  const thead = document.getElementById('tableHead');
  const headRow = '<tr>' + FIELDS.map(f => {
    const cls = sortKey === f.key ? 'sorted' : '';
    const dir = sortKey === f.key && sortAsc ? ' desc' : '';
    return `<th data-key="${f.key}" class="${cls}${dir}">${esc(f.label)}</th>`;
  }).join('') + '</tr>';

  const filterRow = '<tr class="filter-row">' + FIELDS.map(f => {
    const cur = filters[f.key] || '';
    const opts = ['<option value="">All</option>',
      ...uniques[f.key].map(v =>
        `<option value="${esc(v)}"${v === cur ? ' selected' : ''}>${esc(v)}</option>`
      )
    ].join('');
    return `<th><select class="filter-select" data-key="${f.key}">${opts}</select></th>`;
  }).join('') + '</tr>';

  thead.innerHTML = headRow + filterRow;

  thead.querySelectorAll('tr:first-child th').forEach(th => {
    th.addEventListener('click', () => {
      const k = th.dataset.key;
      if (sortKey === k) sortAsc = !sortAsc;
      else { sortKey = k; sortAsc = false; }
      renderTable(data);
    });
  });

  thead.querySelectorAll('.filter-select').forEach(sel => {
    sel.addEventListener('change', onFilterChange);
  });

  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = sorted.map((r, idx) => {
    const cells = FIELDS.map(f => {
      const v = r[f.key];
      let html;
      switch (f.fmt) {
        case 'num': html = `<td class="num">${fmtNum(v, f.dec||0)}</td>`; break;
        case 'pct': html = `<td class="num ${valClass(v)}">${fmtPct(v, f.dec||1)}</td>`; break;
        case 'kpct':
          const cls = valClass(v);
          html = `<td class="num ${cls}"><strong>${fmtPct(v, 2)}</strong></td>`;
          break;
        case 'score':
          html = `<td class="num ${valClass(v)}">${v != null && !isNaN(v) ? v.toFixed(4) : '-'}</td>`;
          break;
        case 'bool':
          html = `<td>${v ? '<span class="tag tag-ok">\u2713</span>' : '<span class="tag tag-fail">\u2717</span>'}</td>`;
          break;
        default:
          html = `<td>${esc(String(v ?? ''))}</td>`;
      }
      return html;
    }).join('');
    const tsId = r.timeseries_id != null ? r.timeseries_id : idx;
    return `<tr data-ts-id="${esc(String(tsId))}">${cells}</tr>`;
  }).join('');

  tbody.querySelectorAll('tr').forEach(tr => {
    tr.addEventListener('click', () => {
      const tsId = tr.dataset.tsId;
      if (!tsId) return;
      document.querySelectorAll('tbody tr.selected').forEach(el => el.classList.remove('selected'));
      tr.classList.add('selected');
      window.open('scenario.html?id=' + tsId, '_blank');
    });
  });
}

function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// ── Main render ───────────────────────────────────────────────────
function render(data) {
  rawData = data;
  filters = {};
  filteredData = data;
  document.getElementById('emptyState').style.display = 'none';
  document.getElementById('content').style.display = 'block';
  renderCards(data);
  renderCharts(data);
  renderTable(data);
  document.getElementById('rowCount').textContent =
    `${data.length} run${data.length !== 1 ? 's' : ''}`;
}

// ── File loading ─────────────────────────────────────────────────
function loadFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('fileLabel').textContent = file.name;
    try {
      const data = parseCSV(e.target.result);
      if (!data || data.length === 0) throw new Error('No data rows found');
      render(data);
    } catch(err) {
      alert('Error parsing CSV: ' + err.message);
    }
  };
  reader.readAsText(file);
}

document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

// Drag and drop
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

// ── Auto-load from URL param? ───────────────────────────────────
const params = new URLSearchParams(window.location.search);
const csvUrl = params.get('csv');
if (csvUrl) {
  fetch(csvUrl).then(r => r.text()).then(text => {
    try { render(parseCSV(text)); } catch(e) { console.error(e); }
  }).catch(() => {});
}

// Auto-load results.csv when served from a web server
if (!csvUrl && window.location.protocol.startsWith('http')) {
  fetch('results.csv').then(r => {
    if (r.ok) return r.text();
    throw new Error('Not found');
  }).then(text => {
    try { render(parseCSV(text)); } catch(e) { console.error(e); }
  }).catch(() => {});
}
