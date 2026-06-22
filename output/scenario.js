// ── State ─────────────────────────────────────────────────────────
let timelineChart = null;

const stateBgPlugin = {
  id: 'stateBg',
  beforeDraw(chart) {
    const bands = chart.data.stateBands;
    if (!bands || !bands.length) return;
    const { ctx, chartArea: { top, bottom } } = chart;
    const xScale = chart.scales.x;
    ctx.save();
    bands.forEach(band => {
      const x1 = xScale.getPixelForValue(band.startIdx);
      const x2 = xScale.getPixelForValue(band.endIdx);
      if (x1 === undefined || x2 === undefined) return;
      ctx.fillStyle = band.state === 'running'
        ? 'rgba(0, 201, 167, 0.07)'
        : 'rgba(255, 94, 122, 0.07)';
      ctx.fillRect(x1, top, x2 - x1, bottom - top);
    });
    ctx.restore();
  },
};

function destroyTimeline() {
  if (timelineChart) { try { timelineChart.destroy(); } catch(e) {} timelineChart = null; }
}

function renderTimeline(data) {
  destroyTimeline();
  document.getElementById('dropZone').style.display = 'none';
  const n = data.carbon_intensity.length;
  if (n === 0) return;

  const labels = data.timestamps.map(t => {
    const d = new Date(t);
    return d.toLocaleString(undefined, {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'});
  });
  const idx = data.carbon_intensity.map((_, i) => i);

  const stateBands = [];
  if (data.state && data.state.length) {
    let start = 0;
    for (let i = 1; i <= data.state.length; i++) {
      if (i === data.state.length || data.state[i] !== data.state[start]) {
        stateBands.push({ startIdx: start, endIdx: i - 1, state: data.state[start] });
        start = i;
      }
    }
  }

  let ciMin = Infinity, ciMax = -Infinity;
  data.carbon_intensity.forEach(v => {
    if (v != null && isFinite(v)) { ciMin = Math.min(ciMin, v); ciMax = Math.max(ciMax, v); }
  });
  const ciPad = (ciMax - ciMin) * 0.1 || 20;
  ciMin = Math.max(0, ciMin - ciPad);
  ciMax = ciMax + ciPad;

  const thetaPause = data.theta_pause;
  const thetaResume = data.theta_resume;

  document.getElementById('timelineTitle').textContent =
    '\u03B8\u209A=' + thetaPause + ' \u03B8\u1D63=' + thetaResume + ' \u2022 ' + n + ' steps';

  const canvas = document.getElementById('chartTimeline');
  timelineChart = new Chart(canvas, {
    type: 'line',
    data: {
      labels: idx,
      datasets: [
        {
          label: 'Carbon Intensity',
          data: data.carbon_intensity,
          borderColor: '#2e7dff',
          backgroundColor: 'rgba(46,125,255,0.1)',
          fill: true,
          pointRadius: 0,
          borderWidth: 1.5,
          yAxisID: 'y',
          order: 2,
        },
        {
          label: '\u03B8_pause',
          data: idx.map(() => thetaPause),
          borderColor: '#ff5e7a',
          borderDash: [4, 4],
          pointRadius: 0,
          borderWidth: 1.5,
          yAxisID: 'y',
          order: 1,
        },
        {
          label: '\u03B8_resume',
          data: idx.map(() => thetaResume),
          borderColor: '#00c9a7',
          borderDash: [4, 4],
          pointRadius: 0,
          borderWidth: 1.5,
          yAxisID: 'y',
          order: 1,
        },
        {
          label: 'Cumulative Emissions',
          data: data.emissions_kg,
          borderColor: '#ff9f43',
          pointRadius: 0,
          borderWidth: 1.5,
          yAxisID: 'y1',
          order: 0,
        },
      ],
      stateBands,
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          position: 'top',
          labels: { boxWidth: 14, padding: 14, font: { size: 11 } },
        },
        tooltip: {
          callbacks: {
            title: ctx => labels[ctx[0].dataIndex] || '',
            afterBody: ctx => {
              const i = ctx[0].dataIndex;
              const s = data.state && data.state[i] ? data.state[i] : '?';
              return 'State: ' + s;
            }
          }
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'Step' },
          ticks: { maxTicksLimit: 20 },
        },
        y: {
          type: 'linear',
          position: 'left',
          title: { display: true, text: 'gCO\u2082eq/kWh' },
          min: ciMin,
          max: ciMax,
          grid: { drawOnChartArea: true },
        },
        y1: {
          type: 'linear',
          position: 'right',
          title: { display: true, text: 'kg CO\u2082' },
          grid: { drawOnChartArea: false },
        },
      },
    },
    plugins: [stateBgPlugin],
  });
}

// ── Load from JSON string (shared by fetch + file drop) ──────────
function loadData(data) {
  if (!data || !data.carbon_intensity || !data.carbon_intensity.length) {
    document.getElementById('timelineTitle').textContent = 'Invalid or empty timeseries file';
    return;
  }
  renderTimeline(data);
}

// ── File drop (local file:// usage, or fallback) ─────────────────
function loadFile(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('fileLabel').textContent = file.name;
    try {
      loadData(JSON.parse(e.target.result));
    } catch(err) {
      alert('Error parsing JSON: ' + err.message);
    }
  };
  reader.readAsText(file);
}

document.getElementById('fileInput').addEventListener('change', e => {
  if (e.target.files[0]) loadFile(e.target.files[0]);
});

const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('dragover'); });
dz.addEventListener('dragleave', () => dz.classList.remove('dragover'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('dragover');
  if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
});

// ── Load from URL param or file:// ────────────────────────────────
const params = new URLSearchParams(window.location.search);
const id = params.get('id');

if (window.location.protocol.startsWith('http') && id != null) {
  fetch('timeseries/' + id + '.json').then(r => {
    if (!r.ok) throw new Error('Not found');
    return r.json();
  }).then(data => {
    loadData(data);
  }).catch(() => {
    document.getElementById('dropZone').style.display = 'flex';
    document.getElementById('timelineTitle').textContent = 'No timeseries data for id=' + id + ' — drop a .json file';
  });
} else if (id != null) {
  document.getElementById('dropZone').style.display = 'flex';
  document.getElementById('timelineTitle').textContent = 'Drop the timeseries .json file for id=' + id;
} else if (window.location.protocol.startsWith('http')) {
  document.getElementById('timelineTitle').textContent = 'No scenario id specified (?id=...)';
} else {
  document.getElementById('dropZone').style.display = 'flex';
  document.getElementById('timelineTitle').textContent = 'Drop a timeseries .json file';
}
