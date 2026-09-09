let selectedFiles = [];
let currentJobId = null;
let currentOutputPath = null;
let liveJobId = null;
let lastProgressPercent = 0;
let lastProgressTerminal = false;

const $ = (id) => document.getElementById(id);

function log(message, obj) {
  const box = $('logBox');
  const line = `[${new Date().toLocaleTimeString()}] ${message}` + (obj ? ` ${JSON.stringify(obj)}` : '');
  box.textContent = line + '\n' + box.textContent;
}

function normalizeProgressPercent(percent, statusText) {
  const status = String(statusText || '').toLowerCase();
  if (['complete', 'completed', 'success', 'finished', 'job complete', 'conversion complete'].some(x => status.includes(x))) return 100;
  if (status.includes('failed') || status.includes('stopped')) return 100;
  const value = Number(percent);
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function resetProgressTracker() {
  lastProgressPercent = 0;
  lastProgressTerminal = false;
}

function setProgress(percent, statusText, stepText) {
  const status = String(statusText || '').toLowerCase();
  const isTerminal = ['complete', 'completed', 'success', 'finished', 'job complete', 'conversion complete'].some(x => status.includes(x))
    || status.includes('failed') || status.includes('stopped');
  let pct = normalizeProgressPercent(percent, statusText);

  // While a job is running, never let local/nested converter steps make the
  // global loader move backward.  Only a new job reset or a terminal state can
  // override the monotonic display.
  const isReset = pct <= 1 && ['job started', 'uploading files'].some(x => status.includes(x));
  if (isReset) {
    resetProgressTracker();
  } else if (!isTerminal && !lastProgressTerminal && pct < lastProgressPercent) {
    pct = lastProgressPercent;
  }
  if (isTerminal) {
    pct = 100;
    lastProgressTerminal = true;
  }
  lastProgressPercent = pct;

  const bar = $('progressBar');
  if (bar) {
    bar.style.width = `${pct}%`;
    bar.textContent = pct >= 100 ? 'Full (100%)' : `${pct}%`;
    bar.setAttribute('aria-valuenow', String(pct));
    bar.setAttribute('aria-valuemin', '0');
    bar.setAttribute('aria-valuemax', '100');
    bar.setAttribute('role', 'progressbar');
  }
  if (statusText && $('jobStatus')) $('jobStatus').textContent = statusText;
  if (stepText && $('currentStep')) $('currentStep').textContent = `Current step: ${stepText} (${pct}%)`;
  if (pct >= 100 && $('completedStep') && stepText) $('completedStep').textContent = `Completed step: ${stepText}`;
}

function showTab(tabId) {
  document.querySelectorAll('.panel').forEach(p => {
    p.classList.remove('active');
    p.style.display = '';
  });
  document.querySelectorAll('button.tab').forEach(b => b.classList.remove('active'));
  const panel = $(tabId);
  if (panel) panel.classList.add('active');
  const tab = document.querySelector(`button[data-tab="${tabId}"]`);
  if (tab) tab.classList.add('active');
}


function ensureAdvancedAnalysisVisible(scrollToSection = false) {
  const section = $('advancedMethodsTab');
  if (section && scrollToSection) {
    showTab('advancedMethodsTab');
    setTimeout(() => section.scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
  }
}

document.querySelectorAll('button.tab').forEach(btn => btn.addEventListener('click', () => showTab(btn.dataset.tab)));

async function checkHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    $('serverStatus').textContent = `backend online — workspace: ${data.workspace}`;
    $('serverStatus').className = 'status ok';
  } catch (e) {
    $('serverStatus').textContent = 'backend offline';
    $('serverStatus').className = 'status failed';
  }
}

function refreshFileList() {
  const ul = $('fileList');
  ul.innerHTML = '';
  selectedFiles.forEach(file => {
    const li = document.createElement('li');
    li.textContent = `${file.name} — ${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    ul.appendChild(li);
  });
}


function appendSelectedFiles(form) {
  selectedFiles.forEach(f => form.append('files', f, f.webkitRelativePath || f.name));
}

function collectOptions() {
  const val = (id) => $(id).value.trim();
  const numOrNull = (id) => val(id) ? Number(val(id)) : null;
  return {
    output_dir: val('outputDir') || null,
    sampling_rate: numOrNull('samplingRate'),
    signal_path: val('signalPath') || null,
    orientation: val('orientation'),
    include_aux: $('includeAux').checked,
    original_units: val('originalUnits') || null,
    target_units: val('targetUnits') || null,
    scale_factor: numOrNull('scaleFactor'),
    offset: numOrNull('offset'),
    export_format: val('exportFormat') || 'npy',
    save_signal_csv: $('saveSignalCsv').checked,
    csv_max_mb: 250,
    preprocess: $('preprocess').checked,
    demean: $('demean').checked,
    detrend: $('detrend').checked,
    normalization: val('normalization') || null,
    make_windows: $('makeWindows').checked,
    window_seconds: numOrNull('windowSeconds') || 2,
    step_seconds: numOrNull('stepSeconds') || 1,
  };
}

function connectJobEvents(jobId) {
  currentJobId = jobId;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}/events`);
  showTab('resultsTab');
  setProgress(0, 'Job started', 'Connecting to heartbeat...');
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'heartbeat') return;
    log(data.message || data.step || data.type, data);
    if (data.percent !== undefined) setProgress(data.percent, data.status || 'running', data.step || data.message);
    if (data.status === 'complete') {
      setProgress(100, 'Conversion complete', data.step || 'Complete');
      $('completedStep').textContent = `Completed step: ${data.step || 'Complete'}`;
      currentOutputPath = data.output_dir || (data.result && data.result.output_dir) || currentOutputPath;
      $('outputPath').textContent = currentOutputPath || '';
      $('resultJson').textContent = JSON.stringify(data.result || data, null, 2);
      ws.close();
    }
    if (data.status === 'failed') {
      setProgress(100, 'Job failed', data.step || 'Failed');
      $('jobStatus').className = 'status failed big-status';
      $('resultJson').textContent = JSON.stringify(data, null, 2);
      ws.close();
    }
    if (data.status === 'running') {
      $('jobStatus').className = 'status big-status';
    }
  };
  ws.onerror = () => log('WebSocket error');
}

async function startConvert(fullAnalyze = false) {
  if (!selectedFiles.length) { alert('Choose or drop at least one file.'); return; }
  const form = new FormData();
  appendSelectedFiles(form);
  const options = collectOptions();
  options.make_windows = fullAnalyze ? true : options.make_windows;
  form.append('options_json', JSON.stringify(options));
  if (options.output_dir) form.append('output_dir', options.output_dir);
  setProgress(1, 'Uploading files', 'Uploading files to local backend...');
  showTab('resultsTab');
  const res = await fetch('/api/jobs/convert-upload', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

async function inspectUploads() {
  if (!selectedFiles.length) { alert('Choose or drop at least one file.'); return; }
  const form = new FormData();
  appendSelectedFiles(form);
  const res = await fetch('/api/jobs/inspect-upload', { method: 'POST', body: form });
  const data = await res.json();
  showTab('resultsTab');
  $('resultJson').textContent = JSON.stringify(data, null, 2);
  log('Inspection complete', data);
}

async function startCompare() {
  const groupA = $('groupA').value.split('\n').map(x => x.trim()).filter(Boolean);
  const groupB = $('groupB').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (!groupA.length || !groupB.length) { alert('Provide at least one converted folder in Group A and Group B.'); return; }
  const payload = {
    group_a: groupA,
    group_b: groupB,
    output_dir: $('compareOutput').value.trim() || null,
    options: { comparison_name: $('comparisonName').value.trim() || 'comparison' }
  };
  const res = await fetch('/api/jobs/compare', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

async function startLive() {
  const payload = {
    source: $('liveSignal').value.trim(),
    channels_csv: $('liveChannels').value.trim() || null,
    metadata_json: $('liveMetadata').value.trim() || null,
    fs: $('liveFs').value.trim() || null,
    channel_profile: $('liveProfile').value || 'auto'
  };
  if (!payload.source) { alert('Provide a signal.npy path.'); return; }
  const res = await fetch('/api/jobs/live', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  liveJobId = data.job_id;
  connectJobEvents(data.job_id);
}

async function stopLive() {
  if (!liveJobId) { alert('No live job id recorded.'); return; }
  await fetch(`/api/jobs/${liveJobId}/stop-live`, { method: 'POST' });
}

$('chooseFilesBtn').addEventListener('click', () => $('fileInput').click());
$('fileInput').addEventListener('change', (e) => { selectedFiles = Array.from(e.target.files); refreshFileList(); });
if ($('folderInput')) $('folderInput').addEventListener('change', (e) => { selectedFiles = Array.from(e.target.files); refreshFileList(); });
if ($('chooseFolderBtn')) $('chooseFolderBtn').addEventListener('click', () => $('folderInput').click());
$('clearFilesBtn').addEventListener('click', () => { selectedFiles = []; $('fileInput').value = ''; if ($('folderInput')) $('folderInput').value = ''; refreshFileList(); });
$('inspectBtn').addEventListener('click', inspectUploads);
$('convertBtn').addEventListener('click', () => startConvert(false));
$('fullAnalyzeBtn').addEventListener('click', () => startConvert(true));
$('compareBtn').addEventListener('click', startCompare);
$('startLiveBtn').addEventListener('click', startLive);
$('stopLiveBtn').addEventListener('click', stopLive);
$('openOutputBtn').addEventListener('click', async () => { if (currentOutputPath) await fetch(`/api/open-output?path=${encodeURIComponent(currentOutputPath)}`); });
$('copyOutputBtn').addEventListener('click', () => { if (currentOutputPath) navigator.clipboard.writeText(currentOutputPath); });

const dropZone = $('dropZone');
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  selectedFiles = Array.from(e.dataTransfer.files || []);
  refreshFileList();
});

checkHealth();

// ---- v0.8 NeuroMouse integration ----
function openUrl(url) { window.open(url, '_blank'); }

function showNeuroMouseOpenLink(url, label = 'Open generated NeuroMouse dataset from this job') {
  if (!url) return;
  const output = $('outputPath');
  const absolute = new URL(url, location.origin).href;
  const datasetUrl = datasetFetchUrlFromNeuroMouseUrl(url);
  const datasetAbs = datasetUrl ? new URL(datasetUrl, location.origin).href : '';
  if (output) output.innerHTML = `${currentOutputPath || ''}<br><a class="action-link" href="${absolute}" target="_blank" rel="noopener">${label}</a>${datasetAbs ? `<br><small>generated data.json: ${datasetAbs}</small>` : ''}`;
  const box = $('neuromouseGeneratedLinkBox');
  if (box) {
    box.innerHTML = `<a class="primary-link" href="${absolute}" target="_blank" rel="noopener">${label}</a><br><small>${absolute}</small>${datasetAbs ? `<br><small>generated data.json: ${datasetAbs}</small>` : ''}`;
  }
}

function openBackendNeuroMouseUrl(url, label = 'Open generated NeuroMouse dataset') {
  if (!url) return;
  // Keep the Neuro Signal app page open. Earlier versions navigated this same
  // tab to /neuromouse-job/<job_id>/ as soon as a job completed, which felt like
  // the window closed. Now we only reveal a link/button and let the user choose.
  try {
    const absolute = new URL(url, location.origin);
    if (!absolute.searchParams.has('t')) absolute.searchParams.set('t', Date.now().toString());
    const datasetUrl = datasetFetchUrlFromNeuroMouseUrl(absolute.toString());
    if (datasetUrl) {
      const dsAbs = new URL(datasetUrl, location.origin);
      localStorage.setItem('NEURO_SIGNAL_LAST_BACKEND_DATASET_URL', dsAbs.pathname + dsAbs.search);
    }
    localStorage.setItem('NEURO_SIGNAL_LAST_NEUROMOUSE_URL', absolute.pathname + absolute.search);
    showNeuroMouseOpenLink(absolute.toString(), label);
    log('Generated NeuroMouse dataset is ready; current page stays open', { url: absolute.toString(), dataset_url: datasetUrl });
  } catch {
    showNeuroMouseOpenLink(url, label);
    log('Generated NeuroMouse dataset is ready; current page stays open', { url });
  }
}


async function analyzeUploadsInNeuroMouse() {
  if (!selectedFiles.length) { alert('Choose or drop at least one file first.'); return; }
  const form = new FormData();
  appendSelectedFiles(form);
  const options = collectOptions();
  options.neuromouse_max_analysis_samples = 240000;
  options.neuromouse_max_windows = 600;
  form.append('options_json', JSON.stringify(options));
  if (options.output_dir) form.append('output_dir', options.output_dir);
  setProgress(1, 'NeuroMouse analysis started', 'Uploading and converting files...');
  showTab('resultsTab');
  const res = await fetch('/api/jobs/analyze-neuromouse-upload', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

async function buildNeuroMouseFromConverted() {
  const dirs = $('neuromouseConvertedDirs').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (!dirs.length) { alert('Provide at least one converted recording folder.'); return; }
  const payload = {
    recording_dirs: dirs,
    output_dir: $('neuromouseOutput').value.trim() || null,
    options: { sampling_rate: $('samplingRate').value.trim() || null }
  };
  const res = await fetch('/api/jobs/neuromouse-from-converted', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

async function compareGroupsInNeuroMouse() {
  const groupA = $('groupA').value.split('\n').map(x => x.trim()).filter(Boolean);
  const groupB = $('groupB').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (!groupA.length || !groupB.length) { alert('Provide at least one converted folder in Group A and Group B.'); return; }
  const payload = {
    group_a: groupA,
    group_b: groupB,
    output_dir: $('compareOutput').value.trim() || null,
    options: { comparison_name: $('comparisonName').value.trim() || 'neuromouse_comparison', sampling_rate: $('samplingRate').value.trim() || null }
  };
  const res = await fetch('/api/jobs/compare-neuromouse', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

function openNeuroMouseLiveReplay() {
  const source = $('neuromouseLiveSignal').value.trim() || $('liveSignal').value.trim();
  if (!source) { alert('Provide a signal.npy path.'); return; }
  const params = new URLSearchParams();
  params.set('source', source);
  const channels = $('neuromouseLiveChannels').value.trim() || $('liveChannels').value.trim();
  const metadata = $('neuromouseLiveMetadata').value.trim() || $('liveMetadata').value.trim();
  const fs = $('neuromouseLiveFs').value.trim() || $('liveFs').value.trim();
  const speed = $('neuromouseLiveSpeed').value.trim() || '1';
  if (channels) params.set('channels_csv', channels);
  if (metadata) params.set('metadata_json', metadata);
  if (fs) params.set('fs', fs);
  params.set('speed', speed);
  const wsProto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${wsProto}://${location.host}/ws/neuromouse/live?${params.toString()}`;
  openUrl(`/neuromouse/?live_ws=${encodeURIComponent(wsUrl)}`);
}

// Extend connectJobEvents to reveal NeuroMouse links when present.
const _oldConnectJobEvents = connectJobEvents;
connectJobEvents = function(jobId) {
  currentJobId = jobId;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}/events`);
  showTab('resultsTab');
  setProgress(0, 'Job started', 'Connecting to heartbeat...');
  ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'heartbeat') return;
    log(data.message || data.step || data.type, data);
    if (data.percent !== undefined) setProgress(data.percent, data.status || 'running', data.step || data.message);
    if (data.status === 'complete') {
      setProgress(100, data.step || 'Job complete', data.step || 'Complete');
      $('completedStep').textContent = `Completed step: ${data.step || 'Complete'}`;
      currentOutputPath = data.output_dir || (data.result && data.result.output_dir) || currentOutputPath;
      $('outputPath').textContent = currentOutputPath || '';
      $('resultJson').textContent = JSON.stringify(data.result || data, null, 2);
      const r = data.result || {};
      if ((r.primary_neuromouse_url || r.primary_neuromouse_url)) openBackendNeuroMouseUrl((r.primary_neuromouse_url || r.primary_neuromouse_url));
      if ((r.neuromouse_comparison_url || r.neuromouse_comparison_url)) openBackendNeuroMouseUrl((r.neuromouse_comparison_url || r.neuromouse_comparison_url));
      ws.close();
    }
    if (data.status === 'failed') {
      setProgress(100, 'Job failed', data.step || 'Failed');
      $('jobStatus').className = 'status failed big-status';
      $('resultJson').textContent = JSON.stringify(data, null, 2);
      ws.close();
    }
    if (data.status === 'running') $('jobStatus').className = 'status big-status';
  };
  ws.onerror = () => log('WebSocket error');
};

if ($('openNeuroMouseBtn')) $('openNeuroMouseBtn').addEventListener('click', () => openUrl('/neuromouse/?demo=1'));
if ($('neuromouseAnalyzeBtn')) $('neuromouseAnalyzeBtn').addEventListener('click', analyzeUploadsInNeuroMouse);
if ($('analyzeNeuroMouseBtn2')) $('analyzeNeuroMouseBtn2').addEventListener('click', analyzeUploadsInNeuroMouse);
if ($('neuromouseFromConvertedBtn')) $('neuromouseFromConvertedBtn').addEventListener('click', buildNeuroMouseFromConverted);
if ($('neuromouseCompareBtn')) $('neuromouseCompareBtn').addEventListener('click', compareGroupsInNeuroMouse);
if ($('neuromouseCompareBtn2')) $('neuromouseCompareBtn2').addEventListener('click', compareGroupsInNeuroMouse);
if ($('openNeuroMouseLiveBtn')) $('openNeuroMouseLiveBtn').addEventListener('click', openNeuroMouseLiveReplay);

// ---- v0.9 complete NeuroMouse workbench + quick visualization layer ----
let latestNeuroMouseUrl = null;
let latestNeuroMouseComparisonUrl = null;
let latestNeuroMouseData = null;
let latestComparisonManifest = null;

function neuromouseOptions() {
  const maxSamples = Number(($('neuromouseMaxSamples')?.value || '').trim()) || 240000;
  const maxWindows = Number(($('neuromouseMaxWindows')?.value || '').trim()) || 600;
  return {
    neuromouse_max_analysis_samples: maxSamples,
    neuromouse_max_windows: maxWindows,
  };
}

async function analyzeUploadsInNeuroMouse() {
  if (!selectedFiles.length) { alert('Choose or drop at least one file first.'); return; }
  const form = new FormData();
  appendSelectedFiles(form);
  const options = { ...collectOptions(), ...neuromouseOptions() };
  form.append('options_json', JSON.stringify(options));
  if (options.output_dir) form.append('output_dir', options.output_dir);
  setProgress(1, 'NeuroMouse analysis started', 'Uploading and converting files...');
  showTab('resultsTab');
  const res = await fetch('/api/jobs/analyze-neuromouse-upload', { method: 'POST', body: form });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

async function buildNeuroMouseFromConverted() {
  const dirs = $('neuromouseConvertedDirs').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (!dirs.length) { alert('Provide at least one converted recording folder.'); return; }
  const payload = {
    recording_dirs: dirs,
    output_dir: $('neuromouseOutput').value.trim() || null,
    options: { sampling_rate: $('samplingRate').value.trim() || null, ...neuromouseOptions() }
  };
  const res = await fetch('/api/jobs/neuromouse-from-converted', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

async function compareGroupsInNeuroMouse() {
  const groupA = $('groupA').value.split('\n').map(x => x.trim()).filter(Boolean);
  const groupB = $('groupB').value.split('\n').map(x => x.trim()).filter(Boolean);
  if (!groupA.length || !groupB.length) { alert('Provide at least one converted folder in Group A and Group B.'); return; }
  const payload = {
    group_a: groupA,
    group_b: groupB,
    output_dir: $('compareOutput').value.trim() || null,
    options: { comparison_name: $('comparisonName').value.trim() || 'neuromouse_comparison', sampling_rate: $('samplingRate').value.trim() || null, ...neuromouseOptions() }
  };
  const res = await fetch('/api/jobs/compare-neuromouse', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  const data = await res.json();
  if (data.error) { alert(data.error); return; }
  connectJobEvents(data.job_id);
}

function datasetFetchUrlFromNeuroMouseUrl(neuromouseUrl) {
  if (!neuromouseUrl) return null;
  try {
    const u = new URL(neuromouseUrl, location.origin);
    const explicit = u.searchParams.get('dataset') || u.searchParams.get('data_json');
    if (explicit) return explicit;
    const m = u.pathname.match(/^\/neuromouse-job\/([^/]+)\/?$/);
    if (m) return `/api/jobs/${m[1]}/neuromouse/data.json`;
    return null;
  } catch { return null; }
}

function comparisonFetchUrlFromNeuroMouseUrl(neuromouseUrl) {
  if (!neuromouseUrl) return null;
  try {
    const u = new URL(neuromouseUrl, location.origin);
    return u.searchParams.get('comparison');
  } catch { return null; }
}

async function fetchJsonMaybe(url) {
  if (!url) return null;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}: HTTP ${res.status}`);
  return await res.json();
}

function meanRows(matrix) {
  if (!Array.isArray(matrix) || !matrix.length) return [];
  const n = Math.max(...matrix.map(row => Array.isArray(row) ? row.length : 0));
  const out = new Array(n).fill(0);
  const count = new Array(n).fill(0);
  matrix.forEach(row => {
    if (!Array.isArray(row)) return;
    row.forEach((v, i) => {
      const x = Number(v);
      if (Number.isFinite(x)) { out[i] += x; count[i] += 1; }
    });
  });
  return out.map((v, i) => count[i] ? v / count[i] : NaN);
}

function drawLineChart(canvas, xValues, yValues, title, yLabel) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#010505'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(0,229,229,0.35)'; ctx.strokeRect(0.5, 0.5, w-1, h-1);
  ctx.fillStyle = '#dfffff'; ctx.font = '14px monospace'; ctx.fillText(title, 16, 24);
  if (!yValues || yValues.length < 2) { ctx.fillText('No preview data available', 16, 52); return; }
  const vals = yValues.map(Number).filter(Number.isFinite);
  if (!vals.length) { ctx.fillText('No finite values', 16, 52); return; }
  const minY = Math.min(...vals), maxY = Math.max(...vals);
  const pad = Math.max((maxY - minY) * 0.08, 1e-9);
  const yMin = minY - pad, yMax = maxY + pad;
  const left = 54, right = 14, top = 38, bottom = 32;
  const plotW = w - left - right, plotH = h - top - bottom;
  ctx.strokeStyle = 'rgba(127,255,255,0.18)';
  for (let i=0; i<5; i++) {
    const y = top + (plotH * i / 4);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w-right, y); ctx.stroke();
  }
  ctx.fillStyle = '#7bdada'; ctx.font = '11px monospace';
  ctx.fillText(`${yMax.toFixed(3)} ${yLabel || ''}`, 8, top + 4);
  ctx.fillText(`${yMin.toFixed(3)} ${yLabel || ''}`, 8, top + plotH);
  ctx.strokeStyle = '#00ff99'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  yValues.forEach((v, i) => {
    const x = left + plotW * (i / Math.max(1, yValues.length - 1));
    const y = top + plotH * (1 - ((Number(v) - yMin) / (yMax - yMin || 1)));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

async function renderNeuroMouseDatasetPreview(neuromouseUrl) {
  const fetchUrl = datasetFetchUrlFromNeuroMouseUrl(neuromouseUrl);
  if (!fetchUrl) return;
  try {
    const data = await fetchJsonMaybe(fetchUrl);
    latestNeuroMouseData = data;
    const nChannels = data?.meta?.channels?.length || data?.meta?.n_channels || 0;
    const nFrames = data?.geometry?.time?.length || data?.centroid?.time_relative?.length || 0;
    const text = `Loaded NeuroMouse dataset: ${data?.meta?.dataset_id || 'dataset'} · ${nChannels} channels · ${nFrames} frames`;
    if ($('previewMeta')) $('previewMeta').textContent = text;
    if ($('resultsPreviewMeta')) $('resultsPreviewMeta').textContent = text;
    drawLineChart($('centroidPreview'), data?.geometry?.time || data?.centroid?.time_relative || [], meanRows(data?.centroid?.values || data?.geometry?.centroid || []), 'Mean spectral centroid over time', 'Hz');
    const avgPsd = meanRows(data?.welch_psd?.psd || []);
    drawLineChart($('psdPreview'), data?.welch_psd?.frequencies || [], avgPsd.map(v => Math.log10(Math.max(Number(v), 1e-18))), 'Mean log10 Welch PSD across channels', 'log');
    renderNeuroMouseAdvancedPlots(data);
  } catch (err) {
    if ($('previewMeta')) $('previewMeta').textContent = `Preview failed: ${err.message}`;
  }
}

async function renderNeuroMouseComparisonPreview(neuromouseUrl) {
  const fetchUrl = comparisonFetchUrlFromNeuroMouseUrl(neuromouseUrl);
  if (!fetchUrl) return;
  try {
    const manifest = await fetchJsonMaybe(fetchUrl);
    latestComparisonManifest = manifest;
    const rows = manifest.datasets || [];
    const container = $('comparisonPreview');
    if (!container) return;
    let html = `<strong>${manifest.comparison_name || 'NeuroMouse comparison'}</strong><br><small>${manifest.compatibility_note || ''}</small>`;
    html += '<table><thead><tr><th>Group</th><th>Dataset</th><th>Channels</th><th>Open</th></tr></thead><tbody>';
    rows.forEach(row => {
      const url = row.data_json_url ? `/neuromouse/?dataset=${encodeURIComponent(row.data_json_url)}` : '';
      html += `<tr><td>${row.group || ''}</td><td>${row.dataset_id || ''}</td><td>variable</td><td>${url ? `<a href="${url}" target="_blank">open</a>` : ''}</td></tr>`;
    });
    html += '</tbody></table>';
    const report = manifest?.comparison_result?.report_html;
    if (report) html += `<p><a href="/api/file?path=${encodeURIComponent(report)}" target="_blank">Open backend comparison report</a></p>`;
    container.innerHTML = html;
    if ($('resultsPreviewMeta')) $('resultsPreviewMeta').textContent = `Loaded comparison: ${rows.length} NeuroMouse datasets`;
  } catch (err) {
    if ($('comparisonPreview')) $('comparisonPreview').textContent = `Comparison preview failed: ${err.message}`;
  }
}

async function handleCompletedNeuroMouseResult(result) {
  const r = result || {};
  if (r.primary_neuromouse_dataset_url) {
    try { localStorage.setItem('NEURO_SIGNAL_LAST_BACKEND_DATASET_URL', r.primary_neuromouse_dataset_url); } catch {}
  } else if (currentJobId) {
    try { localStorage.setItem('NEURO_SIGNAL_LAST_BACKEND_DATASET_URL', `/api/jobs/${currentJobId}/neuromouse/data.json`); } catch {}
  }
  const primaryUrl = r.primary_neuromouse_url || (currentJobId ? `/neuromouse-job/${currentJobId}/?backend=1&force_backend=1&t=${Date.now()}` : null);
  if (primaryUrl) {
    latestNeuroMouseUrl = primaryUrl;
    try { localStorage.setItem('NEURO_SIGNAL_LAST_NEUROMOUSE_URL', latestNeuroMouseUrl); } catch {}
    await renderNeuroMouseDatasetPreview(latestNeuroMouseUrl);
  }
  if ((r.neuromouse_comparison_url || r.neuromouse_comparison_url)) {
    latestNeuroMouseComparisonUrl = (r.neuromouse_comparison_url || r.neuromouse_comparison_url);
    await renderNeuroMouseComparisonPreview(latestNeuroMouseComparisonUrl);
  }
}

// Final override: job stream + NeuroMouse link reveal + quick preview render.
connectJobEvents = function(jobId) {
  currentJobId = jobId;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/api/jobs/${jobId}/events`);
  showTab('resultsTab');
  resetProgressTracker();
  setProgress(0, 'Job started', 'Connecting to heartbeat...');

  const applyJobEvent = async (data) => {
    const status = String(data.status || '').toLowerCase();
    const eventType = String(data.type || '').toLowerCase();
    const isJobStateEvent = eventType === 'state' || eventType === 'heartbeat';
    const isTerminal = isJobStateEvent && ['complete', 'completed', 'success', 'finished', 'failed', 'stopped'].includes(status);
    const displayPercent = data.percent !== undefined ? data.percent : (isTerminal ? 100 : 0);
    if (data.type !== 'heartbeat' || isTerminal) {
      log(data.message || data.step || data.type, data);
    }
    if (data.percent !== undefined || isTerminal) {
      const label = isTerminal && status === 'complete' ? 'Job complete' : (data.status || 'running');
      setProgress(displayPercent, label, data.step || data.message || (isTerminal ? 'Complete' : 'Running'));
    }

    if (isJobStateEvent && (status === 'complete' || status === 'completed' || status === 'success' || status === 'finished')) {
      setProgress(100, 'Job complete', data.step || 'Complete');
      if ($('completedStep')) $('completedStep').textContent = `Completed step: ${data.step || 'Complete'} (100%)`;
      currentOutputPath = data.output_dir || (data.result && data.result.output_dir) || currentOutputPath;
      if ($('outputPath')) $('outputPath').textContent = currentOutputPath || '';
      if ($('resultJson')) $('resultJson').textContent = JSON.stringify(data.result || data, null, 2);
      const r = data.result || {};
      await handleCompletedNeuroMouseResult(r);
      if (r.primary_neuromouse_url) openBackendNeuroMouseUrl(r.primary_neuromouse_url, 'Open generated NeuroMouse dataset');
      if (r.neuromouse_comparison_url) openBackendNeuroMouseUrl(r.neuromouse_comparison_url, 'Open generated NeuroMouse comparison');
      try { ws.close(); } catch (_) {}
      return;
    }

    if (isJobStateEvent && status === 'failed') {
      setProgress(100, 'Job failed', data.step || 'Failed');
      if ($('jobStatus')) $('jobStatus').className = 'status failed big-status';
      if ($('resultJson')) $('resultJson').textContent = JSON.stringify(data, null, 2);
      try { ws.close(); } catch (_) {}
      return;
    }

    if (status === 'running' && $('jobStatus')) $('jobStatus').className = 'status big-status';
  };

  ws.onmessage = async (ev) => {
    const data = JSON.parse(ev.data);
    if (data.type === 'heartbeat' && !data.status) return;
    await applyJobEvent(data);
  };
  ws.onerror = () => log('WebSocket error');
};

async function fetchLatestNeuroMouseInfo() {
  try {
    const res = await fetch('/api/neuromouse/latest?t=' + Date.now(), { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

async function openLatestNeuroMouse() {
  // Always ask the backend first. A browser-stored /neuromouse-job/<old_id>/
  // URL can become stale after restart or workspace cleanup and caused 404s.
  const latest = await fetchLatestNeuroMouseInfo();
  if (latest?.neuromouse_url) {
    latestNeuroMouseUrl = latest.neuromouse_url;
    try {
      if (latest.dataset_url) localStorage.setItem('NEURO_SIGNAL_LAST_BACKEND_DATASET_URL', latest.dataset_url);
      if (latest.job_dataset_url) localStorage.setItem('NEURO_SIGNAL_LAST_JOB_DATASET_URL', latest.job_dataset_url);
      localStorage.setItem('NEURO_SIGNAL_LAST_NEUROMOUSE_URL', latest.neuromouse_url);
    } catch {}
    openUrl(latest.neuromouse_url);
    return;
  }
  if (latestNeuroMouseUrl) { openUrl(latestNeuroMouseUrl); return; }
  try {
    const stored = localStorage.getItem('NEURO_SIGNAL_LAST_NEUROMOUSE_URL');
    if (stored && (stored.includes('/neuromouse-latest/') || stored.includes('/neuromouse-job/'))) { openUrl(stored); return; }
  } catch {}
  alert('No generated NeuroMouse dataset found yet. Convert or Analyze a dataset first, then use this button.');
}
function openLatestNeuroMouseComparison() { if (latestNeuroMouseComparisonUrl) openUrl(latestNeuroMouseComparisonUrl); else alert('No NeuroMouse comparison has been generated yet.'); }

if ($('openLatestNeuroMouseBtn')) $('openLatestNeuroMouseBtn').addEventListener('click', openLatestNeuroMouse);
if ($('openLatestNeuroMouseComparisonBtn')) $('openLatestNeuroMouseComparisonBtn').addEventListener('click', openLatestNeuroMouseComparison);
if ($('resultsOpenLatestNeuroMouseBtn')) $('resultsOpenLatestNeuroMouseBtn').addEventListener('click', openLatestNeuroMouse);
if ($('resultsOpenLatestNeuroMouseComparisonBtn')) $('resultsOpenLatestNeuroMouseComparisonBtn').addEventListener('click', openLatestNeuroMouseComparison);

// ---- v0.9.5 saved raw backend job log output ----
let latestRawLogUrl = null;
let latestRawJsonlUrl = null;
let rawLogRefreshTimer = null;

function setRawJobLogLinks(jobId) {
  if (!jobId) return;
  latestRawLogUrl = `/api/jobs/${jobId}/raw-log.txt`;
  latestRawJsonlUrl = `/api/jobs/${jobId}/raw-log.jsonl`;
  const viewBtn = $('viewRawLogBtn');
  const dlBtn = $('downloadRawLogBtn');
  const jsonlBtn = $('downloadRawJsonlBtn');
  if (viewBtn) { viewBtn.href = latestRawLogUrl; viewBtn.classList.remove('disabled'); }
  if (dlBtn) { dlBtn.href = latestRawLogUrl; dlBtn.classList.remove('disabled'); }
  if (jsonlBtn) { jsonlBtn.href = latestRawJsonlUrl; jsonlBtn.classList.remove('disabled'); }
  if ($('rawLogPath')) $('rawLogPath').textContent = `Raw job log: ${latestRawLogUrl} · JSONL: ${latestRawJsonlUrl}`;
}

async function refreshRawLogPreview(jobId) {
  if (!jobId || !$('rawLogPreview')) return;
  try {
    const res = await fetch(`/api/jobs/${jobId}/raw-log.txt?t=${Date.now()}`);
    if (!res.ok) return;
    const text = await res.text();
    const lines = text.split('\n');
    $('rawLogPreview').textContent = lines.slice(Math.max(0, lines.length - 160)).join('\n');
  } catch (err) {
    $('rawLogPreview').textContent = `Raw job log preview failed: ${err.message}`;
  }
}

const _connectJobEventsBeforeRawLogPatch = connectJobEvents;
connectJobEvents = function(jobId) {
  setRawJobLogLinks(jobId);
  if (rawLogRefreshTimer) clearInterval(rawLogRefreshTimer);
  refreshRawLogPreview(jobId);
  rawLogRefreshTimer = setInterval(() => refreshRawLogPreview(jobId), 1200);
  return _connectJobEventsBeforeRawLogPatch(jobId);
};



// ---- v0.11.2 NeuroMouse advanced plot sections in the launcher ----
let kuramotoFrameIndex = 0;
let kuramotoPlaying = true;
let kuramotoAnimationHandle = null;

function plotEscape(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));
}

function finiteNumbers(values) {
  return (Array.isArray(values) ? values : []).map(Number).filter(Number.isFinite);
}

function plotCanvas(id, title, missingMessage) {
  const canvas = $(id);
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#010505';
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(0,229,229,0.35)';
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.fillStyle = '#dfffff';
  ctx.font = '15px monospace';
  ctx.fillText(title, 16, 26);
  if (missingMessage) {
    ctx.fillStyle = '#7bdada';
    ctx.font = '13px monospace';
    wrapPlotText(ctx, missingMessage, 16, 56, w - 32, 18);
  }
  return { canvas, ctx, w, h };
}

function wrapPlotText(ctx, text, x, y, maxWidth, lineHeight) {
  const words = String(text || '').split(/\s+/);
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      ctx.fillText(line, x, y);
      line = word;
      y += lineHeight;
    } else {
      line = test;
    }
  }
  if (line) ctx.fillText(line, x, y);
}

function setPlotReadout(id, html) {
  const node = $(id);
  if (node) node.innerHTML = html;
}

function interpolateColor(value, min, max) {
  const t = Math.max(0, Math.min(1, (Number(value) - min) / ((max - min) || 1)));
  const r = Math.round(18 + t * 237);
  const g = Math.round(120 + (1 - Math.abs(t - 0.5) * 2) * 115);
  const b = Math.round(220 - t * 170);
  return `rgb(${r},${g},${b})`;
}

function drawPolarAlphaChronomap(data) {
  const p = data?.polar_chronomap;
  const values = finiteNumbers(p?.posterior_alpha);
  const balance = finiteNumbers(p?.balance);
  const frontal = finiteNumbers(p?.frontal_alpha);
  const plot = plotCanvas('polarAlphaChronomapCanvas', 'Polar Alpha Chronomap', values.length ? '' : 'No polar_chronomap arrays found. Re-run Analyze in NeuroMouse so the backend writes polar_chronomap into data.json.');
  if (!plot || !values.length) {
    setPlotReadout('polarAlphaChronomapSummary', 'Missing <code>polar_chronomap</code> in the loaded NeuroMouse dataset.');
    return;
  }
  const { ctx, w, h } = plot;
  const cx = w / 2;
  const cy = h / 2 + 14;
  const radius = Math.min(w, h) * 0.34;
  const inner = radius * 0.52;
  const allAlpha = values.concat(frontal).filter(Number.isFinite);
  const minAlpha = Math.min(...allAlpha);
  const maxAlpha = Math.max(...allAlpha);
  const minBal = balance.length ? Math.min(...balance) : -1;
  const maxBal = balance.length ? Math.max(...balance) : 1;

  ctx.strokeStyle = 'rgba(127,255,255,0.16)';
  ctx.lineWidth = 1;
  for (let ring = 0; ring < 4; ring++) {
    ctx.beginPath();
    ctx.arc(cx, cy, inner + (radius - inner) * ring / 3, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let i = 0; i < values.length; i++) {
    const angle = -Math.PI / 2 + Math.PI * 2 * i / Math.max(1, values.length - 1);
    const alphaT = (values[i] - minAlpha) / ((maxAlpha - minAlpha) || 1);
    const r1 = inner;
    const r2 = inner + (radius - inner) * Math.max(0.05, alphaT);
    ctx.strokeStyle = interpolateColor(balance[i] ?? 0, minBal, maxBal);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * r1, cy + Math.sin(angle) * r1);
    ctx.lineTo(cx + Math.cos(angle) * r2, cy + Math.sin(angle) * r2);
    ctx.stroke();
  }
  ctx.fillStyle = '#00ff99';
  ctx.font = '13px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('posterior alpha', cx, cy - 4);
  ctx.fillStyle = '#7bdada';
  ctx.fillText(`${values.length} frames`, cx, cy + 16);
  ctx.textAlign = 'left';

  const postMean = values.reduce((a, b) => a + b, 0) / values.length;
  const frontMean = frontal.length ? frontal.reduce((a, b) => a + b, 0) / frontal.length : NaN;
  const balMean = balance.length ? balance.reduce((a, b) => a + b, 0) / balance.length : NaN;
  setPlotReadout('polarAlphaChronomapSummary', `Posterior channels: <strong>${plotEscape((p?.posterior_channels || []).length)}</strong> · Frontal channels: <strong>${plotEscape((p?.frontal_channels || []).length)}</strong> · Mean posterior alpha: <strong>${postMean.toFixed(4)}</strong> · Mean frontal alpha: <strong>${Number.isFinite(frontMean) ? frontMean.toFixed(4) : '—'}</strong> · Mean balance: <strong>${Number.isFinite(balMean) ? balMean.toFixed(4) : '—'}</strong>`);
}

function drawKuramotoFrame(data) {
  const k = data?.kuramoto;
  const phases = Array.isArray(k?.channel_phases) ? k.channel_phases : [];
  const time = Array.isArray(k?.time) ? k.time : [];
  const r = Array.isArray(k?.order_parameter_r) ? k.order_parameter_r : [];
  const psi = Array.isArray(k?.mean_phase_psi) ? k.mean_phase_psi : [];
  const nFrames = Math.max(time.length, r.length, psi.length, ...phases.map(row => Array.isArray(row) ? row.length : 0));
  const plot = plotCanvas('kuramotoAnimationCanvas', 'Kuramoto Animation', nFrames ? '' : 'No Kuramoto arrays found. Re-run Analyze in NeuroMouse so the backend writes alpha-band phase synchrony into data.json.');
  if (!plot || !nFrames) {
    setPlotReadout('kuramotoReadout', 'Missing <code>kuramoto</code> in the loaded NeuroMouse dataset.');
    return;
  }
  const { ctx, w, h } = plot;
  const cx = w / 2;
  const cy = h / 2 + 12;
  const radius = Math.min(w, h) * 0.33;
  const frame = Math.max(0, Math.min(nFrames - 1, kuramotoFrameIndex % nFrames));

  ctx.strokeStyle = 'rgba(127,255,255,0.28)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = 'rgba(127,255,255,0.12)';
  ctx.beginPath(); ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius); ctx.stroke();

  phases.forEach((series, idx) => {
    if (!Array.isArray(series)) return;
    const phase = Number(series[Math.min(frame, series.length - 1)]);
    if (!Number.isFinite(phase)) return;
    const x = cx + Math.cos(phase) * radius;
    const y = cy + Math.sin(phase) * radius;
    ctx.fillStyle = idx === 0 ? '#00ff99' : '#00e5e5';
    ctx.globalAlpha = idx === 0 ? 1 : 0.72;
    ctx.beginPath();
    ctx.arc(x, y, idx === 0 ? 5 : 3.5, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
  const rr = Number(r[frame] ?? 0);
  const pp = Number(psi[frame] ?? 0);
  if (Number.isFinite(rr) && Number.isFinite(pp)) {
    ctx.strokeStyle = '#ffd166';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(pp) * radius * rr, cy + Math.sin(pp) * radius * rr);
    ctx.stroke();
  }
  ctx.fillStyle = '#dfffff';
  ctx.font = '13px monospace';
  ctx.fillText(`frame ${frame + 1}/${nFrames}`, 16, h - 42);
  ctx.fillText(`time ${Number(time[frame] ?? 0).toFixed(3)} s`, 16, h - 22);
  setPlotReadout('kuramotoReadout', `Frame <strong>${frame + 1}</strong> / ${nFrames} · channels <strong>${phases.length}</strong> · order parameter r <strong>${Number.isFinite(rr) ? rr.toFixed(4) : '—'}</strong> · mean phase ψ <strong>${Number.isFinite(pp) ? pp.toFixed(4) : '—'}</strong>`);
}

function startKuramotoAnimation() {
  if (kuramotoAnimationHandle) cancelAnimationFrame(kuramotoAnimationHandle);
  let last = 0;
  const tick = (stamp) => {
    if (latestNeuroMouseData?.kuramoto && kuramotoPlaying && (!last || stamp - last > 120)) {
      kuramotoFrameIndex += 1;
      drawKuramotoFrame(latestNeuroMouseData);
      last = stamp;
    }
    kuramotoAnimationHandle = requestAnimationFrame(tick);
  };
  kuramotoAnimationHandle = requestAnimationFrame(tick);
}

function networkMatrixForSelection(data) {
  const net = data?.channel_network;
  const selected = $('channelNetworkMetricSelect')?.value || 'composite_correlation';
  if (selected === 'composite_correlation') return net?.composite_correlation || [];
  if (selected.startsWith('per_metric:')) return net?.per_metric?.[selected.slice('per_metric:'.length)] || [];
  if (selected === 'plv_alpha') return data?.phase_synchrony?.plv_alpha || [];
  return net?.composite_correlation || [];
}

function populateChannelNetworkMetrics(data) {
  const select = $('channelNetworkMetricSelect');
  if (!select) return;
  const current = select.value;
  const options = [{ value: 'composite_correlation', label: 'Composite correlation' }];
  const perMetric = data?.channel_network?.per_metric || {};
  Object.keys(perMetric).forEach(name => options.push({ value: `per_metric:${name}`, label: `Metric: ${name}` }));
  if (Array.isArray(data?.phase_synchrony?.plv_alpha)) options.push({ value: 'plv_alpha', label: 'PLV alpha' });
  select.innerHTML = options.map(opt => `<option value="${plotEscape(opt.value)}">${plotEscape(opt.label)}</option>`).join('');
  if (options.some(opt => opt.value === current)) select.value = current;
}

function drawChannelNetwork(data) {
  const net = data?.channel_network;
  const channels = Array.isArray(net?.channels) ? net.channels : (data?.meta?.channels || []);
  const matrix = networkMatrixForSelection(data);
  const plot = plotCanvas('channelNetworkCanvas', 'Channel Network', Array.isArray(matrix) && matrix.length ? '' : 'No channel_network matrix found. Re-run Analyze in NeuroMouse so the backend writes channel_network into data.json.');
  if (!plot || !Array.isArray(matrix) || !matrix.length) {
    setPlotReadout('channelNetworkReadout', 'Missing <code>channel_network</code> in the loaded NeuroMouse dataset.');
    return;
  }
  const { ctx, w, h } = plot;
  const threshold = Number($('channelNetworkThreshold')?.value || net?.threshold_strong || 0.7);
  const showWeak = Boolean($('channelNetworkWeakEdges')?.checked);
  const n = Math.min(channels.length || matrix.length, matrix.length, 80);
  const cx = w / 2;
  const cy = h / 2 + 10;
  const radius = Math.min(w, h) * 0.34;
  const positions = Array.from({ length: n }, (_, i) => {
    const angle = -Math.PI / 2 + Math.PI * 2 * i / n;
    return { x: cx + Math.cos(angle) * radius, y: cy + Math.sin(angle) * radius, angle };
  });
  let strongEdges = 0;
  let weakEdges = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const value = Math.abs(Number(matrix[i]?.[j]));
      if (!Number.isFinite(value)) continue;
      const isStrong = value >= threshold;
      if (!isStrong && !showWeak) continue;
      if (isStrong) strongEdges += 1; else weakEdges += 1;
      const p1 = positions[i];
      const p2 = positions[j];
      ctx.strokeStyle = isStrong ? `rgba(0,255,153,${Math.min(0.9, 0.25 + value * 0.65)})` : 'rgba(127,255,255,0.12)';
      ctx.lineWidth = isStrong ? Math.max(1.2, value * 3) : 0.75;
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }
  positions.forEach((p, i) => {
    const degree = (matrix[i] || []).slice(0, n).reduce((acc, v, j) => i !== j && Math.abs(Number(v)) >= threshold ? acc + 1 : acc, 0);
    ctx.fillStyle = degree ? '#00ff99' : '#00e5e5';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5 + Math.min(8, degree), 0, Math.PI * 2);
    ctx.fill();
    if (n <= 40 || i % Math.ceil(n / 32) === 0) {
      ctx.fillStyle = '#dfffff';
      ctx.font = '10px monospace';
      const lx = cx + Math.cos(p.angle) * (radius + 24);
      const ly = cy + Math.sin(p.angle) * (radius + 24);
      ctx.textAlign = lx < cx ? 'right' : 'left';
      ctx.fillText(String(channels[i] || i), lx, ly);
    }
  });
  ctx.textAlign = 'left';
  setPlotReadout('channelNetworkReadout', `Channels <strong>${n}</strong> · threshold <strong>${threshold.toFixed(2)}</strong> · strong edges <strong>${strongEdges}</strong>${showWeak ? ` · weak edges shown <strong>${weakEdges}</strong>` : ''}`);
}

function drawTdaView(data) {
  const tda = data?.tda;
  const h0 = Array.isArray(tda?.h0) ? tda.h0 : [];
  const h1 = Array.isArray(tda?.h1) ? tda.h1 : [];
  const plot = plotCanvas('tdaViewCanvas', 'TDA View', (h0.length || h1.length) ? '' : 'No TDA persistence arrays found. Re-run Analyze in NeuroMouse so the backend writes tda into data.json.');
  if (!plot || (!h0.length && !h1.length)) {
    setPlotReadout('tdaSummary', 'Missing computed <code>tda</code> persistence arrays in the loaded NeuroMouse dataset.');
    return;
  }
  const { ctx, w, h } = plot;
  const pairs = h0.concat(h1).map(pair => [Number(pair?.[0]), Number(pair?.[1])]).filter(([b, d]) => Number.isFinite(b) && Number.isFinite(d));
  const maxDeath = Math.max(...pairs.map(([, d]) => d), 1);
  const minBirth = Math.min(...pairs.map(([b]) => b), 0);
  const left = 54, top = 52, scatterW = w * 0.43, scatterH = h - 92;
  const sx = (x) => left + ((x - minBirth) / ((maxDeath - minBirth) || 1)) * scatterW;
  const sy = (y) => top + scatterH - ((y - minBirth) / ((maxDeath - minBirth) || 1)) * scatterH;
  ctx.strokeStyle = 'rgba(127,255,255,0.18)';
  ctx.strokeRect(left, top, scatterW, scatterH);
  ctx.strokeStyle = 'rgba(255,255,255,0.22)';
  ctx.beginPath();
  ctx.moveTo(sx(minBirth), sy(minBirth));
  ctx.lineTo(sx(maxDeath), sy(maxDeath));
  ctx.stroke();
  h0.forEach(pair => drawTdaPoint(ctx, sx(Number(pair?.[0])), sy(Number(pair?.[1])), '#00e5e5'));
  h1.forEach(pair => drawTdaPoint(ctx, sx(Number(pair?.[0])), sy(Number(pair?.[1])), '#00ff99'));
  ctx.fillStyle = '#7bdada';
  ctx.font = '11px monospace';
  ctx.fillText('birth', left + scatterW - 32, h - 14);
  ctx.save();
  ctx.translate(16, top + 90);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('death', 0, 0);
  ctx.restore();

  const barLeft = left + scatterW + 64;
  const barTop = top;
  const barW = w - barLeft - 24;
  const allBars = h0.map(p => ({ p, type: 'H0' })).concat(h1.map(p => ({ p, type: 'H1' }))).slice(0, 70);
  ctx.fillStyle = '#dfffff';
  ctx.font = '13px monospace';
  ctx.fillText('Barcode', barLeft, 30);
  allBars.forEach((item, i) => {
    const b = Number(item.p?.[0]);
    const d = Number(item.p?.[1]);
    if (!Number.isFinite(b) || !Number.isFinite(d)) return;
    const y = barTop + i * Math.max(3, Math.min(9, (h - barTop - 20) / Math.max(1, allBars.length)));
    const x1 = barLeft + ((b - minBirth) / ((maxDeath - minBirth) || 1)) * barW;
    const x2 = barLeft + ((d - minBirth) / ((maxDeath - minBirth) || 1)) * barW;
    ctx.strokeStyle = item.type === 'H1' ? '#00ff99' : 'rgba(0,229,229,0.72)';
    ctx.lineWidth = item.type === 'H1' ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(x1, y);
    ctx.lineTo(Math.max(x1 + 2, x2), y);
    ctx.stroke();
  });
  const lifetimes = pairs.map(([b, d]) => d - b).filter(Number.isFinite);
  const maxLife = lifetimes.length ? Math.max(...lifetimes) : NaN;
  setPlotReadout('tdaSummary', `Status <strong>${plotEscape(tda?.status || 'unknown')}</strong> · H0 points <strong>${h0.length}</strong> · H1 points <strong>${h1.length}</strong> · point cloud rows <strong>${Array.isArray(tda?.point_cloud) ? tda.point_cloud.length : 0}</strong> · max lifetime <strong>${Number.isFinite(maxLife) ? maxLife.toFixed(4) : '—'}</strong>`);
}

function drawTdaPoint(ctx, x, y, color) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.82;
  ctx.beginPath();
  ctx.arc(x, y, 3.5, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}

function renderNeuroMouseAdvancedPlots(data) {
  if (!data) return;
  populateChannelNetworkMetrics(data);
  drawPolarAlphaChronomap(data);
  drawKuramotoFrame(data);
  drawChannelNetwork(data);
  drawTdaView(data);
  const status = $('advancedPlotsStatus');
  if (status) {
    const nChannels = data?.meta?.channels?.length || data?.meta?.n_channels || 0;
    const frames = data?.geometry?.time?.length || data?.centroid?.time_relative?.length || 0;
    status.textContent = `Loaded ${data?.meta?.dataset_id || 'NeuroMouse dataset'} · ${nChannels} channels · ${frames} frames · advanced plot objects: polar=${Boolean(data?.polar_chronomap)}, kuramoto=${Boolean(data?.kuramoto)}, network=${Boolean(data?.channel_network)}, tda=${Boolean(data?.tda)}`;
  }
  if (!$('neuromouseTab')?.classList.contains('active')) return;
  startKuramotoAnimation();
}

async function fetchNeuroMouseDatasetWithDemoFallback() {
  const urls = [
    { url: '/api/neuromouse/latest/data.json?t=' + Date.now(), label: 'latest backend NeuroMouse analysis' },
    { url: '/neuromouse/data/data.json?t=' + Date.now(), label: 'bundled original NeuroMouse demo' }
  ];
  const errors = [];
  for (const item of urls) {
    try {
      const data = await fetchJsonMaybe(item.url);
      return { data, label: item.label, url: item.url };
    } catch (err) {
      errors.push(`${item.label}: ${err.message}`);
    }
  }
  throw new Error(errors.join(' | '));
}

async function loadLatestAdvancedPlots() {
  ensureAdvancedAnalysisVisible(true);
  const status = $('advancedPlotsStatus');
  if (status) status.textContent = 'Loading NeuroMouse advanced plots; falling back to bundled original demo if no backend analysis exists...';
  try {
    const loaded = await fetchNeuroMouseDatasetWithDemoFallback();
    latestNeuroMouseData = loaded.data;
    renderNeuroMouseAdvancedPlots(loaded.data);
    if (status) {
      const nChannels = loaded.data?.meta?.channels?.length || loaded.data?.meta?.n_channels || 0;
      const nFrames = loaded.data?.geometry?.time?.length || loaded.data?.kuramoto?.time?.length || 0;
      status.textContent = `Loaded ${loaded.label} · ${nChannels} channels · ${nFrames} frames · polar=${Boolean(loaded.data?.polar_chronomap)} · kuramoto=${Boolean(loaded.data?.kuramoto)} · network=${Boolean(loaded.data?.channel_network)} · tda=${Boolean(loaded.data?.tda)}`;
    }
    showTab('advancedMethodsTab');
  } catch (err) {
    if (status) status.textContent = `Could not load NeuroMouse plots from backend or demo: ${err.message}`;
    drawPolarAlphaChronomap(null);
    drawKuramotoFrame(null);
    drawChannelNetwork(null);
    drawTdaView(null);
  }
}

if ($('loadLatestAdvancedPlotsBtn')) $('loadLatestAdvancedPlotsBtn').addEventListener('click', loadLatestAdvancedPlots);
if ($('loadLatestAdvancedPlotsFromImportBtn')) $('loadLatestAdvancedPlotsFromImportBtn').addEventListener('click', loadLatestAdvancedPlots);
if ($('jumpAdvancedPlotsBtn')) $('jumpAdvancedPlotsBtn').addEventListener('click', () => scrollToHiguchiDirect());
if ($('jumpEmbeddedFdBtn')) $('jumpEmbeddedFdBtn').addEventListener('click', () => { showTab('embeddedFdTab'); setTimeout(() => $('embeddedFdDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
if ($('goImportFromPlotsBtn')) $('goImportFromPlotsBtn').addEventListener('click', () => showTab('importTab'));
if ($('openFullNeuroMouseWorkbenchBtn')) $('openFullNeuroMouseWorkbenchBtn').addEventListener('click', openLatestNeuroMouse);
if ($('kuramotoPlayPauseBtn')) $('kuramotoPlayPauseBtn').addEventListener('click', () => { kuramotoPlaying = !kuramotoPlaying; if (latestNeuroMouseData) drawKuramotoFrame(latestNeuroMouseData); });
if ($('kuramotoResetBtn')) $('kuramotoResetBtn').addEventListener('click', () => { kuramotoFrameIndex = 0; if (latestNeuroMouseData) drawKuramotoFrame(latestNeuroMouseData); });
if ($('channelNetworkMetricSelect')) $('channelNetworkMetricSelect').addEventListener('change', () => latestNeuroMouseData && drawChannelNetwork(latestNeuroMouseData));
if ($('channelNetworkThreshold')) $('channelNetworkThreshold').addEventListener('input', () => latestNeuroMouseData && drawChannelNetwork(latestNeuroMouseData));
if ($('channelNetworkWeakEdges')) $('channelNetworkWeakEdges').addEventListener('change', () => latestNeuroMouseData && drawChannelNetwork(latestNeuroMouseData));


// ---- v0.11.2 native Advanced Methods runner ----
let advancedMethods = [];
let latestAdvancedResult = null;

function advEscape(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[ch]));
}

function advFormat(value) {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return String(value);
    if (Math.abs(value) >= 100) return value.toFixed(2);
    return value.toFixed(4);
  }
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (Array.isArray(value)) return `${value.length} item${value.length === 1 ? '' : 's'}`;
  if (typeof value === 'object') return Object.entries(value).slice(0, 6).map(([k, v]) => `${k}: ${advFormat(v)}`).join(', ');
  return String(value);
}

function advPath(obj, path) {
  if (!path) return obj;
  return String(path).split('.').reduce((cur, part) => (cur == null ? undefined : cur[part]), obj);
}

function advColumns(rows) {
  const keys = new Set();
  (rows || []).slice(0, 100).forEach(row => {
    if (row && typeof row === 'object' && !Array.isArray(row)) Object.keys(row).forEach(k => keys.add(k));
  });
  return Array.from(keys);
}

function advRenderTable(rows) {
  const safeRows = Array.isArray(rows) ? rows : [];
  if (!safeRows.length) return '<p>No rows returned.</p>';
  const cols = advColumns(safeRows);
  const shown = safeRows.slice(0, 500);
  return `<table><thead><tr>${cols.map(c => `<th>${advEscape(c)}</th>`).join('')}</tr></thead><tbody>${shown.map(row => `<tr>${cols.map(c => `<td>${advEscape(advFormat(row?.[c]))}</td>`).join('')}</tr>`).join('')}</tbody></table>${safeRows.length > shown.length ? `<p class="hint">Showing first ${shown.length} of ${safeRows.length} rows.</p>` : ''}`;
}

function advRenderMatrix(matrix, channels) {
  const rows = Array.isArray(matrix) ? matrix.filter(Array.isArray) : [];
  if (!rows.length) return '<p>No matrix returned.</p>';
  const max = Math.min(rows.length, 80);
  const names = Array.isArray(channels) ? channels : [];
  let html = '<table class="matrix-table"><thead><tr><th></th>';
  for (let i = 0; i < max; i++) html += `<th>${advEscape(names[i] || i)}</th>`;
  html += '</tr></thead><tbody>';
  for (let r = 0; r < max; r++) {
    html += `<tr><th>${advEscape(names[r] || r)}</th>`;
    for (let c = 0; c < Math.min(rows[r].length, max); c++) html += `<td>${advEscape(advFormat(rows[r][c]))}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table>';
  if (rows.length > max) html += `<p class="hint">Showing ${max} × ${max} of a ${rows.length} × ${rows.length} matrix.</p>`;
  return html;
}

function advRenderSummary(summary) {
  const target = $('advancedSummary');
  if (!target) return;
  if (!summary || typeof summary !== 'object') { target.innerHTML = ''; return; }
  const entries = Object.entries(summary).filter(([_, v]) => typeof v !== 'object' || v === null).slice(0, 10);
  target.innerHTML = entries.map(([k, v]) => `<div class="metric-tile"><span>${advEscape(k)}</span><strong>${advEscape(advFormat(v))}</strong></div>`).join('');
}



// ---- v0.11.13 Higuchi fractal dimension frontend plots ----
function hfdRows(data) {
  return Array.isArray(data?.rows) ? data.rows.filter(r => Number.isFinite(Number(r?.higuchi_fd))) : [];
}

function hfdValueRange(values, padFrac = 0.08) {
  const finite = (values || []).map(Number).filter(Number.isFinite);
  if (!finite.length) return { min: 0, max: 1 };
  let min = Math.min(...finite);
  let max = Math.max(...finite);
  if (max === min) { min -= 0.01; max += 0.01; }
  const pad = Math.max((max - min) * padFrac, 0.001);
  return { min: min - pad, max: max + pad };
}

function hfdMapPoint(value, min, max, a, b) {
  const t = (Number(value) - min) / ((max - min) || 1);
  return a + Math.max(0, Math.min(1, t)) * (b - a);
}

function hfdFormatTick(value) {
  const v = Number(value);
  if (!Number.isFinite(v)) return '';
  const av = Math.abs(v);
  if (av >= 1000 || (av > 0 && av < 0.001)) return v.toExponential(1);
  if (av >= 100) return v.toFixed(0);
  if (av >= 10) return v.toFixed(1);
  return v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
}

function hfdTickValues(min, max, count = 5) {
  const a = Number(min), b = Number(max);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return [];
  if (Math.abs(b - a) < 1e-12) return [a];
  const ticks = [];
  for (let i = 0; i < count; i++) ticks.push(a + (b - a) * i / Math.max(1, count - 1));
  return ticks;
}

function hfdDrawAxes(ctx, left, top, width, height, xLabel, yLabel, xRange = null, yRange = null, opts = {}) {
  ctx.save();
  ctx.strokeStyle = 'rgba(127,255,255,0.34)';
  ctx.lineWidth = 1;
  ctx.strokeRect(left, top, width, height);

  const xTicks = opts.xTicks || (xRange ? hfdTickValues(xRange.min, xRange.max, opts.xTickCount || 5) : []);
  const yTicks = opts.yTicks || (yRange ? hfdTickValues(yRange.min, yRange.max, opts.yTickCount || 5) : []);

  ctx.font = opts.tickFont || '11px monospace';
  ctx.lineWidth = 1;
  ctx.textBaseline = 'middle';
  ctx.fillStyle = '#dfffff';
  ctx.strokeStyle = 'rgba(0,229,229,0.16)';

  if (yRange && yTicks.length) {
    yTicks.forEach(tick => {
      const y = hfdMapPoint(tick, yRange.min, yRange.max, top + height, top);
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + width, y); ctx.stroke();
      ctx.strokeStyle = 'rgba(127,255,255,0.45)';
      ctx.beginPath(); ctx.moveTo(left - 5, y); ctx.lineTo(left, y); ctx.stroke();
      ctx.strokeStyle = 'rgba(0,229,229,0.16)';
      ctx.textAlign = 'right';
      ctx.fillText(hfdFormatTick(tick), left - 8, y);
    });
  }

  if (xRange && xTicks.length) {
    xTicks.forEach(tick => {
      const x = hfdMapPoint(tick, xRange.min, xRange.max, left, left + width);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke();
      ctx.strokeStyle = 'rgba(127,255,255,0.45)';
      ctx.beginPath(); ctx.moveTo(x, top + height); ctx.lineTo(x, top + height + 5); ctx.stroke();
      ctx.strokeStyle = 'rgba(0,229,229,0.16)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(hfdFormatTick(tick), x, top + height + 8);
      ctx.textBaseline = 'middle';
    });
  }

  ctx.fillStyle = '#7bdada';
  ctx.font = opts.labelFont || '12px monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'alphabetic';
  if (xLabel) ctx.fillText(xLabel, left + width / 2, top + height + (xTicks.length ? 38 : 34));
  if (yLabel) {
    ctx.save();
    ctx.translate(Math.max(14, left - (yTicks.length ? 52 : 38)), top + height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
  ctx.restore();
}

function hfdDrawBarPlot(data) {
  const rows = hfdRows(data).slice().sort((a, b) => Number(b.higuchi_fd) - Number(a.higuchi_fd));
  const plot = plotCanvas('higuchiFdBarCanvas', 'Higuchi FD by Channel', rows.length ? '' : 'Run the Higuchi fractal dimension backend method to populate this plot.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 54, top = 48, width = w - 80, height = h - 110;
  const vals = rows.map(r => Number(r.higuchi_fd));
  const range = hfdValueRange(vals, 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'Channel rank', 'HFD', {min: 1, max: rows.length}, range);
  const barW = Math.max(2, width / rows.length * 0.72);
  rows.forEach((row, i) => {
    const x = left + (i + 0.15) * width / rows.length;
    const y = hfdMapPoint(row.higuchi_fd, range.min, range.max, top + height, top);
    ctx.fillStyle = interpolateColor(row.higuchi_fd, range.min, range.max);
    ctx.fillRect(x, y, barW, top + height - y);
    if (rows.length <= 48) {
      ctx.save();
      ctx.translate(x + barW / 2, h - 50);
      ctx.rotate(-Math.PI / 2);
      ctx.fillStyle = '#dfffff';
      ctx.font = '10px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(String(row.channel), 0, 0);
      ctx.restore();
    }
  });
  ctx.fillStyle = '#7bdada';
  ctx.font = '11px monospace';
  ctx.fillText(`max ${range.max.toFixed(3)}`, 10, top + 6);
  ctx.fillText(`min ${range.min.toFixed(3)}`, 10, top + height);
}

function hfdDrawFitsGrid(data) {
  const curves = Array.isArray(data?.curves) ? data.curves.slice(0, 32) : [];
  const plot = plotCanvas('higuchiFitsGridCanvas', 'Log L(k) Fits — all channels', curves.length ? '' : 'No Higuchi fit curves returned.');
  if (!plot || !curves.length) return;
  const { ctx, w, h } = plot;
  const cols = 4;
  const rows = Math.ceil(curves.length / cols);
  const gap = 16;
  const cellW = (w - 40 - gap * (cols - 1)) / cols;
  const cellH = (h - 56 - gap * (rows - 1)) / rows;
  curves.forEach((curve, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const left = 20 + col * (cellW + gap);
    const top = 42 + row * (cellH + gap);
    const logK = finiteNumbers(curve.log_k);
    const logL = finiteNumbers(curve.log_Lk);
    const fit = finiteNumbers(curve.fit_log_Lk);
    if (logK.length < 2 || logL.length < 2) return;
    const xr = hfdValueRange(logK, 0.08);
    const yr = hfdValueRange(logL.concat(fit), 0.12);
    ctx.strokeStyle = 'rgba(127,255,255,0.18)';
    ctx.strokeRect(left, top, cellW, cellH);
    ctx.beginPath();
    logK.forEach((xv, i) => {
      const x = hfdMapPoint(xv, xr.min, xr.max, left + 10, left + cellW - 8);
      const y = hfdMapPoint(logL[i], yr.min, yr.max, top + cellH - 14, top + 12);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00e5e5';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    if (fit.length === logK.length) {
      ctx.beginPath();
      logK.forEach((xv, i) => {
        const x = hfdMapPoint(xv, xr.min, xr.max, left + 10, left + cellW - 8);
        const y = hfdMapPoint(fit[i], yr.min, yr.max, top + cellH - 14, top + 12);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = '#00ff99';
      ctx.setLineDash([4, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = '#dfffff';
    ctx.font = '10px monospace';
    ctx.fillText(`${curve.channel}: ${Number(curve.higuchi_fd).toFixed(3)}`, left + 4, top + 12);
  });
}

function hfdDrawScalpMap(data) {
  const points = Array.isArray(data?.scalp_layout?.points) ? data.scalp_layout.points : [];
  const plot = plotCanvas('higuchiScalpCanvas', 'Scalp / Channel Layout HFD Map', points.length ? '' : 'No scalp/channel layout points returned.');
  if (!plot || !points.length) return;
  const { ctx, w, h } = plot;
  const cx = w / 2, cy = h / 2 + 12;
  const radius = Math.min(w, h) * 0.34;
  const vals = points.map(p => Number(p.higuchi_fd)).filter(Number.isFinite);
  const range = hfdValueRange(vals, 0.08);
  ctx.strokeStyle = 'rgba(127,255,255,0.42)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, radius * 1.1, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(cx, cy - radius * 1.1); ctx.lineTo(cx - 14, cy - radius * 1.25); ctx.lineTo(cx + 14, cy - radius * 1.25); ctx.closePath(); ctx.stroke();
  points.forEach(point => {
    const px = cx + Number(point.x) * radius;
    const py = cy - Number(point.y) * radius;
    const value = Number(point.higuchi_fd);
    ctx.fillStyle = interpolateColor(value, range.min, range.max);
    ctx.strokeStyle = '#00e5e5';
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(px, py, 16, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#001010';
    ctx.font = 'bold 9px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(String(point.channel).slice(0, 5), px, py + 3);
  });
  ctx.textAlign = 'left';
  ctx.fillStyle = '#7bdada';
  ctx.font = '11px monospace';
  ctx.fillText(`HFD ${range.min.toFixed(3)} → ${range.max.toFixed(3)}`, 18, h - 18);
}

function hfdDrawRegionalSummary(data) {
  const rows = Array.isArray(data?.regional_summary) ? data.regional_summary : [];
  const plot = plotCanvas('higuchiRegionalCanvas', 'Regional Mean Higuchi FD ± SEM', rows.length ? '' : 'No regional summary available for these channel labels.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 64, top = 48, width = w - 92, height = h - 112;
  const vals = rows.flatMap(r => [Number(r.mean_hfd) + Number(r.sem_hfd || 0), Number(r.mean_hfd) - Number(r.sem_hfd || 0)]);
  const range = hfdValueRange(vals, 0.14);
  hfdDrawAxes(ctx, left, top, width, height, 'Region', 'Mean HFD', {min: 1, max: rows.length}, range, {xTickCount: Math.min(5, rows.length)});
  const barW = Math.max(10, width / rows.length * 0.55);
  rows.forEach((row, i) => {
    const cx = left + (i + 0.5) * width / rows.length;
    const y = hfdMapPoint(row.mean_hfd, range.min, range.max, top + height, top);
    const zero = top + height;
    ctx.fillStyle = '#00e5e5';
    ctx.globalAlpha = 0.75;
    ctx.fillRect(cx - barW / 2, y, barW, zero - y);
    ctx.globalAlpha = 1;
    const semTop = hfdMapPoint(Number(row.mean_hfd) + Number(row.sem_hfd || 0), range.min, range.max, top + height, top);
    const semBottom = hfdMapPoint(Number(row.mean_hfd) - Number(row.sem_hfd || 0), range.min, range.max, top + height, top);
    ctx.strokeStyle = '#00ff99';
    ctx.beginPath(); ctx.moveTo(cx, semTop); ctx.lineTo(cx, semBottom); ctx.stroke();
    ctx.fillStyle = '#dfffff';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(Number(row.mean_hfd).toFixed(3), cx, Math.min(y - 6, semTop - 6));
    ctx.save(); ctx.translate(cx, h - 50); ctx.rotate(-Math.PI / 5); ctx.fillText(row.region, 0, 0); ctx.restore();
  });
  ctx.textAlign = 'left';
}

function hfdDrawAsymmetry(data) {
  const rows = Array.isArray(data?.asymmetry) ? data.asymmetry : [];
  const plot = plotCanvas('higuchiAsymmetryCanvas', 'Left–Right HFD Asymmetry', rows.length ? '' : 'No 10–20 left/right channel pairs found for asymmetry.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 94, top = 48, width = w - 128, height = h - 78;
  const vals = rows.map(r => Number(r.asymmetry_left_minus_right));
  const maxAbs = Math.max(...vals.map(v => Math.abs(v)), 0.001);
  hfdDrawAxes(ctx, left, top, width, height, 'HFD left − right', 'Pair', {min: -maxAbs, max: maxAbs}, {min: 1, max: rows.length}, {yTickCount: Math.min(5, rows.length)});
  const zeroX = hfdMapPoint(0, -maxAbs, maxAbs, left, left + width);
  ctx.strokeStyle = '#00ff99'; ctx.beginPath(); ctx.moveTo(zeroX, top); ctx.lineTo(zeroX, top + height); ctx.stroke();
  const rowH = height / rows.length;
  rows.forEach((row, i) => {
    const y = top + i * rowH + rowH * 0.22;
    const x = hfdMapPoint(row.asymmetry_left_minus_right, -maxAbs, maxAbs, left, left + width);
    ctx.fillStyle = Number(row.asymmetry_left_minus_right) >= 0 ? '#00e5e5' : '#00ff99';
    ctx.fillRect(Math.min(zeroX, x), y, Math.abs(x - zeroX), Math.max(2, rowH * 0.56));
    ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.fillText(row.pair, left - 8, y + rowH * 0.45);
  });
  ctx.textAlign = 'left';
}

function hfdDrawRollingHeatmap(data) {
  const rolling = data?.rolling || {};
  const matrix = Array.isArray(rolling.matrix) ? rolling.matrix : [];
  const channels = Array.isArray(rolling.channels) ? rolling.channels : [];
  const times = Array.isArray(rolling.times_sec) ? rolling.times_sec : [];
  const plot = plotCanvas('higuchiRollingCanvas', 'Rolling Higuchi FD Stability Map', matrix.length && times.length ? '' : 'Rolling stability was disabled or returned no windows.');
  if (!plot || !matrix.length || !times.length) return;
  const { ctx, w, h } = plot;
  const left = 72, top = 46, width = w - 100, height = h - 88;
  const vals = matrix.flatMap(row => Array.isArray(row) ? row.map(Number).filter(Number.isFinite) : []);
  const range = hfdValueRange(vals, 0.04);
  const nRows = matrix.length;
  const nCols = Math.max(...matrix.map(row => Array.isArray(row) ? row.length : 0), 1);
  const cellW = width / nCols;
  const cellH = height / nRows;
  matrix.forEach((row, r) => {
    (row || []).forEach((value, c) => {
      const v = Number(value);
      if (!Number.isFinite(v)) return;
      ctx.fillStyle = interpolateColor(v, range.min, range.max);
      ctx.fillRect(left + c * cellW, top + r * cellH, Math.ceil(cellW), Math.ceil(cellH));
    });
  });
  hfdDrawAxes(ctx, left, top, width, height, 'Window start time', 'Channel');
  ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace';
  for (let i = 0; i < Math.min(18, nRows); i++) {
    const r = Math.round(i * (nRows - 1) / Math.max(1, Math.min(18, nRows) - 1));
    ctx.fillText(String(channels[r] || r), 8, top + r * cellH + cellH / 2 + 3);
  }
  ctx.fillStyle = '#7bdada';
  ctx.fillText(`${Number(times[0] || 0).toFixed(1)}s`, left, h - 18);
  ctx.fillText(`${Number(times[times.length - 1] || 0).toFixed(1)}s`, left + width - 52, h - 18);
}

function hfdDrawScatter(data) {
  const rows = Array.isArray(data?.complexity_instability) ? data.complexity_instability.filter(r => Number.isFinite(Number(r.mean_hfd)) && Number.isFinite(Number(r.rolling_hfd_std))) : [];
  const plot = plotCanvas('higuchiComplexityCanvas', 'Complexity vs Temporal Instability', rows.length ? '' : 'No rolling stability rows available.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 64, top = 48, width = w - 94, height = h - 88;
  const xr = hfdValueRange(rows.map(r => Number(r.mean_hfd)), 0.1);
  const yr = hfdValueRange(rows.map(r => Number(r.rolling_hfd_std)), 0.16);
  hfdDrawAxes(ctx, left, top, width, height, 'Mean HFD', 'Rolling HFD STD', xr, yr);
  rows.forEach(row => {
    const x = hfdMapPoint(row.mean_hfd, xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(row.rolling_hfd_std, yr.min, yr.max, top + height, top);
    ctx.fillStyle = interpolateColor(Number(row.fit_r2 || 0), 0, 1);
    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#dfffff'; ctx.font = '9px monospace'; ctx.fillText(String(row.channel).slice(0, 6), x + 6, y - 4);
  });
}

function renderHiguchiFdPlots(data) {
  if (!data) return;
  hfdDrawBarPlot(data);
  hfdDrawFitsGrid(data);
  hfdDrawScalpMap(data);
  hfdDrawRegionalSummary(data);
  hfdDrawAsymmetry(data);
  hfdDrawRollingHeatmap(data);
  hfdDrawScatter(data);
}

function renderHiguchiFdPanel(data) {
  const summary = data?.summary || {};
  const outputs = data?.outputs || {};
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Higuchi fractal dimension computed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. These plots are rendered in the frontend from backend JSON, following the same pattern as the working Advanced Analysis views.</div>
    <div class="grid two higuchi-plot-grid">
      <div class="card advanced-plot-card"><h3>HFD by Channel</h3><canvas id="higuchiFdBarCanvas" class="advanced-plot-canvas" width="820" height="440"></canvas></div>
      <div class="card advanced-plot-card"><h3>Scalp / Channel Layout Map</h3><canvas id="higuchiScalpCanvas" class="advanced-plot-canvas" width="820" height="520"></canvas></div>
      <div class="card advanced-plot-card"><h3>Regional Mean HFD ± SEM</h3><canvas id="higuchiRegionalCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
      <div class="card advanced-plot-card"><h3>Left–Right Asymmetry</h3><canvas id="higuchiAsymmetryCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
      <div class="card advanced-plot-card"><h3>Rolling HFD Stability Map</h3><canvas id="higuchiRollingCanvas" class="advanced-plot-canvas" width="820" height="520"></canvas></div>
      <div class="card advanced-plot-card"><h3>Complexity vs Temporal Instability</h3><canvas id="higuchiComplexityCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
    </div>
    <div class="card advanced-plot-card"><h3>Log-L(k) Fits for Channels</h3><canvas id="higuchiFitsGridCanvas" class="advanced-plot-canvas" width="1100" height="920"></canvas></div>
    <h3>Ranked channel table</h3>${advRenderTable(data?.rows || [])}
    <h3>Fit diagnostics</h3>${advRenderTable(data?.fit_diagnostics || [])}
    ${outputHtml}`;
}

function advExtractSummary(payload) {
  const r = payload?.result || {};
  return r.higuchi_fractal_dimension?.summary || r.embedded_fractal_dimension?.summary || r.dimension_saturation_profiling?.summary || r.katz_fractal_dimension?.summary || r.wavelet_hurst_exponent?.summary || r.manual_expert_mfdfa?.summary || r.manual_mfdfa_spectrum?.summary || r.manual_mfdfa_shuffle_surrogate?.summary || r.manual_mfdfa_iaaft_surrogate?.summary || r.wavelet_leader_multifractal?.summary || r.lyapunov_spectrum_custom?.summary || r.arnold_tongues_kuramoto?.summary || r.circle_map_arnold_tongues?.summary || r.circle_map_converged_density?.summary || r.mfdfa_plot_viewer?.summary || r.neuromouse_advanced_plots?.summary || r.band_power_summary?.summary || r.spike_detect?.summary || r.network_burst?.summary || r.electrode_connectivity?.summary || null;
}

function advRenderResult(payload) {
  latestAdvancedResult = payload;
  if ($('advancedRawJson')) $('advancedRawJson').textContent = JSON.stringify(payload, null, 2);
  const panel = $('advancedResultPanel');
  const status = $('advancedStatus');
  if (!panel) return;
  if (!payload?.ok) {
    if (status) { status.textContent = payload?.error || 'Advanced method failed.'; status.className = 'status failed'; }
    panel.textContent = payload?.error || 'Advanced method failed.';
    advRenderSummary(null);
    return;
  }
  if (status) { status.textContent = `Complete: ${payload.method?.name || payload.method_id}`; status.className = 'status ok'; }
  advRenderSummary(advExtractSummary(payload));
  const result = payload.result || {};
  const methodId = payload.method_id || payload.method?.id || '';
  if (methodId === 'higuchi_fractal_dimension') {
    const hfd = result.higuchi_fractal_dimension || {};
    const html = renderHiguchiFdPanel(hfd);
    panel.innerHTML = html;
    if ($('higuchiDirectResult')) $('higuchiDirectResult').innerHTML = html;
    setTimeout(() => renderHiguchiFdPlots(hfd), 30);
  } else if (methodId === 'embedded_fractal_dimension') {
    const efd = result.embedded_fractal_dimension || {};
    const html = renderEmbeddedFdPanel(efd);
    panel.innerHTML = html;
    if ($('embeddedFdResult')) $('embeddedFdResult').innerHTML = html;
    setTimeout(() => renderEmbeddedFdPlots(efd), 30);
  } else if (methodId === 'dimension_saturation_profiling') {
    const dsp = result.dimension_saturation_profiling || {};
    const html = renderSaturationPanel(dsp);
    panel.innerHTML = html;
    if ($('saturationResult')) $('saturationResult').innerHTML = html;
    setTimeout(() => renderSaturationPlots(dsp), 30);
  } else if (methodId === 'katz_fractal_dimension') {
    const kfd = result.katz_fractal_dimension || {};
    const html = renderKatzPanel(kfd);
    panel.innerHTML = html;
    if ($('katzResult')) $('katzResult').innerHTML = html;
    setTimeout(() => renderKatzPlots(kfd), 30);
  } else if (methodId === 'wavelet_hurst_exponent') {
    const wh = result.wavelet_hurst_exponent || {};
    const html = renderWaveletPanel(wh);
    panel.innerHTML = html;
    if ($('waveletResult')) $('waveletResult').innerHTML = html;
    setTimeout(() => renderWaveletPlots(wh), 30);
  } else if (methodId === 'manual_expert_mfdfa') {
    const mfdfa = result.manual_expert_mfdfa || {};
    const html = renderMfdfaPanel(mfdfa);
    panel.innerHTML = html;
    if ($('mfdfaResult')) $('mfdfaResult').innerHTML = html;
  } else if (methodId === 'manual_mfdfa_spectrum') {
    const spectrum = result.manual_mfdfa_spectrum || {};
    const html = renderMfdfaSpectrumPanel(spectrum);
    panel.innerHTML = html;
    if ($('mfdfaSpectrumResult')) $('mfdfaSpectrumResult').innerHTML = html;
  } else if (methodId === 'manual_mfdfa_shuffle_surrogate') {
    const shuffle = result.manual_mfdfa_shuffle_surrogate || {};
    const html = renderMfdfaShufflePanel(shuffle);
    panel.innerHTML = html;
    if ($('mfdfaShuffleResult')) $('mfdfaShuffleResult').innerHTML = html;
  } else if (methodId === 'manual_mfdfa_iaaft_surrogate') {
    const iaaft = result.manual_mfdfa_iaaft_surrogate || {};
    const html = renderMfdfaIaaftPanel(iaaft);
    panel.innerHTML = html;
    if ($('mfdfaIaaftResult')) $('mfdfaIaaftResult').innerHTML = html;
  } else if (methodId === 'wavelet_leader_multifractal') {
    const wl = result.wavelet_leader_multifractal || {};
    const html = renderWaveletLeaderPanel(wl);
    panel.innerHTML = html;
    if ($('waveletLeaderResult')) $('waveletLeaderResult').innerHTML = html;
  } else if (methodId === 'lyapunov_spectrum_custom') {
    const lyap = result.lyapunov_spectrum_custom || {};
    const html = renderLyapunovSpectrumPanel(lyap);
    panel.innerHTML = html;
    if ($('lyapunovSpectrumResult')) $('lyapunovSpectrumResult').innerHTML = html;
  } else if (methodId === 'arnold_tongues_kuramoto') {
    const arnold = result.arnold_tongues_kuramoto || {};
    const html = renderArnoldKuramotoPanel(arnold);
    panel.innerHTML = html;
    if ($('arnoldKuramotoResult')) $('arnoldKuramotoResult').innerHTML = html;
  } else if (methodId === 'circle_map_arnold_tongues') {
    const circle = result.circle_map_arnold_tongues || {};
    const html = renderCircleMapArnoldPanel(circle);
    panel.innerHTML = html;
    if ($('circleMapArnoldResult')) $('circleMapArnoldResult').innerHTML = html;
  } else if (methodId === 'circle_map_converged_density') {
    const density = result.circle_map_converged_density || {};
    const html = renderCircleMapDensityPanel(density);
    panel.innerHTML = html;
    if ($('circleMapDensityResult')) $('circleMapDensityResult').innerHTML = html;
  } else if (methodId === 'mfdfa_plot_viewer') {
    const viewer = result.mfdfa_plot_viewer || {};
    const html = renderMfdfaViewerPanel(viewer);
    panel.innerHTML = html;
    if ($('mfdfaViewerResult')) $('mfdfaViewerResult').innerHTML = html;
  } else if (methodId === 'neuromouse_advanced_plots') {
    const p = result.neuromouse_advanced_plots || {};
    panel.innerHTML = `<div class="method-note">Generated original NeuroMouse-compatible advanced plot data. Open the guaranteed server-rendered plot page below.</div>`
      + `<p><a class="button-like primary" href="${advEscape(p.plot_page_url || '/advanced-analysis')}" target="_blank">Open Polar / Kuramoto / Network / TDA Plots</a> <a class="button-like" href="${advEscape(p.open_in_neuromouse_url || '/neuromouse-latest/')}" target="_blank">Open Original NeuroMouse Workbench</a></p>`
      + advRenderTable([p.summary || {}, p.availability || {}]);
  } else if (payload.method_id === 'band_power_summary') {
    panel.innerHTML = `<div class="method-note">${advEscape(result.band_power_summary?.band || 'custom')} band ${advFormat(result.band_power_summary?.min_hz)}–${advFormat(result.band_power_summary?.max_hz)} Hz</div>` + advRenderTable(result.band_power_summary?.rows || []);
  } else if (payload.method_id === 'spike_detect') {
    panel.innerHTML = `<div class="method-note">Detected spikes/events per channel. Stored spike times are limited per channel to keep result files manageable.</div>` + advRenderTable(result.spike_detect?.rows || []);
  } else if (payload.method_id === 'network_burst') {
    panel.innerHTML = `<div class="method-note">Network burst timeline from detected multi-channel events.</div>` + advRenderTable(result.network_burst?.timeline || []);
  } else if (payload.method_id === 'electrode_connectivity') {
    panel.innerHTML = `<div class="method-note">Pairwise binned spike-train cross-correlation matrix. Strongest links are included below the matrix.</div>` + advRenderMatrix(result.electrode_connectivity?.matrix || [], result.electrode_connectivity?.channels || []) + '<h3>Strongest links</h3>' + advRenderTable(result.electrode_connectivity?.links || []);
  } else {
    panel.innerHTML = `<pre>${advEscape(JSON.stringify(result, null, 2))}</pre>`;
  }
}

function renderAdvancedMethodParams() {
  const select = $('advancedMethodSelect');
  const paramsBox = $('advancedParams');
  if (!select || !paramsBox) return;
  const spec = advancedMethods.find(m => m.id === select.value) || advancedMethods[0];
  if (!spec) return;
  if ($('advancedMethodDescription')) $('advancedMethodDescription').textContent = spec.description || '';
  paramsBox.innerHTML = (spec.parameters || []).map(param => {
    const id = `adv_param_${param.name}`;
    if (param.type === 'select') {
      return `<label>${advEscape(param.label || param.name)}<select data-advanced-param="${advEscape(param.name)}" id="${advEscape(id)}">${(param.options || []).map(opt => `<option value="${advEscape(opt)}" ${String(param.default) === String(opt) ? 'selected' : ''}>${advEscape(opt)}</option>`).join('')}</select></label>`;
    }
    return `<label>${advEscape(param.label || param.name)}<input data-advanced-param="${advEscape(param.name)}" id="${advEscape(id)}" type="${param.type === 'number' ? 'number' : 'text'}" step="any" value="${advEscape(param.default ?? '')}" /></label>`;
  }).join('');
}

async function loadAdvancedMethods() {
  const select = $('advancedMethodSelect');
  if (!select) return;
  try {
    const res = await fetch('/api/advanced-methods?t=' + Date.now(), { cache: 'no-store' });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Could not load advanced methods.');
    advancedMethods = data.methods || [];
    select.innerHTML = advancedMethods.map(m => `<option value="${advEscape(m.id)}">${advEscape(m.name || m.id)}</option>`).join('');
    const higuchi = advancedMethods.find(m => m.id === 'higuchi_fractal_dimension');
    if (higuchi) select.value = higuchi.id;
    if (data.latest_recording_dir && !$('advancedRecordingDir').value.trim()) $('advancedRecordingDir').value = data.latest_recording_dir;
    if (data.latest_recording_dir && $('higuchiDirectRecordingDir') && !$('higuchiDirectRecordingDir').value.trim()) $('higuchiDirectRecordingDir').value = data.latest_recording_dir;
    if (data.latest_recording_dir && $('embeddedFdRecordingDir') && !$('embeddedFdRecordingDir').value.trim()) $('embeddedFdRecordingDir').value = data.latest_recording_dir;
    renderAdvancedMethodParams();
  } catch (err) {
    if ($('advancedMethodDescription')) $('advancedMethodDescription').textContent = `Failed to load advanced methods: ${err.message}`;
  }
}

async function useLatestAdvancedRecording() {
  try {
    const res = await fetch('/api/advanced-methods/latest-recording?t=' + Date.now(), { cache: 'no-store' });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'No latest recording found.');
    $('advancedRecordingDir').value = data.recording_dir;
    if ($('advancedStatus')) { $('advancedStatus').textContent = `Using latest converted recording: ${data.recording_dir}`; $('advancedStatus').className = 'status ok'; }
  } catch (err) {
    alert(err.message);
  }
}

function collectAdvancedParams() {
  const params = {};
  document.querySelectorAll('[data-advanced-param]').forEach(input => {
    const name = input.getAttribute('data-advanced-param');
    const raw = input.value;
    if (raw === '') { params[name] = ''; return; }
    if (input.type === 'number') {
      const n = Number(raw);
      params[name] = Number.isFinite(n) ? n : raw;
    } else {
      params[name] = raw;
    }
  });
  return params;
}

async function runAdvancedMethod() {
  const methodId = $('advancedMethodSelect')?.value;
  const recordingDir = $('advancedRecordingDir')?.value.trim() || 'latest';
  if (!methodId) { alert('Choose an advanced method.'); return; }
  if ($('advancedStatus')) { $('advancedStatus').textContent = `Running ${methodId}...`; $('advancedStatus').className = 'status'; }
  if ($('advancedResultPanel')) $('advancedResultPanel').textContent = 'Running backend method...';
  try {
    const res = await fetch('/api/advanced-methods/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method_id: methodId, recording_dir: recordingDir, params: collectAdvancedParams() })
    });
    const data = await res.json();
    advRenderResult(data);
  } catch (err) {
    advRenderResult({ ok: false, error: err.message });
  }
}

function copyAdvancedResultJson() {
  if (!latestAdvancedResult) { alert('No advanced method result to copy yet.'); return; }
  navigator.clipboard.writeText(JSON.stringify(latestAdvancedResult, null, 2));
}

if ($('advancedMethodSelect')) $('advancedMethodSelect').addEventListener('change', renderAdvancedMethodParams);
if ($('advancedRefreshMethodsBtn')) $('advancedRefreshMethodsBtn').addEventListener('click', loadAdvancedMethods);
if ($('advancedUseLatestBtn')) $('advancedUseLatestBtn').addEventListener('click', useLatestAdvancedRecording);
if ($('advancedRunBtn')) $('advancedRunBtn').addEventListener('click', runAdvancedMethod);
if ($('advancedCopyJsonBtn')) $('advancedCopyJsonBtn').addEventListener('click', copyAdvancedResultJson);
loadAdvancedMethods();


// v0.11.9: open the original NeuroMouse advanced plotting workbench directly.
document.getElementById('openOriginalNeuroMouseAdvancedBtn')?.addEventListener('click', () => {
  window.open('/neuromouse/', '_blank');
});


// ---- v0.11.13 direct Higuchi FD workflow: visible card, one-click run, frontend plots ----
let latestHiguchiDirectResult = null;

function scrollToHiguchiDirect() {
  showTab('higuchiTab');
  const card = $('higuchiDirectCard');
  if (card) setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'start' }), 40);
}

function hfdDirectParams() {
  const params = { mode: $('higuchiDirectMode')?.value || 'ultra' };
  const fields = [
    ['segment_start_sec', 'higuchiDirectStart'],
    ['segment_end_sec', 'higuchiDirectEnd'],
    ['k_max', 'higuchiDirectKMax']
  ];
  fields.forEach(([name, id]) => {
    const raw = $(id)?.value?.trim();
    if (raw !== undefined && raw !== '') {
      const n = Number(raw);
      params[name] = Number.isFinite(n) ? n : raw;
    }
  });
  return params;
}

async function higuchiDirectUseLatest() {
  const status = $('higuchiDirectStatus');
  if (status) { status.textContent = 'Looking for the latest converted recording...'; status.className = 'status'; }
  try {
    const res = await fetch('/api/advanced-methods/latest-recording?t=' + Date.now(), { cache: 'no-store' });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'No latest converted recording found.');
    if ($('higuchiDirectRecordingDir')) $('higuchiDirectRecordingDir').value = data.recording_dir;
    if ($('advancedRecordingDir')) $('advancedRecordingDir').value = data.recording_dir;
    if (status) { status.textContent = `Using latest converted recording: ${data.recording_dir}`; status.className = 'status ok'; }
  } catch (err) {
    if (status) { status.textContent = err.message; status.className = 'status failed'; }
  }
}

async function runHiguchiDirect() {
  scrollToHiguchiDirect();
  const status = $('higuchiDirectStatus');
  const output = $('higuchiDirectResult');
  const recordingDir = $('higuchiDirectRecordingDir')?.value?.trim() || 'latest';
  if (status) { status.textContent = 'Running backend Higuchi fractal dimension analysis...'; status.className = 'status'; }
  if (output) output.innerHTML = '<div class="method-note">Running Higuchi FD in the backend. Results will render here as soon as the JSON returns.</div>';
  try {
    const res = await fetch('/api/advanced-methods/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ method_id: 'higuchi_fractal_dimension', recording_dir: recordingDir, params: hfdDirectParams() })
    });
    const payload = await res.json();
    latestHiguchiDirectResult = payload;
    if (!payload.ok) throw new Error(payload.error || 'Higuchi FD method failed.');
    const hfd = payload.result?.higuchi_fractal_dimension || {};
    if (status) { status.textContent = `Complete: Higuchi FD ${hfd.summary?.mode || ''} mode · ${hfd.summary?.n_channels || ''} channels`; status.className = 'status ok'; }
    if (output) {
      output.innerHTML = renderHiguchiFdPanel(hfd);
      setTimeout(() => renderHiguchiFdPlots(hfd), 30);
    }
    if ($('advancedResultPanel')) {
      $('advancedResultPanel').innerHTML = '<div class="method-note">Higuchi FD also rendered in the dedicated card above. Run other advanced methods here if needed.</div>';
    }
  } catch (err) {
    if (status) { status.textContent = err.message; status.className = 'status failed'; }
    if (output) output.innerHTML = `<div class="method-note">${advEscape(err.message)}</div>`;
  }
}

function copyHiguchiDirectJson() {
  if (!latestHiguchiDirectResult) { alert('No Higuchi result JSON to copy yet.'); return; }
  navigator.clipboard.writeText(JSON.stringify(latestHiguchiDirectResult, null, 2));
}

$('scrollHiguchiDirectBtn')?.addEventListener('click', scrollToHiguchiDirect);
$('scrollImportForHiguchiBtn')?.addEventListener('click', () => showTab('importTab'));
$('higuchiDirectUseLatestBtn')?.addEventListener('click', higuchiDirectUseLatest);
$('higuchiDirectRunBtn')?.addEventListener('click', runHiguchiDirect);
$('higuchiDirectCopyJsonBtn')?.addEventListener('click', copyHiguchiDirectJson);



// ---- v0.11.17 Embedded attractor fractal-dimension frontend plots ----
let latestEmbeddedFdResult = null;

function efdRows(data) {
  return Array.isArray(data?.rows) ? data.rows.filter(r => r && r.status === 'ok') : [];
}

function efdGroupByDim(rows, field) {
  const by = new Map();
  rows.forEach(r => {
    const m = Number(r.emb_dim);
    const v = Number(r[field]);
    if (!Number.isFinite(m) || !Number.isFinite(v)) return;
    if (!by.has(m)) by.set(m, []);
    by.get(m).push(v);
  });
  return Array.from(by.entries()).sort((a, b) => a[0] - b[0]).map(([emb_dim, vals]) => ({ emb_dim, mean: vals.reduce((a, b) => a + b, 0) / vals.length, n: vals.length }));
}

function efdDrawMeanCurve(data, canvasId, field, title, ylabel) {
  const rows = efdRows(data);
  const grouped = efdGroupByDim(rows, field);
  const plot = plotCanvas(canvasId, title, grouped.length ? '' : 'Run Embedded Fractal Dimension to populate this plot.');
  if (!plot || !grouped.length) return;
  const { ctx, w, h } = plot;
  const left = 66, top = 48, width = w - 96, height = h - 92;
  const xs = grouped.map(d => d.emb_dim);
  const ys = grouped.map(d => d.mean);
  const xr = hfdValueRange(xs, 0.08);
  const yr = hfdValueRange(ys, 0.14);
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding dimension m', ylabel, xr, yr);
  ctx.beginPath();
  grouped.forEach((d, i) => {
    const x = hfdMapPoint(d.emb_dim, xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(d.mean, yr.min, yr.max, top + height, top);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#00e5e5'; ctx.lineWidth = 2.4; ctx.stroke();
  grouped.forEach(d => {
    const x = hfdMapPoint(d.emb_dim, xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(d.mean, yr.min, yr.max, top + height, top);
    ctx.fillStyle = '#00ff99'; ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign = 'center'; ctx.fillText(String(d.emb_dim), x, top + height + 16);
    ctx.fillText(d.mean.toFixed(2), x, y - 8);
  });
  ctx.textAlign = 'left';
}

function efdDrawMeanDimensions(data) {
  const mean = Array.isArray(data?.mean_by_dimension) ? data.mean_by_dimension : [];
  const plot = plotCanvas('efdMeanDimensionsCanvas', 'Mean Embedded Fractal Dimensions Across Channels', mean.length ? '' : 'No mean-by-dimension rows returned.');
  if (!plot || !mean.length) return;
  const { ctx, w, h } = plot;
  const left = 66, top = 48, width = w - 96, height = h - 92;
  const xs = mean.map(d => Number(d.emb_dim)).filter(Number.isFinite);
  const ys = mean.flatMap(d => [Number(d.corr_dim_d2), Number(d.boxcount_fd)]).filter(Number.isFinite);
  const xr = hfdValueRange(xs, 0.08);
  const yr = hfdValueRange(ys.concat(xs), 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding dimension m', 'Estimated dimension', xr, yr);
  const drawSeries = (field, color, dash=false) => {
    ctx.beginPath();
    mean.forEach((d, i) => {
      const x = hfdMapPoint(d.emb_dim, xr.min, xr.max, left, left + width);
      const y = hfdMapPoint(Number(d[field]), yr.min, yr.max, top + height, top);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color; ctx.lineWidth = 2.2; if (dash) ctx.setLineDash([5, 4]); ctx.stroke(); ctx.setLineDash([]);
  };
  drawSeries('corr_dim_d2', '#00e5e5');
  drawSeries('boxcount_fd', '#dfffff', true);
  ctx.strokeStyle = 'rgba(0,255,153,0.45)'; ctx.setLineDash([2, 4]);
  ctx.beginPath();
  xs.forEach((m, i) => {
    const x = hfdMapPoint(m, xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(m, yr.min, yr.max, top + height, top);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle = '#dfffff'; ctx.font = '12px monospace';
  ctx.fillText('D2 mean', left + 12, top + 18);
  ctx.fillText('Box FD mean (dashed)', left + 12, top + 36);
  ctx.fillText('ambient m reference (dotted)', left + 12, top + 54);
}

function efdDrawHeatmap(data, canvasId, field, title) {
  const rows = efdRows(data).filter(r => Number.isFinite(Number(r[field])));
  const dims = Array.from(new Set(rows.map(r => Number(r.emb_dim)))).sort((a, b) => a - b);
  const channels = Array.from(new Set(rows.map(r => String(r.channel))));
  const plot = plotCanvas(canvasId, title, rows.length ? '' : 'No rows available for heatmap.');
  if (!plot || !rows.length || !dims.length || !channels.length) return;
  const { ctx, w, h } = plot;
  const left = 72, top = 48, width = w - 100, height = h - 92;
  const vals = rows.map(r => Number(r[field]));
  const range = hfdValueRange(vals, 0.04);
  const cellW = width / dims.length;
  const cellH = height / channels.length;
  const lookup = new Map(rows.map(r => [`${r.channel}|${r.emb_dim}`, Number(r[field]) ]));
  channels.forEach((ch, i) => {
    dims.forEach((m, j) => {
      const v = lookup.get(`${ch}|${m}`);
      if (!Number.isFinite(v)) return;
      ctx.fillStyle = interpolateColor(v, range.min, range.max);
      ctx.fillRect(left + j * cellW, top + i * cellH, Math.ceil(cellW), Math.ceil(cellH));
    });
  });
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding dimension m', 'Channel');
  ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  dims.forEach((m, j) => ctx.fillText(String(m), left + j * cellW + cellW / 2, top + height + 16));
  ctx.textAlign = 'right';
  channels.slice(0, 32).forEach((ch, i) => ctx.fillText(ch.slice(0, 7), left - 6, top + i * cellH + cellH / 2 + 3));
  ctx.textAlign = 'left'; ctx.fillText(`${range.min.toFixed(2)} → ${range.max.toFixed(2)}`, left, h - 14);
}

function efdDrawRanking(data) {
  const rows = Array.isArray(data?.channel_summary) ? data.channel_summary.filter(r => Number.isFinite(Number(r.dimension_complexity_score))) : [];
  const plot = plotCanvas('efdRankingCanvas', 'Channel Ranking by Embedded Fractal-Dimension Complexity', rows.length ? '' : 'No channel summary available.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const ranked = rows.slice().sort((a, b) => Number(b.dimension_complexity_score) - Number(a.dimension_complexity_score)).slice(0, 32).reverse();
  const left = 108, top = 48, width = w - 142, height = h - 74;
  const vals = ranked.map(r => Number(r.dimension_complexity_score));
  const range = hfdValueRange(vals.concat([0]), 0.10);
  hfdDrawAxes(ctx, left, top, width, height, 'Complexity score', 'Channel', range, {min: 1, max: ranked.length}, {yTickCount: Math.min(5, ranked.length)});
  const rowH = height / ranked.length;
  ranked.forEach((row, i) => {
    const y = top + i * rowH + rowH * 0.2;
    const x = hfdMapPoint(row.dimension_complexity_score, range.min, range.max, left, left + width);
    ctx.fillStyle = '#00e5e5'; ctx.globalAlpha = 0.78; ctx.fillRect(left, y, Math.max(2, x - left), Math.max(2, rowH * 0.58)); ctx.globalAlpha = 1;
    ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign = 'right'; ctx.fillText(String(row.channel), left - 8, y + rowH * 0.45);
    ctx.textAlign = 'left'; ctx.fillText(Number(row.dimension_complexity_score).toFixed(2), x + 4, y + rowH * 0.45);
  });
}

function efdDrawD2VsBox(data) {
  const rows = efdRows(data).filter(r => Number.isFinite(Number(r.corr_dim_d2)) && Number.isFinite(Number(r.boxcount_fd)));
  const plot = plotCanvas('efdD2VsBoxCanvas', 'Correlation Dimension vs Box-Counting Dimension', rows.length ? '' : 'No D2/box rows available.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 64, top = 48, width = w - 94, height = h - 88;
  const xr = hfdValueRange(rows.map(r => Number(r.boxcount_fd)), 0.12);
  const yr = hfdValueRange(rows.map(r => Number(r.corr_dim_d2)), 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'Box-counting FD', 'Correlation dimension D2', xr, yr);
  rows.forEach(row => {
    const x = hfdMapPoint(row.boxcount_fd, xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(row.corr_dim_d2, yr.min, yr.max, top + height, top);
    ctx.fillStyle = interpolateColor(Number(row.emb_dim), 2, 10);
    ctx.strokeStyle = '#dfffff'; ctx.lineWidth = 0.6;
    ctx.beginPath(); ctx.arc(x, y, 4 + Number(row.emb_dim) * 0.35, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  });
}

function efdDrawFitQc(data) {
  const rows = efdRows(data);
  const plot = plotCanvas('efdFitQcCanvas', 'Fractal-Dimension Fit Quality Control', rows.length ? '' : 'No rows available.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 64, top = 48, width = w - 94, height = h - 88;
  const fitPoints = rows.flatMap(r => [Number(r.corr_fit_points), Number(r.box_fit_points)]).filter(Number.isFinite);
  const xr = hfdValueRange(fitPoints, 0.14);
  hfdDrawAxes(ctx, left, top, width, height, 'Fit scale points', 'R²', xr, {min: 0, max: 1});
  rows.forEach(row => {
    const x1 = hfdMapPoint(row.corr_fit_points, xr.min, xr.max, left, left + width);
    const y1 = hfdMapPoint(row.corr_fit_r2, 0, 1, top + height, top);
    if (Number.isFinite(x1) && Number.isFinite(y1)) { ctx.fillStyle = '#00e5e5'; ctx.beginPath(); ctx.arc(x1, y1, 4, 0, Math.PI * 2); ctx.fill(); }
    const x2 = hfdMapPoint(row.box_fit_points, xr.min, xr.max, left, left + width);
    const y2 = hfdMapPoint(row.box_fit_r2, 0, 1, top + height, top);
    if (Number.isFinite(x2) && Number.isFinite(y2)) { ctx.strokeStyle = '#dfffff'; ctx.beginPath(); ctx.arc(x2, y2, 5, 0, Math.PI * 2); ctx.stroke(); }
  });
  ctx.strokeStyle = 'rgba(0,255,153,0.55)'; const y095 = hfdMapPoint(0.95, 0, 1, top + height, top); ctx.beginPath(); ctx.moveTo(left, y095); ctx.lineTo(left + width, y095); ctx.stroke();
  ctx.fillStyle = '#dfffff'; ctx.font = '11px monospace'; ctx.fillText('filled=D2 fit, open=box fit, line=0.95 R²', left + 12, top + 18);
}

function efdDrawDiagnostics(data) {
  const diags = Array.isArray(data?.diagnostics) ? data.diagnostics : [];
  const plot = plotCanvas('efdDiagnosticsCanvas', 'Diagnostic Correlation-Dimension Fits', diags.length ? '' : 'No diagnostic curves returned.');
  if (!plot || !diags.length) return;
  const { ctx, w, h } = plot;
  const cols = 3, rows = Math.ceil(diags.length / cols), gap = 16;
  const cellW = (w - 40 - gap * (cols - 1)) / cols;
  const cellH = (h - 56 - gap * (rows - 1)) / rows;
  diags.forEach((d, idx) => {
    const col = idx % cols, row = Math.floor(idx / cols);
    const left = 20 + col * (cellW + gap), top = 42 + row * (cellH + gap);
    const lx = finiteNumbers(d.corr_log_r), ly = finiteNumbers(d.corr_log_C), fit = finiteNumbers(d.corr_fit_line);
    if (lx.length < 2 || ly.length < 2) return;
    const xr = hfdValueRange(lx, 0.08), yr = hfdValueRange(ly.concat(fit), 0.12);
    ctx.strokeStyle = 'rgba(127,255,255,0.18)'; ctx.strokeRect(left, top, cellW, cellH);
    ctx.beginPath(); lx.forEach((xv, i) => { const x = hfdMapPoint(xv, xr.min, xr.max, left + 10, left + cellW - 8); const y = hfdMapPoint(ly[i], yr.min, yr.max, top + cellH - 14, top + 12); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.strokeStyle = '#00e5e5'; ctx.lineWidth = 1.2; ctx.stroke();
    if (fit.length === lx.length) { ctx.beginPath(); lx.forEach((xv, i) => { const x = hfdMapPoint(xv, xr.min, xr.max, left + 10, left + cellW - 8); const y = hfdMapPoint(fit[i], yr.min, yr.max, top + cellH - 14, top + 12); if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.strokeStyle = '#00ff99'; ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.fillText(`${d.channel} m=${d.emb_dim} D2=${Number(d.corr_dim_d2).toFixed(2)}`, left + 4, top + 12);
  });
}

function renderEmbeddedFdPlots(data) {
  if (!data) return;
  efdDrawMeanCurve(data, 'efdD2CurveCanvas', 'corr_dim_d2', 'Correlation Dimension D2 vs Embedding Dimension', 'Mean D2');
  efdDrawMeanCurve(data, 'efdBoxCurveCanvas', 'boxcount_fd', 'Box-Counting Fractal Dimension vs Embedding Dimension', 'Mean box FD');
  efdDrawMeanDimensions(data);
  efdDrawHeatmap(data, 'efdD2HeatmapCanvas', 'corr_dim_d2', 'Correlation Dimension D2 Heatmap');
  efdDrawHeatmap(data, 'efdBoxHeatmapCanvas', 'boxcount_fd', 'Box-Counting FD Heatmap');
  efdDrawRanking(data);
  efdDrawD2VsBox(data);
  efdDrawFitQc(data);
  efdDrawDiagnostics(data);
}

function renderEmbeddedFdPanel(data) {
  const summary = data?.summary || {};
  const outputs = data?.outputs || {};
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Embedded fractal-dimension analysis completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It used ${advEscape(summary.generated_embeddings ?? 0)} generated delay embeddings and ${advEscape(summary.precomputed_embeddings ?? 0)} precomputed embeddings.</div>
    <div class="grid two embedded-plot-grid">
      <div class="card advanced-plot-card"><h3>D2 vs Embedding Dimension</h3><canvas id="efdD2CurveCanvas" class="advanced-plot-canvas" width="820" height="440"></canvas></div>
      <div class="card advanced-plot-card"><h3>Box FD vs Embedding Dimension</h3><canvas id="efdBoxCurveCanvas" class="advanced-plot-canvas" width="820" height="440"></canvas></div>
      <div class="card advanced-plot-card"><h3>Mean Dimensions</h3><canvas id="efdMeanDimensionsCanvas" class="advanced-plot-canvas" width="820" height="440"></canvas></div>
      <div class="card advanced-plot-card"><h3>Channel Complexity Ranking</h3><canvas id="efdRankingCanvas" class="advanced-plot-canvas" width="820" height="520"></canvas></div>
      <div class="card advanced-plot-card"><h3>D2 Heatmap</h3><canvas id="efdD2HeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Box FD Heatmap</h3><canvas id="efdBoxHeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>D2 vs Box FD</h3><canvas id="efdD2VsBoxCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
      <div class="card advanced-plot-card"><h3>Fit Quality Control</h3><canvas id="efdFitQcCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
    </div>
    <div class="card advanced-plot-card"><h3>Diagnostic Correlation-Dimension Fits</h3><canvas id="efdDiagnosticsCanvas" class="advanced-plot-canvas" width="1100" height="720"></canvas></div>
    <h3>Mean by embedding dimension</h3>${advRenderTable(data?.mean_by_dimension || [])}
    <h3>Top channel summary</h3>${advRenderTable((data?.channel_summary || []).slice(0, 32))}
    <h3>All computed rows</h3>${advRenderTable(data?.rows || [])}
    ${outputHtml}`;
}

function embeddedFdParams() {
  const params = { mode: $('embeddedFdMode')?.value || 'ultra' };
  const dims = $('embeddedFdDims')?.value?.trim();
  if (dims) params.embedding_dims = dims;
  [['tau_ms','embeddedFdTauMs'], ['max_channels','embeddedFdMaxChannels']].forEach(([name, id]) => {
    const raw = $(id)?.value?.trim();
    if (raw !== undefined && raw !== '') { const n = Number(raw); params[name] = Number.isFinite(n) ? n : raw; }
  });
  return params;
}

async function embeddedFdUseLatest() {
  const status = $('embeddedFdStatus');
  if (status) { status.textContent = 'Looking for the latest converted recording...'; status.className = 'status'; }
  try {
    const res = await fetch('/api/advanced-methods/latest-recording?t=' + Date.now(), { cache: 'no-store' });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'No latest converted recording found.');
    if ($('embeddedFdRecordingDir')) $('embeddedFdRecordingDir').value = data.recording_dir;
    if ($('advancedRecordingDir')) $('advancedRecordingDir').value = data.recording_dir;
    if (status) { status.textContent = `Using latest converted recording: ${data.recording_dir}`; status.className = 'status ok'; }
  } catch (err) { if (status) { status.textContent = err.message; status.className = 'status failed'; } }
}

async function runEmbeddedFdDirect() {
  showTab('embeddedFdTab');
  const card = $('embeddedFdDirectCard');
  if (card) setTimeout(() => card.scrollIntoView({ behavior: 'smooth', block: 'start' }), 30);
  const status = $('embeddedFdStatus'), output = $('embeddedFdResult');
  const recordingDir = $('embeddedFdRecordingDir')?.value?.trim() || 'latest';
  if (status) { status.textContent = 'Running embedded fractal-dimension backend analysis...'; status.className = 'status'; }
  if (output) output.innerHTML = '<div class="method-note">Running embedded FD in the backend. Results will render here when the JSON returns.</div>';
  try {
    const res = await fetch('/api/advanced-methods/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ method_id: 'embedded_fractal_dimension', recording_dir: recordingDir, params: embeddedFdParams() }) });
    const payload = await res.json();
    latestEmbeddedFdResult = payload;
    if (!payload.ok) throw new Error(payload.error || 'Embedded FD method failed.');
    const efd = payload.result?.embedded_fractal_dimension || {};
    if (status) { status.textContent = `Complete: embedded FD ${efd.summary?.mode || ''} mode · ${efd.summary?.ok_rows || 0} rows`; status.className = 'status ok'; }
    if (output) { output.innerHTML = renderEmbeddedFdPanel(efd); setTimeout(() => renderEmbeddedFdPlots(efd), 30); }
    if ($('advancedMethodSelect')) $('advancedMethodSelect').value = 'embedded_fractal_dimension';
  } catch (err) { if (status) { status.textContent = err.message; status.className = 'status failed'; } if (output) output.innerHTML = `<div class="method-note">${advEscape(err.message)}</div>`; }
}

function copyEmbeddedFdJson() {
  if (!latestEmbeddedFdResult) { alert('No Embedded FD result JSON to copy yet.'); return; }
  navigator.clipboard.writeText(JSON.stringify(latestEmbeddedFdResult, null, 2));
}


// ---- v0.11.17 Dimension saturation profiling frontend plots ----
let latestSaturationResult = null;
function dspRows(data) { return Array.isArray(data?.summary_rows) ? data.summary_rows : []; }
function dspProfileRows(data) { return Array.isArray(data?.profiles) ? data.profiles : []; }
function dspDeltaRows(data) { return Array.isArray(data?.delta_rows) ? data.delta_rows : []; }
function dspClassCounts(data) { return Array.isArray(data?.class_counts) ? data.class_counts : []; }
function dspUniqueSorted(rows, key) { return Array.from(new Set(rows.map(r => r?.[key]).filter(v => v !== null && v !== undefined))).sort((a,b) => Number(a) - Number(b)); }

function renderSaturationPlots(data) {
  if (!data) return;
  dspDrawProfiles(data);
  dspDrawDeltaHeatmap(data);
  dspDrawSaturationBar(data);
  dspDrawTailSlope(data);
  dspDrawClassCounts(data);
}

function dspDrawProfiles(data) {
  const rows = dspProfileRows(data);
  const plot = plotCanvas('dspProfilesCanvas', 'Dimension-saturation profiling from D2(m)', rows.length ? '' : 'No D2(m) rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 62, top = 48, width = w - 96, height = h - 92;
  const dims = dspUniqueSorted(rows, 'emb_dim');
  const vals = finiteNumbers(rows.map(r => r.corr_dim_d2 ?? r.boxcount_fd));
  const xr = hfdValueRange(dims, 0.08);
  const yr = hfdValueRange(vals, 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding dimension m', 'D2', xr, yr);
  const byCh = new Map();
  rows.forEach(r => { const ch = String(r.channel); if (!byCh.has(ch)) byCh.set(ch, []); byCh.get(ch).push(r); });
  const satMap = new Map(dspRows(data).map(r => [String(r.channel), r]));
  byCh.forEach((sub, ch) => {
    sub = sub.slice().sort((a,b) => Number(a.emb_dim) - Number(b.emb_dim));
    ctx.beginPath();
    sub.forEach((r, i) => {
      const value = Number(r.corr_dim_d2 ?? r.boxcount_fd);
      const x = hfdMapPoint(r.emb_dim, xr.min, xr.max, left, left + width);
      const y = hfdMapPoint(value, yr.min, yr.max, top + height, top);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'rgba(0,229,229,0.22)'; ctx.lineWidth = 1.25; ctx.stroke();
    const sat = satMap.get(ch); const ms = Number(sat?.m_sat_abs);
    if (Number.isFinite(ms)) {
      const r = sub.find(p => Number(p.emb_dim) === ms);
      if (r) {
        const value = Number(r.corr_dim_d2 ?? r.boxcount_fd);
        const x = hfdMapPoint(ms, xr.min, xr.max, left, left + width);
        const y = hfdMapPoint(value, yr.min, yr.max, top + height, top);
        ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.stroke();
      }
    }
  });
  ctx.fillStyle = '#dfffff'; ctx.font = '11px monospace'; ctx.fillText('Open circles mark absolute-threshold saturation estimates', left + 8, top + 18);
}

function dspDrawDeltaHeatmap(data) {
  const rows = dspDeltaRows(data);
  const plot = plotCanvas('dspDeltaHeatmapCanvas', 'ΔD2 by channel and embedding step', rows.length ? '' : 'No delta rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const channels = Array.from(new Set(rows.map(r => String(r.channel))));
  const dims = dspUniqueSorted(rows, 'emb_dim_to');
  const matrix = channels.map(ch => dims.map(dim => {
    const row = rows.find(r => String(r.channel) === ch && Number(r.emb_dim_to) === Number(dim));
    return Number(row?.delta_D2);
  }));
  const left = 92, top = 48, width = w - 126, height = h - 86;
  const vals = finiteNumbers(matrix.flat()); const range = hfdValueRange(vals.concat([0]), 0.20);
  ctx.strokeStyle = 'rgba(127,255,255,0.18)'; ctx.strokeRect(left, top, width, height);
  const cellW = width / Math.max(1, dims.length), cellH = height / Math.max(1, channels.length);
  channels.forEach((ch, i) => {
    dims.forEach((dim, j) => {
      const v = matrix[i][j]; if (!Number.isFinite(v)) return;
      ctx.fillStyle = interpolateColor(v, range.min, range.max);
      ctx.globalAlpha = 0.20 + 0.75 * Math.min(1, Math.abs(v) / Math.max(1e-9, Math.max(Math.abs(range.min), Math.abs(range.max))));
      ctx.fillRect(left + j * cellW, top + i * cellH, Math.ceil(cellW), Math.ceil(cellH));
      ctx.globalAlpha = 1;
    });
  });
  ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace';
  channels.forEach((ch, i) => { if (i % Math.ceil(channels.length / 32) === 0 || channels.length <= 34) { ctx.textAlign = 'right'; ctx.fillText(ch, left - 6, top + i * cellH + cellH * 0.65); } });
  dims.forEach((dim, j) => { ctx.textAlign = 'center'; ctx.fillText(String(dim), left + j * cellW + cellW / 2, top + height + 16); });
  ctx.textAlign = 'left'; ctx.fillText('cyan intensity = ΔD2 magnitude/sign-scaled', left, 24);
}

function dspDrawSaturationBar(data) {
  const rows = dspRows(data).slice();
  const plot = plotCanvas('dspSaturationBarCanvas', 'Absolute-threshold saturation dimension by channel', rows.length ? '' : 'No saturation summary rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const maxM = Number(data?.summary?.max_embedding_dim) || Math.max(...rows.map(r => Number(r.m_max)).filter(Number.isFinite), 10);
  rows.sort((a,b) => (Number(a.m_sat_abs || maxM+0.5) - Number(b.m_sat_abs || maxM+0.5)) || String(a.channel).localeCompare(String(b.channel)));
  const left = 58, top = 48, width = w - 88, height = h - 96;
  const yr = {min: 0, max: maxM + 1};
  hfdDrawAxes(ctx, left, top, width, height, 'Channel', 'm_sat_abs', {min: 1, max: rows.length}, yr, {xTickCount: Math.min(5, rows.length)});
  const barW = width / rows.length;
  rows.forEach((r, i) => {
    const val = Number.isFinite(Number(r.m_sat_abs)) ? Number(r.m_sat_abs) : maxM + 0.5;
    const x = left + i * barW + 1;
    const y = hfdMapPoint(val, 0, maxM + 1, top + height, top);
    ctx.fillStyle = Number.isFinite(Number(r.m_sat_abs)) ? '#00e5e5' : 'rgba(127,255,255,0.28)';
    ctx.fillRect(x, y, Math.max(2, barW - 2), top + height - y);
    if (rows.length <= 40) { ctx.save(); ctx.translate(x + barW * .5, top + height + 12); ctx.rotate(-Math.PI/2); ctx.fillStyle = '#dfffff'; ctx.font='9px monospace'; ctx.textAlign='right'; ctx.fillText(String(r.channel),0,0); ctx.restore(); }
  });
  const yMax = hfdMapPoint(maxM, 0, maxM + 1, top + height, top); ctx.strokeStyle='#00ffff'; ctx.setLineDash([5,4]); ctx.beginPath(); ctx.moveTo(left,yMax); ctx.lineTo(left+width,yMax); ctx.stroke(); ctx.setLineDash([]);
}

function dspDrawTailSlope(data) {
  const rows = dspRows(data).slice().sort((a,b) => Number(b.tail_slope_last_points) - Number(a.tail_slope_last_points));
  const plot = plotCanvas('dspTailSlopeCanvas', 'Tail slope of D2(m)', rows.length ? '' : 'No tail-slope rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 62, top = 48, width = w - 92, height = h - 96;
  const vals = rows.map(r => Number(r.tail_slope_last_points)).filter(Number.isFinite); const range = hfdValueRange(vals.concat([0]), 0.2);
  const yr = range;
  hfdDrawAxes(ctx, left, top, width, height, 'Channel', 'Tail slope', {min: 1, max: rows.length}, yr, {xTickCount: Math.min(5, rows.length)});
  const zeroY = hfdMapPoint(0, range.min, range.max, top + height, top); ctx.strokeStyle='rgba(255,255,255,0.5)'; ctx.beginPath(); ctx.moveTo(left, zeroY); ctx.lineTo(left+width, zeroY); ctx.stroke();
  const barW = width / rows.length;
  rows.forEach((r, i) => {
    const v = Number(r.tail_slope_last_points); if (!Number.isFinite(v)) return;
    const x = left + i * barW + 1; const y = hfdMapPoint(v, range.min, range.max, top + height, top);
    ctx.fillStyle = v >= 0 ? '#00ffff' : 'rgba(0,255,153,0.55)';
    ctx.fillRect(x, Math.min(y, zeroY), Math.max(2, barW-2), Math.max(2, Math.abs(zeroY-y)));
  });
}

function dspDrawClassCounts(data) {
  const rows = dspClassCounts(data);
  const plot = plotCanvas('dspClassCountsCanvas', 'Saturation class counts', rows.length ? '' : 'No class counts returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 58, top = 48, width = w - 88, height = h - 92;
  const maxV = Math.max(...rows.map(r => Number(r.count)).filter(Number.isFinite), 1);
  const yr = {min: 0, max: maxV};
  hfdDrawAxes(ctx, left, top, width, height, 'Class', 'Count', {min: 1, max: rows.length}, yr, {xTickCount: Math.min(5, rows.length)});
  const barW = width / rows.length;
  rows.forEach((r, i) => {
    const v = Number(r.count); const x = left + i * barW + barW*.18; const y = hfdMapPoint(v, 0, maxV, top + height, top);
    ctx.fillStyle = '#00e5e5'; ctx.fillRect(x, y, barW*.64, top + height - y);
    ctx.fillStyle = '#dfffff'; ctx.font = '12px monospace'; ctx.textAlign='center'; ctx.fillText(String(r.class_abs), x + barW*.32, top+height+18); ctx.fillText(String(v), x + barW*.32, y-5);
  });
}

function renderSaturationPanel(data) {
  const summary = data?.summary || {};
  const outputs = data?.outputs || {};
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Dimension saturation profiling completed using <strong>${advEscape(summary.use_metric || 'corr_dim_d2')}</strong>. Source: ${advEscape(summary.source_csv || '')}. Embedded FD auto-run: <strong>${summary.embedded_fd_was_run ? 'yes' : 'no'}</strong>.</div>
    <div class="grid two saturation-plot-grid">
      <div class="card advanced-plot-card"><h3>D2(m) saturation profiles</h3><canvas id="dspProfilesCanvas" class="advanced-plot-canvas" width="820" height="500"></canvas></div>
      <div class="card advanced-plot-card"><h3>ΔD2 heatmap</h3><canvas id="dspDeltaHeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Saturation dimension by channel</h3><canvas id="dspSaturationBarCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
      <div class="card advanced-plot-card"><h3>Tail slope barplot</h3><canvas id="dspTailSlopeCanvas" class="advanced-plot-canvas" width="820" height="460"></canvas></div>
      <div class="card advanced-plot-card"><h3>Saturation class counts</h3><canvas id="dspClassCountsCanvas" class="advanced-plot-canvas" width="820" height="420"></canvas></div>
    </div>
    <h3>Saturation summary</h3>${advRenderTable(data?.summary_rows || [])}
    <h3>ΔD2 rows</h3>${advRenderTable(data?.delta_rows || [])}
    ${outputHtml}`;
}

function saturationParams() {
  const params = { mode: $('saturationMode')?.value || 'ultra', use_metric: $('saturationMetric')?.value || 'corr_dim_d2' };
  const fields = [['abs_delta_thresh','saturationAbsThresh'], ['rel_delta_thresh','saturationRelThresh'], ['n_consecutive','saturationConsecutive'], ['tail_points','saturationTailPoints'], ['embedding_dims','saturationDims'], ['tau_ms','saturationTauMs'], ['max_channels','saturationMaxChannels'], ['max_analysis_samples','saturationMaxSamples']];
  fields.forEach(([name, id]) => { const raw = $(id)?.value?.trim(); if (raw !== undefined && raw !== '') { const n = Number(raw); params[name] = Number.isFinite(n) && name !== 'embedding_dims' ? n : raw; } });
  params.run_embedded_if_missing = 'true';
  return params;
}
async function saturationUseLatest() {
  const status = $('saturationStatus'); if (status) { status.textContent = 'Looking for latest converted recording...'; status.className = 'status'; }
  try { const res = await fetch('/api/advanced-methods/latest-recording?t=' + Date.now(), {cache:'no-store'}); const data = await res.json(); if (!data.ok) throw new Error(data.error || 'No latest recording found.'); if ($('saturationRecordingDir')) $('saturationRecordingDir').value = data.recording_dir; if (status) { status.textContent = `Using latest converted recording: ${data.recording_dir}`; status.className = 'status ok'; } } catch (err) { if (status) { status.textContent = err.message; status.className = 'status failed'; } }
}
async function runSaturationDirect() {
  showTab('saturationTab'); const card = $('saturationDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 30);
  const status = $('saturationStatus'), output = $('saturationResult'); const recordingDir = $('saturationRecordingDir')?.value?.trim() || 'latest';
  if (status) { status.textContent = 'Running dimension saturation profiling...'; status.className = 'status'; }
  if (output) output.innerHTML = '<div class="method-note">Running saturation profiling. If Embedded FD is missing, the backend will run it first.</div>';
  try { const res = await fetch('/api/advanced-methods/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ method_id:'dimension_saturation_profiling', recording_dir: recordingDir, params: saturationParams() }) }); const payload = await res.json(); latestSaturationResult = payload; if (!payload.ok) throw new Error(payload.error || 'Saturation profiling failed.'); const dsp = payload.result?.dimension_saturation_profiling || {}; if (status) { status.textContent = `Complete: ${dsp.summary?.n_channels || 0} channels · ${dsp.summary?.n_delta_rows || 0} Δ rows`; status.className = 'status ok'; } if (output) { output.innerHTML = renderSaturationPanel(dsp); setTimeout(() => renderSaturationPlots(dsp), 30); } if ($('advancedMethodSelect')) $('advancedMethodSelect').value = 'dimension_saturation_profiling'; } catch (err) { if (status) { status.textContent = err.message; status.className = 'status failed'; } if (output) output.innerHTML = `<div class="method-note">${advEscape(err.message)}</div>`; }
}
function copySaturationJson() { if (!latestSaturationResult) { alert('No saturation profiling result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestSaturationResult, null, 2)); }


// ---- v0.11.18 Katz fractal dimension frontend plots ----
let latestKatzResult = null;
function kfdRows(data) { return Array.isArray(data?.rows) ? data.rows.filter(r => Number.isFinite(Number(r?.katz_fd))) : []; }
function kfdByDimension(data) { return Array.isArray(data?.by_dimension) ? data.by_dimension.filter(r => Number.isFinite(Number(r?.mean_katz_fd))) : []; }
function kfdMatrixRows(data) { return Array.isArray(data?.channel_dimension_matrix) ? data.channel_dimension_matrix : []; }

function renderKatzPlots(data) {
  if (!data) return;
  kfdDrawAllEmbeddings(data);
  kfdDrawMeanByDimension(data);
  kfdDrawHeatmap(data);
  kfdDrawTrajectories(data);
}

function kfdDrawAllEmbeddings(data) {
  const rows = kfdRows(data).slice().sort((a, b) => Number(a.katz_fd) - Number(b.katz_fd)).slice(-80);
  const plot = plotCanvas('kfdAllBarCanvas', 'Katz Fractal Dimension for Each Embedding', rows.length ? '' : 'No valid Katz FD rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 150, top = 42, width = w - 188, height = h - 72;
  const vals = rows.map(r => Number(r.katz_fd));
  const range = hfdValueRange(vals.concat([0]), 0.10);
  hfdDrawAxes(ctx, left, top, width, height, 'Katz Fractal Dimension', 'Embedding', range, {min: 1, max: rows.length}, {yTickCount: Math.min(5, rows.length)});
  const rowH = height / rows.length;
  rows.forEach((row, i) => {
    const y = top + i * rowH + rowH * 0.18;
    const x = hfdMapPoint(Number(row.katz_fd), range.min, range.max, left, left + width);
    ctx.fillStyle = '#00ffff';
    ctx.globalAlpha = 0.82;
    ctx.fillRect(left, y, Math.max(2, x - left), Math.max(2, rowH * 0.62));
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#dfffff';
    ctx.font = rowH < 10 ? '8px monospace' : '10px monospace';
    ctx.textAlign = 'right';
    const label = String(row.stem || `${row.embedding_dimension}dembedded_${row.channel}`).slice(0, 24);
    ctx.fillText(label, left - 8, y + rowH * 0.48);
  });
}

function kfdDrawMeanByDimension(data) {
  const rows = kfdByDimension(data).slice().sort((a,b) => Number(a.embedding_dimension) - Number(b.embedding_dimension));
  const plot = plotCanvas('kfdMeanCanvas', 'Mean Katz FD vs Embedding Dimension', rows.length ? '' : 'No by-dimension summary returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 62, top = 48, width = w - 96, height = h - 92;
  const xs = rows.map(r => Number(r.embedding_dimension));
  const ys = rows.map(r => Number(r.mean_katz_fd));
  const sds = rows.map(r => Number(r.std_katz_fd) || 0);
  const xr = hfdValueRange(xs, 0.08);
  const yr = hfdValueRange(ys.flatMap((y,i) => [y - sds[i], y + sds[i]]), 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding Dimension', 'Mean Katz Fractal Dimension', xr, yr);
  ctx.fillStyle = 'rgba(0,255,255,0.15)';
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = hfdMapPoint(Number(r.embedding_dimension), xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(Number(r.mean_katz_fd) - (Number(r.std_katz_fd) || 0), yr.min, yr.max, top + height, top);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  rows.slice().reverse().forEach((r) => {
    const x = hfdMapPoint(Number(r.embedding_dimension), xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(Number(r.mean_katz_fd) + (Number(r.std_katz_fd) || 0), yr.min, yr.max, top + height, top);
    ctx.lineTo(x, y);
  });
  ctx.closePath(); ctx.fill();
  ctx.beginPath();
  rows.forEach((r, i) => {
    const x = hfdMapPoint(Number(r.embedding_dimension), xr.min, xr.max, left, left + width);
    const y = hfdMapPoint(Number(r.mean_katz_fd), yr.min, yr.max, top + height, top);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = '#00ffff'; ctx.lineWidth = 1.8; ctx.stroke();
  rows.forEach((r) => { const x = hfdMapPoint(Number(r.embedding_dimension), xr.min, xr.max, left, left + width); const y = hfdMapPoint(Number(r.mean_katz_fd), yr.min, yr.max, top + height, top); ctx.fillStyle = '#00ffff'; ctx.beginPath(); ctx.arc(x, y, 4, 0, Math.PI*2); ctx.fill(); });
}

function kfdDrawHeatmap(data) {
  const matrix = kfdMatrixRows(data);
  const plot = plotCanvas('kfdHeatmapCanvas', 'Katz FD by Channel and Embedding Dimension', matrix.length ? '' : 'No channel-dimension matrix returned.');
  if (!plot || !matrix.length) return;
  const { ctx, w, h } = plot;
  const dims = Array.from(new Set(matrix.flatMap(r => (r.values || []).map(v => Number(v.embedding_dimension)).filter(Number.isFinite)))).sort((a,b) => a-b);
  const channels = matrix.map(r => String(r.channel));
  const left = 92, top = 48, width = w - 126, height = h - 86;
  const vals = finiteNumbers(matrix.flatMap(r => (r.values || []).map(v => Number(v.katz_fd))));
  const range = hfdValueRange(vals, 0.08);
  const cellW = width / Math.max(1, dims.length), cellH = height / Math.max(1, channels.length);
  const lookup = new Map();
  matrix.forEach(r => (r.values || []).forEach(v => lookup.set(`${r.channel}|${v.embedding_dimension}`, Number(v.katz_fd))));
  channels.forEach((ch, i) => {
    dims.forEach((dim, j) => {
      const v = lookup.get(`${ch}|${dim}`);
      if (!Number.isFinite(v)) return;
      ctx.fillStyle = interpolateColor(v, range.min, range.max);
      ctx.fillRect(left + j * cellW, top + i * cellH, Math.ceil(cellW), Math.ceil(cellH));
    });
  });
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding Dimension', 'Channel');
  ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  dims.forEach((dim, j) => ctx.fillText(String(dim), left + j * cellW + cellW/2, top + height + 16));
  ctx.textAlign = 'right'; channels.forEach((ch, i) => { if (channels.length <= 34 || i % Math.ceil(channels.length / 32) === 0) ctx.fillText(ch.slice(0, 8), left - 6, top + i * cellH + cellH * .65); });
  ctx.textAlign = 'left'; ctx.fillText(`${range.min.toFixed(2)} → ${range.max.toFixed(2)}`, left, h - 12);
}

function kfdDrawTrajectories(data) {
  const matrix = kfdMatrixRows(data);
  const plot = plotCanvas('kfdTrajectoriesCanvas', 'Katz FD Trajectories Across Embedding Dimensions', matrix.length ? '' : 'No channel trajectories returned.');
  if (!plot || !matrix.length) return;
  const { ctx, w, h } = plot;
  const all = matrix.flatMap(r => (r.values || []).map(v => ({dim: Number(v.embedding_dimension), val: Number(v.katz_fd)}))).filter(p => Number.isFinite(p.dim) && Number.isFinite(p.val));
  const left = 62, top = 48, width = w - 96, height = h - 92;
  const xr = hfdValueRange(all.map(p => p.dim), 0.08), yr = hfdValueRange(all.map(p => p.val), 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'Embedding Dimension', 'Katz Fractal Dimension', xr, yr);
  matrix.forEach(row => {
    const vals = (row.values || []).map(v => ({dim: Number(v.embedding_dimension), val: Number(v.katz_fd)})).filter(p => Number.isFinite(p.dim) && Number.isFinite(p.val)).sort((a,b)=>a.dim-b.dim);
    if (vals.length < 2) return;
    ctx.beginPath();
    vals.forEach((p, i) => { const x = hfdMapPoint(p.dim, xr.min, xr.max, left, left+width); const y = hfdMapPoint(p.val, yr.min, yr.max, top+height, top); if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.strokeStyle = 'rgba(0,255,255,0.65)'; ctx.lineWidth = 1.0; ctx.stroke();
  });
}

function renderKatzPanel(data) {
  const summary = data?.summary || {};
  const outputs = data?.outputs || {};
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Katz fractal-dimension analysis completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It used ${advEscape(summary.generated_embeddings ?? 0)} generated delay embeddings and ${advEscape(summary.precomputed_embeddings ?? 0)} precomputed embeddings.</div>
    <div class="grid two katz-plot-grid">
      <div class="card advanced-plot-card"><h3>All Embeddings Katz FD</h3><canvas id="kfdAllBarCanvas" class="advanced-plot-canvas" width="820" height="720"></canvas></div>
      <div class="card advanced-plot-card"><h3>Mean Katz FD vs Dimension</h3><canvas id="kfdMeanCanvas" class="advanced-plot-canvas" width="820" height="440"></canvas></div>
      <div class="card advanced-plot-card"><h3>Channel × Dimension Heatmap</h3><canvas id="kfdHeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Channel Trajectories</h3><canvas id="kfdTrajectoriesCanvas" class="advanced-plot-canvas" width="820" height="500"></canvas></div>
    </div>
    <h3>Mean Katz FD by embedding dimension</h3>${advRenderTable(data?.by_dimension || [])}
    <h3>Mean Katz FD by channel</h3>${advRenderTable(data?.by_channel || [])}
    <h3>Per-embedding results</h3>${advRenderTable(data?.rows || [])}
    ${outputHtml}`;
}

function katzParams() {
  const params = { mode: $('katzMode')?.value || 'ultra' };
  const dims = $('katzDims')?.value?.trim();
  if (dims) params.embedding_dims = dims;
  [['tau_ms','katzTauMs'], ['max_channels','katzMaxChannels'], ['max_analysis_samples','katzMaxSamples']].forEach(([name, id]) => {
    const raw = $(id)?.value?.trim();
    if (raw !== undefined && raw !== '') { const n = Number(raw); params[name] = Number.isFinite(n) ? n : raw; }
  });
  return params;
}
async function katzUseLatest() {
  const status = $('katzStatus'); if (status) { status.textContent = 'Looking for latest converted recording...'; status.className = 'status'; }
  try { const res = await fetch('/api/advanced-methods/latest-recording?t=' + Date.now(), {cache:'no-store'}); const data = await res.json(); if (!data.ok) throw new Error(data.error || 'No latest recording found.'); if ($('katzRecordingDir')) $('katzRecordingDir').value = data.recording_dir; if (status) { status.textContent = `Using latest converted recording: ${data.recording_dir}`; status.className = 'status ok'; } } catch (err) { if (status) { status.textContent = err.message; status.className = 'status failed'; } }
}
async function runKatzDirect() {
  showTab('katzTab'); const card = $('katzDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 30);
  const status = $('katzStatus'), output = $('katzResult'); const recordingDir = $('katzRecordingDir')?.value?.trim() || 'latest';
  if (status) { status.textContent = 'Running Katz fractal-dimension analysis...'; status.className = 'status'; }
  if (output) output.innerHTML = '<div class="method-note">Running Katz FD in the backend. Results will render here when the JSON returns.</div>';
  try { const res = await fetch('/api/advanced-methods/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ method_id:'katz_fractal_dimension', recording_dir: recordingDir, params: katzParams() }) }); const payload = await res.json(); latestKatzResult = payload; if (!payload.ok) throw new Error(payload.error || 'Katz FD failed.'); const kfd = payload.result?.katz_fractal_dimension || {}; if (status) { status.textContent = `Complete: ${kfd.summary?.ok_rows || 0} Katz rows`; status.className = 'status ok'; } if (output) { output.innerHTML = renderKatzPanel(kfd); setTimeout(() => renderKatzPlots(kfd), 30); } if ($('advancedMethodSelect')) $('advancedMethodSelect').value = 'katz_fractal_dimension'; } catch (err) { if (status) { status.textContent = err.message; status.className = 'status failed'; } if (output) output.innerHTML = `<div class="method-note">${advEscape(err.message)}</div>`; }
}
function copyKatzJson() { if (!latestKatzResult) { alert('No Katz FD result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestKatzResult, null, 2)); }


// ---- v0.11.19 Manual Wavelet Hurst frontend plots ----
let latestWaveletResult = null;
let latestMfdfaResult = null;

function whRows(data) { return Array.isArray(data?.rows) ? data.rows.filter(r => r && r.status === 'ok' && Number.isFinite(Number(r.hurst_exponent))) : []; }
function whChannelResults(data) { return Array.isArray(data?.channel_results) ? data.channel_results : []; }

function whDrawRawOffset(data) {
  const raw = data?.raw_plot || {};
  const time = finiteNumbers(raw.time_sec);
  const traces = Array.isArray(raw.zscore_traces) ? raw.zscore_traces : [];
  const channels = Array.isArray(raw.channels) ? raw.channels : [];
  const plot = plotCanvas('waveletRawCanvas', 'EEG Segment Used for Manual Wavelet Hurst Analysis', time.length && traces.length ? '' : 'No raw offset traces returned.');
  if (!plot || !time.length || !traces.length) return;
  const { ctx, w, h } = plot;
  const left = 82, top = 48, width = w - 118, height = h - 94;
  const offset = Number(raw.offset || 5.0);
  const n = Math.min(traces.length, channels.length || traces.length);
  const yMin = -2, yMax = Math.max(offset * Math.max(1, n - 1) + 2, 4);
  const xr = hfdValueRange(time, 0.02), yr = {min: yMin, max: yMax};
  hfdDrawAxes(ctx, left, top, width, height, 'Time from segment start (s)', 'Channel offset z-score', xr, yr);
  for (let i = 0; i < n; i++) {
    const arr = Array.isArray(traces[i]) ? traces[i].map(Number) : [];
    if (!arr.length) continue;
    ctx.beginPath();
    const m = Math.min(time.length, arr.length);
    for (let j = 0; j < m; j++) {
      const x = hfdMapPoint(time[j], xr.min, xr.max, left, left + width);
      const y = hfdMapPoint(arr[j] + i * offset, yr.min, yr.max, top + height, top);
      if (j === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = 'rgba(0,255,255,0.60)'; ctx.lineWidth = 0.65; ctx.stroke();
    if (n <= 36) { ctx.fillStyle = '#dfffff'; ctx.font = '9px monospace'; ctx.textAlign = 'right'; ctx.fillText(String(channels[i] || i), left - 8, hfdMapPoint(i * offset, yr.min, yr.max, top + height, top)); }
  }
  ctx.textAlign = 'left';
}

function whDrawBar(data, canvasId, field, title, xlabel, referenceLines = []) {
  const rows = whRows(data).slice().sort((a,b) => Number(a[field]) - Number(b[field]));
  const plot = plotCanvas(canvasId, title, rows.length ? '' : 'No wavelet Hurst rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 120, top = 48, width = w - 156, height = h - 76;
  const vals = rows.map(r => Number(r[field]));
  const range = hfdValueRange(vals.concat(referenceLines.map(r => r.value)), 0.10);
  hfdDrawAxes(ctx, left, top, width, height, xlabel, 'Channel', range, {min: 1, max: rows.length}, {yTickCount: Math.min(5, rows.length)});
  referenceLines.forEach(line => {
    if (!Number.isFinite(Number(line.value))) return;
    const x = hfdMapPoint(Number(line.value), range.min, range.max, left, left + width);
    ctx.strokeStyle = line.color || '#dfffff'; ctx.setLineDash(line.dash || [5,4]); ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + height); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = line.color || '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign = 'left'; ctx.fillText(line.label || hfdFormatTick(line.value), x + 4, top + 12);
  });
  const rowH = height / rows.length;
  rows.forEach((row, i) => {
    const y = top + i * rowH + rowH * 0.18;
    const x = hfdMapPoint(Number(row[field]), range.min, range.max, left, left + width);
    ctx.fillStyle = '#00ffff'; ctx.globalAlpha = 0.84; ctx.fillRect(left, y, Math.max(2, x - left), Math.max(2, rowH * 0.62)); ctx.globalAlpha = 1;
    ctx.fillStyle = '#dfffff'; ctx.font = rowH < 10 ? '8px monospace' : '10px monospace'; ctx.textAlign = 'right'; ctx.fillText(String(row.channel), left - 8, y + rowH * 0.48);
    if (rowH >= 14) { ctx.textAlign = 'left'; ctx.fillText(hfdFormatTick(row[field]), x + 4, y + rowH * 0.48); }
  });
}

function whDrawScalingCurves(data) {
  const results = whChannelResults(data).filter(r => Array.isArray(r.log2_scales_fit) && Array.isArray(r.log2_stdevs_fit) && r.log2_scales_fit.length >= 2);
  const plot = plotCanvas('waveletScalingAllCanvas', 'Manual Wavelet Scaling Curves Across Channels', results.length ? '' : 'No scaling curves returned.');
  if (!plot || !results.length) return;
  const { ctx, w, h } = plot;
  const left = 66, top = 48, width = w - 102, height = h - 92;
  const xs = finiteNumbers(results.flatMap(r => r.log2_scales_fit));
  const ys = finiteNumbers(results.flatMap(r => r.log2_stdevs_fit));
  const xr = hfdValueRange(xs, 0.08), yr = hfdValueRange(ys, 0.12);
  hfdDrawAxes(ctx, left, top, width, height, 'log2(scale)', 'log2(std(detail coeffs))', xr, yr);
  results.forEach(row => {
    const xvals = finiteNumbers(row.log2_scales_fit), yvals = finiteNumbers(row.log2_stdevs_fit);
    if (xvals.length < 2 || yvals.length < 2) return;
    ctx.beginPath();
    for (let i=0; i<Math.min(xvals.length, yvals.length); i++) { const x = hfdMapPoint(xvals[i], xr.min, xr.max, left, left+width); const y = hfdMapPoint(yvals[i], yr.min, yr.max, top+height, top); if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }
    ctx.strokeStyle = 'rgba(0,255,255,0.35)'; ctx.lineWidth = 1.0; ctx.stroke();
  });
}

function whDrawScalingGallery(data) {
  const results = whChannelResults(data).filter(r => Number.isFinite(Number(r.hurst_exponent)) && Number.isFinite(Number(r.fit_r2))).sort((a,b) => Number(b.fit_r2) - Number(a.fit_r2)).slice(0, 9);
  const plot = plotCanvas('waveletScalingGalleryCanvas', 'Scaling Curve Gallery — top fit channels', results.length ? '' : 'No scaling gallery data returned.');
  if (!plot || !results.length) return;
  const { ctx, w, h } = plot;
  const cols = 3, rows = Math.ceil(results.length / cols), gap = 18;
  const cellW = (w - 40 - gap * (cols - 1)) / cols, cellH = (h - 56 - gap * (rows - 1)) / rows;
  results.forEach((row, idx) => {
    const col = idx % cols, rr = Math.floor(idx / cols);
    const left = 20 + col * (cellW + gap), top = 42 + rr * (cellH + gap);
    const xvals = finiteNumbers(row.log2_scales_fit), yvals = finiteNumbers(row.log2_stdevs_fit), fit = finiteNumbers(row.fit_line);
    if (xvals.length < 2 || yvals.length < 2) return;
    const xr = hfdValueRange(xvals, 0.08), yr = hfdValueRange(yvals.concat(fit), 0.12);
    hfdDrawAxes(ctx, left + 32, top + 12, cellW - 40, cellH - 44, 'log2(scale)', 'log2(std)', xr, yr, {xTickCount:3, yTickCount:3, tickFont:'8px monospace', labelFont:'9px monospace'});
    ctx.beginPath();
    xvals.forEach((xv,i)=>{ const x=hfdMapPoint(xv,xr.min,xr.max,left+32,left+cellW-8); const y=hfdMapPoint(yvals[i],yr.min,yr.max,top+cellH-32,top+24); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
    ctx.strokeStyle='#00ffff'; ctx.lineWidth=1.2; ctx.stroke();
    if (fit.length === xvals.length) { ctx.beginPath(); xvals.forEach((xv,i)=>{ const x=hfdMapPoint(xv,xr.min,xr.max,left+32,left+cellW-8); const y=hfdMapPoint(fit[i],yr.min,yr.max,top+cellH-32,top+24); if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y); }); ctx.strokeStyle='#dfffff'; ctx.setLineDash([4,3]); ctx.stroke(); ctx.setLineDash([]); }
    ctx.fillStyle='#dfffff'; ctx.font='10px monospace'; ctx.textAlign='left'; ctx.fillText(`${row.channel} H=${Number(row.hurst_exponent).toFixed(3)} R²=${Number(row.fit_r2).toFixed(3)}`, left+4, top+10);
  });
}

function whDrawScatter(data) {
  const rows = whRows(data);
  const plot = plotCanvas('waveletHurstVsR2Canvas', 'Hurst Exponent vs Scaling Fit Reliability', rows.length ? '' : 'No H/R² rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 66, top = 48, width = w - 102, height = h - 90;
  const xr = hfdValueRange(rows.map(r=>Number(r.hurst_exponent)), 0.12), yr = hfdValueRange(rows.map(r=>Number(r.fit_r2)).concat([0.9, 0.95]), 0.08);
  hfdDrawAxes(ctx, left, top, width, height, 'Hurst exponent', 'Scaling fit R²', xr, yr);
  rows.forEach(row => { const x=hfdMapPoint(row.hurst_exponent,xr.min,xr.max,left,left+width); const y=hfdMapPoint(row.fit_r2,yr.min,yr.max,top+height,top); ctx.fillStyle='#00ffff'; ctx.strokeStyle='#dfffff'; ctx.beginPath(); ctx.arc(x,y,4+Number(row.n_levels || 0)*0.6,0,Math.PI*2); ctx.fill(); ctx.stroke(); });
}

function whDrawHeatmap(data, canvasId, title, fieldName) {
  const results = whChannelResults(data).filter(r => Array.isArray(r[fieldName]) && r[fieldName].length);
  const plot = plotCanvas(canvasId, title, results.length ? '' : 'No heatmap data returned.');
  if (!plot || !results.length) return;
  const { ctx, w, h } = plot;
  const channels = results.slice().sort((a,b)=>Number(b.hurst_exponent)-Number(a.hurst_exponent)).map(r=>r.channel);
  const maxLevels = Math.max(...results.map(r => r[fieldName].length));
  const left = 92, top = 48, width = w - 126, height = h - 88;
  const vals = finiteNumbers(results.flatMap(r => r[fieldName]));
  const useLog = fieldName === 'detail_stdevs';
  const plottedVals = useLog ? vals.map(v => Math.log2(v + 1e-12)) : vals;
  const range = hfdValueRange(plottedVals, 0.04);
  const cellW = width / Math.max(1, maxLevels), cellH = height / Math.max(1, channels.length);
  const lookup = new Map(results.map(r=>[r.channel, r]));
  channels.forEach((ch,i)=>{
    const arr = lookup.get(ch)?.[fieldName] || [];
    for(let j=0;j<maxLevels;j++){ let v=Number(arr[j]); if(!Number.isFinite(v)) continue; if(useLog) v=Math.log2(v+1e-12); ctx.fillStyle=interpolateColor(v, range.min, range.max); ctx.fillRect(left+j*cellW, top+i*cellH, Math.ceil(cellW), Math.ceil(cellH)); }
  });
  hfdDrawAxes(ctx, left, top, width, height, 'Wavelet detail level', 'Channel');
  ctx.fillStyle = '#dfffff'; ctx.font = '10px monospace'; ctx.textAlign='center';
  for(let j=0;j<maxLevels;j++) ctx.fillText(`L${j+1}`, left+j*cellW+cellW/2, top+height+16);
  ctx.textAlign='right'; channels.forEach((ch,i)=>{ if(channels.length<=34 || i%Math.ceil(channels.length/32)===0) ctx.fillText(String(ch).slice(0,8), left-6, top+i*cellH+cellH*.65); });
  ctx.textAlign='left'; ctx.fillText(`${hfdFormatTick(range.min)} → ${hfdFormatTick(range.max)}`, left, h-12);
}

function whDrawFdProxy(data) { whDrawBar(data, 'waveletFdProxyCanvas', 'fractal_dimension_proxy', 'Fractal-Dimension Proxy from Wavelet Hurst: D ≈ 2 - H', 'Fractal-dimension proxy'); }

function whDrawRegional(data) {
  const rows = Array.isArray(data?.regional_summary) ? data.regional_summary.filter(r => Number.isFinite(Number(r.mean_hurst))) : [];
  const plot = plotCanvas('waveletRegionalCanvas', 'Regional Summary of Wavelet Hurst Exponent', rows.length ? '' : 'No regional summary returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 132, top = 48, width = w - 168, height = h - 80;
  const vals = rows.flatMap(r => [Number(r.mean_hurst) - (Number(r.std_hurst)||0), Number(r.mean_hurst) + (Number(r.std_hurst)||0)]);
  const range = hfdValueRange(vals, 0.10);
  hfdDrawAxes(ctx, left, top, width, height, 'Mean Hurst exponent ± SD', 'Region', range, {min: 1, max: rows.length}, {yTickCount: Math.min(5, rows.length)});
  const rowH = height / rows.length;
  rows.forEach((row,i)=>{ const y=top+i*rowH+rowH*.2; const x=hfdMapPoint(row.mean_hurst,range.min,range.max,left,left+width); ctx.fillStyle='#00ffff'; ctx.globalAlpha=.84; ctx.fillRect(left,y,Math.max(2,x-left),Math.max(2,rowH*.58)); ctx.globalAlpha=1; const xerr=hfdMapPoint(Number(row.mean_hurst)+(Number(row.std_hurst)||0),range.min,range.max,left,left+width); ctx.strokeStyle='#dfffff'; ctx.beginPath(); ctx.moveTo(x,y+rowH*.3); ctx.lineTo(xerr,y+rowH*.3); ctx.stroke(); ctx.fillStyle='#dfffff'; ctx.font='10px monospace'; ctx.textAlign='right'; ctx.fillText(String(row.region),left-8,y+rowH*.45); ctx.textAlign='left'; ctx.fillText(`n=${row.n} R²=${Number(row.mean_r2).toFixed(2)}`,x+4,y+rowH*.45); });
}

function whDrawAsymmetry(data) {
  const rows = Array.isArray(data?.asymmetry) ? data.asymmetry.filter(r => Number.isFinite(Number(r.H_left_minus_right))) : [];
  const plot = plotCanvas('waveletAsymmetryCanvas', 'Homologous Left-Right Hurst Asymmetry', rows.length ? '' : 'No homologous left/right pairs returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot;
  const left = 100, top = 48, width = w - 134, height = h - 78;
  const vals = rows.map(r => Number(r.H_left_minus_right)); const maxAbs = Math.max(...vals.map(v=>Math.abs(v)), 0.001);
  hfdDrawAxes(ctx, left, top, width, height, 'H_left - H_right', 'Pair', {min:-maxAbs,max:maxAbs}, {min:1,max:rows.length}, {yTickCount:Math.min(5,rows.length)});
  const zero = hfdMapPoint(0,-maxAbs,maxAbs,left,left+width); ctx.strokeStyle='#dfffff'; ctx.beginPath(); ctx.moveTo(zero,top); ctx.lineTo(zero,top+height); ctx.stroke();
  const rowH = height / rows.length;
  rows.slice().sort((a,b)=>Number(a.H_left_minus_right)-Number(b.H_left_minus_right)).forEach((row,i)=>{ const y=top+i*rowH+rowH*.2; const x=hfdMapPoint(row.H_left_minus_right,-maxAbs,maxAbs,left,left+width); ctx.fillStyle='#00ffff'; ctx.globalAlpha=.84; ctx.fillRect(Math.min(zero,x),y,Math.abs(x-zero),Math.max(2,rowH*.58)); ctx.globalAlpha=1; ctx.fillStyle='#dfffff'; ctx.font='10px monospace'; ctx.textAlign='right'; ctx.fillText(String(row.pair),left-8,y+rowH*.45); });
}

function whDrawTopographic(data) {
  const pts = Array.isArray(data?.topographic_points) ? data.topographic_points.filter(p => Number.isFinite(Number(p.hurst_exponent))) : [];
  const plot = plotCanvas('waveletTopoCanvas', 'Approximate Scalp Map of Wavelet Hurst Exponent', pts.length ? '' : 'No topographic points returned.');
  if (!plot || !pts.length) return;
  const { ctx, w, h } = plot; const cx=w/2, cy=h/2+12, radius=Math.min(w,h)*.34; const vals=pts.map(p=>Number(p.hurst_exponent)); const range=hfdValueRange(vals,.08);
  ctx.strokeStyle='rgba(0,255,255,.45)'; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(cx,cy,radius*1.12,0,Math.PI*2); ctx.stroke(); ctx.beginPath(); ctx.moveTo(cx,cy-radius*1.12); ctx.lineTo(cx-14,cy-radius*1.27); ctx.lineTo(cx+14,cy-radius*1.27); ctx.closePath(); ctx.stroke();
  pts.forEach(p=>{ const x=cx+Number(p.x)*radius, y=cy-Number(p.y)*radius, v=Number(p.hurst_exponent); ctx.fillStyle=interpolateColor(v,range.min,range.max); ctx.strokeStyle='#dfffff'; ctx.beginPath(); ctx.arc(x,y,16,0,Math.PI*2); ctx.fill(); ctx.stroke(); ctx.fillStyle='#001010'; ctx.font='bold 9px monospace'; ctx.textAlign='center'; ctx.fillText(String(p.channel).slice(0,5),x,y+3); });
  ctx.textAlign='left'; ctx.fillStyle='#dfffff'; ctx.font='11px monospace'; ctx.fillText(`Hurst ${hfdFormatTick(range.min)} → ${hfdFormatTick(range.max)}`, 18, h-18);
}

function whDrawTemporalStability(data) {
  const rows = Array.isArray(data?.temporal_summary) ? data.temporal_summary.filter(r => Number.isFinite(Number(r.mean_hurst))) : [];
  const plot = plotCanvas('waveletTemporalCanvas', 'Temporal Stability of Manual Wavelet Hurst Estimate', rows.length ? '' : 'No temporal stability rows returned.');
  if (!plot || !rows.length) return;
  const { ctx, w, h } = plot; const left=66, top=48, width=w-102, height=h-92; const xs=rows.map(r=>Number(r.window)); const ys=rows.flatMap(r=>[Number(r.mean_hurst), Number(r.mean_r2)]).filter(Number.isFinite); const xr=hfdValueRange(xs,.08), yr=hfdValueRange(ys,.10); hfdDrawAxes(ctx,left,top,width,height,'Window index','Mean Hurst / mean R²',xr,yr);
  [['mean_hurst','#00ffff'],['mean_r2','#dfffff']].forEach(([field,color])=>{ ctx.beginPath(); rows.forEach((r,i)=>{ const x=hfdMapPoint(r.window,xr.min,xr.max,left,left+width); const y=hfdMapPoint(r[field],yr.min,yr.max,top+height,top); if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y); }); ctx.strokeStyle=color; ctx.lineWidth=2; ctx.stroke(); rows.forEach(r=>{ const x=hfdMapPoint(r.window,xr.min,xr.max,left,left+width); const y=hfdMapPoint(r[field],yr.min,yr.max,top+height,top); ctx.fillStyle=color; ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.fill(); }); });
  ctx.fillStyle='#dfffff'; ctx.font='11px monospace'; ctx.fillText('Hurst', left+12, top+16); ctx.fillText('R²', left+12, top+32);
}

function whDrawTemporalHeatmap(data) {
  const matrix = Array.isArray(data?.temporal_matrix) ? data.temporal_matrix : [];
  const plot = plotCanvas('waveletTemporalHeatmapCanvas', 'Temporal Hurst Heatmap by Channel', matrix.length ? '' : 'No temporal channel matrix returned.');
  if (!plot || !matrix.length) return;
  const { ctx, w, h } = plot; const windows=Array.from(new Set(matrix.flatMap(r=>(r.values||[]).map(v=>Number(v.window)).filter(Number.isFinite)))).sort((a,b)=>a-b); const channels=matrix.map(r=>String(r.channel)); const left=92, top=48, width=w-126, height=h-88; const vals=finiteNumbers(matrix.flatMap(r=>(r.values||[]).map(v=>Number(v.hurst_exponent)))); const range=hfdValueRange(vals,.04); const cellW=width/Math.max(1,windows.length), cellH=height/Math.max(1,channels.length); const lookup=new Map(); matrix.forEach(r=>(r.values||[]).forEach(v=>lookup.set(`${r.channel}|${v.window}`,Number(v.hurst_exponent)))); channels.forEach((ch,i)=>{ windows.forEach((win,j)=>{ const v=lookup.get(`${ch}|${win}`); if(!Number.isFinite(v))return; ctx.fillStyle=interpolateColor(v,range.min,range.max); ctx.fillRect(left+j*cellW,top+i*cellH,Math.ceil(cellW),Math.ceil(cellH)); }); }); hfdDrawAxes(ctx,left,top,width,height,'Window','Channel'); ctx.fillStyle='#dfffff'; ctx.font='10px monospace'; ctx.textAlign='center'; windows.forEach((win,j)=>ctx.fillText(String(win),left+j*cellW+cellW/2,top+height+16)); ctx.textAlign='right'; channels.forEach((ch,i)=>{ if(channels.length<=34 || i%Math.ceil(channels.length/32)===0) ctx.fillText(ch.slice(0,8),left-6,top+i*cellH+cellH*.65); }); ctx.textAlign='left'; ctx.fillText(`${hfdFormatTick(range.min)} → ${hfdFormatTick(range.max)}`,left,h-12);
}

function renderWaveletPlots(data) {
  whDrawRawOffset(data);
  const rows = whRows(data); const hvals = rows.map(r=>Number(r.hurst_exponent)); const rvals = rows.map(r=>Number(r.fit_r2));
  whDrawBar(data, 'waveletHurstBarCanvas', 'hurst_exponent', 'Manual Wavelet-Based Hurst Exponent per EEG Channel', 'Hurst exponent', [ {value: hvals.reduce((a,b)=>a+b,0)/Math.max(1,hvals.length), label:'mean H', color:'#dfffff'}, {value: hvals.slice().sort((a,b)=>a-b)[Math.floor(hvals.length/2)] || 0, label:'median H', color:'#66ffff', dash:[2,3]} ]);
  whDrawBar(data, 'waveletFitR2BarCanvas', 'fit_r2', 'Manual Wavelet Hurst Fit Quality per EEG Channel', 'Linear scaling fit R²', [ {value: 0.95, label:'R²=0.95', color:'#dfffff'}, {value: 0.90, label:'R²=0.90', color:'#66ffff', dash:[2,3]} ]);
  whDrawScalingCurves(data); whDrawScalingGallery(data); whDrawScatter(data); whDrawHeatmap(data, 'waveletStdevHeatmapCanvas', 'Wavelet Detail Coefficient Scale Map: log2(std(detail))', 'detail_stdevs'); whDrawHeatmap(data, 'waveletEnergyHeatmapCanvas', 'Relative Wavelet Detail Energy by Channel', 'relative_detail_energies'); whDrawFdProxy(data); whDrawRegional(data); whDrawAsymmetry(data); whDrawTopographic(data); whDrawTemporalStability(data); whDrawTemporalHeatmap(data);
}

function renderWaveletPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {};
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Manual Wavelet Hurst completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode using <strong>${advEscape(summary.wavelet || '')}</strong>. Mean H=${advEscape(advFormat(summary.mean_hurst))}; mean fit R²=${advEscape(advFormat(summary.mean_fit_r2))}.</div>
    <div class="grid two wavelet-plot-grid">
      <div class="card advanced-plot-card"><h3>Raw EEG offset traces</h3><canvas id="waveletRawCanvas" class="advanced-plot-canvas" width="900" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Hurst exponent by channel</h3><canvas id="waveletHurstBarCanvas" class="advanced-plot-canvas" width="820" height="720"></canvas></div>
      <div class="card advanced-plot-card"><h3>Fit R² by channel</h3><canvas id="waveletFitR2BarCanvas" class="advanced-plot-canvas" width="820" height="720"></canvas></div>
      <div class="card advanced-plot-card"><h3>Scaling curves</h3><canvas id="waveletScalingAllCanvas" class="advanced-plot-canvas" width="820" height="500"></canvas></div>
      <div class="card advanced-plot-card"><h3>Scaling curve gallery</h3><canvas id="waveletScalingGalleryCanvas" class="advanced-plot-canvas" width="900" height="680"></canvas></div>
      <div class="card advanced-plot-card"><h3>Hurst vs fit quality</h3><canvas id="waveletHurstVsR2Canvas" class="advanced-plot-canvas" width="820" height="520"></canvas></div>
      <div class="card advanced-plot-card"><h3>Detail stdev heatmap</h3><canvas id="waveletStdevHeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Relative energy heatmap</h3><canvas id="waveletEnergyHeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Fractal dimension proxy</h3><canvas id="waveletFdProxyCanvas" class="advanced-plot-canvas" width="820" height="720"></canvas></div>
      <div class="card advanced-plot-card"><h3>Regional summary</h3><canvas id="waveletRegionalCanvas" class="advanced-plot-canvas" width="820" height="500"></canvas></div>
      <div class="card advanced-plot-card"><h3>Hemisphere asymmetry</h3><canvas id="waveletAsymmetryCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
      <div class="card advanced-plot-card"><h3>Topographic Hurst map</h3><canvas id="waveletTopoCanvas" class="advanced-plot-canvas" width="700" height="700"></canvas></div>
      <div class="card advanced-plot-card"><h3>Temporal stability</h3><canvas id="waveletTemporalCanvas" class="advanced-plot-canvas" width="820" height="500"></canvas></div>
      <div class="card advanced-plot-card"><h3>Temporal channel heatmap</h3><canvas id="waveletTemporalHeatmapCanvas" class="advanced-plot-canvas" width="820" height="560"></canvas></div>
    </div>
    <h3>Per-channel Hurst results</h3>${advRenderTable(data?.rows || [])}
    <h3>Regional summary</h3>${advRenderTable(data?.regional_summary || [])}
    ${outputHtml}`;
}

function waveletParams() {
  const params = { mode: $('waveletMode')?.value || 'ultra', wavelet: $('waveletName')?.value || 'haar', temporal_stability: ($('waveletTemporal')?.value || 'true') === 'true' };
  [['max_level','waveletMaxLevel'], ['temporal_windows','waveletTemporalWindows'], ['max_channels','waveletMaxChannels'], ['max_analysis_samples','waveletMaxSamples']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function waveletUseLatest() { const status=$('waveletStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('waveletRecordingDir')) $('waveletRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runWaveletDirect() { showTab('waveletTab'); const card=$('waveletDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('waveletStatus'), output=$('waveletResult'), recordingDir=$('waveletRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Manual Wavelet Hurst analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running wavelet Hurst in the backend. Plots will render here when JSON returns.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'wavelet_hurst_exponent', recording_dir:recordingDir, params:waveletParams()})}); const payload=await res.json(); latestWaveletResult=payload; if(!payload.ok) throw new Error(payload.error || 'Wavelet Hurst failed.'); const wh=payload.result?.wavelet_hurst_exponent || {}; if(status){status.textContent=`Complete: ${wh.summary?.n_channels || 0} channels · mean H=${advFormat(wh.summary?.mean_hurst)}`; status.className='status ok';} if(output){output.innerHTML=renderWaveletPanel(wh); setTimeout(()=>renderWaveletPlots(wh),30);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='wavelet_hurst_exponent'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyWaveletJson(){ if(!latestWaveletResult){ alert('No Wavelet Hurst result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestWaveletResult,null,2)); }



function mfdfaImageUrl(plot) {
  if (!plot) return '';
  if (plot.url) return plot.url;
  if (plot.path) return '/api/file?path=' + encodeURIComponent(plot.path);
  return '';
}
function renderMfdfaPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `MFDFA plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No MFDFA plot paths returned.</div>';
  return `<div class="method-note">Manual Expert MFDFA completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. These are the backend-generated black/cyan Matplotlib plots from the notebook-style workflow, shown directly so labels, ticks, and colormaps stay intact.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>q values: ${advFormat(summary.q_count)}</span><span>mean Δα: ${advFormat(summary.mean_alpha_width)}</span><span>mean h(0): ${advFormat(summary.mean_hq_q0)}</span><span>mean R²: ${advFormat(summary.mean_hq_fit_r2)}</span></div>
    ${plotHtml}
    <h3>MFDFA summary table</h3>${advRenderTable(data?.rows || [])}
    <h3>Hemisphere asymmetry</h3>${advRenderTable(data?.asymmetry || [])}
    ${outputHtml}`;
}
function mfdfaParams() {
  const params = { mode: $('mfdfaMode')?.value || 'ultra', temporal_stability: ($('mfdfaTemporal')?.value || 'false') === 'true', save_per_channel_plots: ($('mfdfaPerChannelPlots')?.value || 'false') === 'true' };
  [['q_min','mfdfaQMin'], ['q_max','mfdfaQMax'], ['q_count','mfdfaQCount'], ['poly_order','mfdfaPolyOrder'], ['max_channels','mfdfaMaxChannels'], ['max_analysis_samples','mfdfaMaxSamples'], ['segment_start_sec','mfdfaSegmentStart'], ['segment_end_sec','mfdfaSegmentEnd']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function mfdfaUseLatest() { const status=$('mfdfaStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('mfdfaRecordingDir')) $('mfdfaRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runMfdfaDirect() { showTab('mfdfaTab'); const card=$('mfdfaDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('mfdfaStatus'), output=$('mfdfaResult'), recordingDir=$('mfdfaRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Manual Expert MFDFA analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running MFDFA in the backend. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'manual_expert_mfdfa', recording_dir:recordingDir, params:mfdfaParams()})}); const payload=await res.json(); latestMfdfaResult=payload; if(!payload.ok) throw new Error(payload.error || 'Manual Expert MFDFA failed.'); const mfdfa=payload.result?.manual_expert_mfdfa || {}; if(status){status.textContent=`Complete: ${mfdfa.summary?.n_channels || 0} channels · mean Δα=${advFormat(mfdfa.summary?.mean_alpha_width)}`; status.className='status ok';} if(output){output.innerHTML=renderMfdfaPanel(mfdfa);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='manual_expert_mfdfa'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyMfdfaJson(){ if(!latestMfdfaResult){ alert('No Manual Expert MFDFA result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestMfdfaResult,null,2)); }


let latestMfdfaSpectrumResult = null;
function renderMfdfaSpectrumPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `MFDFA spectrum plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No MFDFA spectrum plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Manual MFDFA Spectrum completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It generated per-channel h(q), tau(q), and f(alpha) triptychs plus the spectrum-width ranking using backend Matplotlib so labels and ticks stay visible.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>scales: ${advFormat(summary.scale_count)}</span><span>q values: ${advFormat(summary.q_count)}</span><span>mean Δα: ${advFormat(summary.mean_alpha_width)}</span><span>poly order: ${advFormat(summary.poly_order)}</span></div>
    ${plotHtml}
    <h3>MFDFA spectrum summary table</h3>${advRenderTable(data?.rows || [])}
    ${outputHtml}`;
}
function mfdfaSpectrumParams() {
  const params = { mode: $('mfdfaSpectrumMode')?.value || 'ultra' };
  [['q_min','mfdfaSpectrumQMin'], ['q_max','mfdfaSpectrumQMax'], ['q_count','mfdfaSpectrumQCount'], ['poly_order','mfdfaSpectrumPolyOrder'], ['scale_min_power','mfdfaSpectrumScaleMin'], ['scale_max_power','mfdfaSpectrumScaleMax'], ['n_scales','mfdfaSpectrumNScales'], ['max_channels','mfdfaSpectrumMaxChannels'], ['max_analysis_samples','mfdfaSpectrumMaxSamples'], ['segment_start_sec','mfdfaSpectrumSegmentStart'], ['segment_end_sec','mfdfaSpectrumSegmentEnd']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function mfdfaSpectrumUseLatest() { const status=$('mfdfaSpectrumStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('mfdfaSpectrumRecordingDir')) $('mfdfaSpectrumRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runMfdfaSpectrumDirect() { showTab('mfdfaSpectrumTab'); const card=$('mfdfaSpectrumDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('mfdfaSpectrumStatus'), output=$('mfdfaSpectrumResult'), recordingDir=$('mfdfaSpectrumRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Manual MFDFA Spectrum analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running compact MFDFA spectrum in the backend. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'manual_mfdfa_spectrum', recording_dir:recordingDir, params:mfdfaSpectrumParams()})}); const payload=await res.json(); latestMfdfaSpectrumResult=payload; if(!payload.ok) throw new Error(payload.error || 'Manual MFDFA Spectrum failed.'); const spectrum=payload.result?.manual_mfdfa_spectrum || {}; if(status){status.textContent=`Complete: ${spectrum.summary?.n_channels || 0} channels · mean Δα=${advFormat(spectrum.summary?.mean_alpha_width)}`; status.className='status ok';} if(output){output.innerHTML=renderMfdfaSpectrumPanel(spectrum);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='manual_mfdfa_spectrum'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyMfdfaSpectrumJson(){ if(!latestMfdfaSpectrumResult){ alert('No Manual MFDFA Spectrum result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestMfdfaSpectrumResult,null,2)); }


let latestMfdfaShuffleResult = null;
function renderMfdfaShufflePanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `MFDFA shuffle plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No MFDFA shuffle plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">MFDFA Shuffle Surrogate completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It ran all available channels by default, comparing original Δα to shuffled surrogate Δα distributions in backend Matplotlib so labels, ticks, legends, and histograms stay visible.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>shuffles/channel: ${advFormat(summary.n_shuffles)}</span><span>scales: ${advFormat(summary.scale_count)}</span><span>q values: ${advFormat(summary.q_count)}</span><span>mean original Δα: ${advFormat(summary.mean_original_alpha_width)}</span><span>mean shuffle Δα: ${advFormat(summary.mean_shuffle_alpha_width)}</span><span>mean Δ: ${advFormat(summary.mean_delta_width)}</span><span>top Δ channel: ${advEscape(summary.top_delta_channel || '')}</span></div>
    ${plotHtml}
    <h3>Shuffle-surrogate summary table</h3>${advRenderTable(data?.rows || [])}
    <h3>Shuffle widths</h3>${advRenderTable(data?.shuffle_rows || [])}
    ${outputHtml}`;
}
function mfdfaShuffleParams() {
  const params = { mode: $('mfdfaShuffleMode')?.value || 'ultra' };
  [['n_shuffles','mfdfaShuffleNShuffles'], ['random_seed','mfdfaShuffleSeed'], ['q_min','mfdfaShuffleQMin'], ['q_max','mfdfaShuffleQMax'], ['q_count','mfdfaShuffleQCount'], ['poly_order','mfdfaShufflePolyOrder'], ['scale_min_power','mfdfaShuffleScaleMin'], ['scale_max_power','mfdfaShuffleScaleMax'], ['n_scales','mfdfaShuffleNScales'], ['max_channels','mfdfaShuffleMaxChannels'], ['max_analysis_samples','mfdfaShuffleMaxSamples'], ['segment_start_sec','mfdfaShuffleSegmentStart'], ['segment_end_sec','mfdfaShuffleSegmentEnd']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function mfdfaShuffleUseLatest() { const status=$('mfdfaShuffleStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('mfdfaShuffleRecordingDir')) $('mfdfaShuffleRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runMfdfaShuffleDirect() { showTab('mfdfaShuffleTab'); const card=$('mfdfaShuffleDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('mfdfaShuffleStatus'), output=$('mfdfaShuffleResult'), recordingDir=$('mfdfaShuffleRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running MFDFA Shuffle Surrogate analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running all-channel original-vs-shuffled MFDFA width comparison. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'manual_mfdfa_shuffle_surrogate', recording_dir:recordingDir, params:mfdfaShuffleParams()})}); const payload=await res.json(); latestMfdfaShuffleResult=payload; if(!payload.ok) throw new Error(payload.error || 'MFDFA Shuffle Surrogate failed.'); const shuffle=payload.result?.manual_mfdfa_shuffle_surrogate || {}; if(status){status.textContent=`Complete: ${shuffle.summary?.n_channels || 0} channels · mean Δ=${advFormat(shuffle.summary?.mean_delta_width)}`; status.className='status ok';} if(output){output.innerHTML=renderMfdfaShufflePanel(shuffle);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='manual_mfdfa_shuffle_surrogate'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyMfdfaShuffleJson(){ if(!latestMfdfaShuffleResult){ alert('No MFDFA Shuffle Surrogate result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestMfdfaShuffleResult,null,2)); }

let latestMfdfaIaaftResult = null;
function renderMfdfaIaaftPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `MFDFA IAAFT plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No MFDFA IAAFT plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">MFDFA IAAFT Surrogate completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It ran all available channels by default, comparing original Δα to IAAFT surrogate Δα distributions in backend Matplotlib so labels, ticks, legends, and histograms stay visible.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>IAAFT/channel: ${advFormat(summary.n_surrogates)}</span><span>iters: ${advFormat(summary.iaaft_iters)}</span><span>scales: ${advFormat(summary.scale_count)}</span><span>q values: ${advFormat(summary.q_count)}</span><span>mean original Δα: ${advFormat(summary.mean_original_alpha_width)}</span><span>mean IAAFT Δα: ${advFormat(summary.mean_iaaft_alpha_width)}</span><span>mean Δ: ${advFormat(summary.mean_delta_width)}</span><span>top Δ channel: ${advEscape(summary.top_delta_channel || '')}</span></div>
    ${plotHtml}
    <h3>IAAFT-surrogate summary table</h3>${advRenderTable(data?.rows || [])}
    <h3>IAAFT surrogate widths</h3>${advRenderTable(data?.surrogate_rows || [])}
    ${outputHtml}`;
}
function mfdfaIaaftParams() {
  const params = { mode: $('mfdfaIaaftMode')?.value || 'ultra' };
  [['n_surrogates','mfdfaIaaftNSurrogates'], ['iaaft_iters','mfdfaIaaftIters'], ['random_seed','mfdfaIaaftSeed'], ['q_min','mfdfaIaaftQMin'], ['q_max','mfdfaIaaftQMax'], ['q_count','mfdfaIaaftQCount'], ['poly_order','mfdfaIaaftPolyOrder'], ['scale_min_power','mfdfaIaaftScaleMin'], ['scale_max_power','mfdfaIaaftScaleMax'], ['n_scales','mfdfaIaaftNScales'], ['max_channels','mfdfaIaaftMaxChannels'], ['max_analysis_samples','mfdfaIaaftMaxSamples'], ['segment_start_sec','mfdfaIaaftSegmentStart'], ['segment_end_sec','mfdfaIaaftSegmentEnd']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function mfdfaIaaftUseLatest() { const status=$('mfdfaIaaftStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('mfdfaIaaftRecordingDir')) $('mfdfaIaaftRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runMfdfaIaaftDirect() { showTab('mfdfaIaaftTab'); const card=$('mfdfaIaaftDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('mfdfaIaaftStatus'), output=$('mfdfaIaaftResult'), recordingDir=$('mfdfaIaaftRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running MFDFA IAAFT Surrogate analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running all-channel original-vs-IAAFT MFDFA width comparison. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'manual_mfdfa_iaaft_surrogate', recording_dir:recordingDir, params:mfdfaIaaftParams()})}); const payload=await res.json(); latestMfdfaIaaftResult=payload; if(!payload.ok) throw new Error(payload.error || 'MFDFA IAAFT Surrogate failed.'); const iaaft=payload.result?.manual_mfdfa_iaaft_surrogate || {}; if(status){status.textContent=`Complete: ${iaaft.summary?.n_channels || 0} channels · mean Δ=${advFormat(iaaft.summary?.mean_delta_width)}`; status.className='status ok';} if(output){output.innerHTML=renderMfdfaIaaftPanel(iaaft);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='manual_mfdfa_iaaft_surrogate'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyMfdfaIaaftJson(){ if(!latestMfdfaIaaftResult){ alert('No MFDFA IAAFT Surrogate result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestMfdfaIaaftResult,null,2)); }



let latestWaveletLeaderResult = null;
function renderWaveletLeaderPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `Wavelet leader plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No wavelet-leader plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Wavelet Leader Multifractal completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It computes manual Haar wavelet leaders, zeta(q), alpha(q), and f(alpha) for all available channels by default, using backend-generated black/cyan Matplotlib plots.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>q values: ${advFormat(summary.q_count)}</span><span>mean Δα: ${advFormat(summary.mean_alpha_width)}</span><span>top channel: ${advEscape(summary.top_alpha_width_channel || '')}</span><span>plots: ${advFormat(summary.plot_count)}</span></div>
    ${plotHtml}
    <h3>Wavelet leader summary table</h3>${advRenderTable(data?.rows || [])}
    <h3>zeta / alpha / f(alpha) rows</h3>${advRenderTable(data?.zeta_rows || [])}
    ${outputHtml}`;
}
function waveletLeaderParams() {
  const params = { mode: $('waveletLeaderMode')?.value || 'ultra' };
  [['q_min','waveletLeaderQMin'], ['q_max','waveletLeaderQMax'], ['q_count','waveletLeaderQCount'], ['fit_j_min','waveletLeaderFitJMin'], ['fit_j_max','waveletLeaderFitJMax'], ['min_leaders_per_scale','waveletLeaderMinLeaders'], ['max_channels','waveletLeaderMaxChannels'], ['max_analysis_samples','waveletLeaderMaxSamples'], ['segment_start_sec','waveletLeaderSegmentStart'], ['segment_end_sec','waveletLeaderSegmentEnd']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function waveletLeaderUseLatest() { const status=$('waveletLeaderStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('waveletLeaderRecordingDir')) $('waveletLeaderRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runWaveletLeaderDirect() { showTab('waveletLeaderTab'); const card=$('waveletLeaderDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('waveletLeaderStatus'), output=$('waveletLeaderResult'), recordingDir=$('waveletLeaderRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Wavelet Leader Multifractal analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running all-channel manual Haar wavelet-leader multifractal analysis. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'wavelet_leader_multifractal', recording_dir:recordingDir, params:waveletLeaderParams()})}); const payload=await res.json(); latestWaveletLeaderResult=payload; if(!payload.ok) throw new Error(payload.error || 'Wavelet Leader Multifractal failed.'); const wl=payload.result?.wavelet_leader_multifractal || {}; if(status){status.textContent=`Complete: ${wl.summary?.n_channels || 0} channels · mean Δα=${advFormat(wl.summary?.mean_alpha_width)}`; status.className='status ok';} if(output){output.innerHTML=renderWaveletLeaderPanel(wl);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='wavelet_leader_multifractal'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyWaveletLeaderJson(){ if(!latestWaveletLeaderResult){ alert('No Wavelet Leader Multifractal result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestWaveletLeaderResult,null,2)); }


let latestLyapunovSpectrumResult = null;
function renderLyapunovSpectrumPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `Lyapunov spectrum plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No Lyapunov spectrum plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Lyapunov Spectrum completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It computes local Jacobian/QR Lyapunov exponents from precomputed or generated delay embeddings for all available channels by default, then displays backend-generated black/cyan expert plots so labels, ticks, legends, and colorbars stay visible.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>embeddings: ${advFormat(summary.n_files)}</span><span>success: ${advFormat(summary.successful_files)}</span><span>dims: ${advEscape((summary.embedding_dims || []).join ? summary.embedding_dims.join(',') : '')}</span><span>mean LLE: ${advFormat(summary.mean_LLE)}</span><span>max LLE: ${advFormat(summary.max_LLE)}</span><span>mean KS: ${advFormat(summary.mean_KS_entropy_proxy)}</span><span>hyperchaos: ${advFormat(summary.hyperchaotic_candidates)}</span></div>
    ${plotHtml}
    <h3>Lyapunov expert metrics table</h3>${advRenderTable(data?.rows || [])}
    ${outputHtml}`;
}
function lyapunovSpectrumParams() {
  const params = { mode: $('lyapunovSpectrumMode')?.value || 'ultra' };
  [['embedding_dims','lyapunovSpectrumDims'], ['tau_ms','lyapunovSpectrumTauMs'], ['tau_samples','lyapunovSpectrumTauSamples'], ['k_neighbors','lyapunovSpectrumKNeighbors'], ['theiler','lyapunovSpectrumTheiler'], ['stride','lyapunovSpectrumStride'], ['max_steps_per_file','lyapunovSpectrumMaxSteps'], ['max_points','lyapunovSpectrumMaxPoints'], ['max_channels','lyapunovSpectrumMaxChannels'], ['max_files','lyapunovSpectrumMaxFiles'], ['max_analysis_samples','lyapunovSpectrumMaxSamples'], ['segment_start_sec','lyapunovSpectrumSegmentStart'], ['segment_end_sec','lyapunovSpectrumSegmentEnd'], ['random_seed','lyapunovSpectrumSeed']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function lyapunovSpectrumUseLatest() { const status=$('lyapunovSpectrumStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('lyapunovSpectrumRecordingDir')) $('lyapunovSpectrumRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runLyapunovSpectrumDirect() { showTab('lyapunovSpectrumTab'); const card=$('lyapunovSpectrumDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('lyapunovSpectrumStatus'), output=$('lyapunovSpectrumResult'), recordingDir=$('lyapunovSpectrumRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Lyapunov Spectrum analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running local Jacobian/QR Lyapunov spectra across all available channels. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'lyapunov_spectrum_custom', recording_dir:recordingDir, params:lyapunovSpectrumParams()})}); const payload=await res.json(); latestLyapunovSpectrumResult=payload; if(!payload.ok) throw new Error(payload.error || 'Lyapunov Spectrum failed.'); const lyap=payload.result?.lyapunov_spectrum_custom || {}; if(status){status.textContent=`Complete: ${lyap.summary?.successful_files || 0}/${lyap.summary?.n_files || 0} embeddings · mean LLE=${advFormat(lyap.summary?.mean_LLE)}`; status.className='status ok';} if(output){output.innerHTML=renderLyapunovSpectrumPanel(lyap);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='lyapunov_spectrum_custom'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyLyapunovSpectrumJson(){ if(!latestLyapunovSpectrumResult){ alert('No Lyapunov Spectrum result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestLyapunovSpectrumResult,null,2)); }


let latestArnoldKuramotoResult = null;
function renderArnoldKuramotoPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `Arnold/Kuramoto plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No Arnold/Kuramoto plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Arnold Tongues / Kuramoto completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It estimates EEG-derived natural frequencies, simulates a phase-forced Kuramoto grid, and renders backend-generated drive-locking, frequency-error, synchronization, and contrast plots.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>grid: ${advFormat(summary.a_count)}×${advFormat(summary.b_count)}</span><span>points: ${advFormat(summary.grid_points)}</span><span>mean r: ${advFormat(summary.mean_sync)}</span><span>max r: ${advFormat(summary.max_sync)}</span><span>mean drive lock: ${advFormat(summary.mean_drive_lock)}</span><span>max drive lock: ${advFormat(summary.max_drive_lock)}</span><span>mean freq error: ${advFormat(summary.mean_frequency_error)}</span><span>best a: ${advFormat(summary.best_a)}</span><span>best b: ${advFormat(summary.best_b)}</span><span>plots: ${advFormat(summary.plot_count)}</span></div>
    ${plotHtml}
    <h3>Natural frequencies</h3>${advRenderTable(data?.frequency_rows || [])}
    <h3>Arnold tongue grid rows</h3>${advRenderTable(data?.grid_rows || [])}
    ${outputHtml}`;
}
function arnoldKuramotoParams() {
  const params = { mode: $('arnoldKuramotoMode')?.value || 'ultra' };
  [['K','arnoldKuramotoK'], ['a_min','arnoldKuramotoAMin'], ['a_max','arnoldKuramotoAMax'], ['a_count','arnoldKuramotoACount'], ['b_min','arnoldKuramotoBMin'], ['b_max','arnoldKuramotoBMax'], ['b_count','arnoldKuramotoBCount'], ['t_end','arnoldKuramotoTEnd'], ['n_t_eval','arnoldKuramotoNT'], ['transient_fraction','arnoldKuramotoTransient'], ['freq_min','arnoldKuramotoFreqMin'], ['freq_max','arnoldKuramotoFreqMax'], ['max_channels','arnoldKuramotoMaxChannels'], ['max_analysis_samples','arnoldKuramotoMaxSamples'], ['segment_start_sec','arnoldKuramotoSegmentStart'], ['segment_end_sec','arnoldKuramotoSegmentEnd'], ['random_seed','arnoldKuramotoSeed']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function arnoldKuramotoUseLatest() { const status=$('arnoldKuramotoStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('arnoldKuramotoRecordingDir')) $('arnoldKuramotoRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runArnoldKuramotoDirect() { showTab('arnoldKuramotoTab'); const card=$('arnoldKuramotoDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('arnoldKuramotoStatus'), output=$('arnoldKuramotoResult'), recordingDir=$('arnoldKuramotoRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Arnold/Kuramoto analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running EEG-derived phase-forced Kuramoto Arnold tongue grid with drive-locking and frequency-error diagnostics. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'arnold_tongues_kuramoto', recording_dir:recordingDir, params:arnoldKuramotoParams()})}); const payload=await res.json(); latestArnoldKuramotoResult=payload; if(!payload.ok) throw new Error(payload.error || 'Arnold/Kuramoto failed.'); const arnold=payload.result?.arnold_tongues_kuramoto || {}; if(status){status.textContent=`Complete: ${arnold.summary?.a_count || 0}×${arnold.summary?.b_count || 0} grid · max drive lock=${advFormat(arnold.summary?.max_drive_lock)}`; status.className='status ok';} if(output){output.innerHTML=renderArnoldKuramotoPanel(arnold);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='arnold_tongues_kuramoto'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyArnoldKuramotoJson(){ if(!latestArnoldKuramotoResult){ alert('No Arnold/Kuramoto result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestArnoldKuramotoResult,null,2)); }


let latestCircleMapArnoldResult = null;
function renderCircleMapArnoldPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `Circle map plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No Circle Map Arnold Tongues plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Circle Map Arnold Tongues completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It extracts EEG-derived circular mean phase with the Hilbert transform, then computes fixed-point locking probability over Ω and K. This is fixed-point locking, not a full p:q rotation-number Arnold tongue calculation.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>grid: ${advFormat(summary.K_count)}×${advFormat(summary.omega_count)}</span><span>points: ${advFormat(summary.grid_points)}</span><span>iterations: ${advFormat(summary.iterations)}</span><span>phase samples: ${advFormat(summary.phase_samples_used)}</span><span>locked max: ${advFormat(summary.locked_max)}</span><span>locked mean: ${advFormat(summary.locked_mean)}</span><span>phase mean: ${advFormat(summary.phase_mean_cycles)}</span><span>plots: ${advFormat(summary.plot_count)}</span></div>
    ${plotHtml}
    <h3>Top locked grid cells</h3>${advRenderTable(data?.top_rows || [])}
    <h3>Circle-map grid rows</h3>${advRenderTable(data?.grid_rows || [])}
    <h3>Phase samples</h3>${advRenderTable(data?.phase_rows || [])}
    ${outputHtml}`;
}
function circleMapArnoldParams() {
  const params = { mode: $('circleMapArnoldMode')?.value || 'ultra' };
  [['n_omega','circleMapArnoldNOmega'], ['n_K','circleMapArnoldNK'], ['omega_min','circleMapArnoldOmegaMin'], ['omega_max','circleMapArnoldOmegaMax'], ['K_min','circleMapArnoldKMin'], ['K_max','circleMapArnoldKMax'], ['iterations','circleMapArnoldIterations'], ['tol','circleMapArnoldTol'], ['max_phase_samples','circleMapArnoldMaxPhaseSamples'], ['lock_threshold','circleMapArnoldLockThreshold'], ['max_channels','circleMapArnoldMaxChannels'], ['max_analysis_samples','circleMapArnoldMaxSamples'], ['segment_start_sec','circleMapArnoldSegmentStart'], ['segment_end_sec','circleMapArnoldSegmentEnd']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function circleMapArnoldUseLatest() { const status=$('circleMapArnoldStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('circleMapArnoldRecordingDir')) $('circleMapArnoldRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runCircleMapArnoldDirect() { showTab('circleMapArnoldTab'); const card=$('circleMapArnoldDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('circleMapArnoldStatus'), output=$('circleMapArnoldResult'), recordingDir=$('circleMapArnoldRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Circle Map Arnold Tongues analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running EEG-derived circular mean phase extraction and vectorized circle-map fixed-point-locking grid. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'circle_map_arnold_tongues', recording_dir:recordingDir, params:circleMapArnoldParams()})}); const payload=await res.json(); latestCircleMapArnoldResult=payload; if(!payload.ok) throw new Error(payload.error || 'Circle Map Arnold Tongues failed.'); const circle=payload.result?.circle_map_arnold_tongues || {}; if(status){status.textContent=`Complete: ${circle.summary?.K_count || 0}×${circle.summary?.omega_count || 0} grid · max locked=${advFormat(circle.summary?.locked_max)}`; status.className='status ok';} if(output){output.innerHTML=renderCircleMapArnoldPanel(circle);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='circle_map_arnold_tongues'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyCircleMapArnoldJson(){ if(!latestCircleMapArnoldResult){ alert('No Circle Map Arnold Tongues result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestCircleMapArnoldResult,null,2)); }


let latestCircleMapDensityResult = null;
function renderCircleMapDensityPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `Circle density plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No Circle Map Converged Density plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">Circle Map Converged Density completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It bandpass-filters EEG before Hilbert phase extraction when enabled, uses each channel's phase distribution as initial conditions, and estimates final-state probability density across K at fixed Ω.</div>
    <div class="summary-pill-row"><span>channels: ${advFormat(summary.n_channels)}</span><span>K points: ${advFormat(summary.K_count)}</span><span>bins: ${advFormat(summary.n_bins)}</span><span>iterations: ${advFormat(summary.iterations)}</span><span>Ω: ${advFormat(summary.Omega)}</span><span>mean conc.: ${advFormat(summary.mean_normalized_concentration)}</span><span>max conc.: ${advFormat(summary.max_normalized_concentration)}</span><span>mean R: ${advFormat(summary.mean_resultant_length)}</span><span>plots: ${advFormat(summary.plot_count)}</span></div>
    ${plotHtml}
    <h3>Top concentration channels</h3>${advRenderTable(data?.top_channels || [])}
    <h3>Channel summary</h3>${advRenderTable(data?.channel_rows || [])}
    <h3>Aggregate by K</h3>${advRenderTable(data?.aggregate_rows || [])}
    <h3>Final-state density grid</h3>${advRenderTable(data?.density_rows || [])}
    ${outputHtml}`;
}
function circleMapDensityParams() {
  const params = { mode: $('circleMapDensityMode')?.value || 'ultra', use_bandpass: ($('circleMapDensityUseBandpass')?.value || 'true') === 'true' };
  [['Omega','circleMapDensityOmega'], ['n_K','circleMapDensityNK'], ['K_min','circleMapDensityKMin'], ['K_max','circleMapDensityKMax'], ['iterations','circleMapDensityIterations'], ['n_bins','circleMapDensityNBins'], ['max_samples_per_channel','circleMapDensityMaxPhaseSamples'], ['phase_min_hz','circleMapDensityPhaseMin'], ['phase_max_hz','circleMapDensityPhaseMax'], ['filter_order','circleMapDensityFilterOrder'], ['max_channels','circleMapDensityMaxChannels'], ['max_analysis_samples','circleMapDensityMaxSamples'], ['segment_start_sec','circleMapDensitySegmentStart'], ['segment_end_sec','circleMapDensitySegmentEnd'], ['random_seed','circleMapDensitySeed']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function circleMapDensityUseLatest() { const status=$('circleMapDensityStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('circleMapDensityRecordingDir')) $('circleMapDensityRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runCircleMapDensityDirect() { showTab('circleMapDensityTab'); const card=$('circleMapDensityDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('circleMapDensityStatus'), output=$('circleMapDensityResult'), recordingDir=$('circleMapDensityRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running Circle Map Converged Density analysis...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Running EEG Hilbert-phase extraction and final-state probability-density circle-map analysis. Backend-generated black/cyan plots will appear here when complete.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'circle_map_converged_density', recording_dir:recordingDir, params:circleMapDensityParams()})}); const payload=await res.json(); latestCircleMapDensityResult=payload; if(!payload.ok) throw new Error(payload.error || 'Circle Map Converged Density failed.'); const density=payload.result?.circle_map_converged_density || {}; if(status){status.textContent=`Complete: ${density.summary?.K_count || 0} K values · max concentration=${advFormat(density.summary?.max_normalized_concentration)}`; status.className='status ok';} if(output){output.innerHTML=renderCircleMapDensityPanel(density);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='circle_map_converged_density'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyCircleMapDensityJson(){ if(!latestCircleMapDensityResult){ alert('No Circle Map Converged Density result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestCircleMapDensityResult,null,2)); }

let latestMfdfaViewerResult = null;
function renderMfdfaViewerPanel(data) {
  const summary = data?.summary || {}; const outputs = data?.outputs || {}; const plots = Array.isArray(data?.plot_paths) ? data.plot_paths : [];
  const plotHtml = plots.length ? `<div class="grid two mfdfa-plot-grid">${plots.map((p, idx) => {
    const url = mfdfaImageUrl(p); const title = p.title || `MFDFA viewer plot ${idx+1}`;
    return `<div class="card advanced-plot-card image-plot-card"><h3>${advEscape(title)}</h3><a href="${advEscape(url)}" target="_blank"><img class="analysis-image-plot" src="${advEscape(url)}" alt="${advEscape(title)}" loading="lazy" /></a></div>`;
  }).join('')}</div>` : '<div class="method-note">No MFDFA viewer plot paths returned.</div>';
  const outputHtml = Object.keys(outputs).length ? `<h3>Saved files</h3>${advRenderTable(Object.entries(outputs).map(([name, path]) => ({ name, path })))}` : '';
  return `<div class="method-note">MFDFA Plot Viewer completed in <strong>${advEscape(summary.mode || 'ultra')}</strong> mode. It scanned Manual Expert MFDFA per-channel PNGs and summary CSVs, then generated contact sheets, ranked galleries, individual pages, inventory, rankings, metric heatmap, and width/asymmetry plots in the same black/cyan style.</div>
    <div class="summary-pill-row"><span>images found: ${advFormat(summary.images_found)}</span><span>displayed: ${advFormat(summary.images_displayed)}</span><span>sort: ${advEscape(summary.sort_mode || '')}</span><span>viewer plots: ${advFormat(summary.viewer_plot_count)}</span><span>ran MFDFA first: ${summary.generated_mfdfa_first ? 'yes' : 'no'}</span></div>
    ${plotHtml}
    <h3>Top ranked channels</h3>${advRenderTable(data?.top_rows || [])}
    <h3>Display manifest</h3>${advRenderTable(data?.display_rows || [])}
    ${outputHtml}`;
}
function mfdfaViewerParams() {
  const params = { mode: $('mfdfaViewerMode')?.value || 'ultra', run_mfdfa_if_missing: ($('mfdfaViewerRunMissing')?.value || 'true') === 'true', sort_mode: $('mfdfaViewerSortMode')?.value || 'multifractal_complexity_score' };
  [['max_images_to_show','mfdfaViewerMaxImages'], ['plots_per_page','mfdfaViewerPlotsPerPage'], ['max_channels','mfdfaViewerMaxChannels'], ['max_analysis_samples','mfdfaViewerMaxSamples']].forEach(([name,id]) => { const raw=$(id)?.value?.trim(); if(raw!==undefined && raw!==''){ const n=Number(raw); params[name]=Number.isFinite(n)?n:raw; } });
  return params;
}
async function mfdfaViewerUseLatest() { const status=$('mfdfaViewerStatus'); if(status){status.textContent='Looking for latest converted recording...'; status.className='status';} try{ const res=await fetch('/api/advanced-methods/latest-recording?t='+Date.now(),{cache:'no-store'}); const data=await res.json(); if(!data.ok) throw new Error(data.error || 'No latest recording found.'); if($('mfdfaViewerRecordingDir')) $('mfdfaViewerRecordingDir').value=data.recording_dir; if(status){status.textContent=`Using latest converted recording: ${data.recording_dir}`; status.className='status ok';} } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} } }
async function runMfdfaViewerDirect() { showTab('mfdfaViewerTab'); const card=$('mfdfaViewerDirectCard'); if(card) setTimeout(()=>card.scrollIntoView({behavior:'smooth',block:'start'}),30); const status=$('mfdfaViewerStatus'), output=$('mfdfaViewerResult'), recordingDir=$('mfdfaViewerRecordingDir')?.value?.trim() || 'latest'; if(status){status.textContent='Running MFDFA Plot Viewer...'; status.className='status';} if(output) output.innerHTML='<div class="method-note">Scanning MFDFA per-channel images and summary CSVs. If missing, Manual Expert MFDFA may run first with per-channel plots enabled.</div>'; try{ const res=await fetch('/api/advanced-methods/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({method_id:'mfdfa_plot_viewer', recording_dir:recordingDir, params:mfdfaViewerParams()})}); const payload=await res.json(); latestMfdfaViewerResult=payload; if(!payload.ok) throw new Error(payload.error || 'MFDFA Plot Viewer failed.'); const viewer=payload.result?.mfdfa_plot_viewer || {}; if(status){status.textContent=`Complete: ${viewer.summary?.images_displayed || 0} displayed images · ${viewer.summary?.viewer_plot_count || 0} viewer plots`; status.className='status ok';} if(output){output.innerHTML=renderMfdfaViewerPanel(viewer);} if($('advancedMethodSelect')) $('advancedMethodSelect').value='mfdfa_plot_viewer'; } catch(err){ if(status){status.textContent=err.message; status.className='status failed';} if(output) output.innerHTML=`<div class="method-note">${advEscape(err.message)}</div>`; } }
function copyMfdfaViewerJson(){ if(!latestMfdfaViewerResult){ alert('No MFDFA Viewer result JSON to copy yet.'); return; } navigator.clipboard.writeText(JSON.stringify(latestMfdfaViewerResult,null,2)); }


function toggleMoreOptions(force) {
  const optional = $('optionalTabs');
  if (!optional) return;
  const show = force === undefined ? optional.hidden : !!force;
  optional.hidden = !show;
  document.body.classList.toggle('more-options-open', show);
  const btn = $('moreOptionsBtn') || $('homeShowMoreBtn');
  if (btn) btn.textContent = show ? 'Hide extra tabs' : 'More tools';
}

function selectAdvancedMethod(methodId) {
  showTab('advancedMethodsTab');
  const sel = $('advancedMethodSelect');
  if (sel && advancedMethods.some(m => m.id === methodId)) { sel.value = methodId; renderAdvancedMethodParams(); }
}


$('saturationUseLatestBtn')?.addEventListener('click', saturationUseLatest);
$('saturationRunBtn')?.addEventListener('click', runSaturationDirect);
$('saturationCopyJsonBtn')?.addEventListener('click', copySaturationJson);
$('homeSaturationBtn')?.addEventListener('click', () => { showTab('saturationTab'); const card = $('saturationDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('katzUseLatestBtn')?.addEventListener('click', katzUseLatest);
$('katzRunBtn')?.addEventListener('click', runKatzDirect);
$('katzCopyJsonBtn')?.addEventListener('click', copyKatzJson);

$('mfdfaUseLatestBtn')?.addEventListener('click', mfdfaUseLatest);
$('mfdfaRunBtn')?.addEventListener('click', runMfdfaDirect);
$('mfdfaCopyJsonBtn')?.addEventListener('click', copyMfdfaJson);
$('homeMfdfaBtn')?.addEventListener('click', () => { showTab('mfdfaTab'); const card = $('mfdfaDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpMfdfaBtn')?.addEventListener('click', () => { showTab('mfdfaTab'); const card = $('mfdfaDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('mfdfaSpectrumUseLatestBtn')?.addEventListener('click', mfdfaSpectrumUseLatest);
$('mfdfaSpectrumRunBtn')?.addEventListener('click', runMfdfaSpectrumDirect);
$('mfdfaSpectrumCopyJsonBtn')?.addEventListener('click', copyMfdfaSpectrumJson);
$('homeMfdfaSpectrumBtn')?.addEventListener('click', () => { showTab('mfdfaSpectrumTab'); const card = $('mfdfaSpectrumDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpMfdfaSpectrumBtn')?.addEventListener('click', () => { showTab('mfdfaSpectrumTab'); const card = $('mfdfaSpectrumDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('mfdfaShuffleUseLatestBtn')?.addEventListener('click', mfdfaShuffleUseLatest);
$('mfdfaShuffleRunBtn')?.addEventListener('click', runMfdfaShuffleDirect);
$('mfdfaShuffleCopyJsonBtn')?.addEventListener('click', copyMfdfaShuffleJson);
$('homeMfdfaShuffleBtn')?.addEventListener('click', () => { showTab('mfdfaShuffleTab'); const card = $('mfdfaShuffleDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpMfdfaShuffleBtn')?.addEventListener('click', () => { showTab('mfdfaShuffleTab'); const card = $('mfdfaShuffleDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('mfdfaIaaftUseLatestBtn')?.addEventListener('click', mfdfaIaaftUseLatest);
$('mfdfaIaaftRunBtn')?.addEventListener('click', runMfdfaIaaftDirect);
$('mfdfaIaaftCopyJsonBtn')?.addEventListener('click', copyMfdfaIaaftJson);
$('homeMfdfaIaaftBtn')?.addEventListener('click', () => { showTab('mfdfaIaaftTab'); const card = $('mfdfaIaaftDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpMfdfaIaaftBtn')?.addEventListener('click', () => { showTab('mfdfaIaaftTab'); const card = $('mfdfaIaaftDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('waveletLeaderUseLatestBtn')?.addEventListener('click', waveletLeaderUseLatest);
$('waveletLeaderRunBtn')?.addEventListener('click', runWaveletLeaderDirect);
$('waveletLeaderCopyJsonBtn')?.addEventListener('click', copyWaveletLeaderJson);
$('homeWaveletLeaderBtn')?.addEventListener('click', () => { showTab('waveletLeaderTab'); const card = $('waveletLeaderDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpWaveletLeaderBtn')?.addEventListener('click', () => { showTab('waveletLeaderTab'); const card = $('waveletLeaderDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('lyapunovSpectrumUseLatestBtn')?.addEventListener('click', lyapunovSpectrumUseLatest);
$('lyapunovSpectrumRunBtn')?.addEventListener('click', runLyapunovSpectrumDirect);
$('lyapunovSpectrumCopyJsonBtn')?.addEventListener('click', copyLyapunovSpectrumJson);
$('homeLyapunovSpectrumBtn')?.addEventListener('click', () => { showTab('lyapunovSpectrumTab'); const card = $('lyapunovSpectrumDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpLyapunovSpectrumBtn')?.addEventListener('click', () => { showTab('lyapunovSpectrumTab'); const card = $('lyapunovSpectrumDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('arnoldKuramotoUseLatestBtn')?.addEventListener('click', arnoldKuramotoUseLatest);
$('arnoldKuramotoRunBtn')?.addEventListener('click', runArnoldKuramotoDirect);
$('arnoldKuramotoCopyJsonBtn')?.addEventListener('click', copyArnoldKuramotoJson);
$('homeArnoldKuramotoBtn')?.addEventListener('click', () => { showTab('arnoldKuramotoTab'); const card = $('arnoldKuramotoDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpArnoldKuramotoBtn')?.addEventListener('click', () => { showTab('arnoldKuramotoTab'); const card = $('arnoldKuramotoDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('circleMapArnoldUseLatestBtn')?.addEventListener('click', circleMapArnoldUseLatest);
$('circleMapArnoldRunBtn')?.addEventListener('click', runCircleMapArnoldDirect);
$('circleMapArnoldCopyJsonBtn')?.addEventListener('click', copyCircleMapArnoldJson);
$('homeCircleMapArnoldBtn')?.addEventListener('click', () => { showTab('circleMapArnoldTab'); const card = $('circleMapArnoldDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpCircleMapArnoldBtn')?.addEventListener('click', () => { showTab('circleMapArnoldTab'); const card = $('circleMapArnoldDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('circleMapDensityUseLatestBtn')?.addEventListener('click', circleMapDensityUseLatest);
$('circleMapDensityRunBtn')?.addEventListener('click', runCircleMapDensityDirect);
$('circleMapDensityCopyJsonBtn')?.addEventListener('click', copyCircleMapDensityJson);
$('homeCircleMapDensityBtn')?.addEventListener('click', () => { showTab('circleMapDensityTab'); const card = $('circleMapDensityDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpCircleMapDensityBtn')?.addEventListener('click', () => { showTab('circleMapDensityTab'); const card = $('circleMapDensityDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('mfdfaViewerUseLatestBtn')?.addEventListener('click', mfdfaViewerUseLatest);
$('mfdfaViewerRunBtn')?.addEventListener('click', runMfdfaViewerDirect);
$('mfdfaViewerCopyJsonBtn')?.addEventListener('click', copyMfdfaViewerJson);
$('homeMfdfaViewerBtn')?.addEventListener('click', () => { showTab('mfdfaViewerTab'); const card = $('mfdfaViewerDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpMfdfaViewerBtn')?.addEventListener('click', () => { showTab('mfdfaViewerTab'); const card = $('mfdfaViewerDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('waveletUseLatestBtn')?.addEventListener('click', waveletUseLatest);
$('waveletRunBtn')?.addEventListener('click', runWaveletDirect);
$('waveletCopyJsonBtn')?.addEventListener('click', copyWaveletJson);
$('homeWaveletBtn')?.addEventListener('click', () => { showTab('waveletTab'); const card = $('waveletDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpWaveletBtn')?.addEventListener('click', () => { showTab('waveletTab'); const card = $('waveletDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('homeKatzBtn')?.addEventListener('click', () => { showTab('katzTab'); const card = $('katzDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpKatzBtn')?.addEventListener('click', () => { showTab('katzTab'); const card = $('katzDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('jumpSaturationBtn')?.addEventListener('click', () => { showTab('saturationTab'); const card = $('saturationDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });

$('embeddedFdUseLatestBtn')?.addEventListener('click', embeddedFdUseLatest);
$('embeddedFdRunBtn')?.addEventListener('click', runEmbeddedFdDirect);
$('embeddedFdCopyJsonBtn')?.addEventListener('click', copyEmbeddedFdJson);
$('homeUploadBtn')?.addEventListener('click', () => showTab('importTab'));
$('homeCompareBtn')?.addEventListener('click', () => showTab('compareTab'));
$('homeHiguchiBtn')?.addEventListener('click', () => { showTab('higuchiTab'); const card = $('higuchiDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('homeEmbeddedFdBtn')?.addEventListener('click', () => { showTab('embeddedFdTab'); const card = $('embeddedFdDirectCard'); if (card) setTimeout(() => card.scrollIntoView({behavior:'smooth', block:'start'}), 40); });
$('moreOptionsBtn')?.addEventListener('click', () => toggleMoreOptions());
$('homeShowMoreBtn')?.addEventListener('click', () => toggleMoreOptions(true));

// v0.11.17: default landing is the simple Home dashboard.
if ($('homeTab')) {
  if (location.hash === '#higuchiDirectCard') { showTab('higuchiTab'); setTimeout(() => $('higuchiDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#embeddedFdDirectCard') { showTab('embeddedFdTab'); setTimeout(() => $('embeddedFdDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#saturationDirectCard') { showTab('saturationTab'); setTimeout(() => $('saturationDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#katzDirectCard') { showTab('katzTab'); setTimeout(() => $('katzDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#mfdfaDirectCard') { showTab('mfdfaTab'); setTimeout(() => $('mfdfaDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#mfdfaSpectrumDirectCard') { showTab('mfdfaSpectrumTab'); setTimeout(() => $('mfdfaSpectrumDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#mfdfaShuffleDirectCard') { showTab('mfdfaShuffleTab'); setTimeout(() => $('mfdfaShuffleDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#waveletLeaderDirectCard') { showTab('waveletLeaderTab'); setTimeout(() => $('waveletLeaderDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#lyapunovSpectrumDirectCard') { showTab('lyapunovSpectrumTab'); setTimeout(() => $('lyapunovSpectrumDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#arnoldKuramotoDirectCard') { showTab('arnoldKuramotoTab'); setTimeout(() => $('arnoldKuramotoDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#circleMapArnoldDirectCard') { showTab('circleMapArnoldTab'); setTimeout(() => $('circleMapArnoldDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#circleMapDensityDirectCard') { showTab('circleMapDensityTab'); setTimeout(() => $('circleMapDensityDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#mfdfaViewerDirectCard') { showTab('mfdfaViewerTab'); setTimeout(() => $('mfdfaViewerDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else if (location.hash === '#waveletDirectCard') { showTab('waveletTab'); setTimeout(() => $('waveletDirectCard')?.scrollIntoView({behavior:'smooth', block:'start'}), 60); }
  else { showTab('homeTab'); }
}
