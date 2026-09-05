const token = document.querySelector('#token');
const rows = document.querySelector('#sessions');
const job = document.querySelector('#job');
const dashboardStatus = document.querySelector('#dashboard-status');
const diagnostics = document.querySelector('#diagnostics');
const objectDiagnostics = document.querySelector('#object-diagnostics');

token.value = localStorage.getItem('iphone3d-token') || '';

class DashboardError extends Error {
  constructor(kind, message, status = null) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

function headers() {
  const value = token.value.trim();
  return {
    ...(value ? { Authorization: `Bearer ${value}` } : {}),
    'X-Protocol-Version': '2',
  };
}

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: { ...headers(), ...(options.headers || {}) },
    });
  } catch {
    throw new DashboardError('network', 'Cannot reach receiver');
  }
  if (!response.ok) {
    const detail = (await response.text()).slice(0, 512);
    if (response.status === 401) throw new DashboardError('auth', 'Authentication required', 401);
    if (response.status >= 500) throw new DashboardError('server', 'Server error', response.status);
    throw new DashboardError('request', detail || `Request failed (HTTP ${response.status})`, response.status);
  }
  return response.json();
}

function bytes(value) {
  return new Intl.NumberFormat(undefined, {
    style: 'unit', unit: 'megabyte', unitDisplay: 'short', maximumFractionDigits: 1,
  }).format(value / 1048576);
}

function showSessionMessage(message) {
  rows.replaceChildren();
  const row = document.createElement('tr');
  const cell = document.createElement('td');
  cell.colSpan = 6;
  cell.textContent = message;
  row.append(cell);
  rows.append(row);
}

function sessionErrorMessage(error) {
  if (error instanceof DashboardError) return error.message;
  return 'Unable to load verified sessions';
}

async function loadSessions() {
  dashboardStatus.textContent = 'Loading sessions…';
  try {
    const data = await api('/api/v2/sessions');
    rows.replaceChildren();
    if (!data.sessions.length) {
      dashboardStatus.textContent = 'No verified sessions';
      showSessionMessage('No verified sessions');
      return;
    }
    dashboardStatus.textContent = `${data.sessions.length} verified session${data.sessions.length === 1 ? '' : 's'}`;
    for (const item of data.sessions) {
      const row = document.createElement('tr');
      for (const value of [item.session_id, item.scan_mode === 'object' ? 'Object' : 'Scene', item.frame_count, bytes(item.total_bytes), item.created_at || '-']) {
        const cell = document.createElement('td');
        cell.textContent = value;
        row.append(cell);
      }
      const actions = document.createElement('td');
      for (const [label, action] of [
        ...(item.scan_mode === 'object' ? [
          ['Build Object Point Cloud', () => start(item.session_id, 'object_pointcloud')],
          ['View Object Cloud', () => view(item.session_id, true)],
          ['Mask diagnostics', () => void showObjectDiagnostics(item.session_id)],
        ] : []),
        ['Build point cloud', () => start(item.session_id, 'pointcloud')],
        ['Build mesh', () => start(item.session_id, 'mesh')],
        ['View', () => view(item.session_id, false)],
      ]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.onclick = action;
        actions.append(button);
      }
      row.append(actions);
      rows.append(row);
    }
  } catch (error) {
    const message = sessionErrorMessage(error);
    dashboardStatus.textContent = message;
    showSessionMessage(message);
  }
}

async function loadDiagnostics() {
  try {
    const value = await api('/api/v2/diagnostics');
    diagnostics.textContent = value.torch_cuda_available ? `CUDA: ${value.nvidia_gpu}` : 'CPU processing';
  } catch (error) {
    diagnostics.textContent = sessionErrorMessage(error);
  }
}

async function loadDashboard() {
  localStorage.setItem('iphone3d-token', token.value.trim());
  await loadSessions();
  void loadDiagnostics();
}

async function start(id, kind) {
  try {
    await api(`/api/v2/sessions/${id}/jobs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind, device: 'auto' }),
    });
    void poll(id, kind);
  } catch (error) {
    job.textContent = sessionErrorMessage(error);
  }
}

async function poll(id, kind) {
  try {
    const state = await api(`/api/v2/sessions/${id}/job`);
    job.textContent = `${state.state.toUpperCase()} ${state.progress || 0}% — ${state.message || ''}`;
    if (!['done', 'failed'].includes(state.state)) setTimeout(() => void poll(id), 1000);
    else if (state.state === 'done') {
      if (kind === 'object_pointcloud') void showObjectDiagnostics(id);
      void loadSessions();
    }
  } catch (error) {
    job.textContent = sessionErrorMessage(error);
  }
}

let viewerModule;
async function view(id, objectCloud) {
  try {
    viewerModule ||= await import('./viewer.js?v=2');
    await viewerModule.viewSession(id, { headers, job, artifactNames: objectCloud ? ['object/object_clean.ply'] : undefined });
  } catch {
    job.textContent = '3D viewer unavailable';
  }
}

function clearObjectDiagnostics() {
  objectDiagnostics.replaceChildren();
}

async function showObjectDiagnostics(id) {
  clearObjectDiagnostics();
  try {
    const report = await api(`/api/v2/sessions/${id}/artifacts/object/processing.json`);
    const summary = document.createElement('p');
    summary.textContent = `Object diagnostics — Frames: ${report.processed_frames}/${report.input_frames}; Foreground points: ${report.foreground_points}; Clean points: ${report.clean_points}; Processing: ${report.timing_seconds.total.toFixed(2)} s`;
    objectDiagnostics.append(summary);
    for (const mask of report.outputs.diagnostic_masks || []) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = `Preview mask ${mask}`;
      button.onclick = async () => {
        try {
          viewerModule ||= await import('./viewer.js?v=2');
          await viewerModule.showMask(id, mask, { headers, job });
        } catch {
          job.textContent = '3D viewer unavailable';
        }
      };
      objectDiagnostics.append(button);
    }
    for (const warning of report.warnings || []) {
      const warningText = document.createElement('p');
      warningText.textContent = `Warning: ${warning}`;
      objectDiagnostics.append(warningText);
    }
  } catch (error) {
    objectDiagnostics.textContent = sessionErrorMessage(error);
  }
}

document.querySelector('#connect').onclick = () => void loadDashboard();
void loadDashboard();
