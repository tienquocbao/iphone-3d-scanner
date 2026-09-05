const token = document.querySelector('#token');
const rows = document.querySelector('#sessions');
const job = document.querySelector('#job');
const dashboardStatus = document.querySelector('#dashboard-status');
const diagnostics = document.querySelector('#diagnostics');
const objectDiagnostics = document.querySelector('#object-diagnostics');

token.value = localStorage.getItem('iphone3d-token') || '';
let nksrCapability = null;

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
      for (const [label, action, disabled = false, title = ''] of [
        ...(item.scan_mode === 'object' ? [
          ['Build Object Point Cloud', () => start(item.session_id, 'object_pointcloud')],
          ['View Object Cloud', () => view(item.session_id, true)],
          ['Mask diagnostics', () => void showObjectDiagnostics(item.session_id)],
          ...(item.pass_count >= 2 ? [['Build Registered Object Cloud', () => start(item.session_id, 'registered_object_pointcloud')], ['View Registered', () => view(item.session_id, 'registered')]] : []),
          ['Build Object Mesh (TSDF)', () => start(item.session_id, 'object_tsdf')],
          ...(item.object_tsdf_state !== 'missing' ? [['View TSDF Mesh', () => view(item.session_id, 'tsdf')], ['TSDF diagnostics', () => void showTSDFDiagnostics(item.session_id, item.object_tsdf_state)]] : []),
          ['Build Object Mesh (Poisson)', () => start(item.session_id, 'object_poisson')],
          ...(item.object_poisson_state !== 'missing' ? [['View Poisson Mesh', () => view(item.session_id, 'poisson')], ['Poisson diagnostics', () => void showSurfaceDiagnostics(item.session_id, 'poisson', item.object_poisson_state)]] : []),
          ['Build Object Mesh (BPA)', () => start(item.session_id, 'object_bpa')],
          ...(item.object_bpa_state !== 'missing' ? [['View BPA Mesh', () => view(item.session_id, 'bpa')], ['BPA diagnostics', () => void showSurfaceDiagnostics(item.session_id, 'bpa', item.object_bpa_state)]] : []),
          ...((item.object_tsdf_state !== 'missing' || item.object_poisson_state !== 'missing' || item.object_bpa_state !== 'missing') ? [['Compare object backends', () => void showSurfaceComparison(item.session_id)]] : []),
          ...(nksrCapability?.available && item.object_reconstruction_ready
            ? [['Build Object Mesh (NKSR)', () => start(item.session_id, 'object_nksr')]]
            : [[nksrCapability?.available ? 'NKSR registration required' : nksrCapability ? 'NKSR unavailable' : 'NKSR capability checking…', () => {}, true, nksrCapability?.available ? 'Build the required object cloud/registration first' : nksrCapability?.reason || 'Checking optional backend']]),
          ...(item.object_nksr_state !== 'missing' ? [['View NKSR Mesh', () => view(item.session_id, 'nksr')], ['NKSR diagnostics', () => void showNKSRDiagnostics(item.session_id, item.object_nksr_state)]] : []),
        ] : []),
        ['Build point cloud', () => start(item.session_id, 'pointcloud')],
        ['Build mesh', () => start(item.session_id, 'mesh')],
        ['View', () => view(item.session_id, false)],
      ]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = label;
        button.onclick = action;
        button.disabled = disabled;
        button.title = title;
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
    nksrCapability = value.nksr;
    const base = value.torch_cuda_available ? `CUDA: ${value.nvidia_gpu}` : 'CPU processing';
    diagnostics.textContent = `${base} · NKSR: ${value.nksr.available ? `Available (${value.nksr.cuda_available ? 'CUDA' : 'CPU'})` : 'Unavailable'}`;
    void loadSessions();
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
    if (!['done', 'failed'].includes(state.state)) setTimeout(() => void poll(id, kind), 1000);
    else if (state.state === 'done') {
      if (kind === 'object_pointcloud') void showObjectDiagnostics(id);
      if (kind === 'object_tsdf') void showTSDFDiagnostics(id, 'current');
      if (kind === 'object_nksr') void showNKSRDiagnostics(id, 'current');
      if (kind === 'object_poisson') void showSurfaceDiagnostics(id, 'poisson', 'current');
      if (kind === 'object_bpa') void showSurfaceDiagnostics(id, 'bpa', 'current');
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
    const artifactNames = objectCloud === 'nksr'
      ? ['object/reconstruction/nksr/object_nksr_clean.ply']
      : objectCloud === 'poisson'
        ? ['object/reconstruction/poisson/object_poisson_clean.ply']
        : objectCloud === 'bpa'
          ? ['object/reconstruction/bpa/object_bpa_clean.ply']
      : objectCloud === 'tsdf'
        ? ['object/reconstruction/tsdf/object_tsdf_clean.ply']
      : objectCloud === 'registered'
        ? ['object/object_registered_clean.ply']
        : objectCloud
          ? ['object/object_clean.ply']
          : undefined;
    await viewerModule.viewSession(id, { headers, job, artifactNames });
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

async function showTSDFDiagnostics(id, state) {
  clearObjectDiagnostics();
  try {
    const report = await api(`/api/v2/sessions/${id}/artifacts/object/reconstruction/tsdf/reconstruction.json`);
    const mesh = report.mesh.clean;
    const summary = document.createElement('p');
    summary.textContent = `Object mesh — Backend: TSDF; Integrated: ${report.integrated_frames}/${report.input_frames}; Rejected: ${report.rejected_frames.length}; Vertices: ${mesh.vertices}; Triangles: ${mesh.triangles}; Processing: ${report.timings.total_seconds.toFixed(2)} s`;
    objectDiagnostics.append(summary);
    if (state === 'stale') {
      const warning = document.createElement('p');
      warning.textContent = 'STALE — object registration changed; rebuild the TSDF mesh.';
      objectDiagnostics.append(warning);
    }
    for (const warningText of report.warnings || []) {
      const warning = document.createElement('p');
      warning.textContent = `Warning: ${warningText}`;
      objectDiagnostics.append(warning);
    }
  } catch (error) {
    objectDiagnostics.textContent = sessionErrorMessage(error);
  }
}

async function showNKSRDiagnostics(id, state) {
  clearObjectDiagnostics();
  try {
    const report = await api(`/api/v2/sessions/${id}/artifacts/object/reconstruction/nksr/reconstruction.json`);
    const mesh = report.mesh;
    const summary = document.createElement('p');
    summary.textContent = `Object mesh — Backend: NKSR; Device: ${report.device}; Input points: ${report.input_points}; Vertices: ${mesh.clean_vertices}; Triangles: ${mesh.clean_triangles}; Processing: ${report.processing_seconds.toFixed(2)} s`;
    objectDiagnostics.append(summary);
    if (state === 'stale') {
      const warning = document.createElement('p');
      warning.textContent = 'STALE — object registration changed; rebuild the NKSR mesh.';
      objectDiagnostics.append(warning);
    }
    const consistency = report.observed_point_consistency || {};
    for (const [label, value] of [['NKSR', consistency.observed_to_nksr], ['TSDF', consistency.observed_to_tsdf]]) {
      if (!value) continue;
      const metric = document.createElement('p');
      metric.textContent = `${label} observed-point distance — median: ${value.median_m.toFixed(4)} m; p95: ${value.p95_m.toFixed(4)} m`;
      objectDiagnostics.append(metric);
    }
  } catch (error) {
    objectDiagnostics.textContent = sessionErrorMessage(error);
  }
}

async function showSurfaceDiagnostics(id, backend, state) {
  clearObjectDiagnostics();
  try {
    const report = await api(`/api/v2/sessions/${id}/artifacts/object/reconstruction/${backend}/reconstruction.json`);
    const mesh = report.mesh.clean;
    const summary = document.createElement('p');
    summary.textContent = `Object mesh — Backend: ${backend.toUpperCase()}; Input points: ${report.input_points}; Vertices: ${mesh.vertices}; Triangles: ${mesh.triangles}; Processing: ${report.processing_seconds.toFixed(2)} s`;
    objectDiagnostics.append(summary);
    if (backend === 'poisson') {
      const density = report.density_filter;
      const details = document.createElement('p');
      details.textContent = `Poisson density trim — threshold: ${density.threshold.toFixed(3)}; removed vertices: ${density.removed_vertices}`;
      objectDiagnostics.append(details);
    } else {
      const details = document.createElement('p');
      details.textContent = `BPA radii: ${(report.bpa.ball_radii_m || []).map((value) => value.toFixed(4)).join(', ')} m`;
      objectDiagnostics.append(details);
    }
    const normal = report.normal_diagnostics;
    const normalDetails = document.createElement('p');
    normalDetails.textContent = `Normals: ${normal.orientation_method}; sensor flips: ${normal.flipped_by_sensor_count}; radius: ${normal.normal_radius_m.toFixed(4)} m`;
    objectDiagnostics.append(normalDetails);
    if (state === 'stale') {
      const warning = document.createElement('p');
      warning.textContent = `STALE — object registration changed; rebuild the ${backend.toUpperCase()} mesh.`;
      objectDiagnostics.append(warning);
    }
    for (const warningText of report.warnings || []) {
      const warning = document.createElement('p');
      warning.textContent = `Warning: ${warningText}`;
      objectDiagnostics.append(warning);
    }
  } catch (error) {
    objectDiagnostics.textContent = sessionErrorMessage(error);
  }
}

async function showSurfaceComparison(id) {
  clearObjectDiagnostics();
  try {
    const report = await api(`/api/v2/sessions/${id}/artifacts/object/reconstruction/comparison.json`);
    const note = document.createElement('p');
    note.textContent = report.note;
    objectDiagnostics.append(note);
    for (const [backend, value] of Object.entries(report.backends || {})) {
      const consistency = value.observed_point_consistency;
      const line = document.createElement('p');
      const runtime = value.runtime_seconds == null ? '-' : `${Number(value.runtime_seconds).toFixed(2)} s`;
      const observed = consistency ? `; observed median/p95: ${consistency.median_m.toFixed(4)}/${consistency.p95_m.toFixed(4)} m` : '';
      line.textContent = `${backend.toUpperCase()} — ${value.vertices} vertices, ${value.triangles} triangles, ${runtime}${observed}`;
      objectDiagnostics.append(line);
    }
  } catch (error) {
    objectDiagnostics.textContent = sessionErrorMessage(error);
  }
}

document.querySelector('#connect').onclick = () => void loadDashboard();
void loadDashboard();
