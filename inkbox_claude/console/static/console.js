const API = '/console/api';

const state = {
  overview: null,
  people: [],
  roles: [],
  scopes: [],
  requests: [],
  tasks: [],
  health: null,
  settings: [],
  settingStatus: {},
  selectedPerson: null,
  selectedWaiting: 0,
  requestFilter: 'all',
  search: '',
  eventSource: null,
  reconnectTimer: null,
  reconnectAttempt: 0,
  noticeTimer: null,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { 'Accept': 'application/json', ...(options.body ? { 'Content-Type': 'application/json' } : {}), ...options.headers },
  });
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof payload === 'object'
      ? payload.error || payload.detail || payload.message || `Request failed (${response.status})`
      : payload || `Request failed (${response.status})`;
    throw new ApiError(message, response.status, payload);
  }
  return payload;
}

const post = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body) });

function escapeHTML(value = '') {
  return String(value).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}

function formatTime(value) {
  if (!value) return 'unknown';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }).format(date);
}

function formatDuration(seconds) {
  if (!Number.isFinite(Number(seconds))) return 'unknown';
  let remaining = Math.max(0, Number(seconds));
  const days = Math.floor(remaining / 86400); remaining %= 86400;
  const hours = Math.floor(remaining / 3600); remaining %= 3600;
  const minutes = Math.floor(remaining / 60);
  return [days && `${days}d`, (hours || days) && `${hours}h`, `${minutes}m`].filter(Boolean).join(' ');
}

function showNotice(message, type = 'info', timeout = 5000) {
  const notice = $('#notice');
  notice.textContent = message;
  notice.className = `notice${type === 'error' ? ' is-error' : ''}`;
  notice.hidden = false;
  clearTimeout(state.noticeTimer);
  if (timeout) state.noticeTimer = setTimeout(() => { notice.hidden = true; }, timeout);
}

function setStarting(isStarting, detail = '') {
  $('#startup-banner').hidden = !isStarting;
  if (detail) $('#startup-detail').textContent = detail;
}

function scopeChips(scopes = []) {
  return scopes.length
    ? scopes.map(scope => `<span class="scope-chip">${escapeHTML(scope)}</span>`).join('')
    : '<span class="muted">no scopes</span>';
}

async function loadSnapshot({ quiet = false } = {}) {
  const endpoints = [
    ['overview', '/overview'], ['people', '/people'], ['roles', '/roles'], ['scopes', '/scopes'],
    ['tasks', '/tasks'], ['health', '/health'], ['settings', '/settings'],
  ];
  try {
    const baseResults = await Promise.all(endpoints.map(async ([key, path]) => [key, await api(path)]));
    const requestResults = await Promise.all(['pending', 'running', 'done', 'failed'].map(requestState => api(`/requests?state=${encodeURIComponent(requestState)}`)));
    for (const [key, value] of baseResults) state[key] = value;
    state.requests = requestResults.flat().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    setStarting(false);
    renderAll();
    if (!quiet) showNotice('Console refreshed.');
  } catch (error) {
    if (error.status === 503) {
      setStarting(true, error.message || 'The console will retry when the API is ready.');
      renderUnavailable();
      window.setTimeout(() => loadSnapshot({ quiet: true }), 4000);
      return;
    }
    setStarting(false);
    showNotice(`Could not load console: ${error.message}`, 'error', 0);
    renderUnavailable();
  }
}

function renderUnavailable() {
  for (const selector of ['#waiting-list', '#people-list', '#tasks-list', '#health-content', '#settings-list']) {
    const target = $(selector);
    if (target && !target.children.length) target.innerHTML = '<div class="empty-state">Data is not available yet.</div>';
  }
}

function renderAll() {
  renderRequests();
  renderPeople();
  renderRoles();
  renderTasks();
  renderHealth();
  renderSettings();
}

function pendingRequests() {
  return state.requests.filter(request => request.state === 'pending');
}

function renderRequests() {
  const waiting = pendingRequests();
  state.selectedWaiting = Math.max(0, Math.min(state.selectedWaiting, waiting.length - 1));
  const count = waiting.length;
  $('#waiting-count').textContent = count;
  $('#waiting-summary').textContent = count ? `${count} request${count === 1 ? '' : 's'} need a decision.` : 'No decisions are waiting.';
  $('#nav-waiting-count').hidden = !count;
  $('#nav-waiting-count').textContent = count;
  $('#waiting-list').setAttribute('aria-busy', 'false');
  $('#waiting-list').innerHTML = count
    ? waiting.map((request, index) => requestCard(request, index)).join('')
    : '<div class="waiting-empty"><strong>Nothing waiting.</strong>New requests will appear here as they arrive.</div>';

  const counts = ['running', 'done', 'failed'].map(name => `${state.requests.filter(r => r.state === name).length} ${name}`).join(' / ');
  $('#history-counts').textContent = counts;
  const history = state.requests.filter(request => request.state !== 'pending' && (state.requestFilter === 'all' || request.state === state.requestFilter));
  $('#request-history').innerHTML = history.length ? history.map(historyRow).join('') : '<div class="empty-state">No requests in this state.</div>';
}

function requestCard(request, index) {
  return `<article class="request-card${index === state.selectedWaiting ? ' is-selected' : ''}" data-request-id="${escapeHTML(request.id)}" data-waiting-index="${index}" tabindex="${index === state.selectedWaiting ? '0' : '-1'}" aria-label="Waiting request from ${escapeHTML(request.sender_name || request.sender)}">
    <div class="request-main">
      <div class="request-who">
        <div class="sender">${escapeHTML(request.sender_name || request.sender || 'Unknown sender')}</div>
        <div class="surface">${escapeHTML(request.surface || 'unknown surface')}</div>
        <div class="request-id">${escapeHTML(request.id)}</div>
      </div>
      <div>
        <p class="request-summary">${escapeHTML(request.summary || 'No summary')}</p>
        <div class="scope-line">${scopeChips(request.scopes)}</div>
      </div>
      <div class="request-actions">
        <button class="button button-approve" type="button" data-decision="yes">Approve</button>
        <button class="button button-reject" type="button" data-decision="no">Reject</button>
        <button class="button button-quiet" type="button" data-show-edit>Edit</button>
      </div>
    </div>
    <details class="original-message" open>
      <summary>Original message · ${escapeHTML(formatTime(request.created_at))}</summary>
      <blockquote>${escapeHTML(request.original_message || 'No original message recorded.')}</blockquote>
    </details>
    <div class="decision-note" hidden>
      <textarea aria-label="Edited instruction" placeholder="Required: enter the edited instruction"></textarea>
      <button class="button button-primary" type="button" data-decision="edit">Run edit</button>
    </div>
  </article>`;
}

function historyRow(request) {
  return `<div class="history-row">
    <span class="state state-${escapeHTML(request.state)}">${escapeHTML(request.state)}</span>
    <span>${escapeHTML(request.sender_name || request.sender || 'Unknown')}</span>
    <span class="summary">${escapeHTML(request.summary || request.original_message || 'No summary')}</span>
    <time datetime="${escapeHTML(request.created_at || '')}">${escapeHTML(formatTime(request.created_at))}</time>
  </div>`;
}

async function decide(requestId, decision, note) {
  const card = $(`[data-request-id="${CSS.escape(requestId)}"]`);
  if (!card) return;
  if (decision === 'edit' && !note?.trim()) {
    showNotice('An edited instruction is required.', 'error');
    card.querySelector('textarea')?.focus();
    return;
  }
  $$('button', card).forEach(button => { button.disabled = true; });
  try {
    const payload = { id: requestId, decision };
    if (note?.trim()) payload.note = note.trim();
    const result = await post('/requests/decide', payload);
    const actualState = result?.state || (decision === 'no' ? 'rejected' : 'approved');
    showNotice(`Request ${requestId} is ${actualState}.`);
  } catch (error) {
    const actual = error.payload?.state || error.payload?.current_state || error.payload?.request?.state;
    const detail = actual ? `${error.message} Current state: ${actual}.` : error.message;
    showNotice(`Decision not applied: ${detail}`, 'error', 8000);
  } finally {
    await refreshRequests();
  }
}

async function refreshRequests() {
  try {
    const [overview, ...requests] = await Promise.all([
      api('/overview'),
      ...['pending', 'running', 'done', 'failed'].map(requestState => api(`/requests?state=${requestState}`)),
    ]);
    state.overview = overview;
    state.requests = requests.flat().sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    renderRequests();
  } catch (error) {
    showNotice(`Request status could not be refreshed: ${error.message}`, 'error');
  }
}

function filteredPeople() {
  const term = state.search.trim().toLowerCase();
  if (!term) return state.people;
  return state.people.filter(person => [person.person, person.key, person.role, person.note, ...(person.scopes || []), ...(person.effective_scopes || [])].some(value => String(value || '').toLowerCase().includes(term)));
}

function renderPeople() {
  const people = filteredPeople();
  $('#people-count').textContent = `${people.length}/${state.people.length}`;
  if (state.selectedPerson && !state.people.some(person => person.key === state.selectedPerson)) state.selectedPerson = null;
  if (!state.selectedPerson && people.length) state.selectedPerson = people[0].key;
  $('#people-list').innerHTML = people.length ? people.map(person => {
    const automatic = (person.effective_scopes || []).length;
    const stillAsks = Math.max(0, state.scopes.length - automatic);
    return `<button class="person-row${person.key === state.selectedPerson ? ' is-selected' : ''}" type="button" role="option" aria-selected="${person.key === state.selectedPerson}" data-person-key="${escapeHTML(person.key)}">
    <span class="person-role">${escapeHTML(person.role || 'no role')}</span>
    <span class="person-name">${escapeHTML(person.person || 'Unnamed')}</span>
    <span class="person-key">${escapeHTML(person.key)}</span>
    <span class="person-impact"><span class="auto">${automatic} run</span> / <span class="ask">${stillAsks} ask</span></span>
  </button>`;
  }).join('') : '<div class="empty-state">No people match this search.</div>';
  renderPersonEditor();
}

function renderPersonEditor(draft = null) {
  const person = draft || state.people.find(item => item.key === state.selectedPerson);
  if (!person) {
    $('#person-editor').innerHTML = '<div class="empty-state">No people configured. Add a handle to start with zero permissions.</div>';
    return;
  }
  const isNew = Boolean(person.__new);
  const selectedRole = state.roles.find(role => role.name === person.role);
  const inherited = new Set(selectedRole?.scopes || []);
  const extras = new Set(person.scopes || []);
  const effective = new Set([...inherited, ...extras]);
  const interrupted = state.scopes.map(scope => scope.name).filter(name => !effective.has(name));
  $('#person-editor').innerHTML = `<form id="person-form">
    <div class="editor-head">
      <div>
        <h3>${isNew ? 'New person' : escapeHTML(person.person || person.key)}</h3>
        <span class="person-key">${isNew ? 'Defaults to no permissions' : escapeHTML(person.key)}</span>
      </div>
    </div>
    <div class="editor-grid">
      <label class="field">Handle
        <input name="key" class="mono" required value="${escapeHTML(person.key || '')}" placeholder="email, phone, Telegram id, or name" ${isNew ? '' : 'readonly'}>
      </label>
      <label class="field">Display name
        <input name="person" value="${escapeHTML(person.person || '')}" placeholder="Name">
      </label>
      <label class="field">Role
        <select name="role">
          <option value="">No role</option>
          ${state.roles.map(role => `<option value="${escapeHTML(role.name)}" ${role.name === person.role ? 'selected' : ''}>${escapeHTML(role.name)} · ${role.scopes.length} scopes</option>`).join('')}
        </select>
      </label>
      <label class="field">Note
        <textarea name="note" placeholder="Why this access exists">${escapeHTML(person.note || '')}</textarea>
      </label>
    </div>
    <div class="consequence">
      <div><span class="consequence-label auto">Runs without asking · ${effective.size}</span><div class="scope-plain-list">${effective.size ? [...effective].sort().map(escapeHTML).join('<br>') : 'Nothing'}</div></div>
      <div><span class="consequence-label ask">Still asks you · ${interrupted.length}</span><div class="scope-plain-list">${interrupted.length ? interrupted.sort().map(escapeHTML).join('<br>') : 'Nothing'}</div></div>
    </div>
    <div class="scope-editor-title"><h3>Extra scopes</h3><p>Checked here in addition to the role.</p></div>
    <div class="scope-groups">${renderScopeGroups(extras, inherited)}</div>
    <div class="editor-actions">
      ${isNew ? '<button class="button button-quiet" type="button" data-cancel-person>Cancel</button>' : '<button class="button button-danger" type="button" data-delete-person>Delete person</button>'}
      <div class="save-actions"><button class="button button-primary" type="submit">Save person</button></div>
    </div>
  </form>`;
}

function renderScopeGroups(selected = new Set(), inherited = new Set()) {
  const term = state.search.trim().toLowerCase();
  const groups = Map.groupBy
    ? Map.groupBy(state.scopes, scope => scope.group || 'Other')
    : state.scopes.reduce((map, scope) => map.set(scope.group || 'Other', [...(map.get(scope.group || 'Other') || []), scope]), new Map());
  return [...groups.entries()].map(([group, scopes]) => {
    const matches = scope => !term || `${scope.name} ${scope.purpose || ''} ${group}`.toLowerCase().includes(term);
    const groupHidden = term && !scopes.some(matches);
    return `<fieldset class="scope-group" ${groupHidden ? 'hidden' : ''}><legend>${escapeHTML(group)}</legend><div class="scope-options">${scopes.map(scope => {
      const isInherited = inherited.has(scope.name);
      return `<label class="scope-option${isInherited ? ' is-inherited' : ''}" ${matches(scope) ? '' : 'hidden'}>
        <input type="checkbox" name="scopes" value="${escapeHTML(scope.name)}" ${selected.has(scope.name) ? 'checked' : ''} ${isInherited ? 'disabled' : ''}>
        <span class="scope-option-text"><span class="scope-name">${escapeHTML(scope.name)}${isInherited ? '<span class="inherited-mark">role</span>' : ''}</span><span class="scope-purpose">${escapeHTML(scope.purpose || `${scope.tools_count || 0} tools`)}</span></span>
      </label>`;
    }).join('')}</div></fieldset>`;
  }).join('');
}

async function savePerson(form) {
  const data = new FormData(form);
  const body = {
    key: data.get('key').trim(),
    person: data.get('person').trim(),
    role: data.get('role') || '',
    scopes: data.getAll('scopes'),
    note: data.get('note').trim(),
  };
  if (!body.key) return showNotice('A handle is required.', 'error');
  setFormBusy(form, true);
  try {
    await post('/people', body);
    state.selectedPerson = body.key;
    await refreshPermissions();
    showNotice(`Saved ${body.person || body.key}.`);
  } catch (error) {
    showNotice(`Person not saved: ${error.message}`, 'error', 8000);
    setFormBusy(form, false);
  }
}

async function deletePerson(key) {
  if (!window.confirm(`Delete ${key}? Their permissions will be removed.`)) return;
  try {
    await post('/people/delete', { key });
    state.selectedPerson = null;
    await refreshPermissions();
    showNotice(`Deleted ${key}.`);
  } catch (error) { showNotice(`Person not deleted: ${error.message}`, 'error'); }
}

async function refreshPermissions() {
  const [people, roles, scopes] = await Promise.all([api('/people'), api('/roles'), api('/scopes')]);
  state.people = people; state.roles = roles; state.scopes = scopes;
  renderPeople(); renderRoles();
}

function renderRoles() {
  $('#roles-count').textContent = `${state.roles.length} configured`;
  $('#roles-list').innerHTML = state.roles.length ? state.roles.map(role => `<div class="role-row" data-role-name="${escapeHTML(role.name)}">
    <div><span class="role-name">${escapeHTML(role.name)}</span><div class="muted">${escapeHTML(role.note || 'No note')}</div></div>
    <div class="role-scopes">${role.scopes.length ? role.scopes.map(escapeHTML).join(' · ') : 'No scopes'}</div>
    <div class="mono muted">${role.people_count} ${role.people_count === 1 ? 'person' : 'people'}</div>
    <div><button class="button button-quiet" type="button" data-edit-role>Edit</button></div>
  </div>`).join('') : '<div class="empty-state">No roles configured. People default to no permissions.</div>';
}

function openRoleEditor(role = { name: '', note: '', scopes: [], __new: true }) {
  const host = role.__new ? document.createElement('div') : $(`[data-role-name="${CSS.escape(role.name)}"]`);
  if (role.__new) { host.className = 'role-row'; $('#roles-list').prepend(host); }
  const selected = new Set(role.scopes || []);
  host.innerHTML = `<form class="role-editor">
    <div class="editor-grid">
      <label class="field">Role name<input name="name" required value="${escapeHTML(role.name)}" ${role.__new ? '' : 'readonly'}></label>
      <label class="field">Note<textarea name="note" placeholder="Who this role is for">${escapeHTML(role.note || '')}</textarea></label>
    </div>
    <div class="scope-groups">${renderScopeGroups(selected)}</div>
    <div class="editor-actions">
      ${role.__new ? '<span></span>' : '<button class="button button-danger" type="button" data-delete-role>Delete role</button>'}
      <div class="save-actions"><button class="button button-quiet" type="button" data-cancel-role>Cancel</button><button class="button button-primary" type="submit">Save role</button></div>
    </div>
  </form>`;
  $('input', host)?.focus();
}

async function saveRole(form) {
  const data = new FormData(form);
  const body = { name: data.get('name').trim(), scopes: data.getAll('scopes'), note: data.get('note').trim() };
  if (!body.name) return showNotice('A role name is required.', 'error');
  setFormBusy(form, true);
  try {
    await post('/roles', body);
    await refreshPermissions();
    showNotice(`Saved ${body.name}. Everyone with this role is now updated.`);
  } catch (error) { showNotice(`Role not saved: ${error.message}`, 'error'); setFormBusy(form, false); }
}

async function deleteRole(name) {
  const role = state.roles.find(item => item.name === name);
  const impact = role?.people_count ? ` ${role.people_count} ${role.people_count === 1 ? 'person' : 'people'} will lose its inherited scopes.` : '';
  if (!window.confirm(`Delete role ${name}?${impact}`)) return;
  try {
    await post('/roles/delete', { name });
    await refreshPermissions();
    showNotice(`Deleted role ${name}.`);
  } catch (error) { showNotice(`Role not deleted: ${error.message}`, 'error'); }
}

function setFormBusy(form, busy) {
  $$('button, input, select, textarea', form).forEach(control => { control.disabled = busy; });
}

function renderTasks() {
  $('#tasks-list').setAttribute('aria-busy', 'false');
  $('#tasks-list').innerHTML = state.tasks.length ? state.tasks.map(task => `<button class="task-row" type="button" data-task-id="${escapeHTML(task.id)}">
    <span class="mono">${escapeHTML(task.id)}</span>
    <span class="summary">${escapeHTML(task.title || 'Untitled task')}</span>
    <span class="state state-${escapeHTML(task.state)}">${escapeHTML(task.state)}</span>
    <time datetime="${escapeHTML(task.updated_at || '')}">${escapeHTML(formatTime(task.updated_at))}</time>
    <span class="participant-list">${escapeHTML((task.participants || []).join(', '))}</span>
  </button>`).join('') : '<div class="empty-state">No recent tasks.</div>';
}

async function openTask(taskId) {
  const dialog = $('#task-dialog');
  $('#ledger-id').textContent = taskId;
  $('#ledger-title').textContent = 'Loading ledger…';
  $('#ledger-content').textContent = '';
  dialog.showModal();
  try {
    const task = await api(`/tasks/${encodeURIComponent(taskId)}`);
    $('#ledger-title').textContent = task.title || 'Task ledger';
    $('#ledger-content').textContent = task.ledger || 'No ledger entries recorded.';
  } catch (error) {
    $('#ledger-title').textContent = 'Ledger unavailable';
    $('#ledger-content').textContent = error.message;
  }
}

function renderHealth() {
  $('#health-content').setAttribute('aria-busy', 'false');
  if (!state.health) return;
  const interrupted = state.health.interrupted || [];
  const errors = state.health.errors || [];
  $('#health-content').innerHTML = `<dl class="health-grid">
    <div class="health-stat"><dt>Tunnel</dt><dd class="state-${state.health.tunnel === 'up' || state.health.tunnel === true ? 'done' : 'failed'}">${escapeHTML(String(state.health.tunnel))}</dd></div>
    <div class="health-stat"><dt>Uptime</dt><dd>${escapeHTML(formatDuration(state.health.uptime_s))}</dd></div>
    <div class="health-stat"><dt>Last restart</dt><dd>${escapeHTML(formatTime(state.health.last_restart))}</dd></div>
  </dl>
  <div class="health-columns">
    <div><h3>Interrupted runs · ${interrupted.length}</h3><ul class="health-list">${interrupted.length ? interrupted.map(item => `<li>${escapeHTML(typeof item === 'string' ? item : JSON.stringify(item))}</li>`).join('') : '<li>None</li>'}</ul></div>
    <div><h3>Recent errors · ${errors.length}</h3><ul class="health-list">${errors.length ? errors.map(item => `<li>${escapeHTML(typeof item === 'string' ? item : JSON.stringify(item))}</li>`).join('') : '<li>None</li>'}</ul></div>
  </div>`;
}

async function refreshHealth() {
  try { state.health = await api('/health'); renderHealth(); }
  catch (error) { showNotice(`Health not refreshed: ${error.message}`, 'error'); }
}

async function refreshTasks() {
  try { state.tasks = await api('/tasks'); renderTasks(); }
  catch (error) { showNotice(`Tasks not refreshed: ${error.message}`, 'error'); }
}

const settingSaveTimers = new Map();

function isSeriousSetting(setting) {
  const description = `${setting.name} ${setting.label} ${setting.help}`.toLowerCase();
  return setting.kind === 'bool' && setting.group === 'Safety'
    && /outbound|send|message/.test(description) && /preview|review|approval|confirm|before|shown/.test(description);
}

function settingControl(setting) {
  const value = setting.value;
  if (setting.kind === 'bool') {
    return `<label class="setting-toggle"><input type="checkbox" data-setting-control data-setting-kind="bool" ${value ? 'checked' : ''}><span class="toggle-track" aria-hidden="true"><span></span></span><span>${value ? 'On' : 'Off'}</span></label>`;
  }
  if (setting.kind === 'number') {
    const bounds = `${setting.min == null ? '' : ` min="${escapeHTML(setting.min)}"`}${setting.max == null ? '' : ` max="${escapeHTML(setting.max)}"`} step="${escapeHTML(setting.step ?? 1)}"`;
    return `<div class="number-control"><input type="range" data-setting-control data-setting-kind="number"${bounds} value="${escapeHTML(value)}" aria-label="${escapeHTML(setting.label)} slider"><input type="number" data-setting-control data-setting-kind="number"${bounds} value="${escapeHTML(value)}" aria-label="${escapeHTML(setting.label)} value"></div>`;
  }
  if (setting.kind === 'choice') {
    return `<select data-setting-control data-setting-kind="choice" aria-label="${escapeHTML(setting.label)}">${(setting.choices || []).map(choice => `<option value="${escapeHTML(choice)}" ${choice === value ? 'selected' : ''}>${escapeHTML(choice)}</option>`).join('')}</select>`;
  }
  return `<textarea data-setting-control data-setting-kind="text" rows="1" aria-label="${escapeHTML(setting.label)}">${escapeHTML(value ?? '')}</textarea>`;
}

function renderSettings() {
  const host = $('#settings-list');
  host.setAttribute('aria-busy', 'false');
  if (!state.settings.length) {
    host.innerHTML = '<div class="empty-state">No settings are available.</div>';
    return;
  }
  const groups = state.settings.reduce((map, setting) => {
    const group = setting.group || 'Other';
    if (!map.has(group)) map.set(group, []);
    map.get(group).push(setting);
    return map;
  }, new Map());
  host.innerHTML = [...groups.entries()].map(([group, settings], groupIndex) => `<section class="settings-group" aria-labelledby="settings-group-${groupIndex}">
    <h3 id="settings-group-${groupIndex}">${escapeHTML(group)}</h3>
    <div class="settings-panel">${settings.map(setting => {
      const serious = isSeriousSetting(setting);
      const status = state.settingStatus[setting.name];
      return `<div class="setting-row${serious ? ' is-serious' : ''}" data-setting-name="${escapeHTML(setting.name)}">
        <div class="setting-copy">
          <div class="setting-label">${escapeHTML(setting.label)}${serious ? '<span class="serious-mark">Serious</span>' : ''}</div>
          <div class="setting-help">${escapeHTML(setting.help || '')}</div>
          <code class="setting-name">${escapeHTML(setting.name)}</code>
        </div>
        <div class="setting-value">
          ${settingControl(setting)}
          <div class="setting-meta">
            <span class="setting-source source-${escapeHTML(setting.source)}">${escapeHTML(setting.source)}</span>
            ${setting.source === 'console' || setting.value !== setting.default ? '<button class="button button-danger setting-reset" type="button" data-reset-setting>Reset</button>' : ''}
            <span class="setting-save-state${status?.type === 'error' ? ' is-error' : ''}" role="status">${escapeHTML(status?.text || '')}</span>
          </div>
        </div>
      </div>`;
    }).join('')}</div>
  </section>`).join('');
  $$('textarea[data-setting-control]', host).forEach(growSettingTextarea);
}

function growSettingTextarea(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${textarea.scrollHeight + 2}px`;
}

function scheduleSettingSave(setting, value, delay) {
  clearTimeout(settingSaveTimers.get(setting.name));
  state.settingStatus[setting.name] = { type: 'saving', text: 'Saving…' };
  const row = $(`[data-setting-name="${CSS.escape(setting.name)}"]`);
  const status = $('.setting-save-state', row);
  if (status) { status.textContent = 'Saving…'; status.classList.remove('is-error'); }
  settingSaveTimers.set(setting.name, setTimeout(() => saveSetting(setting, value), delay));
}

async function saveSetting(setting, value) {
  clearTimeout(settingSaveTimers.get(setting.name));
  settingSaveTimers.delete(setting.name);
  try {
    const result = await post('/settings', { name: setting.name, value });
    state.settings = result.settings || state.settings;
    state.settingStatus[setting.name] = { type: 'saved', text: 'Saved' };
    renderSettings();
    showNotice(`Saved ${setting.label}.`);
  } catch (error) {
    state.settingStatus[setting.name] = { type: 'error', text: error.message };
    renderSettings();
    showNotice(`${setting.label} not saved: ${error.message}`, 'error', 8000);
  }
}

async function refreshSettings() {
  try { state.settings = await api('/settings'); renderSettings(); }
  catch (error) { showNotice(`Settings not refreshed: ${error.message}`, 'error'); }
}

function connectEvents() {
  clearTimeout(state.reconnectTimer);
  state.eventSource?.close();
  setStreamState('connecting');
  const source = new EventSource('/console/events');
  state.eventSource = source;
  source.onopen = () => { state.reconnectAttempt = 0; setStreamState('live'); };
  source.addEventListener('ping', () => setStreamState('live'));
  source.addEventListener('request.created', refreshRequests);
  source.addEventListener('request.state_changed', refreshRequests);
  source.addEventListener('permissions.changed', async () => {
    try { await refreshPermissions(); } catch (error) { showNotice(`Permissions not refreshed: ${error.message}`, 'error'); }
  });
  source.addEventListener('health.changed', refreshHealth);
  source.addEventListener('settings.changed', refreshSettings);
  source.onerror = async () => {
    source.close();
    if (state.eventSource !== source) return;
    setStreamState('offline');
    await loadSnapshot({ quiet: true });
    const delay = Math.min(30000, 1000 * (2 ** state.reconnectAttempt++)) + Math.floor(Math.random() * 500);
    state.reconnectTimer = setTimeout(connectEvents, delay);
  };
}

function setStreamState(streamState) {
  const dot = $('#stream-dot');
  dot.className = `stream-dot${streamState === 'live' ? ' is-live' : streamState === 'connecting' ? ' is-connecting' : ''}`;
  $('#stream-label').textContent = streamState;
}

function moveWaiting(direction) {
  const waiting = pendingRequests();
  if (!waiting.length) return;
  state.selectedWaiting = (state.selectedWaiting + direction + waiting.length) % waiting.length;
  renderRequests();
  const selected = $(`[data-waiting-index="${state.selectedWaiting}"]`);
  selected?.focus({ preventScroll: true });
  selected?.scrollIntoView({ block: 'nearest', behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth' });
}

function isTypingTarget(target) {
  return target.matches('input, textarea, select, [contenteditable="true"]');
}

document.addEventListener('click', event => {
  const waitingCard = event.target.closest('[data-waiting-index]');
  if (waitingCard) {
    state.selectedWaiting = Number(waitingCard.dataset.waitingIndex);
    $$('.request-card').forEach((card, index) => card.classList.toggle('is-selected', index === state.selectedWaiting));
  }
  const decisionButton = event.target.closest('[data-decision]');
  if (decisionButton) {
    const card = decisionButton.closest('[data-request-id]');
    decide(card.dataset.requestId, decisionButton.dataset.decision, card.querySelector('textarea')?.value);
  }
  const editButton = event.target.closest('[data-show-edit]');
  if (editButton) {
    const note = editButton.closest('.request-card').querySelector('.decision-note');
    note.hidden = !note.hidden;
    if (!note.hidden) $('textarea', note).focus();
  }
  const personButton = event.target.closest('[data-person-key]');
  if (personButton) { state.selectedPerson = personButton.dataset.personKey; renderPeople(); }
  const editRole = event.target.closest('[data-edit-role]');
  if (editRole) openRoleEditor(state.roles.find(role => role.name === editRole.closest('[data-role-name]').dataset.roleName));
  const cancelRole = event.target.closest('[data-cancel-role]');
  if (cancelRole) renderRoles();
  const deleteRoleButton = event.target.closest('[data-delete-role]');
  if (deleteRoleButton) deleteRole(deleteRoleButton.closest('[data-role-name]').dataset.roleName);
  const taskButton = event.target.closest('[data-task-id]');
  if (taskButton) openTask(taskButton.dataset.taskId);
  const closeButton = event.target.closest('[data-close-dialog]');
  if (closeButton) $(`#${closeButton.dataset.closeDialog}`).close();
  if (event.target.closest('[data-open-shortcuts]')) $('#shortcuts-dialog').showModal();
  const resetSetting = event.target.closest('[data-reset-setting]');
  if (resetSetting) {
    const setting = state.settings.find(item => item.name === resetSetting.closest('[data-setting-name]').dataset.settingName);
    if (setting) saveSetting(setting, null);
  }
});

document.addEventListener('submit', event => {
  event.preventDefault();
  if (event.target.id === 'person-form') savePerson(event.target);
  if (event.target.matches('.role-editor')) saveRole(event.target);
});

document.addEventListener('change', event => {
  if (event.target.matches('[data-setting-control]')) {
    const row = event.target.closest('[data-setting-name]');
    const setting = state.settings.find(item => item.name === row.dataset.settingName);
    if (!setting) return;
    if (setting.kind === 'number' || setting.kind === 'text') return;
    const value = setting.kind === 'bool' ? event.target.checked : setting.kind === 'number' ? Number(event.target.value) : event.target.value;
    if (isSeriousSetting(setting) && setting.value === true && value === false
      && !window.confirm('Turn this off? Outbound messages will be sent without being shown to you first.')) {
      event.target.checked = true;
      const label = event.target.closest('.setting-toggle')?.querySelector('span:last-child');
      if (label) label.textContent = 'On';
      return;
    }
    if (setting.kind === 'bool') event.target.closest('.setting-toggle').querySelector('span:last-child').textContent = value ? 'On' : 'Off';
    saveSetting(setting, value);
  }
  if (event.target.matches('#person-form select[name="role"], #person-form input[name="scopes"]')) {
    const form = event.target.form;
    const current = state.people.find(person => person.key === state.selectedPerson) || { __new: true };
    const data = new FormData(form);
    renderPersonEditor({
      ...current,
      __new: !state.selectedPerson,
      key: data.get('key'), person: data.get('person'), role: data.get('role'), note: data.get('note'), scopes: data.getAll('scopes'),
    });
  }
});

document.addEventListener('input', event => {
  if (!event.target.matches('[data-setting-control]')) return;
  const row = event.target.closest('[data-setting-name]');
  const setting = state.settings.find(item => item.name === row.dataset.settingName);
  if (!setting) return;
  if (setting.kind === 'number') {
    $$('input', row).forEach(input => { if (input !== event.target) input.value = event.target.value; });
    scheduleSettingSave(setting, Number(event.target.value), 350);
  }
  if (setting.kind === 'text') {
    growSettingTextarea(event.target);
    scheduleSettingSave(setting, event.target.value, 600);
  }
});

document.addEventListener('keydown', event => {
  if (event.key === '?' && !isTypingTarget(event.target)) { event.preventDefault(); $('#shortcuts-dialog').showModal(); return; }
  if (event.key === '/' && !isTypingTarget(event.target)) { event.preventDefault(); $('#permission-search').focus(); return; }
  if (isTypingTarget(event.target) || $('dialog[open]')) return;
  if (event.key === 'j') { event.preventDefault(); moveWaiting(1); }
  if (event.key === 'k') { event.preventDefault(); moveWaiting(-1); }
  if ((event.key === 'y' || event.key === 'n') && pendingRequests().length) {
    event.preventDefault();
    const request = pendingRequests()[state.selectedWaiting];
    decide(request.id, event.key === 'y' ? 'yes' : 'no');
  }
});

$('#permission-search').addEventListener('input', event => { state.search = event.target.value; renderPeople(); });
$('#add-person').addEventListener('click', () => { state.selectedPerson = null; renderPersonEditor({ key: '', person: '', role: '', scopes: [], note: '', __new: true }); $('#person-editor input').focus(); });
$('#person-editor').addEventListener('click', event => {
  if (event.target.closest('[data-cancel-person]')) renderPeople();
  if (event.target.closest('[data-delete-person]')) deletePerson(state.selectedPerson);
});
$('#add-role').addEventListener('click', () => { $('details.roles-block').open = true; openRoleEditor(); });
$$('[data-request-filter]').forEach(button => button.addEventListener('click', () => {
  state.requestFilter = button.dataset.requestFilter;
  $$('[data-request-filter]').forEach(item => item.classList.toggle('is-active', item === button));
  renderRequests();
}));
$('#retry-all').addEventListener('click', () => loadSnapshot());
$('#refresh-tasks').addEventListener('click', refreshTasks);
$('#refresh-health').addEventListener('click', refreshHealth);
$$('dialog').forEach(dialog => dialog.addEventListener('click', event => {
  if (event.target === dialog) dialog.close();
}));

loadSnapshot({ quiet: true });
connectEvents();
