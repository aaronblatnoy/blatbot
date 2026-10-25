"""Metadata retrieval, tree narrowing and complete-body regression tests."""
import asyncio
import time
from types import SimpleNamespace

from inkbox_claude.gate.ledger import continuation_id, index_entry, render_index
from inkbox_claude.gate.store import Store
from inkbox_claude.gate.taskpick import TaskPicker
from inkbox_claude.gate import manager as gm


def test_index_is_metadata_only_and_stale_tasks_remain_searchable(tmp_path, monkeypatch):
    store = Store(str(tmp_path / 'gate.db'))
    fresh = store.create_task('TAMID interviews')
    old = store.create_task('Ancient needle')
    store.task_event(fresh['id'], 'note', 'PRIVATE BODY' * 1000)
    store.set_task_waiting_on(fresh['id'], 'Test participant')
    now = time.time()
    store._db.execute('UPDATE tasks SET updated_at=? WHERE id=?', (now - 15 * 86400, old['id']))
    store._db.commit()
    monkeypatch.setattr(store, 'get_task', lambda *a: (_ for _ in ()).throw(AssertionError('hydrated')))
    result = store.ledger_index(is_approver=True, now=now)
    assert result['dropped'] == 1
    assert [t['id'] for t in result['tasks']] == [fresh['id']]
    text = render_index(result['tasks'])
    assert 'PRIVATE BODY' not in text
    assert 'tamid_workspace' in text
    assert 'waiting: Test participant' in text
    monkeypatch.undo()
    assert store.query_tasks(text='Ancient')['tasks'][0]['id'] == old['id']


def test_index_access_and_recency(tmp_path):
    store = Store(str(tmp_path / 'gate.db'))
    owned = store.create_task('Own', [('one@example.com', 'One')])
    store.create_task('Secret', [('two@example.com', 'Two')])
    linked = store.create_task('Thread task')
    result = store.ledger_index(is_approver=False, person='one@example.com', thread_ids=[linked['id']])
    assert [t['id'] for t in result['tasks']] == [linked['id'], owned['id']]


def test_tree_two_levels_and_confident_overflow(monkeypatch):
    calls = []
    class Client:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def post(self, url, headers, json):
            calls.append(json)
            answers = {key: {'noul': (0.1 if key == 'server' else 0.4 if key == 'tamid_workspace' else 0.95 if key in ('T1', 'T2', 'T3') else 0.8)} for key in json['questions']}
            if len(calls) == 2:
                answers['T4'] = {'noul': 0.95}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'answers': answers})
    monkeypatch.setattr('inkbox_claude.gate.taskpick.httpx.AsyncClient', Client)
    entries = [{'id': i, 'categories': ['tamid_workspace'], 'line': f'T{i} | title'} for i in range(1, 6)]
    entries.append({'id': 6, 'categories': ['server'], 'line': 'T6 | secret'})
    result = asyncio.run(TaskPicker(api_key='test').judge_ledger_tree(message='interviews', history=[], entries=entries))
    assert len(calls) == 2
    assert 'T6' not in calls[1]['questions']
    assert set(result['ids']) == {1, 2, 3} | {4}
    assert set(calls[0]['questions']) == {'tamid_workspace', 'server'}
    assert calls[0]['state']['ledger_index_by_category']['server'] == ['T6 | secret']


def test_continuation_is_conservative():
    rows = [{'id': 7}]
    history = [{'created_at': time.time()}]
    assert continuation_id('try again', rows, [7], history) == 7
    assert continuation_id('T7 please', rows, [], []) == 7
    assert continuation_id('New job please', rows, [7], history) is None
    assert continuation_id('try again', rows, [7], [{'created_at': 1}]) is None
    assert continuation_id('T8 please', rows, [7], history) is None


def test_session_fast_path_loads_only_complete_chosen_body(tmp_path):
    from test_gate import make_manager, approver_meta, APPROVER_CONV
    m, _ = make_manager(tmp_path)
    session = m.get(APPROVER_CONV)
    session.mode = 'imessage'
    session.reply_meta = approver_meta()
    chosen = m.store.create_task('TAMID interviews')
    other = m.store.create_task('Server task')
    body = ('complete source ' * 2000).strip()
    m.store.task_event(chosen['id'], 'note', body, chat_id=session.record_id)
    class Picker:
        async def judge_ledger_tree(self, **kw):
            raise AssertionError('fast path must not call Jev')
    m.task_picker = Picker()
    text = asyncio.run(session.indexed_task_memory([], f"T{chosen['id']}"))
    assert body in text
    assert f"Task T{other['id']} " not in text
    assert [t['id'] for t in session.candidate_tasks()] == [chosen['id']]


def test_request_judgments_overlap_and_merge_without_overwrite():
    from inkbox_claude.gate.decidegraph import judge_request
    from inkbox_claude.gate.router import RouterRequest
    started = set()
    async def barrier(name):
        started.add(name)
        for _ in range(100):
            if len(started) == 3: return
            await asyncio.sleep(0)
        raise AssertionError('judgments were serialized')
    class Session:
        async def jev_scopes(self, out, strict):
            await barrier('scopes')
            out.request.scopes = ['calendar']
        async def jev_knowledge(self, out):
            await barrier('vault')
            out.request.scopes = ['web', 'vault:test']
            out.request.knowledge_notes = ['notes/test.md']
        async def jev_event(self, *args): await barrier('event')
    req = RouterRequest(summary='test', prompt='source', scopes=['web'])
    result = asyncio.run(judge_request({'request': req, 'task': {}, 'body': 'x', 'prior': []},
                                      {'configurable': {'session': Session()}}))
    assert result['request'].scopes == ['calendar', 'vault:test']
    assert result['request'].knowledge_notes == ['notes/test.md']
    assert req.scopes == ['web']


def test_email_matching_long_non_email_and_normal_addresses():
    from inkbox_claude.gate.jevagent import _EMAIL_RE, _LABELED_ID_RE
    assert _EMAIL_RE.findall('x' * 90000) == []
    assert _LABELED_ID_RE.findall('x' * 90000) == []
    assert _LABELED_ID_RE.findall('ID: abc, spreadsheet_id=def') == ['abc', 'def']
    assert _EMAIL_RE.findall('Contact <first.last+tag@example.org> or second@example.com') == [
        'first.last+tag@example.org', 'second@example.com']


def test_lookup_shares_body_budget_and_retains_overlapping_match(tmp_path):
    from test_gate import make_manager, approver_meta, APPROVER_CONV
    from inkbox_claude.gate.router import TaskQuery
    m, _ = make_manager(tmp_path)
    session = m.get(APPROVER_CONV)
    session.mode = 'imessage'
    session.reply_meta = approver_meta()
    tasks = [m.store.create_task(f'Task subject {i}') for i in range(4)]
    session._ledger_selected_ids = [t['id'] for t in tasks[:3]]
    session._ledger_candidates = tasks[:3]
    session._ledger_memory = '\n\n'.join(m.store.task_memory_for_task(t['id']) for t in tasks[:3])
    session._found_task_ids = []
    m.router.next_query = TaskQuery(text='subject', limit=3)
    async def go():
        return await session.lookup_tasks([], 'find subject', session._ledger_memory)
    found = asyncio.run(go())
    shown = session._ledger_memory + found
    assert shown.count('Task T') <= 3
    assert f"Task T{tasks[-1]['id']} " in shown
    assert f"Task T{tasks[2]['id']} " in shown
