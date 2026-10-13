from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.boundary.sqlite_received import SQLiteReceivedStore
from virtual_companion.text.stream_host import StreamHostBridge


def setup_received(tmp_path):
    store = SQLiteReceivedStore(tmp_path / 'received.sqlite3')
    item = store.accept(config(), 'received', '你好')
    run = claim(store, item)
    store.approve_segment(run, reviewed(run), now_us=100)
    event = store.dispatch_segment(run, 1, now_us=100)
    return store, run, event


def test_received_releases_host_wait_without_claiming_visibility(tmp_path):
    store, run, event = setup_received(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(bridge.present, event)
        assert bridge.ready.wait(1)
        receipt = store.receive_segment(A, run.input_id, 1,
            fingerprint=event['fingerprint'], host_receipt='device-received-one')
        bridge.receipt_committed()
        assert waiting.result(1) == receipt
    state = store.stream_state(A, run.input_id)
    assert state['received_units'] == 1 and state['confirmed_units'] == 0
    assert state['confirmed_text'] == '' and state['unknown_units'] == 0
    assert not [s for s in store.background_sources(A) if s['role'] == 'assistant']
    assert store.finish_stream(run, outcome='received', now_us=100)['outcome'] == 'received'
    reopened = SQLiteReceivedStore(store.path)
    assert reopened.received_segment(A, run.input_id, 1) == receipt
    reopened.confirm_segment(A, run.input_id, 1, fingerprint=event['fingerprint'],
                             host_receipt='actual-visible-one', channel='text_visible')
    assert reopened.stream_state(A, run.input_id)['confirmed_units'] == 1
    assert reopened.stream_state(A, run.input_id)['outcome'] == 'received'


def test_received_predecessor_allows_next_dispatch_but_out_of_order_visible_is_not_prefix(tmp_path):
    store, run, event = setup_received(tmp_path)
    store.receive_segment(A, run.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='device-received-one')
    store.approve_segment(run, reviewed(run, ('我在这里。', '你继续。')), now_us=100)
    second = store.dispatch_segment(run, 2, now_us=100)
    store.receive_segment(A, run.input_id, 2, fingerprint=second['fingerprint'],
                          host_receipt='device-received-two')
    store.confirm_segment(A, run.input_id, 2, fingerprint=second['fingerprint'],
                          host_receipt='actual-visible-two', channel='text_visible')
    assert store.stream_state(A, run.input_id)['confirmed_text'] == ''
    with pytest.raises(ValueError):
        store.finish_stream(run, outcome='completed', now_us=100)
    store.finish_stream(run, outcome='received', now_us=100)
    with pytest.raises(ValueError):
        store.receive_segment(A, run.input_id, 1, fingerprint=event['fingerprint'],
                              host_receipt='conflicting-device-receipt')


def test_v19_upgrade_keeps_readable_backup_and_does_not_reclassify_unknown(tmp_path):
    old = SQLiteExplicitMemoryStore(tmp_path / 'upgrade.sqlite3')
    item = old.accept(config(), 'old', '旧合成输入')
    run = claim(old, item)
    old.approve_segment(run, reviewed(run), now_us=100)
    old.dispatch_segment(run, 1, now_us=100)
    upgraded = SQLiteReceivedStore(old.path)
    backup = SQLiteExplicitMemoryStore(upgraded.upgrade_backup)
    assert backup.inspect(A, run.input_id)['text'] == '旧合成输入'
    assert upgraded.stream_state(A, run.input_id)['unknown_units'] == 1
    assert upgraded.stream_state(A, run.input_id)['received_units'] == 0
    assert not upgraded.received_segment(A, run.input_id, 1)


def test_backup_failure_prevents_any_v19_schema_change(tmp_path, monkeypatch):
    old = SQLiteExplicitMemoryStore(tmp_path / 'backup-failure.sqlite3')
    item = old.accept(config(), 'old', '原合成输入')

    def unavailable(*args):
        raise OSError('Synthetic backup failure')

    monkeypatch.setattr(SQLiteReceivedStore, 'backup_to', unavailable)
    with pytest.raises(OSError, match='Synthetic backup failure'):
        SQLiteReceivedStore(old.path)
    reopened = SQLiteExplicitMemoryStore(old.path)
    assert reopened.inspect(A, item.input_id)['text'] == '原合成输入'
    with reopened.transaction(write=False) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 19
        assert not db.execute("SELECT 1 FROM sqlite_schema WHERE name='stream_received'").fetchone()
