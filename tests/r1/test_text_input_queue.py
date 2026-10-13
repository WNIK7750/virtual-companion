import pytest
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def test_received_inputs_survive_reopen_in_order_without_replacing_active_attempt(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'chat')
    current = store.begin('当前表达', {})
    binding = {'role': {'version': 'original'}}
    first = store.enqueue_input('补充一', binding)
    binding['role']['version'] = 'changed'
    second = store.enqueue_input('补充二', binding)
    assert store.latest_attempt() == current
    assert [value['content'] for value in store.history()['messages']] == [
        '当前表达', '补充一', '补充二']
    with pytest.raises(ValueError, match='Unfinalized'):
        store.begin_queued(first['turn_id'])
    store.close()
    store = SQLiteConversationStore(tmp_path / 'chat')
    assert store.queued_inputs() == [first, second]
    assert store.latest_attempt()['status'] == 'interrupted'
    with pytest.raises(ValueError, match='Oldest'):
        store.begin_queued(second['turn_id'])
    active = store.begin_queued(first['turn_id'])
    assert active['attempt_id'] == first['attempt_id']
    assert active['binding'] == {'role': {'version': 'original'}}
    assert store.queued_inputs() == [second]
    store.finish(active['attempt_id'], 'listened', '')
    active = store.begin_queued(second['turn_id'])
    assert active['attempt_id'] == second['attempt_id']
    assert len(store.history()['messages']) == 3
    store.close()


def test_queue_pause_is_durable_and_scoped_to_selected_conversation(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'chat')
    original = store.session_id
    received = store.enqueue_input('已收到', {})
    store.pause_input_queue(True)
    store.new_session()
    assert not store.input_queue_paused() and not store.queued_inputs()
    store.select_session(original)
    assert store.input_queue_paused() and store.queued_inputs() == [received]
    store.close()
    store = SQLiteConversationStore(tmp_path / 'chat')
    assert store.input_queue_paused() and store.queued_inputs() == [received]
    store.pause_input_queue(False)
    assert not store.input_queue_paused()
    store.close()


def test_rejected_queue_write_cannot_leave_a_user_message(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'chat')
    with pytest.raises(ValueError, match='capacity'):
        store.enqueue_input('不能半写入', {'oversized': 'x' * 131073})
    assert store.history()['messages'] == [] and store.queued_inputs() == []
    store.close()
