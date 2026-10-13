from virtual_companion.boundary.presentation import semantic_bubble_ranges
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def test_semantic_bubbles_preserve_paragraphs_and_unicode_without_sentence_splitting():
    source = '第一句🌿。第二句。\n\n换一个自然段。'
    ranges = semantic_bubble_ranges(source)
    assert [source[r['start']:r['end']] for r in ranges] == [
        '第一句🌿。第二句。\n\n', '换一个自然段。']
    assert semantic_bubble_ranges('第一句。第二句！') == [{'start': 0, 'end': 8}]


def test_approved_bubbles_bind_source_ranges_but_legacy_manifest_stays_unchanged(tmp_path):
    from test_faithful_stream_units import spanned
    from test_rebuilt_turns import A, claim, config
    from test_segment_store import reviewed
    from virtual_companion.boundary.sqlite_segments import SQLiteSegmentStore

    store = SQLiteSegmentStore(tmp_path / 'bubbles.sqlite3')
    run = claim(store, store.accept(config(), 'faithful', '合成输入'))
    source = '第一段🌿。\n\n第二段。'
    approved = store.approve_segment(run, spanned(run, source, [len(source)]), now_us=100)
    assert approved['bubbles'] == semantic_bubble_ranges(source)
    assert store.segments(A, run.input_id)[0]['bubbles'] == approved['bubbles']
    legacy = SQLiteSegmentStore(tmp_path / 'legacy.sqlite3')
    legacy_run = claim(legacy, legacy.accept(config(), 'joined', '合成输入'))
    assert 'bubbles' not in legacy.approve_segment(legacy_run, reviewed(legacy_run), now_us=100)


def test_confirmed_bubble_ranges_persist_on_same_message_and_legacy_stays_single(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    attempt = store.begin('合成输入', {})
    first = '第一段。\n\n'
    full = first + '第二段🌿。'
    store.reconcile_confirmed(attempt['turn_id'], first, completed=False,
                              bubble_ranges=semantic_bubble_ranges(first))
    message_id = store.history()['messages'][-1]['id']
    store.reconcile_confirmed(attempt['turn_id'], full, completed=True,
                              bubble_ranges=semantic_bubble_ranges(full))
    store.finish(attempt['attempt_id'], 'received', '')
    store.close()
    reopened = SQLiteConversationStore(tmp_path)
    message = reopened.history()['messages'][-1]
    assert message['id'] == message_id and message['content'] == full
    assert message['bubble_ranges'] == semantic_bubble_ranges(full)
    assert reopened.history_page()['messages'][-1]['bubble_ranges'] == message['bubble_ranges']
    legacy = reopened.begin('旧形状输入', {})
    reopened.finish(legacy['attempt_id'], 'completed', '旧消息。\n\n仍为单泡。')
    assert 'bubble_ranges' not in reopened.history()['messages'][-1]
    reopened.close()


def test_later_bubble_keeps_its_arrival_after_an_interleaved_user_input_on_reopen(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    first = store.begin('第一条输入', {})
    prefix = '先说这一段。\n\n'
    full = prefix + '接着说这一段。'
    earlier = store.reserve_bubbles(first['turn_id'], semantic_bubble_ranges(prefix))
    store.reconcile_confirmed(first['turn_id'], prefix, completed=False,
                              bubble_ranges=semantic_bubble_ranges(prefix))
    second = store.enqueue_input('在两段之间插话', {})
    later = store.reserve_bubbles(first['turn_id'], semantic_bubble_ranges(full))
    assert later[0] == earlier[0]
    assert store.reserve_bubbles(first['turn_id'], semantic_bubble_ranges(full)) == later
    store.reconcile_confirmed(first['turn_id'], full, completed=False,
                              bubble_ranges=semantic_bubble_ranges(full))
    store.finish(first['attempt_id'], 'received', '')
    store.close()
    store = SQLiteConversationStore(tmp_path)
    history = store.history()['messages']
    assistant = next(m for m in history if m['role'] == 'assistant')
    user = next(m for m in history if m['turn_id'] == second['turn_id'])
    assert assistant['bubble_orders'] == later
    assert later[0]['order'] < user['timeline_order'] < later[1]['order']
    store.close()
