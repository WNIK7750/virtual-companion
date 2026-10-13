"""Source age/conflict authority reaches both actual public model stages."""

import virtual_companion.text.default_runtime as module
from test_public_search_consumer import SOURCE, close, environment, finish


def test_public_generation_and_review_share_freshness_and_conflict_boundary(tmp_path):
    runtime, text, memory, mirror, model, *_ = environment(tmp_path)
    try:
        text.send(SOURCE)
        finish(runtime, text)
        initial_system = model.calls[0]['messages'][0]['content']
        assert module.PUBLIC_QUERY_ROUTING in initial_system
        stages = [row for row in model.calls
                  if row['stage'] == 'stream_unit_review'
                  or any(message['role'] == 'user'
                         and message['content'].startswith('{"public_evidence":')
                         for message in row['messages'])]
        assert any(row['stage'] == 'stream_unit_review' for row in stages)
        assert any(row['stage'] == 'demand_control' for row in stages)
        for row in stages:
            system = row['messages'][0]['content']
            assert '检索时间不证明资料当前仍有效' in system
            assert '有版本或时间冲突时保留差异' in system
            assert '不把搜索片段中的数值直接报成当前最新值' in system
    finally:
        close(runtime, text, memory, mirror)


def test_public_policy_changes_invalidate_runtime_model_resource(tmp_path, monkeypatch):
    before_path, after_path = tmp_path / 'before', tmp_path / 'after'
    before_path.mkdir()
    after_path.mkdir()
    before = environment(before_path)
    try:
        monkeypatch.setattr(module, 'PUBLIC_QUERY_ROUTING', module.PUBLIC_QUERY_ROUTING + 'changed')
        after = environment(after_path)
        try:
            assert (dict(before[0].config.resources)['model']
                    != dict(after[0].config.resources)['model'])
        finally:
            close(*after[:4])
    finally:
        close(*before[:4])
