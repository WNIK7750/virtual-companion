import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Thread

import pytest
from test_rebuilt_turns import A
from test_stream_history import setup


def test_python_approval_to_node_device_receipt_to_confirmed_history(tmp_path):
    store, run, event = setup(tmp_path)
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
    module = (Path(__file__).resolve().parents[2] / "apps/desktop/src/stream-receipts.mjs").as_uri()
    script = (
        "import { readFileSync } from 'node:fs';\n"
        f"import {{ StreamReceiptJournal }} from {json.dumps(module)};\n"
        "const event = JSON.parse(readFileSync(process.argv[1], 'utf8'));\n"
        "const journal = new StreamReceiptJournal(process.argv[2]);\n"
        "const receipt = await journal.present(event, async value => value.text === event.text);\n"
        "console.log(JSON.stringify({ receipt, state: journal.lookup(event) }));\n"
    )
    result = subprocess.run(  # noqa: S603 -- fixed local Node program and synthetic paths
        [
            shutil.which("node"),
            "--input-type=module",
            "-e",
            script,
            str(event_path),
            str(tmp_path / "device"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=20,
    )
    receipt = json.loads(result.stdout)
    assert receipt["state"]["status"] == "confirmed"
    store.confirm_segment(
        A,
        run.input_id,
        1,
        fingerprint=event["fingerprint"],
        host_receipt=receipt["receipt"],
        channel="text_visible",
    )
    store.finish_stream(run, outcome="completed", now_us=100)
    assert store.stream_state(A, run.input_id)["confirmed_text"] == "我在这里。"


@pytest.mark.parametrize("lost_ack", ["none", "before", "after"])
def test_authenticated_http_device_receipt_reconciliation(tmp_path, lost_ack):
    from virtual_companion.text.stream_host import StreamHostBridge, create_stream_server

    store, run, event = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=10)
    token = "synthetic-test-token-" + "x" * 32
    server = create_stream_server(token, bridge)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = Path(__file__).resolve().parents[2] / "apps/desktop/src"
    journal_uri = json.dumps((root / "stream-receipts.mjs").as_uri())
    client_uri = json.dumps((root / "stream-delivery-client.mjs").as_uri())
    script = (
        f"import {{ StreamReceiptJournal }} from {journal_uri};\n"
        f"import {{ StreamDeliveryClient }} from {client_uri};\n"
        "const [url, token, directory, lost] = process.argv.slice(1);\n"
        "let renders = 0, first = true;\n"
        "const journal = new StreamReceiptJournal(directory);\n"
        "const call = async (action, payload) => {\n"
        "  const drop = action === 'streamAck' && first && lost !== 'none';\n"
        "  if (action === 'streamAck') first = false;\n"
        "  if (drop && lost === 'before') throw Error('ack request lost');\n"
        "  const response = await fetch(url + '/v1/' + action, {method:'POST',\n"
        "    headers:{Authorization:'Bearer ' + token, 'Content-Type':'application/json'},\n"
        "    body:JSON.stringify(payload), signal:AbortSignal.timeout(5000)});\n"
        "  if (!response.ok) throw Error('HTTP ' + response.status);\n"
        "  const result = await response.json();\n"
        "  if (drop) throw Error('ack response lost');\n"
        "  return result;\n"
        "};\n"
        "const presenter = {present:event => journal.present(event,\n"
        "  async () => {renders++; return true;}),\n"
        "  reconcile:event => journal.lookup(event)};\n"
        "const client = new StreamDeliveryClient({call, presenter});\n"
        "let error = null; try {await client.pollOnce();} catch (e) {error = e.message;}\n"
        "const reconciled = await client.reconcile();\n"
        "await client.reconcile();\n"
        "console.log(JSON.stringify({renders, error, reconciled,\n"
        "  state:await call('streamState', {})}));\n"
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            waiting = pool.submit(bridge.present, event)
            assert bridge.ready.wait(2)
            result = subprocess.run(  # noqa: S603 -- fixed Node program and synthetic loopback fixture
                [
                    shutil.which("node"),
                    "--input-type=module",
                    "-e",
                    script,
                    f"http://127.0.0.1:{server.server_port}",
                    token,
                    str(tmp_path / "device"),
                    lost_ack,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=True,
                timeout=15,
            )
            output = json.loads(result.stdout)
            assert waiting.result(2).startswith("stream-visible-")
        assert output["renders"] == 1
        assert (output["error"] is None) == (lost_ack == "none")
        assert output["reconciled"]["confirmed"] == (1 if lost_ack == "before" else 0)
        assert output["state"]["state"]["confirmed_units"] == 1
        assert output["state"]["reconcile"] == []
        store.finish_stream(run, outcome="completed", now_us=100)
        assert store.stream_state(A, run.input_id)["confirmed_text"] == "我在这里。"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
