import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import assert from "node:assert/strict";

import {
  FEEDBACK_ACTIONS,
  FEEDBACK_CHANNELS,
  assertTrustedSender,
  registerFeedbackIpc,
  validateFeedbackPayload,
} from "../src/feedback-ipc.mjs";
import { normalizeFeedbackDraft } from "../src/feedback-ui.mjs";
import {
  FEEDBACK_CONTENT_SECURITY_POLICY,
  createFeedbackWindowPreferences,
  isAllowedNavigation,
  isAllowedNewWindow,
} from "../src/feedback-window-security.mjs";

class FakeIpcMain {
  constructor() {
    this.handlers = new Map();
  }

  handle(channel, handler) {
    this.handlers.set(channel, handler);
  }

  removeHandler(channel) {
    this.handlers.delete(channel);
  }
}

test("registers only the narrow feedback channels and validates sender", async () => {
  const ipcMain = new FakeIpcMain();
  const calls = [];
  const service = { submitFeedback: async (payload) => { calls.push(payload); return { ok: true }; } };
  const dispose = registerFeedbackIpc({ ipcMain, service, allowedSenderUrls: ["app://virtual-companion"] });

  assert.deepEqual([...ipcMain.handlers.keys()], FEEDBACK_ACTIONS.map((action) => FEEDBACK_CHANNELS[action]));
  const handler = ipcMain.handlers.get(FEEDBACK_CHANNELS.submit);
  assert.ok(handler);
  const result = await handler(
    { senderFrame: { url: "app://virtual-companion" } },
    {
      session_id: "session-1",
      turn_id: "turn-1",
      assistant_message_id: "assistant-1",
      quick_tags: ["too_mechanical"],
      freeform_reaction: null,
      idempotency_key: "idem-1",
      version_binding: { schema_version: 1 },
    },
  );
  assert.deepEqual(result, { ok: true });
  assert.equal(calls.length, 1);
  await assert.rejects(
    handler(
      { senderFrame: { url: "https://evil.invalid" } },
      {
        session_id: "session-1",
        turn_id: "turn-1",
        assistant_message_id: "assistant-1",
        quick_tags: ["x"],
        freeform_reaction: null,
        idempotency_key: "idem-2",
        version_binding: { schema_version: 1 },
      },
    ),
    /untrusted/,
  );
  dispose();
  assert.equal(ipcMain.handlers.size, 0);
});

test("rejects non-plain payloads and unknown fields", () => {
  assert.throws(
    () => validateFeedbackPayload("submit", { quick_tags: [], unexpected: true }),
    /unsupported feedback IPC field/,
  );
  assert.throws(() => validateFeedbackPayload("submit", []), /plain object/);
  assert.throws(
    () => assertTrustedSender({ senderFrame: { url: "file:///tmp/app.html" } }, ["app://virtual-companion"]),
    /untrusted/,
  );
  assert.throws(
    () => validateFeedbackPayload("submit", { quick_tags: ["x"] }),
    /required feedback IPC field/,
  );
  assert.throws(
    () => validateFeedbackPayload("confirmExport", {
      preview: {},
      redaction_confirmed: true,
      external_processing_confirmed: true,
      destination: "C:\\outside\\chosen.json",
    }),
    /unsupported feedback IPC field: destination/,
  );
});

test("feedback draft keeps quick tags optional but rejects an empty submission", () => {
  assert.deepEqual(
    normalizeFeedbackDraft({ quickTags: [], freeformReaction: "  有一点机械  " }),
    { quick_tags: [], freeform_reaction: "有一点机械" },
  );
  assert.throws(() => normalizeFeedbackDraft(), /真实感受/);
  assert.throws(() => normalizeFeedbackDraft({ freeformReaction: "x".repeat(2001) }), /too long/);
  assert.throws(() => normalizeFeedbackDraft({ freeformReaction: "<script>alert(1)</script>" }), /plain text/);
});

test("preload and HTML expose no general renderer capability", async () => {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const preload = await readFile(resolve(root, "src/feedback-preload.mjs"), "utf8");
  const html = await readFile(resolve(root, "src/feedback-form.html"), "utf8");
  assert.match(preload, /contextBridge\.exposeInMainWorld\("feedback"/);
  assert.doesNotMatch(preload, /ipcRenderer\.(send|on|once|sendSync)/);
  assert.match(html, /其他（写下你的真实感受）/);
  assert.doesNotMatch(html, /orthers/);
});

test("window security defaults keep the renderer isolated and navigation closed", () => {
  assert.deepEqual(createFeedbackWindowPreferences("/absolute/preload.mjs"), {
    preload: "/absolute/preload.mjs",
    nodeIntegration: false,
    contextIsolation: true,
    sandbox: true,
    webSecurity: true,
    webviewTag: false,
    allowRunningInsecureContent: false,
  });
  assert.equal(isAllowedNavigation("app://virtual-companion"), true);
  assert.equal(isAllowedNavigation("https://evil.invalid"), false);
  assert.equal(isAllowedNewWindow(), false);
  assert.match(FEEDBACK_CONTENT_SECURITY_POLICY, /connect-src 'none'/);
  assert.match(FEEDBACK_CONTENT_SECURITY_POLICY, /object-src 'none'/);
});
