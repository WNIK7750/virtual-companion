export const FEEDBACK_CHANNELS = Object.freeze({
  submit: "feedback.submit",
  previewExport: "feedback.previewExport",
  confirmExport: "feedback.confirmExport",
  revise: "feedback.revise",
  withdraw: "feedback.withdraw",
  delete: "feedback.delete",
  current: "feedback.current",
});

const ACTIONS = Object.freeze({
  submit: {
    method: "submitFeedback",
    fields: ["session_id", "turn_id", "assistant_message_id", "quick_tags", "freeform_reaction", "idempotency_key", "version_binding", "feedback_id"],
    required: ["session_id", "turn_id", "assistant_message_id", "quick_tags", "freeform_reaction", "idempotency_key", "version_binding"],
  },
  previewExport: { method: "previewExport", fields: ["feedback_ids", "context_policy"], required: ["feedback_ids"] },
  confirmExport: {
    method: "confirmExport",
    fields: ["preview", "redaction_confirmed", "external_processing_confirmed", "export_id"],
    required: ["preview", "redaction_confirmed", "external_processing_confirmed"],
  },
  revise: {
    method: "reviseFeedback",
    fields: ["feedback_id", "expected_revision", "quick_tags", "freeform_reaction"],
    required: ["feedback_id", "expected_revision", "quick_tags", "freeform_reaction"],
  },
  withdraw: { method: "withdrawFeedback", fields: ["feedback_id", "expected_revision"], required: ["feedback_id", "expected_revision"] },
  delete: { method: "deleteFeedback", fields: ["feedback_id"], required: ["feedback_id"] },
  current: { method: "getCurrentFeedback", fields: ["feedback_id"], required: ["feedback_id"] },
});

function isPlainObject(value) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
  const prototype = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

export function validateFeedbackPayload(action, payload) {
  const definition = ACTIONS[action];
  if (!definition) throw new Error(`unknown feedback action: ${action}`);
  if (!isPlainObject(payload)) throw new Error("feedback IPC payload must be a plain object");
  const allowed = new Set(definition.fields);
  for (const key of Object.keys(payload)) {
    if (!allowed.has(key)) throw new Error(`unsupported feedback IPC field: ${key}`);
  }
  for (const field of definition.required) {
    if (!(field in payload)) throw new Error(`required feedback IPC field is missing: ${field}`);
  }
  return Object.fromEntries(Object.entries(payload));
}

export function assertTrustedSender(event, allowedSenderUrls) {
  const senderUrl = event?.senderFrame?.url ?? event?.sender?.getURL?.();
  const allowed = Array.isArray(allowedSenderUrls) ? allowedSenderUrls : [allowedSenderUrls];
  if (typeof senderUrl !== "string" || !allowed.includes(senderUrl)) {
    throw new Error("untrusted feedback IPC sender");
  }
}

export function registerFeedbackIpc({ ipcMain, service, allowedSenderUrls }) {
  if (!ipcMain || typeof ipcMain.handle !== "function") throw new Error("ipcMain.handle is required");
  if (!service || typeof service !== "object") throw new Error("feedback service is required");
  const registered = [];
  for (const [action, definition] of Object.entries(ACTIONS)) {
    const channel = FEEDBACK_CHANNELS[action];
    const handler = async (event, payload = {}) => {
      assertTrustedSender(event, allowedSenderUrls);
      const safePayload = validateFeedbackPayload(action, payload);
      const method = service[definition.method];
      if (typeof method !== "function") throw new Error(`feedback service method is unavailable: ${definition.method}`);
      return method.call(service, safePayload);
    };
    ipcMain.handle(channel, handler);
    registered.push(channel);
  }
  return () => {
    for (const channel of registered) {
      if (typeof ipcMain.removeHandler === "function") ipcMain.removeHandler(channel);
    }
  };
}

export const FEEDBACK_ACTIONS = Object.freeze(Object.keys(ACTIONS));
