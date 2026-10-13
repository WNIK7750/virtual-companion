export const MAX_FREEFORM_LENGTH = 2000;

export function normalizeFeedbackDraft({ quickTags = [], freeformReaction = "" } = {}) {
  if (!Array.isArray(quickTags)) throw new Error("quickTags must be an array");
  const tags = [...new Set(quickTags.filter((tag) => typeof tag === "string").map((tag) => tag.trim()).filter(Boolean))];
  const reaction = typeof freeformReaction === "string" ? freeformReaction.trim() : "";
  if (reaction.length > MAX_FREEFORM_LENGTH) throw new Error("freeform reaction is too long");
  if (/<\s*\/?\s*[A-Za-z!][^>]*>/.test(reaction)) throw new Error("freeform reaction must be plain text");
  if (!tags.length && !reaction) throw new Error("请选择标签或写下真实感受");
  return { quick_tags: tags, freeform_reaction: reaction || null };
}

export function bindFeedbackForm(root, { bridge, feedbackContext = {}, onSaved = () => {} } = {}) {
  if (!root || typeof root.querySelector !== "function") throw new Error("feedback form root is required");
  if (!bridge || typeof bridge.submit !== "function") throw new Error("feedback bridge is required");
  const form = root.querySelector("form[data-feedback-form]");
  const reaction = root.querySelector("textarea[name='freeform_reaction']");
  const status = root.querySelector("[data-feedback-status]");
  if (!form || !reaction) throw new Error("feedback form controls are missing");
  const tags = () => [...root.querySelectorAll("input[name='quick_tags']:checked")].map((item) => item.value);
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    try {
      const payload = normalizeFeedbackDraft({ quickTags: tags(), freeformReaction: reaction.value });
      const saved = await bridge.submit({ ...feedbackContext, ...payload });
      if (status) status.textContent = "已保存";
      onSaved(saved);
    } catch (error) {
      if (status) status.textContent = error instanceof Error ? error.message : "保存失败";
    }
  });
  const cancel = root.querySelector("button[data-feedback-cancel]");
  cancel?.addEventListener("click", () => {
    reaction.value = "";
    for (const checkbox of root.querySelectorAll("input[name='quick_tags']")) checkbox.checked = false;
    if (status) status.textContent = "";
  });
}
