import { contextBridge, ipcRenderer } from "electron";

const channels = Object.freeze({
  submit: "feedback.submit",
  previewExport: "feedback.previewExport",
  confirmExport: "feedback.confirmExport",
  revise: "feedback.revise",
  withdraw: "feedback.withdraw",
  delete: "feedback.delete",
  current: "feedback.current",
});

const feedbackBridge = Object.freeze({
  submit: (payload) => ipcRenderer.invoke(channels.submit, payload),
  previewExport: (payload) => ipcRenderer.invoke(channels.previewExport, payload),
  confirmExport: (payload) => ipcRenderer.invoke(channels.confirmExport, payload),
  revise: (payload) => ipcRenderer.invoke(channels.revise, payload),
  withdraw: (payload) => ipcRenderer.invoke(channels.withdraw, payload),
  delete: (payload) => ipcRenderer.invoke(channels.delete, payload),
  current: (payload) => ipcRenderer.invoke(channels.current, payload),
});

contextBridge.exposeInMainWorld("feedback", feedbackBridge);
