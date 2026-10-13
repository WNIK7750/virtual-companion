const { contextBridge, ipcRenderer } = require('electron');
const invoke = async (action, payload = {}) => {
  const result = await ipcRenderer.invoke('companion:action', action, payload);
  if (!result.ok) throw new Error(result.error);
  return result.value;
};
contextBridge.exposeInMainWorld('companion', Object.freeze({
  state: () => invoke('state'), start: () => invoke('start'),
  presentApproved: () => invoke('presentApproved'),
  resumeApproved: fingerprint => invoke('resumeApproved', { fingerprint }),
  cancelRecovery: () => invoke('cancelRecovery'),
  onNotification: callback => {
    const listener = (_event, target) => callback(target);
    ipcRenderer.on('companion:notification-target', listener);
    return () => ipcRenderer.removeListener('companion:notification-target', listener);
  },
  onApproved: callback => {
    const listener = async (_event, payload) => {
      try {
        if (await callback(payload) === true) ipcRenderer.send('companion:stream-rendered', {
          request_id: payload.request_id, fingerprint: payload.fingerprint, visible: true,
        });
      } catch { /* Unknown presentation remains unacknowledged. */ }
    };
    ipcRenderer.on('companion:stream-unit', listener);
    return () => ipcRenderer.removeListener('companion:stream-unit', listener);
  },
  send: text => invoke('send', { text }), stop: () => invoke('stop'),
  resumeInputs: () => invoke('resumeInputs'),
  setQuiet: quiet => invoke('setQuiet', { quiet }),
  retry: () => invoke('retry'), history: (window = {}) => invoke('history', window), newSession: () => invoke('newSession'),
  selectProvider: (provider_id, confirmed = false) => invoke('selectProvider', { provider_id, confirmed }),
  feedback: payload => invoke('feedback', payload),
  memories: (window = {}) => invoke('memories', window),
  memoryDetails: record_id => invoke('memoryDetails', { record_id }),
  memoryAction: payload => invoke('memoryAction', payload),
  memoryAbandon: payload => invoke('memoryAbandon', payload),
  sessions: (offset = 0) => invoke('sessions', { offset }), selectSession: session_id => invoke('selectSession', { session_id }),
}));
