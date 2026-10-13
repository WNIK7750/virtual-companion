export const FEEDBACK_CONTENT_SECURITY_POLICY =
  "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
  + "connect-src 'none'; object-src 'none'; base-uri 'none'; frame-ancestors 'none';";

export function createFeedbackWindowPreferences(preloadPath) {
  if (typeof preloadPath !== "string" || !preloadPath) throw new Error("preloadPath is required");
  return Object.freeze({
    preload: preloadPath,
    nodeIntegration: false,
    contextIsolation: true,
    sandbox: true,
    webSecurity: true,
    webviewTag: false,
    allowRunningInsecureContent: false,
  });
}

export function isAllowedNavigation(url, allowedOrigin = "app://virtual-companion") {
  return typeof url === "string" && url === allowedOrigin;
}

export function isAllowedNewWindow() {
  return false;
}
