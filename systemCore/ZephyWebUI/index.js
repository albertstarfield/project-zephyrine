const path = require("path");
const os = require("os");
const osUtils = require("os-utils");
const platform = os.platform();
const arch = os.arch();
const colorReset = "\x1b[0m";
const colorBrightCyan = "\x1b[96m";
const colorBrightRed = "\x1b[91m";
const colorBrightGreen = "\x1b[92m";
const assistantName = "Adelaide Zephyrine Charlotte";
const appName = `Project ${assistantName}`;
const engineName = `Adelaide Paradigm Engine`;
const consoleLogPrefix = `[${colorBrightCyan}${engineName}_${platform}_${arch}${colorReset}]:`;
const versionTheUnattendedEngineLogPrefix = `[${colorBrightCyan}${engineName}${colorBrightGreen} [Codename : "The Unattended"] ${colorReset}]:`;
const versionFeatherFeetLogPrefix = `[${colorBrightCyan}${engineName}${colorBrightRed}[Codename : "Featherfeet"]${colorReset}]:`;

let logPathFile;
const { createLogger, transports, format } = require("winston");


// Log Configuration  (Consider moving to a separate module)
logPathFile = path.resolve(__dirname, "AdelaideRuntimeCore.log");
log.info(logPathFile);
log.transports.file.file = logPathFile;
log.transports.file.level = "debug";

const logger = createLogger({
  level: "info",
  format: format.combine(
    format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
    format.printf((info) => `${info.timestamp} ${info.level}: ${info.message}`)
  ),
  transports: [new transports.File({ filename: logPathFile, level: "info" })],
});

//custom icon for the dock on darwin based os using QE/CI engine (Consider moving to a separate module if needed)
if (os.platform() == "darwin") {
  const electron = require("electron");
  const app = electron.app;
  const image = electron.nativeImage.createFromPath(
    app.getAppPath() + "/" + path.join("icon", "mac", "icon.icns")
  );
  app.dock.setIcon(image);
  app.name = assistantName;
}

function delay(ms) {
  //  Utility function - consider moving
  return new Promise((resolve) => setTimeout(resolve, ms));
}

app.on("second-instance", () => {
  if (win) {
    if (framebufferBridgeUI.isMinimized()) framebufferBridgeUI.restore();
    framebufferBridgeUI.focus();
  }
});


app.on("activate", () => {
  if (win == null) createWindow();
});
app.on("before-quit", () => {
  // No longer necessary: if (runningShell) runningShell.kill();
});

// --- IPC Event Handlers (Simplified - only keep GUI related ones) ---

// username


// OS Stats (Keep the ones useful for the GUI)

var sysThreads = osUtils.cpuCount(); //  <--  Consider moving to a separate 'systemInfo' module
var threads; //  <--  Consider moving or removing if not essential for GUI
for (let i = 1; i < sysThreads; i = i * 2) {
  threads = i;
}
if (sysThreads == 4) {
  threads = 4;
}



// --- Removed LLM-related code (Examples) ---
// Removed:  All LLM model loading, spawning, interaction, RAG, Backbrain, etc.
// Removed:  All store.get("params") calls and related parameter handling.
// Removed:  All file system operations related to models, training, etc.
// Removed:  All complex logic related to internal thought processing.
// Removed:  All code related to external libraries like `compromise`, `axios`, `duck-duck-scrape`, etc.
// Removed:  The `runningShell` variable and related pty process.

// ---  Example of how to handle a simplified "message" event ---



// --- Placeholder for potential future GUI-related IPC events ---
// Add more IPC event handlers here as needed for your GUI interactions,
// e.g., handling button clicks, file selection (if still relevant), etc.



// ---  Removed signal handlers (since there's no LLM) ---

// Removed: All signal handlers (`process.on(...)`).  They were primarily
// there to handle unhandled rejections/exceptions from the LLM interaction,
// which is no longer present.
