const { BrowserWindow, app, ipcMain, dialog } = require("electron");
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
const username = os.userInfo().username;
const engineName = `Adelaide Paradigm Engine`;
const consoleLogPrefix = `[${colorBrightCyan}${engineName}_${platform}_${arch}${colorReset}]:`;
const versionTheUnattendedEngineLogPrefix = `[${colorBrightCyan}${engineName}${colorBrightGreen} [Codename : "The Unattended"] ${colorReset}]:`;
const versionFeatherFeetLogPrefix = `[${colorBrightCyan}${engineName}${colorBrightRed}[Codename : "Featherfeet"]${colorReset}]:`;
const osUtil = require("os-utils");

const log = require("electron-log");
let logPathFile;
const { createLogger, transports, format } = require("winston");
require("@electron/remote/main").initialize();

let win;

function createWindow() {
  framebufferBridgeUI = new BrowserWindow({
    width: 1200,
    height: 810,
    minWidth: 780,
    minHeight: 600,
    frame: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false, //  <--  Security consideration:  Set to true and use contextBridge
      enableRemoteModule: true, //  <--  Security consideration:  Remove @electron/remote if possible
      devTools: true,
    },
    titleBarStyle: "hidden",
    icon:
      platform == "darwin"
        ? path.join(__dirname, "icon", "mac", "icon.icns")
        : path.join(__dirname, "icon", "png", "128x128.png"),
  });
  require("@electron/remote/main").enable(framebufferBridgeUI.webContents);

  framebufferBridgeUI.loadFile(path.resolve(__dirname, "src", "index.html"));
  framebufferBridgeUI.setMenu(null);
  // framebufferBridgeUI.webContents.openDevTools();

  let isIdle = false; //  <--  Unused in this simplified version

  const setFrameRate = (fps) => {
    //  <--  Unused (for now)
    framebufferBridgeUI.webContents.executeJavaScript(
      `document.body.style.setProperty('--fps', '${fps}')`
    );
  };
}

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

app.on("ready", () => {
  createWindow();
});

app.on("window-all-closed", () => {
  app.quit();
});
app.on("activate", () => {
  if (win == null) createWindow();
});
app.on("before-quit", () => {
  // No longer necessary: if (runningShell) runningShell.kill();
});

// --- IPC Event Handlers (Simplified - only keep GUI related ones) ---

// username
ipcMain.on("username", () => {
  framebufferBridgeUI.webContents.send("username", {
    data: username,
  });
});

// Assistant name
ipcMain.on("assistantName", () => {
  framebufferBridgeUI.webContents.send("assistantName", {
    data: assistantName,
  });
});

ipcMain.on("cpuUsage", () => {
  osUtil.cpuUsage(function (v) {
    framebufferBridgeUI.webContents.send("cpuUsage", { data: v });
  });
});

// OS Stats (Keep the ones useful for the GUI)

var sysThreads = osUtils.cpuCount(); //  <--  Consider moving to a separate 'systemInfo' module
var threads; //  <--  Consider moving or removing if not essential for GUI
for (let i = 1; i < sysThreads; i = i * 2) {
  threads = i;
}
if (sysThreads == 4) {
  threads = 4;
}

ipcMain.on("cpuFree", () => {
  osUtil.cpuFree(function (v) {
    framebufferBridgeUI.webContents.send("cpuFree", { data: v });
  });
});

ipcMain.on("cpuCount", () => {
  const totalAvailableThreads = osUtil.cpuCount(); // Use a descriptive name
  framebufferBridgeUI.webContents.send("cpuCount", {
    data: totalAvailableThreads,
  });
});

ipcMain.on("threadUtilized", () => {
  //  <--  Keep if you display thread usage
  framebufferBridgeUI.webContents.send("threadUtilized", {
    data: threads,
  });
});

ipcMain.on("freemem", () => {
  framebufferBridgeUI.webContents.send("freemem", {
    data: Math.round(osUtil.freemem() / 102.4) / 10,
  });
});

ipcMain.on("totalmem", () => {
  framebufferBridgeUI.webContents.send("totalmem", {
    data: osUtil.totalmem(),
  });
});

ipcMain.on("os", () => {
  framebufferBridgeUI.webContents.send("os", {
    data: platform,
  });
});

// --- Removed LLM-related code (Examples) ---
// Removed:  All LLM model loading, spawning, interaction, RAG, Backbrain, etc.
// Removed:  All store.get("params") calls and related parameter handling.
// Removed:  All file system operations related to models, training, etc.
// Removed:  All complex logic related to internal thought processing.
// Removed:  All code related to external libraries like `compromise`, `axios`, `duck-duck-scrape`, etc.
// Removed:  The `runningShell` variable and related pty process.

// ---  Example of how to handle a simplified "message" event ---

ipcMain.on("message", (_event, { data }) => {
  // This is a simplified example.  You can send data *back* to the
  // renderer process, but there's no LLM to process the message.
  console.log("Received message from renderer:", data);

  // Example of sending a response back to the UI (immediately):
  framebufferBridgeUI.webContents.send("result", {
    data: "Message received. (GUI only, no LLM)",
  });
});

ipcMain.on("stopGeneration", () => {
  // Nothing to do here now, as there is no generation process.
  console.log("Stop generation request (no LLM to stop).");
  framebufferBridgeUI.webContents.send("result", {
    data: "\n\n<end>", //  <--  You might still want to send this to signal completion
  });
});

// --- Placeholder for potential future GUI-related IPC events ---
// Add more IPC event handlers here as needed for your GUI interactions,
// e.g., handling button clicks, file selection (if still relevant), etc.

ipcMain.on("pickFile", () => {
  //  <--  Keep if your GUI still has file selection
  dialog
    .showOpenDialog(win, {
      title: "Choose a File (Placeholder)", // Update title
      // Removed model-specific filters
      properties: ["dontAddToRecent", "openFile"],
    })
    .then((obj) => {
      if (!obj.canceled) {
        framebufferBridgeUI.webContents.send("pickedFile", {
          data: obj.filePaths[0],
        });
      }
    });
});

// ---  Removed signal handlers (since there's no LLM) ---

// Removed: All signal handlers (`process.on(...)`).  They were primarily
// there to handle unhandled rejections/exceptions from the LLM interaction,
// which is no longer present.
