const { BrowserWindow, app, ipcMain, dialog } = require("electron");
const path = require("path");
const os = require("os");
const platform = os.platform();

// --- Global Variables (Keep only essential ones) ---
let win;
let framebufferBridgeUI;

// --- Helper Functions (Keep only if directly related to UI/Main process setup) ---
// (None from the original code are strictly necessary here, so I'm not including any.
//  You can add back helper functions *unrelated* to the LLM if needed.)

// --- Window Creation ---

function createWindow() {
  framebufferBridgeUI = new BrowserWindow({
    width: 1200,
    height: 810,
    minWidth: 780,
    minHeight: 600,
    frame: false, // Keep frameless if you want
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true,
      devTools: true, // Keep for development
    },
    titleBarStyle: "hidden",
    icon:
      platform == "darwin"
        ? path.join(__dirname, "icon", "mac", "icon.icns")
        : path.join(__dirname, "icon", "png", "128x128.png"),
  });

  require("@electron/remote/main").initialize(); // Add this line
  require("@electron/remote/main").enable(framebufferBridgeUI.webContents);

  framebufferBridgeUI.loadFile(path.resolve(__dirname, "src", "index.html"));
  framebufferBridgeUI.setMenu(null); // Keep no menu
  // framebufferBridgeUI.webContents.openDevTools(); // Keep or remove
}

// --- App Event Handlers ---

app.on("ready", () => {
  createWindow();
});

app.on("window-all-closed", () => {
  app.quit();
});

app.on("activate", () => {
  if (win == null) createWindow();
});

// --- IPC Event Handlers (UI Interaction) ---

// --- Basic Examples (from previous response) ---

ipcMain.on("username", () => {
  framebufferBridgeUI.webContents.send("username", {
    data: os.userInfo().username,
  });
});

ipcMain.on("os", () => {
  framebufferBridgeUI.webContents.send("os", {
    data: platform,
  });
});

ipcMain.on("restart", () => {
  app.relaunch();
  app.exit();
});

// --- UI Interaction Examples (Placeholders) ---

// Example: User clicks a button in the UI.
ipcMain.on("button-click", (_event, data) => {
  console.log("Main process received button click:", data);

  // Do something in the main process (e.g., access the file system,
  //  make a network request – but NOT related to the LLM).

  // Send a response back to the renderer:
  framebufferBridgeUI.webContents.send("button-click-response", {
    message: "Button click handled in main process!",
  });
});

// Example: User submits a form in the UI.
ipcMain.on("form-submit", (_event, formData) => {
  console.log("Main process received form data:", formData);

  // Process the form data (e.g., save to a file, validate input).
  //  Again, *no LLM interaction*.

  // Send data back to the renderer (e.g., to update the UI):
  framebufferBridgeUI.webContents.send("form-submit-response", {
    success: true,
    message: "Form submitted successfully!",
  });
});

// Example: Send a message to the renderer to display.
// (This would be triggered by something in the main process, not directly by the UI).
function sendMessageToUI(message) {
  framebufferBridgeUI.webContents.send("display-message", { text: message });
}

// Example usage (you'd call this from within an ipcMain.on handler, or after
//  some other main-process event):
// sendMessageToUI("This is a message from the main process!");

// --- File Dialog Example ---

ipcMain.on("open-file-dialog", (event) => {
  dialog
    .showOpenDialog(framebufferBridgeUI, {
      properties: ["openFile"],
      filters: [
        { name: "Text Files", extensions: ["txt"] },
        { name: "All Files", extensions: ["*"] },
      ],
    })
    .then((result) => {
      if (!result.canceled && result.filePaths.length > 0) {
        // Send the file path back to the renderer.
        event.sender.send("selected-file", result.filePaths[0]);
      }
    })
    .catch((err) => {
      console.log(err);
    });
});

// --- Close Window Example ---
ipcMain.on("close-window", () => {
  if (framebufferBridgeUI && !framebufferBridgeUI.isDestroyed()) {
    framebufferBridgeUI.close();
  }
});

// --- Minimize Window Example ---
ipcMain.on("minimize-window", () => {
  if (framebufferBridgeUI) {
    framebufferBridgeUI.minimize();
  }
});
// --- Maximize-Restore Window Example ---
ipcMain.on("maximize-restore-window", () => {
  if (framebufferBridgeUI) {
    if (framebufferBridgeUI.isMaximized()) {
      framebufferBridgeUI.restore(); // Unmaximize if maximized
    } else {
      framebufferBridgeUI.maximize();
    }
  }
});
