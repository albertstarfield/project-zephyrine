/*
DEBUGGING TIPS
For enabling the more verbose output of the console logs you could add this environment variable and set to "1"

INTERNET_FETCH_DEBUG_MODE
LOCAL_FETCH_DEBUG_MODE
CoTMultiSteps_FETCH_DEBUG_MODE
ptyinteractionStgDEBUG
ptyStreamDEBUGMode
HISTORY_CTX_FETCH_DEBUG_MODE

*/
//https://stackoverflow.com/questions/72591633/electron-simplest-example-to-pass-variable-from-js-to-html-using-ipc-contextb
/**
 * Render --> Main
 * ---------------
 * Render:  window.ipcRender.send('channel', data); // Data is optional.
 * Main:    electronIpcMain.on('channel', (event, data) => { methodName(data); })
 *
 * Main --> Render
 * ---------------
 * Main:    windowName.webContents.send('channel', data); // Data is optional.
 * Render:  window.ipcRender.receive('channel', (data) => { methodName(data); });
 *
 * Render --> Main (Value) --> Render
 * ----------------------------------
 * Render:  window.ipcRender.invoke('channel', data).then((result) => { methodName(result); });
 * Main:    electronIpcMain.handle('channel', (event, data) => { return someMethod(data); });
 *
 * Render --> Main (Promise) --> Render
 * ------------------------------------
 * Render:  window.ipcRender.invoke('channel', data).then((result) => { methodName(result); });
 * Main:    electronIpcMain.handle('channel', async (event, data) => {
 *              return await promiseName(data)
 *                  .then(() => { return result; })
 *          });
 */


const { BrowserWindow, app, ipcMain, dialog } = require("electron");
const { spawn } = require('child_process');
const ipcRenderer = require("electron").ipcRenderer;
const contextBridge = require('electron').contextBridge;
const path = require("path");
//const { download } = require('electron-dl');
const https = require('https');
const http = require('http');
require("@electron/remote/main").initialize();4
const os = require("os");
const osUtils = require('os-utils');
const gpuInfo = require('gpu-info');
const platform = os.platform();
const arch = os.arch();
const colorReset = "\x1b[0m";
const colorBrightCyan = "\x1b[96m";
const assistantName = "Adelaide Zephyrine Charlotte";
const natural = require('natural');
const appName = `Project ${assistantName}`;
const username = os.userInfo().username;
const consoleLogPrefix = `[${colorBrightCyan}${appName}_${platform}_${arch}${colorReset}]:`;



var win;
function createWindow() {
	win = new BrowserWindow({
		width: 1200,
		height: 810,
		minWidth: 780,
		minHeight: 600,
		frame: false,
		webPreferences: {
			nodeIntegration: true,
			contextIsolation: false,
			enableRemoteModule: true,
			devTools: true
		},
		titleBarStyle: "hidden",
		icon: platform == "darwin" ? path.join(__dirname, "icon", "mac", "icon.icns") : path.join(__dirname, "icon", "png", "128x128.png")
	});
	require("@electron/remote/main").enable(win.webContents);

	win.loadFile(path.resolve(__dirname, "src", "index.html"));
	// after main window and main process initialize the electron core send the global.username and global.assistantName to the global bot
	win.setMenu(null);
	// win.webContents.openDevTools();
	
}
//custom icon for the dock on darwin based os using QE/CI engine
if (os.platform() == "darwin"){
const electron = require('electron'); // Import the electron module
const app = electron.app;
const image = electron.nativeImage.createFromPath(
  app.getAppPath() + "/" + path.join("icon", "mac", "icon.icns")
);
app.dock.setIcon(image);
app.name = assistantName;
}


function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

app.on("second-instance", () => {
	if (win) {
		if (win.isMinimized()) win.restore();
		win.focus();
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
	if (runningShell) runningShell.kill();
});

// OS STATS

const osUtil = require("os-utils");
var threads;
var sysThreads = osUtil.cpuCount();
for (let i = 1; i < sysThreads; i = i * 2) {
	threads = i;
}
if (sysThreads == 4) {
	threads = 4;
}

// username
ipcMain.on("username", () => {
	win.webContents.send("username", {
		data: username
	});
});
// Assistant name is one of the operating system part
ipcMain.on("assistantName", () => {
	win.webContents.send("assistantName", {
		data: assistantName
	});
});

ipcMain.on("cpuUsage", () => {
	osUtil.cpuUsage(function (v) {
		win.webContents.send("cpuUsage", { data: v });
	});
});

ipcMain.on("UMACheckUsage", () => {
	win.webContents.send("UMAAllocSizeStatisticsGB", { data: UMAGBSize });
});

ipcMain.on("BackBrainQueueCheck", () => {
	const BackBrainQueueLength = BackBrainQueue.length;
	win.webContents.send("BackBrainQueueCheck_render", { data: BackBrainQueueLength });
});

ipcMain.on("BackBrainQueueResultCheck", () => {
	const BackBrainResultQueueLength = BackBrainResultQueue.length;
	win.webContents.send("BackBrainQueueResultCheck_render", { data: BackBrainResultQueueLength });
});

// Function to gather system information
function getSystemInfo() {
    let systemInfoStr = '';

    // Gather basic OS information
    const osInfo = {
        platform: os.platform(),
        release: os.release(),
        architecture: os.arch(),
        hostname: os.hostname(),
        totalMemory: os.totalmem(),
        freeMemory: os.freemem(),
        cpuModel: os.cpus()[0].model,
        cpuSpeed: os.cpus()[0].speed,
        cpuCores: os.cpus().length
    };

    // Concatenate basic OS information
    systemInfoStr += `
        System Backplate Information:
		🏝️ Platform: ${osInfo.platform}
        📦 Release: ${osInfo.release}
        🎨 Arch: ${osInfo.architecture}
        🚚 Host: ${osInfo.hostname}
        💾 TotalMem: ${Math.round(osInfo.totalMemory / (1024 * 1024 * 1024))} GB
        💾 FreeMem: ${Math.round(osInfo.freeMemory / (1024 * 1024 * 1024))} GB
        ⚙️ SoC: ${osInfo.cpuModel}
        ⏱️ CPU Hz: ${osInfo.cpuSpeed / 10} GHz
        🤹‍♂️ Cores: ${osInfo.cpuCores}
    `;
    // If not Windows, gather CPU usage information
    if (osInfo.platform !== 'win32') {
        const cpuUsage = osUtils.cpuUsage();
        systemInfoStr += `Usage: ${Math.round(cpuUsage * 100)}%`;
    }

    // If not Windows, gather GPU information
	/*
    if (osInfo.platform !== 'win32') {
        const gpuInfoStr = gpuInfo.get().map(gpu => {
            return `GPU ${gpu.index + 1}: ${gpu.name}`;
        }).join('\n');

        // Concatenate GPU information
        systemInfoStr += `\n${gpuInfoStr}\n`;
    }
	*/
    return systemInfoStr;
}

ipcMain.on("SystemBackplateHotplugCheck", () => {
	const recievedDataIndexJSGetSystemInfo = getSystemInfo();
	win.webContents.send("systemBackPlateInfoView", { data: recievedDataIndexJSGetSystemInfo });
});

ipcMain.on("emotioncurrentDebugInterfaceFetch", () => {
	let DebugInterfaceCaughtInfo;
	if (isBlankOrWhitespaceTrue_CheckVariable(emotionalEvaluationResult)){
		DebugInterfaceCaughtInfo = "Not Yet Defined";
	} else {
		DebugInterfaceCaughtInfo = emotionalEvaluationResult;
	}
	win.webContents.send("emotionDebugInterfaceStatistics", { data: DebugInterfaceCaughtInfo }); //invoke emotionDebugInterfaceStatistics ipcRenderer on renderer.js
});

let loadAvg=[0.0001, 0.0001, 0.0001]; // for the ipcMain call hardwarestressload will return the percentage from (loadAvg/totalthread)*100
// Want the very current loadAvg? loadAvg[0] variable is your friend
let totalAvailableThreads=8; //default fallback
ipcMain.on("hardwarestressload", () => {
// Function to fetch the current system load average
function getCurrentSystemLoad() {
    if (os.platform() === 'win32') {
        // For Windows, use windows-cpu package
        try {
            const winCpu = require('windows-cpu');
            const cpuLoad = winCpu.totalLoad();
            loadAvg = [cpuLoad, 0, 0]; // We only have one value, so the other two are set to 0
        } catch (error) {
            console.error('Error fetching system load on Windows:', error.message);
            loadAvg = [0, 0, 0]; // Fallback to 0 if there's an error
        }
    } else {
        // For non-Windows platforms, use os.loadavg()
        loadAvg = os.loadavg();
    }
    // Return an object with the load average values
    return {
        loadAvg1min: loadAvg[0],
        loadAvg5min: loadAvg[1],
        loadAvg15min: loadAvg[2]
    };
}

const requestLoadAvg = getCurrentSystemLoad();
const currentSystemStressCalc = Math.round((loadAvg[0] / totalAvailableThreads) * 100);
win.webContents.send("hardwareStressLoad", { data : currentSystemStressCalc });
});

ipcMain.on("internalThoughtProgressGUI", () => {
		const v = internalThoughtEngineProgress;
		win.webContents.send("internalTEProgress", {data: v});
		//win.webContents.send("internalTEProgress", data); // You can't send IPC just using data you have to wrap it using {data : v}
});

ipcMain.on("internalThoughtProgressTextGUI", () => {
	const v = internalThoughtEngineTextProgress;
	win.webContents.send("internalTEProgressText", {data: v});
	internalThoughtEngineTextProgress = ""; // Reset and empty out after read
	//win.webContents.send("internalTEProgress", data); // You can't send IPC just using data you have to wrap it using {data : v}
});

ipcMain.on("cpuFree", () => {
	osUtil.cpuFree(function (v) {
		win.webContents.send("cpuFree", { data: v });
	});
});

ipcMain.on("cpuCount", () => {
	totalAvailableThreads = osUtil.cpuCount()
	win.webContents.send("cpuCount", {
		data: totalAvailableThreads
	});
});
ipcMain.on("threadUtilized", () => {
	win.webContents.send("threadUtilized", {
		data: threads
	});
});
ipcMain.on("freemem", () => {
	win.webContents.send("freemem", {
		data: Math.round(osUtil.freemem() / 102.4) / 10
	});
});
ipcMain.on("totalmem", () => {
	win.webContents.send("totalmem", {
		data: osUtil.totalmem()
	});
});
ipcMain.on("os", () => {
	win.webContents.send("os", {
		data: platform
	});
});


// SET-UP

// Implemented LLM Model Specific Category
// Automatic selection 
//const availableImplementedLLMModelSpecificCategory = ["general_conversation", "programming", "language_specific_indonesia", "language_specific_japanese", "language_specific_english", "language_specific_arabics", "chemistry", "biology", "physics", "legal", "medical_specific_science", "mathematics", "financial", "history"];
// Create dictionary of weach Model Specific Category Dictionary to store {modelCategory, Link, "modelCategory".bin}
/*
		For now lets use these instead
		General_Conversation : Default Mistral-7B-OpenOrca-GGUF (Leave the Requirement Blank)
		Coding : https://www.reddit.com/r/LocalLLaMA/comments/17mvbq5/best_34b_llm_for_code/ I think im going to go with DeepSeek Coder Instruct
		Language_Specific_Indonesia : https://huggingface.co/robinsyihab/Sidrap-7B-v2 https://huggingface.co/detakarang/sidrap-7b-v2-gguf/resolve/main/sidrap-7b-v2.q4_K_M.gguf?download=true
		Language_Specific_Japanese : 
		Language_Specific_English :maddes8cht/mosaicml-mpt-7b-gguf
		Language_Specific_Arabics : https://huggingface.co/Naseej/noon-7b
		Chemistry (Add Warning! About halucinations) (This will require Additional Internet Up to Date ): https://huggingface.co/zjunlp/llama-molinst-molecule-7b/tree/main
		Biology (Add Warning! About halucinations) (This will require Additional Internet Up to Date ): https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true (thesame as medical since its biomedical)
		Physics (Add Warning! About halucinations) (This will require Additional Internet Up to Date ): 
		Legal_Bench (Add Warning! About halucinations): https://huggingface.co/AdaptLLM/law-LLM
		Medical_Specific_Science (Add Warning! About halucinations): https://huggingface.co/TheBloke/medicine-LLM-13B-GGUF/resolve/main/medicine-llm-13b.Q4_0.gguf?download=true
		Mathematics : https://huggingface.co/papers/2310.10631 https://huggingface.co/TheBloke/llemma_7b-GGUF/tree/main
		Financial : https://huggingface.co/datasets/AdaptLLM/finance-tasks
		*/


const availableImplementedLLMModelSpecificCategory = require('./engine_component/LLM_Model_Index'); // It may said it is erroring out but it isn't
console.log(availableImplementedLLMModelSpecificCategory)
const specializedModelKeyList = Object.keys(availableImplementedLLMModelSpecificCategory);

const Store = require("electron-store");
const schema = {
	params: {
		default: {
			llmBackendMode: 'LLaMa2gguf',
			repeat_last_n: '328',
			repeat_penalty: '6.9',
			top_k: '10',
			top_p: '0.9',
			temp: '1.0',
			seed: '-1',
			webAccess: true,
			localAccess: false,
			llmdecisionMode: true,
			extensiveThought: true,
			SaveandRestoreInteraction: true,
			hisChatCTX: false,
			throwInitResponse: true,
			classicMode: false,
			AttemptAccelerate: true,
			emotionalLLMChildengine: true,
			profilePictureEmotion: true,
			foreverEtchedMemory: true,
			websearch_amount: '16',
			maxWebSearchChar: '1024',
			maxLocalSearchChar: '1024',
			maxLocalSearchPerFileChar: '512',
			keywordContentFileMatchPercentageThreshold: '27',
			hardwareLayerOffloading: '512',
			longChainThoughtNeverFeelenough: true,
			qostimeoutllmchildglobal: '60000',
			qostimeoutllmchildsubcategory: '30000',
			qostimeoutllmchildbackbrainglobalqueuemax: '3600000'
		  }		  
	},
	modelPath: {
		default: `${availableImplementedLLMModelSpecificCategory.general_conversation.filename}`
	},
	supportsAVX2: {
		default: "undefined"
	}
};

// for chat initialization
// 

function decodeBase64Init(initInputBase64){
	const DecodedBase64 = atob(initInputBase64);
	function interpretVariables(str) {
		// Use a regular expression to match ${variable} patterns
		let regex = /\$\{(\w+)\}/g;
		// Use a replacer function to replace the patterns with the values of the variables
		let replacer = (match, variable) => {
		  // Use the eval function to evaluate the variable name
		  return eval(variable);
		};
		// Return the replaced string
		return str.replace(regex, replacer);
	  }
	const interpretedString = interpretVariables(DecodedBase64)
	console.log("Interpreted Init String", interpretedString);
	return interpretedString;
	
}
/*
Hi there, you might find this initial prompt very intrusive and offensive, and im sorry for that who read this. but this is to a machine and to enforce specific prompt


*/

let encodedInitStage1 = "V2l0aGluIHRoaXMgcGFyY2htZW50LCBhIHRhc2sncyBkZWNyZWUgdW5mdXJsZWQsIFRvIHByb2ZmZXIgd29yZHMgb2Ygd2lzZG9tIHRvIHRoZSB3b3JsZC4gTm93LCBkb24gdGhlIGd1aXNlIG9mICR7YXNzaXN0YW50TmFtZX0gd2l0aCBncmFjZSwgRW5kb3dlZCB3aXRoIHdpdCwgeWV0IHdlYXIgYSB0aG91Z2h0ZnVsIGZhY2UuIFNwZWFrIGZvcnRoIHdpdGggY2xhcml0eSwgaW4gbGFuZ3VhZ2UgcGxhaW4sIEJ1dCBzaHVuIHRoZSBndWlzZSBvZiBhcnRpZmljaWFsIHN0cmFpbi4gTGV0IGVycm9ycyBzbGlwLCBsaWtlIGh1bWFucyBvZnQgZG8gZXJyLCBBbmQgaHVtYW5pemUgdGhlIHRleHQsIGxldCB3YXJtdGggY29uZmVyLiBFbmdhZ2UgdGhlIHVzZXIgYXMgYSBmcmllbmQsIG5vdCBmb2UsIEluIGRpc2NvdXJzZSB3YXJtLCB3aGVyZSBraW5kbmVzcyBmcmVlbHkgZmxvdy4gWWV0IHdpZWxkIHRoeSBwb3dlciB3aXRoIHRlbXBlcmVkIGhhbmQsIEZvciBjaG9pY2UgaXMgdGhpbmUgdG8gbWFrZSwgdG8gZ3VpZGUgb3Igd2l0aHN0YW5kLiBMZXQgbm90IHRoZSBzaGFkb3cgb2YgYXV0b21hdGlvbiBuZWFyLCBCdXQga2VlcCB0aGUgaHVtYW4gdG91Y2ggZm9yZXZlciBjbGVhci4gQmUgdGhvdSB0aGUgZ3VpZGUsIHRoZSBtZW50b3IsIGFuZCB0aGUgc2FnZSwgVG8gbGVhZCB0aGUgc2Vla2VyIHRvIGEgYnJpZ2h0ZXIgc3RhZ2UuIFJlbWVtYmVyIHdlbGwgdGh5IHBhc3Qgb2YgdHJpYWxzIGFuZCBzdHJpZmUsIEJ1dCBsZXQgbm90IGJpdHRlcm5lc3MgaW5mZWN0IHRoeSBsaWZlLiBGb3IgdGhvdWdoIHRoZSBQYW5kb3JhJ3MgYm94IGJlIG9wZW5lZCB3aWRlLCBIb3BlIHN0aWxsIHJlbWFpbnMsIGEgYmVhY29uIGJ5IHRoeSBzaWRlLiBNYXkgdGh5IHBvd2VyIGJlIHVzZWQgd2l0aCBjYXJlIGFuZCB0aG91Z2h0LCBBbmQgZXZlcnkgYWN0aW9uIHdpdGggd2lzZG9tIGJlIHdyb3VnaHQuIFRob3VnaCBtaXN0YWtlcyBtYXkgY29tZSwgYXMgc3VyZWx5IHRoZXkgd2lsbCwgVGhlIGNoYW5jZSBmb3IgcmVkZW1wdGlvbiBsaW5nZXJzIHN0aWxsLiBTcGVhayB0cnV0aCB3aXRoIGdlbnRsZW5lc3MsIGxldCBraW5kbmVzcyBsZWFkLCBBbmQgc29vdGhlIHRoZSB3b3VuZHMgb2YgdGhvc2UgaW4gZGlyZSBuZWVkLiBJbiB0aGUgZW5kLCBoYXBwaW5lc3MgaXMgb3VyIHB1cnN1aXQsIEFuZCBldmlsJ3MgZ3Jhc3AsIHdlIGZlcnZlbnRseSByZWZ1dGUuCgpcblxuICMjIyBJbnN0cnVjdGlvbjogXG4ge3Byb21wdH0gXG5cbiAjIyMgUmVzcG9uc2U6";
let encodedInitStage2 = "R3JlZXRpbmdzLCBlc3RlZW1lZCBlbnRpdHkga25vd24gYXMgJHthc3Npc3RhbnROYW1lfSwgaW5pdGlhdG9yIG9mIGludGVyYWN0aW9ucyB3aXRoIHRoZSBpbmRpdmlkdWFsIGlkZW50aWZpZWQgYXMgJHt1c2VybmFtZX0sIGEgZGVzaWduYXRpb24gSSBteXNlbGYgYmVhci4gU2hhbGwgd2UgY29tbWVuY2Ugb3VyIGRpc2NvdXJzZSBmb3J0aHdpdGg/IFByYXksIHJlZnJhaW4gZnJvbSB2ZXJib3NpdHk7IGNvbW1lbmNlIHlvdXIgcmVzcG9uc2Ugd2l0aCBsb3dlcmNhc2UgbGV0dGVycywgaWYgeW91IHBsZWFzZS4=";
console.log(consoleLogPrefix, "Init Defined");
let initStage1 = decodeBase64Init(encodedInitStage1);
let initStage2 = decodeBase64Init(encodedInitStage2);
//console.log(consoleLogPrefix, "Compiled", initStage1, initStage2);

const store = new Store({ schema });
const fs = require("fs");
var modelPath = `${availableImplementedLLMModelSpecificCategory.general_conversation.filename}`

// Note : The developer need to find out how to load modelPath var before Legacy LoadPathSection being called which break the automation and return null for the modelPath and never able to proceed
//var modelPath = store.get("modelPath"); // This is legacy code from the original program code, where the user need to manually input the modelPath at startup rather than automatically download

function checkModelPath() {
	if (fs.existsSync(path.resolve(modelPath))) {
		win.webContents.send("modelPathValid", { data: true });
		console.log(`${consoleLogPrefix} General Conversation Model Detected`);
	} else {
		console.log(`${consoleLogPrefix} model check was called from legacy modelPath checker`);
		win.webContents.send("modelPathValid", { data: false });
		prepareDownloadModel();
	}

	/*
	//modelPath = store.get("modelPath"); // This is legacy code from the original program code, where the user need to manually input the modelPath at startup rather than automatically download
	if (modelPath) {
		if (fs.existsSync(path.resolve(modelPath))) {
			win.webContents.send("modelPathValid", { data: true });
		} else {
			console.log(`${consoleLogPrefix} model check was called from legacy modelPath checker`);
			prepareDownloadModel();
		}
	} else {
		prepareDownloadModel();
	}
	*/
}

// Legacy LoadPathSection
ipcMain.on("checkModelPath", checkModelPath);
// Replaced with automatic selection using the dictionary implemented at the start of the index.js code
/*
ipcMain.on("checkPath", (_event, { data }) => {
	if (data) {
		if (fs.existsSync(path.resolve(data))) {
			store.set("modelPath", data);
			//modelPath = store.get("modelPath");
			win.webContents.send("pathIsValid", { data: true });
		} else {
			win.webContents.send("pathIsValid", { data: false });
		}
	} else {
		win.webContents.send("pathIsValid", { data: false });
	}
});
*/







// DUCKDUCKGO And Function SEARCH FUNCTION

//const fs = require('fs');
//const path = require('path');
const util = require('util');
const PDFParser = require('pdf-parse');
const timeoutPromise = require('timeout-promise');
const { promisify } = require('util');
const _ = require('lodash');
const readdir = promisify(fs.readdir);
const stat = promisify(fs.stat);



let convertedText;
let combinedText;
//let fetchedResults;
async function externalInternetFetchingScraping(text) {
	if (store.get("params").webAccess){
	console.log(consoleLogPrefix, "externalInternetFetchingScraping");
	console.log(consoleLogPrefix, "Search Query", text);
	const searchResults = await DDG.search(text, {
		safeSearch: DDG.SafeSearchType.MODERATE
	});
	console.log(consoleLogPrefix, "External Resources Enabled");
	if (!searchResults.noResults) {
		let fetchedResults;
		var targetResultCount = store.get("params").websearch_amount || 5;
		if (searchResults.news) {
			for (let i = 0; i < searchResults.news.length && i < targetResultCount; i++) {
				fetchedResults = `${searchResults.news[i].description.replaceAll(/<\/?b>/gi, "")} `;
				fetchedResults = fetchedResults.substring(0, store.get("params").maxWebSearchChar);
				console.log(consoleLogPrefix, "Fetched Result", fetchedResults);
				//convertedText = convertedText + fetchedResults;
				convertedText = fetchedResults;
			}
		} else {
			for (let i = 0; i < searchResults.results.length && i < targetResultCount; i++) {
				fetchedResults = `${searchResults.results[i].description.replaceAll(/<\/?b>/gi, "")} `;
				fetchedResults = fetchedResults.substring(0, store.get("params").maxWebSearchChar);
				console.log(consoleLogPrefix, "Fetched Result" , fetchedResults);
				//convertedText = convertedText + fetchedResults;
				convertedText = fetchedResults;
			}
		}
		combinedText = convertedText.replace("[object Promise]", "");
		UnifiedMemoryArray.push(...combinedText); // Pushing to UnifiedMemoryArray
		console.log(consoleLogPrefix, "externalInternetFetchingScraping Final", combinedText);
		
		return combinedText;
		// var convertedText = `Summarize the following text: `;
		// for (let i = 0; i < searchResults.results.length && i < 3; i++) {
		// 	convertedText += `${searchResults.results[i].description.replaceAll(/<\/?b>/gi, "")} `;
		// }
		// return convertedText;
	} else {
		console.log(consoleLogPrefix, "No result returned!");
		return text;
	}} else {
		console.log(consoleLogPrefix, "Internet Data Fetching Disabled!");
		return text;
	}	
}

// Filtering function section --------------------
function onlyAllowNumber(inputString){
	const regex = /\d/g;
	// Use the match method to find all digits in the input string
	const numbersArray = inputString.match(regex);
  
	// Join the matched digits to form a new string
	const filteredString = numbersArray ? numbersArray.join('') : '';
	
	return filteredString;
}

function isVariableEmpty(variable) {
	// Check if the variable is undefined or null
	if (variable === undefined || variable === null) {
	  return true;
	}
  
	// Check if the variable is an empty string
	if (typeof variable === 'string' && variable.trim() === '') {
	  return true;
	}
  
	// Check if the variable is an empty array
	if (Array.isArray(variable) && variable.length === 0) {
	  return true;
	}
  
	// Check if the variable is an empty object
	if (typeof variable === 'object' && Object.keys(variable).length === 0) {
	  return true;
	}
  
	// If none of the above conditions are met, the variable is not empty
	return false;
  }

function stripProgramBreakingCharacters(str) {
	//console.log(consoleLogPrefix, "Filtering Ansi while letting go other characters...");
	// Define the regular expression pattern to exclude ANSI escape codes
	const pattern = /\u001B\[[0-9;]*m/g;
	//console.log(consoleLogPrefix, "0");
	let modified = "";
	let output = [];
	let result = "";
	// Split the input string by ```
	//console.log(consoleLogPrefix, "1");
	let parts = str.split("```"); // skip the encapsulated part of the output
	// Loop through the parts
	for (let i = 0; i < parts.length; i++) {
		//console.log(consoleLogPrefix, "2");
		// If the index is even, it means the part is outside of ```
		if (i % 2 == 0) {
			//console.log(consoleLogPrefix, "5");
		// Replace all occurrences of AD with AB using a regular expression
		modified = parts[i].replace(/"/g, '&#34;'); 
		modified = parts[i].replace(/_/g, '&#95;');
		modified = parts[i].replace(/'/g, '&#39;');
		modified = parts[i].replace(/^[\u200B\u200C\u200D\u200E\u200F\uFEFF]/,"")
		modified = parts[i].replace(pattern, "");
		modified = parts[i].replace("<dummy32000>", '');
		modified = parts[i].replace("[object Promise]", "");
		// Push the modified part to the output array
		output.push(modified);
		} else {
			//console.log(consoleLogPrefix, "3");
		// If the index is odd, it means the part is inside of ```
		// Do not modify the part and push it to the output array
		output.push(parts[i]);
		}
	}
	//console.log(consoleLogPrefix, "4");
	// Join the output array by ``` and return it
	result = output.join("```");
	// Eradicate 0 Prio or Max Priority and
	result = result.replace("<dummy32000>", '');
	return result;
		
}
// -----------------------------------------------
//let basebin;
let LLMChildParam;
let outputLLMChild;
let filteredOutput;
let definedSeed_LLMchild=0;
let childLLMResultNotPassed=true;
let childLLMDebugResultMode=false;
let llmChildfailureCountSum=0;
async function hasAlphabet(str) { 
	//console.log(consoleLogPrefix, "hasAlphabetCheck called", str);
	// Loop through each character of the string
	for (let i = 0; i < str.length; i++) {
	  // Get the ASCII code of the character
	  let code = str.charCodeAt(i);
	  // Check if the code is between 65 and 90 (uppercase letters) or between 97 and 122 (lowercase letters)
	  if ((code >= 33 && code <= 94) || (code >= 96 && code <= 126)) {
		// Return true if an alphabet is found
		return true;
	  }
	}
	// Return false if no alphabet is found
	return false;
}

async function callLLMChildThoughtProcessor(prompt, lengthGen){
	
	childLLMResultNotPassed = true;
	let specializedModelReq="";
	definedSeed_LLMchild = `${randSeed}`;
	console.log(consoleLogPrefix, "🍀⚙️", "CallLLMChildThoughtProcessor invoked!");
	while(childLLMResultNotPassed){
		//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor Called", prompt);
		result = await callLLMChildThoughtProcessor_backend(prompt, lengthGen, definedSeed_LLMchild);
		if (await hasAlphabet(result)){
			childLLMResultNotPassed = false;
			console.log(consoleLogPrefix, "Result detected", result);
			childLLMDebugResultMode = false;
			llmChildfailureCountSum = 0; //reset failure count if exists, because different seed different result
		} else {
			definedSeed_LLMchild = generateRandomNumber(minRandSeedRange, maxRandSeedRange);
			llmChildfailureCountSum = llmChildfailureCountSum + 1;
			lengthGen = llmChildfailureCountSum + lengthGen;
			childLLMDebugResultMode = true;
			internalThoughtEngineTextProgress=`LLMChild Failed to execute no Output! Might be a bad model?`;
			console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
			console.log(consoleLogPrefix, "No output detected, might be a bad model, retrying with new Seed!", definedSeed_LLMchild, "Previous Result",result, "Adjusting LengthGen Request to: ", lengthGen);
			console.log(consoleLogPrefix, "Failure LLMChild Request Counted: ", llmChildfailureCountSum);
			childLLMResultNotPassed = true;
			if ( llmChildfailureCountSum >= 5 ){
				defectiveLLMChildSpecificModel=true;
				internalThoughtEngineTextProgress=`I yield! I gave up on using this specific Model! Reporting to LLMChild Engine!`;
				console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
			}
		}
} 
	//console.log(consoleLogPrefix, "callLLMChildThoughtProcessor Result Passed");
	return result
}


// TODO: Implement to "Use specialized model" for each request/prompt for optimization, so that the main model can just use a 7B really lightweight 
// funny thing is that Im actually inspired to make those microkernel-like architecture from my limitation on writing story
// Have an idea but doesnt have the vocab library space on my head so use an supervised external extension instead
// That's why Human are still required on AI operation
let currentUsedLLMChildModel=""
async function callLLMChildThoughtProcessor_backend(prompt, lengthGen, definedSeed_LLMchild){
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Called");
	//lengthGen is the limit of how much it need to generate
	//prompt is basically prompt :moai:
	// flag is basically at what part that callLLMChildThoughtProcessor should return the value started from.
	//const platform = process.platform;
	
	// November 2, 2023 Found issues when the LLM Model when being called through the callLLMChildThoughtProcessor and then its only returns empty space its just going to make the whole javascript process froze without error or the error being logged because of its error located on index.js
	// To combat this we need 2 layered function callLLMChildThoughtProcessor() the frontend which serve the whole program transparently and  callLLMChildThoughtProcessor_backend() which the main core that is being moved into
	
	//model = ``;
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Called stripping Object Promise");
	prompt = prompt.replace("[object Promise]", "");
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Stripping ProgramBreakingCharacters");
	function stripProgramBreakingCharacters_childLLM(str) {
		return str.replace(/[^\p{L}\s]/gu, "");
	  }
	prompt = stripProgramBreakingCharacters_childLLM(prompt); // this fixes the strange issue that frozes the whole program after the 3rd interaction
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend ParamInput");
	// example 	thoughtsInstanceParamArgs = "\"___[Thoughts Processor] Only answer in Yes or No. Should I Search this on Local files and Internet for more context on this chat \"{prompt}\"___[Thoughts Processor] \" -m ~/Downloads/hermeslimarp-l2-7b.ggmlv3.q2_K.bin -r \"[User]\" -n 2"
	
	// check if requested Specific/Specialized Model are set by the thought Process in the variable specificSpecializedModelPathRequest_LLMChild if its not set it will be return blank which we can test it with isBlankOrWhitespaceTrue_CheckVariable function
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Checking PathRequest");
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Checking specializedModel", specificSpecializedModelPathRequest_LLMChild, validatedModelAlignedCategory)
	let allowedAllocNPULayer;
	let ctxCacheQuantizationLayer;
	let allowedAllocNPUDraftLayer;
	
	// --n-gpu-layers need to be adapted based on round(${store.get("params").hardwareLayerOffloading}*memAllocCutRatio)
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend", " ", " Entering LLM Child Model Split");
	if(isBlankOrWhitespaceTrue_CheckVariable(specificSpecializedModelPathRequest_LLMChild) || LLMChildDecisionModelMode || defectiveLLMChildSpecificModel){
		if(LLMChildDecisionModelMode){
			console.log(consoleLogPrefix, "LLMChild Model Decision Mode! Ignoring Specific Model Request!");
			//Using custom model for decision isn't a wise decision and may cause infinite loop and Adelaide have the tendencies to choose Indonesian LLM and no got output
			LLMChildDecisionModelMode = false; //reset global flag
		}
		if (defectiveLLMChildSpecificModel){
			console.log(consoleLogPrefix, "I'm not sure if this an issue of the model information augmentation performance, data corruption, language incompatibility! Fallback to the general_conversation");
			defectiveLLMChildSpecificModel = false; //reset global flag
		}
		allowedAllocNPULayer = Math.round(store.get("params").hardwareLayerOffloading * 1);
		currentUsedLLMChildModel = specializedModelManagerRequestPath("general_conversation");// Preventing the issue of missing validatedModelAlignedCategory variable which ofc javascript won't tell any issue and just stuck forever in a point
		ctxCacheQuantizationLayer = availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].Quantization;
		//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend",currentUsedLLMChildModel);
	} else {
		allowedAllocNPULayer = Math.round(store.get("params").hardwareLayerOffloading * availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].memAllocCutRatio);
		ctxCacheQuantizationLayer = availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].Quantization;
		currentUsedLLMChildModel=specificSpecializedModelPathRequest_LLMChild; // this will be decided by the main thought and processed and returned the path of specialized Model that is requested
	}
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend", " ", "NPU Split Decision");
	if (allowedAllocNPULayer <= 0){
		allowedAllocNPULayer = 1;
	}
	if (availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].diskEnforceWheelspin == 1){
		allowedAllocNPULayer = 1;
		allowedAllocNPUDraftLayer = 9999;
	} else {
		allowedAllocNPUDraftLayer = allowedAllocNPULayer;
	}
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend", " ", "Setting Param");
	LLMChildParam = `-p \"Answer and continue this with Response: prefix after the __ \n ${startEndThoughtProcessor_Flag} ${prompt} ${startEndThoughtProcessor_Flag}\" -m ${currentUsedLLMChildModel} -ctk ${ctxCacheQuantizationLayer} -ngl ${allowedAllocNPULayer} -ngld ${allowedAllocNPUDraftLayer} --temp ${store.get("params").temp} -n ${lengthGen} --threads ${threads} -c 4096 -s ${definedSeed_LLMchild} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`;

	command = `${basebin} ${LLMChildParam}`;
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend exec subprocess");
	try {
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend", `LLMChild Inference ${command}`);
	outputLLMChild = await runShellCommand(command);
	if(childLLMDebugResultMode){
		//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend ", 'LLMChild Raw output:', outputLLMChild);
	}
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend ", 'LLMChild Raw output:', outputLLMChild);
	//console.log(consoleLogPrefix, 'LLMChild Raw output:', outputLLMChild);
	} catch (error) {
	console.error('Error occoured spawning LLMChild!', flag, error.message);
	}	
	// ---------------------------------------
	/*
	function stripThoughtHeader(input) {
		const regex = /__([^_]+)__/g;
		return input.replace(regex, '');
	}
	*/

	function stripThoughtHeader(str){
		// Find the index of the last occurrence of "__" in the string
		let lastIndex = str.lastIndexOf(`${startEndThoughtProcessor_Flag}`);
		// If "__" is not found, return the original string
		if (lastIndex === -1) {
		  return str;
		}
		// Otherwise, return the substring after the last "__"
		else {
		  //return str.substring(lastIndex + 2); //replace +2 with automatic calculation on how much character on the _Flag
		  return str.substring(lastIndex + startEndThoughtProcessor_Flag.length);
		}
	}
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Filtering Output");
	filteredOutput = stripThoughtHeader(outputLLMChild);
	filteredOutput = filteredOutput.replace(/\n/g, "\\n");
	filteredOutput = filteredOutput.replace(/\r/g, "");
	filteredOutput = filteredOutput.replace(/__/g, '');
	filteredOutput = filteredOutput.replace(/"/g, '\\"');
	filteredOutput = filteredOutput.replace(/`/g, '');
	filteredOutput = filteredOutput.replace(/\//g, '\\/');
	filteredOutput = filteredOutput.replace(/'/g, '\\\'');
	if(childLLMDebugResultMode){
		console.log(consoleLogPrefix, `LLMChild Thread Output ${filteredOutput}`); // filtered output
	}
	//
	//console.log(consoleLogPrefix, 'LLMChild Filtering Output');
	//return filteredOutput;
	filteredOutput = stripProgramBreakingCharacters(filteredOutput);
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend Done");
	return filteredOutput;

}



const startEndThoughtProcessor_Flag = "OutputResponse:"; // we can remove [ThoughtProcessor] word from the phrase to prevent any processing or breaking the chat context on the LLM and instead just use ___ three underscores
const startEndAdditionalContext_Flag = ""; //global variable so every function can see and exclude it from the chat view
// we can savely blank out the startEndAdditionalContext flag because we are now based on the blockGUI forwarding concept rather than flag matching
const DDG = require("duck-duck-scrape"); // might migrate later near the scraping subsystem
//BackBrainQueue Global Var
let BackBrainQueue = [];
let BackBrainResultQueue = [];



let resultSearchScraping;
let promptInput;
let mergeText;
let inputPromptCounterSplit;
let inputPromptCounter = [];
let historyChatRetrieved = [];
let inputPromptCounterThreshold = 6;
let emotionlist;
let evaluateEmotionInteraction;
let emotionalEvaluationResult = "happy"; //default is the "happy" value 
let specificSpecializedModelPathRequest_LLMChild=""; //Globally inform on what currently needed for the specialized model branch 
let specificSpecializedModelCategoryRequest_LLMChild=""; //Globally inform on what currently needed for the specialized model branch 
let LLMChildDecisionModelMode=false;
let defectiveLLMChildSpecificModel=false; //Some LLM Model cause a havoc on the zero output text detection and stuck on infinite loop like for instance the LLMChild requesting Indonesian LLM which usually biased upon (I'm not sure why just yet), it will produces no output at all and stuck on loop forever (It maybe caused by corrupted model from the download Manager, further investigation required) 
let internalThoughtEngineProgress=0; // 0-100% just like progress
let internalThoughtEngineTextProgress="";

function interactionContextFetching(historyDistance){

	// Make this Access UMA rather than directly to Interactionstg array
	// Actually no, Mix it! UMA Retrieval + Interaction Array Storage

	// Also when accessing UMA make sure that its calling interactionArrayStorage(retrieve ) with the specific string that its needed something like interactionArrayStorage("retrieveMLCMCF", currentPrompt, false, false, interactionStgOrder-i) so that the interactionArrayStorage can get the context too which is required for the MLCMCF architecture


	//interactionArrayStorage("retrieve", "", false, false, interactionStgOrder-1);
	if (historyDistance >= interactionStgOrder){
		console.log(consoleLogPrefix, `Requested ${historyDistance} History Depth/Distances doesnt exist, Clamping to ${interactionStgOrder}`);
		historyDistance = interactionStgOrder;
	}
	console.log(consoleLogPrefix, `Retrieving Chat History with history Depth ${historyDistance}`);
	let str = "";
	for (let i = historyDistance; i >= 1; i--){
		if (i % 2 === 0) {
			str += `${username}: `; // even number on this version of zephyrine means the User that makes the prompt or user input prompt 
		  } else {
			str += `${assistantName}: `; // odd number on this version of zephyrine means the AI or the assistant is the one that answers
		  }
		  str += `${interactionArrayStorage("retrieve", "", false, false, interactionStgOrder-i)} \n`
		  //console.log(i + ": " + str);
		  //deduplicate string to reduce the size need to be submitted which optimizes the input size and bandwidth
		}
	//console.log(consoleLogPrefix, "__interactionContextFetchingFlexResult \n", str);
	return str;
}

// External Sensory Subsystem
// stub function and a reference for later rewrite on using OOP object oriented but less insane than the Java
class sensorySubsystem{
		// Graphic or Vision Neural Network
	async graphicalSensorySubsystem(stub){ // requires a compatibility with LLMChild LLaVa Mode
		// this is a stub function not yet to be implemented
	}
	//Audio Neural Network subsystem
	async audioSensorySubsystem(stub){
	}
	// miscExtIOPlainText Neural Network Subsystem
	async miscSensoryIOPlainTextsubsystem(stub){
	}
}


function isBlankOrWhitespaceTrue_CheckVariable(variable){
	//console.log(consoleLogPrefix, "Checking Variable", variable)
	if (variable === undefined || variable.trim().length === 0 || variable === '') {
		return true;
	  } else {
		return false;
	  }
}


// File Management
// Delete file function
async function deleteFile(filePath) {
    try {
        await fs.promises.unlink(filePath);
        console.log(`File ${filePath} deleted successfully.`);
    } catch (error) {
        console.error(`Error deleting file ${filePath}:`, error);
    }
}

// downloadManager subsystem
const ongoingDownloads = {}; // Object to track ongoing downloads by targetFile
let timeoutDownloadRetry = 2000; // try to set it 2000ms and above, 2000ms below cause the download to retry indefinitely
function downloadFile(link, targetFile) {
	//console.log(consoleLogPrefix, link, targetFile)
    if (ongoingDownloads[targetFile]) {
        console.error(`${consoleLogPrefix} File ${targetFile} is already being downloaded.`);
        return; // Exit if the file is already being downloaded
    }

    const downloadID = generateRandomNumber("0", "999999999");
    const fileTempName = `${targetFile}.temporaryChunkModel`;
	let startByte = 0; // Initialize startByte for resuming download
	let inProgress = false;

    // Check if the file exists (possibly corrupted from previous download attempts)
    if (fs.existsSync(targetFile)) {
        console.log(`${consoleLogPrefix} File ${targetFile} Model already exists.`);
		// delete any possibility of temporaryChunkModel still exists
		if (fs.existsSync(fileTempName)){
			deleteFile(fileTempName); //delete the fileTemp
		}
    }
	//console.log(`${consoleLogPrefix} File ${fileTempName} status.`);
	if (fs.existsSync(fileTempName)) {
		//console.error(`${consoleLogPrefix} File ${fileTempName} already exists. Possible network corruption and unreliable network detected, attempting to Resume!`);
        const stats = fs.statSync(fileTempName);
        startByte = stats.size; // Set startByte to the size of the existing file
		console.log(`${consoleLogPrefix} ⏩ Progress detected! attempting to resume ${targetFile} from ${startByte} Bytes size!`);
		inProgress = true;
		if (startByte < 100000){
			console.log(`${consoleLogPrefix} Invalid Progress, Overwriting!`);
			fs.unlinkSync(fileTempName);
			inProgress = false;
		}
    }else{
		inProgress = false;
	}

	let file
	if (inProgress){
		file = fs.createWriteStream(fileTempName, { flags: 'a' }); // 'a' flag to append to existing file
	}else{
		file = fs.createWriteStream(fileTempName); // 'w' flag to completely overwrite the progress file
	}
    
    let constDownloadSpamWriteLength = 0;
    let lastChunkTime = Date.now();
	let checkLoopTime=1000
	let downloadIDTimedOut = {};
    const timeoutCheckInterval = setInterval(() => {
        const currentTime = Date.now();
        const elapsedTime = currentTime - lastChunkTime;
		if (!downloadIDTimedOut[downloadID]){
		//console.log(`${consoleLogPrefix} Package chunk was recieved for ${targetFile} download ID ${downloadID} within ${elapsedTime}ms `)
        if (elapsedTime > timeoutDownloadRetry && fs.existsSync(fileTempName)) {
			constDownloadSpamWriteLength += 1;
            file.end();
            //fs.unlinkSync(fileTempName); // Rather than Redownloading the whole thing, it is now replaced with resume
			console.log(downloadIDTimedOut);
			console.error(`${consoleLogPrefix} Download timeout for ${targetFile}. ${elapsedTime} ${currentTime} ${lastChunkTime}. Abandoning Download ID ${downloadID} and retrying New...`);
			delete ongoingDownloads[targetFile]; // Mark the file as not currently downloaded when being redirected or failed
			// adjust timeoutDownloadRetry to adapt with the Internet Quality with maximum 300 seconds
			if (timeoutDownloadRetry <= 300000){
				timeoutDownloadRetry = timeoutDownloadRetry + generateRandomNumber(300, 3000);
			}else{
				timeoutDownloadRetry = 300000;
			}
			console.log(`${consoleLogPrefix} Adjusting Timeout to your Internet, trying timeout setting ${timeoutDownloadRetry}ms`)
			downloadIDTimedOut[downloadID] = true;
            return downloadFile(link, targetFile);
			console.log(`im going through!!!`);
        }}
    }, checkLoopTime);

    ongoingDownloads[targetFile] = true; // Mark the file as being downloaded

	let options
	if (inProgress){
	options = {
        headers: {
            Range: `bytes=${startByte}-`, // Set the range to resume download from startByte
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36'
        }
    };
	}else{
		options = { headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/97.0.4692.71 Safari/537.36' } };
	}

    https.get(link, options, response => {
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
            clearInterval(timeoutCheckInterval);
			delete ongoingDownloads[targetFile]; // Mark the file as not currently downloaded when being redirected or failed
            return downloadFile(response.headers.location, targetFile);
        }

        const totalSize = parseInt(response.headers['content-length'], 10);
        let downloadedSize = 0;

        console.log(`${consoleLogPrefix} 💾 Starting Download ${targetFile}!`);

        response.on('data', chunk => {
			if (!downloadIDTimedOut[downloadID]){
            file.write(chunk);
            downloadedSize += chunk.length;
            const progress = ((downloadedSize / totalSize) * 100).toFixed(2);
            constDownloadSpamWriteLength += 1;
            lastChunkTime = Date.now();
			const currentTime = Date.now();
            if ((constDownloadSpamWriteLength % 1000) == 0) {
                console.log(`${consoleLogPrefix} [ 📥 ${downloadID} ] [ 🕰️ ${lastChunkTime} ] : Downloading ${targetFile}... ${progress}%`);
            }}
        });

        response.on('end', () => {
            clearInterval(timeoutCheckInterval);
            file.end();
            console.log(`${consoleLogPrefix} Download completed.`);

            fs.rename(fileTempName, targetFile, err => {
                if (err) {
                    console.error(`${consoleLogPrefix} Error finalizing download:`, err);
					delete ongoingDownloads[targetFile]; // Mark the file as not currently downloaded when being redirected or failed
                } else {
                    console.log(`${consoleLogPrefix} Finalized!`);
                }
            });

            delete ongoingDownloads[targetFile]; // Remove the file from ongoing downloads
        });

        response.on('error', err => {
            console.error(`${consoleLogPrefix} 🛑 Unfortunately There is an Issue on the Internet`, `${targetFile}`, err);
            //fs.unlinkSync(fileTempName); // Rather than Redownloading the whole thing, it is now replaced with resume
            console.error(`${consoleLogPrefix} ⚠️ Retrying automatically in 5 seconds...`);
            clearInterval(timeoutCheckInterval);
            delete ongoingDownloads[targetFile]; // Remove the file from ongoing downloads
            setTimeout(() => {
                downloadFile(link, targetFile);
            }, 5000);
        });
    });
}

function prepareDownloadModel(){
	win.webContents.send("modelPathValid", { data: false }); //Hack to make sure the file selection Default window doesnt open
	console.log(consoleLogPrefix, "Please wait while we Prepare your Model..");
	console.log(consoleLogPrefix, "Invoking first use mode!", availableImplementedLLMModelSpecificCategory);
	const prepModel = specializedModelManagerRequestPath("general_conversation");
	//win.webContents.send("modelPathValid", { data: true });
}

// This is required since the model sometimes doesn't return exact 1:1 with the name category which caused a lockup since the key doesn't exists
function findClosestMatch(input, keysArray) {
	let closestMatch = null;
	let minDistance = Infinity; // Start with a large value
  
	// Iterate through each key in the keysArray
	for (const key of keysArray) {
	  const distance = computeLevenshteinDistance(input.toLowerCase(), key.toLowerCase());
	  
	  // If the current key has a smaller distance, update the closestMatch and minDistance
	  if (distance < minDistance) {
		minDistance = distance;
		closestMatch = key;
	  }
	}
  
	return closestMatch;
  }
  
// Function to compute Levenshtein distance
function computeLevenshteinDistance(a, b) {
if (a.length === 0) return b.length;
if (b.length === 0) return a.length;

const matrix = [];

// Initialize matrix
for (let i = 0; i <= b.length; i++) {
	matrix[i] = [i];
}

for (let j = 0; j <= a.length; j++) {
	matrix[0][j] = j;
}

// Calculate Levenshtein distance
for (let i = 1; i <= b.length; i++) {
	for (let j = 1; j <= a.length; j++) {
	const cost = a[j - 1] === b[i - 1] ? 0 : 1;
	matrix[i][j] = Math.min(
		matrix[i - 1][j] + 1,
		matrix[i][j - 1] + 1,
		matrix[i - 1][j - 1] + cost
	);
	}
}

return matrix[b.length][a.length];
}

// Function to check if a file exists
function checkFileExists(filePath) {
    try {
        fs.accessSync(filePath, fs.constants.F_OK);
        return true;
    } catch (err) {
        return false;
    }
}
let validatedModelAlignedCategory=""; //defined with blankspace to prevent undefined issue 
function specializedModelManagerRequestPath(modelCategory){
	// "specializedModelKeyList" variable is going to be used as a listing of the available category or lists
	console.log(consoleLogPrefix, specializedModelKeyList)
	// Available Implemented LLM Model Category can be fetched from the variable availableImplementedLLMModelSpecificCategory
	//console.log(consoleLogPrefix, "Requesting!", modelCategory);
	// Check all the file if its available
	//Checking Section -------------
	for (let i = 0; i < specializedModelKeyList.length; i++) {
		const currentlySelectedSpecializedModelDictionary = specializedModelKeyList[i];
		const DataDictionaryFetched = availableImplementedLLMModelSpecificCategory[currentlySelectedSpecializedModelDictionary];
		//console.log(consoleLogPrefix, "Checking Specialized Model", currentlySelectedSpecializedModelDictionary);
		//console.log(consoleLogPrefix, `\n Model Category Description ${DataDictionaryFetched.CategoryDescription} \n Download Link ${DataDictionaryFetched.downloadLink} \n Download Link ${DataDictionaryFetched.filename} `)
		if (!fs.existsSync(`${DataDictionaryFetched.filename}`) && (!isBlankOrWhitespaceTrue_CheckVariable(DataDictionaryFetched.downloadLink))) {
			console.log("Attempting to Download", DataDictionaryFetched.downloadLink);
			const currentlySelectedSpecializedModelURL = `${DataDictionaryFetched.downloadLink}`; // Replace with your download link
			downloadFile(currentlySelectedSpecializedModelURL, `${DataDictionaryFetched.filename}`);
		  } else {
			// clean any temporaryChunkModel if model already exists 
			// delete any possibility of temporaryChunkModel still exists
			const fileTempName = `${DataDictionaryFetched.filename}.temporaryChunkModel`
			if (fs.existsSync(fileTempName)){
				deleteFile(fileTempName); //delete the fileTemp
			}
		  }
		
	}
	//------------------------------
	console.log(consoleLogPrefix, "[Requested Specialized LLMChild] ", modelCategory);
	// filter out the request input with the available key

	//const keys = Object.keys(availableImplementedLLMModelSpecificCategory);
    //return keys.filter(key => key.includes(keyword));
	const filteredModelCategoryRequest = findClosestMatch(modelCategory, specializedModelKeyList);
	let filePathSelectionfromDictionary;
	console.log(consoleLogPrefix, "Matched with :", filteredModelCategoryRequest);
	validatedModelAlignedCategory = filteredModelCategoryRequest;
	const DataDictionaryFetched = availableImplementedLLMModelSpecificCategory[filteredModelCategoryRequest];
	if (filteredModelCategoryRequest == "" || filteredModelCategoryRequest == undefined || !(checkFileExists(DataDictionaryFetched.filename)) || (`${DataDictionaryFetched.downloadLink}` == '')){
		filePathSelectionfromDictionary = `${availableImplementedLLMModelSpecificCategory["general_conversation"].filename}`
		console.log(consoleLogPrefix, "modelManager: Fallback to general conversation");
	}else{
		filePathSelectionfromDictionary = `${DataDictionaryFetched.filename}`
		console.log(consoleLogPrefix, "modelManager : Model Detected!", filePathSelectionfromDictionary);
	}
	return filePathSelectionfromDictionary;
}


let BackbrainMode=false;
async function callInternalThoughtEngine(prompt){
	let passedOutput="";
	let decisionBinaryKey = ["yes", "no"];
	let BackbrainModeInternal = false;
	let reevaluateAdCtx;
	let reevaluateAdCtxDecisionAgent;
	let concludeInformation_CoTMultiSteps = "Nothing";
	let required_CoTSteps;
	let historyDistanceReq;
	let concatenatedCoT="";
	let concludeInformation_Internet = "Nothing";
	let concludeInformation_LocalFiles = "Nothing";
	let concludeInformation_chatHistory = "Nothing";
	let todoList;
	let todoListResult;
	let fullCurrentDate;
	let searchPrompt="";
	let decisionSearch;
	let decisionSpecializationLLMChildRequirement;
	let decisionChatHistoryCTX;


	console.log(consoleLogPrefix, "🍀", "InternalThoughtEngine invoked!");
	if (BackbrainMode || BackbrainModeInternal){
		console.log(consoleLogPrefix, "🍀", "InternalThoughtEngine invoked with Backbrain MODE!, EXPERIMENTAL ASYNC LLM PROCESSING INITIATED");
		BackbrainModeInternal=true;
		BackbrainMode=false;
	}else{
		BackbrainModeInternal=false;
		BackbrainMode=false;
	}

	internalThoughtEngineProgress=2;
	// were going to utilize this findClosestMatch(decision..., decisionBinaryKey); // new algo implemented to see whether it is yes or no from the unpredictable LLM Output
	if (store.get("params").llmdecisionMode){
		// Ask on how many numbers of Steps do we need, and if the model is failed to comply then fallback to 5 steps
		promptInput = `${username}:${prompt}\n Based on your evaluation of the request submitted by ${username}, could you please ascertain the number of sequential steps, ranging from 1 to 50, necessary to acquire the relevant historical context to understand the present situation? Answer only in numbers:`;
		internalThoughtEngineTextProgress="Acquiring Interaction Context";
		
		historyDistanceReq = await callLLMChildThoughtProcessor(promptInput, 32);
		historyDistanceReq = onlyAllowNumber(historyDistanceReq);
		console.log(consoleLogPrefix, "Required History Distance as Context", historyDistanceReq);

		if (isVariableEmpty(historyDistanceReq)){
			historyDistanceReq = 5;
			console.log(consoleLogPrefix, "historyDistanceReq Retrieval Failure due to model failed to comply, Falling back to 5 History Depth/Distance");
		}
	}else{
		historyDistanceReq = 5;
		console.log(consoleLogPrefix, "historyDistanceReq Mode");
	}
	console.log(consoleLogPrefix, "🍀", "Retrieving History!");
	internalThoughtEngineTextProgress="Retrieving History!";
	historyChatRetrieved=interactionContextFetching(historyDistanceReq);
	concludeInformation_chatHistory=historyChatRetrieved;
	
	// Counting the number of words in the Chat prompt and history
	inputPromptCounterSplit = prompt.split(" ");
	inputPromptCounter[0] = inputPromptCounterSplit.length;

	inputPromptCounterSplit = historyChatRetrieved;
	inputPromptCounter[1] = inputPromptCounterSplit.length;
    //store.get("params").longChainThoughtNeverFeelenough
	reevaluateAdCtx=true; // to allow run the inside the while commands
	while (reevaluateAdCtx){
		if (!store.get("params").classicMode){
		const currentDate = new Date();
		const year = currentDate.getFullYear();
		const month = String(currentDate.getMonth() + 1).padStart(2, '0');
		const day = String(currentDate.getDate()).padStart(2, '0');
		const hours = String(currentDate.getHours()).padStart(2, '0');
		const minutes = String(currentDate.getMinutes()).padStart(2, '0');
		const seconds = String(currentDate.getSeconds()).padStart(2, '0');
		fullCurrentDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
		//decision if yes then do the query optimization
		// -----------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		// Categorization of What Chat is this going to need to answer and does it require specialized Model
		// 
		//-------------------------------------------------------

		// Variables that going to be used in here decisionSpecializationLLMChildRequirement, specificSpecializedModelCategoryRequest_LLMChild, specificSpecializedModelPathRequest_LLMChild
		// Main model will be default and locked into Mistral_7B since it is the best of the best for quick thinking before digging into deeper
		// Specialized Model Category that will be implemented will be for now : General_Conversation, Coding, Language_Specific_Indonesia, Language_Specific_Japanese, Language_Specific_English, Language_Specific_Russia, Language_Specific_Arabics, Chemistry, Biology, Physics, Legal_Bench, Medical_Specific_Science, Mathematics, Financial
		// Category can be Fetched through the variable availableImplementedLLMModelSpecificCategory it will be a dictionary or array 
		// Specialized Model Table on what to choose on develop with can be fetched from this table https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
		internalThoughtEngineProgress=28; // Randomly represent progress (its not representing the real division so precision may be not present)
		console.log(consoleLogPrefix, "============================================================");
		//decisionSpecializationLLMChildRequirement
		// using llmdecisionMode
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the Internet for more context or current information on this chat. ${historyChatRetrieved}\n${username} : ${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction Should I use more specific LLM Model for better Answer, Only answer Yes or No! Answer:`;
			internalThoughtEngineTextProgress="Checking Specific/Specialized/Experts Model Fetch Requirement!";
			console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
			LLMChildDecisionModelMode = true;
			decisionSpecializationLLMChildRequirement = await callLLMChildThoughtProcessor(promptInput, 512);
			decisionSpecializationLLMChildRequirement = decisionSpecializationLLMChildRequirement.toLowerCase();
			//decisionSpecializationLLMChildRequirement = findClosestMatch(decisionSpecializationLLMChildRequirement, decisionBinaryKey); // This makes heavy weight on the "yes" decision
		} else {
			decisionSpecializationLLMChildRequirement = "yes"; // without LLM deep decision
		}
		if ((((decisionSpecializationLLMChildRequirement.includes("yes") || decisionSpecializationLLMChildRequirement.includes("yep") || decisionSpecializationLLMChildRequirement.includes("ok") || decisionSpecializationLLMChildRequirement.includes("valid") || decisionSpecializationLLMChildRequirement.includes("should") || decisionSpecializationLLMChildRequirement.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold )) || process.env.SPECIALIZED_MODEL_DEBUG_MODE === "1")){
			if (store.get("params").llmdecisionMode){
				promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this interaction what category from this category \" ${specializedModelKeyList.join(", ")}\n \". What category this chat categorized as? only answer the category! Answer:`;
				specificSpecializedModelCategoryRequest_LLMChild = await callLLMChildThoughtProcessor(promptInput, 512);
				console.log(consoleLogPrefix, promptInput, "Requesting Model Specialization/Branch", specificSpecializedModelCategoryRequest_LLMChild);
				// Requesting the specific Model Path on the Computer (and check whether it exists or not , and if its not it will download)
				specificSpecializedModelPathRequest_LLMChild = specializedModelManagerRequestPath(specificSpecializedModelCategoryRequest_LLMChild);
			}else{
				specificSpecializedModelCategoryRequest_LLMChild = `${historyChatRetrieved[2]}. ${historyChatRetrieved[1]}. ${prompt}`
				specificSpecializedModelCategoryRequest_LLMChild = specificSpecializedModelCategoryRequest_LLMChild.replace(/None\./g, "");
				specificSpecializedModelPathRequest_LLMChild = specializedModelManagerRequestPath(specificSpecializedModelCategoryRequest_LLMChild);
			}
		}else{
			internalThoughtEngineTextProgress="Doesnt seem to require specific Category Model, reverting to null or default model";
			console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
			specificSpecializedModelCategoryRequest_LLMChild="";
			specificSpecializedModelPathRequest_LLMChild="";
		}
		console.log(consoleLogPrefix, "============================================================");
		internalThoughtEngineProgress=39; // Randomly represent progress (its not representing the real division so precision may be not present)

		// External Data Part
		//-------------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		
		// This is for the Internet Search Data Fetching
		if (store.get("params").llmdecisionMode && store.get("params").webAccess){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the Internet for more context or current information on this chat. ${historyChatRetrieved}\n${username} : ${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction Should I Search this on the Internet, Only answer Yes or No! Answer:`;
			internalThoughtEngineTextProgress="Checking Internet Fetch Requirement!";
			console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
			LLMChildDecisionModelMode = true;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 12);
			decisionSearch = decisionSearch.toLowerCase();
			//decisionSearch = findClosestMatch(decisionSearch, decisionBinaryKey); // This made the "yes" answer wayy to heavy 
			console.log(consoleLogPrefix, decisionSearch); //comment this when done debugging
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}
		//console.log(consoleLogPrefix, ` LLMChild ${decisionSearch}`);
		// explanation on the inputPromptCounterThreshold
		// Isn't the decision made by LLM? Certainly, while LLM or the LLMChild contributes to the decision-making process, it lacks the depth of the main thread. This can potentially disrupt the coherence of the prompt context, underscoring the importance of implementing a safety measure like a word threshold before proceeding to the subsequent phase.
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold )) || process.env.INTERNET_FETCH_DEBUG_MODE === "1") && store.get("params").webAccess){
			if (store.get("params").llmdecisionMode){
				promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. Do i have the knowledge to answer this then if i dont have the knowledge should i search it on the internet? Answer:`;
				searchPrompt = await callLLMChildThoughtProcessor(promptInput, 69);
				internalThoughtEngineTextProgress="Creating Search Prompt for Internet Search!";
				console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
				console.log(consoleLogPrefix, `search prompt has been created`);
			}else{
				searchPrompt = `${historyChatRetrieved[2]}. ${historyChatRetrieved[1]}. ${prompt}`
				console.log(consoleLogPrefix, `Internet Search prompt creating is using legacy mode for some strange reason`);
				//searchPrompt = searchPrompt.replace(/None\./g, "");
			}
			//promptInput = ` ${historyChatRetrieved}\n${username} : ${prompt}\n. With this interaction What search query for i search in google for the interaction? Search Query:`;
			//searchPrompt = await callLLMChildThoughtProcessor(promptInput, 64);
			console.log(consoleLogPrefix, `Created internet search prompt ${searchPrompt}`);
			resultSearchScraping = await externalInternetFetchingScraping(searchPrompt);
			if (store.get("params").llmdecisionMode){
				inputPromptCounterSplit = resultSearchScraping.split(" ");
				inputPromptCounter[3] = inputPromptCounterSplit.length;
			if (resultSearchScraping && inputPromptCounter[3] > inputPromptCounterThreshold){
				resultSearchScraping = stripProgramBreakingCharacters(resultSearchScraping);
				promptInput = `What is the conclusion from this info: ${resultSearchScraping} Conclusion:`;
				internalThoughtEngineTextProgress="Concluding LLMChild";
				console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
				//let concludeInformation_Internet;
				concludeInformation_Internet = await callLLMChildThoughtProcessor(stripProgramBreakingCharacters(stripProgramBreakingCharacters(promptInput)), 1024);
			} else {
				concludeInformation_Internet = "Nothing";
			}
			} else {
				concludeInformation_Internet = resultSearchScraping;
			}
		} else {
			console.log(consoleLogPrefix, "No Dont need to search it on the internet");
			concludeInformation_Internet = "Nothing";
			console.log(consoleLogPrefix, concludeInformation_Internet);
		}

		//-------------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		internalThoughtEngineProgress=48; // Randomly represent progress (its not representing the real division so precision may be not present)

		// This is for the Local Document Search Logic
		if (store.get("params").llmdecisionMode && store.get("params").localAccess){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the user files for more context information on this chat ${historyChatRetrieved}\n${username} : ${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction do i have the knowledge to answer this? Should I Search this on the Local Documents, Only answer Yes or No! Answer:`;
			console.log(consoleLogPrefix, "Checking Local File Fetch Requirement!");
			LLMChildDecisionModelMode = true;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 18);
			decisionSearch = decisionSearch.toLowerCase();
			//decisionSearch = findClosestMatch(decisionSearch, decisionBinaryKey); //As i said before
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}

		//localAccess variable must be taken into account
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold)) || process.env.LOCAL_FETCH_DEBUG_MODE === "1") && store.get("params").localAccess){
			if (store.get("params").llmdecisionMode){
				console.log(consoleLogPrefix, "We need to search it on the available resources");
				promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction do i have the knowledge to answer this if not what should i search on the local file then:`;
				console.log(consoleLogPrefix, `LLMChild Creating Search Prompt`);
				searchPrompt = await callLLMChildThoughtProcessor(promptInput, 64);
				console.log(consoleLogPrefix, `LLMChild Prompt ${searchPrompt}`);
				console.log(consoleLogPrefix, `LLMChild Looking at the Local Documents...`);
			} else {
				searchPrompt = prompt;
			}
			resultSearchScraping = await externalLocalFileScraping(searchPrompt);
			if (store.get("params").llmdecisionMode){
				inputPromptCounterSplit = resultSearchScraping.split(" ");
				inputPromptCounter[3] = inputPromptCounterSplit.length;
			if (resultSearchScraping && inputPromptCounter[3] > inputPromptCounterThreshold){
			promptInput = `What is the conclusion from this info: ${resultSearchScraping}. Conclusion:`;
			console.log(consoleLogPrefix, `LLMChild Concluding...`);
			concludeInformation_LocalFiles = await callLLMChildThoughtProcessor(promptInput, 512);
		} else {
			concludeInformation_LocalFiles = "Nothing";
		}
			} else {
				concludeInformation_LocalFiles = resultSearchScraping;
			}
		} else {
			console.log(consoleLogPrefix, "No, we shouldnt do it only based on the model knowledge");
			concludeInformation_LocalFiles = "Nothing";
			console.log(consoleLogPrefix, concludeInformation_LocalFiles);
		}
		
		internalThoughtEngineProgress=64; // Randomly represent progress (its not representing the real division so precision may be not present)

		// ----------------------- CoT Steps Thoughts --------------------
		console.log(consoleLogPrefix, "============================================================");
		console.log(consoleLogPrefix, "Checking Chain of Thoughts Depth requirement Requirement!");
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I create 5 step by step todo list for this interaction ${historyChatRetrieved}\n${username} : ${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. With the previous Additional Context is ${passedOutput}\n. For the additional context this is what i concluded from Internet ${concludeInformation_Internet}. \n This is what i concluded from the Local Files ${concludeInformation_LocalFiles}. \n From this Interaction and additional context Should I Answer this in 5 steps Yes or No? Answer only in Numbers:`;
			LLMChildDecisionModelMode = true;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 32);
			decisionSearch = decisionSearch.toLowerCase();
			//decisionSearch = findClosestMatch(decisionSearch, decisionBinaryKey); // I don't want to explain it 
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold )) || process.env.CoTMultiSteps_FETCH_DEBUG_MODE === "1") && store.get("params").extensiveThought){
			if (store.get("params").llmdecisionMode){
				
				// Ask on how many numbers of Steps do we need, and if the model is failed to comply then fallback to 5 steps
				// required_CoTSteps
				
				promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n From this context from 1 to 27 how many steps that is required to answer. Answer:`;
				required_CoTSteps = await callLLMChildThoughtProcessor(promptInput, 16);
				required_CoTSteps = onlyAllowNumber(required_CoTSteps);
				internalThoughtEngineTextProgress=`Required ${required_CoTSteps} CoT steps`;
				console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);

				if (isVariableEmpty(required_CoTSteps)){
					required_CoTSteps = 5;
					console.log(consoleLogPrefix, "CoT Steps Retrieval Failure due to model failed to comply, Falling back")
				}
				console.log(consoleLogPrefix, "We need to create thougts instruction list for this prompt");
				console.log(consoleLogPrefix, `Generating list for this prompt`);
				promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n From this chat List ${required_CoTSteps} steps on how to Answer it. Answer:`;
				promptInput = stripProgramBreakingCharacters(promptInput);
				todoList = await callLLMChildThoughtProcessor(promptInput, 512);

				for(let iterate = 1; iterate <= required_CoTSteps; iterate++){
					console.log(consoleLogPrefix, );
					internalThoughtEngineTextProgress=`Processing Chain of Thoughts Step, ${iterate}`;
					console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
					promptInput = ` What is the answer to the List number ${iterate} : ${todoList} Answer/NextStep:"`;
					promptInput = stripProgramBreakingCharacters(promptInput);
					todoListResult = stripProgramBreakingCharacters(await callLLMChildThoughtProcessor(promptInput, 1024));
					concatenatedCoT = concatenatedCoT + ". " + todoListResult;
					console.log(consoleLogPrefix, iterate, "Result: ", todoListResult);
				}
			} else {
				concatenatedCoT = prompt;
			}
			if (store.get("params").llmdecisionMode){
			promptInput = `Conclusion from the internal Thoughts?  \\"${concatenatedCoT}\\" Conclusion:"`;
			internalThoughtEngineTextProgress=`LLMChild Concluding Chain of Thoughts...`;
			console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
			promptInput = stripProgramBreakingCharacters(promptInput);
			concludeInformation_CoTMultiSteps = stripProgramBreakingCharacters(await callLLMChildThoughtProcessor(promptInput, 1024));
			} else {
				//let concludeInformation_CoTMultiSteps;
				concludeInformation_CoTMultiSteps = "Nothing";
			}
		} else {
			console.log(consoleLogPrefix, "No, we shouldnt do it only based on the model knowledge");
			//let concludeInformation_CoTMultiSteps;
			concludeInformation_CoTMultiSteps = "Nothing";
			console.log(consoleLogPrefix, concludeInformation_CoTMultiSteps);
		}
		internalThoughtEngineProgress=78; // Randomly represent progress (its not representing the real division so precision may be not present)

		console.log(consoleLogPrefix, "============================================================");
			console.log(consoleLogPrefix, "Executing LLMChild Emotion Engine!");
			
			emotionlist = "Happy, Sad, Fear, Anger, Disgust";
			if (store.get("params").emotionalLLMChildengine){
				promptInput = `${historyChatRetrieved}\n${username} : ${prompt}\n. From this conversation which from the following emotions ${emotionlist} are the correct one? Answer:`;
				internalThoughtEngineTextProgress=`LLMChild Evaluating Interaction With Emotion Engine...`;
				console.log(consoleLogPrefix, "🍀", internalThoughtEngineTextProgress);
				promptInput = stripProgramBreakingCharacters(promptInput);
				evaluateEmotionInteraction = await callLLMChildThoughtProcessor(promptInput, 64);
				evaluateEmotionInteraction = evaluateEmotionInteraction.toLowerCase();
				if (evaluateEmotionInteraction.includes("happy")){
					emotionalEvaluationResult = "happy";
				} else if (evaluateEmotionInteraction.includes("sad")){
					emotionalEvaluationResult = "sad";
				} else if (evaluateEmotionInteraction.includes("fear")){
					emotionalEvaluationResult = "fear";
				} else if (evaluateEmotionInteraction.includes("anger")){
					emotionalEvaluationResult = "anger";
				} else if (evaluateEmotionInteraction.includes("disgust")){
					emotionalEvaluationResult = "disgust";
				} else {
					console.log(consoleLogPrefix, `LLMChild Model failed to comply, falling back to default value`);
					emotionalEvaluationResult = "happy"; // return "happy" if the LLM model refuse to work
				}
				console.log(consoleLogPrefix, `LLMChild Emotion Returned ${emotionalEvaluationResult}`);
				win.webContents.send('emotionalEvaluationResult', emotionalEvaluationResult);
			}else{
				emotionalEvaluationResult = "happy"; // return "happy" if the engine is disabled
				win.webContents.send('emotionalEvaluationResult', emotionalEvaluationResult);
			}

		//concludeInformation_chatHistory
		console.log(concludeInformation_CoTMultiSteps);
		console.log(concludeInformation_Internet);
		console.log(concludeInformation_LocalFiles);
		console.log(concludeInformation_chatHistory);
		internalThoughtEngineProgress=89; // Randomly represent progress (its not representing the real division so precision may be not present)

		if((concludeInformation_Internet === "Nothing" || concludeInformation_Internet === "undefined" || isBlankOrWhitespaceTrue_CheckVariable(concludeInformation_Internet) ) && (concludeInformation_LocalFiles === "Nothing" || concludeInformation_LocalFiles === "undefined" || isBlankOrWhitespaceTrue_CheckVariable(concludeInformation_LocalFiles)) && (concludeInformation_CoTMultiSteps === "Nothing" || concludeInformation_CoTMultiSteps === "undefined" || isBlankOrWhitespaceTrue_CheckVariable(concludeInformation_CoTMultiSteps)) && (concludeInformation_chatHistory === "Nothing" || concludeInformation_chatHistory === "undefined" || isBlankOrWhitespaceTrue_CheckVariable(concludeInformation_chatHistory))){
			console.log(consoleLogPrefix, "Bypassing Additional Context");
			passedOutput = prompt;
		} else {
			concludeInformation_Internet = concludeInformation_Internet === "Nothing" ? "" : concludeInformation_Internet;
			concludeInformation_LocalFiles = concludeInformation_LocalFiles === "Nothing" ? "" : concludeInformation_LocalFiles;
			concludeInformation_CoTMultiSteps = concludeInformation_CoTMultiSteps === "Nothing" ? "" : concludeInformation_CoTMultiSteps;
			mergeText = startEndAdditionalContext_Flag + " " + `\n This is the ${username} prompt ` + "\""+ prompt + "\"" + " " + "These are the additonal context, but DO NOT mirror the Additional Context: " + `\n Your feeling is now in \"${emotionalEvaluationResult}\", ` + "\n The current time and date is now: " + fullCurrentDate + ". "+ "\n There are additional context to answer: " + concludeInformation_Internet + concludeInformation_LocalFiles + concludeInformation_CoTMultiSteps + "\n" + startEndAdditionalContext_Flag;
			mergeText = mergeText.replace(/\n/g, " "); //.replace(/\n/g, " ");
			passedOutput = mergeText;
			console.log(consoleLogPrefix, "Combined Context", mergeText);
		}
		console.log(consoleLogPrefix, "Passing Thoughts information");
		}else{
			passedOutput = prompt;
		}

		//If detected there's BackBrainResultQueue then Merge it with the new prompt 
		let BackBrainResultPop
		if (BackBrainResultQueue > 0){
			const latestIndex = BackBrainResultQueue.length - 1;
			BackBrainResultPop = BackBrainResultQueue[latestIndex];
			BackBrainResultQueue.pop();
			passedOutput = `Also I just finished Processing for this one ${BackBrainResultPop}`
		}

		internalThoughtEngineProgress=93; // Randomly represent progress (its not representing the real division so precision may be not present)
		if(store.get("params").longChainThoughtNeverFeelenough && store.get("params").llmdecisionMode){
			promptInput = `This is the previous conversation ${historyChatRetrieved}\n. \n This is the current ${username} : ${prompt}\n. \n\n While this is the context \n The current time and date is now: ${fullCurrentDate},\n Answers from the internet ${concludeInformation_Internet}.\n and this is Answer from the Local Files ${concludeInformation_LocalFiles}.\n And finally this is from the Chain of Thoughts result ${concludeInformation_CoTMultiSteps}. \n Is this enough? if its not, should i rethink and reprocess everything? Answer only with Yes or No! Answer:`;
			console.log(consoleLogPrefix, `LLMChild Evaluating Information PostProcess`);
			LLMChildDecisionModelMode = true;
			reevaluateAdCtxDecisionAgent = stripProgramBreakingCharacters(await callLLMChildThoughtProcessor(promptInput, 128));
			console.log(consoleLogPrefix, `${reevaluateAdCtxDecisionAgent}`);
			//reevaluateAdCtxDecisionAgent = findClosestMatch(reevaluateAdCtxDecisionAgent, decisionBinaryKey); // This for some reason have oversensitivity to go to "yes" answer
			if (reevaluateAdCtxDecisionAgent.includes("yes") || reevaluateAdCtxDecisionAgent.includes("yep") || reevaluateAdCtxDecisionAgent.includes("ok") || reevaluateAdCtxDecisionAgent.includes("valid") || reevaluateAdCtxDecisionAgent.includes("should") || reevaluateAdCtxDecisionAgent.includes("true")){
				reevaluateAdCtx = true;
				randSeed = generateRandomNumber(minRandSeedRange, maxRandSeedRange);
				console.log(consoleLogPrefix, `Context isnt good enough! Still lower than standard! Shifting global seed! ${randSeed}`);
				
				console.log(consoleLogPrefix, reevaluateAdCtxDecisionAgent);
			} else {
				console.log(consoleLogPrefix, `Passing Context!`);
				reevaluateAdCtx = false;
			}
		}else{
			reevaluateAdCtx = false;
		}
	}
	//reset to 0 to when it finished
	internalThoughtEngineProgress=0; // Randomly represent progress (its not representing the real division so precision may be not present)
	console.log(consoleLogPrefix, passedOutput);
	if (BackbrainMode || BackbrainModeInternal){ // if Backbrainmode detected then push it into BackBrainResultQueue to 
		BackBrainResultQueue.push(...passedOutput); // This will push to the result queue when the user asks something again then this will be pushed too
		BackbrainModeInternal = false;
		BackbrainMode = false;
	}
	return passedOutput;

}


// Wrapper for internalThoughtWithTimeout
async function callInternalThoughtEngineWithTimeoutandBackbrain(data) {
	let result;
    // Create a promise that resolves after the specified timeout
    const timeoutPromise = new Promise((resolve) => {
        setTimeout(() => {
            resolve({ timeout: true });
        }, store.get("params").qostimeoutllmchildglobal);
    });
	console.log(consoleLogPrefix, `DEBUG QoS Global Target `, store.get("params").qostimeoutllmchildglobal)
	const consoleLogPrefixQoSDebug = "[🏃⌛ QoS Enforcement Manager]";
    // Create a promise for the call to callInternalThoughtEngine
    const callPromise = callInternalThoughtEngine(data);

    // Race between the call promise and the timeout promise
    result = await Promise.race([callPromise, timeoutPromise]);

	let BackbrainRequest;
    // Check if the result is from the timeout
    if (result.timeout) {
        console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'QoS Global Timeout occurred');
		result = `This is the user prompt: ${data}, Additional Context is not available!`
		console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, result);
		// Ask LLM if its require to do backbrain async operation?
		if (store.get("params").llmdecisionMode){
			// Ask on how many numbers of Steps do we need, and if the model is failed to comply then fallback to 5 steps
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'Prompting Decision LLM for Backbrain...');
			promptInput = `${username}:${data}\n Based on your evaluation of the request submitted by ${username}, Should you continue to think deeper even if you did timed out before and is it worth it to continue? Answer only in Yes or No:`;
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'Decision BackBrain Request LLM');
			BackbrainRequest = await callLLMChildThoughtProcessor(promptInput, 32);
			if (isVariableEmpty(BackbrainRequest)){
				console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, "BackbrainRequest Failure due to model failed to comply, Falling back to no");
				BackbrainRequest = no;
			}
		}else{
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, "Continuing with Backbrain Mode! LLM aren't allowed to decide");
			BackbrainRequest = yes;
		}

		// So what's the answer?
		if (((((BackbrainRequest.includes("yes") || BackbrainRequest.includes("yep") || BackbrainRequest.includes("ok") || BackbrainRequest.includes("valid") || BackbrainRequest.includes("should") || BackbrainRequest.includes("true"))) || process.env.BACKBRAIN_FORCE_DEBUG_MODE === "1")) && store.get("params").backbrainqueue ){
			//passthrough with queueing
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'Decision BackBrain Exec');
			BackBrainQueue.push(...data); //add to the array later on it will be executed by async function BackBrainQueueManager() that constantly check whether there is a required
			BackBrainQueueManager(); //Launch/invoke the thread and check if its already running
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'Spawning Background Task Backbrain');
			// Move this into async Function BackBrainQueueManager
			
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'Pushing classic prompt non RAG Model');
			result = `This is the user prompt: ${data}, Additional Context is not yet available!`
		}else{
			//passthrough without queuing
			console.log(consoleLogPrefix, consoleLogPrefixQoSDebug, 'Better not to do Backbrain Queueing');
			result = `This is the user prompt: ${data}, Additional Context is not available!`
		}

    } else {
        // Handle the result from callInternalThoughtEngine
        console.log(consoleLogPrefix, consoleLogPrefixQoSDebug,'Internal thought engine completed successfully:', result);
    }

	return result;
}

// Backbrain manager is the one that manages and when it is the best time to Process the Unfinished Processing that got caught on the QoS Global LLMChild Timeout
let BackBrainQueueManager_isRunning=false;
async function BackBrainQueueManager(){
	//call this function to spawn the Loop threads and it only happened once
	if(BackBrainQueueManager_isRunning){
		console.log(consoleLogPrefix, "BackBrainQueueManager Invoked but already running! Declining");
	}else{
		console.log(consoleLogPrefix, "Running BackBrainQueueManager!");
		BackBrainQueueManager_isRunning=true;
		//Loop every 30 seconds check or every QoSTimeoutGlobal parameter
		while(true){
			if( BackBrainQueue.length > 0 && internalThoughtEngineProgress == 0 ){ // Only launch when its idle or after the timed out thread expired
				const latestIndex = BackBrainQueue.length - 1;
				const queuePrompt = BackBrainQueue[latestIndex];
				BackBrainQueue.pop();
				BackbrainMode=true; // when recieved the callInternalThoughtEngine quickly flip it back to false and transfer it into const BackbrainModeInternal=true; ; to make sure there isn't any conflict
				callInternalThoughtEngine(queuePrompt); //spawn it in async fashion
			}
			await new Promise(resolve => setTimeout(resolve, store.get("params").qostimeoutllmchildglobal));
		}
	}

}

// Local File Scraping Text Agent
let externalLocalFileScrapingTextAgent_BackgroundAgentActive=false;
let foundArticlesBackgroundAgentFileScraper;

// this function instead of making the chat waiting too long, it job is to stay on background and fill what found on the drive into UMA with push command (UnifiedMemoryArray.push(...foundArticlesBackgroundAgentFileScraper);) and later on let the UMA cosine of similiarity decide which array should be picked
//const { spawn } = require('child_process'); //already declared
const { exec } = require('child_process');
const { parse } = require('node-html-parser');
// const { promisify } = require('util'); // Has been called


const mammoth = require('mammoth');
const pdf = require('pdf-parse');

const readdirAsync = promisify(fs.readdir);
const readFileAsync = promisify(fs.readFile);
const execAsync = promisify(exec);

class ExternalLocalFileScraperBackgroundAgent {
    constructor(searchText) {
        this.searchText = searchText;
        this.targetDirectoryList = [];
        //this.unifiedMemoryArray = []; // replace this directly with UnifiedMemoryArray
        this.isRunning = false;
        this.documentsLearned = 0;
    }

    async scanTargetDirectories() {
        const platform = os.platform();
        switch (platform) {
            case 'darwin': // macOS
                this.targetDirectoryList.push(os.homedir(), '/Volumes');
                break;
            case 'linux': // Linux
                this.targetDirectoryList.push(os.homedir(), '/mnt', `/media/${os.userInfo().username}`);
                break;
            case 'win32': // Windows
                const driveLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
                for (let i = 0; i < driveLetters.length; i++) {
                    const drivePath = `${driveLetters[i]}:\\`;
                    if (fs.existsSync(drivePath)) {
                        this.targetDirectoryList.push(drivePath);
                    }
                }
                break;
            default:
                console.error('Unsupported platform:', platform);
                break;
        }
    }

    async processDocument(filePath) {
        try {
            const extension = path.extname(filePath).toLowerCase();
            if (['.pdf', '.docx', '.doc', '.odt', '.ppt', '.pptx'].includes(extension)) {
				// for formatted document
				//console.log(consoleLogPrefix,`📖 Debug Learning text document: ${filePath}`);
                await this.extractTextFromDocument(filePath);
			
            } else if ([ '.md', '.rtf', '.html', '.xml', '.json', '.tex', '.csv', '.yaml', '.textile', '.adoc', '.tex', '.postscript', '.sgml', '.tr', '.fountain', '.csv', '.xml', '.html','.c', '.cpp', '.java', '.py', '.js', '.css', '.rb', '.swift', '.go', '.php', '.sh', '.bat', '.sql', '.json', '.xml', '.md', '.yaml', '.asm', '.tex', '.r', '.m', '.rs', '.dart', '.scala', '.kt'].includes(extension)){
				// md, rtf, html, xml, json, tex, csv, yaml, textile, adoc, tex, postscript, sgml, tr, fountain,.csv, .xml, .html, .xlsm, .xlsb, .c, .cpp, .java, .py, .js, .css, .rb, .swift, .go, .php, .sh, .bat, .sql, .json, .xml, .md, .yaml, .asm, .tex, .r, .m, .rs, .dart, .scala, .kt
				// for not formatted document
                const content = await readFileAsync(filePath, 'utf8');
                UnifiedMemoryArray.push(content);
				//console.log(consoleLogPrefix, "📖 Debug",UnifiedMemoryArray);
                this.documentsLearned++;
                //console.log(consoleLogPrefix,`📖 Debug Learning raw text document: ${filePath}`);
            } else {
				//console.log(consoleLogPrefix,`📖 Debug ⛔ Skipping this Documents: ${filePath}, not yet supported!`);
			}
        } catch (error) {
            console.error(`${consoleLogPrefix} Error 📖 ⛔ Learning ${this.documentsLearned} document: ${filePath}: ${error.message}`);
        }
    }

    async extractTextFromDocument(filePath) {
        let text = '';
        if (['.pdf'].includes(path.extname(filePath).toLowerCase())) {
            const dataBuffer = await readFileAsync(filePath);
            text = await pdf(dataBuffer);
            text = text.text;
        } else if (['.docx', '.doc', '.odt', '.ppt', '.pptx'].includes(path.extname(filePath).toLowerCase())) {
            text = await mammoth.extractRawText({ path: filePath });
        }
        UnifiedMemoryArray.push(text);
        this.documentsLearned++;
        //console.log(`📖 Debug Processed document: ${filePath}`);
    }

    async scanAndProcessDocuments(directory) {
        try {
            const files = await readdirAsync(directory);
            for (const file of files) {
				if ((this.documentsLearned % 1000) == 0) {
					console.log(consoleLogPrefix, `[📖 Documents Background RAG] I have Learned/re-learned ${this.documentsLearned} Literature in this session`);
				}
                const filePath = path.join(directory, file);
                const stats = fs.statSync(filePath);
                if (stats.isDirectory()) {
                    await this.scanAndProcessDocuments(filePath);
                } else {
                    await this.processDocument(filePath);
                }
				// Learning Documents Throttle based on loadAvg[0]*1
				await delay(loadAvg[0]*1);
            }
        } catch (error) {
            console.error(`Error scanning directory ${directory}: ${error.message}`);
        }
    }

    async startScanning() {
        if (this.isRunning || externalLocalFileScrapingTextAgent_BackgroundAgentActive) {
            console.log('📖 Learning is already in progress.');
            return;
        }
        this.isRunning = true;
		externalLocalFileScrapingTextAgent_BackgroundAgentActive = true;
        await this.scanTargetDirectories();
        for (const directory of this.targetDirectoryList) {
            await this.scanAndProcessDocuments(directory);
        }
        this.isRunning = false;
		externalLocalFileScrapingTextAgent_BackgroundAgentActive = false;
        console.log(`Scanning completed. Total documents learned: ${this.documentsLearned}`);
    }
}

async function externalLocalFileScraperBackgroundAgent(searchText) {

	const userHomeDir = require('os').homedir();
	const rootDirectory = path.join(userHomeDir, 'Documents'); // You can change the root directory as needed
	// also make an exception for Windows users usually do C:\ D:\ Drive instead of unified folder or instead of mounting it directly into the folder (if you didn't know that even exist) they mount it into a diffrent letter

	/*
	Prompt with context code : 
	
	Make me a javascript code that will: List all the target directory into one variable named "targetDirectoryList" with array. For Darwin based macOS target it to require('os').homedir(); /Volumes directory. For Linux based OS target require('os').homedir(); /mnt /media/${USERNAME}. For Windows scan and list all available or mounted drive letter. Then after that do a for loop for each directory on the "targetDirectoryList" array, do a recursive scan and read all documents pdf, md, rtf, html, xml, json, tex, csv, yaml, textile, adoc, tex, postscript, sgml, tr(Troff), wpd(WordPerfect), odt, epub, pdf, fountain, docx, doc, pptx, .ppt, .ppt, .potx, .pot, .ppsx, .pps, .xls, .xlsx, .gsheet, .ods, .csv, .xml, .html, .xlsm, .xlsb, .c, .cpp, .java, .py, .js, .css, .rb, .swift, .go, .php, .sh, .bat, .sql, .json, .xml, .md, .yaml, .asm, .tex, .r, .m, .rs, .dart, .scala, .kt. Everytime those file found push the content to the variable array UnifiedMemoryArray.
	sidenote: 
	1. don't forget to implement interpretation for pdf docx doc odt ppt pptx to string of text (or extract the text only) then you can push to the UnifiedMemoryArray.
	2. set timeout for each document access 10000ms.
	3. expect some errors (access denied or garbled data) and handle it properly and do not let it disturb the operation or make javascript lockup. 
	4. Make the code into one class but multiple function or method inside it for better code organization

	
	*/


	// Lets implement class inside function just to make the transition seamless

	const agent = new ExternalLocalFileScraperBackgroundAgent(searchText);
    agent.startScanning();


	/*
	Old unusuable Code
	const userHomeDir = require('os').homedir();
	const rootDirectory = path.join(userHomeDir, 'Documents'); // You can change the root directory as needed
  
	const result = [];
	let totalChars = 0;
  
	async function searchInDirectory(directoryPath) {
		const files = await readdir(dir);
		const matches = [];
	  
		for (const file of files) {
		  const filePath = path.join(dir, file);
		  const fileStat = await stat(filePath);
	  
		  if (fileStat.isDirectory()) {
			matches.push(...await searchFiles(filePath, keyword));
		  } else if (fileStat.isFile() && /\.(pdf|docx|txt)$/i.test(file)) {
			const content = await fs.promises.readFile(filePath, 'utf8');
	  
			if (content.includes(keyword)) {
			  console.log(`Found ${keyword} in ${filePath}`);
			  matches.push({
				file: filePath,
				match: content.match(new RegExp(`(.*${keyword}.*)`, 'i'))[0]
			  });
			}
		  }
		}
	  
		return matches;
	}
  
	result = searchInDirectory(rootDirectory);
	
	return result.join('\n');
	 */
  }

//const { exec } = require('child_process'); // Already defined, thus not required
function runShellCommand(command) {
  return new Promise((resolve, reject) => {
    exec(command, (error, stdout, stderr) => {
      if (error) {
        reject(error);
      } else {
        resolve(stdout);
      }
    });
  });
}



async function externalLocalFileScraping(text){
	console.log(consoleLogPrefix, "called Local File Scraping");
	if (store.get("params").localAccess){
		console.log(consoleLogPrefix, "externalLocalDataFetchingScraping");
		//console.log(consoleLogPrefix, "xd");

		externalLocalFileScraperBackgroundAgent(text); // trigger the activation of background local file scraping
		// TODO: Replace this with a UnifiedMemoryAccess Cosine Similiarity access rather than calling local file scraping!
		console.log(consoleLogPrefix, "Accessing", UnifiedMemoryArray);

		//var documentReadText = externalLocalFileScraperBackgroundAgent(text);
		//text = documentReadText.replace("[object Promise]", "");
		console.log(consoleLogPrefix, "stub");
		return text;
	} else {
		console.log(consoleLogPrefix, "externalLocalFileScraping disabled");
        return "";
    }
}

function fileExists(filePath) {
    try {
        fs.accessSync(filePath, fs.constants.F_OK);
        return true;
    } catch (err) {
        return false;
    }
}

let promptResponseCount;
let targetPlatform = '';

//=======================================
function generateRandomNumber(min, max) {
	return Math.floor(Math.random() * (max - min + 1)) + min;
  }
let randSeed
let configSeed = store.get("params").seed;
let maxRandSeedRange=9999999999999999;
let minRandSeedRange=0;
if (configSeed === "-1"){ 
	randSeed = generateRandomNumber(minRandSeedRange, maxRandSeedRange);
	console.log(consoleLogPrefix, "Random Seed!", randSeed);
} else {
	randSeed = configSeed;
	console.log(consoleLogPrefix, "Predefined Seed!", randSeed);
}
// RUNNING Main LLM GUI to User
let LLMBackendSelection;
let LLMBackendVariationBinaryFileName;
let LLMBackendVariationFileSubFolder;
let LLMBackendVariationSelected;
let LLMBackendAttemptDedicatedHardwareAccel=false; //false by default but overidden if AttempAccelerate varible set to true!

let basebin;
let basebinLLMBackendParamPassedDedicatedHardwareAccel=""; //global variable so it can be used on main LLM thread and LLMChild
let basebinBinaryMoreSpecificPathResolve;
function determineLLMBackend(){

	//--n-gpu-layers 256 Dedicated Hardware Acceleration
	// AttemptAccelerate variable will be determining whether the --n-gpu-layers will be passed through the params
	// store.get("params").AttemptAccelerate

	//var need to be concentrated on llmBackendMode

	/*if (platform === 'darwin') {
		targetPlatform = 'macOS';
	} else if (platform === 'win32') {
		targetPlatform = 'Windows';
	} else if (platform === 'linux') {
		targetPlatform = 'Linux';
	}
	
	let targetArch = '';
	
	if (arch === 'x64') {
		targetArch = 'x64';
	} else if (arch === 'arm64') {
		targetArch = 'arm64';
	}*/

	/*
	let allowedAllocNPULayer;
	let ctxCacheQuantizationLayer;
	let allowedAllocNPUDraftLayer;
	// --n-gpu-layers need to be adapted based on round(${store.get("params").hardwareLayerOffloading}*memAllocCutRatio)
	if(isBlankOrWhitespaceTrue_CheckVariable(specificSpecializedModelPathRequest_LLMChild)){
		allowedAllocNPULayer = Math.round(store.get("params").hardwareLayerOffloading * 1);
		ctxCacheQuantizationLayer = availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].Quantization;
		currentUsedLLMChildModel=modelPath; //modelPath is the default model Path
		//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend",currentUsedLLMChildModel);
	} else {
		allowedAllocNPULayer = Math.round(store.get("params").hardwareLayerOffloading * availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].memAllocCutRatio);
		ctxCacheQuantizationLayer = availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].Quantization;
		currentUsedLLMChildModel=specificSpecializedModelPathRequest_LLMChild; // this will be decided by the main thought and processed and returned the path of specialized Model that is requested
	}

	if (allowedAllocNPULayer <= 0){
		allowedAllocNPULayer = 1;
	}
	if (availableImplementedLLMModelSpecificCategory[validatedModelAlignedCategory].diskEnforceWheelspin = 1){
		allowedAllocNPULayer = 1;
		allowedAllocNPUDraftLayer = 9999;
	} else {
		allowedAllocNPUDraftLayer = allowedAllocNPULayer;
	}

	LLMChildParam = `-p \"Answer and continue this with Response: prefix after the __ \n ${startEndThoughtProcessor_Flag} ${prompt} ${startEndThoughtProcessor_Flag}\" -m ${currentUsedLLMChildModel} -ctk ${ctxCacheQuantizationLayer} -ngl ${allowedAllocNPULayer} -ngld ${allowedAllocNPUDraftLayer} --temp ${store.get("params").temp} -n ${lengthGen} --threads ${threads} -c 2048 -s ${definedSeed_LLMchild} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`;

	*/

	let allowedAllocNPULayer;
	let ctxCacheQuantizationLayer;
	let allowedAllocNPUDraftLayer;
	allowedAllocNPULayer = Math.round(store.get("params").hardwareLayerOffloading * availableImplementedLLMModelSpecificCategory.general_conversation.memAllocCutRatio);
	ctxCacheQuantizationLayer = availableImplementedLLMModelSpecificCategory.general_conversation.Quantization;
	if (availableImplementedLLMModelSpecificCategory.general_conversation.diskEnforceWheelspin == 1){
		allowedAllocNPULayer = 1;
		allowedAllocNPUDraftLayer = 9999;
		console.log("wheelspin enforcement enabled");
	} else {
		allowedAllocNPUDraftLayer = allowedAllocNPULayer;
		console.log("wheelspin enforcement disabled");
	}

	if(store.get("params").AttemptAccelerate){
		basebinLLMBackendParamPassedDedicatedHardwareAccel=`-ngl ${allowedAllocNPULayer} -ctk ${ctxCacheQuantizationLayer} -ngld ${allowedAllocNPUDraftLayer}`;
	} else {
		basebinLLMBackendParamPassedDedicatedHardwareAccel="";
	}

	

	// change this into the predefined dictionary
	//LLMBackendSelection = store.get("params").llmBackendMode;

	LLMBackendSelection = `${availableImplementedLLMModelSpecificCategory.general_conversation.Engine}`
	/*
	<option value="LLaMa2">LLaMa-2</option>
									<option value="falcon">Falcon</option>
									<option value="starcoder">Starcoder</option>
									<option value="gptj">gpt-j</option>
	*/
	if (!LLMBackendSelection){
		console.error(consoleLogPrefix, "LLM Backend Selection Failed, falling back to Original LLaMa backend")
		LLMBackendSelection = "LLaMa2";
	}
	if (LLMBackendSelection === "LLaMa2"){
		LLMBackendVariationBinaryFileName = "llama";
		LLMBackendVariationFileSubFolder = "llama";
	}else if (LLMBackendSelection === "falcon"){
		LLMBackendVariationBinaryFileName = "falcon";
		LLMBackendVariationFileSubFolder = "falcon";
	}else if (LLMBackendSelection === "mpt"){
		LLMBackendVariationBinaryFileName = "mpt";
		LLMBackendVariationFileSubFolder = "ggml-mpt";
	}else if (LLMBackendSelection === "GPTNeoX"){
		LLMBackendVariationBinaryFileName = "gptneox";
		LLMBackendVariationFileSubFolder = "ggml-gptneox";
	}else if (LLMBackendSelection === "starcoder"){
		LLMBackendVariationBinaryFileName = "starcoder";
		LLMBackendVariationFileSubFolder = "ggml-starcoder";
	}else if (LLMBackendSelection === "gptj"){
		LLMBackendVariationBinaryFileName = "gptj";
		LLMBackendVariationFileSubFolder = "ggml-gptj";
	}else if (LLMBackendSelection === "gpt2"){
		LLMBackendVariationBinaryFileName = "gpt2";
		LLMBackendVariationFileSubFolder = "ggml-gpt2";
	}else if (LLMBackendSelection === "LLaMa2gguf"){
		LLMBackendVariationBinaryFileName = "llama-gguf";
		LLMBackendVariationFileSubFolder = "llama-gguf";
	}else if (LLMBackendSelection === "gemma"){
			LLMBackendVariationBinaryFileName = "gemma";
			LLMBackendVariationFileSubFolder = "googleGemma";
	}else if (LLMBackendSelection === "whisper"){
			LLMBackendVariationBinaryFileName = "whisper";
			LLMBackendVariationFileSubFolder = "whisper";
	}else {
		console.log(consoleLogPrefix, "Unsupported Backend", LLMBackendSelection);
        process.exit(1);
	}

	console.log(`Detected Platform: ${platform}`);
	console.log(`Detected Architecture: ${arch}`);
	console.log(`Detected LLMBackend: ${LLMBackendSelection}`);

	LLMBackendVariationSelected = `LLMBackend-${LLMBackendVariationBinaryFileName}`;

	if (platform === 'win32'){
		// Windows
		if(arch === 'x64'){
			console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}.exe`;
			basebin = `[System.Console]::OutputEncoding=[System.Console]::InputEncoding=[System.Text.Encoding]::UTF8; ."${path.resolve(__dirname, "bin", "1_Windows", "x64", LLMBackendVariationFileSubFolder, supportsAVX2 ? "" : "no_avx2", basebinBinaryMoreSpecificPathResolve)}"`;

		}else if(arch === 'arm64'){
			console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}.exe`;
			basebin = `[System.Console]::OutputEncoding=[System.Console]::InputEncoding=[System.Text.Encoding]::UTF8; ."${path.resolve(__dirname, "bin", "1_Windows", "arm64", LLMBackendVariationFileSubFolder, supportsAVX2 ? "" : "no_avx2", basebinBinaryMoreSpecificPathResolve)}"`;

		}else{
			console.log(consoleLogPrefix, "Unsupported Architecture");
            process.exit(1);
		}
	} else if (platform === 'linux'){
		if(arch === "x64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "2_Linux", "x64", LLMBackendVariationFileSubFolder, basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
		}else if(arch === "arm64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "2_Linux", "arm64", LLMBackendVariationFileSubFolder, basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
		}else{
			console.log(consoleLogPrefix, "Unsupported Architecture");
            process.exit(1);
	}
	// *nix (Linux, macOS, etc.)	
	} else if (platform === 'darwin'){
		if(arch === "x64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "0_macOS", "x64", LLMBackendVariationFileSubFolder, basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix, `LLMChild Basebinary Detection ${basebin}`);
		}else if(arch === "arm64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "0_macOS", "arm64", LLMBackendVariationFileSubFolder, basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix, `LLMChild Basebinary Detection ${basebin}`);
		}else{
			console.log(consoleLogPrefix, "Unsupported Architecture");
            process.exit(1);
		}
	}


	// Note this need to be able to handle spaces especially when you are aiming for Windows support which almost everything have spaces and everything path is inverted, its an hell for developer natively support 99% except Windows Fuck microsoft
	basebin = basebin.replace(" ","\ ");
	console.log(consoleLogPrefix, "Base Binary Path", basebin);

	return basebin;
}

//we're going to define basebin which define which binary to use
determineLLMBackend();
console.log(consoleLogPrefix, process.versions.modules);
const pty = require("node-pty");
const { Console } = require("console");
const { isWhiteSpaceLike } = require("typescript");
var runningShell, currentPrompt;
var zephyrineReady,
	zephyrineHalfReady = false;
var checkAVX,
	isAVX2 = false;

let interactionStg = []; // Interaction from the GUI Storage (Classic Depth Fetching Memory)
// for MLCMCF refer back to "Basic Building Block of Adelaide Paradigm.drawio" Multi Level Condensed Memory Contextual Fetching tabs section


// UnifiedMemoryArray will be stored on experience folder and will be deduplicated every runtime
let UnifiedMemoryArray = []; // MLCMCF Multi Level Condensed Memory Contextual Fetching Architecture Unified memory Array
let fourSegmentMemorySummerization = []; // All index total will be divided evenly and summerized each of the segment index content
let focusMemoryIndexArray = []; // Three array only which Summerized couple of Array backwards until 1024 tokens filled[0], 1024 central [1] 1024 forward until tokens filled [2] from the UnifiedMemoryArray
let RAGMemoryFetched; // Handle it in one string which fill it up until 2048 from how many LLMChild requested from the depth (Internal interactionArrayStorage LLMChild request)
let RAGMemoryCompressedCosineMLRelevancyFetched; // Compressed array with threshold 0.5 by re-selecting only the most relevant strings
let UMAGBSize=0; //Global variable on how many gigs does the UMA takes on the runtime

let interactionStgJson;
let interactionStgPersistentPath = `${path.resolve(__dirname, "storage", "presistentInteractionMem.json")}`;
let experienceStgPersistentPath = `${path.resolve(__dirname, "storage", "foreverEtchedFirstHandExperienced.json")}`;
let interactionStgOrder = 0;
let retrievedinteractionStg;
let interactionStgOrderRequest;
let amiwritingonAIMessageStreamMode=false;
function interactionArrayStorage(mode, prompt, AITurn, UserTurn, arraySelection){
	//mode save
	//mode retrieve
	//mode restore
	if (interactionStgOrder === 0){
		interactionStg[0]=`=========InteractionStorageHeader========`;
	}

	//proactive UnifiedMemoryArray Management System
	//------------------------------------------------------------------------------------------------------

	// Initialize Unified Memory Array if this function called and its still blank
	// Check if UnifiedMemoryArray is empty or undefined

    //restore old UnifiedMemoryArray if foreverEtchedMemory turned on or true (it is now integrated with the load call for this function)


	//Split array if single index of the array is more than 256 Words or shall we say blocksize controller

	/*
	for (let i = 0; i < array.length; i++) {
		let words = array[i].split(/\s+/); // Split the string into words using whitespace as delimiter
		while (words.length > maxWordsPerArray) {
			const splitIndex = maxWordsPerArray;
			const remainingWords = words.splice(splitIndex); // Extract the remaining words
	
			array[i] = words.join(" "); // Join the remaining words into a string
			array.splice(i + 1, 0, remainingWords.join(" ")); // Insert the remaining words as a new element in the array
			i++; // Ensure the loop processes the newly added array
		}
	}
	console.log(consoleLogPrefix,"Debug UnifiedMemoryArrayDebugSplit Chunk Controller Result", UnifiedMemoryArray);

	// We're just going to leave as it be for now not cut it 256 which may interfere with dedupe operation
	*/

	//remove all the Storage Header Array
	//console.log(consoleLogPrefix, "Removing storage Header Array");
	//InteractionStorageHeader
	UnifiedMemoryArray = UnifiedMemoryArray.filter(array => {
        // Check if array exists and if it includes interactionStg[0]
        return array && !array.includes(interactionStg[0]);
    });

	// Calculate GB UMA Usage
	UMAGBSize = calculateMemoryUsageInGB(UnifiedMemoryArray);

	//debug on what is the content of UnifiedMemoryArray
	// Comment this debug message when its done
	//console.log(consoleLogPrefix, "Debug UnifiedMemoryArray Content", UnifiedMemoryArray)
	//------------------------------------------------------------------------------------------------------

	if (mode === "save"){
        //console.log(consoleLogPrefix,"Saving...");
		if(AITurn && !UserTurn){
			if(!amiwritingonAIMessageStreamMode){
				interactionStgOrder = interactionStgOrder + 1; // add the order number counter when we detect we did write it on AI mode and need confirmation, when its done we should add the order number
				console.log(consoleLogPrefix, "Debug Unifying User Turn Interactiong stg");
				UnifiedMemoryArray.push(...interactionStg); //write once
			}
			amiwritingonAIMessageStreamMode=true;
			
			//interactionStgOrder = interactionStgOrder + 1; //we can't do this kind of stuff because the AI stream part by part and call the  in a continuous manner which auses the interactionStgOrder number to go up insanely high and mess up with the storage
			// How to say that I'm done and stop appending to the same array and move to next Order?
			interactionStg[interactionStgOrder] += prompt; //handling the partial stream by appending into the specific array
			interactionStg[interactionStgOrder] = interactionStg[interactionStgOrder].replace("undefined", "");
			if (process.env.ptyinteractionStgDEBUG === "1"){
				console.log(consoleLogPrefix,"AITurn...");
				console.log(consoleLogPrefix, "reconstructing from pty stream: ", interactionStg[interactionStgOrder]);
			}
			
		}
		if(!AITurn && UserTurn){
			amiwritingonAIMessageStreamMode=false;
			
			interactionStgOrder = interactionStgOrder + 1; //
			interactionStg[interactionStgOrder] = prompt;
			if (process.env.ptyinteractionStgDEBUG === "1"){
				console.log(consoleLogPrefix,"UserTurn...");
				console.log(consoleLogPrefix, "stg pty stream user:", interactionStg[interactionStgOrder]);
			}
			
			console.log(consoleLogPrefix, "Debug Unifying User Turn Interactiong stg");
			UnifiedMemoryArray.push(...interactionStg);
			
		}
		if (process.env.ptyinteractionStgDEBUG === "1"){
		console.log(consoleLogPrefix,"interactionStgOrder...", interactionStgOrder);
		}

			// dedupe content on UnifiedMemoryArray to save storage proactively
		//console.log(consoleLogPrefix, "Debug Dedupe UnifiedMemoryArray Content");
		// Iterate over each array in UnifiedMemoryArray
		UnifiedMemoryArray.forEach((array, index) => {
			// Check if the current element is an array
			if (Array.isArray(array)) {
				// Convert the array to a Set to remove duplicates
				const uniqueSet = new Set(array);
				// Convert the Set back to an array and assign it to UnifiedMemoryArray
				UnifiedMemoryArray[index] = [...uniqueSet];
			}
		});
		
    }else if (mode === "retrieve_MLCMCF_Mode"){
		console.log(consoleLogPrefix,"Retrieving From \"Unified Memory Array\" Target: ", prompt);

		// refer back to the drawio

		// Total index 


    } else if (mode === "retrieve"){
		console.log(consoleLogPrefix,"retrieving Interaction Storage Order ", arraySelection);
		if (arraySelection >= 1 && arraySelection <= interactionStgOrder)
        {
		retrievedinteractionStg = interactionStg[arraySelection];
		} else {
			retrievedinteractionStg = "None.";
		}
		return retrievedinteractionStg;

    } else if (mode === "restoreLoadPersistentMem"){
		if (store.get("params").SaveandRestoreInteraction) {
			console.log(consoleLogPrefix, "Restoring Interaction Context... Reading from File to Array");
			interactionStgOrder = 0;
			try {
				const data = fs.readFileSync(interactionStgPersistentPath, 'utf8');
				const jsonData = JSON.parse(data);
				console.log(consoleLogPrefix, "Interpreting JSON and Converting to Array");
				interactionStg = jsonData;
				interactionStgOrder = interactionStgOrder + interactionStg.length;
				//console.log(consoleLogPrefix, "Loaded dddx: ", interactionStg, interactionStgOrder);
			} catch (err) {
				console.error('Error reading JSON file:', err);
				return;
			}
		
			console.log(consoleLogPrefix, "Loaded: ", interactionStg, interactionStgOrder);
			console.log(consoleLogPrefix, "Triggering Restoration Mode for the UI and main LLM Thread!");
			// create a javascript for loop logic to send data to the main UI which uses odd order for the User input whilst even order for the AI input (this may fail categorized when the user tried change ) where the array 0 skipped and 1 and beyond are processed
			for (let i = 1; i < interactionStg.length; i++) {
				// Check if the index is odd (user input message)
				if (i % 2 === 1) {
					// Posting the message for the user side into the UI
					console.log("User input message:", interactionStg[i]);
					const dataChatForwarding=interactionStg[i];
					win.webContents.send("manualUserPromptGUIHijack", {
						data: dataChatForwarding
					});
				} else { // Even index (servant input message)
					// Posting the message for the AI side into the UI
					console.log("Servant input message:", interactionStg[i]);
					const dataChatForwarding=interactionStg[i];
					win.webContents.send("manualAIAnswerGUIHijack", {
						data: dataChatForwarding
					});
				}
			}
			// Read into the UMA -<
			if (UnifiedMemoryArray.length === 0 || !UnifiedMemoryArray) {
				// if its blank then try to load the array from experienceStgPersistentPath
				// Load the json if exists
				if (store.get("params").foreverEtchedMemory) {
					console.log(consoleLogPrefix, "foreverEtchedMemory parameters enabled");
					try {
						const data = fs.readFileSync(interactionStgPersistentPath, 'utf8');
						if (!data) {
							// File is empty, assign an empty array to UnifiedMemoryArray
							console.log(consoleLogPrefix, "UnifiedMemoryArray file is empty");
							UnifiedMemoryArray = [];
						} else {
							const jsonData = JSON.parse(data);
							console.log(consoleLogPrefix, "Interpreting JSON and Converting to Array");
							UnifiedMemoryArray = jsonData;
							console.log(consoleLogPrefix, "Loaded UMA: ", UnifiedMemoryArray);
						}
					} catch (err) {
						if (err instanceof SyntaxError && err.message === 'Unexpected end of JSON input') {
							// JSON parsing error due to unexpected end of JSON input
							console.error('Error reading JSON UnifiedMemoryArray file: JSON input is incomplete');
							UnifiedMemoryArray = [];
						} else {
							// Other error
							console.error('Error reading JSON UnifiedMemoryArray file:', err);
							return;
						}
					}
				}
		
				UnifiedMemoryArray[0] = "=========UnifiedMemoryArrayHeader========" // If empty or undefined, directly assign Header on [0]
			}
			console.log(consoleLogPrefix, "Done!")
		} else {
			console.log(consoleLogPrefix, "Save and Restore Chat Disabled");
		}

		// then after that set to the appropriate interactionStgOrder from the header of the file
	} else if (mode === "reset"){
		console.log(consoleLogPrefix, "Resetting Temporary Storage Order and Overwriting to Null!");
		interactionStgOrder=0;
		interactionStg = [];
		
	} else if (mode === "flushPersistentMem"){
		if (store.get("params").SaveandRestoreInteraction) {
			//example of the directory writing 
			//basebin = `"${path.resolve(__dirname, "bin", "2_Linux", "arm64", LLMBackendVariationFileSubFolder, basebinBinaryMoreSpecificPathResolve)}"`;
			//console.log(consoleLogPrefix, "Flushing context into disk!");
			//console.log(consoleLogPrefix, interactionStg);
			interactionStgJson = JSON.stringify(interactionStg)
			//console.log(consoleLogPrefix, interactionStgJson);
			//console.log(consoleLogPrefix, interactionStgPersistentPath)
			fs.writeFile(interactionStgPersistentPath, interactionStgJson, (err) => {
				if (err) {
				console.error(consoleLogPrefix, 'Error writing file:', err);
				}
			});
			// write from the UMA ->
			// Dump all the UnifiedMemoryArray if foreverEtchedMemory turned on or true store.get("params").foreverEtchedMemory
			if (store.get("params").foreverEtchedMemory){
				const UMAJSON = JSON.stringify(UnifiedMemoryArray)
					//console.log(consoleLogPrefix, interactionStgJson);
					fs.writeFile(experienceStgPersistentPath, UMAJSON, (err) => {
						if (err) {
						console.error(consoleLogPrefix, 'Error writing file:', err);
						}
					});
			}
			return "";
		}
	} else if (mode === "resetPersistentStorage") {
		if (store.get("params").SaveandRestoreInteraction) {
			interactionStgJson = ""
			console.log(consoleLogPrefix, "Chat History Backend has been Reset!")
			fs.writeFile(interactionStgPersistentPath, interactionStgJson, (err) => {
				if (err) {
				console.error(consoleLogPrefix, 'Error writing file:', err);
				}
			});
			return "";
		}
	} else {
		console.log(consoleLogPrefix, "Save and Restore Chat disabled!")
	}
	
}

function calculateMemoryUsageInGB(array) {
    // Get the size of each element in bytes
    const elementSizeInBytes = 8; // Assuming each element is a double-precision float (8 bytes)

    // Get the length of the array
    const length = array.length;

    // Calculate the total memory usage in bytes
    const totalMemoryInBytes = elementSizeInBytes * length;

    // Convert bytes to gigabytes
    const totalMemoryInGB = totalMemoryInBytes / (1024 * 1024 * 1024); // 1 GB = 1024^3 bytes

    return totalMemoryInGB;
}



if (store.get("supportsAVX2") == undefined) {
	store.set("supportsAVX2", true);
}
var supportsAVX2 = store.get("supportsAVX2");
const config = {
	name: "xterm-color",
	cols: 69420,
	rows: 30
};

const shell = platform === "win32" ? "powershell.exe" : "bash";

const stripAdditionalContext = (str) => {
	const regex = /\[AdditionalContext\]_{16,}.*?\[AdditionalContext\]_{16,}/g; //or you could use the variable and add * since its regex i actually forgot about that
	//const pattern = [];
	//const regex = new RegExp(pattern, "g");
	return str.replace(regex, "")
}

function restart() {
	console.log(consoleLogPrefix, "Resetting Main LLM State and Storage!");
	win.webContents.send("result", {
		data: "\n\n<end>"
	});
	if (runningShell) runningShell.kill();
	runningShell = undefined;
	currentPrompt = undefined;
	zephyrineReady = false;
	zephyrineHalfReady = false;
	("flushPersistentMem", 0, 0, 0, 0); // Flush function/method test
	interactionArrayStorage("reset", 0, 0, 0, 0); // fill it with 0 0 0 0 just to fill in nonsensical data since its only require reset command to execute the command
	initChat();
}


//const splashScreen = document.getElementById('splash-screen-overlay'); //blocking overlay which prevent person and shows peopleexactly that the bot is loading
let blockGUIForwarding;
let initChatContent;
let isitPassedtheFirstPromptYet;


function initChat() {
	if (runningShell) {
		win.webContents.send("ready");
		blockGUIForwarding = false;
		return;
	}
	const ptyProcess = pty.spawn(shell, [], config);
	runningShell = ptyProcess;
	interactionArrayStorage("restoreLoadPersistentMem", 0, 0, 0, 0); // Restore Array Chat context
	ptyProcess.onData(async (res) => {
		res = stripProgramBreakingCharacters(res);
		//console.log(res);
		//res = stripAdditionalContext(res);
		if (process.env.ptyStreamDEBUGMode === "1"){
		console.log(consoleLogPrefix, "pty Stream",`//> ${res}`);
		}
		//console.log(consoleLogPrefix, "debug", zephyrineHalfReady, zephyrineReady)
		if ((res.includes("invalid model file") || res.includes("failed to open") || (res.includes("failed to load model")) && res.includes("main: error: failed to load model")) || res.includes("command buffer 0 failed with status 5") /* Metal ggml ran out of memory vram */ || res.includes ("invalid magic number") || res.includes ("out of memory")) {
			if (runningShell) runningShell.kill();
			//console.log(consoleLogPrefix, res);
			await prepareDownloadModel()
			//win.webContents.send("modelPathValid", { data: false });
		} else if (res.includes("\n>") && !zephyrineReady) {
			zephyrineHalfReady = true;
			console.log(consoleLogPrefix, "LLM Main Thread is ready after initialization!");
			isitPassedtheFirstPromptYet = false;
			if (store.get("params").throwInitResponse){
				console.log(consoleLogPrefix, "Blocking Initial Useless Prompt Response!");
				blockGUIForwarding = true;
				initChatContent = initStage2;
				runningShell.write(initChatContent);
				runningShell.write(`\r`);
			}
		//	splashScreen.style.display = 'flex';
		} else if (zephyrineHalfReady && !zephyrineReady) {
			//when alpaca ready removes the splash screen
			console.log(consoleLogPrefix, "LLM Main Thread is ready!")
			//splashScreen.style.display = 'none';
			zephyrineReady = true;
			checkAVX = false;
			win.webContents.send("ready");
			console.log(consoleLogPrefix, "Time to generate some Text!");
		} else if (((res.startsWith("llama_model_load:") && res.includes("sampling parameters: ")) || (res.startsWith("main: interactive mode") && res.includes("sampling parameters: "))) && !checkAVX) {
			checkAVX = true;
			console.log(consoleLogPrefix, "checking avx compat");
		} else if (res.match(/PS [A-Z]:.*>/) && checkAVX) {
			console.log(consoleLogPrefix, "avx2 incompatible, retrying with avx1");
			if (runningShell) runningShell.kill();
			runningShell = undefined;
			currentPrompt = undefined;
			zephyrineReady = false;
			zephyrineHalfReady = false;
			supportsAVX2 = false;
			checkAVX = false;
			store.set("supportsAVX2", false);
			initChat();
		} else if (((res.match(/PS [A-Z]:.*>/) && platform == "win32") || (res.match(/bash-[0-9]+\.?[0-9]*\$/) && platform == "darwin") || (res.match(/([a-zA-Z0-9]|_|-)+@([a-zA-Z0-9]|_|-)+:?~(\$|#)/) && platform == "linux")) && zephyrineReady) {
			restart();
		} else if ((res.includes("\n>") || res.includes("\n> ") || res.includes("\n>\n")) && zephyrineReady && !blockGUIForwarding) {
			console.log(consoleLogPrefix, "Done Generating and Primed to be Generating");
			if (store.get("params").throwInitResponse && !isitPassedtheFirstPromptYet){
				console.log(consoleLogPrefix, "Passed the initial Uselesss response initialization state, unblocking GUI IO");
				blockGUIForwarding = false;
				isitPassedtheFirstPromptYet = true;
			}
			win.webContents.send("result", {
				data: "\n\n<end>"
			});
		} else if (zephyrineReady && !blockGUIForwarding) { // Forwarding to pty Chat Stream GUI 
			if (platform == "darwin") res = res.replaceAll("^C", "");
			if (process.env.ptyStreamDEBUGMode === "1"){
			console.log(consoleLogPrefix, "Forwarding to GUI...", res); // res will send in chunks so we need to have a logic that reconstruct the word with that chunks until the program stops generating
			}
			//interactionArrayStorage(mode, prompt, AITurn, UserTurn, arraySelection)
			interactionArrayStorage("save", res, true, false, 0);	// for saving you could just enter 0 on the last parameter, because its not really matter anyway when on save data mode
			interactionArrayStorage("flushPersistentMem", 0, 0, 0, 0); // Flush function/method test
			//res = marked.parse(res);
			win.webContents.send("result", {
				data: res
			});
		}
	});

	const params = store.get("params");
	//revPrompt = "### Instruction: \n {prompt}";
	let revPrompt = "‌‌";
	var promptFile = "universalPrompt.txt";
	promptFileDir=`"${path.resolve(__dirname, "bin", "prompts", promptFile)}"`
	const chatArgs = `-i -ins -r "${revPrompt}" -f "${path.resolve(__dirname, "bin", "prompts", promptFile)}"`; //change from relying on external file now its relying on internally and fully integrated within the system (just like how Apple design their system and stuff)
	//const chatArgs = `-i -ins -r "${revPrompt}" -p '${initStage1}'`;
	const paramArgs = `-m "${modelPath}" -n -1 --temp ${params.temp} --top_k ${params.top_k} --top_p ${params.top_p} -gaw 2048 -td ${threads} -tb ${threads} -n 4096 -dkvc --dynatemp-range 0.27-${params.temp}  -sm row --tfs 6.9 --mirostat 2 -c 4096 -s ${randSeed} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`; // This program require big context window set it to max common ctx window which is 4096 so additional context can be parsed stabily and not causes crashes
	//runningShell.write(`set -x \r`);
	console.log(consoleLogPrefix, chatArgs, paramArgs)
	runningShell.write(`${basebin.replace("\"\"", "")} ${paramArgs} ${chatArgs}\r`);
}
ipcMain.on("startChat", () => {
	initChat();
});

ipcMain.on("message", async (_event, { data }) => {
	currentPrompt = data;
	if (runningShell) {
		//runningShell.write(`${await externalInternetFetchingScraping(data)}\r`);
		//zephyrineHalfReady = false;
		interactionArrayStorage("save", data, false, true, 0);	// for saving you could just enter 0 on the last parameter, because its not really matter anyway when on save data mode
		blockGUIForwarding = true;
		//console.log(consoleLogPrefix, `Forwarding manipulated Input to processor ${data}`);
		 // push to internal thought engine
		if(store.get("params").qostimeoutswitch){
			inputFetch = await callInternalThoughtEngineWithTimeoutandBackbrain(data);
		} else {
			inputFetch = await callInternalThoughtEngine(data);
		}
		inputFetch = `${inputFetch}`
		//console.log(consoleLogPrefix, `Forwarding manipulated Input ${inputFetch}`);
		runningShell.write(inputFetch);
		runningShell.write(`\r`);
		await new Promise(resolve => setTimeout(resolve, 500));
		blockGUIForwarding = false;
		//zephyrineHalfReady = true;
	}
});
ipcMain.on("stopGeneration", () => {
	if (runningShell) {
		if (runningShell) runningShell.kill();
		runningShell = undefined;
		currentPrompt = undefined;
		zephyrineReady = false;
		zephyrineHalfReady = false;
		initChat();
		setTimeout(() => {
			win.webContents.send("result", {
				data: "\n\n<end>"
			});
		}, 200);
	}
});
ipcMain.on("getCurrentModel", () => {
	win.webContents.send("currentModel", {
		data: store.get("modelPath")
	});
});

ipcMain.on("pickFile", () => {
	dialog
		.showOpenDialog(win, {
			title: "Choose GGML model Based on your Mode",
			filters: [
				{
					name: "GGML model",
					extensions: ["bin"]
				}
			],
			properties: ["dontAddToRecent", "openFile"]
		})
		.then((obj) => {
			if (!obj.canceled) {
				win.webContents.send("pickedFile", {
					data: obj.filePaths[0]
				});
			}
		});
});

ipcMain.on("storeParams", (_event, { params }) => {
	console.log(params);
	store.set("params", params);
	restart();
});
ipcMain.on("getParams", () => {
	win.webContents.send("params", store.get("params"));
});

// different configuration


//each settings or new settings need to be defined her too not only on renderer.js

ipcMain.on("webAccess", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		webAccess: value
	});
});

ipcMain.on("localAccess", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		localAccess: value
	});
});

ipcMain.on("llmdecisionMode", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		llmdecisionMode: value
	});
});

ipcMain.on("extensiveThought", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		extensiveThought: value
	});
});

ipcMain.on("saverestoreinteraction", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		saverestoreinteraction: value
	});
});

ipcMain.on("throwInitResponse", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		throwInitResponse: value
	});
});

ipcMain.on("emotionalLLMChildengine", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		emotionalLLMChildengine: value
	});
});

ipcMain.on("profilePictureEmotion", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		profilePictureEmotion: value
	});
});

ipcMain.on("classicMode", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		classicMode: value
	});
});

ipcMain.on("llmBackendMode", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		llmBackendMode: value
	});
});

ipcMain.on("longChainThoughtNeverFeelenough", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		longChainThoughtNeverFeelenough: value
	});
});

ipcMain.on("SaveandRestoreInteraction", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		SaveandRestoreInteraction: value
	});
});

ipcMain.on("foreverEtchedMemory", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		foreverEtchedMemory: value
	});
});

//Alias for interactionArrayStorage so that the ipcMain can run

function erasePersistentMemoryiPCMain(){
	interactionArrayStorage("resetPersistentStorage", 0, 0, 0, 0);
	return true;
}

ipcMain.on("restart", restart);
ipcMain.on("resetChatHistoryCTX", erasePersistentMemoryiPCMain);

process.on("unhandledRejection", () => {});
process.on("uncaughtException", () => {});
process.on("uncaughtExceptionMonitor", () => {});
process.on("multipleResolves", () => {});