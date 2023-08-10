const { BrowserWindow, app, ipcMain, dialog } = require("electron");
const path = require("path");
require("@electron/remote/main").initialize();

const os = require("os");
const platform = os.platform();
const arch = os.arch();
const appName = "Project Zephyrine"
const consoleLogPrefix = `[${appName}_${platform}_${arch}]:`;

var win;
function createWindow() {
	win = new BrowserWindow({
		width: 1200,
		height: 810,
		minWidth: 960,
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

	win.setMenu(null);
	// win.webContents.openDevTools();
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
ipcMain.on("cpuUsage", () => {
	osUtil.cpuUsage(function (v) {
		win.webContents.send("cpuUsage", { data: v });
	});
});
ipcMain.on("cpuFree", () => {
	osUtil.cpuFree(function (v) {
		win.webContents.send("cpuFree", { data: v });
	});
});

ipcMain.on("cpuCount", () => {
	win.webContents.send("cpuCount", {
		data: osUtil.cpuCount()
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
const Store = require("electron-store");
const schema = {
	params: {
		default: {
			model_type: "alpaca",
			repeat_last_n: "64",
			repeat_penalty: "1.3",
			top_k: "40",
			top_p: "0.9",
			temp: "0.8",
			seed: "-1",
			webAccess: false,
			websearch_amount: "5"
		}
	},
	modelPath: {
		default: "undefined"
	},
	supportsAVX2: {
		default: "undefined"
	}
};
const store = new Store({ schema });
//const fs = require("fs");
var modelPath = store.get("modelPath");

function checkModelPath() {
	modelPath = store.get("modelPath");
	if (modelPath) {
		if (fs.existsSync(path.resolve(modelPath))) {
			win.webContents.send("modelPathValid", { data: true });
		} else {
			win.webContents.send("modelPathValid", { data: false });
		}
	} else {
		win.webContents.send("modelPathValid", { data: false });
	}
}
ipcMain.on("checkModelPath", checkModelPath);

ipcMain.on("checkPath", (_event, { data }) => {
	if (data) {
		if (fs.existsSync(path.resolve(data))) {
			store.set("modelPath", data);
			modelPath = store.get("modelPath");
			win.webContents.send("pathIsValid", { data: true });
		} else {
			win.webContents.send("pathIsValid", { data: false });
		}
	} else {
		win.webContents.send("pathIsValid", { data: false });
	}
});

// DUCKDUCKGO And Function SEARCH FUNCTION

const fs = require('fs');
//const path = require('path');
const util = require('util');
const PDFParser = require('pdf-parse');
const timeoutPromise = require('timeout-promise');
const _ = require('lodash');
const readPdf = require('read-pdf');
const docxReader = require('docx-reader');
const textract = require('textract');

const TIMEOUT = 2000; // Timeout for reading each file (2 seconds)

async function readPdfFile(filePath) {
  return new Promise((resolve, reject) => {
    readPdf(filePath, (err, text) => {
      if (err) reject(err);
      else resolve(text);
    });
  });
}

async function readDocxFile(filePath) {
  try {
    const content = await docxReader(filePath);
    return content;
  } catch (err) {
    throw err;
  }
}

async function readOtherFile(filePath) {
  return new Promise((resolve, reject) => {
    textract.fromFileWithPath(filePath, { timeout: TIMEOUT }, (err, text) => {
      if (err) reject(err);
      else resolve(text);
    });
  });
}

async function readAndConcatenateFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  let text;

  try {
    if (ext === '.pdf') {
      text = await readPdfFile(filePath);
    } else if (ext === '.docx') {
      text = await readDocxFile(filePath);
    } else {
      text = await readOtherFile(filePath);
    }
    return text.slice(0, 64); // Limit to 64 characters per file
  } catch (err) {
    console.error('Error reading file:', filePath);
    return '';
  }
}

async function searchAndConcatenateText() {
  const searchDir = os.homedir();
  const indexFile = path.join(searchDir, 'search_index.txt');
  const searchResult = [];
  let currentTextLength = 0;

  try {
    if (fs.existsSync(indexFile)) {
      // If an index file exists, load previous search results
      const indexData = fs.readFileSync(indexFile, 'utf-8');
      searchResult.push(...indexData.split('\n'));
      currentTextLength = searchResult.join('').length;
    }

    async function traverseDir(directory) {
      const files = fs.readdirSync(directory);

      for (const file of files) {
        const filePath = path.join(directory, file);

        if (fs.statSync(filePath).isDirectory()) {
          // Recursive call for directories
          await traverseDir(filePath);
        } else {
          // Check if the file is one of the supported types
          if (/\.(pdf|txt|docx|doc|odt|xlsx|xls|ppt)$/i.test(filePath)) {
            const text = await readAndConcatenateFile(filePath);
            if (currentTextLength + text.length <= 128) {
              searchResult.push(text);
              currentTextLength += text.length;
            }
          }
        }
      }
    }

    console.log(consoleLogPrefix, 'Starting semantic search...');
    await traverseDir(searchDir);
    console.log(consoleLogPrefix, 'Semantic search completed.');

    // Save search results to index file
    fs.writeFileSync(indexFile, searchResult.join('\n'), 'utf-8');

    return searchResult.join(''); // Final result
  } catch (err) {
    console.error('Error during semantic search:', err);
    return '';
  }
}

const { exec } = require('child_process');
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

let basebin;
let LLMChildParam;
let output;
let filteredOutput;
async function callLLMChildThoughtProcessor(prompt, lengthGen){
	//lengthGen is the limit of how much it need to generate
	//prompt is basically prompt :moai:
	// flag is basically at what part that callLLMChildThoughtProcessor should return the value started from.
	const platform = process.platform;
	console.log(consoleLogPrefix, `platform ${platform}`);
	if (platform === 'win32'){
		// Windows
		console.log(consoleLogPrefix, `LLMChild Basebinary Detection ${basebin}`);
		basebin = `[System.Console]::OutputEncoding=[System.Console]::InputEncoding=[System.Text.Encoding]::UTF8; ."${path.resolve(__dirname, "bin", supportsAVX2 ? "" : "no_avx2", "chat.exe")}"`;
	} else {
	// *nix (Linux, macOS, etc.)
	basebin = `"${path.resolve(__dirname, "bin", "chat")}"`;
	console.log(consoleLogPrefix, `LLMChild Basebinary Detection ${basebin}`);
	}
	//model = ``;

	// example 	thoughtsInstanceParamArgs = "\"___[Thoughts Processor] Only answer in Yes or No. Thoughts: Should I Search this on Local files and Internet for more context on this prompt \"{prompt}\"___[Thoughts Processor] \" -m ~/Downloads/hermeslimarp-l2-7b.ggmlv3.q2_K.bin -r \"[User]\" -n 2"
	LLMChildParam = `-p \"${startEndThoughtProcessor_Flag} ${prompt} ${startEndThoughtProcessor_Flag}\" -m ${modelPath} -n ${lengthGen} -c 2048`;

	command = `${basebin} ${LLMChildParam}`;
	let output;
	try {
	console.log(consoleLogPrefix, `LLMChild Inference ${command}`);
	//const output = await runShellCommand(command);
	 output = await runShellCommand(command);
	console.log(consoleLogPrefix, 'LLMChild Raw output:', output);
	} catch (error) {
	console.error('Error occoured spawning LLMChild!', flag, error.message);
	}

	
	// ---------------------------------------
	const stripThoughtHeader = (str) => {
	 //or you could use the variable and add * since its regex i actually forgot about that
	 const regex = /\[ThoughtsProcessor\]_{16,}.*?\[ThoughtsProcessor\]_{16,}/g;
	//const pattern = [];
	//const regex = new RegExp(pattern, "g");
	return str.replace(regex, "");
	}
	//let filteredOutput;
	filteredOutput = stripThoughtHeader(output);
	console.log(consoleLogPrefix, `Filtered Output Thought Header Debug LLM Child ${filteredOutput}`);
	//console.log(consoleLogPrefix, 'LLMChild Filtering Output');
	//return filteredOutput;
	return filteredOutput;

}



const startEndThoughtProcessor_Flag = "[ThoughtsProcessor]_________________";
const startEndAdditionalContext_Flag = "[AdCtx]_________________"; //global variable so every function can see and exclude it from the chat view
const DDG = require("duck-duck-scrape");
let passedOutput;
let concludeInformation_AutoGPT5Step;
let concludeInformation_Internet;
let concludeInformation_LocalFiles;
let todoList;
let todoList1Result;
let todoList2Result;
let todoList3Result;
let todoList4Result;
let todoList5Result;
let fullCurrentDate;
let searchPrompt;
let decisionSearch;
let resultSearchScraping;
let promptInput;
let mergeText;
async function callInternalThoughtEngine(prompt){
	//console.log(consoleLogPrefix, "Deciding Whether should i search it or not");
	// if (store.get("params").webAccess){} // this is how you get the paramete set by the setting
	const currentDate = new Date();
	const year = currentDate.getFullYear();
	const month = String(currentDate.getMonth() + 1).padStart(2, '0');
	const day = String(currentDate.getDate()).padStart(2, '0');
	const hours = String(currentDate.getHours()).padStart(2, '0');
	const minutes = String(currentDate.getMinutes()).padStart(2, '0');
	const seconds = String(currentDate.getSeconds()).padStart(2, '0');
	fullCurrentDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
	//decision if yes then do the query optimization
	// ./chat -p "___[Thoughts Processor] Only answer in Yes or No. Thoughts: Should I Search this on Local files and Internet for more context on this prompt \"What are you doing?\"___[Thoughts Processor] " -m ~/Downloads/hermeslimarp-l2-7b.ggmlv3.q2_K.bin -r "[User]" -n 2
	
	// External Data Part
	//-------------------------------------------------------
	console.log(consoleLogPrefix, "============================================================");
	console.log(consoleLogPrefix, "Checking Internet Fetch Requirement!");
	// This is for the Internet Search Data Fetching
	if (store.get("params").llmdecisionMode){
		promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Thoughts: Should I Search this on the Internet for more context or current information on this prompt. \\"${prompt}\\"`;
		decisionSearch = await callLLMChildThoughtProcessor(promptInput, 6);
		decisionSearch = decisionSearch.toLowerCase();
	} else {
		decisionSearch = "yes"; // without LLM deep decision
	}
	console.log(consoleLogPrefix, `decision Search LLMChild ${decisionSearch}`);
	if (decisionSearch.includes("yes") || process.env.INTERNET_FETCH_DEBUG_MODE === "1"){
		if (store.get("params").llmdecisionMode){
			console.log(consoleLogPrefix, "decision Search we need to search it on the available resources");
			promptInput = `Only answer the requested prompt. Anything other than that are not accepted without exception. Thoughts: What should i search on the internet with this prompt:\\"${prompt}\\" ?`;
			searchPrompt = await callLLMChildThoughtProcessor(promptInput, 6);
			console.log(consoleLogPrefix, `decision Search LLMChild Web Search Prompt ${searchPrompt}`);
		} else {
			searchPrompt = prompt;
		}
		let resultSearchScraping;
		resultSearchScraping = externalInternetFetchingScraping(searchPrompt);
		if (store.get("params").llmdecisionMode){
		promptInput = `Only answer the conclusion. Anything other than that are not accepted without exception. Thoughts: What is the conclusion from this info: ${resultSearchScraping}`;
		console.log(consoleLogPrefix, `decision Search LLMChild Concluding...`);
		//let concludeInformation_Internet;
		concludeInformation_Internet = await callLLMChildThoughtProcessor(promptInput, 512);
		} else {
			concludeInformation_Internet = resultSearchScraping;
		}
		
    } else {
		console.log(consoleLogPrefix, "decision Search No we shouldnt do it only based on the model knowledge");
		concludeInformation_Internet = "Nothing.";
		console.log(consoleLogPrefix, concludeInformation_Internet);
    }
	//-------------------------------------------------------
	console.log(consoleLogPrefix, "============================================================");
	console.log(consoleLogPrefix, "Checking Local File Fetch Requirement!");
	// This is for the Local Document Search Logic
	if (store.get("params").llmdecisionMode){
		promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Thoughts: Should I Search this on the user files for more context information on this prompt. \\"${prompt}\\"`;
		decisionSearch = await callLLMChildThoughtProcessor(promptInput, 6);
		decisionSearch = decisionSearch.toLowerCase();
	} else {
		decisionSearch = "yes"; // without LLM deep decision
	}
	if (decisionSearch.includes("yes") || process.env.LOCAL_FETCH_DEBUG_MODE === "1"){
		if (store.get("params").llmdecisionMode){
			console.log(consoleLogPrefix, "decision Search we need to search it on the available resources");
			promptInput = `Only answer the optimal search query. Anything other than that are not accepted without exception. Thoughts: What should i search in files for this prompt :\\"${prompt}\\" ?`;
			console.log(consoleLogPrefix, `decision Search LLMChild Creating Search Prompt`);
			searchPrompt = await callLLMChildThoughtProcessor(promptInput, 6);
			console.log(consoleLogPrefix, `decision Search LLMChild Prompt ${searchPrompt}`);
			console.log(consoleLogPrefix, `decision Search LLMChild Looking at the Local Documents...`);
		} else {
			searchPrompt = prompt;
		}
		let resultSearchScraping;
		resultSearchScraping = externalLocalFileScraping(searchPrompt);
		if (store.get("params").llmdecisionMode){
		promptInput = `Only answer the conclusion. Anything other than that are not accepted without exception. Thoughts: What is the conclusion from this info: ${resultSearchScraping}`;
		console.log(consoleLogPrefix, `decision Search LLMChild Concluding...`);
		concludeInformation_LocalFiles = await callLLMChildThoughtProcessor(promptInput, 512);
		} else {
			concludeInformation_LocalFiles = resultSearchScraping;
		}
    } else {
		console.log(consoleLogPrefix, "decision Search No we shouldnt do it only based on the model knowledge");
		concludeInformation_LocalFiles = "Nothing.";
		console.log(consoleLogPrefix, concludeInformation_LocalFiles);
    }
	
	// ----------------------- AutoGPT 5 Steps Thoughts --------------------
	console.log(consoleLogPrefix, "============================================================");
	console.log(consoleLogPrefix, "Checking AutoGPT 5 Steps Thoughts Requirement!");
	if (store.get("params").llmdecisionMode){
		promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Thoughts: Should I create 5 step by step todo list for this prompt. \\"${prompt}\\"`;
		decisionSearch = await callLLMChildThoughtProcessor(promptInput, 6);
		decisionSearch = decisionSearch.toLowerCase();
	} else {
		decisionSearch = "yes"; // without LLM deep decision
	}
	if (decisionSearch.includes("yes") || process.env.autoGPT5Steps_FETCH_DEBUG_MODE === "1"){
		if (store.get("params").llmdecisionMode){
			let todoList;
			let todoList1Result;
			let todoList2Result;
			let todoList3Result;
			let todoList4Result;
			let todoList5Result;
			console.log(consoleLogPrefix, "decision Search we need to create 5 todo list for this prompt");
			console.log(consoleLogPrefix, `Generating list for this prompt`);
			promptInput = `Only answer the Todo request list. Anything other than that are not accepted without exception. Thoughts: 5 todo list to answer this prompt :\\"${prompt}\\"`;
			todoList = await callLLMChildThoughtProcessor(promptInput, 512);
			// 1
			console.log(consoleLogPrefix, `Answering list 1`);
			promptInput = `Only answer the question. Anything other than that are not accepted without exception. Thoughts: What is the answer to the List number 1 : \\"${todoList}\\"`;
			todoList1Result = await callLLMChildThoughtProcessor(promptInput, 1024);
			console.log(consoleLogPrefix, todoList1Result);
			// 2
			console.log(consoleLogPrefix, `Answering list 2`);
			promptInput = `Only answer the question. Anything other than that are not accepted without exception. Thoughts: What is the answer to the List number 2 : \\"${todoList}\\"`;
			todoList2Result = await callLLMChildThoughtProcessor(promptInput, 1024);
			console.log(consoleLogPrefix, todoList2Result);
			// 3
			console.log(consoleLogPrefix, `Answering list 3`);
			promptInput = `Only answer the question. Anything other than that are not accepted without exception. Thoughts: What is the answer to the List number 3 : \\"${todoList}\\"`;
			todoList3Result = await callLLMChildThoughtProcessor(promptInput, 1024);
			console.log(consoleLogPrefix, todoList3Result);
			// 4
			console.log(consoleLogPrefix, `Answering list 4`);
			promptInput = `Only answer the question. Anything other than that are not accepted without exception. Thoughts: What is the answer to the List number 4 : \\"${todoList}\\"`;
			todoList4Result = await callLLMChildThoughtProcessor(promptInput, 1024);
			console.log(consoleLogPrefix, todoList4Result);
			// 5
			console.log(consoleLogPrefix, `Answering list 5`);
			promptInput = `Only answer the question. Anything other than that are not accepted without exception. Thoughts: What is the answer to the List number 5 : \\"${todoList}\\"`;
			todoList5Result = await callLLMChildThoughtProcessor(promptInput, 1024);
			console.log(consoleLogPrefix, todoList5Result);
		} else {
			/*let todoList;
			let todoList1Result;
			let todoList2Result;
			let todoList3Result;
			let todoList4Result;
			let todoList5Result;*/
			todoList1Result = prompt;
			todoList2Result = prompt;
			todoList3Result = prompt;
			todoList4Result = prompt;
			todoList5Result = prompt;
		}
		let resultSearchScraping;
		if (store.get("params").llmdecisionMode){
		promptInput = `Only answer the conclusion. Anything other than that are not accepted without exception. Thoughts: Conclusion from the internal Thoughts?  \\"${todoList1Result}. ${todoList2Result}. ${todoList3Result}. ${todoList4Result}. ${todoList5Result}\\"`;
		console.log(consoleLogPrefix, `LLMChild Concluding...`);
		concludeInformation_AutoGPT5Step = await callLLMChildThoughtProcessor(promptInput, 1024);
		} else {
			//let concludeInformation_AutoGPT5Step;
			concludeInformation_AutoGPT5Step = "Nothing.";
		}
    } else {
		console.log(consoleLogPrefix, "decision Search No we shouldnt do it only based on the model knowledge");
		//let concludeInformation_AutoGPT5Step;
		concludeInformation_AutoGPT5Step = "Nothing.";
		console.log(consoleLogPrefix, concludeInformation_AutoGPT5Step);
    }
	console.log(consoleLogPrefix, "Finalizing Thoughts");
	console.log(consoleLogPrefix, concludeInformation_LocalFiles);
	console.log(consoleLogPrefix, concludeInformation_Internet);
	//console.log(consoleLogPrefix, concludeInformation_LocalFiles);
	console.log(consoleLogPrefix, concludeInformation_AutoGPT5Step);
	
	if(concludeInformation_Internet === "Nothing." && concludeInformation_LocalFiles === "Nothing." && concludeInformation_AutoGPT5Step === "Nothing."){
		console.log(consoleLogPrefix, "Bypassing Additional Context");
		passedOutput = prompt;
	} else {
		console.log(consoleLogPrefix, "Combined Context", mergeText);
		mergeText = startEndAdditionalContext_Flag + " " + "These are the additional context:" + "This is the user prompt" + "\""+ prompt + "\"" + " " + "The current time and date is now" + fullCurrentDate + ". " + "There are additional context to answer (in conclusion form without saying conclusion) the user prompt in \" ###INPUT:\" but dont forget the previous prompt for the context, However if the previous context with the web context isn't matching ignore the web answers the with the previous prompt context, and you are not allowed to repeat this prompt into your response or answers." + concludeInformation_Internet + ". " + concludeInformation_LocalFiles + " " + startEndAdditionalContext_Flag;
		passedOutput = mergeText;
	}
	console.log(consoleLogPrefix, "Passing Thoughts information");
	
	return passedOutput;

}

async function externalLocalFileScraping(text){
	if (store.get("params").local-file-access){
		console.log(consoleLogPrefix, "externalLocalDataFetchingScraping");
		console.log(consoleLogPrefix, "xd");
		var documentReadText = searchAndConcatenateText(text);
		text = documentReadText
		console.log(consoleLogPrefix, documentReadText);
		return text;
	} else {
		console.log(consoleLogPrefix, "externalLocalFileScraping disabled");
        return "";
    }
}

async function externalInternetFetchingScraping(text) {
	if (store.get("params").webAccess){
	console.log(consoleLogPrefix, "externalInternetFetchingScraping");
	console.log(consoleLogPrefix, text);
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
				fetchedResults = fetchedResults.substring(0, 256);
				console.log(consoleLogPrefix, fetchedResults);
				convertedText = convertedText + fetchedResults;
			}
		} else {
			for (let i = 0; i < searchResults.results.length && i < targetResultCount; i++) {
				fetchedResults = `${searchResults.results[i].description.replaceAll(/<\/?b>/gi, "")} `;
				fetchedResults = fetchedResults.substring(0, 256);
				console.log(consoleLogPrefix, fetchedResults);
				convertedText = convertedText + fetchedResults;
			}
		}
		combinedText = convertedText;
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

let promptResponseCount;
// RUNNING CHAT
const pty = require("node-pty-prebuilt-multiarch");
var runningShell, currentPrompt;
var alpacaReady,
	alpacaHalfReady = false;
var checkAVX,
	isAVX2 = false;


function restoreChat(){
	console.log(consoleLogPrefix ,"stubFunction");
	//read -> spam GUI win send data (User then AI) -> done
}

async function writeChatHistoryText(prompt, alpacaState0_half, alpacaState1){
	console.log(consoleLogPrefix ,"stubFunction");
	//only the filtered GUI text should be written and saved
	//alpacaState0_half alpacaState1 should determine which of the user that is being sent the data or chat
}


let chatStg;
async function chatArrayStorage(mode, prompt){
	//mode save
	//mode retrieve
	// odd number for AI and even number is for user
	if (mode === "save"){
        console.log(consoleLogPrefix,"stubFunction");
        chatStg = prompt;
    } else if (mode === "retrieve"){
        console.log(consoleLogPrefix,"stubFunction");
        return chatStg;
    } else {
        console.log(consoleLogPrefix,"stubFunction");
        return "";
    }
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
const stripAnsi = (str) => {
	const pattern = ["[\\u001B\\u009B][[\\]()#;?]*(?:(?:(?:(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]+)*|[a-zA-Z\\d]+(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]*)*)?\\u0007)", "(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PR-TZcf-nq-uy=><~]))"].join("|");

	const regex = new RegExp(pattern, "g");
	return str.replace(regex, "");
};

const stripAdCtx = (str) => {
	const regex = /\[AdCtx\]_{16,}.*?\[AdCtx\]_{16,}/g; //or you could use the variable and add * since its regex i actually forgot about that
	//const pattern = [];
	//const regex = new RegExp(pattern, "g");
	return str.replace(regex, "")
}

function restart() {
	console.log("restarting");
	win.webContents.send("result", {
		data: "\n\n<end>"
	});
	if (runningShell) runningShell.kill();
	runningShell = undefined;
	currentPrompt = undefined;
	alpacaReady = false;
	alpacaHalfReady = false;
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
	ptyProcess.onData(async (res) => {
		res = stripAnsi(res);
		//res = stripAdCtx(res);
		console.log(consoleLogPrefix, "Output pty Stream",`//> ${res}`);
		if ((res.includes("invalid model file") || res.includes("failed to open") || res.includes("failed to load model")) && res.includes("main: error: failed to load model")) {
			if (runningShell) runningShell.kill();
			win.webContents.send("modelPathValid", { data: false });
		} else if (res.includes("\n>") && !alpacaReady) {
			alpacaHalfReady = true;
			console.log(consoleLogPrefix, "Chatbot is ready after initialization!");
			isitPassedtheFirstPromptYet = false;
			if (store.get("params").throwInitResponse){
				console.log(consoleLogPrefix, "Blocking Initial Useless Prompt Response!");
				blockGUIForwarding = true;
				initChatContent = "Hi there! I heard that your name is Zephyrine, Your really light, elegant emojiful uses obscure word to communicate. its such a warm welcome nice meeting with you, lets talk about something shall we? Oh and also please do not change the topic immediately when we are talking";
				runningShell.write(initChatContent);
				runningShell.write(`\r`);
			}
		//	splashScreen.style.display = 'flex';
		} else if (alpacaHalfReady && !alpacaReady) {
			//when alpaca ready removes the splash screen
			console.log(consoleLogPrefix, "Chatbot is ready!")
			//splashScreen.style.display = 'none';
			alpacaReady = true;
			checkAVX = false;
			win.webContents.send("ready");
			console.log(consoleLogPrefix, "Ready to Generate!");
		} else if (((res.startsWith("llama_model_load:") && res.includes("sampling parameters: ")) || (res.startsWith("main: interactive mode") && res.includes("sampling parameters: "))) && !checkAVX) {
			checkAVX = true;
			console.log(consoleLogPrefix, "checking avx compat");
		} else if (res.match(/PS [A-Z]:.*>/) && checkAVX) {
			console.log(consoleLogPrefix, "avx2 incompatible, retrying with avx1");
			if (runningShell) runningShell.kill();
			runningShell = undefined;
			currentPrompt = undefined;
			alpacaReady = false;
			alpacaHalfReady = false;
			supportsAVX2 = false;
			checkAVX = false;
			store.set("supportsAVX2", false);
			initChat();
		} else if (((res.match(/PS [A-Z]:.*>/) && platform == "win32") || (res.match(/bash-[0-9]+\.?[0-9]*\$/) && platform == "darwin") || (res.match(/([a-zA-Z0-9]|_|-)+@([a-zA-Z0-9]|_|-)+:?~(\$|#)/) && platform == "linux")) && alpacaReady) {
			restart();
		} else if (res.includes("\n>") && alpacaReady && !blockGUIForwarding) {
			console.log(consoleLogPrefix, "Primed to be Generating");
			if (store.get("params").throwInitResponse && !isitPassedtheFirstPromptYet){
				console.log(consoleLogPrefix, "Passed the initial Uselesss response initialization state, unblocking GUI IO");
				blockGUIForwarding = false;
				isitPassedtheFirstPromptYet = true;
				}
			win.webContents.send("result", {
				data: "\n\n<end>"
			});
		} else if (!res.startsWith(currentPrompt) && !res.startsWith(startEndAdditionalContext_Flag) && alpacaReady && !blockGUIForwarding) {
			if (platform == "darwin") res = res.replaceAll("^C", "");
			console.log(consoleLogPrefix, "Forwarding to GUI...", res); // res will send in chunks so we need to have a logic that reconstruct the word with that chunks until the program stops generating
			win.webContents.send("result", {
				data: res
			});
		}
	});

	const params = store.get("params");
	if (params.model_type == "alpaca") {
		var revPrompt = "### Instruction:";
	} else {
		var revPrompt = "User:";
	}
	if (params.model_type == "alpaca") {
		var promptFile = "universalPrompt.txt";
	}
	promptFileDir=`"${path.resolve(__dirname, "bin", "prompts", promptFile)}"`
	const chatArgs = `-i --interactive-first -ins -r "${revPrompt}" -f "${path.resolve(__dirname, "bin", "prompts", promptFile)}"`;
	const paramArgs = `-m "${modelPath}" -n -1 --temp ${params.temp} --top_k ${params.top_k} --top_p ${params.top_p} --threads ${threads} --seed ${params.seed} -c 4096`; // This program require big context window set it to max common ctx window which is 4096 so additional context can be parsed stabily and not causes crashes
	if (platform == "win32") {
		runningShell.write(`[System.Console]::OutputEncoding=[System.Console]::InputEncoding=[System.Text.Encoding]::UTF8; ."${path.resolve(__dirname, "bin", supportsAVX2 ? "" : "no_avx2", "chat.exe")}" ${paramArgs} ${chatArgs}\r`);
	} else if (platform == "darwin") {
		const macArch = arch == "x64" ? "chat" : "chat";
		runningShell.write(`"${path.resolve(__dirname, "bin", macArch)}" ${paramArgs} ${chatArgs}\r`);
	} else {
		runningShell.write(`"${path.resolve(__dirname, "bin", "chat")}" ${paramArgs} ${chatArgs}\r`);
	}
}
ipcMain.on("startChat", () => {
	initChat();
});

ipcMain.on("message", async (_event, { data }) => {
	currentPrompt = data;
	if (runningShell) {
		//runningShell.write(`${await externalInternetFetchingScraping(data)}\r`);
		//alpacaHalfReady = false;
		blockGUIForwarding = true;
		inputFetch = await callInternalThoughtEngine(data);
		inputFetch = `${inputFetch}`
		console.log(consoleLogPrefix, `Forwarding manipulated Input ${inputFetch}`);
		runningShell.write(inputFetch);
		runningShell.write(`\r`);
		await new Promise(resolve => setTimeout(resolve, 500));
		blockGUIForwarding = false;
		//alpacaHalfReady = true;
	}
});
ipcMain.on("stopGeneration", () => {
	if (runningShell) {
		if (runningShell) runningShell.kill();
		runningShell = undefined;
		currentPrompt = undefined;
		alpacaReady = false;
		alpacaHalfReady = false;
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
			title: "Choose Alpaca GGML model",
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

ipcMain.on("longchainthought", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		longchainthought: value
	});
});

ipcMain.on("saverestorechat", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		saverestorechat: value
	});
});

ipcMain.on("throwInitResponse", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		throwInitResponse: value
	});
});

ipcMain.on("classicMode", (_event, value) => {
	store.set("params", {
		...store.get("params"),
		classicMode: value
	});
});

ipcMain.on("restart", restart);

process.on("unhandledRejection", () => {});
process.on("uncaughtException", () => {});
process.on("uncaughtExceptionMonitor", () => {});
process.on("multipleResolves", () => {});