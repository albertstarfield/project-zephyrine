/*
DEBUGGING TIPS
For enabling the more verbose output of the console logs you could add this environment variable and set to "1"

INTERNET_FETCH_DEBUG_MODE
LOCAL_FETCH_DEBUG_MODE
CoTMultiSteps_FETCH_DEBUG_MODE
ptyChatStgDEBUG
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
const ipcRenderer = require("electron").ipcRenderer;
const contextBridge = require('electron').contextBridge;
const path = require("path");
require("@electron/remote/main").initialize();

const os = require("os");
const platform = os.platform();
const arch = os.arch();
const appName = "Project Zephyrine"
const colorReset = "\x1b[0m";
const colorBrightCyan = "\x1b[96m";
const consoleLogPrefix = `[${colorBrightCyan}${appName}_${platform}_${arch}${colorReset}]:`;
const assistantName = "Adelaide Zephyrine Charlotte"
const username = os.userInfo().username;


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
	// after main window and main process initialize the electron core send the global.username and global.assistantName to the global bot
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
			llmBackendMode: 'LLaMa2gguf',
			repeat_last_n: '64',
			repeat_penalty: '1.4',
			top_k: '10',
			top_p: '0.9',
			temp: '0.6',
			seed: '-1',
			webAccess: true,
			localAccess: true,
			llmdecisionMode: true,
			extensiveThought: true,
			SaveandRestorechat: false,
			throwInitResponse: true,
			classicMode: false,
			AttemptAccelerate: true,
			emotionalLLMChildengine: true,
			profilePictureEmotion: true,
			websearch_amount: '7',
			maxWebSearchChar: '1024',
			maxLocalSearchChar: '1024',
			maxLocalSearchPerFileChar: '512',
			keywordContentFileMatchPercentageThreshold: '27',
			hardwareLayerOffloading: '2',
			longChainThoughtNeverFeelenough: true	  
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
const fs = require("fs");
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

async function searchAndConcatenateText(searchText) {
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
	//console.log(consoleLogPrefix, result);
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
	definedSeed_LLMchild = `${randSeed}`;
	while(childLLMResultNotPassed){
		//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor Called", prompt);
		result = await callLLMChildThoughtProcessor_backend(prompt, lengthGen, definedSeed_LLMchild);
		if (await hasAlphabet(result)){
			childLLMResultNotPassed = false;
			//console.log(consoleLogPrefix, "Result detected", result);
			childLLMDebugResultMode = false;
		} else {
			definedSeed_LLMchild = generateRandomNumber(minRandSeedRange, maxRandSeedRange);
			llmChildfailureCountSum = llmChildfailureCountSum + 1;
			lengthGen = llmChildfailureCountSum + lengthGen;
			childLLMDebugResultMode = true;
			console.log(consoleLogPrefix, "No output detected, might be a bad model, retrying with new Seed!", definedSeed_LLMchild, "Previous Result",result, "Adjusting LengthGen Request to: ", lengthGen);
			console.log(consoleLogPrefix, "Failure LLMChild Request Counted: ", llmChildfailureCountSum);
			childLLMResultNotPassed = true;
		}
} 
	//console.log(consoleLogPrefix, "callLLMChildThoughtProcessor Result Passed");
	return result
}

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
	LLMChildParam = `-p \"Answer and continue this with Response: prefix after the __ \n ${startEndThoughtProcessor_Flag} ${prompt} ${startEndThoughtProcessor_Flag}\" -m ${modelPath} --temp ${store.get("params").temp} -n ${lengthGen} --threads ${threads} -c 2048 -s ${definedSeed_LLMchild} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`;
	command = `${basebin} ${LLMChildParam}`;
	//console.log(consoleLogPrefix, "______________callLLMChildThoughtProcessor_backend exec subprocess");
	try {
	//console.log(consoleLogPrefix, `LLMChild Inference ${command}`);
	outputLLMChild = await runShellCommand(command);
	if(childLLMDebugResultMode){
		console.log(consoleLogPrefix, 'LLMChild Raw output:', outputLLMChild);
	}
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
	filteredOutput = filteredOutput.replace(/\n/g, " ");
	filteredOutput = filteredOutput.replace(/__/g, '');
	filteredOutput = filteredOutput.replace(/"/g, '');
	filteredOutput = filteredOutput.replace(/`/g, '');
	filteredOutput = filteredOutput.replace(/\//g, '');
	filteredOutput = filteredOutput.replace(/'/g, '');
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
const DDG = require("duck-duck-scrape");
let passedOutput="";
let concludeInformation_CoTMultiSteps;
let required_CoTSteps;
let historyDistanceReq;
let concatenatedCoT="";
let concludeInformation_Internet;
let concludeInformation_LocalFiles;
let concludeInformation_chatHistory;
let todoList;
let todoListResult;
let fullCurrentDate;
let searchPrompt="";
let decisionSearch;
let decisionChatHistoryCTX;
let resultSearchScraping;
let promptInput;
let mergeText;
let inputPromptCounterSplit;
let inputPromptCounter = [];
let historyChatRetrieved = [];
let inputPromptCounterThreshold = 6;
let emotionlist;
let evaluateEmotionInteraction;
let emotionalEvaluationResult;
let reevaluateAdCtx;
let reevaluateAdCtxDecisionAgent;
function historyRequirementRetrieval(historyDistance){
	//chatArrayStorage("retrieve", "", false, false, chatStgOrder-1);
	console.log(consoleLogPrefix, `Retrieving Chat History with history Depth ${historyDistance}`);
	let str = "";
	for (let i = historyDistance; i >= 1; i--){
		if (i % 2 === 0) {
			str += `${username}: `; // even number on this version of zephyrine means the User that makes the prompt or user input prompt 
		  } else {
			str += `${assistantName}: `; // odd number on this version of zephyrine means the AI or the assistant is the one that answers
		  }
		  str += `${chatArrayStorage("retrieve", "", false, false, chatStgOrder-i)} \n`
		  //console.log(i + ": " + str);
		  //deduplicate string to reduce the size need to be submitted which optimizes the input size and bandwidth
		}
	console.log(consoleLogPrefix, "__historyRequirementRetrievalFlexResult \n", str);
	return str;
}
async function callInternalThoughtEngine(prompt){
	if (store.get("params").llmdecisionMode){
		// Ask on how many numbers of Steps do we need, and if the model is failed to comply then fallback to 5 steps
		promptInput = `${username}:${prompt}\n Based on your evaluation of the request submitted by ${username}, could you please ascertain the number of sequential steps, ranging from 1 to 50, necessary to acquire the relevant historical context to understand the present situation? Answer only in numbers:`;
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
	console.log(consoleLogPrefix, "Retrieving History!");
	historyChatRetrieved=historyRequirementRetrieval(historyDistanceReq);
	
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
		// ./chat -p "___[Thoughts Processor] Only answer in Yes or No. Should I Search this on Local files and Internet for more context on this chat \"What are you doing?\"___[Thoughts Processor] " -m ~/Downloads/hermeslimarp-l2-7b.ggmlv3.q2_K.bin -r "[User]" -n 2
		// Chat History Context Call
		// -----------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		// This is for the ChatCTX Request
		//&& store.get("params").hisChatCTX
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the Internet for more context or current information on this chat. ${historyChatRetrieved}\n${username}:${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction Should I Search this on the Previous session out of scope chat history for context Yes or No? Answer:`;
			console.log(consoleLogPrefix, "Checking Chat History Context Call Requirement!");
			decisionChatHistoryCTX = await callLLMChildThoughtProcessor(promptInput, 12);
			decisionChatHistoryCTX = decisionChatHistoryCTX.toLowerCase();
		} else {
			decisionChatHistoryCTX = "yes"; // without LLM deep decision
		}
		//console.log(consoleLogPrefix, ` LLMChild ${decisionSearch}`);
		// explanation on the inputPromptCounterThreshold
		// Isn't the decision made by LLM? Certainly, while LLM or the LLMChild contributes to the decision-making process, it lacks the depth of the main thread. This can potentially disrupt the coherence of the prompt context, underscoring the importance of implementing a safety measure like a word threshold before proceeding to the subsequent phase.
		if ((((decisionChatHistoryCTX.includes("yes") || decisionChatHistoryCTX.includes("yep") || decisionChatHistoryCTX.includes("ok") || decisionChatHistoryCTX.includes("valid") || decisionChatHistoryCTX.includes("should") || decisionChatHistoryCTX.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold )) || process.env.HISTORY_CTX_FETCH_DEBUG_MODE === "1") && store.get("params").hisChatCTX){
			if (store.get("params").llmdecisionMode){
				promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. What should i search on the Internet on this interaction to help with my answer when i dont know the answer? Answer:`;
				searchPrompt = await callLLMChildThoughtProcessor(promptInput, 69);
			}else{
				searchPrompt = `${historyChatRetrieved[2]}. ${historyChatRetrieved[1]}. ${prompt}`
				searchPrompt = searchPrompt.replace(/None\./g, "");
			}
			console.log(consoleLogPrefix, `LLMChild Creating Search Prompt`);
			resultSearchScraping = await externalInternetFetchingScraping(searchPrompt);
			if (store.get("params").llmdecisionMode){
				inputPromptCounterSplit = resultSearchScraping.split(" ");
				inputPromptCounter[3] = inputPromptCounterSplit.length;
			if (resultSearchScraping && inputPromptCounter[3] > inputPromptCounterThreshold){
				resultSearchScraping = stripProgramBreakingCharacters(resultSearchScraping);
				promptInput = `What is the conclusion from this info: ${resultSearchScraping} Conclusion:`;
				console.log(consoleLogPrefix, `LLMChild Concluding...`);
				//let concludeInformation_Internet;
				concludeInformation_Internet = await callLLMChildThoughtProcessor(stripProgramBreakingCharacters(stripProgramBreakingCharacters(promptInput)), 1024);
			} else {
				concludeInformation_Internet = "Nothing.";
			}
			} else {
				concludeInformation_Internet = resultSearchScraping;
			}
		} else {
			console.log(consoleLogPrefix, "No Dont need to search it on the internet");
			concludeInformation_Internet = "Nothing.";
			console.log(consoleLogPrefix, concludeInformation_Internet);
		}
		// External Data Part
		//-------------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		
		// This is for the Internet Search Data Fetching
		if (store.get("params").llmdecisionMode && store.get("params").webAccess){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the Internet for more context or current information on this chat. ${historyChatRetrieved}\n${username}:${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction Should I Search this on the Internet Yes or No? Answer:`;
			console.log(consoleLogPrefix, "Checking Internet Fetch Requirement!");
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 12);
			decisionSearch = decisionSearch.toLowerCase();
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}
		//console.log(consoleLogPrefix, ` LLMChild ${decisionSearch}`);
		// explanation on the inputPromptCounterThreshold
		// Isn't the decision made by LLM? Certainly, while LLM or the LLMChild contributes to the decision-making process, it lacks the depth of the main thread. This can potentially disrupt the coherence of the prompt context, underscoring the importance of implementing a safety measure like a word threshold before proceeding to the subsequent phase.
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold )) || process.env.INTERNET_FETCH_DEBUG_MODE === "1") && store.get("params").webAccess){
			if (store.get("params").llmdecisionMode){
				promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. Do i have the knowledge to answer this then if i dont have the knowledge should i search it on the internet? Answer:`;
				searchPrompt = await callLLMChildThoughtProcessor(promptInput, 69);
			}else{
				searchPrompt = `${historyChatRetrieved[2]}. ${historyChatRetrieved[1]}. ${prompt}`
				searchPrompt = searchPrompt.replace(/None\./g, "");
			}
			

			//promptInput = ` ${historyChatRetrieved}\n${username}:${prompt}\n. With this interaction What search query for i search in google for the interaction? Search Query:`;
			//searchPrompt = await callLLMChildThoughtProcessor(promptInput, 64);
			console.log(consoleLogPrefix, `LLMChild Creating Search Prompt`);
			resultSearchScraping = await externalInternetFetchingScraping(searchPrompt);
			if (store.get("params").llmdecisionMode){
				inputPromptCounterSplit = resultSearchScraping.split(" ");
				inputPromptCounter[3] = inputPromptCounterSplit.length;
			if (resultSearchScraping && inputPromptCounter[3] > inputPromptCounterThreshold){
				resultSearchScraping = stripProgramBreakingCharacters(resultSearchScraping);
				promptInput = `What is the conclusion from this info: ${resultSearchScraping} Conclusion:`;
				console.log(consoleLogPrefix, `LLMChild Concluding...`);
				//let concludeInformation_Internet;
				concludeInformation_Internet = await callLLMChildThoughtProcessor(stripProgramBreakingCharacters(stripProgramBreakingCharacters(promptInput)), 1024);
			} else {
				concludeInformation_Internet = "Nothing.";
			}
			} else {
				concludeInformation_Internet = resultSearchScraping;
			}
		} else {
			console.log(consoleLogPrefix, "No Dont need to search it on the internet");
			concludeInformation_Internet = "Nothing.";
			console.log(consoleLogPrefix, concludeInformation_Internet);
		}

		//-------------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		
		// This is for the Local Document Search Logic
		if (store.get("params").llmdecisionMode && store.get("params").localAccess){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the user files for more context information on this chat ${historyChatRetrieved}\n${username}:${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction do i have the knowledge to answer this? Should I Search this on the Local Documents Yes or No? Answer:`;
			console.log(consoleLogPrefix, "Checking Local File Fetch Requirement!");
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 18);
			decisionSearch = decisionSearch.toLowerCase();
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}

		//localAccess variable must be taken into account
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold)) || process.env.LOCAL_FETCH_DEBUG_MODE === "1") && store.get("params").localAccess){
			if (store.get("params").llmdecisionMode){
				console.log(consoleLogPrefix, "We need to search it on the available resources");
				promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction do i have the knowledge to answer this if not what should i search on the local file then:`;
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
			concludeInformation_LocalFiles = "Nothing.";
		}
			} else {
				concludeInformation_LocalFiles = resultSearchScraping;
			}
		} else {
			console.log(consoleLogPrefix, "No, we shouldnt do it only based on the model knowledge");
			concludeInformation_LocalFiles = "Nothing.";
			console.log(consoleLogPrefix, concludeInformation_LocalFiles);
		}
		

		// ----------------------- CoT Steps Thoughts --------------------
		console.log(consoleLogPrefix, "============================================================");
		console.log(consoleLogPrefix, "Checking Chain of Thoughts Depth requirement Requirement!");
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I create 5 step by step todo list for this interaction ${historyChatRetrieved}\n${username}:${prompt}\n Your Response:`;
			promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. For the additional context this is what i concluded from Internet ${concludeInformation_Internet}. \n This is what i concluded from the Local Files ${concludeInformation_LocalFiles}. \n From this Interaction and additional context Should I Answer this in 5 steps Yes or No? Answer only in Numbers:`;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 32);
			decisionSearch = decisionSearch.toLowerCase();
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold )) || process.env.CoTMultiSteps_FETCH_DEBUG_MODE === "1") && store.get("params").extensiveThought){
			if (store.get("params").llmdecisionMode){
				
				// Ask on how many numbers of Steps do we need, and if the model is failed to comply then fallback to 5 steps
				// required_CoTSteps
				promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n From this context from 1 to 27 how many steps that is required to answer. Answer:`;
				required_CoTSteps = await callLLMChildThoughtProcessor(promptInput, 16);
				required_CoTSteps = onlyAllowNumber(required_CoTSteps);
				console.log(consoleLogPrefix, "Required CoT steps", required_CoTSteps);

				if (isVariableEmpty(required_CoTSteps)){
					required_CoTSteps = 5;
					console.log(consoleLogPrefix, "CoT Steps Retrieval Failure due to model failed to comply, Falling back")
				}
				console.log(consoleLogPrefix, "We need to create thougts instruction list for this prompt");
				console.log(consoleLogPrefix, `Generating list for this prompt`);
				promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n From this chat List ${required_CoTSteps} steps on how to Answer it. Answer:`;
				promptInput = stripProgramBreakingCharacters(promptInput);
				todoList = await callLLMChildThoughtProcessor(promptInput, 512);

				for(let iterate = 1; iterate <= required_CoTSteps; iterate++){
					console.log(consoleLogPrefix, "Processing Chain of Thoughts Step", iterate);
					promptInput = ` What is the answer to the List number ${iterate} : ${todoList} Answer/NextStep:"`;
					promptInput = stripProgramBreakingCharacters(promptInput);
					todoListResult = stripProgramBreakingCharacters(await callLLMChildThoughtProcessor(promptInput, 1024));
					concatenatedCoT = concatenatedCoT + ". " + todoListResult;
					console.log(consoleLogPrefix, iterate, "Result: ", todoListResult);
				}
			} else {
				concatenatedCoT = prompt;
			}
			let resultSearchScraping;
			if (store.get("params").llmdecisionMode){
			promptInput = `Conclusion from the internal Thoughts?  \\"${concatenatedCoT}\\" Conclusion:"`;
			console.log(consoleLogPrefix, `LLMChild Concluding...`);
			promptInput = stripProgramBreakingCharacters(promptInput);
			concludeInformation_CoTMultiSteps = stripProgramBreakingCharacters(await callLLMChildThoughtProcessor(promptInput, 1024));
			} else {
				//let concludeInformation_CoTMultiSteps;
				concludeInformation_CoTMultiSteps = "Nothing.";
			}
		} else {
			console.log(consoleLogPrefix, "No, we shouldnt do it only based on the model knowledge");
			//let concludeInformation_CoTMultiSteps;
			concludeInformation_CoTMultiSteps = "Nothing.";
			console.log(consoleLogPrefix, concludeInformation_CoTMultiSteps);
		}
		console.log(consoleLogPrefix, "============================================================");
			console.log(consoleLogPrefix, "Executing LLMChild Emotion Engine!");
			
			emotionlist = "Happy, Sad, Fear, Anger, Disgust";
			if (store.get("params").emotionalLLMChildengine){
				promptInput = `${historyChatRetrieved}\n${username}:${prompt}\n. From this conversation which from the following emotions ${emotionlist} are the correct one? Answer:`;
				console.log(consoleLogPrefix, `LLMChild Evaluating Interaction With Emotion Engine...`);
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
		if((concludeInformation_Internet === "Nothing." || concludeInformation_Internet === "undefined" ) && (concludeInformation_LocalFiles === "Nothing." || concludeInformation_LocalFiles === "undefined") && (concludeInformation_CoTMultiSteps === "Nothing." || concludeInformation_CoTMultiSteps === "undefined") && (concludeInformation_chatHistory === "Nothing." || concludeInformation_chatHistory === "undefined")){
			console.log(consoleLogPrefix, "Bypassing Additional Context");
			passedOutput = prompt;
		} else {
			concludeInformation_Internet = concludeInformation_Internet === "Nothing." ? "" : concludeInformation_Internet;
			concludeInformation_LocalFiles = concludeInformation_LocalFiles === "Nothing." ? "" : concludeInformation_LocalFiles;
			concludeInformation_CoTMultiSteps = concludeInformation_CoTMultiSteps === "Nothing." ? "" : concludeInformation_CoTMultiSteps;
			mergeText = startEndAdditionalContext_Flag + " " + `This is the ${username} prompt ` + "\""+ prompt + "\"" + " " + "These are the additonal context, but DO NOT mirror the Additional Context: " + ` Your feeling is now in \"${emotionalEvaluationResult}\", ` + "The current time and date is now: " + fullCurrentDate + ". "+ "There are additional context to answer: " + concludeInformation_Internet + concludeInformation_LocalFiles + concludeInformation_CoTMultiSteps + startEndAdditionalContext_Flag;
			mergeText = mergeText.replace(/\n/g, " "); //.replace(/\n/g, " ");
			passedOutput = mergeText;
			console.log(consoleLogPrefix, "Combined Context", mergeText);
		}
		console.log(consoleLogPrefix, "Passing Thoughts information");
		}else{
			passedOutput = prompt;
		}
		if(store.get("params").longChainThoughtNeverFeelenough && store.get("params").llmdecisionMode){
			promptInput = `This is the conversation ${historyChatRetrieved}\n${username}:${prompt}\n. While this is the context \n The current time and date is now: ${fullCurrentDate}, Answers from the internet ${concludeInformation_Internet}. and this is Answer from the Local Files ${concludeInformation_LocalFiles}. And finally this is from the Chain of Thoughts result ${concludeInformation_CoTMultiSteps}. \n From the user named ${username} did i or will i make the user said that my respond is bad, horrible, not professional, trash? Answer only with Yes or No? Answer:`;
			console.log(consoleLogPrefix, `LLMChild Evaluating Information PostProcess`);
			reevaluateAdCtxDecisionAgent = stripProgramBreakingCharacters(await callLLMChildThoughtProcessor(promptInput, 64));
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
	console.log(consoleLogPrefix, passedOutput);
	return passedOutput;

}

async function externalLocalFileScraping(text){
	console.log(consoleLogPrefix, "called Local File Scraping");
	if (store.get("params").localAccess){
		console.log(consoleLogPrefix, "externalLocalDataFetchingScraping");
		//console.log(consoleLogPrefix, "xd");
		var documentReadText = searchAndConcatenateText(text);
		text = documentReadText.replace("[object Promise]", "");
		console.log(consoleLogPrefix, documentReadText);
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
let maxRandSeedRange=99999999;
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
let LLMBackendVariationFileName;
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

	if(store.get("params").AttemptAccelerate){
		basebinLLMBackendParamPassedDedicatedHardwareAccel=`--n-gpu-layers ${store.get("params").hardwareLayerOffloading}`;
	} else {
		basebinLLMBackendParamPassedDedicatedHardwareAccel="";
	}

	LLMBackendSelection = store.get("params").llmBackendMode;
	/*
	<option value="LLaMa2">LLaMa-2</option>
									<option value="falcon">Falcon</option>
									<option value="mpt">MPT</option>
									<option value="GPTNeoX">GPT-NEO-X</option>
									<option value="starcoder">Starcoder</option>
									<option value="gptj">gpt-j</option>
									<option value="gpt2">gpt-2</option>
	*/
	if (!LLMBackendSelection){
		console.error(consoleLogPrefix, "LLM Backend Selection Failed, falling back to Original LLaMa backend")
		LLMBackendSelection = "LLaMa2";
	}
	if (LLMBackendSelection === "LLaMa2"){
		LLMBackendVariationFileName = "llama";
		LLMBackendVariationFileSubFolder = "llama";
	}else if (LLMBackendSelection === "falcon"){
		LLMBackendVariationFileName = "falcon";
		LLMBackendVariationFileSubFolder = "falcon";
	}else if (LLMBackendSelection === "mpt"){
		LLMBackendVariationFileName = "mpt";
		LLMBackendVariationFileSubFolder = "ggml-mpt";
	}else if (LLMBackendSelection === "GPTNeoX"){
		LLMBackendVariationFileName = "gptneox";
		LLMBackendVariationFileSubFolder = "ggml-gptneox";
	}else if (LLMBackendSelection === "starcoder"){
		LLMBackendVariationFileName = "starcoder";
		LLMBackendVariationFileSubFolder = "ggml-starcoder";
	}else if (LLMBackendSelection === "gptj"){
		LLMBackendVariationFileName = "gptj";
		LLMBackendVariationFileSubFolder = "ggml-gptj";
	}else if (LLMBackendSelection === "gpt2"){
		LLMBackendVariationFileName = "gpt2";
		LLMBackendVariationFileSubFolder = "ggml-gpt2";
	}else if (LLMBackendSelection === "LLaMa2gguf"){
		LLMBackendVariationFileName = "llama-gguf";
		LLMBackendVariationFileSubFolder = "llama-gguf";
	}else {
		console.log(consoleLogPrefix, "Unsupported Backend", LLMBackendSelection);
        process.exit(1);
	}

	console.log(`Detected Platform: ${platform}`);
	console.log(`Detected Architecture: ${arch}`);
	console.log(`Detected LLMBackend: ${LLMBackendSelection}`);

	LLMBackendVariationSelected = `LLMBackend-${LLMBackendVariationFileName}`;

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
	console.log(consoleLogPrefix, "Base Binary Path", basebin);
	basebin = basebin.replace(" ","\ ");
	console.log(consoleLogPrefix, "Base Binary Path", basebin);

	return basebin;
}

//we're going to define basebin which define which binary to use
determineLLMBackend();
console.log(consoleLogPrefix, process.versions.modules);
const pty = require("node-pty");
var runningShell, currentPrompt;
var zephyrineReady,
	zephyrineHalfReady = false;
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


let chatStg = [];
let chatStgJson;
let chatStgPersistentPath = `${path.resolve(__dirname, "storage", "presistentInteractionMem.json")}`;
let chatStgOrder = 0;
let retrievedChatStg;
let chatStgOrderRequest;
let amiwritingonAIMessageStreamMode=false;
function chatArrayStorage(mode, prompt, AITurn, UserTurn, arraySelection){
	//mode save
	//mode retrieve
	//mode restore
	// odd number for AI and even number is for user
	if (chatStgOrder === 0){
		chatStg[0]="=========ChatStorageHeader========";
	}
	
	if (mode === "save"){
        //console.log(consoleLogPrefix,"Saving...");
		if(AITurn && !UserTurn){
			if(!amiwritingonAIMessageStreamMode){
				chatStgOrder = chatStgOrder + 1; // add the order number counter when we detect we did write it on AI mode and need confirmation, when its done we should add the order number
			}
			amiwritingonAIMessageStreamMode=true;
			
			//chatStgOrder = chatStgOrder + 1; //we can't do this kind of stuff because the AI stream part by part and call the chatArrayStorage in a continuous manner which auses the chatStgOrder number to go up insanely high and mess up with the storage
			// How to say that I'm done and stop appending to the same array and move to next Order?
			chatStg[chatStgOrder] += prompt; //handling the partial stream by appending into the specific array
			chatStg[chatStgOrder] = chatStg[chatStgOrder].replace("undefined", "");
			if (process.env.ptyChatStgDEBUG === "1"){
				console.log(consoleLogPrefix,"AITurn...");
				console.log(consoleLogPrefix, "reconstructing from pty stream: ", chatStg[chatStgOrder]);
			}
		}
		if(!AITurn && UserTurn){
			amiwritingonAIMessageStreamMode=false;
			
			chatStgOrder = chatStgOrder + 1; //
			chatStg[chatStgOrder] = prompt;
			if (process.env.ptyChatStgDEBUG === "1"){
				console.log(consoleLogPrefix,"UserTurn...");
				console.log(consoleLogPrefix, "stg pty stream user:", chatStg[chatStgOrder]);
			}
		}
		if (process.env.ptyChatStgDEBUG === "1"){
		console.log(consoleLogPrefix,"ChatStgOrder...", chatStgOrder);
		}
    } else if (mode === "retrieve"){
		console.log(consoleLogPrefix,"retrieving Chat Storage Order ", arraySelection);
		if (arraySelection >= 1 && arraySelection <= chatStgOrder)
        {
		retrievedChatStg = chatStg[arraySelection];
		} else {
			retrievedChatStg = "None.";
		}
		return retrievedChatStg;
    } else if (mode === "restoreLoadPersistentMem"){
		if (store.get("params").SaveandRestorechat) {
			console.log(consoleLogPrefix,"Restoring Chat... Reading from File to Array");
			chatStgOrder=0;
			chatStg = fs.readFile(filename, 'utf8', (err, data) => {
				if (err) {
				console.error('Error reading file:', err);
				callback([]);
				} else {
				try {
					const array = JSON.parse(data);
					callback(array);
				} catch (parseError) {
					console.error('Error parsing JSON:', parseError);
					callback([]);
				}
				}
			});
			console.log(consoleLogPrefix, chatStg);
			console.log(consoleLogPrefix, "Triggering Restoration Mode for the UI and main LLM Thread!");
			console.log(consoleLogPrefix, "Stub!");
		} else {
			console.log(consoleLogPrefix, "Save and Restore Chat Disabled");
		}

		// then after that set to the appropriate chatStgOrder from the header of the file
	} else if (mode === "reset"){
		console.log(consoleLogPrefix, "Resetting Temporary Storage Order and Overwriting to Null!");
		chatStgOrder=0;
		chatStg = [];
		
	} else if (mode === "flushPersistentMem"){
		if (store.get("params").SaveandRestorechat) {
			//example of the directory writing 
			//basebin = `"${path.resolve(__dirname, "bin", "2_Linux", "arm64", LLMBackendVariationFileSubFolder, basebinBinaryMoreSpecificPathResolve)}"`;
			console.log(consoleLogPrefix, "Stub Function but Trying Flushing into Mem!");
			console.log(consoleLogPrefix, chatStg);
			chatStgJson = JSON.stringify(chatStg)
			console.log(consoleLogPrefix, chatStgJson);
			console.log(consoleLogPrefix, chatStgPersistentPath)
			fs.writeFile(chatStgPersistentPath, chatStgJson, (err) => {
				if (err) {
				console.error(consoleLogPrefix, 'Error writing file:', err);
				} else {
				console.log(consoleLogPrefix, 'Array data saved to file:', chatStgPersistentPath);
				}
			});
		} else {
			console.log(consoleLogPrefix,"stubFunction");
			return "";
		}
	} else {
		console.log(consoleLogPrefix, "Save and Restore Chat disabled!")
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
	chatArrayStorage("flushPersistentMem", 0, 0, 0, 0); // Flush function/method test
	chatArrayStorage("reset", 0, 0, 0, 0); // fill it with 0 0 0 0 just to fill in nonsensical data since its only require reset command to execute the command
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
		res = stripProgramBreakingCharacters(res);
		//res = stripAdditionalContext(res);
		if (process.env.ptyStreamDEBUGMode === "1"){
		console.log(consoleLogPrefix, "pty Stream",`//> ${res}`);
		}
		//console.log(consoleLogPrefix, "debug", zephyrineHalfReady, zephyrineReady)
		if ((res.includes("invalid model file") || res.includes("failed to open") || (res.includes("failed to load model")) && res.includes("main: error: failed to load model")) || res.includes("command buffer 0 failed with status 5") /* Metal ggml ran out of memory vram */ || res.includes ("invalid magic number") || res.includes ("out of memory")) {
			if (runningShell) runningShell.kill();
			win.webContents.send("modelPathValid", { data: false });
		} else if (res.includes("\n>") && !zephyrineReady) {
			zephyrineHalfReady = true;
			console.log(consoleLogPrefix, "LLM Main Thread is ready after initialization!");
			isitPassedtheFirstPromptYet = false;
			if (store.get("params").throwInitResponse){
				console.log(consoleLogPrefix, "Blocking Initial Useless Prompt Response!");
				blockGUIForwarding = true;
				initChatContent = `Greetings and salutations, may the morning rays grace ${assistantName}! May ${username}'s day shine as brightly as the stars that adorn the sky. I am ${username}, extending my introduction; let our conversation find its footing as we engage in discourse, if you would be so kind.`;
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
		} else if ((res.includes("\n>") || res.includes("\n>\n")) && zephyrineReady && !blockGUIForwarding) {
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
			//chatArrayStorage(mode, prompt, AITurn, UserTurn, arraySelection)
			chatArrayStorage("save", res, true, false, 0);	// for saving you could just enter 0 on the last parameter, because its not really matter anyway when on save data mode
			//res = marked.parse(res);
			win.webContents.send("result", {
				data: res
			});
		}
	});

	const params = store.get("params");
	//revPrompt = "### Instruction: \n {prompt}";
	revPrompt = "‌‌ ";
	var promptFile = "universalPrompt.txt";
	promptFileDir=`"${path.resolve(__dirname, "bin", "prompts", promptFile)}"`
	const chatArgs = `-i --interactive-first -ins -r "${revPrompt}" -f "${path.resolve(__dirname, "bin", "prompts", promptFile)}"`;
	const paramArgs = `-m "${modelPath}" -n -1 --temp ${params.temp} --top_k ${params.top_k} --top_p ${params.top_p} --threads ${threads} -c 2048 -s ${randSeed} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`; // This program require big context window set it to max common ctx window which is 4096 so additional context can be parsed stabily and not causes crashes
	//runningShell.write(`set -x \r`);
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
		chatArrayStorage("save", data, false, true, 0);	// for saving you could just enter 0 on the last parameter, because its not really matter anyway when on save data mode
		blockGUIForwarding = true;
		inputFetch = await callInternalThoughtEngine(data);
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
		llmBackendMode: value
	});
});

ipcMain.on("restart", restart);

process.on("unhandledRejection", () => {});
process.on("uncaughtException", () => {});
process.on("uncaughtExceptionMonitor", () => {});
process.on("multipleResolves", () => {});