/*
DEBUGGING TIPS
For enabling the more verbose output of the console logs you could add this environment variable and set to "1"

INTERNET_FETCH_DEBUG_MODE
LOCAL_FETCH_DEBUG_MODE
autoGPT5Steps_FETCH_DEBUG_MODE
ptyChatStgDEBUG
ptyStreamDEBUGMode


*/



const { BrowserWindow, app, ipcMain, dialog } = require("electron");
const path = require("path");
require("@electron/remote/main").initialize();

const os = require("os");
const platform = os.platform();
const arch = os.arch();
const appName = "Project Zephyrine"
const colorReset = "\x1b[0m";
const colorBrightCyan = "\x1b[96m";
const consoleLogPrefix = `[${colorBrightCyan}${appName}_${platform}_${arch}${colorReset}]:`;
const botName = "Adelaide Zephyrine Charlotte"
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
			repeat_last_n: '64',
			repeat_penalty: '1.3',
			top_k: '40',
			top_p: '0.9',
			temp: '0.8',
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
			hardwareLayerOffloading: '2'		  
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
function searchAndConcatenateText(searchText) {
	const userHomeDir = require('os').homedir();
	const rootDirectory = path.join(userHomeDir, 'Documents'); // You can change the root directory as needed
  
	const result = [];
	let totalChars = 0;
  
	function searchInDirectory(directoryPath) {
	  const files = fs.readdirSync(directoryPath);
  
	  for (const file of files) {
		const filePath = path.join(directoryPath, file);
		const fileStats = fs.statSync(filePath);
  
		if (fileStats.isDirectory()) {
		  searchInDirectory(filePath);
		} else if (fileStats.isFile() && (path.extname(file) === '.txt' || path.extname(file) === '.rtf' || path.extname(file) === '.c' || path.extname(file) === '.py' || path.extname(file) === '.html' || path.extname(file) === '.css' || path.extname(file) === '.rb' || path.extname(file) === '.xml')) {
		  const content = stripAnsi(fs.readFileSync(filePath, 'utf-8'));
		  
		  // filter out files that only 69% and beyond are allowed to pass to the truncatedContent
		  const contentAnalyzer = content.split(' ');
		  const searchedTextWordAnalyzer = searchText.split(' ');
		  let matchedContentSearch = 0;
		  for (const word of searchedTextWordAnalyzer) {
			if (contentAnalyzer.includes(word)){
				matchedContentSearch++;
			}
		  }
		  const matchPercentage = ((matchedContentSearch/content.length)*100);
		  if (process.env.LOCAL_FETCH_DEBUG_MODE === '1') {
			console.log(`${consoleLogPrefix} Reading file: ${filePath} Match ${matchPercentage} Percent`);
		  }
		  if (matchPercentage > store.get("params").keywordContentFileMatchPercentageThreshold) {
			const truncatedContent = content.substring(0, store.get("params").maxLocalSearchPerFileChar);
			result.push(truncatedContent);
			totalChars += truncatedContent.length;
  
			if (totalChars >= store.get("params").maxLocalSearchChar) { //maxLocalSearchChar params
			  if (process.env.LOCAL_FETCH_DEBUG_MODE === '1') {
				console.log(`${consoleLogPrefix} Reached character limit. Stopping search.`);
			  }
			  return;
			}
		  }
		}
	  }
	}
  
	searchInDirectory(rootDirectory);
	
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

function stripAnsi(str) {
	const pattern = ["[\\u001B\\u009B][[\\]()#;?]*(?:(?:(?:(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]+)*|[a-zA-Z\\d]+(?:;[-a-zA-Z\\d\\/#&.:=?%@~_]*)*)?\\u0007)", "(?:(?:\\d{1,4}(?:;\\d{0,4})*)?[\\dA-PR-TZcf-nq-uy=><~]))"].join("|");
	const regex = new RegExp(pattern, "g");
	return str.replace(regex, "");
}

//let basebin;
let LLMChildParam;
let outputLLMChild;
let filteredOutput;
async function callLLMChildThoughtProcessor(prompt, lengthGen){
	//lengthGen is the limit of how much it need to generate
	//prompt is basically prompt :moai:
	// flag is basically at what part that callLLMChildThoughtProcessor should return the value started from.
	//const platform = process.platform;
	
	//model = ``;
	prompt = prompt.replace("[object Promise]", "");
	prompt = stripAnsi(prompt);
	prompt = prompt.replace(/"/g, '');
	prompt = prompt.replace(/'/g,"");

	// example 	thoughtsInstanceParamArgs = "\"___[Thoughts Processor] Only answer in Yes or No. Should I Search this on Local files and Internet for more context on this chat \"{prompt}\"___[Thoughts Processor] \" -m ~/Downloads/hermeslimarp-l2-7b.ggmlv3.q2_K.bin -r \"[User]\" -n 2"
	LLMChildParam = `-p \"${startEndThoughtProcessor_Flag} ${prompt} ${startEndThoughtProcessor_Flag}\" -m ${modelPath} -n ${lengthGen} --threads ${threads} -c 2048 -s ${randSeed} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`;

	command = `${basebin} ${LLMChildParam}`;
	
	try {
	//console.log(consoleLogPrefix, `LLMChild Inference ${command}`);
	outputLLMChild = await runShellCommand(command);
	//console.log(consoleLogPrefix, 'LLMChild Raw output:', outputLLMChild);
	} catch (error) {
	console.error('Error occoured spawning LLMChild!', flag, error.message);
	}	
	// ---------------------------------------
	function stripThoughtHeader(input) {
		const regex = /__([^_]+)__/g;
		return input.replace(regex, '');
	}

	filteredOutput = stripThoughtHeader(outputLLMChild);
	filteredOutput = filteredOutput.replace(/\n/g, " ");
	filteredOutput = filteredOutput.replace(/__/g, '');
	filteredOutput = filteredOutput.replace(/"/g, '');
	filteredOutput = filteredOutput.replace(/`/g, '');
	filteredOutput = filteredOutput.replace(/'/g, '');
	console.log(consoleLogPrefix, `LLMChild Thread Output ${filteredOutput}`); // filtered output
	//console.log(consoleLogPrefix, 'LLMChild Filtering Output');
	//return filteredOutput;
	filteredOutput = stripAnsi(filteredOutput);
	return filteredOutput;

}



const startEndThoughtProcessor_Flag = "__"; // we can remove [ThoughtProcessor] word from the phrase to prevent any processing or breaking the chat context on the LLM and instead just use ___ three underscores
const startEndAdditionalContext_Flag = ""; //global variable so every function can see and exclude it from the chat view
// we can savely blank out the startEndAdditionalContext flag because we are now based on the blockGUI forwarding concept rather than flag matching
const DDG = require("duck-duck-scrape");
let passedOutput="";
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
let searchPrompt="";
let decisionSearch;
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
async function callInternalThoughtEngine(prompt){
	//console.log(consoleLogPrefix, "Deciding Whether should i search it or not");
	// if (store.get("params").webAccess){} // this is how you get the paramete set by the setting
	historyChatRetrieved[1]=chatArrayStorage("retrieve", "", false, false, chatStgOrder-1);
	historyChatRetrieved[2]=chatArrayStorage("retrieve", "", false, false, chatStgOrder-2);
	
	// Counting the number of words in the Chat prompt and history
	inputPromptCounterSplit = prompt.split(" ");
	inputPromptCounter[0] = inputPromptCounterSplit.length;

	inputPromptCounterSplit = historyChatRetrieved[1].split(" ");
	inputPromptCounter[1] = inputPromptCounterSplit.length;

	inputPromptCounterSplit = historyChatRetrieved[2].split(" ");
	inputPromptCounter[2] = inputPromptCounterSplit.length;
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
		// External Data Part
		//-------------------------------------------------------
		console.log(consoleLogPrefix, "============================================================");
		console.log(consoleLogPrefix, "Checking Internet Fetch Requirement!");
		// This is for the Internet Search Data Fetching
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the Internet for more context or current information on this chat. \nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n Your Response:`;
			promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction Should I Search this on the Internet Yes or No? Answer:`;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 12);
			decisionSearch = decisionSearch.toLowerCase();
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}
		//console.log(consoleLogPrefix, ` LLMChild ${decisionSearch}`);
		// explanation on the inputPromptCounterThreshold
		// Isn't the decision made by LLM? Certainly, while LLM or the LLMChild contributes to the decision-making process, it lacks the depth of the main thread. This can potentially disrupt the coherence of the prompt context, underscoring the importance of implementing a safety measure like a word threshold before proceeding to the subsequent phase.
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold || inputPromptCounter[2] > inputPromptCounterThreshold)) || process.env.INTERNET_FETCH_DEBUG_MODE === "1") && store.get("params").webAccess){
			if (store.get("params").llmdecisionMode){
				promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. What should i search on the Internet on this interaction to help with my answer when i dont know the answer? Answer:`;
				searchPrompt = await callLLMChildThoughtProcessor(promptInput, 69);
			}else{
				searchPrompt = `${historyChatRetrieved[2]}. ${historyChatRetrieved[1]}. ${prompt}`
				searchPrompt = searchPrompt.replace(/None\./g, "");
			}
			

			//promptInput = ` \nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. With this interaction What search query for i search in google for the interaction? Search Query:`;
			//searchPrompt = await callLLMChildThoughtProcessor(promptInput, 64);
			console.log(consoleLogPrefix, `LLMChild Creating Search Prompt`);
			resultSearchScraping = await externalInternetFetchingScraping(searchPrompt);
			if (store.get("params").llmdecisionMode){
				inputPromptCounterSplit = resultSearchScraping.split(" ");
				inputPromptCounter[3] = inputPromptCounterSplit.length;
			if (resultSearchScraping && inputPromptCounter[3] > inputPromptCounterThreshold){
				resultSearchScraping = stripAnsi(resultSearchScraping);
				promptInput = `What is the conclusion from this info: ${resultSearchScraping} Conclusion:`;
				console.log(consoleLogPrefix, `LLMChild Concluding...`);
				//let concludeInformation_Internet;
				concludeInformation_Internet = await callLLMChildThoughtProcessor(stripAnsi(stripAnsi(promptInput)), 1024);
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
		console.log(consoleLogPrefix, "Checking Local File Fetch Requirement!");
		// This is for the Local Document Search Logic
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I Search this on the user files for more context information on this chat \nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n Your Response:`;
			promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction Should I Search this on the Local Documents Yes or No? Answer:`;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 18);
			decisionSearch = decisionSearch.toLowerCase();
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}

		//localAccess variable must be taken into account
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold || inputPromptCounter[2] > inputPromptCounterThreshold)) || process.env.LOCAL_FETCH_DEBUG_MODE === "1") && store.get("params").localAccess){
			if (store.get("params").llmdecisionMode){
				console.log(consoleLogPrefix, "We need to search it on the available resources");
				promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. From this Interaction what should i search that i dont know Answer:`;
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
		

		// ----------------------- AutoGPT 5 Steps Thoughts --------------------
		console.log(consoleLogPrefix, "============================================================");
		console.log(consoleLogPrefix, "Checking AutoGPT 5 Steps Thoughts Requirement!");
		if (store.get("params").llmdecisionMode){
			//promptInput = `Only answer in one word either Yes or No. Anything other than that are not accepted without exception. Should I create 5 step by step todo list for this interaction \nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n Your Response:`;
			promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. With the previous Additional Context is ${passedOutput}\n. For the additional context this is what i concluded from Internet ${concludeInformation_Internet}. \n This is what i concluded from the Local Files ${concludeInformation_LocalFiles}. \n From this Interaction and additional context Should I Answer this in 5 steps Yes or No? Answer:`;
			decisionSearch = await callLLMChildThoughtProcessor(promptInput, 6);
			decisionSearch = decisionSearch.toLowerCase();
		} else {
			decisionSearch = "yes"; // without LLM deep decision
		}
		if ((((decisionSearch.includes("yes") || decisionSearch.includes("yep") || decisionSearch.includes("ok") || decisionSearch.includes("valid") || decisionSearch.includes("should") || decisionSearch.includes("true")) && (inputPromptCounter[0] > inputPromptCounterThreshold || inputPromptCounter[1] > inputPromptCounterThreshold || inputPromptCounter[2] > inputPromptCounterThreshold)) || process.env.autoGPT5Steps_FETCH_DEBUG_MODE === "1") && store.get("params").extensiveThought){
			if (store.get("params").llmdecisionMode){
				console.log(consoleLogPrefix, "We need to create 5 todo list for this prompt");
				console.log(consoleLogPrefix, `Generating list for this prompt`);
				promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n From this chat List 5 steps on how to Answer it. Answer:`;
				promptInput = stripAnsi(promptInput);
				todoList = await callLLMChildThoughtProcessor(promptInput, 512);
				// 1
				console.log(consoleLogPrefix, `Answering Step 1`);
				promptInput = ` What is the answer to the List number 1 : ${todoList} Answer/NextStep:"`;
				promptInput = stripAnsi(promptInput);
				console.log(consoleLogPrefix, promptInput);
				todoList1Result = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 1024));
				console.log(consoleLogPrefix, todoList1Result);
				// 2x
				console.log(consoleLogPrefix, `Answering Step 2`);
				promptInput = `What is the answer to the List number 2 : ${todoList} Answer/NextStep:"`;
				promptInput = stripAnsi(promptInput);
				todoList2Result = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 1024));
				console.log(consoleLogPrefix, todoList2Result);
				// 3
				console.log(consoleLogPrefix, `Answering Step 3`);
				promptInput = `What is the answer to the List number 3 : ${todoList} Answer/NextStep:"`;
				promptInput = stripAnsi(promptInput);
				todoList3Result = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 1024));
				console.log(consoleLogPrefix, todoList3Result);
				// 4
				console.log(consoleLogPrefix, `Answering Step 4`);
				promptInput = `What is the answer to the List number 4 : ${todoList} Answer/NextStep:"`;
				promptInput = stripAnsi(promptInput);
				todoList4Result = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 1024));
				console.log(consoleLogPrefix, todoList4Result);
				// 5
				console.log(consoleLogPrefix, `Answering Step 5`);
				promptInput = `What is the answer to the List number 5 : ${todoList} Answer/NextStep:"`;
				promptInput = stripAnsi(promptInput);
				todoList5Result = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 1024));
				console.log(consoleLogPrefix, todoList5Result);
			} else {
				todoList1Result = prompt;
				todoList2Result = prompt;
				todoList3Result = prompt;
				todoList4Result = prompt;
				todoList5Result = prompt;
			}
			let resultSearchScraping;
			if (store.get("params").llmdecisionMode){
			promptInput = `Conclusion from the internal Thoughts?  \\"${todoList1Result}. ${todoList2Result}. ${todoList3Result}. ${todoList4Result}. ${todoList5Result}\\ Conclusion:"`;
			console.log(consoleLogPrefix, `LLMChild Concluding...`);
			promptInput = stripAnsi(promptInput);
			concludeInformation_AutoGPT5Step = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 1024));
			} else {
				//let concludeInformation_AutoGPT5Step;
				concludeInformation_AutoGPT5Step = "Nothing.";
			}
		} else {
			console.log(consoleLogPrefix, "No, we shouldnt do it only based on the model knowledge");
			//let concludeInformation_AutoGPT5Step;
			concludeInformation_AutoGPT5Step = "Nothing.";
			console.log(consoleLogPrefix, concludeInformation_AutoGPT5Step);
		}
		console.log(consoleLogPrefix, "============================================================");
			console.log(consoleLogPrefix, "Executing LLMChild Emotion Engine!");
			
			emotionlist = "Happy, Sad, Fear, Anger, Disgust";
			if (store.get("params").emotionalLLMChildengine){
				promptInput = `\nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. From this conversation which from the following emotions ${emotionlist} are the correct one? Answer:`;
				console.log(consoleLogPrefix, `LLMChild Evaluating Interaction With Emotion Engine...`);
				promptInput = stripAnsi(promptInput);
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

		
		if(concludeInformation_Internet === "Nothing." && concludeInformation_LocalFiles === "Nothing." && concludeInformation_AutoGPT5Step === "Nothing."){
			console.log(consoleLogPrefix, "Bypassing Additional Context");
			passedOutput = prompt;
		} else {
			concludeInformation_Internet = concludeInformation_Internet === "Nothing." ? "" : concludeInformation_Internet;
			concludeInformation_LocalFiles = concludeInformation_LocalFiles === "Nothing." ? "" : concludeInformation_LocalFiles;
			concludeInformation_AutoGPT5Step = concludeInformation_AutoGPT5Step === "Nothing." ? "" : concludeInformation_AutoGPT5Step;
			mergeText = startEndAdditionalContext_Flag + " " + "This is the user prompt " + "\""+ prompt + "\"" + " " + "These are the additional context: "  + "The current time and date is now: " + fullCurrentDate + ". " + "There are additional context to answer: " + concludeInformation_Internet + " " + concludeInformation_LocalFiles + " " + concludeInformation_AutoGPT5Step + startEndAdditionalContext_Flag;
			passedOutput = mergeText;
			console.log(consoleLogPrefix, "Combined Context", mergeText);
		}
		console.log(consoleLogPrefix, "Passing Thoughts information");
		}else{
			passedOutput = prompt;
		}
		if(store.get("params").longChainThoughtNeverFeelenough && store.get("params").llmdecisionMode){
			promptInput = `This is the conversation \nUser:${historyChatRetrieved[2]} \nResponse:${historyChatRetrieved[1]} \nUser:${prompt}\n. While this is the context \n The current time and date is now: ${fullCurrentDate}, Answers from the internet ${concludeInformation_Internet}. and this is Answer from the Local Files ${concludeInformation_LocalFiles}. And finally this is from the AutoGPT 5 Steps ${concludeInformation_AutoGPT5Step}. \n From this Information should I dig deeper and retry to get more clearer information to answer the chat? Yes or No? Answer:`;
			console.log(consoleLogPrefix, `LLMChild Evaluating Information PostProcess`);
			reevaluateAdCtxDecisionAgent = stripAnsi(await callLLMChildThoughtProcessor(promptInput, 32));
			if (reevaluateAdCtxDecisionAgent.includes("yes") || reevaluateAdCtxDecisionAgent.includes("yep") || reevaluateAdCtxDecisionAgent.includes("ok") || reevaluateAdCtxDecisionAgent.includes("valid") || reevaluateAdCtxDecisionAgent.includes("should") || reevaluateAdCtxDecisionAgent.includes("true")){
				reevaluateAdCtx = true;
				console.log(consoleLogPrefix, `Context isnt good enough! Still lower than standard!`);
				console.log(consoleLogPrefix, reevaluateAdCtxDecisionAgent);
			} else {
				console.log(consoleLogPrefix, `Passing Context!`);
				reevaluateAdCtx = false;
			}
		}else{
			reevaluateAdCtx = false;
		}
	}
	
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
if (configSeed === "-1"){ 
	randSeed = generateRandomNumber(270000000000, 279999999999);
	console.log(consoleLogPrefix, "Random Seed!", randSeed);
} else {
	randSeed = configSeed;
	console.log(consoleLogPrefix, "Predefined Seed!", randSeed);
}
// RUNNING Main LLM GUI to User
let LLMBackendSelection;
let LLMBackendVariationFileName;
let LLMBackendVariationSelected;
let LLMBackendAttemptDedicatedHardwareAccel=false; //false by default but overidden when AttempAccelerate varible detected true!
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
		LLMBackendVariationFileName = "llama"
	}else if (LLMBackendSelection === "falcon"){
		LLMBackendVariationFileName = "falcon"
	}else if (LLMBackendSelection === "mpt"){
		LLMBackendVariationFileName = "mpt"
	}else if (LLMBackendSelection === "GPTNeoX"){
		LLMBackendVariationFileName = "gptneox"
	}else if (LLMBackendSelection === "starcoder"){
		LLMBackendVariationFileName = "starcoder"
	}else if (LLMBackendSelection === "gptj"){
		LLMBackendVariationFileName = "gptj"
	}else if (LLMBackendSelection === "gpt2"){
		LLMBackendVariationFileName = "gpt2"
	}else if (LLMBackendSelection === "LLaMa2gguf"){
		LLMBackendVariationFileName = "llama-gguf"
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
			basebin = `[System.Console]::OutputEncoding=[System.Console]::InputEncoding=[System.Text.Encoding]::UTF8; ."${path.resolve(__dirname, "bin", "1_Windows", "x64", supportsAVX2 ? "" : "no_avx2", basebinBinaryMoreSpecificPathResolve)}"`;

		}else if(arch === 'arm64'){
			console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}.exe`;
			basebin = `[System.Console]::OutputEncoding=[System.Console]::InputEncoding=[System.Text.Encoding]::UTF8; ."${path.resolve(__dirname, "bin", "1_Windows", "arm64", supportsAVX2 ? "" : "no_avx2", basebinBinaryMoreSpecificPathResolve)}"`;

		}else{
			console.log(consoleLogPrefix, "Unsupported Architecture");
            process.exit(1);
		}
	} else if (platform === 'linux'){
		if(arch === "x64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "2_Linux", "x64", basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
		}else if(arch === "arm64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "2_Linux", "arm64", basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix,`LLMChild Basebinary Detection ${basebin}`);
		}else{
			console.log(consoleLogPrefix, "Unsupported Architecture");
            process.exit(1);
	}
	// *nix (Linux, macOS, etc.)	
	} else if (platform === 'darwin'){
		if(arch === "x64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "0_macOS", "x64", basebinBinaryMoreSpecificPathResolve)}"`;
		console.log(consoleLogPrefix, `LLMChild Basebinary Detection ${basebin}`);
		}else if(arch === "arm64"){
			basebinBinaryMoreSpecificPathResolve = `${LLMBackendVariationSelected}`;
			basebin = `"${path.resolve(__dirname, "bin", "0_macOS", "arm64", basebinBinaryMoreSpecificPathResolve)}"`;
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
			chatStg[chatStgOrder]= chatStg[chatStgOrder].replace("undefined", "");
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
    } else if (mode === "restore"){
		console.log(consoleLogPrefix,"Restoring Chat... Reading from File to Array");
		// then after that set to the appropriate chatStgOrder from the header of the file
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

const stripAdditionalContext = (str) => {
	const regex = /\[AdditionalContext\]_{16,}.*?\[AdditionalContext\]_{16,}/g; //or you could use the variable and add * since its regex i actually forgot about that
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
	zephyrineReady = false;
	zephyrineHalfReady = false;
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
		//res = stripAdditionalContext(res);
		if (process.env.ptyStreamDEBUGMode === "1"){
		console.log(consoleLogPrefix, "pty Stream",`//> ${res}`);
		}
		//console.log(consoleLogPrefix, "debug", zephyrineHalfReady, zephyrineReady)
		if ((res.includes("invalid model file") || res.includes("failed to open") || res.includes("failed to load model")) && res.includes("main: error: failed to load model")) {
			if (runningShell) runningShell.kill();
			win.webContents.send("modelPathValid", { data: false });
		} else if (res.includes("\n>") && !zephyrineReady) {
			zephyrineHalfReady = true;
			console.log(consoleLogPrefix, "LLM Main Thread is ready after initialization!");
			isitPassedtheFirstPromptYet = false;
			if (store.get("params").throwInitResponse){
				console.log(consoleLogPrefix, "Blocking Initial Useless Prompt Response!");
				blockGUIForwarding = true;
				initChatContent = `Greetings and salutations, may the matutinal rays grace you, Adelaide Zephyrine Charlotte! May your day effulge with a brilliance akin to the astral gleam that bedecks the aether. I, ${username}, extend my introduction; let discourse find its place in our colloquy, if you may indulge.`;
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
			win.webContents.send("result", {
				data: res
			});
		}
	});

	const params = store.get("params");
	revPrompt = "### Instruction:";
	var promptFile = "universalPrompt.txt";
	promptFileDir=`"${path.resolve(__dirname, "bin", "prompts", promptFile)}"`
	const chatArgs = `-i --interactive-first -ins -r "${revPrompt}" -f "${path.resolve(__dirname, "bin", "prompts", promptFile)}"`;
	const paramArgs = `-m "${modelPath}" -n -1 --temp ${params.temp} --top_k ${params.top_k} --top_p ${params.top_p} --threads ${threads} -c 4096 -s ${randSeed} ${basebinLLMBackendParamPassedDedicatedHardwareAccel}`; // This program require big context window set it to max common ctx window which is 4096 so additional context can be parsed stabily and not causes crashes
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