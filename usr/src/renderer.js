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
//https://stackoverflow.com/questions/72591633/electron-simplest-example-to-pass-variable-from-js-to-html-using-ipc-contextb
const remote = require("@electron/remote");
const { ipcRenderer, dialog } = require("electron");

const win = remote.getCurrentWindow();

document.onreadystatechange = (event) => {
	ipcRenderer.send("os");
	if (document.readyState == "complete") {
		handleWindowControls();
	}
	document.querySelector("#path-dialog-bg > div > div.dialog-button > button.secondary").style.display = "none";
	document.querySelector("#path-dialog-bg > div > div.dialog-title > h3").innerText = "Couldn't load model";
	ipcRenderer.send("checkModelPath");
};

ipcRenderer.on("os", (_error, { data }) => {
	document.querySelector("html").classList.add(data);
});

ipcRenderer.on("modelPathValid", (_event, { data }) => {
	if (data) {
		ipcRenderer.send("startChat");
	} else {
		ipcRenderer.send("getCurrentModel");
		document.getElementById("path-dialog-bg").classList.remove("hidden");
	}
});
/*
document.querySelector("#path-dialog-bg > div > div.dialog-button > button.primary").addEventListener("click", () => {
	var path = document.querySelector("#path-dialog input[type=text]").value.replaceAll('"', "");
	ipcRenderer.send("checkPath", { data: path });
});
*/
// Replaced with automatic selection

document.querySelector("#path-dialog-bg > div > div.dialog-button > button.secondary").addEventListener("click", () => {
	document.getElementById("path-dialog-bg").classList.add("hidden");
});

ipcRenderer.on("pathIsValid", (_event, { data }) => {
	console.log(data);
	if (data) {
		document.querySelector("#path-dialog > p.error-text").style.display = "none";
		document.getElementById("path-dialog-bg").classList.add("hidden");
		ipcRenderer.send("restart");
	} else {
		document.querySelector("#path-dialog > p.error-text").style.display = "block";
	}
});
/*
// Legacy alpaca-electron manual selection model file code

document.querySelector("#path-dialog > div > button").addEventListener("click", () => {
	ipcRenderer.send("pickFile");
});
ipcRenderer.on("pickedFile", (_error, { data }) => {
	document.querySelector("#path-dialog input[type=text]").value = data;
});

document.querySelector("#path-dialog input[type=text]").addEventListener("keypress", (e) => {
	if (e.keyCode === 13) {
		e.preventDefault();
		document.querySelector("#path-dialog-bg .dialog-button button.primary").click();
	}
});


*/

ipcRenderer.on("currentModel", (_event, { data }) => {
	document.querySelector("#path-dialog input[type=text]").value = data;
});


window.onbeforeunload = (event) => {
	win.removeAllListeners();
};

function handleWindowControls() {
	document.getElementById("min-button").addEventListener("click", (event) => {
		win.minimize();
	});

	document.getElementById("max-button").addEventListener("click", (event) => {
		win.maximize();
	});

	document.getElementById("restore-button").addEventListener("click", (event) => {
		win.unmaximize();
	});

	document.getElementById("close-button").addEventListener("click", (event) => {
		win.close();
	});

	toggleMaxRestoreButtons();
	win.on("maximize", toggleMaxRestoreButtons);
	win.on("unmaximize", toggleMaxRestoreButtons);

	function toggleMaxRestoreButtons() {
		if (win.isMaximized()) {
			document.body.classList.add("maximized");
		} else {
			document.body.classList.remove("maximized");
		}
	}
}

var gen = 0;
const form = document.getElementById("form");
const stopButton = document.getElementById("stop");
const input = document.getElementById("input");
const messages = document.getElementById("messages");

input.addEventListener("keydown", () => {
	setTimeout(() => {
		input.style.height = "auto";
		input.style.height = input.scrollHeight + "px";
	});
});
input.addEventListener("keyup", () => {
	setTimeout(() => {
		input.style.height = "auto";
		input.style.height = input.scrollHeight + "px";
	});
});

let isRunningModel = false;
const loading = (on) => {
	if (on) {
		document.querySelector(".loading").classList.remove("hidden");
	} else {
		document.querySelector(".loading").classList.add("hidden");
	}
};

form.addEventListener("submit", (e) => {
	e.preventDefault();
	e.stopPropagation();
	if (form.classList.contains("running-model")) return;
	if (input.value) {
		var prompt = input.value.replaceAll("\n", "\\n");
		//Reverse engineering conclusion note : This is for sending the prompt or the User input to the GUI
		// NO its not, this ipcRenderer is for sending the prompt to index.js then to whatever engine_component or can be said as interprocesscommunication for processing, not for rendering into the GUI
		ipcRenderer.send("message", { data: prompt }); 
		// Reverse Engineering note : this might be the one that sends to the GUI! 
		// but what is input.value? where does it come from? From the look of the var prompt = input.value.replaceAll("\n", "\\n"); it seems it stores the user message but what is the .value? 
		// Ah, so the input variable is already or asyncrhonously determined by the renderer.js for the value subvariable
		// so to forward thing basically we just do
		/*
		say(message, `user${gen}`, true);
		gen++;

		//Haha reverse engineering done!
		*/
		say(input.value, `user${gen}`, true); // so ${gen} is for determining what part of the chat history that its belongs to, Nice!
		loading(prompt);
		input.value = "";
		isRunningModel = true;
		stopButton.removeAttribute("disabled");
		form.setAttribute("class", isRunningModel ? "running-model" : "");
		gen++;
		setTimeout(() => {
			input.style.height = "auto";
			input.style.height = input.scrollHeight + "px";
		});
	}
});
input.addEventListener("keydown", (e) => {
	if (e.keyCode === 13) {
		e.preventDefault();
		if (e.shiftKey) {
			document.execCommand("insertLineBreak");
		} else {
			form.requestSubmit();
		}
	}
});

stopButton.addEventListener("click", (e) => {
	e.preventDefault();
	e.stopPropagation();
	ipcRenderer.send("stopGeneration");
	stopButton.setAttribute("disabled", "true");
	setTimeout(() => {
		isRunningModel = false;
		form.setAttribute("class", isRunningModel ? "running-model" : "");
		input.style.height = "34px";
	}, 5);
});

const sha256 = async (input) => {
	const textAsBuffer = new TextEncoder().encode(input);
	const hashBuffer = await window.crypto.subtle.digest("SHA-256", textAsBuffer);
	const hashArray = Array.from(new Uint8Array(hashBuffer));
	const hash = hashArray.map((item) => item.toString(16).padStart(2, "0")).join("");
	return hash;
};
// target Emotional Support Add---------------------------------------------------

let profilePictureEmotions;
ipcRenderer.on('emotionalEvaluationResult', (event, data) => {
	// Use the received data in your renderer process
	console.log('Received emotionalEvaluationResult:', data);
	profilePictureEmotions = data;
  });  

const marked = require('marked') // call marked marked.js to convert output to Markdown

const say = (msg, id, isUser) => {
	// Sidenote : Reverse Engineering Conclusion : Only captures one data stream iteration
	let item = document.createElement("li");
	if (id) item.setAttribute("data-id", id);
	
	// step 0 convert it into if structure
	//item.classList.add(isUser ? "user-msg" : document.getElementById("web-access").checked ? "bot-web-msg" : "bot-msg"); // changes bot response image

	// step 1 Understanding how the profile picture changes based on web-access switch
	/*
	if (isUser) {
		item.classList.add("user-msg");
	} else if (document.getElementById("web-access").checked) {
		item.classList.add("bot-web-msg");
	} else {
		item.classList.add("bot-msg");
	}*/

	//step 2 add into 5 emotion mode step

	// Because of the save and load doesn't contain profilePictureEmotions currently, when its being loaded the program will error out thus if its uninitialized or undefined we can set it to default "happy"
	if (typeof profilePictureEmotions === 'undefined') {
		profilePictureEmotions = 'happy'; // Set the variable to a default value or any other desired value
	}


	if (isUser) {
		item.classList.add("user-msg");
	} else if (profilePictureEmotions.toLowerCase() === "happy") {
		item.classList.add("bot-default");
	} else if (profilePictureEmotions.toLowerCase() === "fear") {
		item.classList.add("bot-concernedFear");
	} else if (profilePictureEmotions.toLowerCase() === "disgust") {
		item.classList.add("bot-disgust");
	} else if (profilePictureEmotions.toLowerCase() === "anger") {
		item.classList.add("bot-anger");
	} else if (profilePictureEmotions.toLowerCase() === "sad") {
		item.classList.add("bot-sad");
	} else {
		item.classList.add("bot-default");
	}
	console.log(msg); //debug message

	//escape html tag
	if (isUser) {
		//msg = marked.parse(msg);
		msg = msg.replaceAll(/</g, "&lt;");
		msg = msg.replaceAll(/>/g, "&gt;");
	}
	//console.log("reverseEngMessage", msg)
	item.innerHTML = msg;
	if (document.getElementById("bottom").getBoundingClientRect().y - 40 < window.innerHeight) {
		setTimeout(() => {
			bottom.scrollIntoView({ behavior: "smooth", block: "end" });
		}, 100);
	}
	messages.append(item);
};
//console.log("responsesTraceDebug", responses)
var responses = []; // This is probably appending the continous stream of llama-2 LLM

Date.prototype.timeNow = function () {
	return (this.getHours() < 10 ? "0" : "") + this.getHours() + ":" + (this.getMinutes() < 10 ? "0" : "") + this.getMinutes() + ":" + (this.getSeconds() < 10 ? "0" : "") + this.getSeconds();
};
Date.prototype.today = function () {
	return (this.getDate() < 10 ? "0" : "") + this.getDate() + "/" + (this.getMonth() + 1 < 10 ? "0" : "") + (this.getMonth() + 1) + "/" + this.getFullYear();
};



/* Renderer that handeled the result data from the binary data into html */

function HTMLInterpreterPreparation(stream){
	stream = stream.replaceAll(/\r?\n\x1B\[\d+;\d+H./g, "");
	stream = stream.replaceAll(/\x08\r?\n?/g, "");

	stream = stream.replaceAll("\\t", "&nbsp;&nbsp;&nbsp;&nbsp;"); //tab characters
	stream = stream.replaceAll("\\b", "&nbsp;"); //no break space
	stream = stream.replaceAll("\\f", "&nbsp;"); //no break space
	stream = stream.replaceAll("\\r", "\n"); //sometimes /r is used in codeblocks

	stream = stream.replaceAll("\\n", "\n"); //convert line breaks back
	stream = stream.replaceAll("\\\n", "\n"); //convert line breaks back
	stream = stream.replaceAll('\\\\\\""', '"'); //convert quotes back

	stream = stream.replaceAll(/\[name\]/gi, "Alpaca");

	stream = stream.replaceAll(/(<|\[|#+)((current|local)_)?time(>|\]|#+)/gi, new Date().timeNow());
	stream = stream.replaceAll(/(<|\[|#+)(current_)?date(>|\]|#+)/gi, new Date().today());
	stream = stream.replaceAll(/(<|\[|#+)day_of_(the_)?week(>|\]|#+)/gi, new Date().toLocaleString("en-us", { weekday: "long" }));

	stream = stream.replaceAll(/(<|\[|#+)(current_)?year(>|\]|#+)/gi, new Date().getFullYear());
	stream = stream.replaceAll(/(<|\[|#+)(current_)?month(>|\]|#+)/gi, new Date().getMonth() + 1);
	stream = stream.replaceAll(/(<|\[|#+)(current_)?day(>|\]|#+)/gi, new Date().getDate());

	//escape html tag
	stream = stream.replaceAll(/</g, "&lt;"); // translates < into the special character input so it can be inputted without being changed on the terminal I/O
	stream = stream.replaceAll(/>/g, "&gt;"); // translates > into the special character input so it can be inputted without being changed on the terminal I/O
	return stream;
}

function firstLineParagraphCleaner(stream){
	let str = stream;
	let arr = str.split('\n');
	arr[0] = arr[0].replace(/<p>|<\/p>/g, '');
	let result = arr.join('\n');
	return result;
}


// It seems that this is the one that handles the continous stream result of the binary call then forwrad it to html
let prefixConsoleLogStreamCapture = "[LLMMainBackendStreamCapture]: ";
// add some custom remote ipc async routines (in this case the experiment is going to be conducted on the index.js where we manually inject the user input data to the GUI, this is important for the save and restore function of the program)

// this is for recieving (load User input) message from the saved message routines
ipcRenderer.on("manualUserPromptGUIHijack", async (_event, { data }) => { 
	var userPretendInput = data;
	say(userPretendInput, `user${gen}`, true);
	gen++;
});
// this is for recieving (load AI Output) message from the saved message routines
ipcRenderer.on("manualAIAnswerGUIHijack", async (_event, { data }) => { 
	const id = gen;
	var response = data;
	let existing = document.querySelector(`[data-id='${id}']`);

	if (existing) {
		//console.log("responsesStreamCaptureTrace_init:", responses[id]);
		if (!responses[id]) {
			responses[id] = document.querySelector(`[data-id='${id}']`).innerHTML;
		}
		responses[id] = responses[id] + response;
		//console.log("responsesStreamCaptureTrace_first:", responses[id]);

		if (responses[id].startsWith("<br>")) {
			responses[id] = responses[id].replace("<br>", "");
		}
		if (responses[id].startsWith("\n")) {
			responses[id] = responses[id].replace("\n", "");
		}
		// if scroll is within 8px of the bottom, scroll to bottom
		if (document.getElementById("bottom").getBoundingClientRect().y - 40 < window.innerHeight) {
			setTimeout(() => {
				bottom.scrollIntoView({ behavior: "smooth", block: "end" });
			}, 100);
		}
		
		//existing.innerHTML = responses[id];
	} else {
		say(response, id);
	}
	// okay it seems that it works but its saying about "Uncaught (in promise) TypeError: Cannot read properties of undefined (reading 'toLowerCase()')

	// Oh its about the profile picture emotion issue we can just stub it out using the normal profile picture, nobody will notice for now (define using default)
	profilePictureEmotions = "happy"; 
	existing.innerHTML = response; // no submitting this thing will result in error "Uncaught (in promise) TypeError: Cannot set properties of null (setting 'innerHTML')"
	// Something is missing. Ah th emissing part is that on the let existing need to end with .innerHTML, Nope. Still erroring out.

	

});

ipcRenderer.on("result", async (_event, { data }) => {
	var response = data;
	const id = gen;
	let existing = document.querySelector(`[data-id='${id}']`);
	let totalStreamResultData;
	loading(false);
	if (data == "\n\n<end>") {
		setTimeout(() => {
			isRunningModel = false;
			console.log(prefixConsoleLogStreamCapture, "Stream Ended!");
			console.log(prefixConsoleLogStreamCapture, "Assuming LLM Main Backend Done Typing!");
			console.log(prefixConsoleLogStreamCapture, "Sending Stream to GUI");
			//totalStreamResultData = responses[id];
			totalStreamResultData = marked.parse(responses[id]);
			totalStreamResultData = firstLineParagraphCleaner(totalStreamResultData);
			console.log("responsesStreamCaptureTrace_markedConverted:", totalStreamResultData);
			//totalStreamResultData = HTMLInterpreterPreparation(totalStreamResultData); //legacy old alpaca-electron interpretation
			console.log("responsesStreamCaptureTrace_markedConverted_InterpretedPrep:", totalStreamResultData);
			existing.innerHTML = totalStreamResultData; // existing innerHTML is a final form which send the stream into the GUI
			console.log(prefixConsoleLogStreamCapture, "It should be done");
			form.setAttribute("class", isRunningModel ? "running-model" : "");
		}, 200);
	} else {
		document.body.classList.remove("llama");
		document.body.classList.remove("alpaca");
		isRunningModel = true;
		form.setAttribute("class", isRunningModel ? "running-model" : "");
		if (existing) {
			//console.log("responsesStreamCaptureTrace_init:", responses[id]);
			if (!responses[id]) {
				responses[id] = document.querySelector(`[data-id='${id}']`).innerHTML;
			}
			responses[id] = responses[id] + response;
			//console.log("responsesStreamCaptureTrace_first:", responses[id]);

			if (responses[id].startsWith("<br>")) {
				responses[id] = responses[id].replace("<br>", "");
			}
			if (responses[id].startsWith("\n")) {
				responses[id] = responses[id].replace("\n", "");
			}
			// if scroll is within 8px of the bottom, scroll to bottom
			if (document.getElementById("bottom").getBoundingClientRect().y - 40 < window.innerHeight) {
				setTimeout(() => {
					bottom.scrollIntoView({ behavior: "smooth", block: "end" });
				}, 100);
			}
			
			//existing.innerHTML = responses[id];
		} else {
			say(response, id);
		}
	}
});

document.querySelectorAll("#feed-placeholder-llama button.card").forEach((e) => {
	e.addEventListener("click", () => {
		let text = e.innerText.replace('"', "").replace('" →', "");
		input.value = text;
	});
});
document.querySelectorAll("#feed-placeholder-alpaca button.card").forEach((e) => {
	e.addEventListener("click", () => {
		let text = e.innerText.replace('"', "").replace('" →', "");
		input.value = text;
	});
});
let username;
let assistantName;
ipcRenderer.send("username") //send request of that specific data from all process that is running
ipcRenderer.send("assistantName") //send request
ipcRenderer.on("username", (_event, { data }) => {
	username = data;	
	console.log("The username is: ", username);
});
ipcRenderer.on("assistantName", (_event, { data }) => {
	assistantName = data;
	console.log("I am :", assistantName);
});

const cpuText = document.querySelector("#cpu .text");
const ramText = document.querySelector("#ram .text");
const cpuBar = document.querySelector("#cpu .bar-inner");
const ramBar = document.querySelector("#ram .bar-inner");

var cpuCount, threadUtilized, totalmem, cpuPercent, freemem;
ipcRenderer.send("cpuCount");
ipcRenderer.send("threadUtilized");
ipcRenderer.send("totalmem");
ipcRenderer.on("cpuCount", (_event, { data }) => {
	cpuCount = data;
});
ipcRenderer.on("threadUtilized", (_event, { data }) => {
	threadUtilized = data;
});
ipcRenderer.on("totalmem", (_event, { data }) => {
	totalmem = Math.round(data / 102.4) / 10;
});

setInterval(async () => {
	ipcRenderer.send("cpuUsage");
	ipcRenderer.send("freemem");
}, 60000);
// For some reason cpuUsage and freemem everytime its updating its eating huge amount of GPU power ?

ipcRenderer.on("cpuUsage", (_event, { data }) => {
	cpuPercent = Math.round(data * 100);
	cpuText.innerText = `CPU: ${cpuPercent}%, ${threadUtilized}/${cpuCount} threads`;
	cpuBar.style.transform = `scaleX(${cpuPercent / 100})`;
});
ipcRenderer.on("freemem", (_event, { data }) => {
	freemem = data;
	ramText.innerText = `Memory: ${Math.round((totalmem - freemem) * 10) / 10}GB/${totalmem}GB`;
	ramBar.style.transform = `scaleX(${(totalmem - freemem) / totalmem})`;
});

document.getElementById("clear").addEventListener("click", () => {
	input.value = "";
	setTimeout(() => {
		input.style.height = "auto";
		input.style.height = input.scrollHeight + "px";
	});
});

document.getElementById("chat-reset").addEventListener("click", () => {
	stopButton.click();
	stopButton.removeAttribute("disabled");
	ipcRenderer.send("restart");
	ipcRenderer.send("resetChatHistoryCTX");
	document.querySelectorAll("#messages li").forEach((element) => {
		element.remove();
	});
	setTimeout(() => {
		document.querySelectorAll("#messages li").forEach((element) => {
			element.remove();
		});
	}, 100);
});

// since the program now come with predefined tuned model this won't be required and if its implemented it will make the program runs in buggy state or chaos like the button won't be able clicked without error and etc
/*
document.getElementById("change-model").addEventListener("click", () => {
	ipcRenderer.send("getCurrentModel");
	document.querySelector("#path-dialog-bg > div > div.dialog-button > button.secondary").style.display = "";
	document.querySelector("#path-dialog-bg > div > div.dialog-title > h3").innerText = "Change model path";
	document.getElementById("path-dialog-bg").classList.remove("hidden");
});
*/

ipcRenderer.send("getParams");
document.getElementById("settings").addEventListener("click", () => {
	document.getElementById("settings-dialog-bg").classList.remove("hidden");
	ipcRenderer.send("getParams");
});

ipcRenderer.send("getParams");
document.getElementById("aboutSection").addEventListener("click", () => {
	document.getElementById("info-dialog-bg").classList.remove("hidden");
	ipcRenderer.send("getParams");
});

ipcRenderer.on("params", (_event, data) => {
	//document.getElementById("LLMBackendMode").value = data.llmBackendMode; // since the program now come with predefined tuned model this won't be required and if its implemented it will make the program runs in buggy state or chaos like the button won't be able clicked without error and etc, and selection are implemented on dictionary on index.js in Engine section of the Data
	document.getElementById("repeat_last_n").value = data.repeat_last_n;
	document.getElementById("repeat_penalty").value = data.repeat_penalty;
	document.getElementById("top_k").value = data.top_k;
	document.getElementById("top_p").value = data.top_p;
	document.getElementById("temp").value = data.temp;
	document.getElementById("seed").value = data.seed;
	document.getElementById("web-access").checked = data.webAccess;
	document.getElementById("local-file-access").checked = data.localAccess;
	document.getElementById("LLMChildDecision").checked = data.llmdecisionMode;
	document.getElementById("longchainthought").checked = data.extensiveThought;
	document.getElementById("saverestorechat").checked = data.SaveandRestorechat;
	document.getElementById("historyChatCtx").checked = data.hisChatCTX;
	document.getElementById("foreveretchedmemory").checked = data.foreverEtchedMemory;
	document.getElementById("throwInitialGarbageResponse").checked = data.throwInitResponse;
	document.getElementById("classicmode").checked = data.classicMode;
	document.getElementById("attemptaccelerate").checked = data.AttemptAccelerate;
	document.getElementById("hardwarelayeroffloading").value = data.hardwareLayerOffloading;
	document.getElementById("emotionalllmchildengine").checked = data.emotionalLLMChildengine;
	document.getElementById("profilepictureemotion").checked = data.profilePictureEmotion;
	document.getElementById("longchainthought-neverfeelenough").checked = data.longChainThoughtNeverFeelenough;
});
document.querySelector("#settings-dialog-bg > div > div.dialog-button > button.primary").addEventListener("click", () => {
	ipcRenderer.send("storeParams", {
		params: {
			//llmBackendMode: document.getElementById("LLMBackendMode").value || document.getElementById("LLMBackendMode").value.placeholder, // since the program now come with predefined tuned model this won't be required and if its implemented it will make the program runs in buggy state or chaos like the button won't be able clicked without error and etc, and selection are implemented on dictionary on index.js in Engine section of the Data
			repeat_last_n: document.getElementById("repeat_last_n").value || document.getElementById("repeat_last_n").placeholder,
			repeat_penalty: document.getElementById("repeat_penalty").value || document.getElementById("repeat_penalty").placeholder,
			top_k: document.getElementById("top_k").value || document.getElementById("top_k").placeholder,
			top_p: document.getElementById("top_p").value || document.getElementById("top_p").placeholder,
			temp: document.getElementById("temp").value || document.getElementById("temp").placeholder,
			seed: document.getElementById("seed").value || document.getElementById("seed").placeholder,
			webAccess: document.getElementById("web-access").checked,
			localAccess: document.getElementById("local-file-access").checked,
            llmdecisionMode: document.getElementById("LLMChildDecision").checked,
            extensiveThought: document.getElementById("longchainthought").checked,
            SaveandRestorechat: document.getElementById("saverestorechat").checked,
			hisChatCTX: document.getElementById("historyChatCtx").checked,
			foreverEtchedMemory: document.getElementById("foreveretchedmemory").checked,
			throwInitResponse: document.getElementById("throwInitialGarbageResponse").checked,
			classicMode: document.getElementById("classicmode").checked,
			AttemptAccelerate: document.getElementById("attemptaccelerate").checked,
			emotionalLLMChildengine: document.getElementById("emotionalllmchildengine").checked,
			profilePictureEmotion: document.getElementById("profilepictureemotion").checked,
			websearch_amount: document.getElementById("websearch_amount").value || document.getElementById("websearch_amount").placeholder,
			maxWebSearchChar: document.getElementById("max_websearch_character").value || document.getElementById("max_websearch_character").placeholder,
			maxLocalSearchChar: document.getElementById("max_localsearch_character").value || document.getElementById("max_localsearch_character").placeholder,
			maxLocalSearchPerFileChar: document.getElementById("max_localsearch_perfile_character").value || document.getElementById("max_localsearch_perfile_character").placeholder,
			keywordContentFileMatchPercentageThreshold: document.getElementById("keywordcontentfilematchpercentthreshold").value || document.getElementById("keywordcontentfilematchpercentthreshold").placeholder,
			hardwareLayerOffloading: document.getElementById("hardwarelayeroffloading").value || document.getElementById("hardwarelayeroffloading").placeholder,
			longChainThoughtNeverFeelenough: document.getElementById("longchainthought-neverfeelenough").checked
		}
	});
	document.getElementById("settings-dialog-bg").classList.add("hidden");
});

document.querySelector("#settings-dialog-bg > div > div.dialog-button > button.secondary").addEventListener("click", () => {
	document.getElementById("settings-dialog-bg").classList.add("hidden");
});

document.querySelector("#info-dialog-bg > div > div.dialog-button > button.secondary").addEventListener("click", () => {
	document.getElementById("info-dialog-bg").classList.add("hidden");
});

document.getElementById("web-access").addEventListener("change", () => {
	ipcRenderer.send("webAccess", document.getElementById("web-access").checked);
});

document.getElementById("local-file-access").addEventListener("change", () => {
	ipcRenderer.send("localAccess", document.getElementById("local-file-access").checked);
});

document.getElementById("LLMChildDecision").addEventListener("change", () => {
	ipcRenderer.send("llmdecisionMode", document.getElementById("LLMChildDecision").checked);
});

document.getElementById("longchainthought").addEventListener("change", () => {
	ipcRenderer.send("extensiveThought", document.getElementById("longchainthought").checked);
});

document.getElementById("saverestorechat").addEventListener("change", () => {
	ipcRenderer.send("SaveandRestorechat", document.getElementById("saverestorechat").checked);
});

document.getElementById("historyChatCtx").addEventListener("change", () => {
	ipcRenderer.send("hisChatCTX", document.getElementById("historyChatCtx").checked);
});

document.getElementById("attemptaccelerate").addEventListener("change", () => {
	ipcRenderer.send("AttemptAccelerate", document.getElementById("attemptaccelerate").checked);
});

document.getElementById("emotionalllmchildengine").addEventListener("change", () => {
	ipcRenderer.send("emotionalLLMChildengine", document.getElementById("emotionalllmchildengine").checked);
});

document.getElementById("profilepictureemotion").addEventListener("change", () => {
	ipcRenderer.send("profilePictureEmotion", document.getElementById("profilepictureemotion").checked);
});

document.getElementById("longchainthought-neverfeelenough").addEventListener("change", () => {
	ipcRenderer.send("longChainThoughtNeverFeelenough", document.getElementById("longchainthought-neverfeelenough").checked);
});


//default value
//document.getElementById("model").value = "alpaca";
document.getElementById("repeat_last_n").value = 64;
document.getElementById("repeat_penalty").value = 1.3;
document.getElementById("top_k").value = 420;
document.getElementById("top_p").value = 90;
document.getElementById("temp").value = 0.9;

