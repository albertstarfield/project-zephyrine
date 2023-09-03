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

document.querySelector("#path-dialog-bg > div > div.dialog-button > button.primary").addEventListener("click", () => {
	var path = document.querySelector("#path-dialog input[type=text]").value.replaceAll('"', "");
	ipcRenderer.send("checkPath", { data: path });
});

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

document.querySelector("#path-dialog > div > button").addEventListener("click", () => {
	ipcRenderer.send("pickFile");
});
ipcRenderer.on("pickedFile", (_error, { data }) => {
	document.querySelector("#path-dialog input[type=text]").value = data;
});

ipcRenderer.on("currentModel", (_event, { data }) => {
	document.querySelector("#path-dialog input[type=text]").value = data;
});

document.querySelector("#path-dialog input[type=text]").addEventListener("keypress", (e) => {
	if (e.keyCode === 13) {
		e.preventDefault();
		document.querySelector("#path-dialog-bg .dialog-button button.primary").click();
	}
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
		ipcRenderer.send("message", { data: prompt });
		say(input.value, `user${gen}`, true);
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

const say = (msg, id, isUser) => {
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

	console.log(msg);

	//escape html tag
	if (isUser) {
		msg = msg.replaceAll(/</g, "&lt;");
		msg = msg.replaceAll(/>/g, "&gt;");
	}

	item.innerHTML = msg;
	if (document.getElementById("bottom").getBoundingClientRect().y - 40 < window.innerHeight) {
		setTimeout(() => {
			bottom.scrollIntoView({ behavior: "smooth", block: "end" });
		}, 100);
	}
	messages.append(item);
};
var responses = [];

Date.prototype.timeNow = function () {
	return (this.getHours() < 10 ? "0" : "") + this.getHours() + ":" + (this.getMinutes() < 10 ? "0" : "") + this.getMinutes() + ":" + (this.getSeconds() < 10 ? "0" : "") + this.getSeconds();
};
Date.prototype.today = function () {
	return (this.getDate() < 10 ? "0" : "") + this.getDate() + "/" + (this.getMonth() + 1 < 10 ? "0" : "") + (this.getMonth() + 1) + "/" + this.getFullYear();
};

ipcRenderer.on("result", async (_event, { data }) => {
	var response = data;
	loading(false);
	if (data == "\n\n<end>") {
		setTimeout(() => {
			isRunningModel = false;
			form.setAttribute("class", isRunningModel ? "running-model" : "");
		}, 200);
	} else {
		document.body.classList.remove("llama");
		document.body.classList.remove("alpaca");
		isRunningModel = true;
		form.setAttribute("class", isRunningModel ? "running-model" : "");
		const id = gen;
		let existing = document.querySelector(`[data-id='${id}']`);
		if (existing) {
			if (!responses[id]) {
				responses[id] = document.querySelector(`[data-id='${id}']`).innerHTML;
			}

			responses[id] = responses[id] + response;

			if (responses[id].startsWith("<br>")) {
				responses[id] = responses[id].replace("<br>", "");
			}
			if (responses[id].startsWith("\n")) {
				responses[id] = responses[id].replace("\n", "");
			}

			responses[id] = responses[id].replaceAll(/\r?\n\x1B\[\d+;\d+H./g, "");
			responses[id] = responses[id].replaceAll(/\x08\r?\n?/g, "");

			responses[id] = responses[id].replaceAll("\\t", "&nbsp;&nbsp;&nbsp;&nbsp;"); //tab characters
			responses[id] = responses[id].replaceAll("\\b", "&nbsp;"); //no break space
			responses[id] = responses[id].replaceAll("\\f", "&nbsp;"); //no break space
			responses[id] = responses[id].replaceAll("\\r", "\n"); //sometimes /r is used in codeblocks

			responses[id] = responses[id].replaceAll("\\n", "\n"); //convert line breaks back
			responses[id] = responses[id].replaceAll("\\\n", "\n"); //convert line breaks back
			responses[id] = responses[id].replaceAll('\\\\\\""', '"'); //convert quotes back

			responses[id] = responses[id].replaceAll(/\[name\]/gi, "Alpaca");

			responses[id] = responses[id].replaceAll(/(<|\[|#+)((current|local)_)?time(>|\]|#+)/gi, new Date().timeNow());
			responses[id] = responses[id].replaceAll(/(<|\[|#+)(current_)?date(>|\]|#+)/gi, new Date().today());
			responses[id] = responses[id].replaceAll(/(<|\[|#+)day_of_(the_)?week(>|\]|#+)/gi, new Date().toLocaleString("en-us", { weekday: "long" }));

			responses[id] = responses[id].replaceAll(/(<|\[|#+)(current_)?year(>|\]|#+)/gi, new Date().getFullYear());
			responses[id] = responses[id].replaceAll(/(<|\[|#+)(current_)?month(>|\]|#+)/gi, new Date().getMonth() + 1);
			responses[id] = responses[id].replaceAll(/(<|\[|#+)(current_)?day(>|\]|#+)/gi, new Date().getDate());

			//escape html tag
			responses[id] = responses[id].replaceAll(/</g, "&lt;");
			responses[id] = responses[id].replaceAll(/>/g, "&gt;");

			// if scroll is within 8px of the bottom, scroll to bottom
			if (document.getElementById("bottom").getBoundingClientRect().y - 40 < window.innerHeight) {
				setTimeout(() => {
					bottom.scrollIntoView({ behavior: "smooth", block: "end" });
				}, 100);
			}
			existing.innerHTML = responses[id];
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

document.getElementById("clear-chat").addEventListener("click", () => {
	stopButton.click();
	stopButton.removeAttribute("disabled");
	document.querySelectorAll("#messages li").forEach((element) => {
		element.remove();
	});
	setTimeout(() => {
		document.querySelectorAll("#messages li").forEach((element) => {
			element.remove();
		});
	}, 100);
});
document.getElementById("change-model").addEventListener("click", () => {
	ipcRenderer.send("getCurrentModel");
	document.querySelector("#path-dialog-bg > div > div.dialog-button > button.secondary").style.display = "";
	document.querySelector("#path-dialog-bg > div > div.dialog-title > h3").innerText = "Change model path";
	document.getElementById("path-dialog-bg").classList.remove("hidden");
});

ipcRenderer.send("getParams");
document.getElementById("settings").addEventListener("click", () => {
	document.getElementById("settings-dialog-bg").classList.remove("hidden");
	ipcRenderer.send("getParams");
});
ipcRenderer.on("params", (_event, data) => {
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
	document.getElementById("throwInitialGarbageResponse").checked = data.throwInitResponse;
	document.getElementById("classicmode").checked = data.classicMode;
	document.getElementById("attemptaccelerate").checked = data.AttemptAccelerate;
	document.getElementById("hardwarelayeroffloading").value = data.hardwareLayerOffloading;
	//document.getElementById("LLMBackendMode").value = data.llmBackendMode;
	document.getElementById("emotionalllmchildengine").checked = data.emotionalLLMChildengine;
	document.getElementById("profilepictureemotion").checked = data.profilePictureEmotion;
	document.getElementById("longchainthought-neverfeelenough").checked = data.longChainThoughtNeverFeelenough;
});
document.querySelector("#settings-dialog-bg > div > div.dialog-button > button.primary").addEventListener("click", () => {
	ipcRenderer.send("storeParams", {
		params: {
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

document.getElementById("attemptaccelerate").addEventListener("change", () => {
	ipcRenderer.send("AttemptAccelerate", document.getElementById("attemptaccelerate").checked);
});

document.getElementById("emotionalllmchildengine").addEventListener("change", () => {
	ipcRenderer.send("emotionalLLMChildengine", document.getElementById("emotionalllmchildengine").checked);
});

document.getElementById("profilepictureemotion").addEventListener("change", () => {
	ipcRenderer.send("profilePictureEmotion", document.getElementById("profilepictureemotion").checked);
});

document.getElementById("LLMBackendMode").addEventListener("change", () => {
	const value = document.getElementById("LLMBackendMode").value;
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

