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

const path = require("path");
let logPathFile;
const { createLogger, transports, format } = require('winston');

function generateRandomNumber(min, max) {
	return Math.floor(Math.random() * (max - min + 1)) + min;
}

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

logPathFile = path.resolve(__dirname, "AdelaideFrontendHandler.log");
console.log(logPathFile);


const logger = createLogger({
    level: 'info',
    format: format.combine(
        format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        format.printf(info => `${info.timestamp} ${info.level}: ${info.message}`)
    ),
    transports: [
        // Console transport (removed to prevent logging to console)
        // new transports.Console(),

        // File transport for info level messages only
        new transports.File({ filename: logPathFile, level: 'info' })
    ]
});

document.addEventListener("DOMContentLoaded", function() {
	//Messages
    const feedPlaceholder = document.getElementById("messages");
	feedPlaceholder.classList.add("fade-in");
	
    //document.getElementById("sky").classList.remove("hidden");
	//document.getElementById("sky").classList.add("fade-in");

	//document.getElementById("sky").classList.add("hidden");
	//document.getElementById("sky").classList.remove("fade-in");
    // Function to show the element with fade-in effect
    function fadeInElement() {
        feedPlaceholder.classList.remove("hidden");
    }

    // Function to hide the element with fade-out effect
    function fadeOutElement() {
        feedPlaceholder.classList.add("hidden");
    }

    // Call the fadeInElement function after a short delay to ensure smoother transition
	// Intro Fade-in with the feed
    setTimeout(fadeInElement, 500); // Adjust the delay as needed

});


document.addEventListener("DOMContentLoaded", function() {
    const content = document.getElementById("content");
	const skyStarHolder = document.getElementById("sky")
	document.getElementById("sky").classList.add("fade-in");
	
    // Function to fade in content gradually
    function fadeInContent() {
        content.classList.remove("hidden");
        content.style.opacity = 0;
        let opacity = 0;
        const fadeInInterval = setInterval(() => {
            opacity += 0.05; // Increase opacity gradually
            content.style.opacity = opacity;
            if (opacity >= 1) {
                clearInterval(fadeInInterval);
            }
        }, 50); // Adjust the interval for smoother or faster fade-in effect
    }

    // Call the fadeInContent function after a short delay to ensure smoother transition
    setTimeout(fadeInContent, 500); // Adjust the delay as needed
});


document.addEventListener("DOMContentLoaded", function() {
    const sky = document.getElementById("sky");
    const stars = [];

    // Function to generate a random number within a range
    function getRandom(min, max) {
        return Math.random() * (max - min) + min;
    }

    // Function to create a star
    function createStar() {
        const star = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        star.setAttribute("width", generateRandomNumber(1,5));
        star.setAttribute("height", generateRandomNumber(1,5));
        star.setAttribute("viewBox", "0 0 24 24");
        star.innerHTML = `<path d="M9.15316 5.40838C10.4198 3.13613 11.0531 2 12 2C12.9469 2 13.5802 3.13612 14.8468 5.40837L15.1745 5.99623C15.5345 6.64193 15.7144 6.96479 15.9951 7.17781C16.2757 7.39083 16.6251 7.4699 17.3241 7.62805L17.9605 7.77203C20.4201 8.32856 21.65 8.60682 21.9426 9.54773C22.2352 10.4886 21.3968 11.4691 19.7199 13.4299L19.2861 13.9372C18.8096 14.4944 18.5713 14.773 18.4641 15.1177C18.357 15.4624 18.393 15.8341 18.465 16.5776L18.5306 17.2544C18.7841 19.8706 18.9109 21.1787 18.1449 21.7602C17.3788 22.3417 16.2273 21.8115 13.9243 20.7512L13.3285 20.4768C12.6741 20.1755 12.3469 20.0248 12 20.0248C11.6531 20.0248 11.3259 20.1755 10.6715 20.4768L10.0757 20.7512C7.77268 21.8115 6.62118 22.3417 5.85515 21.7602C5.08912 21.1787 5.21588 19.8706 5.4694 17.2544L5.53498 16.5776C5.60703 15.8341 5.64305 15.4624 5.53586 15.1177C5.42868 14.773 5.19043 14.4944 4.71392 13.9372L4.2801 13.4299C2.60325 11.4691 1.76482 10.4886 2.05742 9.54773C2.35002 8.60682 3.57986 8.32856 6.03954 7.77203L6.67589 7.62805C7.37485 7.4699 7.72433 7.39083 8.00494 7.17781C8.28555 6.96479 8.46553 6.64194 8.82547 5.99623L9.15316 5.40838Z" fill="#ffe070"></path>`;
        star.style.position = "absolute";
        star.style.top = getRandom(0, sky.offsetHeight - 24) + "px";
        star.style.left = getRandom(0, sky.offsetWidth - 24) + "px";
        sky.appendChild(star);

        // Store star object
        stars.push({
            element: star,
            x: parseFloat(star.style.left),
            y: parseFloat(star.style.top),
            xSpeed: getRandom(1, 256),
            ySpeed: getRandom(1, 256)
        });
    }

    // Function to move stars and handle collisions
    function moveStars() {
        stars.forEach(star => {
            star.x += star.xSpeed;
            star.y += star.ySpeed;

            if (star.x < 0 || star.x > sky.offsetWidth - 24) {
                star.xSpeed *= -1; // Reverse direction on collision with left or right boundary
            }

            if (star.y < 0 || star.y > sky.offsetHeight - 24) {
                star.ySpeed *= -1; // Reverse direction on collision with top or bottom boundary
            }

            star.element.style.left = star.x + "px";
            star.element.style.top = star.y + "px";
        });

        // Call moveStars() recursively
        setTimeout(moveStars, 60000); //ms frame request
		//requestAnimationFrame(moveStars)
    }

    // Function to generate stars
    function generateStars() {
        for (let i = 0; i < 50; i++) {
            createStar();
        }
    }

    generateStars();
    moveStars();
});








document.onreadystatechange = (event) => {
	ipcRenderer.send("os");
	console.log("docReadyState", document.readyState);
	if (document.readyState == "complete") {
		handleWindowControls();
	}
	document.querySelector("#path-dialog-bg > div > div.dialog-button > button.secondary").style.display = "none";
	document.querySelector("#path-dialog-bg > div > div.dialog-title > h2").innerText = "Couldn't load model";
	ipcRenderer.send("checkModelPath");
};

ipcRenderer.on("os", (_error, { data }) => {
	document.querySelector("html").classList.add(data);
});

ipcRenderer.on("modelPathValid", (_event, { data }) => {
	if (data) {
		ipcRenderer.send("startInteraction");
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

/*

We dont use Loading spinning wheel anymore

// Loading Logo Spinning W
const loading = (on) => {
	if (on) {
		document.querySelector(".loading").classList.remove("hidden");
	} else {
		document.querySelector(".loading").classList.add("hidden");
	}
};

*/


// submit is where you click send and get processed by the mainLLM on alpaca-electron or dalai flip flop old model
// Reverse Engineer Conclusion I : This is form DOM event listener, it listens if you clicks the submit or autocomplete button 
form.addEventListener("submit", (e) => {
	console.debug("[DEBUG MULTISUBMISSION]: I Recieved a submission command")
	if (input.value){
		var prompt = input.value.replaceAll("\n", "\\n");
		console.debug("[DEBUG MULTISUBMISSION]: It has a value! Forwarding to the system!")
		say(input.value, `user${gen}`, true); // so ${gen} is for determining what part of the chat history that its belongs to, Nice!
		gen++;
		console.debug("[DEBUG MULTISUBMISSION]: Forwarding to main backend Engine!")
		ipcRenderer.send("message", { data: prompt }); 
		console.debug("[DEBUG MULTISUBMISSION]: Resetting GUI input form!")
		input.value = "";
	}
	
	e.preventDefault(); // if i disable this, it will make the UI reset every time i click and Loss control of the mainLLM prompting drive
	e.stopPropagation();
	// This stops the propagation of the process of the event listener if its already invoked single time
	if (form.classList.contains("running-model")) return;
	if (input.value) {

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
		say(input.value, `user${gen}`, true); // so ${gen} is for determining what part of the chat history that its belongs to, Nice!
		gen++;
		*/		
		input.value = "";
		isRunningModel = true;
		stopButton.removeAttribute("disabled");
		form.setAttribute("class", isRunningModel ? "running-model" : "");
		
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

const marked = require('marked'); // call marked marked.js to convert output to Markdown
const katex = require('katex'); // LaTeX Conversion! We're going scientific m8!

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
	var userPretendInput = data.interactionTextData; //marked.parse(responses[id]);
	say(userPretendInput, `user${gen}`, true);
	gen++;
});
// this is for recieving (load AI Output) message from the saved message routines
ipcRenderer.on("manualAIAnswerGUIHijack", async (_event, { data }) => { 
	const id = gen;
	var response = data.interactionTextData;
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
	//profilePictureEmotions = "happy"; 
	// Since the data now have the emotions we can use the emotion data to replace the profilePictureEmotions based on what is saved
	profilePictureEmotions = data.emotion;
	existing.innerHTML = response; // no submitting this thing will result in error "Uncaught (in promise) TypeError: Cannot set properties of null (setting 'innerHTML')"
	gen++; //add into new rows
	// Something is missing. Ah th emissing part is that on the let existing need to end with .innerHTML, Nope. Still erroring out.
});



ipcRenderer.on("result", async (_event, { data }) => {
	var response = data;
	const id = gen;
	let existing = document.querySelector(`[data-id='${id}']`);
	let totalStreamResultData;
	//loading(false);
	//console.debug(prefixConsoleLogStreamCapture, "DataStream", data);
	/*
	can be invoked through main index.js or the adelaide paradigm engine by using this
	clear
	framebufferBridgeUI.webContents.send("result", {
					data: "\n\n<end>"
				});
	*/
	if (data == "\n\n<end>") {
		setTimeout(() => { // this timeout is for the final render, if somehow after the 
			isRunningModel = false;
			doneGenerating = false;
			//existing.style.opacity = 0;
			//existing.style.transition = 'opacity 1s ease-in-out';
			//console.log(prefixConsoleLogStreamCapture, "Stream Ended!");
			//console.log(prefixConsoleLogStreamCapture, "Assuming LLM Main Backend Done Typing!");
			//console.log(prefixConsoleLogStreamCapture, "Sending Stream to GUI");
			//totalStreamResultData = responses[id];
			console.log("Response Raw DEBUG",responses[id]);
			// So i have found the issue where it finishes responding but the text disappears its because the marked.parse throw some error and the text failed to go through. so instead of complete failure, we can just use try and catch the error and forcefully do the thing
			try{
				totalStreamResultData = marked.parse(responses[id]);
			}catch(error){
				console.error(prefixConsoleLogStreamCapture, "ERROR Marked, Preventing Disappearing", error, "Captured Content:", responses[id])
				totalStreamResultData = responses[id];
			}
			if (!totalStreamResultData == undefined){
				console.error("Capture Stream Broken!")
				totalStreamResultData="🛑 An Unexpected Error has been occoured and Adelaide Engine lost context capture from mainLLM Thread. Please restart the program and try again!🛑 "
			}
			/*
			*/
			totalStreamResultData = firstLineParagraphCleaner(totalStreamResultData);
			console.log("responsesStreamCaptureTrace_markedConverted:", totalStreamResultData);
			//totalStreamResultData = HTMLInterpreterPreparation(totalStreamResultData); //legacy old alpaca-electron interpretation
			console.log("responsesStreamCaptureTrace_markedConverted_InterpretedPrep:", totalStreamResultData);
			existing.innerHTML = totalStreamResultData; // existing innerHTML is a final form which send the stream into the GUI (so basically it replaces all the chunks of message from the responses[id] on the existing.innerHTML)
			console.log(prefixConsoleLogStreamCapture, "It should be done");

			//responses[id] forward to Automata Mode if its Turned on then it will continue
			// Why i put the Automata Triggering ipcRenderer in here? : because when the stream finished, it stopped in here
			ipcRenderer.send("AutomataLLMMainResultReciever", { data: responses[id] });
			ipcRenderer.send("saveAdelaideMessage", { data: responses[id] }); // Save Adelaide Paradigm Engine Message into the database by calling the IPC Call Function
			ipcRenderer.send("CheckAsyncMessageQueue"); // Multiple-User submission support
			console.log(prefixConsoleLogStreamCapture, "current ID of Output", id);

			form.setAttribute("class", isRunningModel ? "running-model" : "");
			existing.style.opacity = 1;
			gen++; //add into new rows
		}, 150); // previously 30 now adjust it to 60 just to make sure it doesn't cut the stream midway
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
			
			existing.innerHTML = responses[id]; // This sends each stream chunk creating a typing effect like on AI Website
		} else {
			say(response, id);
		}
	}
});

document.querySelectorAll("#feed-placeholder-llama button.card").forEach((e) => {
	e.addEventListener("click", () => {
		let text = e.innerText.replace('\"', "").replace('" →', "");
		input.value = text;
	});
});
document.querySelectorAll("#feed-placeholder-alpaca button.card").forEach((e) => {
	e.addEventListener("click", () => {
		let text = e.innerText.replace('\"', "").replace('" →', "");
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

const cpuText = document.querySelector("#cpu .text"); //("id .text")
const ramText = document.querySelector("#ram .text");
const cpuBar = document.querySelector("#cpu .bar-inner");
const ramBar = document.querySelector("#ram .bar-inner");
const emotionindicatorText = document.querySelector("#emotionindicator .text");
const LLMChildEngineIndicatorText = document.querySelector("#LLMChildEngineIndicator .text");
const LLMChildEngineIndicatorTextBar = document.querySelector("#LLMChildEngineIndicator .bar-inner");
const SystemBackplateInfoText = document.querySelector("#SystemBackendPlateInfo .text");

const HardwareStressLoadText = document.querySelector("#stressload .text");
const HardwareStressLoadBar = document.querySelector("#stressload .bar-inner");

const UMALoadText = document.querySelector("#UMAMemAlloc .text"); //GB usage
const UMALoadBar = document.querySelector("#UMAMemAlloc .bar-inner");
const maxDefaultUMAJavascriptAlloc = 3.8 ;//tested before the js v8 engine crashes due to ran out of memory 3.8GB

let maxBackBrainQueue_Fallback=3; //I Don't know the max queue for the BackBrain just yet so i'm just going to assume 3

const BackBrainQueueText = document.querySelector("#BackBrainQueue .text"); 
const BackBrainQueueBar = document.querySelector("#BackBrainQueue .bar-inner");

const BackBrainResultQueueText = document.querySelector("#BackBrainFinished .text"); 
const BackBrainResultQueueBar = document.querySelector("#BackBrainFinished .bar-inner");

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
//Render Alloc buffet Graphics Processing Allocator Manager

// This basically set and send the data into ipcRenderer cpuUsage which manipulate the "green bar", maybe we can learn from this to create a progress bar 
function engineStatsFetchReq(){
	ipcRenderer.send("cpuUsage");
	ipcRenderer.send("freemem");
	ipcRenderer.send("hardwarestressload");
	ipcRenderer.send("emotioncurrentDebugInterfaceFetch");
	ipcRenderer.send("UMACheckUsage");
	ipcRenderer.send("BackBrainQueueCheck");
	ipcRenderer.send("BackBrainQueueResultCheck");
	ipcRenderer.send("timingDegradationCheckFactor");
	ipcRenderer.send("timingDegradationCheck");
}

engineStatsFetchReq
setInterval(async () => {
	engineStatsFetchReq();
	
}, generateRandomNumber(60000,120000)); //Randomize so it doesn't run in one frame

// For some reason cpuUsage and freemem everytime its updating its eating huge amount of GPU power ?
// internalTEProgress for recieving internalThoughtEngineProgress;
// const cpuText = document.querySelector("#cpu .text"); //("#id .text") make an example from how to choose the specific part of the html that wanted to be changed

let dynamicTips = document.querySelector("#dynamicTipsUIPart .info-bottom-changable");
let dynamicTipsBar = document.querySelector("#dynamicTipsUIPart .loading-bar");

function requestInternalThoughtStat(){
	ipcRenderer.send("internalThoughtProgressGUI");
	ipcRenderer.send("internalThoughtProgressTextGUI");
}

requestInternalThoughtStat();
setInterval(async () => {
	requestInternalThoughtStat();
	//console.log("fetching progress!"); // this is my first time programming polling ipc so yeah i know its cringy to have a verbose message in here too though, but i guess everyone has its own starting point :/
}, generateRandomNumber(1000,5000));



ipcRenderer.send("SystemBackplateHotplugCheck");
setInterval(async () => {
	ipcRenderer.send("SystemBackplateHotplugCheck");
}, generateRandomNumber(6000,10000));


let dynamiTipsProgress="";
ipcRenderer.on("internalTEProgressText", (_event, { data }) => {
	dynamiTipsProgress=data;
});


ipcRenderer.on("timingDegradationFactorReciever_renderer", (_event, { data }) => {
	console.debug("Factor Degradation GUI", data);
});

ipcRenderer.on("timingDegradationReciever_renderer", (_event, { data }) => {
	console.debug("Timing Degradation GUI (ms)", data);
});



ipcRenderer.on("systemBackPlateInfoView", (_event, { data }) => {
	SystemBackplateInfoText.style.opacity = 0;
	SystemBackplateInfoText.style.transition = 'opacity 0.5s ease-in-out';
	SystemBackplateInfoText.style.textAlign = 'left';
	setTimeout(() => {
		SystemBackplateInfoText.innerText = `${data}`;
		SystemBackplateInfoText.style.opacity = 1;
	}, 1000);
});

ipcRenderer.on("emotionDebugInterfaceStatistics", (_event, {data}) => {
	emotionindicatorText.style.opacity = 0;
	emotionindicatorText.style.transition = 'opacity 0.5s ease-in-out';
	// later on the bar is going to be color spectrum representing the emotion
	setTimeout(() => {
		emotionindicatorText.innerText = `❤️‍🩹 Current Emotion : ${data}`;
		emotionindicatorText.style.opacity = 1;
	}, 1000);
});

ipcRenderer.on("BackBrainQueueCheck_render", (_event, {data}) => {
	BackBrainQueueText.style.opacity = 0;
	BackBrainQueueText.style.transition = 'opacity 0.5s ease-in-out';
	BackBrainQueueBar.style.transition = 'transform 0.5s ease-in-out';
	BackBrainQueueBar.style.transform = `scaleX(${data/maxBackBrainQueue_Fallback})`; //since it uses 0.0 to 1.0
	if (BackBrainQueueBar.style.transform > 1){BackBrainQueueBar.style.transform=1}
	// later on the bar is going to be color spectrum representing the emotion
	setTimeout(() => {
		BackBrainQueueText.innerText = `BackBrain Async Queue Line : ${data}`;
		BackBrainQueueText.style.opacity = 1;
	}, 1000);
});

ipcRenderer.on("BackBrainQueueResultCheck_render", (_event, {data}) => {
	BackBrainResultQueueText.style.opacity = 0;
	BackBrainResultQueueText.style.transition = 'opacity 0.5s ease-in-out';
	BackBrainResultQueueBar.style.transition = 'transform 0.5s ease-in-out';
	BackBrainResultQueueBar.style.transform = `scaleX(${data/maxBackBrainQueue_Fallback})`; //since it uses 0.0 to 1.0
	if (BackBrainResultQueueBar.style.transform > 1){BackBrainResultQueueBar.style.transform=1}
	// later on the bar is going to be color spectrum representing the emotion
	setTimeout(() => {
		BackBrainResultQueueText.innerText = `BackBrain Submission Queue: ${data}`;
		BackBrainResultQueueText.style.opacity = 1;
	}, 1000);
});

ipcRenderer.on("internalTEProgress", (_event, { data }) => {
	if (data == 0){
		dynamicTips.innerText = "Random Tips: Shift + Enter for multiple lines";
		LLMChildEngineIndicatorText.innerText = "💤 LLMChild not invoked";
		dynamicTips.style.transition = 'opacity 0.5s ease-in-out';
		dynamicTips.style.opacity = 1;
		dynamicTipsBar.style.width = `${data}%`;
		LLMChildEngineIndicatorTextBar.style.transition = 'width 0.5s ease-in-out'
		LLMChildEngineIndicatorTextBar.style.width = `${data}%`;
		dynamicTipsBar.style.transition = 'width 0.5s ease-in-out';
	}else{
		dynamicTips.style.transition = 'opacity 1.5s ease-in-out';
		dynamicTips.style.opacity = 0;
		LLMChildEngineIndicatorText.style.opacity=0;
		dynamicTipsBar.style.width = `${data}%`;
		LLMChildEngineIndicatorTextBar.style.width = `${data}%`;
		setTimeout(() => {
			if(dynamiTipsProgress == ""){
				dynamicTips.innerText = "Waiting Adelaide Engine to Give Progress Report!"
				LLMChildEngineIndicatorText.innerText = `🤔 Adelaide Engine Invoked`
			} else {
				dynamicTips.innerText = dynamiTipsProgress;
				LLMChildEngineIndicatorText.innerText = `🤔 Adelaide Engine Processing : ${dynamicTips.innerText}`
			}
			 // no We can't do this, we need to replace it with something more dynamic and more saying verbose what its doing
			// Add fade-in effect to the new text
			// change the engine control panel or activity monitor on the info
			dynamicTips.style.transition = 'opacity 0.5s ease-in-out';
			LLMChildEngineIndicatorText.style.transition = 'opacity 0.5s ease-in-out';
			dynamicTips.style.opacity = 1;
		}, 1000); // Adjust the delay as needed to match the transition duration
	}

});


ipcRenderer.on("cpuUsage", (_event, { data }) => {
	cpuPercent = Math.round(data * 100);
	cpuText.style.opacity = 0;
	//cpuText.innerText = `🧠 ${cpuPercent}%, ${threadUtilized}/${cpuCount} threads`;
	cpuText.style.transition = 'opacity 1.5s ease-in-out';
	cpuBar.style.transition = 'transform 0.5s ease-in-out';
	cpuBar.style.transform = `scaleX(${cpuPercent / 100})`;
	setTimeout(() => {
		cpuText.innerText = `🧠 CPU Usage ${cpuPercent}%`;
		// Add fade-in effect to the new text
		cpuText.style.transition = 'opacity 0.5s ease-in-out';
		cpuText.style.opacity = 1;
	}, 1000); // Adjust the delay as needed to match the transition duration
});
ipcRenderer.on("freemem", (_event, { data }) => {
	freemem = data;
	ramText.style.opacity = 0;
	ramText.style.transition = 'opacity 1.5s ease-in-out';
	ramBar.style.transition = 'transform 0.5s ease-in-out';
	ramBar.style.transform = `scaleX(${(totalmem - freemem) / totalmem})`;
	setTimeout(() => {
		//ramText.innerText = `💾 ${Math.round((totalmem - freemem) * 10) / 10}GB/${totalmem}GB`;
		ramText.innerText = `💾 Allocated RAM ${Math.round((totalmem - freemem) * 10) / 10}GB`;
		// Add fade-in effect to the new text
		ramText.style.transition = 'opacity 0.5s ease-in-out';
		ramText.style.opacity = 1;
	}, 1000); // Adjust the delay as needed to match the transition duration
});

ipcRenderer.on("UMAAllocSizeStatisticsGB", (_event, { data }) => {
	const UMAallocGB = data;
	UMALoadText.style.opacity = 0;
	UMALoadText.style.transition = 'opacity 1.5s ease-in-out';
	UMALoadBar.style.transition = 'transform 0.5s ease-in-out';
	UMALoadBar.style.transform = `scaleX(${ UMAallocGB / maxDefaultUMAJavascriptAlloc })`;
	setTimeout(() => {
		UMALoadText.innerText = `UMA MLCMCF Alloc ${UMAallocGB}/${maxDefaultUMAJavascriptAlloc} GB`;
		// Add fade-in effect to the new text
		UMALoadText.style.transition = 'opacity 0.5s ease-in-out';
		UMALoadText.style.opacity = 1;
	}, 1000); // Adjust the delay as needed to match the transition duration
});

ipcRenderer.on("hardwareStressLoad", (_event, { data }) => {
	stressPercent = data;
	HardwareStressLoadText.style.opacity = 0;
	HardwareStressLoadText.style.transition = 'opacity 1.5s ease-in-out';
	HardwareStressLoadBar.style.transition = 'transform 0.5s ease-in-out';
	HardwareStressLoadBar.style.transform = `scaleX(${stressPercent/100})`; //since it uses 0.0 to 1.0
	setTimeout(() => {
		//ramText.innerText = `💾 ${Math.round((totalmem - freemem) * 10) / 10}GB/${totalmem}GB`;
		HardwareStressLoadText.innerText = `🔥 Hardware Stress ${stressPercent} %`;
		// Add fade-in effect to the new text
		HardwareStressLoadText.style.transition = 'opacity 0.5s ease-in-out';
		HardwareStressLoadText.style.opacity = 1;
	}, 1000); // Adjust the delay as needed to match the transition duration
});


document.getElementById("clear").addEventListener("click", () => {
	input.value = "";
	setTimeout(() => {
		input.style.height = "auto";
		input.style.height = input.scrollHeight + "px";
	});
});

document.getElementById("interaction-session-reset").addEventListener("click", () => {
	stopButton.click();
	stopButton.removeAttribute("disabled");
	ipcRenderer.send("restart");
	ipcRenderer.send("resetInteractionHistoryCTX");
	document.querySelectorAll("#messages li").forEach((element) => {
		element.style.opacity = 1;
		element.style.transition = 'opacity 1s ease-in-out';
		element.style.opacity = 0;
		element.remove();
	});
	setTimeout(() => {
		document.querySelectorAll("#messages li").forEach((element) => {
			element.style.opacity = 1;
			element.style.transition = 'opacity 1s ease-in-out';
			element.style.opacity = 0;
			element.remove();
		});
	}, 1000);
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

ipcRenderer.send("getParams");
document.getElementById("interactionHistorySession").addEventListener("click", () => {
	document.getElementById("interactionHistory-dialog-bg").classList.remove("hidden");
	ipcRenderer.send("getParams");
});


ipcRenderer.send("getParams");
document.getElementById("autonomousHandlessInteraction").addEventListener("click", () => {
	document.getElementById("AutonomousHandlessAssistantSwitch-dialog-bg").classList.remove("hidden");
	ipcRenderer.send("getParams");
});

ipcRenderer.on("params", (_event, data) => {
	// don't forget to scroll down to the bottom of index.js to update the value too
	//document.getElementById("LLMBackendMode").value = data.llmBackendMode; // since the program now come with predefined tuned model this won't be required and if its implemented it will make the program runs in buggy state or chaos like the button won't be able clicked without error and etc, and selection are implemented on dictionary on index.js in Engine section of the Data
	document.getElementById("repeat_last_n").value = data.repeat_last_n;
	document.getElementById("repeat_penalty").value = data.repeat_penalty;
	document.getElementById("top_k").value = data.top_k;
	document.getElementById("top_p").value = data.top_p;
	document.getElementById("temp").value = data.temp;
	document.getElementById("seed").value = data.seed;
	document.getElementById("QoSTimeoutLLMChildGlobal").value = data.qostimeoutllmchildglobal;
	document.getElementById("QoSTimeoutLLMChildSubCategory").value = data.qostimeoutllmchildsubcategory;
	document.getElementById("QoSTimeoutLLMChildBackBrainGlobalQueueMax").value = data.qostimeoutllmchildbackbrainglobalqueuemax;
	document.getElementById("QoSTimeoutSwitch").checked = data.qostimeoutswitch;
	document.getElementById("BackbrainQueue").checked = data.backbrainqueue;
	document.getElementById("web-access").checked = data.webAccess;
	document.getElementById("local-file-access").checked = data.localAccess;
	document.getElementById("LLMChildDecision").checked = data.llmdecisionMode;
	document.getElementById("longchainthought").checked = data.extensiveThought;
	document.getElementById("saverestoreinteraction").checked = data.SaveandRestoreInteraction;
	document.getElementById("historyChatCtx").checked = data.hisChatCTX;
	document.getElementById("foreveretchedmemory").checked = data.foreverEtchedMemory;
	document.getElementById("throwInitialGarbageResponse").checked = data.throwInitResponse;
	document.getElementById("classicmode").checked = data.classicMode;
	document.getElementById("attemptaccelerate").checked = data.AttemptAccelerate;
	document.getElementById("hardwarelayeroffloading").value = data.hardwareLayerOffloading;
	document.getElementById("sideloadexperienceuma").checked = data.sideloadExperienceUMA;
	document.getElementById("emotionalllmchildengine").checked = data.emotionalLLMChildengine;
	document.getElementById("profilepictureemotion").checked = data.profilePictureEmotion;
	document.getElementById("longchainthought-neverfeelenough").checked = data.longChainThoughtNeverFeelenough;
	document.getElementById("AutomateLoopback").checked = data.automateLoopback;
	document.getElementById("openaiapiserverhost").checked = data.openAPIServer;
	document.getElementById("ragprepromptprocesscontexting").checked = data.ragPrePromptProcessContexting;
	document.getElementById("selfreintegrate").checked = data.selfReintegrate;
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
			qostimeoutllmchildglobal: document.getElementById("QoSTimeoutLLMChildGlobal").value || document.getElementById("QoSTimeoutLLMChildGlobal").placeholder,
			qostimeoutllmchildsubcategory: document.getElementById("QoSTimeoutLLMChildSubCategory").value || document.getElementById("QoSTimeoutLLMChildSubCategory").placeholder,
			qostimeoutllmchildbackbrainglobalqueuemax: document.getElementById("QoSTimeoutLLMChildBackBrainGlobalQueueMax").value || document.getElementById("QoSTimeoutLLMChildBackBrainGlobalQueueMax").placeholder,
			qostimeoutswitch: document.getElementById("QoSTimeoutSwitch").checked,
			backbrainqueue: document.getElementById("BackbrainQueue").checked,
			ragPrePromptProcessContexting: document.getElementById("ragprepromptprocesscontexting").checked,
			webAccess: document.getElementById("web-access").checked,
			openaiapiserverhost: document.getElementById("openaiapiserverhost").checked,
			automateLoopback: document.getElementById("AutomateLoopback").checked,
			localAccess: document.getElementById("local-file-access").checked,
            llmdecisionMode: document.getElementById("LLMChildDecision").checked,
            extensiveThought: document.getElementById("longchainthought").checked,
            SaveandRestoreInteraction: document.getElementById("saverestoreinteraction").checked,
			hisChatCTX: document.getElementById("historyChatCtx").checked,
			foreverEtchedMemory: document.getElementById("foreveretchedmemory").checked,
			throwInitResponse: document.getElementById("throwInitialGarbageResponse").checked,
			classicMode: document.getElementById("classicmode").checked,
			AttemptAccelerate: document.getElementById("attemptaccelerate").checked,
			emotionalLLMChildengine: document.getElementById("emotionalllmchildengine").checked,
			profilePictureEmotion: document.getElementById("profilepictureemotion").checked,
			websearch_amount: document.getElementById("websearch_amount").value || document.getElementById("websearch_amount").placeholder,
			hardwareLayerOffloading: document.getElementById("hardwarelayeroffloading").value || document.getElementById("hardwarelayeroffloading").placeholder,
			longChainThoughtNeverFeelenough: document.getElementById("longchainthought-neverfeelenough").checked,
			selfReintegrate: document.getElementById("selfreintegrate").checked,
			sideloadExperienceUMA: document.getElementById("sideloadexperienceuma").checked
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

// We are adding an event listener to two buttons inside the interaction history dialog box.
// The first button with class "secondary" and the second button with class "primary"
// When either of these buttons is clicked, we want to hide the interaction history dialog box.

document.querySelector("#interactionHistory-dialog-bg > div > div.dialog-button > button.secondary").addEventListener("click", () => {
  document.getElementById("interactionHistory-dialog-bg").classList.add("hidden");
});

document.querySelector("#interactionHistory-dialog-bg > div > div.dialog-button > button.primary").addEventListener("click", () => {
    document.getElementById("interactionHistory-dialog-bg").classList.add("hidden");
});


document.querySelector("#AutonomousHandlessAssistantSwitch-dialog-bg > div > div.dialog-button > button.secondary").addEventListener("click", () => {
	document.getElementById("AutonomousHandlessAssistantSwitch-dialog-bg").classList.add("hidden");
  });
  
  document.querySelector("#AutonomousHandlessAssistantSwitch-dialog-bg > div > div.dialog-button > button.primary").addEventListener("click", () => {
	  document.getElementById("AutonomousHandlessAssistantSwitch-dialog-bg").classList.add("hidden");
  });

// This event listener listens for changes in the "BackbrainQueue" checkbox.
document.getElementById("BackbrainQueue").addEventListener("change", () => {
  // Sends a message to the main process indicating whether the "BackbrainQueue"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("backbrainqueue", document.getElementById("BackbrainQueue").checked);
});

// This event listener listens for changes in the "AutomateLoopback" checkbox.
document.getElementById("AutomateLoopback").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "AutomateLoopback"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("automateLoopback", document.getElementById("AutomateLoopback").checked);
});

// This event listener listens for changes in the "QoSTimeoutSwitch" checkbox.
document.getElementById("QoSTimeoutSwitch").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "QoSTimeoutSwitch"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("qostimeoutswitch", document.getElementById("QoSTimeoutSwitch").checked);
});

// This event listener listens for changes in the "web-access" checkbox.
document.getElementById("web-access").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "web-access"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("webAccess", document.getElementById("web-access").checked);
});

// This event listener listens for changes in the "ragprepromptprocesscontexting" checkbox.
document.getElementById("ragprepromptprocesscontexting").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "ragprepromptprocesscontexting"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("webAccess", document.getElementById("ragprepromptprocesscontexting").checked);
});

// This event listener listens for changes in the "local-file-access" checkbox.
document.getElementById("local-file-access").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "local-file-access"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("localAccess", document.getElementById("local-file-access").checked);
});

// This event listener listens for changes in the "LLMChildDecision" checkbox.
document.getElementById("LLMChildDecision").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "LLMChildDecision"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("llmdecisionMode", document.getElementById("LLMChildDecision").checked);
});

// This event listener listens for changes in the "longchainthought" checkbox.
document.getElementById("longchainthought").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "longchainthought"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("extensiveThought", document.getElementById("longchainthought").checked);
});

// This event listener listens for changes in the "saverestoreinteraction" checkbox.
document.getElementById("saverestoreinteraction").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "saverestoreinteraction"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("SaveandRestoreInteraction", document.getElementById("saverestoreinteraction").checked);
});

// This event listener listens for changes in the "historyChatCtx" checkbox.
document.getElementById("historyChatCtx").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "historyChatCtx"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("hisChatCTX", document.getElementById("historyChatCtx").checked);
});

// This event listener listens for changes in the "attemptaccelerate" checkbox.
document.getElementById("attemptaccelerate").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "attemptaccelerate"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("AttemptAccelerate", document.getElementById("attemptaccelerate").checked);
});

// This event listener listens for changes in the "emotionalllmchildengine" checkbox.
document.getElementById("emotionalllmchildengine").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "emotionalllmchildengine"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("emotionalLLMChildengine", document.getElementById("emotionalllmchildengine").checked);
});

// This event listener listens for changes in the "profilepictureemotion" checkbox.
document.getElementById("profilepictureemotion").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "profilepictureemotion"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("profilePictureEmotion", document.getElementById("profilepictureemotion").checked);
});

// This event listener listens for changes in the "longchainthought-neverfeelenough" checkbox.
document.getElementById("longchainthought-neverfeelenough").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "longchainthought-neverfeelenough"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("longChainThoughtNeverFeelenough", document.getElementById("longchainthought-neverfeelenough").checked);
});

// This event listener listens for changes in the "sideloadexperienceuma" checkbox.
document.getElementById("sideloadexperienceuma").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "sideloadexperienceuma"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("sideloadExperienceUMA", document.getElementById("sideloadexperienceuma").checked);
});


document.getElementById("selfreintegrate").addEventListener("change", () => {
    // Sends a message to the main process indicating whether the "selfreintegrate"
    // checkbox is checked or not. This allows the main process to know how to proceed.
    ipcRenderer.send("selfReintegrate", document.getElementById("selfreintegrate").checked);
});


/*
//default value
//document.getElementById("model").value = "alpaca";
document.getElementById("repeat_last_n").value = 64;
document.getElementById("repeat_penalty").value = 1.3;
document.getElementById("top_k").value = 420;
document.getElementById("top_p").value = 90;
document.getElementById("temp").value = 0.9;

*/