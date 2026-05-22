import { setGlobalDispatcher, Agent } from "undici";
setGlobalDispatcher(new Agent());
fetch("http://127.0.0.1:11421/v1/chat/completions").catch(console.error);
fetch("https://api.openai.com/v1/chat/completions").catch(console.error);
