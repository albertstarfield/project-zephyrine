const originalFetch = global.fetch;
global.fetch = async (...args) => {
  console.log('>>> GLOBAL FETCH URL:', args[0]);
  try {
    const res = await originalFetch(...args);
    return res;
  } catch (err) {
    console.log('>>> GLOBAL FETCH ERROR:', err.message);
    throw err;
  }
};
const http = require('http');
const origHttpReq = http.request;
http.request = (...args) => {
  console.log('>>> HTTP REQUEST:', args[0]?.href || args[0]);
  return origHttpReq(...args);
};
const https = require('https');
const origHttpsReq = https.request;
https.request = (...args) => {
  console.log('>>> HTTPS REQUEST:', args[0]?.href || args[0]);
  return origHttpsReq(...args);
};
require('/opt/homebrew/Cellar/qwen-code/0.15.11/libexec/lib/node_modules/@qwen-code/qwen-code/cli.js');
