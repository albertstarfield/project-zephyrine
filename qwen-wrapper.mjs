import http from 'http';
import https from 'https';

const originalFetch = global.fetch;
global.fetch = async (...args) => {
  console.log('>>> GLOBAL FETCH URL:', args[0]);
  
  if (args[1] && args[1].dispatcher) {
    console.log('>>> DELETING DISPATCHER FROM FETCH OPTIONS!');
    delete args[1].dispatcher;
  }

  try {
    const res = await originalFetch(...args);
    return res;
  } catch (err) {
    console.log('>>> GLOBAL FETCH ERROR:', err.message);
    throw err;
  }
};

const origHttpReq = http.request;
http.request = (...args) => {
  console.log('>>> HTTP REQUEST:', args[0]?.href || args[0]);
  return origHttpReq(...args);
};

const origHttpsReq = https.request;
https.request = (...args) => {
  console.log('>>> HTTPS REQUEST:', args[0]?.href || args[0]);
  return origHttpsReq(...args);
};

await import('file:///opt/homebrew/Cellar/qwen-code/0.15.11/libexec/lib/node_modules/@qwen-code/qwen-code/cli.js');
