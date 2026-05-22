import http from 'http';
import https from 'https';

const originalFetch = global.fetch;
global.fetch = async (...args) => {
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

await import('file:///opt/homebrew/Cellar/qwen-code/0.15.11/libexec/lib/node_modules/@qwen-code/qwen-code/cli.js');
