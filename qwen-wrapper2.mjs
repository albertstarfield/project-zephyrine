import http from 'http';
import https from 'https';

const originalFetch = global.fetch;
global.fetch = async (...args) => {
  console.log('>>> GLOBAL FETCH URL:', args[0]);
  console.log('>>> GLOBAL FETCH OPTIONS:', JSON.stringify(args[1], null, 2));
  try {
    const res = await originalFetch(...args);
    return res;
  } catch (err) {
    console.log('>>> GLOBAL FETCH ERROR:', err.message);
    throw err;
  }
};

await import('file:///opt/homebrew/Cellar/qwen-code/0.15.11/libexec/lib/node_modules/@qwen-code/qwen-code/cli.js');
