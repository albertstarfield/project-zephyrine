#!/bin/bash
cd /Users/albertstarfield/LibraryTube/OpenIntellegentiaPlatform/Adelaide_Lite
alr exec -- ./bin/adelaide_server &
SERVER_PID=$!
sleep 5
curl -X POST http://127.0.0.1:11420/api/chat -d '{"model":"zephy","messages":[{"role":"user","content":"What is anarchy?"}]}'
kill $SERVER_PID
