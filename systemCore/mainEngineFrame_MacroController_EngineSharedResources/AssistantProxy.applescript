-- AssistantProxy.applescript
-- Version with Timeouts and Enhanced Logging

use AppleScript version "2.4" -- Yosemite (10.10) or later
use scripting additions

-- Main handler called by osascript from Python
on handleAction given parameters:{actionType:actionType, actionParamsJSON:actionParamsJSON}
	log "AssistantProxy: ----- New Action Request -----"
	log "AssistantProxy: Received actionType=" & actionType & ", params=" & actionParamsJSON

	set outputResult to "Error: Action processing failed." -- Default error

	-- Main try block for action execution and error handling
	try
		-- === ACTION HANDLERS ===

		if actionType is "prime_permissions" then
			set prime_results to {}
			log "Attempting to prime permissions (Read Attempt)..."
			
			-- Try accessing Calendar count with timeout
			log "Priming: Telling Calendar..."
			try
				with timeout of 15 seconds
					tell application "Calendar"
						log "Priming: Accessing Calendar count..."
						set calCount to count of calendars
						set end of prime_results to "Calendar access attempt finished (Count: " & calCount & ")."
						log "Priming: Calendar OK."
					end tell
				end timeout
			on error errMsg number errNum
				log "Priming Calendar Error: " & errMsg & " (" & errNum & ")"
				set end of prime_results to "Calendar access failed/denied (" & errNum & ")."
			end try
			
			-- Try reading Contacts count with timeout
			log "Priming: Telling Contacts..."
			try
				with timeout of 15 seconds
					tell application "Contacts"
						log "Priming: Accessing Contacts count..."
						set contactCount to count of people
						set end of prime_results to "Contacts access attempt finished (Count: " & contactCount & ")."
						log "Priming: Contacts OK."
					end tell
				end timeout
			on error errMsg number errNum
				log "Priming Contacts Error: " & errMsg & " (" & errNum & ")"
				set end of prime_results to "Contacts access failed/denied (" & errNum & ")."
			end try
			
			-- Try reading Reminders count with timeout
			log "Priming: Telling Reminders..."
			try
				with timeout of 15 seconds
					tell application "Reminders"
						log "Priming: Accessing Reminders count..."
						set listCount to count of lists
						set end of prime_results to "Reminders access attempt finished (Count: " & listCount & ")."
						log "Priming: Reminders OK."
					end tell
				end timeout
			on error errMsg number errNum
				log "Priming Reminders Error: " & errMsg & " (" & errNum & ")"
				set end of prime_results to "Reminders access failed/denied (" & errNum & ")."
			end try
			
			set AppleScript's text item delimiters to ", "
			set outputResult to "Permission priming sequence finished. Results: " & (prime_results as text)
			set AppleScript's text item delimiters to ""

		else if actionType is "open_app_or_file" then
			set targetPath to getParameter(actionParamsJSON, "target", "")
			if targetPath is not "" then
				log "Executing: open " & quoted form of targetPath
				do shell script "open " & quoted form of targetPath
				set outputResult to "Attempted to open: " & targetPath
			else
				set outputResult to "Error: Missing target path to open."
			end if

		else if actionType is "get_disk_space" then
			log "Executing: df -h / ..."
			set dfOutput to do shell script "df -h / | tail -n 1 | awk '{print $4 \" available\"}'"
			set outputResult to "Disk Space Check: " & dfOutput

		else if actionType is "search_web" then
			set queryText to getParameter(actionParamsJSON, "query", "")
			if queryText is not "" then
				log "Executing: open web search for " & quoted form of queryText
				do shell script "open 'https://www.google.com/search?q=' & quoted form of queryText & ''"
				set outputResult to "Opened web search for: " & queryText
			else
				set outputResult to "Error: Missing query for web search."
			end if

		else if actionType is "schedule_meeting" then
			log "Executing: schedule_meeting (basic - activating Calendar)"
			try
				with timeout of 15 seconds
					tell application "Calendar" to activate
				end timeout
				set outputResult to "Opened Calendar to schedule meeting."
			on error errMsg number errNum
				log "Error activating Calendar: " & errMsg
				set outputResult to "Error activating Calendar: " & errMsg
			end try

		else if actionType is "define_word" then
			set theWord to getParameter(actionParamsJSON, "word", "")
			if theWord is not "" then
				log "Executing: define_word web search for " & quoted form of theWord
				do shell script "open 'https://www.google.com/search?q=define+' & quoted form of theWord & ''"
				set outputResult to "Opened web search for definition of: " & theWord
			else
				set outputResult to "Error: Missing word for definition."
			end if

		else if actionType is "send_text" then
			set contactName to getParameter(actionParamsJSON, "contact_name", "")
			set messageBody to getParameter(actionParamsJSON, "message_body", "")
			if contactName is not "" and messageBody is not "" then
				log "Executing: send_text (Placeholder) to " & contactName
				-- Requires complex scripting and permissions
				-- try
				--   with timeout of 30 seconds
				--     tell application "Messages"
				--       set targetService to 1st service whose service type = iMessage
				--       set targetBuddy to buddy contactName of targetService
				--       send messageBody to targetBuddy
				--     end tell
				--   end timeout
				--   set outputResult to "Sent text to " & contactName & "."
				-- on error errMsg number errNum
				--   log "Error sending text: " & errMsg
				--   set outputResult to "Error sending text: " & errMsg
				-- end try
				set outputResult to "Prepared text for " & contactName & ". (Messages integration needed)"
			else
				set outputResult to "Error: Missing contact name or message body for send_text."
			end if

		-- === ADD MORE 'else if' BLOCKS FOR OTHER ACTIONS HERE ===

		else
			-- Handle unknown action type if none of the above matched
			set outputResult to "Error: Unknown action type '" & actionType & "'."
			log outputResult
		end if -- *** This closes the main if/else if block for actions ***

	-- Catch errors from the main action execution try block
	on error errMsg number errNum
		log "Error executing action " & actionType & " (Outer Try Block): " & errMsg & " (" & errNum & ")"
		set outputResult to "Error encountered while trying to perform action '" & actionType & "': " & errMsg
	end try

	log "AssistantProxy: Action " & actionType & " finished. Result: " & outputResult
	return outputResult -- Return the final result string

end handleAction


-- Helper function to get parameter value from JSON string (basic parsing)
on getParameter(jsonString, paramName, defaultValue)
	set searchKey to ("\"" & paramName & "\":\"")
	set endKey to "\""
	set AppleScript's text item delimiters to searchKey
	try
		set tempList to text items of jsonString
		if (count of tempList) > 1 then
			set nextPart to item 2 of tempList
			set AppleScript's text item delimiters to endKey
			set paramValue to text item 1 of nextPart
			set AppleScript's text item delimiters to "" -- Reset delimiters
			set paramValue to replaceText(paramValue, "\\\\", "\\") -- Unescape \\ -> \
			set paramValue to replaceText(paramValue, "\\\"", "\"") -- Unescape \" -> "
			if paramValue is not "" then return paramValue
		end if
	on error errMsg number errNum
		log "Note: Basic parameter parsing failed for " & paramName & ". Error: " & errMsg
	end try
	set AppleScript's text item delimiters to "" -- Ensure delimiters are reset
	return defaultValue
end getParameter


-- Helper function to replace text occurrences
on replaceText(sourceText, searchString, replaceString)
	set AppleScript's text item delimiters to searchString
	set textItems to text items of sourceText
	set AppleScript's text item delimiters to replaceString
	set resultText to textItems as string
	set AppleScript's text item delimiters to ""
	return resultText
end replaceText