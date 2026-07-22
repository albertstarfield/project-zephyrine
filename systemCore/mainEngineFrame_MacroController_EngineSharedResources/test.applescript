use AppleScript version "2.5"
use framework "Foundation"

set searchWord to "strawberry"
set searchURL to "https://www.google.com/search?q=" & quoted form of searchWord

try
    tell application "Safari" to make new document with properties {URL:searchURL}
catch error message as errorMessage
    return errorMessage as text
end try

return "Search performed in Safari. Please open Safari and click on the link to see results."