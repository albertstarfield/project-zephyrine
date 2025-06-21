package System_Info is
   -- Platform-agnostic interface
   function Uptime_Seconds return Long_Integer;
   -- Returns system uptime in seconds (-1 on error)
   
private
   -- Platform-specific implementations
   type Implementation is (Linux, Darwin, Windows, Unsupported);
   function Detect_Platform return Implementation;
   Current_Platform : constant Implementation := Detect_Platform;
end System_Info;