package System_Info is
   -- Returns system uptime in seconds or -1 on error
   function Uptime_Seconds return Long_Integer;
   
private
   type Implementation is (Linux, Darwin, Windows, Unsupported);
   function Detect_Platform return Implementation;
   pragma Inline(Detect_Platform);
   
   Current_Platform : constant Implementation := Detect_Platform;
end System_Info;