package body System_Info is
   function Detect_Platform return Implementation is
   begin
      #ifdef linux then
         return Linux;
      #elsif Darwin then
         return Darwin;
      #elsif Windows then
         return Windows;
      #else
         return Unsupported;
      #end if;
   end Detect_Platform;

   Current_Platform : constant Implementation := Detect_Platform;

   function Uptime_Seconds return Long_Integer is
   begin
      case Current_Platform is
         when Linux => return Linux.Uptime_Seconds;
         when Darwin => return Darwin.Uptime_Seconds;
         when others => return -1;  -- Robust error handling
      end case;
   end Uptime_Seconds;
end System_Info;