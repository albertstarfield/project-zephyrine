with Interfaces.C;
with System;  -- For Clock function
with System.Config;  -- For OS detection

package body System_Info is
   function Detect_Platform return Implementation is
      use System.Config;
   begin
      if OS = "Linux" then
         return Linux;
      elsif OS = "Darwin" then
         return Darwin;
      elsif OS = "Windows" then
         return Windows;
      else
         return Unsupported;
      end if;
   end Detect_Platform;

   -- Platform-specific implementations
   function Linux_Uptime return Long_Integer;
   function Darwin_Uptime return Long_Integer;
   function Windows_Uptime return Long_Integer;

   function Uptime_Seconds return Long_Integer is
   begin
      case Current_Platform is
         when Linux    => return Linux_Uptime;
         when Darwin  => return Darwin_Uptime;
         when Windows => return Windows_Uptime;
         when others  => return -1;
      end case;
   end Uptime_Seconds;

   -- Linux implementation
   function Linux_Uptime return Long_Integer is
      type sysinfo_t is record
         uptime : Interfaces.C.long;
      end record;
      pragma Convention(C, sysinfo_t);

      function sysinfo(buf : access sysinfo_t) return Interfaces.C.int;
      pragma Import(C, sysinfo, "sysinfo");

      Info : aliased sysinfo_t;
   begin
      if sysinfo(Info'Access) = 0 then
         return Long_Integer(Info.uptime);
      else
         return -1;
      end if;
   end Linux_Uptime;

   -- macOS implementation
   function Darwin_Uptime return Long_Integer is
      MIB : constant array (0..1) of Interfaces.C.int := (1, 21);  -- CTL_KERN, KERN_BOOTTIME
      
      type Timeval is record
         tv_sec  : Interfaces.C.long;
         tv_usec : Interfaces.C.long;
      end record;
      pragma Convention(C, Timeval);

      function sysctl(
         name    : access Interfaces.C.int;
         namelen : Interfaces.C.unsigned;
         oldp    : access Timeval;
         oldlenp : access Interfaces.C.size_t;
         newp    : System.Address;
         newlen  : Interfaces.C.size_t) return Interfaces.C.int;
      pragma Import(C, sysctl);

      Boot_Time : aliased Timeval;
      Len       : aliased Interfaces.C.size_t := Timeval'Size / 8;
   begin
      if sysctl(MIB'Access, 2, Boot_Time'Access, Len'Access, 
                System.Null_Address, 0) = 0 
      then
         declare
            Now : constant Long_Integer := 
              Long_Integer(System.Clock / System.Clocks_Per_Second);
         begin
            return Now - Long_Integer(Boot_Time.tv_sec);
         end;
      else
         return -1;
      end if;
   end Darwin_Uptime;

   -- Windows implementation
   function Windows_Uptime return Long_Integer is
      function GetTickCount64 return Interfaces.C.unsigned_long_long;
      pragma Import(C, GetTickCount64, "GetTickCount64");
   begin
      return Long_Integer(GetTickCount64 / 1000);
   end Windows_Uptime;
end System_Info;