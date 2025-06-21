with Interfaces.C;
with System;

package body System_Info is

   -- Platform Detection --
   function Detect_Platform return Implementation is
      -- Windows detection variables
      function GetVersion return Interfaces.C.unsigned_long;
      pragma Import(C, GetVersion, "GetVersion");
      
      Is_Windows : Boolean := False;
      Dummy_Version : Interfaces.C.unsigned_long;
      
      -- POSIX detection variables
      function uname(buf : System.Address) return Interfaces.C.int;
      pragma Import(C, uname, "uname");
      
      type UtsName is record
         sysname : Interfaces.C.char_array(1 .. 65);
      end record;
      pragma Convention(C, UtsName);
      
      Name : aliased UtsName := (sysname => (others => Interfaces.C.nul));
      Sysname : String(1 .. 65);
   begin
      -- Windows check
      begin
         if GetVersion'Address /= System.Null_Address then
            Dummy_Version := GetVersion;
            Is_Windows := True;
         end if;
      exception
         when others => null;
      end;

      if Is_Windows then
         return Windows;
      end if;

      -- POSIX check
      if uname(Name'Access) = 0 then
         Sysname := Interfaces.C.To_Ada(Name.sysname);
         if Sysname'Length >= 5 and then Sysname(1 .. 5) = "Linux" then
            return Linux;
         elsif Sysname'Length >= 6 and then Sysname(1 .. 6) = "Darwin" then
            return Darwin;
         end if;
      end if;
      
      return Unsupported;
   exception
      when others =>
         return Unsupported;
   end Detect_Platform;

   -- Linux Implementation --
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

   -- macOS Implementation --
   function Darwin_Uptime return Long_Integer is
      MIB : constant array(0 .. 1) of Interfaces.C.int := (1, 21);  -- CTL_KERN, KERN_BOOTTIME
      
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
      pragma Import(C, sysctl, "sysctl");

      Boot_Time : aliased Timeval;
      Len       : aliased Interfaces.C.size_t := Interfaces.C.size_t(Timeval'Size / 8);
      Now       : Long_Integer;
   begin
      if sysctl(MIB'Unchecked_Access, 2, Boot_Time'Access, Len'Access, 
                System.Null_Address, 0) = 0 then
         Now := Long_Integer(System.Clock / System.Tick);
         return Now - Long_Integer(Boot_Time.tv_sec);
      else
         return -1;
      end if;
   end Darwin_Uptime;

   -- Windows Implementation --
   function Windows_Uptime return Long_Integer is
      function GetTickCount64 return Interfaces.C.unsigned_long_long;
      pragma Import(C, GetTickCount64, "GetTickCount64");
      
      Result : Interfaces.C.unsigned_long_long;
      Divider : constant := 1000;
   begin
      Result := GetTickCount64;
      return Long_Integer(Result / Interfaces.C.unsigned_long_long(Divider));
   exception
      when others =>
         return -1;
   end Windows_Uptime;

   -- Public Interface --
   function Uptime_Seconds return Long_Integer is
   begin
      case Current_Platform is
         when Linux    => return Linux_Uptime;
         when Darwin  => return Darwin_Uptime;
         when Windows => return Windows_Uptime;
         when others  => return -1;
      end case;
   end Uptime_Seconds;

end System_Info;