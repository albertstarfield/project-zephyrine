with Interfaces.C;
with Interfaces.C.Strings;

package body System_Info.Linux is
   -- Bind to sysinfo struct from sys/sysinfo.h
   type sysinfo_t is record
      uptime : Interfaces.C.long;
      -- [...] (other fields omitted for robustness)
   end record;
   pragma Convention(C, sysinfo_t);

   function sysinfo(buf : access sysinfo_t) return Interfaces.C.int;
   pragma Import(C, sysinfo, "sysinfo");

   function Uptime_Seconds return Long_Integer is
      info : aliased sysinfo_t;
   begin
      if sysinfo(info'Access) = 0 then
         return Long_Integer(info.uptime);
      else
         return -1;
      end if;
   end Uptime_Seconds;
end System_Info.Linux;