with Interfaces.C;

package body System_Info.Darwin is
   -- Bind to BSD sysctl
   function sysctl(
        name    : access Interfaces.C.int;
        namelen : Interfaces.C.unsigned;
        oldp    : access Interfaces.C.long;
        oldlenp : access Interfaces.C.size_t;
        newp    : System.Address;
        newlen  : Interfaces.C.size_t) return Interfaces.C.int;
   pragma Import(C, sysctl);

   function Uptime_Seconds return Long_Integer is
      mib    : aliased array(0..1) of Interfaces.C.int := (1, 21);  // CTL_KERN, KERN_BOOTTIME
      boottime : aliased Interfaces.C.long;
      size   : aliased Interfaces.C.size_t := Interfaces.C.size_t'Size/8;
   begin
      if sysctl(mib'Access, 2, boottime'Access, size'Access, System.Null_Address, 0) = 0 then
         return Long_Integer(Interfaces.C.long(System.Clock) - boottime);
      else
         return -1;
      end if;
   end Uptime_Seconds;
end System_Info.Darwin;