with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;

package PX4_FFI_Bindings is
   --  Native C bindings for PX4 MAVLink communication
   --  Designed for ELP3 (4000Hz / 250us latency target)

   --  Initialize the MAVLink UDP Socket to the PX4 SITL or Hardware
   function Initialize_PX4_Socket (Port : Integer) return Integer;
   pragma Import (C, Initialize_PX4_Socket, "initialize_px4_socket");

   --  Send a GNC command natively
   procedure Send_GNC_Command (Roll, Pitch, Yaw, Thrust : Float);
   pragma Import (C, Send_GNC_Command, "send_gnc_command");

   --  Ada wrapper for the LLM to call
   procedure Execute_GNC_Tool (Params : String);

end PX4_FFI_Bindings;
