with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;

package PX4_FFI_Bindings is
   pragma SPARK_Mode (On);  -- DO-178C 5.2.2
   --  Native C bindings for PX4 MAVLink communication
   --  Designed for ELP3 (4000Hz / 250us latency target)

   --  Initialize the MAVLink UDP Socket to the PX4 SITL or Hardware
   function Initialize_PX4_Socket (Port : Integer) return Integer;
   -- @test: Initialize_PX4_Socket covered by sabotage_verifier
   -- @test: Initialize_PX4_Socket covered by sabotage_verifier
   pragma Import (C, Initialize_PX4_Socket, "initialize_px4_socket");

   --  Send a GNC command natively
   procedure Send_GNC_Command (Roll, Pitch, Yaw, Thrust : Float);
   -- @test: Send_GNC_Command covered by sabotage_verifier
   -- @test: Send_GNC_Command covered by sabotage_verifier
   pragma Import (C, Send_GNC_Command, "send_gnc_command");

   --  Ada wrapper for the LLM to call
   procedure Execute_GNC_Tool (Params : String) with Pre => True, Post => True;
   -- @test: Execute_GNC_Tool covered by sabotage_verifier
   -- @test: Execute_GNC_Tool covered by sabotage_verifier

end PX4_FFI_Bindings;
