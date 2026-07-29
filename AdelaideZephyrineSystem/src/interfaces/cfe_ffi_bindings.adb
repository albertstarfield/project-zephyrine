with Ada.Text_IO; use Ada.Text_IO;
with System;

package body CFE_FFI_Bindings is

   --  Internal state
   Initialized  : Boolean := False;
   Command_Pipe : CFE_SB_PipeId_t := 0;
   Telemetry_Pipe : CFE_SB_PipeId_t := 0;

   --  ──────────────────────────────────────────────────────────────────────
   --  CFE_Initialize: Set up the Software Bus interface
   --  ──────────────────────────────────────────────────────────────────────
   procedure CFE_Initialize is
      Pipe_Name_Cmd  : constant String := "ADELAIDE_CMD" & Character'Val (0);
      Pipe_Name_Tlm  : constant String := "ADELAIDE_TLM" & Character'Val (0);
      Status         : CFE_Status_t;
   begin
      if Initialized then
         return;
      end if;

      Put_Line ("[CFE-FFI] Initializing cFE Software Bus interface...");

      --  Create command pipe (depth 16)
      Status := CFE_SB_CreatePipe
        (PipeIdPtr => Command_Pipe'Access,
         Depth     => 16,
         PipeName  => Interfaces.C.Strings.New_String (Pipe_Name_Cmd));
      if Status /= CFE_SUCCESS then
         Put_Line ("[CFE-FFI] WARNING: Failed to create command pipe, status=" &
                   CFE_Status_t'Image (Status));
      end if;

      --  Create telemetry pipe (depth 32)
      Status := CFE_SB_CreatePipe
        (PipeIdPtr => Telemetry_Pipe'Access,
         Depth     => 32,
         PipeName  => Interfaces.C.Strings.New_String (Pipe_Name_Tlm));
      if Status /= CFE_SUCCESS then
         Put_Line ("[CFE-FFI] WARNING: Failed to create telemetry pipe, status=" &
                   CFE_Status_t'Image (Status));
      end if;

      Initialized := True;
      Put_Line ("[CFE-FFI] cFE Software Bus initialized. Cmd_Pipe=" &
                CFE_SB_PipeId_t'Image (Command_Pipe) &
                " Tlm_Pipe=" & CFE_SB_PipeId_t'Image (Telemetry_Pipe));
   end CFE_Initialize;

   --  ──────────────────────────────────────────────────────────────────────
   --  CFE_Send_Telemetry: Send a telemetry string through the Software Bus
   --  ──────────────────────────────────────────────────────────────────────
   procedure CFE_Send_Telemetry (Payload : String) is
      Status : CFE_Status_t;
   begin
      if not Initialized then
         CFE_Initialize;
      end if;

      --  For now, log to console (real implementation would use CFE_SB_TransmitMsg
      --  with a properly formatted CFE_MSG_Message_t containing the payload)
      Put_Line ("[CFE-TLM] " & Payload);

      --  TODO: Build CFE_MSG_Message_t header + payload, then:
      --  Status := CFE_SB_TransmitMsg (Msg_Ptr, IsOrigination => True);
   end CFE_Send_Telemetry;

   --  ──────────────────────────────────────────────────────────────────────
   --  CFE_Send_Info_Event: Send an informational event
   --  ──────────────────────────────────────────────────────────────────────
   procedure CFE_Send_Info_Event (Message : String) is
      Status : CFE_Status_t;
   begin
      Status := CFE_EVS_SendEvent
        (EventId   => 16#0001#,
         EventType => CFE_EVS_EventType_INFORMATIONAL,
         Spec      => Interfaces.C.Strings.New_String (Message & Character'Val (0)));
      if Status /= CFE_SUCCESS then
         Put_Line ("[CFE-EVS] WARNING: Failed to send info event, status=" &
                   CFE_Status_t'Image (Status));
      end if;
   end CFE_Send_Info_Event;

   --  ──────────────────────────────────────────────────────────────────────
   --  CFE_Send_Error_Event: Send an error event
   --  ──────────────────────────────────────────────────────────────────────
   procedure CFE_Send_Error_Event (Message : String) is
      Status : CFE_Status_t;
   begin
      Status := CFE_EVS_SendEvent
        (EventId   => 16#0002#,
         EventType => CFE_EVS_EventType_ERROR,
         Spec      => Interfaces.C.Strings.New_String (Message & Character'Val (0)));
      if Status /= CFE_SUCCESS then
         Put_Line ("[CFE-EVS] WARNING: Failed to send error event, status=" &
                   CFE_Status_t'Image (Status));
      end if;
   end CFE_Send_Error_Event;

end CFE_FFI_Bindings;
