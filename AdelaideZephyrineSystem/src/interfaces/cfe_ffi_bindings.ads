with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;

--  Native C bindings for NASA cFE Software Bus
--  Wraps the cFE core API for Ada access (pipe management, subscribe, send/receive)
--  Designed for flight software integration with Adelaide Zephyrine System
package CFE_FFI_Bindings is

   --  ──────────────────────────────────────────────────────────────────────
   --  cFE Software Bus Types (opaque handles)
   --  ──────────────────────────────────────────────────────────────────────
   type CFE_SB_PipeId_t is new Interfaces.C.unsigned;
   type CFE_SB_MsgId_t is new Interfaces.C.unsigned;
   type CFE_Status_t is new Interfaces.C.int;

   --  Constants
   CFE_SUCCESS       : constant CFE_Status_t := 0;
   CFE_SB_POLL       : constant Interfaces.C.int := 0;
   CFE_SB_PEND_FOREVER : constant Interfaces.C.int := -1;

   --  ──────────────────────────────────────────────────────────────────────
   --  Pipe Management
   --  ──────────────────────────────────────────────────────────────────────

   --  Create a new software bus pipe
   --  Returns CFE_SUCCESS on success
   function CFE_SB_CreatePipe
     (PipeIdPtr : access CFE_SB_PipeId_t;
      Depth     : Interfaces.C.unsigned_short;
      PipeName  : Interfaces.C.Strings.chars_ptr)
      return CFE_Status_t;
   pragma Import (C, CFE_SB_CreatePipe, "CFE_SB_CreatePipe");

   --  Delete a software bus pipe
   function CFE_SB_DeletePipe
     (PipeId : CFE_SB_PipeId_t)
      return CFE_Status_t;
   pragma Import (C, CFE_SB_DeletePipe, "CFE_SB_DeletePipe");

   --  ──────────────────────────────────────────────────────────────────────
   --  Subscription Management
   --  ──────────────────────────────────────────────────────────────────────

   --  Subscribe to a message (default QoS)
   function CFE_SB_Subscribe
     (MsgId  : CFE_SB_MsgId_t;
      PipeId : CFE_SB_PipeId_t)
      return CFE_Status_t;
   pragma Import (C, CFE_SB_Subscribe, "CFE_SB_Subscribe");

   --  Unsubscribe from a message
   function CFE_SB_Unsubscribe
     (MsgId  : CFE_SB_MsgId_t;
      PipeId : CFE_SB_PipeId_t)
      return CFE_Status_t;
   pragma Import (C, CFE_SB_Unsubscribe, "CFE_SB_Unsubscribe");

   --  ──────────────────────────────────────────────────────────────────────
   --  Message Send/Receive
   --  ──────────────────────────────────────────────────────────────────────

   --  Transmit a message (IsOrigination = true for new messages)
   function CFE_SB_TransmitMsg
     (MsgPtr        : System.Address;
      IsOrigination : Interfaces.C.bool)
      return CFE_Status_t;
   pragma Import (C, CFE_SB_TransmitMsg, "CFE_SB_TransmitMsg");

   --  Set user data length in a message
   procedure CFE_SB_SetUserDataLength
     (MsgPtr     : System.Address;
      DataLength : Interfaces.C.size_t);
   pragma Import (C, CFE_SB_SetUserDataLength, "CFE_SB_SetUserDataLength");

   --  Timestamp a message with current spacecraft time
   procedure CFE_SB_TimeStampMsg (MsgPtr : System.Address);
   pragma Import (C, CFE_SB_TimeStampMsg, "CFE_SB_TimeStampMsg");

   --  Get pointer to user data in a message
   function CFE_SB_GetUserData (MsgPtr : System.Address) return System.Address;
   pragma Import (C, CFE_SB_GetUserData, "CFE_SB_GetUserData");

   --  Get length of user data in a message
   function CFE_SB_GetUserDataLength (MsgPtr : System.Address) return Interfaces.C.size_t;
   pragma Import (C, CFE_SB_GetUserDataLength, "CFE_SB_GetUserDataLength");

   --  ──────────────────────────────────────────────────────────────────────
   --  Event Service (EVS) — Send events to ground/telmetry
   --  ──────────────────────────────────────────────────────────────────────

   --  Event type constants
   CFE_EVS_EventType_INFORMATIONAL : constant Interfaces.C.unsigned := 0;
   CFE_EVS_EventType_ERROR         : constant Interfaces.C.unsigned := 1;
   CFE_EVS_EventType_DEBUG         : constant Interfaces.C.unsigned := 2;

   --  Send an event (printf-style formatted string)
   function CFE_EVS_SendEvent
     (EventId    : Interfaces.C.unsigned_short;
      EventType  : Interfaces.C.unsigned;
      Spec       : Interfaces.C.Strings.chars_ptr)
      return CFE_Status_t;
   pragma Import (C, CFE_EVS_SendEvent, "CFE_EVS_SendEvent");

   --  ──────────────────────────────────────────────────────────────────────
   --  Ada Wrapper Procedures
   --  ──────────────────────────────────────────────────────────────────────

   --  Initialize the cFE Software Bus interface
   --  Creates a pipe and subscribes to standard telemetry
   procedure CFE_Initialize;

   --  Send a telemetry message through the Software Bus
   procedure CFE_Send_Telemetry (Payload : String);

   --  Send an informational event
   procedure CFE_Send_Info_Event (Message : String);

   --  Send an error event
   procedure CFE_Send_Error_Event (Message : String);

end CFE_FFI_Bindings;
