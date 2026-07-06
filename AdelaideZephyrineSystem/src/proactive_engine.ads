pragma SPARK_Mode (Off);
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Calendar;          use Ada.Calendar;
with Model_Types;           use Model_Types;

--  ============================================================================
--  PROACTIVE ENGINE
--  ============================================================================
--  "Zephy Intelligence & Curiosity" — the assistant's inner voice.
--
--  Purpose:
--    Enables the assistant to initiate conversations, not just respond.
--    This is the core of STS (System Two) Handless Mode: the assistant
--    can ask questions, share observations, and engage the user proactively.
--
--  Key Concepts:
--    - Handless Mode: When active, the assistant can speak first without
--      user prompting.  On first activation, Adelaide greets:
--      "Hello There! I'm Adelaide, nice to meet you!"
--    - Acoustic Dynamic Trigger: If ambient sound changes (e.g., someone
--      enters the room), the assistant may ask a question.
--    - Calendar Schedule: The assistant can be scheduled to ask questions
--      at specific times (e.g., morning check-in, evening summary).
--    - Curiosity Engine: Generates questions based on accumulated knowledge,
--      recent conversations, and the user's interests.
--
--  Why This Exists:
--    -打破了 user <-> assistant 的单向交互模式
--    - 让助手能主动发起对话，像真人一样
--    - STS (System Two) 需要 handless mode 来实现自主行为
--  ============================================================================
package Proactive_Engine is

   --  Handless mode state
   type Handless_Mode_State is (Off, Activating, Active);

   --  Initialize the proactive engine
   procedure Initialize;

   --  Activate handless mode (assistant can now initiate)
   --  On first activation, Adelaide greets the user.
   procedure Activate_Handless_Mode;

   --  Deactivate handless mode
   procedure Deactivate_Handless_Mode;

   --  Check if handless mode is active
   function Is_Handless_Mode_Active return Boolean;

   --  Trigger a proactive question based on acoustic dynamics
   --  Called when ambient sound changes significantly
   procedure Trigger_Acoustic_Question;

   --  Schedule a proactive question at a specific time
   procedure Schedule_Question (At_Time : Time; Topic : String);

   --  Schedule a repeating proactive question (e.g., every hour)
   procedure Schedule_Repeating_Question (Interval : Duration; Topic : String);

   --  Tick: check and fire any pending proactive questions
   --  Called from the main loop or a dedicated task
   procedure Tick;

   --  Get the last proactive question asked (for logging)
   function Get_Last_Question return String;

   --  Get the last proactive answer given (for logging)
   function Get_Last_Answer return String;

   --  Audio queue for Handless STS proactive injection
   procedure Queue_Audio (PCM : String);
   function Has_Pending_Audio return Boolean;
   function Pop_Pending_Audio return String;

end Proactive_Engine;
