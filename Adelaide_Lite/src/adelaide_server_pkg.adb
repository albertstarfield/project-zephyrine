with Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Exceptions;
with AWS.Client;
with AWS.Headers;
with AWS.Messages;
with AWS.Response.Set;
with AWS.Resources.Streams;
with GNATCOLL.JSON;
with Math_Utils;
with Model_Manager; use Model_Manager;
with Database_Manager;
with Ada.Strings.Fixed;
with Ada.Calendar; use Ada.Calendar;
with Streaming_Queue;

package body Adelaide_Server_Pkg is

   OLLAMA_PORT : constant String := "11435";
   OLLAMA_URL  : constant String := "http://localhost:" & OLLAMA_PORT;

   --  Minimal Stream Registry for compilation
   protected Stream_Registry is
      procedure Register (ID : String; Q : Streaming_Queue.Queue_Access);
      procedure Unregister (ID : String);
      procedure Push_Log (ID : String; Log : String);
   end Stream_Registry;

   protected body Stream_Registry is
      procedure Register (ID : String; Q : Streaming_Queue.Queue_Access) is null;
      procedure Unregister (ID : String) is null;
      procedure Push_Log (ID : String; Log : String) is null;
   end Stream_Registry;

   --  Minimal Generator Task for compilation
   task type Generator_Task is
      entry Start
        (Stream_Ptr     : Streaming_Queue.Queue_Access;
         Prompt_Val     : String;
         Session_ID_Val : String;
         URI_Str_Val    : String;
         Start_Time_Val : Ada.Calendar.Time;
         Level_Val      : Model_Manager.ELP_Level := Model_Manager.ELP1);
   end Generator_Task;

   type Generator_Task_Access is access Generator_Task;

   task body Generator_Task is
   begin
      accept Start
        (Stream_Ptr     : Streaming_Queue.Queue_Access;
         Prompt_Val     : String;
         Session_ID_Val : String;
         URI_Str_Val    : String;
         Start_Time_Val : Ada.Calendar.Time;
         Level_Val      : Model_Manager.ELP_Level := Model_Manager.ELP1) do
         null;
      end Start;
   end Generator_Task;

   --  Minimal Dispatch for heartbeat test
   function Dispatch (Request : AWS.Status.Data) return AWS.Response.Data is
      use AWS.Status;
   begin
      Ada.Text_IO.Put_Line (Ada.Text_IO.Standard_Error, "[Request] " & URI (Request));
      Ada.Text_IO.Flush (Ada.Text_IO.Standard_Error);
      
      return AWS.Response.Build (Content_Type => "text/plain",
                                 Message_Body => "OK");
   end Dispatch;

end Adelaide_Server_Pkg;
