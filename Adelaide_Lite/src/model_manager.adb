with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar; use type Ada.Calendar.Time;
with Database_Manager;
with Tool_Manager;
with Scheduler_Manager;
with Llama_Interface;
use Llama_Interface;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Unchecked_Conversion;

package body Model_Manager is
   use Streaming_Queue;

   function Llama_Batch_Get_One
     (T : System.Address; N : int) return Llama_Batch;
   pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

   task type WCET_Printer;
   task body WCET_Printer is
   begin
      loop
         delay 30.0;
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Light_Red) &
                               "[WCET]" & AnsiAda.Reset &
                               " Current Pipeline WCET: " & Current_WCET'Img & " seconds");
      end loop;
   end WCET_Printer;

   Printer_Task : WCET_Printer;

   type Model_Record is record
      Model       : Llama_Model := Null_Model;
      Context     : Llama_Context := Null_Context;
      Path        : Unbounded_String;
      Loaded      : Boolean := False;
      In_Use      : Boolean := False;
      Last_Used   : Time := Time_First;
      Current_Ctx : unsigned := 0;
   end record;

   Models : array (Model_Type) of Model_Record;

   type Model_Type_Refs is array (Model_Type) of aliased Model_Type;
   Model_Refs : constant Model_Type_Refs :=
     [Qwen_0_8B      => Qwen_0_8B,
      Qwen_4B        => Qwen_4B,
      Qwen_Embedding => Qwen_Embedding,
      MMProj         => MMProj];

   type Owner_Array is array (Model_Type) of ELP_Level;
   type Busy_Array is array (Model_Type) of Boolean;

   --  PRIORITY MODEL GATE:
   --  Manages access to the model contexts.
   --  ELP1 requests preempt running ELP0 requests.
   protected Priority_Model_Gate is
      procedure Request_ELP1;
      entry Acquire_ELP1 (Model_Type);
      procedure Release_ELP1 (Kind : Model_Type);
      entry Acquire_ELP0 (Model_Type) (Success : out Boolean);
      procedure Release_ELP0 (Kind : Model_Type);
      function Should_Abort return Boolean;
      function Is_ELP0_Owner (Kind : Model_Type) return Boolean;
   private
      ELP1_Pending      : Natural := 0;
      ELP1_Active_Count : Natural := 0;
      Busy              : Busy_Array := [others => False];
      Owner             : Owner_Array := [others => ELP0];
   end Priority_Model_Gate;

   protected body Priority_Model_Gate is
      procedure Request_ELP1 is
      begin
         ELP1_Pending := ELP1_Pending + 1;
      end Request_ELP1;

      entry Acquire_ELP1 (for K in Model_Type) when not Busy (K) is
      begin
         ELP1_Pending := ELP1_Pending - 1;
         Busy (K) := True;
         Owner (K) := ELP1;
         ELP1_Active_Count := ELP1_Active_Count + 1;
      end Acquire_ELP1;

      procedure Release_ELP1 (Kind : Model_Type) is
      begin
         Busy (Kind) := False;
         Owner (Kind) := ELP0;
         if ELP1_Active_Count > 0 then
            ELP1_Active_Count := ELP1_Active_Count - 1;
         end if;
      end Release_ELP1;

      entry Acquire_ELP0 (for K in Model_Type) (Success : out Boolean)
         when not Busy (K)
           or else ELP1_Pending > 0
           or else ELP1_Active_Count > 0 is
      begin
         if ELP1_Pending > 0 or else ELP1_Active_Count > 0 then
            Success := False;
         else
            Busy (K) := True;
            Owner (K) := ELP0;
            Success := True;
         end if;
      end Acquire_ELP0;

      procedure Release_ELP0 (Kind : Model_Type) is
      begin
         Busy (Kind) := False;
      end Release_ELP0;

      function Should_Abort return Boolean is
      begin
         return ELP1_Pending > 0 or else ELP1_Active_Count > 0;
      end Should_Abort;

      function Is_ELP0_Owner (Kind : Model_Type) return Boolean is
      begin
         return Owner (Kind) = ELP0;
      end Is_ELP0_Owner;
   end Priority_Model_Gate;

   task Idle_Monitor is
      pragma Storage_Size (1024 * 1024);
      entry Start;
   end Idle_Monitor;

   task body Idle_Monitor is
      Next_Check : Time;
      Interval   : constant Time_Span := Seconds (1);
      Timeout    : constant Time_Span := Seconds (30);
      Now        : Time;
   begin
      accept Start;
      loop
         Next_Check := Clock + Interval;
         Now := Clock;
         for Kind in Model_Type loop
            if Models (Kind).Loaded and then
               not Models (Kind).In_Use and then
               (Now - Models (Kind).Last_Used) > Timeout
            then
               Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Idle]" &
                         AnsiAda.Reset & " Unloading " &
                         Model_Type'Image (Kind));
               Unload_Model (Kind);
            end if;
         end loop;
         delay until Next_Check;
      end loop;
   end Idle_Monitor;

   function Wrap_ChatML (Sys : String; Msg : String) return String is
   begin
      return "<|im_start|>system" & ASCII.LF & Sys & "<|im_end|>" & ASCII.LF &
             "<|im_start|>user" & ASCII.LF & Msg & "<|im_end|>" & ASCII.LF &
             "<|im_start|>assistant" & ASCII.LF;
   end Wrap_ChatML;

   procedure Initialize is
   begin
      Llama_Backend_Init;
      Database_Manager.Initialize;
      Models (Qwen_0_8B).Path := To_Unbounded_String
        ("../llama.cpp/models/qwen3.5/Qwen3.5-0.8B-Q4_K_S.gguf");
      Models (Qwen_4B).Path   := To_Unbounded_String
        ("../llama.cpp/models/qwen3.5/Qwen3.5-4B-Q4_K_S.gguf");
      Models (Qwen_Embedding).Path := To_Unbounded_String
        ("../llama.cpp/models/qwen3.5/Qwen3-Embedding-0.6B-Q8_0.gguf");
      Models (MMProj).Path := To_Unbounded_String
        ("../llama.cpp/models/qwen3.5/mmproj-0.8B-F16.gguf");
      Idle_Monitor.Start;
   end Initialize;

   procedure Load_Model
     (Kind          : Model_Type;
      Success       : out Boolean;
      Requested_Ctx : Positive := 4096)
   is
      M_Params   : Llama_Model_Params := Llama_Model_Default_Params;
      C_Params   : Llama_Context_Params := Llama_Context_Default_Params;
      Path_C     : chars_ptr := New_String (To_String (Models (Kind).Path));
      Actual_Ctx : unsigned;
   begin
      Actual_Ctx := (if Requested_Ctx <= 4096 then 4096
                     else (if Requested_Ctx <= 16384 then 16384 else 32768));
      Success := False;
      if Models (Kind).Loaded then
         if unsigned (Requested_Ctx) <= Models (Kind).Current_Ctx then
            Models (Kind).Last_Used := Clock;
            Success := True;
            return;
         end if;
         Unload_Model (Kind);
      end if;

      Put_Line ("[+] Loading " & Model_Type'Image (Kind) &
                " (N_CTX=" & Actual_Ctx'Img & ")");
      M_Params.N_Gpu_Layers := -1;
      Models (Kind).Model := Llama_Model_Load_From_File (Path_C, M_Params);
      Free (Path_C);

      if Models (Kind).Model /= Null_Model then
         C_Params.N_Ctx := Actual_Ctx;
         C_Params.N_Batch := 4096;
         C_Params.N_Ubatch := 1024;
         C_Params.N_Threads := 8;
         C_Params.N_Threads_Batch := 8;
         C_Params.Abort_Callback := Llama_Abort_Callback'Address;
         C_Params.Abort_Callback_Data := Model_Refs (Kind)'Address;
         Models (Kind).Context :=
           Llama_Init_From_Model (Models (Kind).Model, C_Params);
         if Models (Kind).Context /= Null_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Models (Kind).Current_Ctx := Actual_Ctx;
            Success := True;
         else
            Llama_Model_Free (Models (Kind).Model);
            Models (Kind).Model := Null_Model;
         end if;
      end if;
   end Load_Model;

   procedure Unload_Model (Kind : Model_Type) is
   begin
      if Models (Kind).Loaded then
         Llama_Free (Models (Kind).Context);
         Llama_Model_Free (Models (Kind).Model);
         Models (Kind).Context := Null_Context;
         Models (Kind).Model := Null_Model;
         Models (Kind).Loaded := False;
         Models (Kind).Current_Ctx := 0;
      end if;
   end Unload_Model;

   procedure Force_Unload_And_Reload (Kind : Model_Type) is
      Success : Boolean;
   begin
      Unload_Model (Kind);
      Load_Model (Kind, Success);
   end Force_Unload_And_Reload;

   function Get_Context
     (Kind : Model_Type) return Llama_Interface.Llama_Context is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Context;
   end Get_Context;

   function Get_Model
     (Kind : Model_Type) return Llama_Interface.Llama_Model is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Model;
   end Get_Model;

   function Llama_Abort_Callback (Data : System.Address) return Boolean is
      use System;
      type Model_Type_Ptr is access all Model_Type;
      function To_Ptr is new Ada.Unchecked_Conversion
        (System.Address, Model_Type_Ptr);
      Ptr : Model_Type_Ptr;
   begin
      if Data = System.Null_Address then
         return False;
      end if;
      Ptr := To_Ptr (Data);
      return Priority_Model_Gate.Is_ELP0_Owner (Ptr.all)
        and then Priority_Model_Gate.Should_Abort;
   end Llama_Abort_Callback;

   function Should_Abort_ELP0 return Boolean is
   begin
      return Priority_Model_Gate.Should_Abort;
   end Should_Abort_ELP0;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type is
   begin
      if Name = "adelaide-hybrid"
        or else Name = "qwen3.5:4b"
        or else Name = "metamodel"
        or else Name = "adelaide-metamodel"
      then
         return Qwen_4B;
      elsif Name = "qwen-embedding" or else Name = "adelaide-embedding" then
         return Qwen_Embedding;
      else
         return Qwen_0_8B;
      end if;
   end Get_Kind_For_Model_Name;

   function Is_Loaded (Kind : Model_Type) return Boolean is
   begin
      return Models (Kind).Loaded;
   end Is_Loaded;

   function Count_Tokens (Text : String) return Positive is
   begin
      return Text'Length / 4 + 1;
   end Count_Tokens;

   function Get_Request_Category
     (Msg        : String;
      Session_ID : String := "";
      Level      : ELP_Level := ELP1) return String
   is
      pragma Unreferenced (Session_ID, Level);
   begin
      if Index (Msg, "code") > 0 or else Index (Msg, "program") > 0 then
         return "Technical";
      else
         return "General";
      end if;
   end Get_Request_Category;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "";
      Level         : ELP_Level := ELP1) return Natural
   is
      pragma Unreferenced (Response_Text, Prompt, Session_ID, Level);
      Score : Natural := 5;
   begin
      if Search_Used then
         Score := Score + 2;
      end if;
      if Has_Citations then
         Score := Score + 3;
      end if;
      return Score;
   end Grade_Response_Quality;

   procedure Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Str_Piece  : String)
   is
      pragma Unreferenced (Session_ID);
   begin
      if Stream /= null then
         Stream.Push (Str_Piece);
      end if;
   end Push_Chunk;

   function Generator_Callback (Prompt : String) return String is
   begin
      return "Callback response to " & Prompt;
   end Generator_Callback;

   --  SINGLE EMBEDDING HELPER
   procedure Get_Single_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural)
   is
      Success  : Boolean;
      Kind     : constant Model_Type := Qwen_Embedding;
      Vocab    : Llama_Vocab;
      Tokens   : array (1 .. 32768) of Llama_Token;
      N_Toks   : int;
      Prompt_C : chars_ptr := New_String (Prompt);
   begin
      Priority_Model_Gate.Acquire_ELP1 (Kind);
      Load_Model (Kind, Success);
      if not Success then
         Priority_Model_Gate.Release_ELP1 (Kind);
         Length := 0;
         return;
      end if;

      Models (Kind).Last_Used := Clock;
      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize
        (Vocab, Prompt_C, int (Prompt'Length), Tokens (1)'Address,
         32768, True, True);
      Free (Prompt_C);
      if N_Toks <= 0 then
         Priority_Model_Gate.Release_ELP1 (Kind);
         Length := 0;
         return;
      end if;

      declare
         B : constant Llama_Batch :=
           Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         Llama_Interface.Llama_Memory_Clear
           (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);
         Llama_Set_Embeddings (Models (Kind).Context, True);
         if Llama_Decode (Models (Kind).Context, B) /= 0 then
            Priority_Model_Gate.Release_ELP1 (Kind);
            Length := 0;
            return;
         end if;
      end;

      declare
         function Llama_Model_N_Embd (M : Llama_Model) return int;
         pragma Import (C, Llama_Model_N_Embd, "llama_model_n_embd");
         Dim : constant int := Llama_Model_N_Embd (Models (Kind).Model);
         Ptr : constant System.Address :=
           Llama_Get_Embeddings (Models (Kind).Context);
         type Float_Array is array (1 .. Integer (Dim)) of Float;
         pragma Convention (C, Float_Array);
         Embed : Float_Array;
         for Embed'Address use Ptr;
      begin
         if Integer (Dim) <= Result'Length then
            for I in 1 .. Integer (Dim) loop
               Result (Result'First + I - 1) := Embed (I);
            end loop;
            Length := Integer (Dim);
         else
            Length := 0;
         end if;
         Priority_Model_Gate.Release_ELP1 (Kind);
      end;
   exception
      when others =>
         Priority_Model_Gate.Release_ELP1 (Kind);
         Length := 0;
   end Get_Single_Embedding;

   --  GET EMBEDDING (WITH CHUNKING > 800 CHARS)
   procedure Get_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural)
   is
   begin
      if Prompt'Length <= 800 then
         Get_Single_Embedding (Prompt, Result, Length);
      else
         declare
            Num_Chunks : Natural := 0;
            Sum_Vec    : Math_Utils.Vector (Result'Range) := [others => 0.0];
            Dim        : Natural := 0;
            Start_Idx  : Positive := Prompt'First;
            End_Idx    : Positive;
         begin
            while Start_Idx <= Prompt'Last loop
               End_Idx := Start_Idx + 800 - 1;
               if End_Idx > Prompt'Last then
                  End_Idx := Prompt'Last;
               end if;
               declare
                  Sub_Prompt : constant String :=
                    Prompt (Start_Idx .. End_Idx);
                  Sub_Vec    : Math_Utils.Vector (Result'Range) :=
                    [others => 0.0];
                  Sub_Len    : Natural := 0;
               begin
                  Get_Single_Embedding (Sub_Prompt, Sub_Vec, Sub_Len);
                  if Sub_Len > 0 then
                     if Num_Chunks = 0 then
                        Dim := Sub_Len;
                     end if;
                     for I in 1 .. Dim loop
                        Sum_Vec (Result'First + I - 1) :=
                          Sum_Vec (Result'First + I - 1) +
                          Sub_Vec (Sub_Vec'First + I - 1);
                     end loop;
                     Num_Chunks := Num_Chunks + 1;
                  end if;
               end;
               Start_Idx := End_Idx + 1;
            end loop;

            if Num_Chunks > 0 and then Dim > 0 then
               for I in 1 .. Dim loop
                  Result (Result'First + I - 1) :=
                    Sum_Vec (Result'First + I - 1) / Float (Num_Chunks);
               end loop;
               Length := Dim;
            else
               Length := 0;
            end if;
         end;
      end if;
   end Get_Embedding;

   --  STREAM PARSER HELPERS
   type Think_State_Type is (State_Think, State_Answer);

   type Stream_Parser_State is record
      Orch_Think_Open : Boolean := False;
      Header_Closed   : Boolean := False;
      Think_State     : Think_State_Type := State_Think;
      Buffer          : Unbounded_String := Null_Unbounded_String;
      Closing_Buffer  : Unbounded_String := Null_Unbounded_String;
      Sanitize_Buffer : Unbounded_String := Null_Unbounded_String;
   end record;

   procedure Process_And_Push_Char
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Parser     : in out Stream_Parser_State;
      C          : Character)
   is
   begin
      if not Parser.Header_Closed then
         Append (Parser.Buffer, C);
         declare
            Buf_Str : constant String := To_String (Parser.Buffer);
         begin
            if Buf_Str = "<think>" then
               Parser.Header_Closed := True;
               Parser.Think_State := State_Think;
               Parser.Buffer := Null_Unbounded_String;
            elsif Buf_Str'Length >= 7 then
               Parser.Header_Closed := True;
               if Parser.Orch_Think_Open then
                  Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
                  Parser.Orch_Think_Open := False;
               end if;
               Push_Chunk (Stream, Session_ID, Buf_Str);
               Parser.Buffer := Null_Unbounded_String;
               Parser.Think_State := State_Answer;
            end if;
         end;
         return;
      end if;

      if Parser.Think_State = State_Think then
         Append (Parser.Closing_Buffer, C);
         declare
            Cls_Str : constant String := To_String (Parser.Closing_Buffer);
         begin
            if Cls_Str = "</think>" then
               if Parser.Orch_Think_Open then
                  Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
                  Parser.Orch_Think_Open := False;
               else
                  Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
               end if;
               Parser.Think_State := State_Answer;
               Parser.Closing_Buffer := Null_Unbounded_String;
            elsif Cls_Str'Length >= 8 then
               Push_Chunk
                 (Stream, Session_ID,
                  Cls_Str (Cls_Str'First .. Cls_Str'First));
               Parser.Closing_Buffer :=
                 To_Unbounded_String
                   (Cls_Str (Cls_Str'First + 1 .. Cls_Str'Last));
            end if;
         end;
         return;
      end if;

      Append (Parser.Sanitize_Buffer, C);
      declare
         S_Str : constant String := To_String (Parser.Sanitize_Buffer);
      begin
         if S_Str = "<think>" or else S_Str = "</think>" then
            Parser.Sanitize_Buffer := Null_Unbounded_String;
         elsif S_Str'Length >= 8 then
            Push_Chunk
              (Stream, Session_ID, S_Str (S_Str'First .. S_Str'First));
            Parser.Sanitize_Buffer :=
              To_Unbounded_String (S_Str (S_Str'First + 1 .. S_Str'Last));
         end if;
      end;
   end Process_And_Push_Char;

   procedure Process_And_Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Parser     : in out Stream_Parser_State;
      Chunk      : String)
   is
   begin
      for I in Chunk'Range loop
         Process_And_Push_Char (Stream, Session_ID, Parser, Chunk (I));
      end loop;
   end Process_And_Push_Chunk;

   procedure Flush_Parser
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Parser     : in out Stream_Parser_State)
   is
   begin
      if not Parser.Header_Closed then
         declare
            Buf_Str : constant String := To_String (Parser.Buffer);
         begin
            if Parser.Orch_Think_Open then
               Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
            end if;
            Push_Chunk (Stream, Session_ID, Buf_Str);
         end;
      elsif Parser.Think_State = State_Think then
         declare
            Cls_Str : constant String := To_String (Parser.Closing_Buffer);
         begin
            if Parser.Orch_Think_Open then
               Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
            else
               Push_Chunk (Stream, Session_ID, Cls_Str);
            end if;
         end;
      else
         declare
            S_Str : constant String := To_String (Parser.Sanitize_Buffer);
         begin
            Push_Chunk (Stream, Session_ID, S_Str);
         end;
      end if;
   end Flush_Parser;

   function Sanitize_Think_Tags (Text : String) return String is
      Res : Unbounded_String;
      I   : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>" then
            I := I + 7;
         elsif I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>" then
            I := I + 8;
         else
            Append (Res, Text (I));
            I := I + 1;
         end if;
      end loop;
      return To_String (Res);
   end Sanitize_Think_Tags;

   --  GENERATE (CORE GGUF INFERENCE WITH PREEMPTION SUPPORT)
   procedure Generate
     (Kind            : Model_Type;
      Prompt          : String;
      Result          : out Unbounded_String;
      Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID      : String := "";
      Requested_Ctx   : Positive := 4096;
      Stream          : Streaming_Queue.Queue_Access := null;
      Orch_Think_Open : Boolean := False;
      Level           : ELP_Level := ELP1)
   is
      Success  : Boolean;
      Vocab    : Llama_Vocab;
      Tokens   : array (1 .. 32768) of Llama_Token;
      N_Toks   : int;
      Sampler  : Llama_Sampler;
      S_Params : Llama_Sampler_Chain_Params;
      Prompt_C : chars_ptr := New_String (Prompt);
      Parser   : Stream_Parser_State;
   begin
      pragma Unreferenced (Images);
      Result := Null_Unbounded_String;

      if Level = ELP0 then
         declare
            Acq_OK : Boolean;
         begin
            Priority_Model_Gate.Acquire_ELP0 (Kind) (Acq_OK);
            if not Acq_OK then
               Result := To_Unbounded_String ("ERROR: Preempted");
               return;
            end if;
         end;
      else
         Priority_Model_Gate.Request_ELP1;
         Priority_Model_Gate.Acquire_ELP1 (Kind);
      end if;

      Load_Model (Kind, Success, Requested_Ctx);
      if not Success then
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         Result := To_Unbounded_String ("ERROR: Load failed");
         return;
      end if;

      Llama_Interface.Llama_Memory_Clear
        (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);

      Models (Kind).In_Use := True;
      Models (Kind).Last_Used := Clock;
      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize
        (Vocab, Prompt_C, int (Prompt'Length), Tokens (1)'Address,
         32768, True, True);
      Free (Prompt_C);
      if N_Toks < 0 then
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         Result := To_Unbounded_String ("ERROR: Tokenization failed");
         return;
      end if;

      --  CHUNKED DECODING
      declare
         Batch_Size  : constant int := 4096;
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
      begin
         Llama_Interface.Llama_Memory_Clear
           (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);

         while Tokens_Left > 0 loop

            if Level = ELP0 and then Should_Abort_ELP0 then
               Models (Kind).In_Use := False;
               Priority_Model_Gate.Release_ELP0 (Kind);
               Result := To_Unbounded_String ("");
               return;
            end if;

            declare
               To_Decode : constant int :=
                 (if Tokens_Left > Batch_Size
                  then Batch_Size
                  else Tokens_Left);
               B : constant Llama_Batch :=
                 Llama_Batch_Get_One
                   (Tokens (Integer (Current_Pos) + 1)'Address, To_Decode);
            begin
               if Llama_Decode (Models (Kind).Context, B) /= 0 then
                  Models (Kind).In_Use := False;
                  if Level = ELP0 then
                     Priority_Model_Gate.Release_ELP0 (Kind);
                  else
                     Priority_Model_Gate.Release_ELP1 (Kind);
                  end if;
                  Result := To_Unbounded_String ("ERROR: Decode failed");
                  return;
               end if;
               Tokens_Left := Tokens_Left - To_Decode;
               Current_Pos := Current_Pos + To_Decode;
            end;
         end loop;
      end;

      S_Params := Llama_Sampler_Chain_Default_Params;
      Sampler := Llama_Sampler_Chain_Init (S_Params);
      Llama_Sampler_Chain_Add
        (Sampler, Llama_Sampler_Init_Penalties (64, 1.1, 0.1, 0.1));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_K (40));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_P (0.9, 1));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Temp (0.7));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Dist (1234));

      Parser.Orch_Think_Open := Orch_Think_Open;

      for I in 1 .. 2048 loop
         if Level = ELP0 and then Should_Abort_ELP0 then
            exit;
         end if;

         declare
            Token : constant Llama_Token :=
              Llama_Sampler_Sample (Sampler, Models (Kind).Context, -1);
            Piece : array (1 .. 256) of aliased Character;
            Len   : int;
         begin
            if Llama_Vocab_Is_Eog (Vocab, Token) then
               exit;
            end if;
            Len := Llama_Token_To_Piece
              (Vocab, Token, Piece (1)'Address, 256, 0, True);
            if Len > 0 then
               declare
                  Str_Piece : String (1 .. Integer (Len));
               begin
                  for J in 1 .. Integer (Len) loop
                     Str_Piece (J) := Piece (J);
                     Append (Result, Piece (J));
                  end loop;

                  if Stream /= null then
                     Process_And_Push_Chunk
                       (Stream, Session_ID, Parser, Str_Piece);
                  end if;
               end;
            end if;

            declare
               B : constant Llama_Batch :=
                 Llama_Batch_Get_One (Token'Address, 1);
            begin
               if Llama_Decode (Models (Kind).Context, B) /= 0 then
                  exit;
               end if;
            end;
         end;
      end loop;

      if Stream /= null then
         Flush_Parser (Stream, Session_ID, Parser);
      end if;

      Llama_Sampler_Free (Sampler);
      Models (Kind).In_Use := False;

      if Level = ELP0 then
         Priority_Model_Gate.Release_ELP0 (Kind);
      else
         Priority_Model_Gate.Release_ELP1 (Kind);
      end if;
   exception
      when others =>
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         Result := To_Unbounded_String ("ERROR: Decode failed");
   end Generate;

   --  HYBRID_GENERATE (MULTI-HOP REASONING PIPELINE)
   procedure Hybrid_Generate
     (Prompt     : String;
      Result     : out Unbounded_String;
      Images     : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
      Session_ID : String := "";
      Stream     : Streaming_Queue.Queue_Access := null;
      Level      : ELP_Level := ELP1;
      Agentic    : Boolean := False;
      Raw_Prompt : Boolean := False)
   is
      Whimsical_Adelaide : constant String :=
        "You are Adelaide Zephyrine Charlotte, a senior engineer. " &
        "Provide brilliant responses based on verified information.";
      Internal_State : Unbounded_String := Null_Unbounded_String;
      Current_Response : Unbounded_String;
      Current_Hop : Positive := 1;
      T0, T1      : Ada.Calendar.Time;
      Emb_Vec     : Math_Utils.Vector (1 .. 1536) := [others => 0.0];
      Emb_Len     : Natural;
   begin
      T0 := Ada.Calendar.Clock;

      if Stream /= null then
         Stream.Push
           ("<think>" & ASCII.LF & "[Adelaide Core Orchestration]" & ASCII.LF);
      end if;

      Get_Embedding (Prompt, Emb_Vec, Emb_Len);

      declare
         Cached_Res : constant String :=
           Database_Manager.Get_Cached_Response
             (Emb_Vec (1 .. Emb_Len), Current_WCET);
      begin
         if Cached_Res /= "" then
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                      "[Hybrid]" & AnsiAda.Reset &
                      " Cache HIT. Returning cached response.");
            Result := To_Unbounded_String (Cached_Res);
            if Stream /= null then
               Stream.Push (Cached_Res);
            end if;
            return;
         end if;
      end;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                "[Hybrid]" & AnsiAda.Reset &
                " Starting reasoning chain...");

      --  1. Factual checking
      if not Agentic
        and then
        (Index (Prompt, "What is") > 0
         or else Index (Prompt, "Who is") > 0
         or else Index (Prompt, "tell me about") > 0)
      then
         declare
            Start_Tag : constant String := "<|im_start|>user";
            End_Tag   : constant String := "<|im_end|>";
            S_Idx     : Natural := Index (Prompt, Start_Tag, Ada.Strings.Backward);
            E_Idx     : Natural;
            Raw_Q     : Unbounded_String;
            Gen_Q     : Unbounded_String;
         begin
            if S_Idx > 0 then
               S_Idx := S_Idx + Start_Tag'Length;
               E_Idx := Index (Prompt (S_Idx .. Prompt'Last), End_Tag);
               if E_Idx > 0 then
                  Raw_Q := To_Unbounded_String
                    (Trim (Prompt (S_Idx .. E_Idx - 1), Ada.Strings.Both));
               else
                  Raw_Q := To_Unbounded_String
                    (Trim (Prompt (S_Idx .. Prompt'Last), Ada.Strings.Both));
               end if;
            else
               Raw_Q := To_Unbounded_String (Trim (Prompt, Ada.Strings.Both));
            end if;

            declare
               Actual_Prompt : constant String :=
                 "Generate ONLY a concise 2-4 keyword search query for the following request: """ &
                 To_String (Raw_Q) & """. NO EXPLANATIONS. NO QUOTES. JUST KEYWORDS.";
            begin
               Model_Manager.Generate
                 (Kind            => Model_Manager.Qwen_0_8B,
                  Prompt          => Actual_Prompt,
                  Result          => Gen_Q,
                  Level           => Level);
            end;

            declare
               Final_Q : constant String :=
                 (if Length (Gen_Q) > 0 and then To_String (Gen_Q) /= "ERROR: Preempted"
                  then To_String (Gen_Q) else To_String (Raw_Q));
               R : constant Tool_Manager.Tool_Result :=
                 Tool_Manager.Execute_Tool ("searchglobalref", Final_Q);
            begin
               Append
                 (Internal_State,
                  "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
               if Stream /= null then
                  Stream.Push ("[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
               end if;
            end;
         end;
      end if;

      loop
         if Level = ELP0 and then Should_Abort_ELP0 then
            Result := To_Unbounded_String ("");
            return;
         end if;

         declare
            Router_Sys : constant String :=
              "You are the Router. You decide if a tool is needed. " &
              "If the user says hello or greets you, output [FINISH]. " &
              "If you need to search, use [ACTION: search(query)]. " &
              "If you need to read a file, use [ACTION: cat(filename)]. " &
              "If you need to calculate math, use [ACTION: math(expression)]. " &
              "If you need to execute code, use [ACTION: code(python_script)]. " &
              "If you want to schedule a proactive thought for later, use [ACTION: schedule(seconds, query)]. " &
              "If you are done, output [FINISH]. " &
              "Output ONLY the tag.";
            Paging_Instr : constant String :=
              "Current Data: " & To_String (Internal_State);
            Step_Raw     : Unbounded_String;

            function Get_Router_Prompt return String is
            begin
               if Raw_Prompt then
                  declare
                     Sub_Str : constant String :=
                       "<|im_start|>assistant" & ASCII.LF;
                     Idx     : constant Natural :=
                       Index (Prompt, Sub_Str, Going => Ada.Strings.Backward);
                  begin
                     if Idx > 0 then
                        return Prompt (Prompt'First .. Idx - 1) &
                               "System Override: " & Router_Sys & ASCII.LF &
                               Paging_Instr & ASCII.LF & Sub_Str;
                     else
                        return Prompt & ASCII.LF & "System Override: " &
                               Router_Sys & ASCII.LF & Paging_Instr &
                               ASCII.LF & Sub_Str;
                     end if;
                  end;
               else
                  return Wrap_ChatML (Router_Sys, Paging_Instr & ASCII.LF & Prompt);
               end if;
            end Get_Router_Prompt;
         begin
            declare
               Ctx : Positive := 2048;
            begin
               loop
                  Generate
                    (Qwen_0_8B,
                     Get_Router_Prompt,
                     Step_Raw, GNATCOLL.JSON.Empty_Array, Session_ID, Ctx,
                     Stream, False, Level);
                  exit when To_String (Step_Raw) /= "ERROR: Decode failed"
                    or else Ctx >= 16384;
                  Ctx := Ctx * 2;
               end loop;
            end;

            declare
               Step : constant String :=
                 Trim (To_String (Step_Raw), Ada.Strings.Both);
            begin
               Put_Line (" [Hybrid] Hop" & Current_Hop'Img & ": " & Step);

               if Index (Step, "[ACTION:") > 0 then
                  declare
                     S_Pos : constant Natural := Index (Step, "[ACTION:") + 8;
                     E_Pos : constant Natural := Index (Step, "]", S_Pos);
                  begin
                     if E_Pos > S_Pos then
                        declare
                           A_Full : constant String :=
                             Step (S_Pos .. E_Pos - 1);
                           P_Pos  : constant Natural :=
                             Index (A_Full, "(");
                           EP_Pos : constant Natural :=
                             (if P_Pos > 0 then Index (A_Full, ")", P_Pos) else 0);
                        begin
                           if P_Pos > 0 and then EP_Pos > P_Pos then
                              declare
                                 T_Name : constant String :=
                                   Trim
                                     (A_Full (A_Full'First .. P_Pos - 1),
                                      Ada.Strings.Both);
                                 T_Pars : constant String :=
                                   Trim
                                     (A_Full (P_Pos + 1 .. EP_Pos - 1),
                                      Ada.Strings.Both);
                              begin
                                 if T_Name = "schedule" then
                                    declare
                                       Comma_Idx : constant Natural := Index (T_Pars, ",");
                                    begin
                                       if Comma_Idx > 0 then
                                          declare
                                             Time_Str : constant String := Trim (T_Pars (T_Pars'First .. Comma_Idx - 1), Ada.Strings.Both);
                                             Prompt_Str : constant String := Trim (T_Pars (Comma_Idx + 1 .. T_Pars'Last), Ada.Strings.Both);
                                             Delay_Secs : Integer;
                                          begin
                                             Delay_Secs := Integer'Value (Time_Str);
                                             Scheduler_Manager.Schedule (Delay_Secs, Prompt_Str);
                                             Append (Internal_State, "[SCHEDULED]: " & Prompt_Str & ASCII.LF);
                                          exception
                                             when others => null;
                                          end;
                                       end if;
                                    end;
                                 elsif T_Pars'Length < 256 and then
                                    Index
                                      (To_String (Internal_State),
                                       T_Name & "(" & T_Pars & ")") = 0
                                 then
                                    if Agentic then
                                       Result := To_Unbounded_String
                                         ("[TOOL_CALL: " & T_Name &
                                          "(" & T_Pars & ")]");
                                       return;
                                    end if;
                                    declare
                                       R : constant Tool_Manager.Tool_Result :=
                                         Tool_Manager.Execute_Tool
                                           (T_Name, T_Pars);
                                    begin
                                       Append
                                         (Internal_State,
                                          "[TOOL (" & T_Name & ")]: " &
                                          To_String (R.Output) & ASCII.LF);
                                       if Stream /= null then
                                          Stream.Push
                                            (ASCII.LF & "[TOOL (" & T_Name & ")]: " &
                                             To_String (R.Output) & ASCII.LF);
                                       end if;
                                    end;
                                 else
                                    exit;
                                 end if;
                              end;
                           end if;
                        end;
                     end if;
                  end;
               elsif Index (Step, "[FINISH]") > 0 then
                  exit;
               else
                  exit;
               end if;
            end;
         end;
         Current_Hop := Current_Hop + 1;
         exit when Current_Hop > 5;
      end loop;

      declare
         function Get_Final_Prompt return String is
         begin
            if Raw_Prompt then
               declare
                  Sub_Str : constant String :=
                    "<|im_start|>assistant" & ASCII.LF;
                  Idx     : constant Natural :=
                    Index (Prompt, Sub_Str, Going => Ada.Strings.Backward);
               begin
                  if Idx > 0 then
                     return Prompt (Prompt'First .. Idx - 1) &
                            "Fact-Check: " & To_String (Internal_State) &
                            ASCII.LF & Sub_Str;
                  else
                     return Prompt & ASCII.LF & "Fact-Check: " &
                            To_String (Internal_State) & ASCII.LF & Sub_Str;
                  end if;
               end;
            else
               return Wrap_ChatML
                 (Whimsical_Adelaide,
                  "User: " & Prompt & ASCII.LF &
                  "Fact-Check: " & To_String (Internal_State));
            end if;
         end Get_Final_Prompt;

         Synth_Prompt : constant String := Get_Final_Prompt;
      begin
         declare
            Ctx : Positive := 4096;
         begin
            loop
               Generate
                 (Qwen_4B, Synth_Prompt, Current_Response, Images, Session_ID,
                  Ctx, Stream, True, Level);
               exit when To_String (Current_Response) /= "ERROR: Decode failed"
                 or else Ctx >= 32768;
               Ctx := Ctx * 2;
            end loop;
         end;

         Result := Current_Response;
         declare
            B64_Str : Unbounded_String := To_Unbounded_String ("");
         begin
            if GNATCOLL.JSON.Length (Images) > 0 then
               B64_Str := To_Unbounded_String
                 (String'(GNATCOLL.JSON.Get
                   (GNATCOLL.JSON.Get (Images, 1))));
            end if;
            Database_Manager.Remember
              (Prompt, To_String (Current_Response), To_String (B64_Str));
         end;
      end;

      Database_Manager.Add_To_Cache (Prompt, Emb_Vec (1 .. Emb_Len), To_String (Current_Response));

      T1 := Ada.Calendar.Clock;
      declare
         Dur : constant Duration := T1 - T0;
      begin
         if Dur > Current_WCET then
            Current_WCET := Dur;
         end if;
         if Level = ELP0 then
            if Dur > Current_WCET_ELP0 then
               Current_WCET_ELP0 := Dur;
            end if;
         else
            if Dur > Current_WCET_ELP1 then
               Current_WCET_ELP1 := Dur;
            end if;
         end if;
      end;

      if Stream = null then
         --  Format non-streaming response with merged think block
         declare
            Tag_Idx : constant Natural :=
              Index (To_String (Current_Response), "</think>");
            Merged  : Unbounded_String;
         begin
            if Tag_Idx > 0 then
               declare
                  Resp_Str : constant String := To_String (Current_Response);
                  Part1    : constant String :=
                    Resp_Str (Resp_Str'First .. Tag_Idx - 1);
                  Part2    : constant String :=
                    Resp_Str (Tag_Idx + 8 .. Resp_Str'Last);
               begin
                  Merged :=
                    To_Unbounded_String
                      ("<think>" & ASCII.LF &
                       "[Adelaide Core Orchestration]" & ASCII.LF &
                       To_String (Internal_State) & ASCII.LF &
                       Sanitize_Think_Tags (Part1) & "</think>" &
                       ASCII.LF & Sanitize_Think_Tags (Part2));
               end;
            else
               Merged :=
                 To_Unbounded_String
                   ("<think>" & ASCII.LF &
                    "[Adelaide Core Orchestration]" & ASCII.LF &
                    To_String (Internal_State) & "</think>" &
                    ASCII.LF &
                    Sanitize_Think_Tags (To_String (Current_Response)));
            end if;
            Result := Merged;
         end;
      else
         Result := Current_Response;
      end if;
   end Hybrid_Generate;

end Model_Manager;
