with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Database_Manager;
with Tool_Manager;
with Llama_Interface; use Llama_Interface;
with Watchdog_Manager;
with System;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Real_Time; use Ada.Real_Time;
with Streaming_Queue; use type Streaming_Queue.Queue_Access;
with GNAT.OS_Lib;
with Ada.Directories;
with Ada.Numerics.Discrete_Random;
with Ada.Characters.Handling;
with Verification_Manager;
with GNATCOLL.JSON;

package body Model_Manager is

   type Model_Record is record
      Model      : Llama_Model := Null_Model;
      Context    : Llama_Context := Null_Context;
      Path       : Unbounded_String;
      Loaded     : Boolean := False;
      In_Use     : Boolean := False;
      Last_Used  : Time := Time_First;
      Current_Ctx : unsigned := 0;
   end record;

   Models : array (Model_Type) of Model_Record;

   type Busy_Array is array (Model_Type) of Boolean;

   --  QUEUE MANAGER: Serialize access to models
   --  to prevent concurrent decode crashes.
   protected Model_Gate is
      entry Acquire (Model_Type);
      procedure Release (Kind : Model_Type);
   private
      Busy : Busy_Array := (others => False);
   end Model_Gate;

   protected body Model_Gate is
      entry Acquire (for K in Model_Type) when not Busy (K) is
      begin
         Busy (K) := True;
      end Acquire;

      procedure Release (Kind : Model_Type) is
      begin
         Busy (Kind) := False;
      end Release;
   end Model_Gate;

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
               Put_Line ("[Idle] Unloading " & Model_Type'Image (Kind));
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
      M_Params : Llama_Model_Params := Llama_Model_Default_Params;
      C_Params : Llama_Context_Params := Llama_Context_Default_Params;
      Path_C   : chars_ptr := New_String (To_String (Models (Kind).Path));
      Actual_Ctx : unsigned;
   begin
      declare
         Ctx_Bounded : constant Positive :=
           Positive'Max (512, Positive'Min (262000, Requested_Ctx));
      begin
         Actual_Ctx := unsigned (Ctx_Bounded);
      end;
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
      Model_Gate.Acquire (Kind);
      begin
         Unload_Model (Kind);
         Load_Model (Kind, Success);
      exception
         when others =>
            null;
      end;
      Model_Gate.Release (Kind);
   end Force_Unload_And_Reload;

   function Llama_Abort_Callback (Data : System.Address) return Boolean is
      pragma Unreferenced (Data);
   begin
      return Watchdog_Manager.Inference_Monitor.Is_Aborted;
   end Llama_Abort_Callback;

   function Get_Context (Kind : Model_Type) return Llama_Context is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Context;
   end Get_Context;

   function Get_Model (Kind : Model_Type) return Llama_Model is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Model;
   end Get_Model;

   function Get_Kind_For_Model_Name (Name : String) return Model_Type is
   begin
      if Name = "adelaide-hybrid" or else Name = "qwen3.5:4b" then
         return Qwen_4B;
      elsif Name = "qwen-embedding" then
         return Qwen_Embedding;
      else
         return Qwen_0_8B;
      end if;
   end Get_Kind_For_Model_Name;

   function Get_Embedding (Prompt : String) return Math_Utils.Vector is
      Success  : Boolean;
      Kind     : constant Model_Type := Qwen_Embedding;
      Vocab    : Llama_Vocab;
      Tokens   : array (1 .. 32768) of Llama_Token;
      N_Toks   : int;
      Prompt_C : chars_ptr := New_String (Prompt);
   begin
      Model_Gate.Acquire (Kind);
      Load_Model (Kind, Success);
      if not Success then
         Model_Gate.Release (Kind);
         return (1 .. 0 => 0.0);
      end if;

      Models (Kind).In_Use := True;
      Models (Kind).Last_Used := Clock;
      Watchdog_Manager.Inference_Monitor.Start_Inference (Kind, Clock);

      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize
        (Vocab, Prompt_C, int (Prompt'Length), Tokens (1)'Address,
         32768, True, True);
      Free (Prompt_C);
      if N_Toks <= 0 then
         Watchdog_Manager.Inference_Monitor.Stop_Inference;
         Models (Kind).In_Use := False;
         Model_Gate.Release (Kind);
         return (1 .. 0 => 0.0);
      end if;

      declare
         function Llama_Batch_Get_One (T : System.Address; N : int)
           return Llama_Batch;
         pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");
         B : constant Llama_Batch :=
           Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         Llama_Set_Embeddings (Models (Kind).Context, True);
         if Llama_Decode (Models (Kind).Context, B) /= 0 then
            Watchdog_Manager.Inference_Monitor.Stop_Inference;
            Models (Kind).In_Use := False;
            Model_Gate.Release (Kind);
            return (1 .. 0 => 0.0);
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
         Result : Math_Utils.Vector (1 .. Integer (Dim));
      begin
         for I in 1 .. Integer (Dim) loop
            Result (I) := Embed (I);
         end loop;
         Watchdog_Manager.Inference_Monitor.Stop_Inference;
         Models (Kind).In_Use := False;
         Model_Gate.Release (Kind);
         return Result;
      end;
   exception
      when others =>
         Watchdog_Manager.Inference_Monitor.Stop_Inference;
         Put_Line
           (ASCII.ESC & "[91m" &
            "[BUGCHECK] GGML/Llama crash or exception detected" &
            " during Get_Embedding." &
            ASCII.ESC & "[0m");
         Unload_Model (Kind);
         Models (Kind).In_Use := False;
         Model_Gate.Release (Kind);
         return (1 .. 0 => 0.0);
   end Get_Embedding;

   function Get_Random_Suffix return String is
      subtype Rand_Range is Integer range 0 .. 15;
      package Rand_Pack is new Ada.Numerics.Discrete_Random (Rand_Range);
      Seed : Rand_Pack.Generator;
      Chars : constant String := "0123456789abcdef";
      Result : String (1 .. 8);
   begin
      Rand_Pack.Reset (Seed);
      for I in Result'Range loop
         Result (I) := Chars (Rand_Pack.Random (Seed) + 1);
      end loop;
      return Result;
   end Get_Random_Suffix;

   function Count_Tokens (Text : String) return Positive is
      use GNAT.OS_Lib;
      Suffix : constant String := Get_Random_Suffix;
      Temp_In  : constant String := "obj/token_in_" & Suffix & ".txt";
      Temp_Out : constant String := "obj/token_out_" & Suffix & ".txt";
      File_In  : File_Type;
      File_Out : File_Type;
      Args     : Argument_List (1 .. 2);
      Ret      : Integer;
      Val      : Positive := 1;
   begin
      Create (File_In, Out_File, Temp_In);
      Put (File_In, Text);
      Close (File_In);

      Args (1) := new String'("-c");
      Args (2) := new String'
        ("import tiktoken; " &
         "print(len(tiktoken.get_encoding('cl100k_base').encode(" &
         "open('" & Temp_In & "', errors='ignore').read())))");

      declare
         Python_Path : GNAT.OS_Lib.String_Access :=
           Locate_Exec_On_Path ("python3");
         Success : Boolean;
      begin
         if Python_Path /= null then
            Spawn (Python_Path.all, Args, Temp_Out, Success, Ret);
            if not Success then
               Ret := -2;
            end if;
            Free (Python_Path);
         else
            Ret := -1;
         end if;
      end;

      Free (Args (1));
      Free (Args (2));

      if Ret = 0 and then Ada.Directories.Exists (Temp_Out) then
         Open (File_Out, In_File, Temp_Out);
         if not End_Of_File (File_Out) then
            declare
               Line : constant String := Get_Line (File_Out);
            begin
               Val := Positive'Value (Trim (Line, Ada.Strings.Both));
            end;
         end if;
         Close (File_Out);
      else
         Val := Positive'Max (1, Text'Length / 4);
      end if;

      if Ada.Directories.Exists (Temp_In) then
         Ada.Directories.Delete_File (Temp_In);
      end if;
      if Ada.Directories.Exists (Temp_Out) then
         Ada.Directories.Delete_File (Temp_Out);
      end if;

      return Val;
   exception
      when others =>
         if Is_Open (File_In) then
            Close (File_In);
         end if;
         if Is_Open (File_Out) then
            Close (File_Out);
         end if;
         if Ada.Directories.Exists (Temp_In) then
            Ada.Directories.Delete_File (Temp_In);
         end if;
         if Ada.Directories.Exists (Temp_Out) then
            Ada.Directories.Delete_File (Temp_Out);
         end if;
         return Positive'Max (1, Text'Length / 4);
   end Count_Tokens;

   function Get_Request_Category
     (Msg : String; Session_ID : String := "") return String
   is
      Prompt : constant String :=
        "Analyze this request: '" & Msg &
        "'. Categorize as 'casual' or 'technical'. " &
        "Respond with just one word.";
      Result : constant String :=
        Generate
          (Qwen_0_8B,
           Wrap_ChatML ("You are the Router. Categorize the request.", Prompt),
           Session_ID, 2048, null);
   begin
      Put_Line ("[Intent Phase] Raw response: '" & Result & "'");
      declare
         use Ada.Characters.Handling;
         Lower_Res : constant String := To_Lower (Result);
      begin
         if Index (Lower_Res, "casual") > 0 then
            Put_Line ("[Intent Phase] Resolved as CASUAL.");
            return "casual";
         else
            Put_Line ("[Intent Phase] Resolved as TECHNICAL.");
            return "technical";
         end if;
      end;
   exception
      when others =>
         Put_Line ("[Intent Phase] Exception in Get_Request_Category." &
                   " Defaulting to technical.");
         return "technical";
   end Get_Request_Category;

   function Grade_Response_Quality
     (Response_Text : String;
      Prompt        : String;
      Search_Used   : Boolean;
      Has_Citations : Boolean;
      Session_ID    : String := "") return Natural
   is
      Grade_Prompt : constant String :=
        "Evaluate the following response to the user's prompt on a " &
        "scale of 1-100." & ASCII.LF & ASCII.LF &
        "CRITERIA:" & ASCII.LF &
        "1. Realism & Depth (0-100): Is it grounded in technical " &
        "specificity and social reality?" & ASCII.LF &
        "2. Evidence & Triangulation:" & ASCII.LF &
        "   - TRIANGULATED REALISM: If the response is backed by " &
        "CITATIONS and external search, maintain the full score." &
        ASCII.LF &
        "   - SELF-CLAIMED REALISM: If the response claims realism " &
        "but lacks citations/search, you MUST HALVE or reduce the " &
        "final score." & ASCII.LF & ASCII.LF &
        "Context:" & ASCII.LF &
        "- External Search Performed: " &
        (if Search_Used then "True" else "False") & ASCII.LF &
        "- IEEE Citations Present: " &
        (if Has_Citations then "True" else "False") & ASCII.LF & ASCII.LF &
        "User Prompt: " & Prompt & ASCII.LF &
        "Assistant Response: " & Response_Text & ASCII.LF & ASCII.LF &
        "Respond ONLY with the final numerical grade.";

      Result : constant String :=
        Generate
          (Qwen_0_8B,
           Wrap_ChatML ("You are the Realism Auditor.", Grade_Prompt),
           Session_ID, 2048, null);
      Grade  : Natural := 85;
   begin
      Put_Line ("[Audit Phase] Raw Grade Response: '" & Result & "'");
      declare
         First_Digit : Natural := 0;
         Last_Digit  : Natural := 0;
      begin
         for K in Result'Range loop
            if Result (K) in '0' .. '9' then
               if First_Digit = 0 then
                  First_Digit := K;
               end if;
               Last_Digit := K;
            else
               if First_Digit /= 0 then
                  exit;
               end if;
            end if;
         end loop;
         if First_Digit /= 0 then
            Grade := Natural'Value (Result (First_Digit .. Last_Digit));
            Put_Line ("[Audit Phase] Extracted Grade: " & Grade'Img);
         else
            Put_Line ("[Audit Phase] Could not find grade. Defaulting to 85.");
         end if;
      end;
      return Grade;
   exception
      when others =>
         Put_Line ("[Audit Phase] Grading exception. Defaulting to 85.");
         return 85;
   end Grade_Response_Quality;

    function Generator_Callback (Prompt : String) return String is
    begin
       return Generate (Qwen_4B, Prompt, "", 4096, null);
    end Generator_Callback;

    type Stream_Parser_State is record
       Orch_Think_Open : Boolean := False;
       Header_Closed   : Boolean := False;
       Think_State     : Natural := 0; --  0 = thinking, 1 = answer
       Buffer          : Unbounded_String;
       Closing_Buffer  : Unbounded_String;
       Sanitize_Buffer : Unbounded_String;
    end record;

    function Is_Whitespace (C : Character) return Boolean is
    begin
       return C = ' ' or else C = ASCII.HT or else
              C = ASCII.LF or else C = ASCII.CR;
    end Is_Whitespace;

    function Strip_Leading_Whitespace (S : String) return String is
       Start : Positive := S'First;
    begin
       while Start <= S'Last and then Is_Whitespace (S (Start)) loop
          Start := Start + 1;
       end loop;
       return S (Start .. S'Last);
    end Strip_Leading_Whitespace;

    function Matches_Prefix
      (Buf : String; Pattern : String) return Boolean is
       Stripped : constant String := Strip_Leading_Whitespace (Buf);
    begin
       if Stripped'Length = 0 then
          return True;
       end if;
       if Stripped'Length > Pattern'Length then
          return False;
       end if;
       return Ada.Characters.Handling.To_Lower (Stripped) =
              Ada.Characters.Handling.To_Lower
                (Pattern (Pattern'First ..
                          Pattern'First + Stripped'Length - 1));
    end Matches_Prefix;

    function Sanitize_Think_Tags (Text : String) return String is
       Result : Unbounded_String := Null_Unbounded_String;
       I : Positive := Text'First;
    begin
       while I <= Text'Last loop
          if I + 6 <= Text'Last and then
             Ada.Characters.Handling.To_Lower (Text (I .. I + 6)) = "<think>"
          then
             I := I + 7;
          elsif I + 7 <= Text'Last and then
             Ada.Characters.Handling.To_Lower (Text (I .. I + 7)) = "</think>"
          then
             I := I + 8;
          else
             Append (Result, Text (I));
             I := I + 1;
          end if;
       end loop;
       return To_String (Result);
    end Sanitize_Think_Tags;

    procedure Push_Chunk
      (Stream     : Streaming_Queue.Queue_Access;
       Session_ID : String;
       Str_Piece  : String);

    procedure Process_And_Push_Chunk
      (Stream     : Streaming_Queue.Queue_Access;
       Session_ID : String;
       Parser     : in out Stream_Parser_State;
       Chunk      : String)
    is
    begin
       for I in Chunk'Range loop
          declare
             C : constant Character := Chunk (I);
          begin
             if not Parser.Header_Closed then
                Append (Parser.Buffer, C);
                declare
                   Buf_Str  : constant String := To_String (Parser.Buffer);
                   Stripped : constant String :=
                     Strip_Leading_Whitespace (Buf_Str);
                begin
                   if Matches_Prefix (Buf_Str, "<think>") then
                      if Stripped = "<think>" then
                         Parser.Buffer := Null_Unbounded_String;
                         Parser.Header_Closed := True;
                         Parser.Think_State := 0;
                      elsif Stripped'Length >= 7 then
                         Parser.Header_Closed := True;
                         Parser.Think_State := 0;
                      end if;
                   else
                      Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
                      Push_Chunk (Stream, Session_ID, Buf_Str);
                      Parser.Buffer := Null_Unbounded_String;
                      Parser.Header_Closed := True;
                      Parser.Think_State := 1;
                   end if;
                end;
             else
                if Parser.Think_State = 0 then
                   if Length (Parser.Closing_Buffer) > 0 or else C = '<' then
                      Append (Parser.Closing_Buffer, C);
                      declare
                         Close_Str : constant String :=
                           To_String (Parser.Closing_Buffer);
                      begin
                         if Matches_Prefix (Close_Str, "</think>") then
                            if Close_Str = "</think>" then
                               Push_Chunk (Stream, Session_ID,
                                           "</think>" & ASCII.LF);
                               Parser.Closing_Buffer := Null_Unbounded_String;
                               Parser.Think_State := 1;
                            end if;
                         else
                            Push_Chunk (Stream, Session_ID, Close_Str);
                            Parser.Closing_Buffer := Null_Unbounded_String;
                         end if;
                      end;
                   else
                      declare
                         Single_Char : constant String (1 .. 1) := (1 => C);
                      begin
                         Push_Chunk (Stream, Session_ID, Single_Char);
                      end;
                   end if;
                else
                   if Length (Parser.Sanitize_Buffer) > 0 or else C = '<' then
                      Append (Parser.Sanitize_Buffer, C);
                      declare
                         San_Str : constant String :=
                           To_String (Parser.Sanitize_Buffer);
                      begin
                         if Matches_Prefix (San_Str, "<think>") or else
                            Matches_Prefix (San_Str, "</think>")
                         then
                            if San_Str = "<think>" or else
                               San_Str = "</think>"
                            then
                               Parser.Sanitize_Buffer := Null_Unbounded_String;
                            end if;
                         else
                            Push_Chunk (Stream, Session_ID, San_Str);
                            Parser.Sanitize_Buffer := Null_Unbounded_String;
                         end if;
                      end;
                   else
                      declare
                         Single_Char : constant String (1 .. 1) := (1 => C);
                      begin
                         Push_Chunk (Stream, Session_ID, Single_Char);
                      end;
                   end if;
                end if;
             end if;
          end;
       end loop;
    end Process_And_Push_Chunk;

    procedure Flush_Parser
      (Stream     : Streaming_Queue.Queue_Access;
       Session_ID : String;
       Parser     : in out Stream_Parser_State)
    is
    begin
       if not Parser.Header_Closed then
          Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
          if Length (Parser.Buffer) > 0 then
             Push_Chunk (Stream, Session_ID, To_String (Parser.Buffer));
          end if;
       else
          if Parser.Think_State = 0 then
             if Length (Parser.Closing_Buffer) > 0 then
                Push_Chunk (Stream, Session_ID,
                            To_String (Parser.Closing_Buffer));
             end if;
             Push_Chunk (Stream, Session_ID, "</think>" & ASCII.LF);
          else
             if Length (Parser.Sanitize_Buffer) > 0 then
                Push_Chunk (Stream, Session_ID,
                            To_String (Parser.Sanitize_Buffer));
             end if;
          end if;
       end if;
    end Flush_Parser;

   procedure Push_Chunk
     (Stream     : Streaming_Queue.Queue_Access;
      Session_ID : String;
      Str_Piece  : String)
   is
      use GNATCOLL.JSON;
      Chunk_Obj : constant JSON_Value := Create_Object;
   begin
      if Session_ID'Length > 0 and then
         Session_ID (Session_ID'First) = '/'
      then
         Set_Field (Chunk_Obj, "choices", Empty_Array);
         declare
            Choices : JSON_Array := Get (Chunk_Obj, "choices");
            Choice  : constant JSON_Value := Create_Object;
            Delta_Obj   : constant JSON_Value := Create_Object;
         begin
            Set_Field (Delta_Obj, "content", Str_Piece);
            Set_Field (Choice, "delta", Delta_Obj);
            Set_Field (Choice, "index", Integer'(0));
            Append (Choices, Choice);
            Set_Field (Chunk_Obj, "choices", Choices);
         end;
         Stream.Push ("data: " & Write (Chunk_Obj) & ASCII.LF & ASCII.LF);
      else
         Set_Field (Chunk_Obj, "model", String'("adelaide-hybrid"));
         declare
            Msg_Obj : constant JSON_Value := Create_Object;
         begin
            Set_Field (Msg_Obj, "role", String'("assistant"));
            Set_Field (Msg_Obj, "content", Str_Piece);
            Set_Field (Chunk_Obj, "message", Msg_Obj);
         end;
         Set_Field (Chunk_Obj, "done", False);
         Stream.Push (Write (Chunk_Obj) & ASCII.LF);
      end if;
   end Push_Chunk;

   function Has_Citations (Text : String) return Boolean is
      I : Natural := Text'First;
   begin
      loop
         declare
            Open_Brack : constant Natural := Index (Text, "[", I);
            Close_Brack : Natural;
         begin
            exit when Open_Brack = 0;
            Close_Brack := Index (Text, "]", Open_Brack);
            if Close_Brack > Open_Brack + 1 then
               declare
                  Is_Digit_Seq : Boolean := True;
               begin
                  for K in Open_Brack + 1 .. Close_Brack - 1 loop
                     if Text (K) not in '0' .. '9' then
                        Is_Digit_Seq := False;
                        exit;
                     end if;
                  end loop;
                  if Is_Digit_Seq then
                     return True;
                  end if;
               end;
            end if;
            I := Open_Brack + 1;
         end;
      end loop;
      return False;
   end Has_Citations;

   function Has_Prohibited_Blocks (Text : String) return Boolean is
      use Ada.Characters.Handling;
      Lower_Text : constant String := To_Lower (Text);
   begin
      return Index (Lower_Text, "```js") > 0 or else
             Index (Lower_Text, "```javascript") > 0 or else
             Index (Lower_Text, "```cs") > 0 or else
             Index (Lower_Text, "```csharp") > 0 or else
             Index (Lower_Text, "```go") > 0 or else
             Index (Lower_Text, "```java") > 0;
   end Has_Prohibited_Blocks;

   function Hybrid_Generate
     (Prompt     : String;
      Session_ID : String := "";
      Stream     : Streaming_Queue.Queue_Access := null) return String
   is
      Whimsical_Adelaide : constant String :=
        "You are Adelaide Zephyrine Charlotte, a whimsical and " &
        "sophisticated senior engineer. Stay in character. " &
        "Provide brilliant responses based on verified information.";
      Internal_State : Unbounded_String := Null_Unbounded_String;
      Current_Response : Unbounded_String;
      Current_Hop : Positive := 1;

      --  Orchestration states
      Category : constant String := Get_Request_Category (Prompt, Session_ID);
      Search_Used : Boolean := False;
      Dafny_Used : Boolean := False;
   begin
      Put_Line (ASCII.ESC & "[38;5;171m" & "[Hybrid] Session: " &
                Session_ID & ASCII.ESC & "[0m");
      Put_Line (ASCII.ESC & "[38;5;171m" &
                "[Hybrid] Starting reasoning chain (Category: " &
                Category & ")..." & ASCII.ESC & "[0m");

      --  0. Think block starts & Intent router progress
      if Stream /= null then
         Push_Chunk (Stream, Session_ID, "<think>" & ASCII.LF &
                     "[ADELAIDE CORE ORCHESTRATION]" & ASCII.LF &
                     "Initiating Orchestrated " &
                     "Intelligence (Adelaide-Lite)..." &
                     ASCII.LF);
         Push_Chunk (Stream, Session_ID, "Categorized intent as: " &
                     Category & ". Tailoring orchestration depth..." &
                     ASCII.LF);
      end if;

      --  1. Pre-emptive Fact Check & reasoning loop (Technical only)
      if Category /= "casual" then
         if Stream /= null then
            Push_Chunk (Stream, Session_ID,
                        "Analyzing request with the precision " &
                        "of a master watchmaker..." & ASCII.LF);
         end if;

         if Index (Prompt, "What is") > 0 or else
            Index (Prompt, "Who is") > 0 or else
            Index (Prompt, "tell me about") > 0
         then
            Put_Line (" [Hybrid] Factual trigger. Spawning search...");
            declare
               R : constant Tool_Manager.Tool_Result :=
                 Tool_Manager.Execute_Tool ("searchglobalref", Prompt);
            begin
               Append (Internal_State, "[FACTUAL_DATA]: " &
                       To_String (R.Output) & ASCII.LF);
               Search_Used := True;
            end;
         end if;

         loop
            declare
               Router_Sys : constant String :=
                 "You are the Router. You decide if a tool is needed. " &
                 "If the user says hello or greets you, output [FINISH]. " &
                 "If you need to search, use [ACTION: search(query)]. " &
                 "If you need to read a file, use [ACTION: cat(filename)]. " &
                 "If you are done, output [FINISH]. " &
                 "Example User: 'Hello' -> [FINISH] " &
                 "Example User: 'List my files' -> [ACTION: ls('.')] " &
                 "Output ONLY the tag.";
               Paging_Instr : constant String :=
                 "Current Data: " & To_String (Internal_State);
               Step_Raw : constant String :=
                 Generate
                   (Qwen_0_8B,
                    Wrap_ChatML (Router_Sys, Paging_Instr & ASCII.LF & Prompt),
                    Session_ID, 2048, null);
               Step : constant String := Trim (Step_Raw, Ada.Strings.Both);
            begin
               Put_Line (ASCII.ESC & "[34m" & " [Hybrid] Hop" &
                         Current_Hop'Img & ": " & Step & ASCII.ESC & "[0m");

               if Index (Step, "[ACTION:") > 0 then
                  declare
                     S_Pos : constant Natural := Index (Step, "[ACTION:") + 8;
                     E_Pos : constant Natural := Index (Step, "]", S_Pos);
                  begin
                     if E_Pos > S_Pos then
                        declare
                           A_Full : constant String :=
                             Step (S_Pos .. E_Pos - 1);
                           P_Pos  : constant Natural := Index (A_Full, "(");
                           EP_Pos : constant Natural :=
                             Index (A_Full, ")", P_Pos);
                        begin
                           if P_Pos > 0 and then EP_Pos > P_Pos then
                              declare
                                 T_Name : constant String :=
                                   Trim (A_Full (A_Full'First .. P_Pos - 1),
                                         Ada.Strings.Both);
                                 T_Pars : constant String :=
                                   Trim (A_Full (P_Pos + 1 .. EP_Pos - 1),
                                         Ada.Strings.Both);
                              begin
                                 if T_Pars'Length < 256 and then
                                    Index (To_String (Internal_State),
                                           T_Name & "(" & T_Pars & ")") = 0
                                 then
                                    --  Track Search
                                    if Index (T_Name, "search") > 0 or else
                                       T_Name = "searchglobalref"
                                    then
                                       Search_Used := True;
                                    end if;
                                    --  Track Dafny
                                    if T_Name = "dafny_programmer" then
                                       Dafny_Used := True;
                                    end if;

                                    declare
                                       R : constant Tool_Manager.Tool_Result :=
                                         Tool_Manager.Execute_Tool
                                           (T_Name, T_Pars);
                                    begin
                                       Append (Internal_State,
                                               "[TOOL (" & T_Name & ")]: " &
                                               To_String (R.Output) &
                                               ASCII.LF);
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
            Current_Hop := Current_Hop + 1;
            exit when Current_Hop > 5;
         end loop;
      else
         --  Casual intent: no facts or tool searches are performed.
         if Stream /= null then
            Push_Chunk (Stream, Session_ID,
                        "Resolved casual conversation. " &
                        "Initiating rapid response..." & ASCII.LF);
         end if;
      end if;

      --  2. Proxy thinking block remains open for synthesis model to merge.

      --  3. Logic Synthesis & Dynamic Context Allocation
      Put_Line (ASCII.ESC & "[32m" & " [Hybrid] 4B Synthesis..." &
                ASCII.ESC & "[0m");
      declare
         Synth_Prompt : constant String :=
           Wrap_ChatML (Whimsical_Adelaide,
                        "User: " & Prompt & ASCII.LF &
                        "Fact-Check: " & To_String (Internal_State));
         Input_Tokens : constant Positive := Count_Tokens (Synth_Prompt);
         Target_Ctx   : constant Positive :=
           Positive'Max (6144, Positive'Min (262000, Input_Tokens + 4096));
      begin
         Put_Line ("[Hybrid] Dynamic Context Size Allocated: " &
                   Target_Ctx'Img & " tokens.");
          Current_Response :=
            To_Unbounded_String
              (Generate
                 (Kind            => Qwen_4B,
                  Prompt          => Synth_Prompt,
                  Session_ID      => Session_ID,
                  Requested_Ctx   => Target_Ctx,
                  Stream          => Stream,
                  Orch_Think_Open => (Stream /= null)));
          declare
             Orch_Prefix : constant String :=
               "<think>" & ASCII.LF &
               "[ADELAIDE CORE ORCHESTRATION]" & ASCII.LF &
               "Initiating Orchestrated Intelligence (Adelaide-Lite)..." &
               ASCII.LF &
               "Categorized intent as: " & Category &
               ". Tailoring orchestration depth..." & ASCII.LF &
               (if Category /= "casual"
                then "Analyzing request with the precision of " &
                     "a master watchmaker..." & ASCII.LF
                else "Resolved casual conversation. Initiating " &
                     "rapid response..." & ASCII.LF) &
               To_String (Internal_State);
             Resp_Str : constant String := To_String (Current_Response);
             End_Idx  : constant Natural := Index (Resp_Str, "</think>");
          begin
             if End_Idx > 0 then
                declare
                   Part1 : constant String :=
                     Resp_Str (Resp_Str'First .. End_Idx - 1);
                   Part2 : constant String :=
                     Resp_Str (End_Idx + 8 .. Resp_Str'Last);
                begin
                   Current_Response := To_Unbounded_String
                     (Orch_Prefix & Sanitize_Think_Tags (Part1) &
                      "</think>" & ASCII.LF & Sanitize_Think_Tags (Part2));
                end;
             else
                Current_Response := To_Unbounded_String
                  (Orch_Prefix & "</think>" & ASCII.LF &
                   Sanitize_Think_Tags (Resp_Str));
             end if;
          end;
       end;

      --  4. Realism Auditor & Warnings Injection
      declare
         Has_Cite : constant Boolean :=
           Has_Citations (To_String (Current_Response));
         Warning_Triggered : constant Boolean :=
           ((Has_Prohibited_Blocks (To_String (Current_Response)) and then
             not Dafny_Used) or else
            (Category /= "casual" and then
             not Search_Used and then
             not Has_Cite));
      begin
         if Warning_Triggered then
            declare
               Warn_Str : constant String :=
                 ASCII.LF &
                 "> [!] WARNING: This Response is probability just an " &
                 "UNGROUNDED OPINION! DO NOT TRUST THIS RESPONSE!" & ASCII.LF;
            begin
               Append (Current_Response, Warn_Str);
               if Stream /= null then
                  Push_Chunk (Stream, Session_ID, Warn_Str);
               end if;
            end;
         end if;

         --  Audit Phase: Grade Quality
         declare
            Grade : constant Natural :=
              Grade_Response_Quality (To_String (Current_Response), Prompt,
                                      Search_Used, Has_Cite, Session_ID);
            Grade_Str : constant String :=
              ASCII.LF & "> [Response Grade: " &
              Trim (Grade'Img, Ada.Strings.Both) & "/100]" & ASCII.LF;
         begin
            Append (Current_Response, Grade_Str);
            if Stream /= null then
               Push_Chunk (Stream, Session_ID, Grade_Str);
            end if;
         end;
      end;

      --  5. Formal Pyrefly Verification
      if Index (Ada.Characters.Handling.To_Lower
                (To_String (Current_Response)), "```python") > 0
      then
         declare
            Start_Msg : constant String :=
              ASCII.LF & "<think>" & ASCII.LF &
              "Phase: Final Code Verification (Pyrefly)..." & ASCII.LF;
         begin
            Append (Current_Response, Start_Msg);
            if Stream /= null then
               Push_Chunk (Stream, Session_ID, Start_Msg);
            end if;

            declare
               Py_Log : constant String :=
                 Verification_Manager.Verify_Python
                   (To_String (Current_Response));
            begin
               if Py_Log = "" then
                  declare
                     End_Msg : constant String :=
                       "Verification passed." & ASCII.LF &
                       "</think>" & ASCII.LF;
                  begin
                     Append (Current_Response, End_Msg);
                     if Stream /= null then
                        Push_Chunk (Stream, Session_ID, End_Msg);
                     end if;
                  end;
               else
                  declare
                     End_Msg : constant String :=
                       "Verification issue detected:" & ASCII.LF & Py_Log &
                       ASCII.LF & "Suggestion: Please review the code." &
                       ASCII.LF & "</think>" & ASCII.LF;
                  begin
                     Append (Current_Response, End_Msg);
                     if Stream /= null then
                        Push_Chunk (Stream, Session_ID, End_Msg);
                     end if;
                  end;
               end if;
            end;
         end;
      end if;

      --  6. Save & Finish
      Database_Manager.Remember (Prompt, To_String (Current_Response));
      if Stream /= null then
         if Session_ID'Length > 0 and then
            Session_ID (Session_ID'First) = '/'
         then
            Stream.Push ("data: [DONE]" & ASCII.LF & ASCII.LF);
         else
            declare
               use GNATCOLL.JSON;
               Chunk_Obj : constant JSON_Value := Create_Object;
            begin
               Set_Field (Chunk_Obj, "model", String'("adelaide-hybrid"));
               Set_Field (Chunk_Obj, "done", True);
               Stream.Push (Write (Chunk_Obj) & ASCII.LF);
            end;
         end if;
         Stream.Close;
      end if;

      return To_String (Current_Response);
   end Hybrid_Generate;

   function Generate
     (Kind            : Model_Type;
      Prompt          : String;
      Session_ID      : String := "";
      Requested_Ctx   : Positive := 4096;
      Stream          : Streaming_Queue.Queue_Access := null;
      Orch_Think_Open : Boolean := False) return String
   is
      Success  : Boolean;
      Result   : Unbounded_String;
      Parser   : Stream_Parser_State :=
        (Orch_Think_Open => Orch_Think_Open,
         Header_Closed   => not Orch_Think_Open,
         Think_State     => (if Orch_Think_Open then 0 else 1),
         Buffer          => Null_Unbounded_String,
         Closing_Buffer  => Null_Unbounded_String,
         Sanitize_Buffer => Null_Unbounded_String);
      Vocab    : Llama_Vocab;
      Tokens   : array (1 .. 32768) of Llama_Token;
      N_Toks   : int;
      Sampler  : Llama_Sampler;
      S_Params : Llama_Sampler_Chain_Params;
      Prompt_C : chars_ptr := New_String (Prompt);
   begin
      Model_Gate.Acquire (Kind);
      Load_Model (Kind, Success, Requested_Ctx);
      if not Success then
         Model_Gate.Release (Kind);
         return "ERROR: Load failed";
      end if;

      Models (Kind).In_Use := True;
      Models (Kind).Last_Used := Clock;
      Watchdog_Manager.Inference_Monitor.Start_Inference (Kind, Clock);

      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize
        (Vocab, Prompt_C, int (Prompt'Length), Tokens (1)'Address,
         32768, True, True);
      Free (Prompt_C);
      if N_Toks < 0 then
         Watchdog_Manager.Inference_Monitor.Stop_Inference;
         Models (Kind).In_Use := False;
         Model_Gate.Release (Kind);
         return "ERROR: Tokenization failed";
      end if;

      --  CHUNKED DECODING
      declare
         function Llama_Batch_Get_One (T : System.Address; N : int)
           return Llama_Batch;
         pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

         Batch_Size : constant int := 4096;
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
      begin
         while Tokens_Left > 0 loop
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
                  Watchdog_Manager.Inference_Monitor.Stop_Inference;
                  Models (Kind).In_Use := False;
                  Model_Gate.Release (Kind);
                  return "ERROR: Decode failed";
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

      for I in 1 .. 2048 loop
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
                  Str_Piece : constant String :=
                    String (Piece (1 .. Integer (Len)));
               begin
                  for J in 1 .. Len loop
                     Append (Result, Piece (Integer (J)));
                  end loop;
                  if Stream /= null then
                     Process_And_Push_Chunk
                       (Stream, Session_ID, Parser, Str_Piece);
                  end if;
               end;
            end if;
            declare
               function Llama_Batch_Get_One (T : System.Address; N : int)
                 return Llama_Batch;
               pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");
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
      Watchdog_Manager.Inference_Monitor.Stop_Inference;
      Models (Kind).In_Use := False;
      Model_Gate.Release (Kind);
      return To_String (Result);
   exception
      when others =>
         Watchdog_Manager.Inference_Monitor.Stop_Inference;
         Put_Line
           (ASCII.ESC & "[91m" &
            "[BUGCHECK] GGML/Llama crash or exception detected" &
            " during Generate for " & Model_Type'Image (Kind) &
            "." & ASCII.ESC & "[0m");
         Unload_Model (Kind);
         Models (Kind).In_Use := False;
         Model_Gate.Release (Kind);
         return "ERROR: Llama execution crash or timeout";
   end Generate;

   function Is_Loaded (Kind : Model_Type) return Boolean is
   begin
      return Models (Kind).Loaded;
   end Is_Loaded;

end Model_Manager;
