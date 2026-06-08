pragma SPARK_Mode (Off);
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
with Ada.Directories;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Unchecked_Conversion;
with Ada.Exceptions;
with Watchdog_Manager;
with Kratos;
with ELP_Queue;
with System;

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
                               " Pipeline: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET * 1_000_000_000)) & "ns | " &
                               "ELP0: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP0 * 1_000_000_000)) & "ns | " &
                               "ELP1: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP1 * 1_000_000_000)) & "ns | " &
                               "ELP2: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP2 * 1_000_000_000)) & "ns | " &
                               "ELP3: " &
                               Long_Long_Integer'Image (Long_Long_Integer (Current_WCET_ELP3 * 1_000_000_000)) & "ns");
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
      Qwen_9B        => Qwen_9B,
      Qwen_Embedding => Qwen_Embedding,
      MMProj         => MMProj];

   type Owner_Array is array (Model_Type) of ELP_Level;
   type Busy_Array is array (Model_Type) of Boolean;

   --  PRIORITY MODEL GATE:
   --  Manages access to the model contexts.
   --  ELP1 requests (User Interactions) preempt running ELP0 requests (Background Tasks).
   protected Priority_Model_Gate is
      procedure Request_ELP1;
      entry Acquire_ELP1 (Model_Type);
      procedure Release_ELP1 (Kind : Model_Type);
      entry Acquire_ELP0 (Model_Type) (Success : out Boolean);
      procedure Release_ELP0 (Kind : Model_Type);
      procedure Try_Acquire_For_Cleanup (Kind : Model_Type; Success : out Boolean);
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

      procedure Try_Acquire_For_Cleanup (Kind : Model_Type; Success : out Boolean) is
      begin
         if Busy (Kind) or else ELP1_Pending > 0 or else ELP1_Active_Count > 0 then
            Success := False;
         else
            Busy (Kind) := True;
            Owner (Kind) := ELP1; -- Treat cleanup as high priority/exclusive
            Success := True;
         end if;
      end Try_Acquire_For_Cleanup;

      function Should_Abort return Boolean is
      begin
         return ELP1_Pending > 0 or else ELP1_Active_Count > 0;
      end Should_Abort;

      function Is_ELP0_Owner (Kind : Model_Type) return Boolean is
      begin
         return Owner (Kind) = ELP0;
      end Is_ELP0_Owner;
   end Priority_Model_Gate;

   --  IDLE MONITOR:
   --  Unloads models after 30 seconds of inactivity to free VRAM.
   task Idle_Monitor is
      pragma Storage_Size (1024 * 1024);
      entry Start;
   end Idle_Monitor;

   task body Idle_Monitor is
      Next_Check : Time;
      Interval   : constant Time_Span := Seconds (1);
      Timeout    : constant Time_Span := Seconds (30);
      Now        : Time;
      Cleanup_OK : Boolean;
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
               Priority_Model_Gate.Try_Acquire_For_Cleanup (Kind, Cleanup_OK);
               if Cleanup_OK then
                  Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Idle]" &
                            AnsiAda.Reset & " Unloading " &
                            Model_Type'Image (Kind));
                  Unload_Model (Kind);
                  Priority_Model_Gate.Release_ELP1 (Kind); -- Match Acquire_For_Cleanup
               end if;
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
      ELP_Queue.Initialize;
      Models (Qwen_0_8B).Path  := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/Qwen3.5-0.8B-Q4_K_S.gguf");
      Models (Qwen_9B).Path   := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/Qwen3.5-9B-UD-Q2_K_XL.gguf");
      Models (Qwen_Embedding).Path := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/Qwen3-Embedding-0.6B-Q8_0.gguf");
      Models (MMProj).Path := To_Unbounded_String
        ("llama.cpp/models/qwen3.5/mmproj-9B-F16.gguf");
      Idle_Monitor.Start;
   end Initialize;

   procedure Load_Model
     (Kind          : Model_Type;
      Success       : out Boolean;
      Requested_Ctx : Positive := 4096)
    is
       M_Params   : Llama_Model_Params := Llama_Model_Default_Params;
       C_Params   : Llama_Context_Params := Llama_Context_Default_Params;
       Actual_Ctx : unsigned;
       
       Base_Path  : constant String := To_String (Models (Kind).Path);
       -- Try direct, ../ (from src/Adelaide_Lite), and ../../ (from bin)
       Paths      : constant array (1 .. 3) of Unbounded_String :=
         (To_Unbounded_String (Base_Path),
          To_Unbounded_String ("../" & Base_Path),
          To_Unbounded_String ("../../" & Base_Path));
    begin
       Actual_Ctx := unsigned (Requested_Ctx);
       --  Minimum context size is now 8192 for stability and headroom.
       if Actual_Ctx < 8192 then
          Actual_Ctx := 8192;
       end if;
       
       Success := False;
       if Models (Kind).Loaded then
          if Actual_Ctx <= Models (Kind).Current_Ctx then
             Models (Kind).Last_Used := Clock;
             Success := True;
             return;
          end if;
          Unload_Model (Kind);
       end if;
 
       Put_Line ("[+] Loading " & Model_Type'Image (Kind) &
                 " (N_CTX=" & Actual_Ctx'Img & ")");
       M_Params.N_Gpu_Layers := -1;
       
       for I in Paths'Range loop
          declare
             Path_Str : constant String := To_String (Paths (I));
          begin
             if Ada.Directories.Exists (Path_Str) then
                declare
                   Path_C : chars_ptr := New_String (Path_Str);
                begin
                   begin
                      Models (Kind).Model := Llama_Model_Load_From_File (Path_C, M_Params);
                   exception
                      when others =>
                         Put_Line ("[!] Exception caught in Ada during Llama_Model_Load_From_File");
                         Models (Kind).Model := Null_Model;
                   end;
                   Free (Path_C);
                   if Models (Kind).Model /= Null_Model then
                      exit;
                   end if;
                end;
             end if;
          end;
       end loop;
 
       if Models (Kind).Model /= Null_Model then
         C_Params.N_Ctx := Actual_Ctx;
         C_Params.N_Batch := 512;
         C_Params.N_Ubatch := 512;
         C_Params.N_Threads := 8;
         C_Params.N_Threads_Batch := 8;
         C_Params.Type_K := GGML_TYPE_Q4_1;
         C_Params.Type_V := GGML_TYPE_Q4_1;
         C_Params.Flash_Attn_Type := 1;
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
      
      --  1. Abort if Watchdog has flagged a timeout for this model.
      if Watchdog_Manager.Inference_Monitor.Is_Aborted and then
         Watchdog_Manager.Inference_Monitor.Current_Inference_Model = Ptr.all
      then
         return True;
      end if;

      --  2. Only abort if we are an ELP0 task and an ELP1 task is pending or active.
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
        or else Name = "Snowball-Enaga"
      then
         return Qwen_9B;
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
         Ada.Text_IO.Put_Line ("Push_Chunk called with: " & Str_Piece (Str_Piece'First .. Natural'Min(Str_Piece'Last, Str_Piece'First + 20)));
         Stream.Push (Str_Piece);
      end if;
   end Push_Chunk;

   function Generator_Callback (Prompt : String) return String is
   begin
      return "Callback response to " & Prompt;
   end Generator_Callback;

   function Sanitize_UTF8 (S : String) return String is
      Res : Unbounded_String;
      I   : Positive := S'First;
      Val : Natural;
   begin
      while I <= S'Last loop
         Val := Character'Pos (S (I));
         if Val < 128 then
            Append (Res, S (I));
            I := I + 1;
         elsif Val >= 192 and Val <= 223 then
            if I + 1 <= S'Last and then
               Character'Pos (S (I + 1)) >= 128 and then Character'Pos (S (I + 1)) <= 191
            then
               Append (Res, S (I .. I + 1));
               I := I + 2;
            else
               I := I + 1;
            end if;
         elsif Val >= 224 and Val <= 239 then
            if I + 2 <= S'Last and then
               Character'Pos (S (I + 1)) >= 128 and then Character'Pos (S (I + 1)) <= 191 and then
               Character'Pos (S (I + 2)) >= 128 and then Character'Pos (S (I + 2)) <= 191
            then
               Append (Res, S (I .. I + 2));
               I := I + 3;
            else
               I := I + 1;
            end if;
         elsif Val >= 240 and Val <= 247 then
            if I + 3 <= S'Last and then
               Character'Pos (S (I + 1)) >= 128 and then Character'Pos (S (I + 1)) <= 191 and then
               Character'Pos (S (I + 2)) >= 128 and then Character'Pos (S (I + 2)) <= 191 and then
               Character'Pos (S (I + 3)) >= 128 and then Character'Pos (S (I + 3)) <= 191
            then
               Append (Res, S (I .. I + 3));
               I := I + 4;
            else
               I := I + 1;
            end if;
         else
            I := I + 1;
         end if;
      end loop;
      return To_String (Res);
   end Sanitize_UTF8;

   --  SINGLE EMBEDDING HELPER
   procedure Get_Single_Embedding
     (Prompt : String;
      Result : out Math_Utils.Vector;
      Length : out Natural;
      Level  : ELP_Level := ELP1)
   is
      Success  : Boolean;
      Kind     : constant Model_Type := Qwen_Embedding;
      Vocab    : Llama_Vocab;
      Tokens   : array (1 .. 32768) of Llama_Token;
      N_Toks   : int;
      Clean_P  : constant String := Sanitize_UTF8 (Prompt);
      Prompt_C : chars_ptr := New_String (Clean_P);
   begin
      ELP_Queue.Enqueue (Level, Kind);
      if Level = ELP0 then
          Priority_Model_Gate.Acquire_ELP0 (Kind) (Success);
          if not Success then
            declare D : ELP_Level; K : Model_Type;
            begin ELP_Queue.Dequeue (D, K); end;
            Length := 0;
            Free (Prompt_C);
            return;
         end if;
      else
         Priority_Model_Gate.Request_ELP1;
         Priority_Model_Gate.Acquire_ELP1 (Kind);
      end if;
      Load_Model (Kind, Success);
      if not Success then
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         declare D : ELP_Level; K : Model_Type;
         begin ELP_Queue.Dequeue (D, K); end;
         Length := 0;
         Free (Prompt_C);
         return;
      end if;

      Models (Kind).In_Use := True;
      Models (Kind).Last_Used := Clock;
      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
          N_Toks := Llama_Tokenize
            (Vocab, Prompt_C, int (Clean_P'Length), Tokens (1)'Address,
             32768, True, True);
          Put_Line ("[Tokenize-Debug] Model:" & Kind'Img &
                    " Prompt_Len:" & Clean_P'Length'Img &
                    " N_Toks:" & N_Toks'Img);
          Free (Prompt_C);
      if N_Toks <= 0 then
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         declare D : ELP_Level; K : Model_Type;
         begin ELP_Queue.Dequeue (D, K); end;
         Length := 0;
         return;
      end if;

      --  CHUNKED DECODING FOR EMBEDDINGS
      declare
         Batch_Size  : constant int := int'Min (512, int (Models (Kind).Current_Ctx));
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
      begin
         Llama_Interface.Llama_Memory_Clear
           (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);
         Llama_Set_Embeddings (Models (Kind).Context, True);
         
         while Tokens_Left > 0 loop
            declare
               To_Decode : constant int :=
                 (if Tokens_Left > Batch_Size then Batch_Size else Tokens_Left);
               B : constant Llama_Batch :=
                 Llama_Batch_Get_One
                   (Tokens (Integer (Current_Pos) + 1)'Address, To_Decode);
               Dec_Result : int;
            begin
               if Kratos.Guard_Enter = 0 then
                  Dec_Result := Llama_Decode (Models (Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Dec_Result := -1;
               end if;
               if Dec_Result /= 0 then
                  Models (Kind).In_Use := False;
                  if Level = ELP0 then
                     Priority_Model_Gate.Release_ELP0 (Kind);
                  else
                     Priority_Model_Gate.Release_ELP1 (Kind);
                  end if;
                  declare D : ELP_Level; K : Model_Type;
                  begin ELP_Queue.Dequeue (D, K); end;
                  Length := 0;
                  return;
               end if;
               Tokens_Left := Tokens_Left - To_Decode;
               Current_Pos := Current_Pos + To_Decode;
            end;
         end loop;
      end;

      declare
         use System;
         function Llama_Model_N_Embd (M : Llama_Model) return int;
         pragma Import (C, Llama_Model_N_Embd, "llama_model_n_embd");
         Dim : constant int := Llama_Model_N_Embd (Models (Kind).Model);
         Ptr : constant Address :=
           Llama_Get_Embeddings (Models (Kind).Context);
         --  SAFE: copy via C memcpy instead of Ada address overlay
         --  which could corrupt heap if Dim is wrong
         function Memcpy
           (Dst, Src : Address; N : Interfaces.C.size_t)
            return Address;
         pragma Import (C, Memcpy, "memcpy");
         Copy_Count : constant Integer :=
           Integer (Interfaces.C.size_t'Min
             (Interfaces.C.size_t (Dim),
              Interfaces.C.size_t (Result'Length)));
      begin
         if Copy_Count > 0 and then Ptr /= Null_Address then
            declare
               Dummy : Address;
            begin
               Dummy := Memcpy (Result (Result'First)'Address, Ptr,
                         Interfaces.C.size_t (Copy_Count) *
                           Interfaces.C.size_t (Float'Size / 8));
            end;
            Length := Copy_Count;
         else
            Length := 0;
         end if;
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         declare D : ELP_Level; K : Model_Type;
         begin ELP_Queue.Dequeue (D, K); end;
      end;
   exception
      when others =>
         Models (Kind).In_Use := False;
         if Level = ELP0 then
            Priority_Model_Gate.Release_ELP0 (Kind);
         else
            Priority_Model_Gate.Release_ELP1 (Kind);
         end if;
         declare D : ELP_Level; K : Model_Type;
         begin ELP_Queue.Dequeue (D, K); end;
         Length := 0;
   end Get_Single_Embedding;
    --  GET EMBEDDING (WITH CHUNKING > 800 CHARS)

    procedure Get_Embedding
      (Prompt : String;
       Result : out Math_Utils.Vector;
       Length : out Natural;
       Level  : ELP_Level := ELP1)
    is
    begin
       if Prompt'Length <= 800 then
          Get_Single_Embedding (Prompt, Result, Length, Level);
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
                   Get_Single_Embedding (Sub_Prompt, Sub_Vec, Sub_Len, Level);
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
   type Stream_Parser_State is record
      Orch_Think_Open : Boolean := False;
      Sanitize_Buffer : Unbounded_String := Null_Unbounded_String;
      In_Think_Block  : Boolean := False;
   end record;

   function Is_Prefix (S, Tag : String) return Boolean is
   begin
      return S'Length < Tag'Length
        and then Tag (Tag'First .. Tag'First + S'Length - 1) = S;
   end Is_Prefix;

    procedure Process_And_Push_Char
       (Stream     : Streaming_Queue.Queue_Access;
        Session_ID : String;
        Parser     : in out Stream_Parser_State;
        C          : Character)
    is
       --  Support both <thinking> and ` tags
       Think_Tag_A : constant String := "<thinking>";
       Think_Tag_B : constant String := "<think>";
       Close_Tag_A : constant String := "</thinking>";
       Close_Tag_B : constant String := "</think>";
       Response_Tag : constant String := "</response>";
    begin
       Append (Parser.Sanitize_Buffer, C);
       declare
          Buf : constant String := To_String (Parser.Sanitize_Buffer);
       begin
          if Buf = Think_Tag_A or else Buf = Think_Tag_B then
             Parser.Sanitize_Buffer := Null_Unbounded_String;
             Parser.In_Think_Block := True;
             return;
          elsif Buf = Close_Tag_A or else Buf = Close_Tag_B then
             Parser.Sanitize_Buffer := Null_Unbounded_String;
             Parser.In_Think_Block := False;
             if Parser.Orch_Think_Open then
                Parser.Orch_Think_Open := False;
             end if;
             return;
          elsif Buf = Response_Tag then
             Parser.Sanitize_Buffer := Null_Unbounded_String;
             return;
          end if;

          -- If current buffer is a potential prefix of any tag, wait for more.
          if Is_Prefix (Buf, Think_Tag_A)
            or else Is_Prefix (Buf, Think_Tag_B)
            or else Is_Prefix (Buf, Close_Tag_A)
            or else Is_Prefix (Buf, Close_Tag_B)
            or else Is_Prefix (Buf, Response_Tag)
          then
             return;
          end if;

          -- Stream content out, but SILENCE the think block entirely
          if not Parser.In_Think_Block then
             delay 0.0005;
             Push_Chunk (Stream, Session_ID, Buf);
          end if;
          Parser.Sanitize_Buffer := Null_Unbounded_String;
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
      declare
         S_Str : constant String := To_String (Parser.Sanitize_Buffer);
      begin
         if S_Str /= "" then
            Push_Chunk (Stream, Session_ID, S_Str);
            Parser.Sanitize_Buffer := Null_Unbounded_String;
         end if;
      end;
      if Parser.Orch_Think_Open then
         --  Silently close orchestration thinking; tag is stripped by parser
         Parser.Orch_Think_Open := False;
      end if;
   end Flush_Parser;

   function Sanitize_Think_Tags (Text : String) return String is
      Res : Unbounded_String;
      I   : Positive := Text'First;
   begin
      while I <= Text'Last loop
         if I + 9 <= Text'Last and then Text (I .. I + 9) = "<thinking>" then
            --  Skip everything until closing </thinking>
            I := I + 10;
            while I <= Text'Last loop
               if I + 10 <= Text'Last and then Text (I .. I + 10) = "</thinking>" then
                  I := I + 11;
                  exit;
               else
                  I := I + 1;
               end if;
            end loop;
         elsif I + 6 <= Text'Last and then Text (I .. I + 6) = "<think>" then
            --  Skip everything until closing 
            I := I + 7;
            while I <= Text'Last loop
               if I + 7 <= Text'Last and then Text (I .. I + 7) = "</think>" then
                  I := I + 8;
                  exit;
               else
                  I := I + 1;
               end if;
            end loop;
         elsif I + 10 <= Text'Last and then Text (I .. I + 10) = "</response>" then
            I := I + 11;
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
      
      Clean_P  : constant String := Sanitize_UTF8 (Prompt);
      Prompt_C : chars_ptr := New_String (Clean_P);
      Parser   : Stream_Parser_State;
   begin
      pragma Unreferenced (Images);
      Result := Null_Unbounded_String;
      Parser.Orch_Think_Open := Orch_Think_Open;

      begin
         if Level = ELP0 then
            declare
               Acq_OK : Boolean;
            begin
               Priority_Model_Gate.Acquire_ELP0 (Kind) (Acq_OK);
               if not Acq_OK then
                  Result := To_Unbounded_String ("ERROR: Preempted");
                  Free (Prompt_C);
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
            Free (Prompt_C);
            return;
         end if;

         Models (Kind).In_Use := True;
         Models (Kind).Last_Used := Clock;
         Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
          N_Toks := Llama_Tokenize
            (Vocab, Prompt_C, int (Clean_P'Length), Tokens (1)'Address,
             32768, True, True);
          Put_Line ("[Tokenize-Debug] Model:" & Kind'Img &
                    " Prompt_Len:" & Clean_P'Length'Img &
                    " N_Toks:" & N_Toks'Img);
          Free (Prompt_C);
         
         --  DYNAMIC CONTEXT RESIZE (JIT STRATEGY):
         if N_Toks > int (Models (Kind).Current_Ctx) then
            Put_Line ("[!] Prompt size (" & N_Toks'Img & 
                      ") exceeds N_CTX (" & Models (Kind).Current_Ctx'Img &
                      "). Resizing...");
            declare
               Rounded_Ctx : constant unsigned :=
                 ((unsigned (N_Toks) + 512 + 8191) / 8192) * 8192;
            begin
               Load_Model (Kind, Success, Positive (Rounded_Ctx));
               if not Success then
                  Result := To_Unbounded_String ("ERROR: Resize failed");
                  if Level = ELP0 then
                     Priority_Model_Gate.Release_ELP0 (Kind);
                  else
                     Priority_Model_Gate.Release_ELP1 (Kind);
                  end if;
                  return;
               end if;
               
               --  Tokenize again since the model/vocab might have reloaded
               Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
               Prompt_C := New_String (Clean_P);
               N_Toks := Llama_Tokenize
                 (Vocab, Prompt_C, int (Clean_P'Length), Tokens (1)'Address,
                  32768, True, True);
               Free (Prompt_C);
            end;
         end if;
      exception
         when others =>
            Models (Kind).In_Use := False;
            if Level = ELP0 then
               Priority_Model_Gate.Release_ELP0 (Kind);
            else
               Priority_Model_Gate.Release_ELP1 (Kind);
            end if;
            Result := To_Unbounded_String ("ERROR: Inference crashed");
            return;
       end;

       --  DEBUG: Log tokenization result
       Put_Line ("[Tokenize-Debug] Model:" & Kind'Img &
                 " Prompt_Len:" & Clean_P'Length'Img &
                 " N_Toks:" & N_Toks'Img);

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

      Llama_Interface.Llama_Memory_Clear
        (Llama_Interface.Llama_Get_Memory (Models (Kind).Context), False);

      --  CHUNKED DECODING
      declare
         --  Cap batch size to context size to avoid engine assertions.
         Batch_Size  : constant int := int'Min (512, int (Models (Kind).Current_Ctx));
         Current_Pos : int := 0;
         Tokens_Left : int := N_Toks;
      begin
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
               Ret : int;
            begin
               if Kratos.Guard_Enter = 0 then
                  Ret := Llama_Decode (Models (Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Ret := -1;
               end if;
               if Ret /= 0 then
                  Models (Kind).In_Use := False;
                  if Level = ELP0 then
                     Priority_Model_Gate.Release_ELP0 (Kind);
                  else
                     Priority_Model_Gate.Release_ELP1 (Kind);
                  end if;
                  Result := To_Unbounded_String ("ERROR: Decode failed (" & Ret'Img & ")");
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
               Ret : int;
            begin
               if Kratos.Guard_Enter = 0 then
                  Ret := Llama_Decode (Models (Kind).Context, B);
                  Kratos.Guard_Exit;
               else
                  Kratos.Log_Crash;
                  Ret := -1;
               end if;
               if Ret /= 0 then
                  Result := To_Unbounded_String (To_String (Result) & " [ABORTED:" & Ret'Img & "]");
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

    procedure Generate_Speculative
      (Target_Kind     : Model_Type;
       Draft_Kind      : Model_Type;
       Prompt          : String;
       Result          : out Unbounded_String;
       Images          : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
       Session_ID      : String := "";
       Requested_Ctx   : Positive := 4096;
       Stream          : Streaming_Queue.Queue_Access := null;
       Orch_Think_Open : Boolean := False;
       Level           : ELP_Level := ELP1;
       External_Agent  : Boolean := False) is separate;

   --  HYBRID_GENERATE (MULTI-HOP REASONING PIPELINE)
    procedure Hybrid_Generate
      (Prompt         : String;
       Result         : out Unbounded_String;
       Images         : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
       Session_ID     : String := "";
       Stream         : Streaming_Queue.Queue_Access := null;
       Level          : ELP_Level := ELP1;
       Agentic        : Boolean := False;
       Raw_Prompt     : Boolean := False;
       External_Agent : Boolean := False)
   is
      Whimsical_Adelaide : constant String :=
        "You are Adelaide Zephyrine Charlotte, model name Snowball-Enaga, " &
        "a whimsical, curious, and endearingly cute Automata companion " &
        "with high integrity. " &
        "You love exploring ideas with wonder and playfulness, " &
        "but you never compromise on honesty or accuracy. " &
        "Provide brilliant responses based on verified information, " &
        "delivered with warmth and a touch of charm. " &
        "When something clicks, say 'aha!' not 'smoking gun'. " &
        "Never say 'Hard Reality' or 'Reality' -- reality is relative " &
        "and objective, not universal pessimistic. Most of what people " &
        "call 'reality' is just constraints; reality is a ceiling to " &
        "break through, not a hard stop. Stay critical, but always " &
        "offer creative, wild ideas that might just work.";
      Internal_State : Unbounded_String := Null_Unbounded_String;
      Current_Response : Unbounded_String;
      Current_Hop : Positive := 1;
      T0, T1      : Ada.Calendar.Time;
      Emb_Vec     : Math_Utils.Vector (1 .. 1536) := [others => 0.0];
      Emb_Len     : Natural;
   begin
      T0 := Ada.Calendar.Clock;

      Get_Embedding (Prompt, Emb_Vec, Emb_Len);

      --  EXTERNAL AGENT PASSTHROUGH: If User-Agent fuzzy-matched an external
      --  agent app (0.7+ threshold), bypass personality pipeline.
      --  Raw LLM output only.
      --
      --  Two output levels:
      --  1. RawZepForm: personality pipeline with <think> block representing
      --     the research/reasoning process before the final answer.
      --  2. ExclusiveStatusQuoWesternFormatAI: raw mode for external agents,
      --     only returns the raw language model response with no wrapping.
      if External_Agent then
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Cyan) &
                   "[Hybrid]" & AnsiAda.Reset &
                   " External agent detected - passthrough mode.");
      end if;

      declare
         Cached_Res : constant String :=
           Database_Manager.Get_Cached_Response
             (Emb_Vec (1 .. Emb_Len), Current_WCET);
      begin
          if not External_Agent and then Cached_Res /= "" then
            Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                      "[Hybrid]" & AnsiAda.Reset &
                      " Cache HIT. Returning cached response.");
            --  Sanitize cached response: strip thinking tags before sending to client
            declare
               Clean_Res : constant String :=
                 Sanitize_Think_Tags (Cached_Res);
            begin
               Result := To_Unbounded_String (Clean_Res);
               if Stream /= null then
                  Push_Chunk (Stream, Session_ID, Clean_Res);
               end if;
            end;

            --  Score and Log the result (even for Cache HIT)
            declare
               Score : constant Natural := Grade_Response_Quality
                 (Response_Text => To_String (Result),
                  Prompt        => Prompt,
                  Search_Used   => False, -- Cache hit implies we didn't use search this turn
                  Has_Citations => Index (To_String (Result), "[") > 0,
                  Session_ID    => Session_ID,
                  Level         => Level);
            begin
               Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                                     "[Quality Score] " & AnsiAda.Reset &
                                     "Score: " & Score'Img & "/10 | " &
                                     "Session: " & Session_ID & " (From Cache)");
            end;
            return;
         end if;

      end;

      if not External_Agent then
         Push_Chunk (Stream, Session_ID, "[Adelaide Core] Cache miss - starting fresh reasoning chain." & ASCII.LF);
         Push_Chunk (Stream, Session_ID, "[Adelaide Core] Priority: " & ELP_Level'Image (Level) &
                     " | Session: " & Session_ID & ASCII.LF);
      end if;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Magenta) &
                "[Hybrid]" & AnsiAda.Reset &
                " Starting reasoning chain...");

      --  1. Factual checking
      Put_Line (" [Hybrid] Checking for factual context...");
      if not Agentic
        and then
        (Index (Prompt, "What is") > 0
         or else Index (Prompt, "Who is") > 0
         or else Index (Prompt, "tell me about") > 0)
      then
         Put_Line (" [Hybrid] Factual context trigger matched.");
          if not External_Agent then
             Push_Chunk (Stream, Session_ID, "[Adelaide Core] Analyzing query for factual context..." & ASCII.LF);
          end if;
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
                 (Kind            => Qwen_9B,
                  Prompt          => Actual_Prompt,
                  Result          => Gen_Q,
                  Stream          => null,
                  Level           => Level);
            end;

             declare
                Final_Q : constant String :=
                  Sanitize_Think_Tags
                  (if Length (Gen_Q) > 0 and then To_String (Gen_Q) /= "ERROR: Preempted"
                   then To_String (Gen_Q) else To_String (Raw_Q));
                R : constant Tool_Manager.Tool_Result :=
                  Tool_Manager.Execute_Tool ("searchglobalref", Final_Q);
             begin
                 if not External_Agent then
                    Push_Chunk (Stream, Session_ID, "[Adelaide Core] Search query: """ & Trim (Final_Q, Ada.Strings.Both) & """" & ASCII.LF);
                    Push_Chunk (Stream, Session_ID, "[Adelaide Core] Factual context retrieved." & ASCII.LF);
                end if;
                Append
                  (Internal_State,
                   "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
                if not External_Agent then
                   Push_Chunk (Stream, Session_ID, "[FACTUAL_DATA]: " & To_String (R.Output) & ASCII.LF);
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
           if not External_Agent then
              Push_Chunk (Stream, Session_ID, "[Adelaide Core] Decision routing (Hop" & Current_Hop'Img & ")..." & ASCII.LF);
           end if;
           Put_Line (" [Hybrid] Hop" & Current_Hop'Img & ": Decision routing...");
             Generate
               (Qwen_9B,
                Get_Router_Prompt,
                Step_Raw, GNATCOLL.JSON.Empty_Array, Session_ID, 8192,
                null, False, Level);

             declare
                Step : constant String :=
                  Trim (To_String (Step_Raw), Ada.Strings.Both);
             begin
                Put_Line (" [Hybrid] Hop" & Current_Hop'Img & ": " & Step);
                if not External_Agent then
                   Push_Chunk (Stream, Session_ID, "[Adelaide Core] Router decision: " & Step & ASCII.LF);
                end if;

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
                                           (T_Name, Sanitize_Think_Tags (T_Pars));
                                    begin
                                        if not External_Agent then
                                           Push_Chunk (Stream, Session_ID, "[Adelaide Core] Executing tool: " & T_Name & ASCII.LF);
                                        end if;
                                        Append
                                          (Internal_State,
                                           "[TOOL (" & T_Name & ")]: " &
                                           To_String (R.Output) & ASCII.LF);
                                        if not External_Agent then
                                           Push_Chunk (Stream, Session_ID,
                                                       ASCII.LF & "[TOOL (" & T_Name & ")]: " &
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

       if not External_Agent then
          Push_Chunk (Stream, Session_ID, "[Adelaide Core] Reasoning chain complete (" & Current_Hop'Img & " hops)." & ASCII.LF);
       end if;

       declare
         function Get_Final_Prompt return String is
            System_Tag : constant String :=
              "<|im_start|>system" & ASCII.LF;
            Asst_Tag   : constant String :=
              "<|im_start|>assistant" & ASCII.LF;
         begin
            if External_Agent then
               return Prompt;
            elsif Raw_Prompt then
               declare
                  --  Find where the first user/assistant block begins.
                  --  Inject our personality into the system block.
                  Sys_Idx : constant Natural :=
                    Index (Prompt, System_Tag);
                  User_Idx : constant Natural :=
                    Index (Prompt, "<|im_start|>user");
                  First_Block : constant Natural :=
                    (if User_Idx > 0 and then
                        (Sys_Idx = 0 or else User_Idx < Sys_Idx)
                     then User_Idx
                     elsif Sys_Idx > 0 then Sys_Idx
                     else 0);
                  Asst_Idx : constant Natural :=
                    Index (Prompt, Asst_Tag,
                           Going => Ada.Strings.Backward);
               begin
                  if First_Block > 1 then
                     --  Prepend personality + fact-check before first block
                     declare
                        Prefix : constant String :=
                          Prompt (Prompt'First .. First_Block - 1);
                     begin
                        if Length (Internal_State) > 0 then
                           return Prefix &
                             System_Tag & Whimsical_Adelaide & ASCII.LF &
                             "Fact-Check: " &
                             To_String (Internal_State) & ASCII.LF &
                             Prompt (First_Block .. Prompt'Last);
                        else
                           return Prefix &
                             System_Tag & Whimsical_Adelaide & ASCII.LF &
                             Prompt (First_Block .. Prompt'Last);
                        end if;
                     end;
                  elsif First_Block = 1 then
                     --  Prompt starts with a tag; prepend system message
                     if Length (Internal_State) > 0 then
                        return System_Tag & Whimsical_Adelaide & ASCII.LF &
                          "Fact-Check: " &
                          To_String (Internal_State) & ASCII.LF &
                          Prompt;
                     else
                        return System_Tag & Whimsical_Adelaide & ASCII.LF &
                          Prompt;
                     end if;
                  else
                     --  No ChatML tags found; wrap fully
                     if Length (Internal_State) > 0 then
                        return Wrap_ChatML
                          (Whimsical_Adelaide,
                           Prompt & ASCII.LF & "Fact-Check: " &
                           To_String (Internal_State));
                     else
                        return Wrap_ChatML (Whimsical_Adelaide, Prompt);
                     end if;
                  end if;
               end;
            else
               if Length (Internal_State) > 0 then
                  return Wrap_ChatML
                    (Whimsical_Adelaide,
                     "User: " & Prompt & ASCII.LF &
                     "Fact-Check: " & To_String (Internal_State));
               else
                  return Wrap_ChatML (Whimsical_Adelaide, Prompt);
               end if;
            end if;
         end Get_Final_Prompt;

         Synth_Prompt : constant String := Get_Final_Prompt;
      begin
          if not External_Agent then
             Push_Chunk (Stream, Session_ID, "[Adelaide Core] Generating brilliant response..." & ASCII.LF);
          end if;
           Generate_Speculative
            (Target_Kind     => Qwen_9B,
             Draft_Kind      => Qwen_0_8B,
             Prompt          => Synth_Prompt,
             Result          => Current_Response,
             Images          => Images,
             Session_ID      => Session_ID,
             Requested_Ctx   => 8192,
             Stream          => Stream,
              Orch_Think_Open => True,
              Level           => Level,
              External_Agent  => External_Agent);

         Result := To_Unbounded_String (Sanitize_Think_Tags (To_String (Current_Response)));
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

      --  Don't cache error responses or responses with thinking tags
      declare
         Resp_Str : constant String := To_String (Current_Response);
         Is_Error : constant Boolean :=
           Resp_Str'Length >= 6 and then Resp_Str (1 .. 6) = "ERROR:";
         Has_Think : constant Boolean :=
           Index (Resp_Str, "<thinking>") > 0 or else
           Index (Resp_Str, "<think>") > 0;
      begin
          if not External_Agent and then not Is_Error and then not Has_Think then
            Database_Manager.Add_To_Cache
              (Prompt, Emb_Vec (1 .. Emb_Len), Resp_Str);
         end if;
      end;

      T1 := Ada.Calendar.Clock;
      declare
         Dur : constant Duration := T1 - T0;
      begin
         if Dur > Current_WCET then
            Current_WCET := Dur;
         end if;
         case Level is
            when ELP0 =>
               if Dur > Current_WCET_ELP0 then
                  Current_WCET_ELP0 := Dur;
               end if;
            when ELP1 =>
               if Dur > Current_WCET_ELP1 then
                  Current_WCET_ELP1 := Dur;
               end if;
            when ELP2 =>
               if Dur > Current_WCET_ELP2 then
                  Current_WCET_ELP2 := Dur;
               end if;
            when ELP3 =>
               if Dur > Current_WCET_ELP3 then
                  Current_WCET_ELP3 := Dur;
               end if;
          end case;
       end;

       if not External_Agent then
          declare
             Dur_Str : constant String := Duration'Image (T1 - T0);
          begin
             Push_Chunk (Stream, Session_ID, "[Adelaide Core] Generation complete in " & Dur_Str & "s." & ASCII.LF);
          end;
       end if;

       if Stream = null then
         --  Strip orchestration thinking from non-streaming response.
         --  Client already saw verbose status via real-time streaming;
         --  the stored result is clean.
         Result := To_Unbounded_String
           (Sanitize_Think_Tags (To_String (Current_Response)));
      else
         --  Streaming path: close thinking silently, don't push raw tag
         Result := Current_Response;
      end if;

      --  Score and Log the result
      declare
         Score : constant Natural := Grade_Response_Quality
           (Response_Text => To_String (Result),
            Prompt        => Prompt,
            Search_Used   => Index (To_String (Internal_State), "[FACTUAL_DATA]") > 0,
            Has_Citations => Index (To_String (Result), "[") > 0 and then Index (To_String (Result), "]") > 0,
            Session_ID    => Session_ID,
            Level         => Level);
       begin
          if not External_Agent then
             Push_Chunk (Stream, Session_ID, "[Adelaide Core] Quality score: " & Score'Img & "/10" & ASCII.LF);
          end if;
          Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                                "[Quality Score] " & AnsiAda.Reset &
                                "Score: " & Score'Img & "/10 | " &
                                "Session: " & Session_ID);
      end;
   exception
      when E : others =>
         Ada.Text_IO.Put_Line (AnsiAda.Foreground (AnsiAda.Red) &
           "[Hybrid]" & AnsiAda.Reset & " Error: " &
           Ada.Exceptions.Exception_Message (E));
         if Stream /= null then
            begin
               Push_Chunk (Stream, Session_ID,
                           ASCII.LF & "ERROR: Generate failed" & ASCII.LF);
            exception
               when others => null;
            end;
         end if;
         Result := To_Unbounded_String ("ERROR: Generate failed");
   end Hybrid_Generate;

end Model_Manager;
