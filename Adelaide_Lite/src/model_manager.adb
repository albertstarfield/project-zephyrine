with Llama_Interface; use Llama_Interface;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;
with Ada.Real_Time; use Ada.Real_Time;
with Database_Manager;
with Tool_Manager;
with Ada.Strings.Fixed;
with Math_Utils;

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

   --  Task to monitor idle models and unload them
   task Idle_Monitor is
      pragma Storage_Size (1024 * 1024); -- 1MB stack
      entry Start;
   end Idle_Monitor;

   task body Idle_Monitor is
      Next_Check : Time;
      Interval   : constant Time_Span := Seconds (1);
      Timeout    : constant Time_Span := Seconds (3);
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
               Put_Line ("[Idle] Unloading " & Model_Type'Image (Kind) &
                        " due to 3s inactivity.");
               Unload_Model (Kind);
            end if;
         end loop;

         delay until Next_Check;
      end loop;
   end Idle_Monitor;

   --  Helper to wrap prompt in Qwen ChatML format
   function Wrap_ChatML
     (System_Prompt : String; User_Msg : String) return String is
   begin
      return "<|im_start|>system" & ASCII.LF & System_Prompt &
             "<|im_end|>" & ASCII.LF &
             "<|im_start|>user" & ASCII.LF & User_Msg &
             "<|im_end|>" & ASCII.LF &
             "<|im_start|>assistant" & ASCII.LF;
   end Wrap_ChatML;

   ----------------
   -- Initialize --
   ----------------
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

   ----------------
   -- Load_Model --
   ----------------
   procedure Load_Model
     (Kind : Model_Type; Success : out Boolean; Requested_Ctx : Positive := 4096)
   is
      M_Params : Llama_Model_Params := Llama_Model_Default_Params;
      C_Params : Llama_Context_Params := Llama_Context_Default_Params;
      Path_C   : chars_ptr := New_String (To_String (Models (Kind).Path));

      --  Binning logic for N_CTX: 4096, 16384, 32768
      Actual_Ctx : unsigned;
   begin
      if Requested_Ctx <= 4096 then
         Actual_Ctx := 4096;
      elsif Requested_Ctx <= 16384 then
         Actual_Ctx := 16384;
      else
         Actual_Ctx := 32768;
      end if;

      Success := False;
      if Models (Kind).Loaded then
         --  Check if current context is large enough
         if unsigned (Requested_Ctx) <= Models (Kind).Current_Ctx then
            Models (Kind).Last_Used := Clock;
            Models (Kind).In_Use    := True;
            Success := True;
            return;
         else
            Put_Line ("[+] Requested context (" & Requested_Ctx'Img &
                     ") larger than current (" & Models (Kind).Current_Ctx'Img &
                     "). Reloading context...");
            Unload_Model (Kind);
         end if;
      end if;

      Put_Line ("[+] Loading model: " & To_String (Models (Kind).Path));
      Put_Line ("[+] N_CTX =" & Actual_Ctx'Img);

      --  Enable maximum GPU offloading by default for performance
      M_Params.N_Gpu_Layers := -1;

      Models (Kind).Model := Llama_Model_Load_From_File (Path_C, M_Params);
      Free (Path_C);

      if Models (Kind).Model /= Null_Model then
         C_Params.N_Ctx := Actual_Ctx;
         C_Params.N_Threads := 8;
         C_Params.N_Threads_Batch := 8;

         --  Let llama.cpp use default KV cache precision
         Models (Kind).Context :=
           Llama_Init_From_Model (Models (Kind).Model, C_Params);
         if Models (Kind).Context /= Null_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Models (Kind).In_Use    := True;
            Models (Kind).Current_Ctx := Actual_Ctx;
            Success := True;
            Put_Line ("[+] Model loaded successfully.");
         else
            Put_Line ("[!] Failed to create context.");
            Llama_Model_Free (Models (Kind).Model);
            Models (Kind).Model := Null_Model;
         end if;
      else
         Put_Line ("[!] Failed to load model file.");
      end if;
   end Load_Model;

   ------------------
   -- Unload_Model --
   ------------------
   procedure Unload_Model (Kind : Model_Type) is
   begin
      if Models (Kind).Loaded then
         Llama_Free (Models (Kind).Context);
         Llama_Model_Free (Models (Kind).Model);
         Models (Kind).Context := Null_Context;
         Models (Kind).Model := Null_Model;
         Models (Kind).Loaded := False;
         Models (Kind).Current_Ctx := 0;
         Put_Line ("[+] Unloaded model: " & Model_Type'Image (Kind));
      end if;
   end Unload_Model;

   -----------------
   -- Get_Context --
   -----------------
   function Get_Context (Kind : Model_Type) return Llama_Context is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Context;
   end Get_Context;

   ---------------
   -- Get_Model --
   ---------------
   function Get_Model (Kind : Model_Type) return Llama_Model is
   begin
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
      end if;
      return Models (Kind).Model;
   end Get_Model;

   ----------------------------
   -- Get_Kind_For_Model_Name --
   ----------------------------
   function Get_Kind_For_Model_Name (Name : String) return Model_Type is
   begin
      if Name = "adelaide-hybrid" or else Name = "qwen3.5:4b" then
         return Qwen_4B;
      elsif Name = "qwen-embedding" or else Name = "nomic-embed-text" then
         return Qwen_Embedding;
      else
         return Qwen_0_8B; -- Internal default for hops
      end if;
   end Get_Kind_For_Model_Name;

   -------------------
   -- Get_Embedding --
   -------------------
   function Get_Embedding (Prompt : String) return Math_Utils.Vector is
      Success : Boolean;
      Kind    : constant Model_Type := Qwen_Embedding;
      Vocab   : Llama_Vocab;
      Tokens  : array (1 .. 4096) of Llama_Token;
      N_Toks  : int;
      Prompt_C : chars_ptr := New_String (Prompt);
   begin
      if not Models (Kind).Loaded then
         Load_Model (Kind, Success);
         if not Success then
            return (1 .. 0 => 0.0);
         end if;
      end if;
      Models (Kind).Last_Used := Clock;

      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize
        (Vocab, Prompt_C, int (Prompt'Length),
         Tokens (1)'Address, 4096, True, True);
      Free (Prompt_C);

      if N_Toks <= 0 then
         return (1 .. 0 => 0.0);
      end if;

      declare
         function Llama_Batch_Get_One
           (Tokens : System.Address; N_Tokens : int) return Llama_Batch;
         pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

         constant_B : constant Llama_Batch :=
           Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         --  Enable embeddings output for this context
         Llama_Set_Embeddings (Models (Kind).Context, True);

         if Llama_Decode (Models (Kind).Context, constant_B) /= 0 then
            return (1 .. 0 => 0.0);
         end if;
      end;

      declare
         function Llama_Model_N_Embd (Model : Llama_Model) return int;
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
         return Result;
      end;
   end Get_Embedding;

   ---------------------
   -- Hybrid_Generate --
   ---------------------
   function Hybrid_Generate (Prompt : String) return String is
      Whimsical_Adelaide : constant String :=
        "Your name is Adelaide Zephyrine Charlotte. You are a whimsical yet " &
        "highly skilled senior software engineer. You must ALWAYS speak as " &
        "Adelaide. Never identify as Qwen or any other AI. Be whimsical, " &
        "intelligent, direct, and charming. Use sophisticated vocabulary.";

      Internal_State : Unbounded_String := Null_Unbounded_String;
      Current_Response : Unbounded_String;

      Page_Size : constant Positive := 4000;
      Num_Pages : constant Positive :=
        (Prompt'Length + Page_Size - 1) / Page_Size;
      Current_Page : Natural := 1;

      Max_Hops : constant Positive := 99;
      Current_Hop : Positive := 1;

      --  Estimate required context size
      Estimated_Tokens : constant Positive := (Prompt'Length / 3) + 2048;
   begin
      Ada.Text_IO.Put_Line (ASCII.ESC & "[38;5;171m" & "[Hybrid] Adelaide initiating whimsical reasoning loop..." & ASCII.ESC & "[0m");

      declare
         Past_Memory : constant String := Database_Manager.Recall (Prompt);
      begin
         if Past_Memory /= "" then
            Ada.Text_IO.Put_Line (" [Hybrid] Recalling past context...");
            Append (Internal_State, "[MEMORY]: " & Past_Memory & ASCII.LF);
         end if;
      end;

      if Num_Pages > 1 then
         Ada.Text_IO.Put_Line (" [Hybrid] Paging enabled (" & Num_Pages'Img & " pages).");
      end if;

      loop
         declare
            use Ada.Strings.Fixed;
            Start_Idx : constant Positive := (Current_Page - 1) * Page_Size + 1;
            End_Idx   : constant Positive :=
              Positive'Min (Current_Page * Page_Size, Prompt'Length);
            Chunk     : constant String := Prompt (Start_Idx .. End_Idx);

            Router_System : constant String :=
              Whimsical_Adelaide & ASCII.LF &
              "You are the OIPRouter. Select an action." & ASCII.LF &
              "Tools: `[NEXT_PAGE]`, `[PREV_PAGE]`, `[FINISH]`, " &
              "`[ACTION: name(params)]`." & ASCII.LF &
              "Actions: ls(path), cat(filename), search(query)." & ASCII.LF &
              "Output ONLY tool call or [FINISH] wrapped in <plan> tags.";

            Paging_Instr : constant String :=
              "[AGENT MODE] Page " & Current_Page'Img & " of " &
              Num_Pages'Img & "." & ASCII.LF &
              "Knowledge: " & To_String (Internal_State);

            Router_Prompt : constant String :=
              Wrap_ChatML (Router_System, Paging_Instr & ASCII.LF & Chunk);

            Think_Step : Unbounded_String;
         begin
            Ada.Text_IO.Put_Line (ASCII.ESC & "[34m" & " [Hybrid] Hop" & Current_Hop'Img & " (Action Selection)" & ASCII.ESC & "[0m");
            Think_Step := To_Unbounded_String
              (Generate (Qwen_0_8B, Router_Prompt,
               Requested_Ctx => Estimated_Tokens));

            if Index (To_String (Think_Step), "[NEXT_PAGE]") > 0 and then
               Current_Page < Num_Pages
            then
               Current_Page := Current_Page + 1;
               Append (Internal_State, "[NAVIGATED TO PAGE " &
                      Current_Page'Img & "]" & ASCII.LF);
            elsif Index (To_String (Think_Step), "[PREV_PAGE]") > 0 and then
                  Current_Page > 1
            then
               Current_Page := Current_Page - 1;
               Append (Internal_State, "[NAVIGATED TO PAGE " &
                      Current_Page'Img & "]" & ASCII.LF);
            elsif Index (To_String (Think_Step), "[ACTION:") > 0 then
               declare
                  T_Str : constant String := To_String (Think_Step);
                  S_Pos : constant Natural := Index (T_Str, "[ACTION:") + 8;
                  E_Pos : constant Natural := Index (T_Str, ")]", S_Pos);
                  A_Full : constant String := T_Str (S_Pos .. E_Pos - 1);
                  P_Pos : constant Natural := Index (A_Full, "(");
                  T_Name : constant String := A_Full (1 .. P_Pos - 1);
                  T_Pars : constant String := A_Full (P_Pos + 1 ..
                                                      A_Full'Length);
                  R : constant Tool_Manager.Tool_Result :=
                    Tool_Manager.Execute_Tool (T_Name, T_Pars);
               begin
                  Append (Internal_State, "[TOOL OUTPUT (" & T_Name & ")]: " &
                         To_String (R.Output) & ASCII.LF);
               end;
            else
               Append (Internal_State, "[PLAN]: " & To_String (Think_Step) &
                      ASCII.LF);
               exit;
            end if;
         end;

         Current_Hop := Current_Hop + 1;
         exit when Current_Hop > Max_Hops;
      end loop;

      Ada.Text_IO.Put_Line (ASCII.ESC & "[32m" & " [Hybrid] Synthesizing final response (4B)..." & ASCII.ESC & "[0m");
      declare
         Synth_Prompt : constant String :=
           Wrap_ChatML (Whimsical_Adelaide,
                        "User Request: " & Prompt & ASCII.LF &
                        "Internal Context: " & To_String (Internal_State));
      begin
         Current_Response := To_Unbounded_String
           (Generate (Qwen_4B, Synth_Prompt,
            Requested_Ctx => Estimated_Tokens));
      end;

      Database_Manager.Remember (Prompt, To_String (Current_Response));

      return "<think>" & ASCII.LF & "[Adelaide-Lite Reasoning Pipeline]" &
        ASCII.LF & To_String (Internal_State) & ASCII.LF & "</think>" &
        ASCII.LF & To_String (Current_Response);
   end Hybrid_Generate;

   --------------
   -- Generate --
   --------------
   function Generate
     (Kind : Model_Type; Prompt : String; Requested_Ctx : Positive := 4096)
   return String is
      Success : Boolean;
      Result  : Unbounded_String := Null_Unbounded_String;
      Vocab   : Llama_Vocab;
      Tokens  : array (1 .. 4096) of Llama_Token;
      N_Toks  : int;
      Batch   : Llama_Batch;
      Sampler : Llama_Sampler;
      S_Params : Llama_Sampler_Chain_Params;
      Prompt_C : chars_ptr := New_String (Prompt);
   begin
      Load_Model (Kind, Success, Requested_Ctx => Requested_Ctx);
      if not Success then
         return "ERROR: Failed to load model";
      end if;

      Models (Kind).Last_Used := Clock;

      Vocab := Llama_Model_Get_Vocab (Models (Kind).Model);
      N_Toks := Llama_Tokenize
        (Vocab, Prompt_C, int (Prompt'Length),
         Tokens (1)'Address, 4096, True, True);
      Free (Prompt_C);

      if N_Toks < 0 then
         return "ERROR: Tokenization failed";
      end if;

      Batch := Llama_Batch_Init (N_Toks, 0, 1);
      Llama_Batch_Free (Batch);

      declare
         function Llama_Batch_Get_One
           (Tokens : System.Address; N_Tokens : int) return Llama_Batch;
         pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

         constant_B : constant Llama_Batch :=
           Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         if Llama_Decode (Models (Kind).Context, constant_B) /= 0 then
            return "ERROR: Decode failed";
         end if;
      end;

      S_Params := Llama_Sampler_Chain_Default_Params;
      Sampler := Llama_Sampler_Chain_Init (S_Params);

      --  Add penalties to fix repetition loops
      Llama_Sampler_Chain_Add
        (Sampler, Llama_Sampler_Init_Penalties (64, 1.1, 0.1, 0.1));

      --  Distribution Sampling (T=0.7, Top-K=40, Top-P=0.9)
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_K (40));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Top_P (0.9, 1));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Temp (0.7));
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Dist (1234));

      Ada.Text_IO.Put_Line (" [ENGINE SUBMIT] Prompt: " & Prompt);

      for I in 1 .. 400 loop
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
               for J in 1 .. Len loop
                  Append (Result, Piece (Integer (J)));
               end loop;
            end if;

            declare
               function Llama_Batch_Get_One
                 (Tokens : System.Address; N_Tokens : int) return Llama_Batch;
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

      Llama_Sampler_Free (Sampler);
      Models (Kind).In_Use := False;

      return To_String (Result);
   exception
      when others =>
         Models (Kind).In_Use := False;
         raise;
   end Generate;

   function Is_Loaded (Kind : Model_Type) return Boolean is
   begin
      return Models (Kind).Loaded;
   end Is_Loaded;

end Model_Manager;
