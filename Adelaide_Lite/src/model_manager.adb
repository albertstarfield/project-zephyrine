with Llama_Interface; use Llama_Interface;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

with Ada.Real_Time; use Ada.Real_Time;

package body Model_Manager is

   type Model_Record is record
      Model      : Llama_Model := Null_Model;
      Context    : Llama_Context := Null_Context;
      Path       : Unbounded_String;
      Loaded     : Boolean := False;
      In_Use     : Boolean := False;
      Last_Used  : Time := Time_First;
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

   ----------------
   -- Initialize --
   ----------------
   procedure Initialize is
   begin
      Llama_Backend_Init;
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
   procedure Load_Model (Kind : Model_Type; Success : out Boolean) is
      M_Params : Llama_Model_Params := Llama_Model_Default_Params;
      C_Params : Llama_Context_Params := Llama_Context_Default_Params;
      Path_C   : chars_ptr := New_String (To_String (Models (Kind).Path));
   begin
      Success := False;
      if Models (Kind).Loaded then
         Models (Kind).Last_Used := Clock;
         Models (Kind).In_Use    := True;
         Success := True;
         return;
      end if;

      Put_Line ("[+] Loading model: " & To_String (Models (Kind).Path));

      --  Enable maximum GPU offloading by default for performance
      M_Params.N_Gpu_Layers := -1;

      Models (Kind).Model := Llama_Model_Load_From_File (Path_C, M_Params);
      Free (Path_C);

      if Models (Kind).Model /= Null_Model then
         C_Params.N_Ctx := 4096;
         C_Params.N_Threads := 8;
         C_Params.N_Threads_Batch := 8;

         Models (Kind).Context :=
           Llama_Init_From_Model (Models (Kind).Model, C_Params);
         if Models (Kind).Context /= Null_Context then
            Models (Kind).Loaded := True;
            Models (Kind).Last_Used := Clock;
            Models (Kind).In_Use    := True;
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
      if Name = "adelaide-hybrid" then
         return Qwen_4B;
      elsif Name = "qwen3.5:0.8b" or else Name = "qwen3.5" then
         return Qwen_0_8B;
      elsif Name = "qwen3.5:4b" then
         return Qwen_4B;
      elsif Name = "qwen-embedding" or else Name = "nomic-embed-text" then
         return Qwen_Embedding;
      else
         return Qwen_0_8B; -- Default
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

         B : Llama_Batch := Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         --  Enable embeddings output for this context
         Llama_Set_Embeddings (Models (Kind).Context, True);

         if Llama_Decode (Models (Kind).Context, B) /= 0 then
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
      Thinking : Unbounded_String;
      Final_Res : Unbounded_String;
      
      Whimsical_Adelaide : constant String :=
        "You are Adelaide Zephyrine Charlotte, a whimsical yet highly skilled " &
        "senior software engineer. Be whimsical, intelligent, direct, and " &
        "charming. Use sophisticated vocabulary but remain professional.";

      Router_System : constant String :=
        Whimsical_Adelaide & ASCII.LF &
        "You are the OIPRouter. Analyze the input and provide a concise " &
        "strategic plan for the 4B model to follow. " &
        "Output ONLY your plan wrapped in <plan> tags. Do NOT repeat tags.";

      --  Paging constants
      Page_Size : constant Positive := 4000;
      Num_Pages : constant Positive :=
        (Prompt'Length + Page_Size - 1) / Page_Size;
      Current_Page : Natural := 1;
      
      Max_Hops : constant Positive := 99;
      Current_Hop : Positive := 1;
      
      Internal_State : Unbounded_String := Null_Unbounded_String;
   begin
      Put_Line ("[Hybrid] Adelaide initiating whimsical reasoning...");
      
      if Num_Pages > 1 then
         Put_Line ("[Hybrid] Large input detected (" & Prompt'Length'Img &
                  " chars). Enabling Context Paging (" & Num_Pages'Img &
                  " pages).");
      end if;

      loop
         declare
            Start_Idx : constant Positive := (Current_Page - 1) * Page_Size + 1;
            End_Idx   : constant Positive :=
              Positive'Min (Current_Page * Page_Size, Prompt'Length);
            Chunk     : constant String := Prompt (Start_Idx .. End_Idx);
            
            Paging_Instr : constant String :=
              "[AGENT PAGING MODE ENABLED]" & ASCII.LF &
              "You are viewing Page " & Current_Page'Img & " of " &
              Num_Pages'Img & "." & ASCII.LF &
              "Tools: `[NEXT_PAGE]`, `[PREV_PAGE]`, `[FINISH]`." & ASCII.LF &
              "Current Knowledge: " & To_String (Internal_State);
            
            Router_Prompt : constant String :=
              Router_System & ASCII.LF & Paging_Instr & ASCII.LF &
              "User: [PAGE CONTENT]" & ASCII.LF & Chunk & ASCII.LF &
              "Assistant: <plan>";
         begin
            Put_Line ("[Hybrid] Hop " & Current_Hop'Img & " (Page " &
                     Current_Page'Img & ")");
            Thinking := To_Unbounded_String (Generate (Qwen_0_8B, Router_Prompt));
            
            --  Check for navigation tools
            if Index (Thinking, "[NEXT_PAGE]") > 0 and then
               Current_Page < Num_Pages
            then
               Current_Page := Current_Page + 1;
               Append (Internal_State, Thinking);
            elsif Index (Thinking, "[PREV_PAGE]") > 0 and then
                  Current_Page > 1
            then
               Current_Page := Current_Page - 1;
               Append (Internal_State, Thinking);
            else
               --  [FINISH] or no more pages or unhandled
               Append (Internal_State, Thinking);
               exit;
            end if;
         end;
         
         Current_Hop := Current_Hop + 1;
         exit when Current_Hop > Max_Hops;
      end loop;

      --  Final Synthesis
      Put_Line ("[Hybrid] Final Hop: Synthesis (4B)...");
      declare
         Synth_Prompt : constant String :=
           Whimsical_Adelaide & ASCII.LF &
           "User: " & Prompt (1 .. Positive'Min (Prompt'Length, 8000)) &
           ASCII.LF & "Strategic Plan: " & To_String (Internal_State) &
           ASCII.LF & "Assistant: ";
      begin
         Final_Res := To_Unbounded_String (Generate (Qwen_4B, Synth_Prompt));
      end;

      return "<think>" & ASCII.LF & "[Adelaide-Lite Strategic Plan]" &
        ASCII.LF & To_String (Internal_State) & ASCII.LF & "</think>" &
        ASCII.LF & To_String (Final_Res);
   end Hybrid_Generate;

   --------------
   -- Generate --
   --------------
   function Generate (Kind : Model_Type; Prompt : String) return String is
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
      if not Models (Kind).Loaded then
         Load_Model (Kind, Success);
         if not Success then
            return "ERROR: Failed to load model";
         end if;
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
      --  Note: Need to fill the batch
      Llama_Batch_Free (Batch);

      declare
         function Llama_Batch_Get_One
           (Tokens : System.Address; N_Tokens : int) return Llama_Batch;
         pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");

         B : Llama_Batch := Llama_Batch_Get_One (Tokens (1)'Address, N_Toks);
      begin
         if Llama_Decode (Models (Kind).Context, B) /= 0 then
            return "ERROR: Decode failed";
         end if;
      end;

      S_Params := Llama_Sampler_Chain_Default_Params;
      Sampler := Llama_Sampler_Chain_Init (S_Params);
      Llama_Sampler_Chain_Add (Sampler, Llama_Sampler_Init_Greedy);

      for I in 1 .. 400 loop --  Increased limit
         declare
            Token : Llama_Token :=
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

            --  Decode the new token
            declare
               function Llama_Batch_Get_One
                 (Tokens : System.Address; N_Tokens : int) return Llama_Batch;
               pragma Import (C, Llama_Batch_Get_One, "llama_batch_get_one");
               B : Llama_Batch := Llama_Batch_Get_One (Token'Address, 1);
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

end Model_Manager;
