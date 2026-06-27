pragma SPARK_Mode (On);

with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;
with Math_Utils;
with Model_Manager;
with Model_Types;
with Database_Manager;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package body Embedding_Batcher is

   --  [INTERNAL STATE] Bounded buffers to avoid non-SPARK dynamic containers
   --  Sizing for Max_Batch_Size prompts.
   type Prompt_Buffer_Type is array (1 .. Max_Batch_Size) of Unbounded_String;
   type Name_Buffer_Type is array (1 .. Max_Batch_Size) of String (1 .. 256);
   type Name_Len_Type is array (1 .. Max_Batch_Size) of Natural;

    Buffer      : Prompt_Buffer_Type := [others => Null_Unbounded_String];
    Names       : Name_Buffer_Type    := [others => [others => ' ']];
    Name_Lens   : Name_Len_Type       := [others => 0];
   Current_Idx : Natural := 0;

   --  Internal procedure to dispatch the current buffer to the Tensor Accelerator
    procedure Dispatch_Batch (Level : Model_Types.ELP_Level) is
       Prompts : Math_Utils.Prompt_List (1 .. Max_Batch_Size) := [others => Null_Unbounded_String];
       Results : Math_Utils.Embedding_Vector_List (1 .. Max_Batch_Size) := [others => [others => 0.0]];
       Lengths : Math_Utils.Natural_List (1 .. Max_Batch_Size) := [others => 0];
    begin
       if Current_Idx = 0 then
          return;
       end if;
 
       Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Batcher]" & AnsiAda.Reset & 
                 " Dispatching batch of " & Current_Idx'Img & " requests to Tensor Accelerator...");
 
       -- Copy buffer to the arrays required by Model_Manager
       for I in 1 .. Current_Idx loop
          Prompts (I) := Buffer (I);
       end loop;
 
       --  [NON-PROVABLE BOUNDARY] 
       --  The following call enters Model_Manager, which is a bridge to C/Metal.
       --  This is the designated point where SPARK proofs transition to trust.
       Model_Manager.Get_Embeddings_Batch (Prompts, Results, Lengths, Level);
 
       --  Index results into the knowledge base
       for I in 1 .. Current_Idx loop
          if Lengths (I) > 0 then
             declare
                Actual_Name : String (1 .. Name_Lens (I));
             begin
                Actual_Name := Names (I) (1 .. Name_Lens (I));
                 Database_Manager.Add_Literature_Chunk (Actual_Name, To_String (Buffer (I)), Math_Utils.Vector (Results (I)), "hash");
             end;
          end if;
       end loop;
 
       -- Reset buffer
       for I in 1 .. Max_Batch_Size loop
          Buffer (I) := Null_Unbounded_String;
       end loop;
       Current_Idx := 0;
       
       Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[Batcher]" & AnsiAda.Reset & " Batch processed and indexed successfully.");
    end Dispatch_Batch;

   procedure Add_To_Batch (Prompt : String; File_Name : String; Level : Model_Types.ELP_Level) is
      L : constant Natural := File_Name'Length;
   begin
      Current_Idx := Current_Idx + 1;
      --  [DO NOT REMOVE COMMENT EXPLANATION]
      --  FIX 3: Hardware-Aware Padding (Misaligned Buffers)
      --  Pad the text chunk with spaces to ensure the final token count
      --  and memory footprint are aligned to Metal GPU SIMD group boundaries (multiples of 32)
      declare
          Pad_Len : constant Natural := (32 - (Prompt'Length mod 32)) mod 32;
      begin
          if Pad_Len = 0 then
              Buffer (Current_Idx) := To_Unbounded_String (Prompt);
          else
              declare
                  Padded_Prompt : String (1 .. Prompt'Length + Pad_Len);
              begin
                  Padded_Prompt (1 .. Prompt'Length) := Prompt;
                  for I in Prompt'Length + 1 .. Padded_Prompt'Last loop
                      Padded_Prompt (I) := ' ';
                  end loop;
                  Buffer (Current_Idx) := To_Unbounded_String (Padded_Prompt);
              end;
          end if;
      end;
      
      if L > 256 then
         Names (Current_Idx) := File_Name (File_Name'First .. File_Name'First + 255);
         Name_Lens (Current_Idx) := 256;
      else
         Names (Current_Idx) := [others => ' '];
         Names (Current_Idx) (1 .. L) := File_Name;
         Name_Lens (Current_Idx) := L;
      end if;

      if Current_Idx >= Max_Batch_Size then
         Dispatch_Batch (Level);
      end if;
   end Add_To_Batch;

   procedure Flush_Batch (Level : Model_Types.ELP_Level) is
   begin
      if Current_Idx > 0 then
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[Batcher]" & AnsiAda.Reset & " Flushing remaining " & Current_Idx'Img & " requests...");
         Dispatch_Batch (Level);
      end if;
   end Flush_Batch;

end Embedding_Batcher;
