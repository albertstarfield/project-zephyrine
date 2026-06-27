pragma SPARK_Mode (On);

with Math_Utils;
with Model_Manager;
with Model_Types;
with Database_Manager;

package Embedding_Batcher is

   -- Maximum batch size to prevent Tensor Accelerator command buffer churn
   Max_Batch_Size : constant Natural := 32;

   -- Adds a prompt to the buffer and triggers a batch process if the buffer is full
   procedure Add_To_Batch
     (Prompt : String;
      File_Name : String;
      Level : Model_Types.ELP_Level);

   -- Forces the remaining buffered prompts to be processed
   procedure Flush_Batch
     (Level : Model_Types.ELP_Level);

end Embedding_Batcher;
