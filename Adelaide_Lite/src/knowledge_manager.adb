with Ada.Text_IO; use Ada.Text_IO;
with Model_Manager;
with Database_Manager;

package body Knowledge_Manager is

   task Indexing_Task is
      entry Start;
   end Indexing_Task;

   task Thought_Task is
      entry Start;
   end Thought_Task;

   procedure Initialize is
   begin
      Put_Line ("[Knowledge] Initializing Knowledge Base...");
      Database_Manager.Initialize;
   end Initialize;

   procedure Start_Tasks is
   begin
      Indexing_Task.Start;
      Thought_Task.Start;
   end Start_Tasks;

   procedure Process_File (Path : String) is
   begin
      Put_Line ("[Knowledge] Indexing: " & Path);
   end Process_File;

   task body Indexing_Task is
   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            Put_Line ("[Knowledge] Indexing preempted.");
            delay 1.0;
         else
            delay 5.0;
         end if;
      end loop;
   end Indexing_Task;

   task body Thought_Task is
   begin
      accept Start;
      loop
         if Model_Manager.Should_Abort_ELP0 then
            Put_Line ("[Knowledge] Thinking preempted.");
            delay 1.0;
         else
            delay 10.0;
         end if;
      end loop;
   end Thought_Task;

end Knowledge_Manager;
