with Ada.Text_IO; use Ada.Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Containers.Doubly_Linked_Lists;
with Model_Manager;

package body Scheduler_Manager is

   type Scheduled_Event is record
      Trigger_Time : Time;
      Prompt       : Unbounded_String;
   end record;

   package Event_Lists is new Ada.Containers.Doubly_Linked_Lists (Scheduled_Event);
   use Event_Lists;

   protected Event_Queue is
      procedure Add (Item : Scheduled_Event);
      procedure Get_Next (Item : out Scheduled_Event; Found : out Boolean);
   private
      List : Event_Lists.List;
   end Event_Queue;

   protected body Event_Queue is
      procedure Add (Item : Scheduled_Event) is
      begin
         List.Append (Item);
      end Add;

      procedure Get_Next (Item : out Scheduled_Event; Found : out Boolean) is
         Cur : Cursor := List.First;
         Now : constant Time := Clock;
      begin
         Found := False;
         while Has_Element (Cur) loop
            if Element (Cur).Trigger_Time <= Now then
               Item := Element (Cur);
               List.Delete (Cur);
               Found := True;
               return;
            end if;
            Cur := Next (Cur);
         end loop;
      end Get_Next;
   end Event_Queue;

   task type Scheduler_Task_Type is
      pragma Storage_Size (8 * 1024 * 1024); -- 8MB Stack
   end Scheduler_Task_Type;
   type Scheduler_Task_Access is access Scheduler_Task_Type;
   Worker : Scheduler_Task_Access := null;

   task body Scheduler_Task_Type is
      Evt : Scheduled_Event;
      Has_Evt : Boolean;
      LLM_Result : Unbounded_String;
   begin
      loop
         Event_Queue.Get_Next (Evt, Has_Evt);
         if Has_Evt then
            Put_Line ("[Scheduler] Triggering proactive thought: " & To_String (Evt.Prompt));
            Model_Manager.Hybrid_Generate
              (Prompt     => To_String (Evt.Prompt),
               Result     => LLM_Result,
               Session_ID => "server-scheduler",
               Agentic    => True,
               Level      => Model_Manager.ELP0);
         else
            delay 1.0;
         end if;
      end loop;
   end Scheduler_Task_Type;

   procedure Initialize is
   begin
      if Worker = null then
         Worker := new Scheduler_Task_Type;
      end if;
   end Initialize;

   procedure Schedule (Delay_Seconds : Integer; Prompt : String) is
      Evt : Scheduled_Event;
   begin
      Evt.Trigger_Time := Clock + Seconds (Delay_Seconds);
      Evt.Prompt := To_Unbounded_String (Prompt);
      Event_Queue.Add (Evt);
      Put_Line ("[Scheduler] Scheduled proactive thought in" & Delay_Seconds'Img & " seconds.");
   end Schedule;

end Scheduler_Manager;
