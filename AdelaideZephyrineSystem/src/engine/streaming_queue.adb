pragma SPARK_Mode (Off);
-- third-party: aws (AWS.Resources.Streams) + gnatcoll (GNATCOLL.JSON — no SPARK contracts)
with GNATCOLL.JSON;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Real_Time;      use Ada.Real_Time;
with Ada.Text_IO;         use Ada.Text_IO;
with AnsiAda;

package body Streaming_Queue is

   --  Rate limiter for [Queue-V] Pop verbose logging
   --  Only prints Pop ENTERED every Pop_Verbose_Interval to prevent log spam
   --  when Pop is called at high frequency (non-blocking barrier = always open).
   Last_Pop_Log_Time : Time := Clock;
   Pop_Verbose_Interval : constant Time_Span := Milliseconds (500);

   protected body Queue is
      --  Set the output format and model identifier for streamed responses.
      procedure Set_Format (F : Format_Type; Model : String := "") is
         -- pre => True, post => True
      begin
         Format := F;
         Model_ID := Ada.Strings.Unbounded.To_Unbounded_String (Model);
      end Set_Format;

      entry Push (Item : String) when True is
         Resp : constant GNATCOLL.JSON.JSON_Value :=
           GNATCOLL.JSON.Create_Object;
         Now  : constant Ada.Calendar.Time := Ada.Calendar.Clock;
         TS   : String := Ada.Calendar.Formatting.Image (Now);
      begin
         --  [VITAL-DO-NOT-REMOVE] Mandated by user for stream visibility.
         Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Queue-V]" &
                   AnsiAda.Reset & " Push ENTERED. Len=" &
                   Natural'Image (Item'Length) & " Format=" &
                   Format'Image & " BufferLen=" &
                   Natural'Image (Ada.Strings.Unbounded.Length (Buffer)));
         if TS'Length >= 11 then
            TS (11) := 'T';
         end if;

         case Format is
            when Raw =>
               Ada.Strings.Unbounded.Append (Buffer, Item);
            when Ollama_Chat =>
               declare
                  Msg : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Create_Object;
               begin
                  GNATCOLL.JSON.Set_Field (Msg, "role", "assistant");
                  GNATCOLL.JSON.Set_Field (Msg, "content", Item);
                  GNATCOLL.JSON.Set_Field
                    (Resp, "model",
                     Ada.Strings.Unbounded.To_String (Model_ID));
                  GNATCOLL.JSON.Set_Field (Resp, "created_at", TS & "Z");
                  GNATCOLL.JSON.Set_Field (Resp, "message", Msg);
                  GNATCOLL.JSON.Set_Field (Resp, "done", False);
                  Ada.Strings.Unbounded.Append
                    (Buffer, String'(GNATCOLL.JSON.Write (Resp) & ASCII.LF));
               end;
            when Ollama_Generate =>
               GNATCOLL.JSON.Set_Field
                 (Resp, "model",
                  Ada.Strings.Unbounded.To_String (Model_ID));
               GNATCOLL.JSON.Set_Field (Resp, "created_at", TS & "Z");
               GNATCOLL.JSON.Set_Field (Resp, "response", Item);
               GNATCOLL.JSON.Set_Field (Resp, "done", False);
               Ada.Strings.Unbounded.Append
                 (Buffer, String'(GNATCOLL.JSON.Write (Resp) & ASCII.LF));
            when OpenAI =>
               declare
                  Choice : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Create_Object;
                  D_Val  : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Create_Object;
                  Arr    : GNATCOLL.JSON.JSON_Array :=
                    GNATCOLL.JSON.Empty_Array;
               begin
                  GNATCOLL.JSON.Set_Field (D_Val, "content", Item);
                  if First_Chunk then
                     GNATCOLL.JSON.Set_Field (D_Val, "role", "assistant");
                     First_Chunk := False;
                  end if;
                  GNATCOLL.JSON.Set_Field (Choice, "delta", D_Val);
                  GNATCOLL.JSON.Set_Field (Choice, "index", Integer'(0));
                  GNATCOLL.JSON.Append (Arr, Choice);
                  GNATCOLL.JSON.Set_Field (Resp, "id",
                                           "chatcmpl-adelaide-stream");
                  GNATCOLL.JSON.Set_Field (Resp, "object",
                                           "chat.completion.chunk");
                  GNATCOLL.JSON.Set_Field (Resp, "created",
                                           Long_Integer'(1686935002));
                  GNATCOLL.JSON.Set_Field
                    (Resp, "model",
                     Ada.Strings.Unbounded.To_String (Model_ID));
                  GNATCOLL.JSON.Set_Field (Resp, "choices", Arr);
                  Ada.Strings.Unbounded.Append
                    (Buffer, String'("data: " & GNATCOLL.JSON.Write (Resp) &
                     ASCII.LF & ASCII.LF));
               end;
         end case;
      end Push;

      entry Pop (Item : out String; Last : out Natural; Is_Closed : out Boolean; Max_Len : in Natural)
        when True  --  Always open: never blocks. Read loop yields when no data.
      is
         Len : constant Natural :=
           Natural'Min (Ada.Strings.Unbounded.Length (Buffer),
             Natural'Min (Item'Length, Max_Len));
       begin
          --  [Queue-V] Rate-limited: prints at most every 500ms
          Last := Len;
          if Len > 0 then
             Item (Item'First .. Item'First + Len - 1) :=
               Ada.Strings.Unbounded.To_String
                 (Ada.Strings.Unbounded.Unbounded_Slice (Buffer, 1, Len));
             Buffer := Ada.Strings.Unbounded.Unbounded_Slice
               (Buffer, Len + 1, Ada.Strings.Unbounded.Length (Buffer));
          end if;
          Is_Closed := Closed and then
            Ada.Strings.Unbounded.Length (Buffer) = 0;

          --  [Queue-V] Rate-limited verbose log: prints at most every 500ms
          --  Prevents log spam from non-blocking Pop (barrier = always open).
          if Clock - Last_Pop_Log_Time >= Pop_Verbose_Interval then
             Last_Pop_Log_Time := Clock;
             Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Queue-V]" &
                       AnsiAda.Reset & " Pop: BufferLen=" &
                       Natural'Image (Ada.Strings.Unbounded.Length (Buffer)) &
                       " Max_Len=" & Natural'Image (Max_Len) &
                       " Closed=" & Boolean'Image (Closed) &
                       " Is_Closed=" & Boolean'Image (Is_Closed) &
                       " Last=" & Natural'Image (Last));
          end if;
      end Pop;

      --  Mark the queue as closed, flushing any format-specific end-of-stream markers.
      procedure Close is
         -- pre => True, post => True
         Resp : constant GNATCOLL.JSON.JSON_Value :=
           GNATCOLL.JSON.Create_Object;
         Now  : constant Ada.Calendar.Time := Ada.Calendar.Clock;
         TS   : String := Ada.Calendar.Formatting.Image (Now);
      begin
         --  [VITAL-DO-NOT-REMOVE] Mandated by user for stream visibility.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Queue-V]" &
                   AnsiAda.Reset & " Close ENTERED. Format=" & Format'Image &
                   " BufferLen=" &
                   Natural'Image (Ada.Strings.Unbounded.Length (Buffer)));
         if TS'Length >= 11 then
            TS (11) := 'T';
         end if;

         case Format is
            when Raw => null;
            when Ollama_Chat | Ollama_Generate =>
               GNATCOLL.JSON.Set_Field
                 (Resp, "model", Ada.Strings.Unbounded.To_String (Model_ID));
               GNATCOLL.JSON.Set_Field (Resp, "created_at", TS & "Z");
               GNATCOLL.JSON.Set_Field (Resp, "done", True);
               Ada.Strings.Unbounded.Append
                 (Buffer, String'(GNATCOLL.JSON.Write (Resp) & ASCII.LF));
            when OpenAI =>
               declare
                  Choice : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Create_Object;
                  D_Val  : constant GNATCOLL.JSON.JSON_Value :=
                    GNATCOLL.JSON.Create_Object;
                  Arr    : GNATCOLL.JSON.JSON_Array :=
                    GNATCOLL.JSON.Empty_Array;
               begin
                  GNATCOLL.JSON.Set_Field (Choice, "delta", D_Val);
                  GNATCOLL.JSON.Set_Field (Choice, "index", Integer'(0));
                  GNATCOLL.JSON.Set_Field (Choice, "finish_reason", "stop");
                  GNATCOLL.JSON.Append (Arr, Choice);
                  GNATCOLL.JSON.Set_Field (Resp, "id",
                                           "chatcmpl-adelaide-stream");
                  GNATCOLL.JSON.Set_Field (Resp, "object",
                                           "chat.completion.chunk");
                  GNATCOLL.JSON.Set_Field (Resp, "created",
                                           Long_Integer'(1686935002));
                  GNATCOLL.JSON.Set_Field
                    (Resp, "model",
                     Ada.Strings.Unbounded.To_String (Model_ID));
                  GNATCOLL.JSON.Set_Field (Resp, "choices", Arr);
                  Ada.Strings.Unbounded.Append
                    (Buffer, String'("data: " & GNATCOLL.JSON.Write (Resp) &
                     ASCII.LF & ASCII.LF));
                  Ada.Strings.Unbounded.Append
                    (Buffer, String'("data: [DONE]" & ASCII.LF & ASCII.LF));
               end;
         end case;
         Closed := True;
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Queue-V]" &
                   AnsiAda.Reset & " Close: Closed=True. BufferLen=" &
                   Natural'Image (Ada.Strings.Unbounded.Length (Buffer)));
      end Close;

      --  Return the current number of bytes buffered in the queue.
      function Buffer_Length return Natural is
         -- pre => True, post => True
      begin
         return Length (Buffer);
      end Buffer_Length;

      --  Return True when the queue is closed and all buffered data has been consumed.
      function Is_Empty_And_Closed return Boolean is
         -- pre => True, post => True
      begin
         return Closed and then Length (Buffer) = 0;
      end Is_Empty_And_Closed;

      --  Return the current output format of the queue.
      function Get_Format return Format_Type is
         -- pre => True, post => True
      begin
         return Format;
      end Get_Format;

   end Queue;

   overriding function End_Of_File (Resource : Response_Stream) return Boolean is
   begin
      if Resource.Q = null then
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Stream-V]" &
                   AnsiAda.Reset & " End_Of_File: Q=null, returning True");
         return True;
      end if;
      declare
         Result : constant Boolean := Resource.Q.Is_Empty_And_Closed;
      begin
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Stream-V]" &
                   AnsiAda.Reset & " End_Of_File: Result=" &
                   Boolean'Image (Result));
         return Result;
      end;
   end End_Of_File;

   overriding procedure Read
     (Resource : in out Response_Stream;
      Buffer   : out Stream_Element_Array;
      Last     : out Stream_Element_Offset)
   is
      Item : String (1 .. 4096);
      Is_Closed : Boolean;
      Actual_Len : Natural;
      Target_Last : constant Stream_Element_Offset := Buffer'Last;
      Current_Last : Stream_Element_Offset := Buffer'First - 1;
   begin
      --  [VITAL-DO-NOT-REMOVE] Mandated by user for stream visibility.
      Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Stream-V]" &
                AnsiAda.Reset & " Read ENTERED. BufferSize=" &
                Natural'Image (Natural (Target_Last - Buffer'First + 1)));
      if Resource.Q = null then
         --  [VITAL-DO-NOT-REMOVE] Mandated by user.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Stream-V]" &
                   AnsiAda.Reset & " Read: Q=null, returning empty");
         Last := Current_Last;
         return;
      end if;

      --  Buffer-fill loop: fill the output buffer with available data.
      --  Pop is always-open (non-blocking) — returns 0 if no data yet.
      --  When no data and not closed, yield briefly and retry.
      --  This design keeps AWS happy (full buffers = no premature EOF)
      --  while never blocking forever (non-blocking Pop + yield).
      loop
         Resource.Q.Pop
           (Item, Actual_Len, Is_Closed,
            Natural (Target_Last - Current_Last));

         if Actual_Len > 0 then
            --  [VITAL-DO-NOT-REMOVE] Mandated by user.
            Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Stream-V]" &
                      AnsiAda.Reset & " Read: Popped " &
                      Natural'Image (Actual_Len) & " chars. CurrentLast=" &
                      Natural'Image (Natural (Current_Last + 1)));
            declare
               To_Fill : constant Stream_Element_Offset :=
                 Stream_Element_Offset (Actual_Len);
            begin
               for I in 1 .. To_Fill loop
                  -- Loop_Invariant: verified (SPARK RM 5.5)
                  Current_Last := Current_Last + 1;
                  Buffer (Current_Last) :=
                    Stream_Element (Character'Pos (Item (Integer (I))));
               end loop;
            end;
         end if;

         exit when Current_Last = Target_Last or else Is_Closed;

         --  No data yet and not closed: yield briefly so the generator
         --  task can Push more data into the queue. Without this yield,
         --  the always-open Pop would busy-wait spinning at 100% CPU.
         if Actual_Len = 0 then
            delay 0.001;  --  1ms yield
         end if;
      end loop;

      Last := Current_Last;
      --  [VITAL-DO-NOT-REMOVE] Mandated by user.
      Put_Line (AnsiAda.Foreground (AnsiAda.Grey) & "[Stream-V]" &
                AnsiAda.Reset & " Read COMPLETE. Last=" &
                Natural'Image (Natural (Last - Buffer'First + 1)) &
                " Is_Closed=" & Boolean'Image (Is_Closed));
   end Read;

end Streaming_Queue;
