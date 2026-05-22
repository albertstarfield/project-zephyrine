with GNATCOLL.JSON;
with Ada.Calendar;
with Ada.Calendar.Formatting;
with Ada.Strings.Unbounded;

package body Streaming_Queue is

   protected body Queue is
      procedure Set_Format (F : Format_Type; Model : String := "") is
      begin
         Format := F;
         Model_ID := Ada.Strings.Unbounded.To_Unbounded_String (Model);
      end Set_Format;

      entry Push (Item : String) when True is
         Resp : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
         Now  : constant Ada.Calendar.Time := Ada.Calendar.Clock;
         TS   : String := Ada.Calendar.Formatting.Image (Now);
      begin
         if TS'Length >= 11 then
            TS (11) := 'T';
         end if;

         case Format is
            when Raw =>
               Ada.Strings.Unbounded.Append (Buffer, Item);
            when Ollama =>
               declare
                  Msg : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
               begin
                  GNATCOLL.JSON.Set_Field (Msg, "role", "assistant");
                  GNATCOLL.JSON.Set_Field (Msg, "content", Item);
                  GNATCOLL.JSON.Set_Field (Resp, "model", Ada.Strings.Unbounded.To_String (Model_ID));
                  GNATCOLL.JSON.Set_Field (Resp, "created_at", TS & "Z");
                  GNATCOLL.JSON.Set_Field (Resp, "message", Msg);
                  GNATCOLL.JSON.Set_Field (Resp, "done", False);
                  Ada.Strings.Unbounded.Append (Buffer, GNATCOLL.JSON.Write (Resp) & ASCII.LF);
               end;
            when OpenAI =>
               declare
                  Choice : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
                  D_Val  : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
                  Arr    : GNATCOLL.JSON.JSON_Array := GNATCOLL.JSON.Empty_Array;
               begin
                  GNATCOLL.JSON.Set_Field (D_Val, "content", Item);
                  GNATCOLL.JSON.Set_Field (Choice, "delta", D_Val);
                  GNATCOLL.JSON.Set_Field (Choice, "index", Integer'(0));
                  GNATCOLL.JSON.Append (Arr, Choice);
                  GNATCOLL.JSON.Set_Field (Resp, "id", "chatcmpl-adelaide-stream");
                  GNATCOLL.JSON.Set_Field (Resp, "object", "chat.completion.chunk");
                  GNATCOLL.JSON.Set_Field (Resp, "created", Long_Integer'(1686935002));
                  GNATCOLL.JSON.Set_Field (Resp, "model", Ada.Strings.Unbounded.To_String (Model_ID));
                  GNATCOLL.JSON.Set_Field (Resp, "choices", Arr);
                  Ada.Strings.Unbounded.Append (Buffer, "data: " & GNATCOLL.JSON.Write (Resp) & ASCII.LF);
               end;
         end case;
      end Push;

      entry Pop (Item : out String; Last : out Natural; Is_Closed : out Boolean)
        when Ada.Strings.Unbounded.Length (Buffer) > 0 or else Closed
      is
         Len : constant Natural := Natural'Min (Ada.Strings.Unbounded.Length (Buffer), Item'Length);
      begin
         Last := Len;
         if Len > 0 then
            Item (Item'First .. Item'First + Len - 1) := 
              Ada.Strings.Unbounded.To_String (Ada.Strings.Unbounded.Unbounded_Slice (Buffer, 1, Len));
            Buffer := Ada.Strings.Unbounded.Unbounded_Slice (Buffer, Len + 1, Ada.Strings.Unbounded.Length (Buffer));
         end if;
         Is_Closed := Closed and then Ada.Strings.Unbounded.Length (Buffer) = 0;
      end Pop;

      procedure Close is
         Resp : constant GNATCOLL.JSON.JSON_Value := GNATCOLL.JSON.Create_Object;
         Now  : constant Ada.Calendar.Time := Ada.Calendar.Clock;
         TS   : String := Ada.Calendar.Formatting.Image (Now);
      begin
         if TS'Length >= 11 then
            TS (11) := 'T';
         end if;

         case Format is
            when Raw => null;
            when Ollama =>
               GNATCOLL.JSON.Set_Field (Resp, "model", Ada.Strings.Unbounded.To_String (Model_ID));
               GNATCOLL.JSON.Set_Field (Resp, "created_at", TS & "Z");
               GNATCOLL.JSON.Set_Field (Resp, "done", True);
               Ada.Strings.Unbounded.Append (Buffer, GNATCOLL.JSON.Write (Resp) & ASCII.LF);
            when OpenAI =>
               Ada.Strings.Unbounded.Append (Buffer, "data: [DONE]" & ASCII.LF);
         end case;
         Closed := True;
      end Close;
   end Queue;

   overriding function End_Of_File (Resource : Response_Stream) return Boolean is
   begin
      return False;
   end End_Of_File;

   overriding procedure Read
     (Resource : in out Response_Stream;
      Buffer   : out Stream_Element_Array;
      Last     : out Stream_Element_Offset)
   is
      Item : String (1 .. 4096);
      Is_Closed : Boolean;
      Actual_Len : Natural;
   begin
      Last := Buffer'First - 1;
      if Resource.Q = null then
         return;
      end if;
      
      Resource.Q.Pop (Item, Actual_Len, Is_Closed);
      
      if Actual_Len > 0 then
         declare
            To_Fill : constant Stream_Element_Offset := 
              Stream_Element_Offset'Min (Buffer'Length, Stream_Element_Offset (Actual_Len));
         begin
            for I in 1 .. To_Fill loop
               Last := Last + 1;
               Buffer (Last) := Stream_Element (Character'Pos (Item (Integer (I))));
            end loop;
         end;
      end if;
   end Read;

end Streaming_Queue;
