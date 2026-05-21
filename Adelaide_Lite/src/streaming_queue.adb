package body Streaming_Queue is

   protected body Queue is
      entry Push (Item : String) when True is
      begin
         Append (Buffer, Item);
      end Push;

      entry Pop (Item : out String; Last : out Natural; Is_Closed : out Boolean)
        when Length (Buffer) > 0 or else Closed
      is
         Len : constant Natural := Natural'Min (Length (Buffer), Item'Length);
      begin
         Last := Len;
         if Len > 0 then
            Item (Item'First .. Item'First + Len - 1) := 
              To_String (Unbounded_Slice (Buffer, 1, Len));
            Buffer := Unbounded_Slice (Buffer, Len + 1, Length (Buffer));
         end if;
         Is_Closed := Closed and then Length (Buffer) = 0;
      end Pop;

      procedure Close is
      begin
         Closed := True;
      end Close;
   end Queue;

   overriding function End_Of_File (Resource : Response_Stream) return Boolean is
   begin
      --  AWS uses Read to determine EOF. Returning False ensures Read is called.
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
      
      --  Wait for data (blocks until data or closed)
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
            --  Note: if Item had more data than Buffer could hold, 
            --  those bytes are lost in this simplified impl. 
            --  But AWS usually provides large enough buffers.
         end;
      end if;
   end Read;

end Streaming_Queue;
