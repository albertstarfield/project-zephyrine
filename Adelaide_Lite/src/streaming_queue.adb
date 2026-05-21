package body Streaming_Queue is

   protected body Queue is
      entry Push (Item : String) when True is
      begin
         Append (Buffer, Item);
      end Push;

      entry Pop (Item : out String; Is_Closed : out Boolean)
        when Length (Buffer) > 0 or else Closed
      is
         Len : constant Natural := Natural'Min (Length (Buffer), 1024);
      begin
         if Len > 0 then
            Item := To_String (Unbounded_Slice (Buffer, 1, Len));
            Buffer := Unbounded_Slice (Buffer, Len + 1, Length (Buffer));
         else
            Item := "";
         end if;
         Is_Closed := Closed and then Length (Buffer) = 0;
      end Pop;

      procedure Close is
      begin
         Closed := True;
      end Close;
   end Queue;

   overriding function End_Of_File (Resource : Response_Stream) return Boolean is
      Dummy : String (1 .. 0);
      Is_Closed : Boolean;
   begin
      if Resource.Q = null then
         return True;
      end if;
      --  Check if closed without blocking if possible? 
      --  Actually Pop blocks. We might need a non-blocking check.
      return False; -- AWS will call Read anyway.
   end End_Of_File;

   overriding procedure Read
     (Resource : in out Response_Stream;
      Buffer   : out Stream_Element_Array;
      Last     : out Stream_Element_Offset)
   is
      Item : String (1 .. 1024);
      Is_Closed : Boolean;
      Actual_Len : Natural;
   begin
      Last := Buffer'First - 1;
      if Resource.Q = null then
         return;
      end if;
      
      --  Wait for data
      Resource.Q.Pop (Item, Is_Closed);
      Actual_Len := Item'Length;
      
      if Actual_Len > 0 then
         for I in 1 .. Actual_Len loop
            Last := Last + 1;
            Buffer (Last) := Stream_Element (Character'Pos (Item (I)));
         end loop;
      end if;
   end Read;

end Streaming_Queue;
