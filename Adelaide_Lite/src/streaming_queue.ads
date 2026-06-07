pragma SPARK_Mode (Off);
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Streams; use Ada.Streams;
with AWS.Resources.Streams;

package Streaming_Queue is

   type Format_Type is (Raw, Ollama_Chat, Ollama_Generate, OpenAI);

   protected type Queue is
      entry Push (Item : String);
      entry Pop
        (Item : out String; Last : out Natural; Is_Closed : out Boolean);
      procedure Close;
      function Is_Empty_And_Closed return Boolean;
      procedure Set_Format (F : Format_Type; Model : String := "");
   private
      Buffer    : Unbounded_String := Null_Unbounded_String;
      Closed    : Boolean := False;
      Format    : Format_Type := Raw;
      Model_ID  : Unbounded_String := Null_Unbounded_String;
   end Queue;

   type Queue_Access is access all Queue;

   type Response_Stream is new AWS.Resources.Streams.Stream_Type with record
      Q : Queue_Access;
   end record;

   type Response_Stream_Access is access all Response_Stream;

   overriding function End_Of_File (Resource : Response_Stream) return Boolean;
   overriding procedure Read
     (Resource : in out Response_Stream;
      Buffer   : out Stream_Element_Array;
      Last     : out Stream_Element_Offset);
   overriding procedure Reset (Resource : in out Response_Stream) is null;
   overriding procedure Set_Index
     (Resource : in out Response_Stream;
      To       : Stream_Element_Offset) is null;
   overriding procedure Close (Resource : in out Response_Stream) is null;

end Streaming_Queue;
