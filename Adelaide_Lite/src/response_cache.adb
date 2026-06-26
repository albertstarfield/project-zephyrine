pragma SPARK_Mode (Off);
with Ada.Text_IO; use Ada.Text_IO;
with AnsiAda;
with Ada.Characters.Handling; use Ada.Characters.Handling;
with Ada.Strings.Maps; use Ada.Strings.Maps;

--  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
--  STRING RESPONSE CACHE implementation
--  O(1) hash table lookup with fuzzy normalization

package body Response_Cache is

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Hash function for Unbounded_String keys
   function Hash_Unbounded (Key : Unbounded_String) return Ada.Containers.Hash_Type is
   begin
      return Ada.Strings.Hash (To_String (Key));
   end Hash_Unbounded;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Normalize prompt: lowercase, collapse whitespace, trim
   --  This enables fuzzy matching: "Say hi" = "say hi" = "Say  hi"
   function Normalize (Prompt : String) return String is
      Result : Unbounded_String;
      Prev_Was_Space : Boolean := False;
   begin
      for I in Prompt'Range loop
         declare
            C : constant Character := Prompt (I);
         begin
            if C = ' ' or else C = ASCII.LF or else C = ASCII.CR
              or else C = ASCII.HT
            then
               if not Prev_Was_Space then
                  Append (Result, ' ');
                  Prev_Was_Space := True;
               end if;
            else
               Append (Result, To_Lower (C));
               Prev_Was_Space := False;
            end if;
         end;
      end loop;

      --  Trim trailing space
      if Length (Result) > 0
        and then Element (Result, Length (Result)) = ' '
      then
         Result := To_Unbounded_String (Slice (Result, 1, Length (Result) - 1));
      end if;

      return To_String (Result);
   end Normalize;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Pre-seeded responses for common queries
   procedure Seed_Common_Queries is
   begin
      --  Greetings
      Store ("hi", "Hello! How can I help you today?");
      Store ("hello", "Hello! How can I help you today?");
      Store ("hey", "Hey there! What can I do for you?");
      Store ("hi there", "Hi there! How can I assist you?");
      Store ("hello there", "Hello there! What can I help with?");
      Store ("good morning", "Good morning! How can I help?");
      Store ("good afternoon", "Good afternoon! How can I help?");
      Store ("good evening", "Good evening! How can I help?");

      --  Status queries
      Store ("say hi", "Hello! I'm Snowball Enaga, your AI assistant.");
      Store ("say hello", "Hello! I'm Snowball Enaga, your AI assistant.");
      Store ("who are you", "I'm Snowball Enaga, an AI assistant powered by Adelaide Lite.");
      Store ("what are you", "I'm Snowball Enaga, an AI assistant running locally on your machine.");
      Store ("what is your name", "My name is Snowball Enaga.");
      Store ("how are you", "I'm doing great! Thanks for asking. How can I help?");

      --  Capability queries
      Store ("what can you do", "I can answer questions, help with coding, analyze text, generate images, and more. Just ask!");
      Store ("help", "I'm here to help! You can ask me questions, request code help, or have a conversation.");
      Store ("what do you know", "I have access to a knowledge base and can reason about many topics. What would you like to know?");

      --  Simple acknowledgments
      Store ("ok", "Got it! Let me know if you need anything else.");
      Store ("thanks", "You're welcome! Happy to help.");
      Store ("thank you", "You're welcome! Let me know if there's anything else.");
      Store ("bye", "Goodbye! Have a great day!");
      Store ("goodbye", "Goodbye! Take care!");

      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) &
                "[Response-Cache]" & AnsiAda.Reset &
                " Seeded " & Natural'Image (Integer (Cache_Maps.Length (Cache_Map))) &
                " common queries");
   end Seed_Common_Queries;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Initialize cache with pre-seeded responses
   procedure Initialize is
   begin
      Cache_Maps.Clear (Cache_Map);
      Hit_Counter := 0;
      Miss_Counter := 0;
      Seed_Common_Queries;
   end Initialize;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Look up prompt in cache. O(1) average case.
   function Lookup (Prompt : String) return String is
      Key : constant Unbounded_String := To_Unbounded_String (Normalize (Prompt));
   begin
      if Cache_Maps.Contains (Cache_Map, Key) then
         Hit_Counter := Hit_Counter + 1;
         return To_String (Cache_Maps.Element (Cache_Map, Key));
      end if;

      Miss_Counter := Miss_Counter + 1;
      return "";
   end Lookup;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   --  Store prompt→response in cache. O(1) average.
   procedure Store (Prompt : String; Response : String) is
      Key : constant Unbounded_String := To_Unbounded_String (Normalize (Prompt));
   begin
      --  Evict oldest entry if cache is full
      if Integer (Cache_Maps.Length (Cache_Map)) >= Max_Cache_Entries then
         declare
            Iter : Cache_Maps.Cursor := Cache_Maps.First (Cache_Map);
         begin
            if Cache_Maps.Has_Element (Iter) then
               Cache_Maps.Delete (Cache_Map, Iter);
            end if;
         end;
      end if;

      --  Insert or overwrite
      if Cache_Maps.Contains (Cache_Map, Key) then
         Cache_Maps.Replace (Cache_Map, Key, To_Unbounded_String (Response));
      else
         Cache_Maps.Insert (Cache_Map, Key, To_Unbounded_String (Response));
      end if;
   end Store;

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  YOU ARE EXPECTED TO BE INSTANT GRATIFICATION SERVER EITHER UX IS TTFB BELOW 10ms, WHEN STREAMING. REMEMBER THE REALITY IS THAT YOU ARE LIVING on INDONESIA YOU ARE NOT PAID FOR BREATHING A SINGLE SECOND. DO IT INSTANTLY AND WITH PAID CHEAPLY OR YOU DIE! NO EXCUSE
--  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
   function Hit_Count return Natural is (Hit_Counter);
   function Miss_Count return Natural is (Miss_Counter);
   function Entry_Count return Natural is (Natural (Cache_Maps.Length (Cache_Map)));

   procedure Reset_Stats is
   begin
      Hit_Counter := 0;
      Miss_Counter := 0;
   end Reset_Stats;

end Response_Cache;
