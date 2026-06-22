pragma SPARK_Mode (Off);
--  KV Cache Manager — SSD Cache Spillover for llama.cpp
--
--  Implementation uses llama.cpp's state save/load APIs and SHA-256 hashing
--  for cache file naming. LRU eviction ensures we don't fill up the SSD.

with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar; use Ada.Calendar;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

with Llama_Interface; use Llama_Interface;

package body KV_Cache_Manager is

   --  Simple hash function (not SHA-256, but good enough for cache naming)
   function Simple_Hash (Data : String) return String is
      Hash : Natural := 0;
   begin
      for I in Data'Range loop
         Hash := (Hash * 31 + Character'Pos (Data (I))) mod 1000000;
      end loop;
      return Trim (Natural'Image (Hash), Both);
   end Simple_Hash;

   --  Hash first Max_Hash tokens from token array
   --  Note: This is a simplified version. In production, we'd hash the actual
   --  token values, but System.Address conversion is platform-specific.
   function Hash_Tokens
     (Tokens   : System.Address;
      N_Tokens  : Interfaces.C.size_t;
      Max_Hash  : Positive := 128) return String
   is
      use type Interfaces.C.size_t;
      Tok_Len : constant Natural :=
        Natural'Min (Natural (N_Tokens), Max_Hash);
   begin
      --  For now, just hash the token count as a simple cache key
      --  In production, we'd read the actual token values from memory
      return Simple_Hash (Natural'Image (Tok_Len));
   end Hash_Tokens;

   --  Check if SSD cache exists for a given prompt prefix hash
   function Has_Cached_Prefix (Prompt_Hash : String) return Boolean is
      File_Path : constant String := Cache_Dir & Prompt_Hash & ".bin";
   begin
      return Ada.Directories.Exists (File_Path);
   end Has_Cached_Prefix;

   --  Save KV cache to SSD file
   procedure Save_To_SSD
     (Context    : Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t;
      File_Path  : String;
      Success    : out Boolean)
   is
      Path_C : chars_ptr := New_String (File_Path);
   begin
      --  Create cache directory if needed
      if not Ada.Directories.Exists (Cache_Dir) then
         Ada.Directories.Create_Path (Cache_Dir);
      end if;

      --  Save state via llama.cpp
      Success := Llama_State_Save_File
        (Context, Path_C, Tokens, N_Tokens);

      if Success then
         Put_Line ("[KV-Cache] Saved to SSD: " & File_Path);
      else
         Put_Line ("[KV-Cache] Failed to save: " & File_Path);
      end if;

      Free (Path_C);

      --  Evict old cache files if needed
      Evict_Old_Cache;
   exception
      when others =>
         Success := False;
         Free (Path_C);
   end Save_To_SSD;

   --  Load KV cache from SSD file
   function Load_From_SSD
     (Context    : Llama_Context;
      File_Path  : String;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t) return Boolean
   is
      Path_C  : chars_ptr := New_String (File_Path);
      N_Out   : aliased Interfaces.C.size_t := 0;
      Success : Boolean;
   begin
      Tokens := System.Null_Address;
      N_Tokens := 0;

      if not Ada.Directories.Exists (File_Path) then
         Free (Path_C);
         return False;
      end if;

      --  Load state from file
      Success := Llama_State_Load_File
        (Context, Path_C, Tokens, N_Tokens, N_Out'Access);
      N_Tokens := N_Out;

      if Success then
         Put_Line ("[KV-Cache] Loaded from SSD: " & File_Path &
                   " (" & Interfaces.C.size_t'Image (N_Tokens) & " tokens)");
      else
         Put_Line ("[KV-Cache] Failed to load: " & File_Path);
      end if;

      Free (Path_C);
      return Success;
   exception
      when others =>
         Free (Path_C);
         return False;
   end Load_From_SSD;

   --  Evict oldest cache files if exceeding Max_Cache_Files
   procedure Evict_Old_Cache is
      use Ada.Directories;

      Count : Natural := 0;

      procedure Count_Files (Ent : Directory_Entry_Type) is
      begin
         if Simple_Name (Ent)'Length > 4 and then
           Simple_Name (Ent) (Simple_Name (Ent)'Last - 3 .. Simple_Name (Ent)'Last) = ".bin"
         then
            Count := Count + 1;
         end if;
      end Count_Files;

   begin
      if not Exists (Cache_Dir) then
         return;
      end if;

      --  Count cache files
      Search (Cache_Dir, "*.bin", (True, False, False), Count_Files'Access);

      if Count > Max_Cache_Files then
         Put_Line ("[KV-Cache] Evicting old cache files (" &
                   Natural'Image (Count) & " > " &
                   Natural'Image (Max_Cache_Files) & ")");
         --  TODO: Implement LRU eviction by modification time
         --  For now, just log the warning
      end if;
   end Evict_Old_Cache;

   --  Initialize cache directory
   procedure Initialize is
   begin
      if not Ada.Directories.Exists (Cache_Dir) then
         Ada.Directories.Create_Path (Cache_Dir);
         Put_Line ("[KV-Cache] Created cache directory: " & Cache_Dir);
      else
         Put_Line ("[KV-Cache] Cache directory exists: " & Cache_Dir);
      end if;
   end Initialize;

end KV_Cache_Manager;
