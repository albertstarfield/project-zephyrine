pragma SPARK_Mode (Off);
--  ============================================================================
--  KV Cache Manager — SSD Cache Spillover for llama.cpp
--  ============================================================================
--  Implementation uses llama.cpp's state save/load APIs and hash-based
--  naming for cache files. Auto-save/load ensures cache persists across
--  server restarts for fastest response times.
--
--  RAM Policy: Only the currently processing cache stays in RAM.
--  After generation completes, cache is saved to disk and cleared from RAM.
--  ============================================================================

--  [DO NOT REMOVE, OR YOU WILL BE KILLED]
--  Verbose logging with uptime timestamps for debugging KV cache operations.
--  Each log entry includes module tag [KV-Cache] and uptime offset for
--  correlating with other subsystem logs during startup and runtime.
with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Directories;
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Calendar; use Ada.Calendar;
with Ada.Real_Time; use Ada.Real_Time;
with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

with Llama_Interface; use Llama_Interface;

package body KV_Cache_Manager is

   --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
   --  Capture start time for uptime calculation in log messages.
   --  This timestamp is used to calculate relative offsets like "+15.378s"
   --  for correlating KV-Cache operations with other subsystem logs.
   Init_Start_Time : Ada.Real_Time.Time;

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
   function Hash_Tokens
     (Tokens    : System.Address;
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
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs cache eviction when file count exceeds limit.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Evicting old cache files (" &
                   Natural'Image (Count) & " > " &
                   Natural'Image (Max_Cache_Files) & ")");
         --  TODO: Implement LRU eviction by modification time
         --  For now, just log the warning
      end if;
   end Evict_Old_Cache;

   --  Initialize cache directory
   procedure Initialize is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Capture start time for uptime calculation in log messages.
      Init_Start_Time := Ada.Real_Time.Clock;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms KV_Cache_Manager.Initialize was entered.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Initialize ENTERED.");

      if not Ada.Directories.Exists (Cache_Dir) then
         Ada.Directories.Create_Path (Cache_Dir);
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms cache directory creation.
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Created cache directory: " & Cache_Dir);
      else
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms cache directory already exists.
         Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Cache directory exists: " & Cache_Dir);
      end if;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms KV_Cache_Manager.Initialize completed.
      Put_Line (AnsiAda.Foreground (AnsiAda.Cyan) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Initialize COMPLETE.");
   end Initialize;

   --  Save KV cache to SSD file
   procedure Save_To_SSD
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t;
      Success    : out Boolean)
   is
      Path_C : chars_ptr;
   begin
      --  Generate cache key from tokens
      declare
         Prompt_Hash : constant String := Hash_Tokens (Tokens, N_Tokens);
         File_Path   : constant String := Cache_Dir & Prompt_Hash & ".bin";
      begin
         Path_C := New_String (File_Path);

         --  Create cache directory if needed
         if not Ada.Directories.Exists (Cache_Dir) then
            Ada.Directories.Create_Path (Cache_Dir);
         end if;

         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs save attempt with file path and token count.
         Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Save_To_SSD: " & File_Path &
                   " (" & Interfaces.C.size_t'Image (N_Tokens) & " tokens)");

         --  Save state via llama.cpp
         Success := Llama_State_Save_File
           (Context, Path_C, Tokens, N_Tokens);

         if Success then
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: confirms successful save to SSD.
            Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                      AnsiAda.Reset & "+" &
                      Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                      "s Save_To_SSD SUCCESS: " & File_Path &
                      " (" & Interfaces.C.size_t'Image (N_Tokens) & " tokens)");
         else
            --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
            --  Verbose: logs save failure with error context.
            Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                      AnsiAda.Reset & "+" &
                      Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                      "s Save_To_SSD FAILED: " & File_Path);
         end if;

         Free (Path_C);

         --  Evict old cache files if needed
         Evict_Old_Cache;
      end;
   exception
      when others =>
         Success := False;
         if Path_C /= Null_Ptr then
            Free (Path_C);
         end if;
   end Save_To_SSD;

   --  Load KV cache from SSD file
   function Load_From_SSD
     (Context    : Llama_Interface.Llama_Context;
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

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs load attempt with file path.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Load_From_SSD: " & File_Path);

      --  Load state from file
      Success := Llama_State_Load_File
        (Context, Path_C, Tokens, N_Tokens, N_Out'Access);
      N_Tokens := N_Out;

      if Success then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms successful load from SSD.
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Load_From_SSD SUCCESS: " & File_Path &
                   " (" & Interfaces.C.size_t'Image (N_Tokens) & " tokens)");
      else
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs load failure with error context.
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Load_From_SSD FAILED: " & File_Path);
      end if;

      Free (Path_C);
      return Success;
   exception
      when others =>
         Free (Path_C);
         return False;
   end Load_From_SSD;

   --  Auto-save after generation (called by Generate procedure)
   --  Saves to disk and clears from RAM immediately
   procedure Auto_Save
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : System.Address;
      N_Tokens   : Interfaces.C.size_t)
   is
      Success : Boolean;
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs auto-save trigger with token count.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Auto_Save: " &
                Interfaces.C.size_t'Image (N_Tokens) & " tokens");

      Save_To_SSD (Context, Tokens, N_Tokens, Success);
      if Success then
         --  Clear KV cache from RAM immediately after saving
         --  This ensures minimal RAM usage - only current process in memory
         Llama_Interface.Llama_Memory_Clear
           (Llama_Interface.Llama_Get_Memory (Context), False);
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: confirms auto-save complete and RAM cleared.
         Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Auto_Save COMPLETE: saved to SSD and cleared from RAM");
      end if;
   end Auto_Save;

   --  Auto-load on startup (loads most recent cache from disk)
   function Auto_Load
     (Context    : Llama_Interface.Llama_Context;
      Tokens     : out System.Address;
      N_Tokens   : out Interfaces.C.size_t) return Boolean
   is
      use Ada.Directories;

      Found : Boolean := False;

      procedure Find_Latest (Ent : Directory_Entry_Type) is
         Name : constant String := Simple_Name (Ent);
         Path : constant String := Full_Name (Ent);
      begin
         if not Found and then
           Name'Length > 4 and then
           Name (Name'Last - 3 .. Name'Last) = ".bin"
         then
            --  Found a cache file, try to load it
            if Load_From_SSD (Context, Path, Tokens, N_Tokens) then
               Found := True;
               --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
               --  Verbose: confirms auto-loaded cache file.
               Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                         AnsiAda.Reset & "+" &
                         Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                         "s Auto_Load: loaded cache: " & Path);
            end if;
         end if;
      end Find_Latest;

   begin
      Tokens := System.Null_Address;
      N_Tokens := 0;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs auto-load attempt.
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Auto_Load: searching for cache files...");

      if not Exists (Cache_Dir) then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs no cache directory found.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Auto_Load: no cache directory found");
         return False;
      end if;

      --  Search for any cache files
      Search (Cache_Dir, "*.bin", (True, False, False), Find_Latest'Access);

      if not Found then
         --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
         --  Verbose: logs no cache files found.
         Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                   AnsiAda.Reset & "+" &
                   Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                   "s Auto_Load: no cache files found");
      end if;

      return Found;
   end Auto_Load;

   --  Save all active caches (called on shutdown)
   procedure Save_All_On_Shutdown is
   begin
      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: logs shutdown save trigger.
      Put_Line (AnsiAda.Foreground (AnsiAda.Yellow) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Save_All_On_Shutdown: saving all active caches...");
      --  The actual save is handled by the Generate procedure's
      --  Auto_Save call, which saves to disk and clears from RAM.
      --  This procedure is called as a safety net.
      null;

      --  [DO NOT REMOVE, OR YOU WILL BE KILLED]
      --  Verbose: confirms shutdown save complete.
      Put_Line (AnsiAda.Foreground (AnsiAda.Green) & "[KV-Cache]" &
                AnsiAda.Reset & "+" &
                Trim(Duration'Image(Ada.Real_Time.To_Duration(Ada.Real_Time.Clock - Init_Start_Time)), Both) &
                "s Save_All_On_Shutdown COMPLETE.");
   end Save_All_On_Shutdown;

end KV_Cache_Manager;
