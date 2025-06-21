-- All necessary libraries for I/O, C bindings, and containers.
with Ada.Text_IO;
with Ada.Exceptions;
with Interfaces;
with Interfaces.C;
with Interfaces.C.Strings;
with System;
with System.Storage_Elements;
with Ada.Containers.Vectors;
with Ada.Unchecked_Conversion;

-- The main procedure is the single top-level compilation unit.
procedure Watchdog_Thread2 is

   -- ===================================================================
   --  PART 1: GLOBAL DECLARATIONS (between IS and BEGIN)
   -- ===================================================================
   
   use Ada.Text_IO;
   use Interfaces.C;
   use type System.Address;
   use type Interfaces.Unsigned_8;
   use System.Storage_Elements;

   subtype Byte is Interfaces.Unsigned_8;
   type String_Access_Array is array (Positive range <>) of access String;

   -- ===================================================================
   --  PART 2: C_BRIDGE SPECIFICATION
   -- ===================================================================
   package C_Bridge is
      type Process_Id is new int;
      No_Process_Id : constant Process_Id := -1;

      function Spawn (Program : String; Args : String_Access_Array) return Process_Id;
      function Is_Alive (Pid : Process_Id) return Boolean;
      function Kill_Process (Pid : Process_Id) return Boolean;
      function Malloc (Size : size_t) return System.Address;
      procedure Free (Ptr : System.Address);
   end C_Bridge;

   -- ===================================================================
   --  PART 3: IMPLEMENTATIONS (also between IS and BEGIN)
   -- ===================================================================

   package body C_Bridge is
      use Interfaces;
      use Interfaces.C.Strings;
      use type C.int;

      function C_Fork return int;
      pragma Import (C, C_Fork, "fork");
      function C_Execvp (File : chars_ptr; Args : chars_ptr_array) return int;
      pragma Import (C, C_Execvp, "execvp");
      function C_Kill (Pid : int; Sig : int) return int;
      pragma Import (C, C_Kill, "kill");
      function C_Malloc_Body (Size : size_t) return System.Address;
      pragma Import (C, C_Malloc_Body, "malloc");
      procedure C_Free_Body (Ptr : System.Address);
      pragma Import (C, C_Free_Body, "free");

      function Malloc (Size : size_t) return System.Address is
      begin
         return C_Malloc_Body (Size);
      end Malloc;
      procedure Free (Ptr : System.Address) is
      begin
         C_Free_Body(Ptr);
      end Free;

      function Spawn (Program : String; Args : String_Access_Array) return Process_Id is
         Arg_Count : constant size_t := size_t(Args'Length + 1);
         C_Args    : aliased chars_ptr_array (size_t(0) .. Arg_Count);
         C_Program : chars_ptr := New_String (Program);
         Child_PID : int;
      begin
         C_Args(size_t(0)) := New_String (Program);
         for I in Args'Range loop
            C_Args(size_t(I - Args'First + 1)) := New_String (Args(I).all);
         end loop;
         C_Args(Arg_Count) := null_ptr;
         Child_PID := C_Fork;
         if Child_PID < 0 then
            return No_Process_Id;
         elsif Child_PID = 0 then
            declare
               Return_Code : int := C_Execvp (C_Program, C_Args);
               procedure C_Exit (Status : int);
               pragma Import (C, C_Exit, "_exit");
            begin
               C_Exit (1);
            end;
            return No_Process_Id;
         else
            Free(C_Program);
            for Ptr of C_Args loop
               if Ptr /= null_ptr then
                  Free(Ptr);
               end if;
            end loop;
            return Process_Id (Child_PID);
         end if;
      end Spawn;

      function Is_Alive (Pid : Process_Id) return Boolean is
         Signal_0 : constant int := 0;
      begin
         if Pid = No_Process_Id then return False; end if;
         return C_Kill (int (Pid), Signal_0) = 0;
      end Is_Alive;

      function Kill_Process (Pid : Process_Id) return Boolean is
         Sig_Term : constant int := 15;
      begin
         return C_Kill (int (Pid), Sig_Term) = 0;
      end Kill_Process;
   end C_Bridge;

   procedure Perform_Memory_Check (Success : out Boolean) is
      procedure Write_And_Verify_Chunk
        (Memory_Block : System.Address; Size : size_t; Pattern : Byte; Chunk_Success : out Boolean)
      is
         type Byte_Ptr is access all Byte;
         function To_Byte_Ptr is new Ada.Unchecked_Conversion(System.Address, Byte_Ptr);
         Current_Ptr : Byte_Ptr;
      begin
         Chunk_Success := True;
         Put ("  -> Verifying chunk at " & System.Address'Image(Memory_Block) & "... ");
         for I in 0 .. Storage_Offset(Size) - 1 loop
            Current_Ptr := To_Byte_Ptr (Memory_Block + I);
            Current_Ptr.all := Pattern;
         end loop;
         for I in 0 .. Storage_Offset(Size) - 1 loop
            Current_Ptr := To_Byte_Ptr (Memory_Block + I);
            if Current_Ptr.all /= Pattern then
               Put_Line ("FAILURE!");
               Put_Line ("     Mismatch at offset " & Storage_Offset'Image(I));
               Chunk_Success := False;
               return;
            end if;
         end loop;
         Put_Line ("OK.");
      end Write_And_Verify_Chunk;
      
      MB               : constant := 1_048_576;
      Chunk_Size_Bytes : constant size_t := 128 * MB;
      Total_GB_Target  : constant := 1;
      Total_MB_Target  : constant := Total_GB_Target * 1024;
      Number_Of_Chunks : constant := Total_MB_Target / 128;
      
      package Address_Vectors is new Ada.Containers.Vectors (Positive, System.Address);
      Allocated_Chunks : Address_Vectors.Vector;
      
      New_Chunk_Addr   : System.Address;
      Chunk_Is_OK      : Boolean;
   begin
      Put_Line ("--- PRE-FLIGHT: Starting Memory Integrity Stress Test ---");
      Success := True;
      for I in 1 .. Number_Of_Chunks loop
         Put ("  Pass " & Integer'Image(I) & "/" & Integer'Image(Number_Of_Chunks) & ": Allocating 128 MB chunk... ");
         New_Chunk_Addr := C_Bridge.Malloc (Chunk_Size_Bytes);
         if New_Chunk_Addr = System.Null_Address then
            Put_Line ("FAILED.");
            Put_Line ("  -> CRITICAL: OS could not provide memory. Halting.");
            Success := False;
            exit;
         else
            Put_Line ("OK.");
            Allocated_Chunks.Append (New_Chunk_Addr);
            Write_And_Verify_Chunk (New_Chunk_Addr, Chunk_Size_Bytes, Byte(I mod 256), Chunk_Is_OK);
            if not Chunk_Is_OK then
               Put_Line ("  -> CRITICAL: Memory integrity check failed. Halting.");
               Success := False;
               exit;
            end if;
         end if;
      end loop;
      Put_Line ("--- PRE-FLIGHT: Deallocating all test chunks... ---");
      for Ptr of Allocated_Chunks loop
         C_Bridge.Free (Ptr);
      end loop;
      if Success then
         Put_Line ("--- PRE-FLIGHT: Memory Check Passed. ---");
      end if;
      New_Line;
   end Perform_Memory_Check;

   -- ===================================================================
   --  PART 4: MAIN PROGRAM VARIABLES
   -- ===================================================================
   use type C_Bridge.Process_Id;

   Memory_Check_OK : Boolean;
   Program_To_Run  : constant String := "./Watchdog_Thread1";
   Process_Args    : constant String_Access_Array :=
     (new String'("--integrity-check-file=./launcher.py"),
      new String'("--"),
      new String'("python"),
      new String'("AdelaideAlbertCortex.py"));
   Child_PID       : C_Bridge.Process_Id := C_Bridge.No_Process_Id;

-- ===================================================================
--  PART 5: THE MAIN PROGRAM LOGIC (the BEGIN block)
-- ===================================================================
begin
   Put_Line ("=====================================================");
   Put_Line (" ADA WATCHDOG (LEVEL 2) INITIALIZING...");
   Put_Line ("=====================================================");
   
   Perform_Memory_Check (Success => Memory_Check_OK);
   
   if Memory_Check_OK then
      Put_Line ("--- Watchdog entering active monitoring mode. ---");
      New_Line;
      
      loop
         if not C_Bridge.Is_Alive (Child_PID) then
            Put_Line ("ADA_WATCHDOG: Monitored process (Go Watchdog) is not running. Spawning...");
            Child_PID := C_Bridge.Spawn (Program => Program_To_Run, Args => Process_Args);
            if Child_PID /= C_Bridge.No_Process_Id then
               Put_Line ("ADA_WATCHDOG: Successfully spawned Go Watchdog with PID: " & C_Bridge.Process_Id'Image (Child_PID));
            else
               Put_Line ("ADA_WATCHDOG: *** FAILED to spawn Go Watchdog. ***");
               Put_Line ("ADA_WATCHDOG: Retrying in 5 seconds...");
               delay 5.0;
            end if;
         end if;
         delay 5.0;
      end loop;
   else
      Put_Line ("CRITICAL FAILURE: Pre-flight memory check failed. The system is unstable. Halting.");
   end if;

exception
   when E : others =>
      Put_Line ("!!! A FATAL, UNEXPECTED ERROR occurred in the Ada Watchdog: " &
                            Ada.Exceptions.Exception_Message (E));
      delay 10.0;
end Watchdog_Thread2;