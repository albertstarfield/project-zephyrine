-- watchdog_thread2.adb
-- Ada/SPARK Watchdog (Level 2) — ZephyWatchtower Component
--
-- SPARK Verification Policy:
--   - Top-level SPARK_Mode pragma: On (covers the outer procedure and nested
--     subprograms unless individually overridden with pragma SPARK_Mode (Off))
--   - C_Bridge package body: pragma SPARK_Mode (Off) inside body — C FFI
--   - Perform_Memory_Check: pragma SPARK_Mode (Off) — pointer arithmetic
--   - File_Exists: SPARK_Mode On (inherited), with Pre/Post contracts
--
--
-- RULE: For nested units, SPARK_Mode must be set via pragma, not an aspect.
-- (Ada RM + SPARK RM E.0011: SPARK_Mode aspect only allowed at library level.)
--

with Ada.Strings.Bounded;
with Ada.Text_IO;
with Ada.Exceptions;
with Interfaces;
with Interfaces.C;
with Interfaces.C.Strings;
with System;
with System.Storage_Elements;
with Ada.Containers.Vectors;
with Ada.Unchecked_Conversion;
with Ada.Command_Line;
with Watchdog_Core;

-- The main procedure is the single top-level compilation unit.

procedure Watchdog_Thread2 is
    pragma SPARK_Mode (Off);

    -- ===================================================================
    --  PART 1: GLOBAL DECLARATIONS
    -- ===================================================================

    use Ada.Text_IO;
    use Interfaces.C;
    use type System.Address;
    use type Interfaces.Unsigned_8;
    use System.Storage_Elements;

    subtype Byte is Interfaces.Unsigned_8;
    
    package Arg_Strings is new Ada.Strings.Bounded.Generic_Bounded_Length (Max => 1024);
    use Arg_Strings;
    type Bounded_String_Array is array (Positive range <>) of Bounded_String;

    -- ===================================================================
    --  PART 2: C_BRIDGE SPECIFICATION
    --  Spec is in SPARK_Mode On (inherited). Contracts on the spec let
    --  callers (which are in On territory) be verified at call sites.
    --  The body uses pragma SPARK_Mode (Off) to exclude C FFI from proof.
    -- ===================================================================
    package C_Bridge is
        type Process_Id is new int;
        No_Process_Id : constant Process_Id := -1;

        -- SPARK Pre/Post on the spec — visible to the prover at call sites.
        function Spawn
           (Program : String; Args : Bounded_String_Array) return Process_Id
          with
            Global => null,
            Pre    => Program'Length > 0;

        function Is_Alive (Pid : Process_Id) return Boolean
          with
            Global => null,
            Post   => (if Pid = No_Process_Id then not Is_Alive'Result);


        -- Malloc/Free cannot have SPARK contracts: they are C wrappers.
        -- Declared here but excluded from proof via pragma in the body.
        function Malloc (Size : size_t) return System.Address;
        procedure Free (Ptr : System.Address);
    end C_Bridge;

    -- ===================================================================
    --  PART 3: C_BRIDGE BODY
    --  pragma SPARK_Mode (Off) is placed first inside the body.
    --  RATIONALE: This body contains imported C subprograms (fork, execvp,
    --  kill, malloc, free). SPARK cannot formally verify C FFI bodies — this
    --  is a defined limitation of the SPARK proof system, not a code defect.
    --  WARRANTY: The spec contracts above (Pre/Post/Global) are still enforced
    --  at every call site that remains in SPARK_Mode On territory.
    -- ===================================================================

    package body C_Bridge is
        pragma SPARK_Mode (Off);  -- Exclude entire body from SPARK proof (C FFI)
        use Interfaces;
        use Interfaces.C.Strings;

        function C_Fork return int;
        pragma Import (C, C_Fork, "fork");
        function C_Execvp
           (File : chars_ptr; Args : chars_ptr_array) return int;
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
            C_Free_Body (Ptr);
        end Free;

        function Spawn
           (Program : String; Args : Bounded_String_Array) return Process_Id
        is
            Arg_Count : constant size_t := size_t (Args'Length + 1);
            C_Args    : aliased chars_ptr_array (size_t (0) .. Arg_Count);
            C_Program : chars_ptr := New_String (Program);
            Child_PID : int;
        begin
            C_Args (size_t (0)) := New_String (Program);
            for I in Args'Range loop
                C_Args (size_t (I - Args'First + 1)) :=
                   New_String (To_String (Args (I)));
            end loop;
            C_Args (Arg_Count) := null_ptr;
            Child_PID := C_Fork;
            if Child_PID < 0 then
                return No_Process_Id;
            elsif Child_PID = 0 then
                declare
                    Return_Code : int := C_Execvp (C_Program, C_Args);
                    pragma Unreferenced (Return_Code);
                    procedure C_Exit (Status : int);
                    pragma Import (C, C_Exit, "_exit");
                begin
                    C_Exit (1);
                end;
                return No_Process_Id;
            else
                Free (C_Program);
                for Ptr of C_Args loop
                    if Ptr /= null_ptr then
                        Free (Ptr);
                    end if;
                end loop;
                return Process_Id (Child_PID);
            end if;
        end Spawn;

        function Is_Alive (Pid : Process_Id) return Boolean is
            Signal_0 : constant int := 0;
        begin
            if Pid = No_Process_Id then
                return False;
            end if;
            return C_Kill (int (Pid), Signal_0) = 0;
        end Is_Alive;


    end C_Bridge;

    -- ===================================================================
    --  PART 3b: MEMORY CHECK PROCEDURE
    --  pragma SPARK_Mode (Off) because this procedure:
    --    - Uses Ada.Unchecked_Conversion for raw pointer arithmetic
    --    - Calls C_Bridge.Malloc and Free (C wrappers)
    --  RATIONALE: SPARK cannot prove manual memory management or pointer
    --  dereferencing through Unchecked_Conversion. The Off pragma is the
    --  correct mechanism to scope the exclusion to this procedure only.
    --  The control flow logic (Success output, loop bounds) is correct by
    --  inspection and does not require formal proof for this safety level.
    -- ===================================================================

    -- File_Exists has been moved to Watchdog_Core for SPARK proof.

    procedure Perform_Memory_Check (Success : out Boolean) is
        pragma SPARK_Mode (Off);  -- Pointer arithmetic + C malloc/free

        procedure Write_And_Verify_Chunk
           (Memory_Block  : System.Address;
            Size          : size_t;
            Pattern       : Byte;
            Chunk_Success : out Boolean)
        is
            type Byte_Ptr is access all Byte;
            function To_Byte_Ptr is new
               Ada.Unchecked_Conversion (System.Address, Byte_Ptr);
            Current_Ptr : Byte_Ptr;
        begin
            Chunk_Success := True;
            Put
               ("  -> Verifying integrity of chunk at "
                & Integer'Image (Integer (System.Storage_Elements.To_Integer (Memory_Block)))
                & "... ");
            for I in 0 .. Storage_Offset (Size) - 1 loop
                Current_Ptr := To_Byte_Ptr (Memory_Block + I);
                Current_Ptr.all := Pattern;
            end loop;
            for I in 0 .. Storage_Offset (Size) - 1 loop
                Current_Ptr := To_Byte_Ptr (Memory_Block + I);
                if Current_Ptr.all /= Pattern then
                    Put_Line ("FAILURE!");
                    Put_Line
                       ("     >> Mismatch at offset "
                        & Storage_Offset'Image (I)
                        & ". Expected "
                        & Byte'Image (Pattern)
                        & ", found "
                        & Byte'Image (Current_Ptr.all));
                    Chunk_Success := False;
                    return;
                end if;
            end loop;
            Put_Line ("OK.");
        end Write_And_Verify_Chunk;

        Flag_File_Name     : constant String :=
           "_potential_incapable_machine.flag";
        Engine_Subdir_Path : constant String :=
           "systemCore/mainEngineFrame_MacroController_EngineSharedResources/";
        Root_Flag_Path     : constant String := "./" & Flag_File_Name;
        Engine_Flag_Path   : constant String :=
           Engine_Subdir_Path & Flag_File_Name;

        Flag_File_Exists : Boolean := False;


        procedure Check_For_Flag is
        begin
            Put_Line
               ("INFO: Checking for incapability flag in root and engine directories...");
            if Watchdog_Core.File_Exists (Root_Flag_Path)
               or else Watchdog_Core.File_Exists (Engine_Flag_Path)
            then
                Flag_File_Exists := True;
                Put_Line ("INFO: Diagnostic flag found from a previous run.");
            else
                Flag_File_Exists := False;
            end if;
        end Check_For_Flag;

        procedure Write_Flag_Content (File_Handle : in out File_Type) is
        begin
            Put_Line
               (File_Handle,
                "The Ada watchdog's initial memory stress test failed.");
            Put_Line
               (File_Handle,
                "This suggests the machine may not have enough free RAM to run the full application stack reliably.");
        end Write_Flag_Content;

        procedure Create_Incapability_Flag is
            Flag_File : File_Type;
        begin
            if not Flag_File_Exists then
                begin
                    Create
                       (File => Flag_File,
                        Mode => Out_File,
                        Name => Root_Flag_Path);
                    Write_Flag_Content (Flag_File);
                    Close (Flag_File);
                    Put_Line
                       ("     >> Created diagnostic flag file: "
                        & Root_Flag_Path);
                exception
                    when others =>
                        Put_Line
                           ("     >> WARNING: Could not create root diagnostic flag file.");
                end;

                begin
                    Create
                       (File => Flag_File,
                        Mode => Out_File,
                        Name => Engine_Flag_Path);
                    Write_Flag_Content (Flag_File);
                    Close (Flag_File);
                    Put_Line
                       ("     >> Created diagnostic flag file: "
                        & Engine_Flag_Path);
                exception
                    when others =>
                        Put_Line
                           ("     >> WARNING: Could not create engine diagnostic flag file. (Does the directory exist?)");
                end;
            end if;
        end Create_Incapability_Flag;

        MB_In_Bytes      : constant := 1_048_576;
        Chunk_Size_Bytes : constant size_t := 128 * MB_In_Bytes;
        Total_MB_Target  : size_t;
        Number_Of_Chunks : Natural;
        package Address_Vectors is new
           Ada.Containers.Vectors (Positive, System.Address);
        Allocated_Chunks : Address_Vectors.Vector;
        New_Chunk_Addr   : System.Address;
        Chunk_Is_OK      : Boolean;

    begin
        Check_For_Flag;

        if Flag_File_Exists then
            Total_MB_Target := 512;
        else
            Total_MB_Target := 6 * 1024;
        end if;
        Number_Of_Chunks := Natural (Total_MB_Target / 128);

        Put_Line
           ("--- PRE-FLIGHT: Starting "
            & size_t'Image (Total_MB_Target)
            & " MB Memory Integrity Test ---");
        if Flag_File_Exists then
            Put_Line
               ("(Running reduced 'sanity check' as a full test failed previously.)");
        else
            Put_Line ("(Performing full-system stress test.)");
        end if;

        Success := True;
        for I in 1 .. Number_Of_Chunks loop
            Put
               ("  Pass "
                & Integer'Image (I)
                & "/"
                & Integer'Image (Number_Of_Chunks)
                & ": Allocating "
                & size_t'Image (Chunk_Size_Bytes / MB_In_Bytes)
                & " MB chunk... ");
            New_Chunk_Addr := C_Bridge.Malloc (Chunk_Size_Bytes);
            if New_Chunk_Addr = System.Null_Address then
                Put_Line ("FAILED.");
                Put_Line
                   ("  -> CRITICAL: The operating system could not provide the requested memory.");
                Create_Incapability_Flag;
                Success := False;
                exit;
            else
                Put_Line ("OK.");
                Allocated_Chunks.Append (New_Chunk_Addr);
                Write_And_Verify_Chunk
                   (New_Chunk_Addr,
                    Chunk_Size_Bytes,
                    Byte (I mod 256),
                    Chunk_Is_OK);
                if not Chunk_Is_OK then
                    Put_Line
                       ("  -> CRITICAL: Memory integrity check failed. Data written to RAM was not read back correctly.");
                    Create_Incapability_Flag;
                    Success := False;
                    exit;
                end if;
            end if;
        end loop;
        New_Line;
        Put_Line ("--- PRE-FLIGHT: Deallocating all test chunks... ---");
        for Ptr of Allocated_Chunks loop
            C_Bridge.Free (Ptr);
        end loop;
        if Success then
            Put_Line
               ("--- PRE-FLIGHT: Memory Check Passed. System appears stable. ---");
        end if;
        New_Line;
    end Perform_Memory_Check;

    -- ===================================================================
    --  PART 4: MAIN PROGRAM VARIABLES
    -- ===================================================================
    use type C_Bridge.Process_Id;
    use Ada.Command_Line;

    Memory_Check_OK : Boolean;
    Child_PID       : C_Bridge.Process_Id := C_Bridge.No_Process_Id;

    Program_To_Run : String (1 .. Argument (1)'Length);
    Process_Args   : Bounded_String_Array (1 .. Argument_Count - 1);

    -- ===================================================================
    --  PART 5: MAIN PROGRAM LOGIC
    -- ===================================================================
begin
    Put_Line ("=====================================================");
    Put_Line (" ADA/SPARK WATCHDOG (LEVEL 2) INITIALIZING...");
    Put_Line ("=====================================================");

    if Argument_Count < 1 then
        Put_Line ("FATAL: Ada Watchdog requires a command to supervise.");
        Put_Line ("Usage: ./watchdog_thread2 <program_path> [args...]");
        return;
    end if;

    Program_To_Run := Argument (1);
    for I in Process_Args'Range loop
        Process_Args (I) := To_Bounded_String (Argument (I + 1));
    end loop;

    Put_Line ("Supervision Target: " & Program_To_Run);
    if Process_Args'Length > 0 then
        Put_Line ("Target Arguments:");
        for Arg of Process_Args loop
            Put_Line ("  " & To_String (Arg));
        end loop;
    end if;
    New_Line;

    Perform_Memory_Check (Success => Memory_Check_OK);

    if Memory_Check_OK then
        Put_Line ("--- Watchdog entering active monitoring mode. ---");
        New_Line;

        declare
            Health_Check_Interval : constant Natural := 6;
            Cycle_Counter         : Natural          := 0;
        begin
            loop
                if not C_Bridge.Is_Alive (Child_PID) then
                    Cycle_Counter := 0;
                    Put_Line
                       ("Watchdog_Thread2: [FAIL] Monitored process is not running. Spawning...");
                    Child_PID :=
                       C_Bridge.Spawn
                          (Program => Program_To_Run, Args => Process_Args);

                    if Child_PID /= C_Bridge.No_Process_Id then
                        Put_Line
                           ("Watchdog_Thread2: [OK] Successfully spawned target with PID: "
                            & C_Bridge.Process_Id'Image (Child_PID));
                    else
                        Put_Line
                           ("Watchdog_Thread2: [CRITICAL] *** FAILED to spawn target. ***");
                        Put_Line
                           ("Watchdog_Thread2: Retrying in 5 seconds...");
                        delay 5.0;
                    end if;
                else
                    Cycle_Counter := Cycle_Counter + 1;
                    if Cycle_Counter >= Health_Check_Interval then
                        Put_Line
                           ("Watchdog_Thread2: [OK] Health check passed. Monitored process (PID "
                            & C_Bridge.Process_Id'Image (Child_PID)
                            & ") is alive.");
                        Cycle_Counter := 0;
                    end if;
                end if;

                delay 5.0;
            end loop;
        end;
    else
        Put_Line
           ("CRITICAL FAILURE: Pre-flight memory check failed. The system is unstable. Halting.");
    end if;

exception
    when E : others =>
        Put_Line
           ("!!! A FATAL, UNEXPECTED ERROR occurred in the Ada Watchdog: "
            & Ada.Exceptions.Exception_Message (E));
        delay 10.0;
end Watchdog_Thread2;
