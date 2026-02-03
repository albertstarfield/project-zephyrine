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
with Ada.Command_Line;

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

        function Spawn
           (Program : String; Args : String_Access_Array) return Process_Id;
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
           (Program : String; Args : String_Access_Array) return Process_Id
        is
            Arg_Count : constant size_t := size_t (Args'Length + 1);
            C_Args    : aliased chars_ptr_array (size_t (0) .. Arg_Count);
            C_Program : chars_ptr := New_String (Program);
            Child_PID : int;
        begin
            C_Args (size_t (0)) := New_String (Program);
            for I in Args'Range loop
                C_Args (size_t (I - Args'First + 1)) :=
                   New_String (Args (I).all);
            end loop;
            C_Args (Arg_Count) := null_ptr;
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

        function Kill_Process (Pid : Process_Id) return Boolean is
            Sig_Term : constant int := 15;
        begin
            return C_Kill (int (Pid), Sig_Term) = 0;
        end Kill_Process;
    end C_Bridge;

    procedure Perform_Memory_Check (Success : out Boolean) is
        -- This procedure performs an aggressive memory allocation and integrity
        -- check to verify system stability before launching critical processes.

        -- Inner procedure to handle the logic for a single memory chunk.
        procedure Write_And_Verify_Chunk
           (Memory_Block  : System.Address;
            Size          : size_t;
            Pattern       : Byte;
            Chunk_Success : out Boolean)
        is
            -- (This inner procedure is unchanged)
            type Byte_Ptr is access all Byte;
            function To_Byte_Ptr is new
               Ada.Unchecked_Conversion (System.Address, Byte_Ptr);
            Current_Ptr : Byte_Ptr;
        begin
            Chunk_Success := True;
            Put
               ("  -> Verifying integrity of chunk at "
                & System.Address'Image (Memory_Block)
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

        -- NEW: Define constants for both flag file locations.
        Flag_File_Name     : constant String :=
           "_potential_incapable_machine.flag";
        Engine_Subdir_Path : constant String := "systemCore/engineMain/";
        Root_Flag_Path     : constant String := "./" & Flag_File_Name;
        Engine_Flag_Path   : constant String :=
           Engine_Subdir_Path & Flag_File_Name;

        Flag_File_Exists : Boolean := False;

        -- Helper function to check for a file's existence.
        function File_Exists (Path : String) return Boolean is
            File : File_Type;
        begin
            Open (File => File, Mode => In_File, Name => Path);
            Close (File);
            return True;
        exception
            when Name_Error =>
                return False;
        end File_Exists;

        -- Check for the flag in both locations.
        procedure Check_For_Flag is
        begin
            Put_Line
               ("INFO: Checking for incapability flag in root and engine directories...");
            if File_Exists (Root_Flag_Path)
               or else File_Exists (Engine_Flag_Path)
            then
                Flag_File_Exists := True;
                Put_Line ("INFO: Diagnostic flag found from a previous run.");
            else
                Flag_File_Exists := False;
            end if;
        end Check_For_Flag;

        -- Helper procedure to write the flag file content.
        procedure Write_Flag_Content (File_Handle : in out File_Type) is
        begin
            Put_Line
               (File_Handle,
                "The Ada watchdog's initial memory stress test failed.");
            Put_Line
               (File_Handle,
                "This suggests the machine may not have enough free RAM to run the full application stack reliably.");
        end Write_Flag_Content;

        -- Create the incapability flag in both locations.
        procedure Create_Incapability_Flag is
            Flag_File : File_Type;
        begin
            if not Flag_File_Exists then
                -- Create in root directory
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

                -- Create in engine subdirectory
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

        -- (The rest of the procedure is mostly the same, just using the new constants)
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
            Total_MB_Target := 6 * 1024; -- 6 GB
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
        -- (The main allocation loop is unchanged)
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
    use Ada.Command_Line; -- Use the standard library for command line access

    Memory_Check_OK : Boolean;
    Child_PID       : C_Bridge.Process_Id := C_Bridge.No_Process_Id;

    -- These will now be populated from command-line arguments
    Program_To_Run : String (1 .. Argument (1)'Length);
    Process_Args   : String_Access_Array (1 .. Argument_Count - 1);

    -- ===================================================================
    --  PART 5: THE MAIN PROGRAM LOGIC (the BEGIN block)
    -- ===================================================================
begin
    Put_Line ("=====================================================");
    Put_Line (" ADA WATCHDOG (LEVEL 2) INITIALIZING...");
    Put_Line ("=====================================================");

    -- First, check if we were given a command to supervise.
    if Argument_Count < 1 then
        Put_Line ("FATAL: Ada Watchdog requires a command to supervise.");
        Put_Line ("Usage: ./watchdog_thread2 <program_path> [args...]");
        return; -- Exit immediately

    end if;

    -- Populate the program and arguments from the command line.
    -- Argument(0) is the watchdog's own name.
    -- Argument(1) is the program it should run.
    -- Argument(2)... are the arguments for that program.
    Program_To_Run := Argument (1);
    for I in Process_Args'Range loop
        Process_Args (I) := new String'(Argument (I + 1));
    end loop;

    Put_Line ("Supervision Target: " & Program_To_Run);
    if Process_Args'Length > 0 then
        Put_Line ("Target Arguments:");
        for Arg of Process_Args loop
            Put_Line ("  " & Arg.all);
        end loop;
    end if;
    New_Line;

    Perform_Memory_Check (Success => Memory_Check_OK);

    if Memory_Check_OK then
        Put_Line ("--- Watchdog entering active monitoring mode. ---");
        New_Line;

        declare
            -- We will print a health check message every 6 cycles (30 seconds).
            Health_Check_Interval : constant Natural := 6;
            Cycle_Counter         : Natural := 0;
        begin
            loop
                if not C_Bridge.Is_Alive (Child_PID) then
                    -- If the process is dead, reset the counter and restart it.
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
                        delay 5.0; -- Add an extra delay on spawn failure
                    end if;
                else
                    -- The process is alive. Increment the counter.
                    Cycle_Counter := Cycle_Counter + 1;

                    -- If the counter reaches our interval, print a health check and reset it.
                    if Cycle_Counter >= Health_Check_Interval then
                        Put_Line
                           ("Watchdog_Thread2: [OK] Health check passed. Monitored process (PID "
                            & C_Bridge.Process_Id'Image (Child_PID)
                            & ") is alive.");
                        Cycle_Counter := 0;
                    end if;
                end if;

                -- Wait 5 seconds before the next check.
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
