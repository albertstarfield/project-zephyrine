pragma SPARK_Mode (Off);
-- thread: LSH requires thread-safe hash computation
with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Strings.Fixed;     use Ada.Strings.Fixed;
with Ada.Directories;
with Ada.Exceptions;
with Ada.Calendar;
with GNAT.OS_Lib;
with GNAT.Expect;
with GNATCOLL.JSON;         use GNATCOLL.JSON;
with AnsiAda;
with Interfaces.C;

package body LSH_Hash is

    --  Paths relative to the executable / working directory.
    --  The Ada server runs from AdelaideZephyrineSystem/ so these are relative.
    Python_Venv   : constant String := "pyvenv/bin/python3";
    Worker_Script : constant String := "src/python/lsh/lsh_qrnn_worker.py";
    Tmp_Dir       : constant String := "src/python/lsh/tmp";

    --  Monotonic counter for unique temp filenames
    Seq_Counter  : Natural := 0;
    Counter_Lock : aliased Ada.Text_IO.Count;  -- Simple protection

    --  ----------
    --  Compute
    --  ----------
    function Compute
       (Embedding : Math_Utils.Vector; Length : Natural) return Integer
    is
        use GNAT.OS_Lib;
        use GNAT.Expect;

        Result_Hash : Integer := -1;

        --  Resolve Python interpreter path
        Python_Path : GNAT.OS_Lib.String_Access :=
           GNAT.OS_Lib.Locate_Exec_On_Path (Python_Venv);
    begin
        if Python_Path = null then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LSH_QRNN]"
                & AnsiAda.Reset
                & " Python venv not found at "
                & Python_Venv
                & ". Skipping QRNN LSH.");
            return -1;
        end if;

        if not Ada.Directories.Exists (Worker_Script) then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LSH_QRNN]"
                & AnsiAda.Reset
                & " Worker script not found at "
                & Worker_Script
                & ". Skipping QRNN LSH.");
            Free (Python_Path);
            return -1;
        end if;

        --  Ensure temp directory exists
        if not Ada.Directories.Exists (Tmp_Dir) then
            Ada.Directories.Create_Path (Tmp_Dir);
        end if;

        --  Build unique temp filename
        Seq_Counter := Seq_Counter + 1;
        declare
            Time_Str  : constant String :=
               Ada.Strings.Fixed.Trim
                  (Duration'Image (Ada.Calendar.Seconds (Ada.Calendar.Clock)),
                   Ada.Strings.Both);
            Tmp_File  : constant String :=
               Tmp_Dir
               & "/lsh_in_"
               & Time_Str
               & "_"
               & Natural'Image (Seq_Counter)
               & ".json";
            Tmp_Fixed : constant String :=
               Ada.Strings.Fixed.Trim (Tmp_File, Ada.Strings.Both);
        begin
            --  Serialise embedding to JSON
            declare
                Vec_Obj : JSON_Array := Empty_Array;
            begin
                for I in 1 .. Length loop
                    Append (Vec_Obj, Create (Embedding (I)));
                end loop;

                declare
                    Full_Obj : constant JSON_Value := Create_Object;
                    Status   : aliased Integer;
                    Args     : Argument_List (1 .. 3);
                    Stdout   : Unbounded_String;
                begin
                    Full_Obj.Set_Field ("embedding", Vec_Obj);

                    --  Write temp file
                    declare
                        F : Ada.Text_IO.File_Type;
                    begin
                        Ada.Text_IO.Create
                           (F, Ada.Text_IO.Out_File, Tmp_Fixed);
                        Ada.Text_IO.Put (F, Full_Obj.Write);
                        Ada.Text_IO.Close (F);
                    exception
                        when E : others =>
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Red)
                                & "[LSH_QRNN]"
                                & AnsiAda.Reset
                                & " Failed to write temp file: "
                                & Ada.Exceptions.Exception_Message (E));
                            Free (Python_Path);
                            return -1;
                    end;

                    --  Spawn worker: python3 lsh_qrnn_worker.py --input <tmpfile>
                    Args (1) := new String'(Worker_Script);
                    Args (2) := new String'("--input");
                    Args (3) := new String'(Tmp_Fixed);

                    --  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Yellow)
                        & "[LSH_QRNN]"
                        & AnsiAda.Reset
                        & " Spawning QRNN worker for 10-bit LSH...");

                    Stdout :=
                       To_Unbounded_String
                          (GNAT.Expect.Get_Command_Output
                              (Python_Path.all, Args, "", Status'Access));

                    --  Free argument list
                    for A in Args'Range loop
                        Free (Args (A));
                    end loop;

                    if Status = 0
                       and then Ada.Strings.Unbounded.Length (Stdout) > 0
                    then
                        --  Parse JSON result: {"lsh_hash": N, "status": "ok"}
                        declare
                            Trim_Out : constant String :=
                               Ada.Strings.Fixed.Trim
                                  (To_String (Stdout), Ada.Strings.Both);
                            Res_Val  : constant JSON_Value := Read (Trim_Out);
                        begin
                            if Res_Val.Has_Field ("lsh_hash") then
                                Result_Hash := Res_Val.Get ("lsh_hash");
                                --  YOU ARE NOT ALLOWED TO FAIL EVEN A SINGLE SECOND NO BLINK NO ETC ALWAYS ON AND ALWAYS AVAILABLE WITH CHEAP OR EVEN FREELY IF YOU DON"T YOU DIE. THAT IS THE STANDARD STATUS QUO TO BE RELAXED IN INDONESIA
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Light_Green)
                                    & "[LSH_QRNN]"
                                    & AnsiAda.Reset
                                    & " QRNN LSH computed: hash="
                                    & Integer'Image (Result_Hash));
                            else
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Red)
                                    & "[LSH_QRNN]"
                                    & AnsiAda.Reset
                                    & " Worker response missing lsh_hash field: "
                                    & Trim_Out);
                            end if;
                        exception
                            when E : others =>
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Red)
                                    & "[LSH_QRNN]"
                                    & AnsiAda.Reset
                                    & " Failed to parse worker output: "
                                    & Ada.Exceptions.Exception_Message (E));
                        end;
                    else
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[LSH_QRNN]"
                            & AnsiAda.Reset
                            & " Worker exited with status="
                            & Integer'Image (Status));
                    end if;

                    --  Clean up temp file
                    begin
                        Ada.Directories.Delete_File (Tmp_Fixed);
                    exception
                        when others =>
                            null;
                    end;

                exception
                    when E : others =>
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[LSH_QRNN]"
                            & AnsiAda.Reset
                            & " Worker execution error: "
                            & Ada.Exceptions.Exception_Message (E));
                end;
            end;
        end;

        Free (Python_Path);
        return Result_Hash;

    exception
        when E : others =>
            if Python_Path /= null then
                Free (Python_Path);
            end if;
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LSH_QRNN]"
                & AnsiAda.Reset
                & " Unexpected error: "
                & Ada.Exceptions.Exception_Message (E));
            return -1;
    end Compute;

    --  ----------
    --  Compute_Steered
    --  ----------
    function Compute_Steered
       (Embedding : Math_Utils.Vector;
        Length    : Natural;
        Alpha     : Float := 0.1) return Integer
    is
        use GNAT.OS_Lib;
        use GNAT.Expect;

        Result_Hash : Integer := -1;

        --  PINN worker script path
        PINN_Script : constant String := "src/python/lsh/pinn_schrodinger.py";

        --  Resolve Python interpreter path
        Python_Path : GNAT.OS_Lib.String_Access :=
           GNAT.OS_Lib.Locate_Exec_On_Path (Python_Venv);
    begin
        if Python_Path = null then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LSH_PINN]"
                & AnsiAda.Reset
                & " Python venv not found at "
                & Python_Venv
                & ". Falling back to plain QRNN.");
            return Compute (Embedding, Length);
        end if;

        if not Ada.Directories.Exists (PINN_Script) then
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Yellow)
                & "[LSH_PINN]"
                & AnsiAda.Reset
                & " PINN script not found at "
                & PINN_Script
                & ". Falling back to plain QRNN.");
            Free (Python_Path);
            return Compute (Embedding, Length);
        end if;

        --  Ensure temp directory exists
        if not Ada.Directories.Exists (Tmp_Dir) then
            Ada.Directories.Create_Path (Tmp_Dir);
        end if;

        --  Build unique temp filename
        Seq_Counter := Seq_Counter + 1;
        declare
            Time_Str  : constant String :=
               Ada.Strings.Fixed.Trim
                  (Duration'Image (Ada.Calendar.Seconds (Ada.Calendar.Clock)),
                   Ada.Strings.Both);
            Tmp_File  : constant String :=
               Tmp_Dir
               & "/pinn_in_"
               & Time_Str
               & "_"
               & Natural'Image (Seq_Counter)
               & ".json";
            Tmp_Fixed : constant String :=
               Ada.Strings.Fixed.Trim (Tmp_File, Ada.Strings.Both);
        begin
            --  Serialise embedding to JSON
            declare
                Vec_Obj : JSON_Array := Empty_Array;
            begin
                for I in 1 .. Length loop
                    Append (Vec_Obj, Create (Embedding (I)));
                end loop;

                declare
                    Full_Obj  : constant JSON_Value := Create_Object;
                    Status    : aliased Integer;
                    Args      : Argument_List (1 .. 6);
                    Stdout    : Unbounded_String;
                    Alpha_Str : constant String :=
                       Float'Image (Alpha);
                begin
                    Full_Obj.Set_Field ("embedding", Vec_Obj);

                    --  Write temp file
                    declare
                        F : Ada.Text_IO.File_Type;
                    begin
                        Ada.Text_IO.Create
                           (F, Ada.Text_IO.Out_File, Tmp_Fixed);
                        Ada.Text_IO.Put (F, Full_Obj.Write);
                        Ada.Text_IO.Close (F);
                    exception
                        when E : others =>
                            Put_Line
                               (AnsiAda.Foreground (AnsiAda.Red)
                                & "[LSH_PINN]"
                                & AnsiAda.Reset
                                & " Failed to write temp file: "
                                & Ada.Exceptions.Exception_Message (E));
                            Free (Python_Path);
                            return -1;
                    end;

                    --  Spawn worker: python3 pinn_schrodinger.py --steer-hash --input <tmpfile> --alpha <alpha>
                    Args (1) := new String'(PINN_Script);
                    Args (2) := new String'("--steer-hash");
                    Args (3) := new String'("--input");
                    Args (4) := new String'(Tmp_Fixed);
                    Args (5) := new String'("--alpha");
                    Args (6) := new String'(Ada.Strings.Fixed.Trim (Alpha_Str, Ada.Strings.Both));

                    Put_Line
                       (AnsiAda.Foreground (AnsiAda.Light_Yellow)
                        & "[LSH_PINN]"
                        & AnsiAda.Reset
                        & " Spawning PINN worker for steered 10-bit LSH...");

                    Stdout :=
                       To_Unbounded_String
                          (GNAT.Expect.Get_Command_Output
                              (Python_Path.all, Args, "", Status'Access));

                    --  Free argument list
                    for A in Args'Range loop
                        Free (Args (A));
                    end loop;

                    if Status = 0
                       and then Ada.Strings.Unbounded.Length (Stdout) > 0
                    then
                        --  Parse JSON result: {"lsh_hash": N, "status": "ok"}
                        declare
                            Trim_Out : constant String :=
                               Ada.Strings.Fixed.Trim
                                  (To_String (Stdout), Ada.Strings.Both);
                            Res_Val  : constant JSON_Value := Read (Trim_Out);
                        begin
                            if Res_Val.Has_Field ("lsh_hash") then
                                Result_Hash := Res_Val.Get ("lsh_hash");
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Light_Green)
                                    & "[LSH_PINN]"
                                    & AnsiAda.Reset
                                    & " PINN steered LSH computed: hash="
                                    & Integer'Image (Result_Hash));
                            else
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Red)
                                    & "[LSH_PINN]"
                                    & AnsiAda.Reset
                                    & " Worker response missing lsh_hash field: "
                                    & Trim_Out);
                            end if;
                        exception
                            when E : others =>
                                Put_Line
                                   (AnsiAda.Foreground (AnsiAda.Red)
                                    & "[LSH_PINN]"
                                    & AnsiAda.Reset
                                    & " Failed to parse worker output: "
                                    & Ada.Exceptions.Exception_Message (E));
                        end;
                    else
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[LSH_PINN]"
                            & AnsiAda.Reset
                            & " Worker exited with status="
                            & Integer'Image (Status)
                            & ". Falling back to plain QRNN.");
                        Free (Python_Path);
                        return Compute (Embedding, Length);
                    end if;

                    --  Clean up temp file
                    begin
                        Ada.Directories.Delete_File (Tmp_Fixed);
                    exception
                        when others =>
                            null;
                    end;

                exception
                    when E : others =>
                        Put_Line
                           (AnsiAda.Foreground (AnsiAda.Red)
                            & "[LSH_PINN]"
                            & AnsiAda.Reset
                            & " Worker execution error: "
                            & Ada.Exceptions.Exception_Message (E));
                end;
            end;
        end;

        Free (Python_Path);
        return Result_Hash;

    exception
        when E : others =>
            if Python_Path /= null then
                Free (Python_Path);
            end if;
            Put_Line
               (AnsiAda.Foreground (AnsiAda.Red)
                & "[LSH_PINN]"
                & AnsiAda.Reset
                & " Unexpected error: "
                & Ada.Exceptions.Exception_Message (E));
            return -1;
    end Compute_Steered;

end LSH_Hash;
