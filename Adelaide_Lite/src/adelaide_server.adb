pragma SPARK_Mode (Off);

with AnsiAda;
with Ada.Text_IO; use Ada.Text_IO;
with Ada.Exceptions;
with Ada.Command_Line;
with Ada.Real_Time; use Ada.Real_Time;
with Adelaide_Server_Pkg;
with Model_Manager;
with Knowledge_Manager;
with Scheduler_Manager;
with Watchdog_Manager;
with Watchdog_IPC;
with AWS.Config.Set;
with AWS.Server;
with Moonshine_Interface;
with ELP_Queue;

--  ===========================================================================
--  SERVER QUIRKS & DISCOVERED WORKAROUNDS
--  ===========================================================================
--  [QUIRK-S01] [ALL] Pre-existing unload crash (exit code -1) — FIXED
--  After QWEN_0_8B processes a request and ELP0 releases the model, the
--  server could crash with exit code -1 (kratos signal isolation).  Root
--  cause: Idle_Monitor unloaded QWEN_0_8B via Llama_Free after 30s
--  inactivity, triggering a ggml-metal GPU buffer race.  FIXED [macOS]:
--  QWEN_0_8B is now exempt from Idle_Monitor unloading and kept permanently
--  loaded.  LINUX-COMPAT / Android-Termux: On Linux (no ggml-metal) remove
--  the Qwen_0_8B exemption in Idle_Monitor to allow aggressive unloading.
--  For smartphone / Termux targets, also consider lowering Idle_Monitor
--  timeout from 30s to 10-15s for tighter memory pressure response.
--  See QUIRK-M03 for details.
--
--  [QUIRK-S02] [ALL] Port 11420 binding with retry
--  The server tries to bind to port 11420 with up to 3 retries (2s apart).
--  If the port is still in use (e.g., stale server from a previous crash),
--  a [BUGCHECK] message is printed with the port conflict details.
--  Kill the stale process: kill $(lsof -ti:11420)
--
--  [QUIRK-S03] [ALL] ELP Queue utilization logging
--  Every 5 seconds, the main loop logs ELP queue depth and utilization.
--  This is the primary diagnostic for whether background tasks (ELP0) are
--  keeping up or the system is overloaded with user requests (ELP1).
--
--  [QUIRK-S04] [macOS] Moonshine model path
--  The Moonshine STT model is loaded from a hardcoded relative path:
--    "../moonshine/models/download.moonshine.ai/model/medium-streaming-en/quantized"
--  LINUX-COMPAT: This path is the same on Linux, but the model files may
--  reside at a different location.  The path is relative to the CWD at
--  startup (Adelaide_Lite/ when run via run.py).
--  ===========================================================================

procedure Adelaide_Server is
   WS   : AWS.Server.HTTP;
   Conf : AWS.Config.Object := AWS.Config.Get_Current;

   Max_Retries : constant := 3;
   Retry_Count : Natural := 0;
   Started     : Boolean := False;
begin
   begin
      Put_Line ("    ___       __     __      _     __ ");
      Put_Line ("   /   | ____/ /__  / /___ _(_)___/ /__ ");
      Put_Line ("  / /| |/ __  / _ \/ / __ `/ / __  / _ \ ");
      Put_Line (" / ___ / /_/ /  __/ / /_/ / / /_/ /  __/ ");
      Put_Line ("/_/  |_\__,_/\___/_/\__,_/_/\__,_/\___/ ");
      Put_Line ("");
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Initializing Adelaide Intelligence Backend...");
      Model_Manager.Initialize;
      Knowledge_Manager.Initialize;
      Scheduler_Manager.Initialize;

      --  Start file-based IPC for the external watchdog process
      Watchdog_IPC.Init;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Connecting to Kokoro-ONNX sidecar (TTS)...");
      --  Kokoro does not require C-level init in Ada since it runs on Python Sidecar
      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Initializing Moonshine (STT)...");
      Moonshine_Interface.Init_Moonshine
        ("../moonshine/models/download.moonshine.ai/model/medium-streaming-en/quantized");

      AWS.Config.Set.Server_Port (Conf, 11420);
      AWS.Config.Set.Server_Host (Conf, "0.0.0.0");
      AWS.Config.Set.Reuse_Address (Conf, True);

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Adelaide-Lite Server starting on port 11420...");

      while not Started and then Retry_Count < Max_Retries loop
         begin
            AWS.Server.Start
              (Web_Server => WS,
               Callback   => Adelaide_Server_Pkg.Dispatch'Access,
               Config     => Conf);
            Started := True;
         exception
            when E : others =>
               Retry_Count := Retry_Count + 1;
               if Retry_Count < Max_Retries then
                  Put_Line
                    ("[Warning] Port 11420 might be bound. Retrying in " &
                     "2 seconds (" &
                     Natural'Image (Retry_Count) & "/" &
                     Natural'Image (Max_Retries) & ")...");
                  delay 2.0;
               else
                  --  Print [BUGCHECK] with the issue verbose in Red font
                  Put_Line (Character'Val (27) & "[31m" &
                            "[BUGCHECK] Failed to bind to port 11420: " &
                            Ada.Exceptions.Exception_Message (E) &
                            Character'Val (27) & "[0m");
                  Ada.Command_Line.Set_Exit_Status
                    (Ada.Command_Line.Failure);
                  return;
               end if;
         end;
      end loop;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Initializing index crawl...");
      Knowledge_Manager.Start_Tasks;

      Put_Line (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
                AnsiAda.Reset & " Server is UP. Press Q to shutdown (or kill if background).");

      --  Queue heartbeat counter
      Heartbeat_Count : Natural := 0;

      --  Avoid Get_Line failure in background
      loop
         Watchdog_Manager.AWS_Server_Monitor.Heartbeat (Clock);
         Watchdog_IPC.Write_Heartbeat;
         Heartbeat_Count := Heartbeat_Count + 1;
         if Heartbeat_Count >= 5 then
            Heartbeat_Count := 0;
            Ada.Text_IO.Put_Line
              (AnsiAda.Foreground (AnsiAda.Light_Blue) & "[Main]" &
               AnsiAda.Reset & " ELP Queue: " &
               ELP_Queue.Utilization'Img & "% full (" &
               ELP_Queue.Depth'Img & " pending)");
         end if;
         delay 1.0;
      end loop;

   exception
      when E : others =>
         Put_Line (AnsiAda.Foreground (AnsiAda.Red) & "[FATAL]" &
                   AnsiAda.Reset & " Server Error: " &
                   Ada.Exceptions.Exception_Message (E));
         Ada.Command_Line.Set_Exit_Status (Ada.Command_Line.Failure);
   end;
end Adelaide_Server;
