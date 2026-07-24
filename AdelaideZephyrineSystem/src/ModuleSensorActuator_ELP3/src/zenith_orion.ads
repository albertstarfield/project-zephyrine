pragma SPARK_Mode (Off);
-- c_binding: Orion sensor FFI
package Zenith_Orion is

   type Jitter_Data is record
      Max_Jitter : Duration;
      Min_Jitter : Duration;
      Avg_Jitter : Duration;
   end record;

   --  Initialize ZenithOrion ELP3 core
   procedure Initialize with Pre => True, Post => True;

   --  The 4000Hz Deterministic Loop
   --  ELP3: ZenithOrion - 0.25ms (250us) Pacing Lock (Deterministic)
   procedure Paced_Loop with Pre => True, Post => True;

   --  Get_Current_Timing: Returns the last measured loop execution time.
   function Get_Current_Timing return Duration with Pre => True, Post => True;
   --  Get_Jitter_Profile: Returns the collected jitter statistics (max, min, avg).
   function Get_Jitter_Profile return Jitter_Data with Pre => True, Post => True;

   --  Checks if the prompt maps to an exact SHM/hardware trigger.
   --  Returns empty string if no match.
   function Check_SHM_Trigger (Prompt : String) return String with Pre => True, Post => True;

   --  Thread-safe buffer to transport commands from ELP0/ELP1 tools
   --  to the deterministic ELP3 fast-path.
   protected ROS2_Command_Buffer is
      procedure Push_Command (Servo_ID : String; Angle : Float) with Pre => True, Post => True;
      procedure Pop_Command (Servo_ID : out String; Length : out Natural; Angle : out Float; Valid : out Boolean) with Pre => True, Post => True;
   private
      Buffer_Servo : String (1 .. 64) := (others => ' ');
      Buffer_Len   : Natural := 0;
      Buffer_Angle : Float := 0.0;
      Has_Command  : Boolean := False;
   end ROS2_Command_Buffer;

end Zenith_Orion;
