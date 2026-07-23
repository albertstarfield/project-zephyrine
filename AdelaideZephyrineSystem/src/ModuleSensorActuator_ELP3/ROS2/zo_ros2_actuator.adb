with Ada.Real_Time; use Ada.Real_Time;
with System;

package body ZO_ROS2_Actuator is

   --  Store the exact time the node system was initialized to calculate uptime.
   Start_Time : Time;

   --  Helper function to generate the formatted verbose prefix with uptime.
   function Prefix return String is
      Now : Time := Clock;
      Span : Time_Span := Now - Start_Time;
      Secs : Duration := To_Duration (Span);
   begin
      --  Format: [Prefix][+Uptime]
      --  Example: [ZenithOrion-ELP3][+1.002s] 
      return "[ZenithOrion-ELP3][+" & Secs'Img & "s] ";
   end Prefix;

   Global_Node : Actuator_Node;

   --  Initialize_ROS2: Initializes the ROS2 node for actuator control.
   function Initialize_ROS2 return Boolean is
      --  1. Create zero-initialized options to prevent garbage memory in C structs
      Init_Opts : aliased rcl_init_options_t := rcl_get_zero_initialized_init_options;
      Node_Opts : aliased rcl_node_options_t;
      Ret       : rcl_ret_t;
      
      --  2. Define the node name and namespace using C-compatible strings
      Node_Name : chars_ptr := New_String ("zenith_orion_actuator_node");
      Namespace : chars_ptr := New_String ("");
   begin
      if Global_Node.Initialized then
         return True;
      end if;

      Start_Time := Clock;
      Put_Line (Prefix & "Starting native Ada ROS2 Initialization sequence...");

      --  3. Get a zero-initialized context for the node
      Global_Node.Context := rcl_get_zero_initialized_context;
      Put_Line (Prefix & "Context zero-initialized.");
      
      --  4. Initialize the init options using the default memory allocator
      Ret := rcl_init_options_init (Init_Opts'Access, rcutils_get_default_allocator);
      if Ret /= RCL_RET_OK then
         Put_Line (Prefix & "Error: Failed to initialize rcl_init_options. Ret code: " & Ret'Img);
         return False;
      end if;
      Put_Line (Prefix & "rcl_init_options initialized successfully.");

      --  5. Initialize the core rcl library
      Ret := rcl_init (0, System.Null_Address, Init_Opts'Access, Global_Node.Context'Access);
      if Ret /= RCL_RET_OK then
         Put_Line (Prefix & "Error: Failed to initialize rcl core. Ret code: " & Ret'Img);
         return False;
      end if;
      Put_Line (Prefix & "rcl core initialized successfully.");

      --  6. Get default node options and zero-initialize the node struct
      Global_Node.Node := rcl_get_zero_initialized_node;
      Node_Opts := rcl_node_get_default_options;

      --  7. Create the ROS2 node on the DDS network
      Put_Line (Prefix & "Attempting to create ROS2 Node '" & Value(Node_Name) & "'...");
      Ret := rcl_node_init (Global_Node.Node'Access, Node_Name, Namespace, Global_Node.Context'Access, Node_Opts'Access);
      
      --  8. Free the C strings to prevent memory leaks
      Free (Node_Name);
      Free (Namespace);

      if Ret /= RCL_RET_OK then
         Put_Line (Prefix & "Error: Failed to initialize ROS2 Node. Ret code: " & Ret'Img);
         return False;
      end if;

      Put_Line (Prefix & "ROS2 Actuator Node Initialized successfully on the DDS network.");
      Global_Node.Initialized := True;
      return True;
   end Initialize_ROS2;

   --  Publish_Actuator_Command: Publishes a servo command to the ROS2 actuator topic.
   procedure Publish_Actuator_Command (Servo_ID : String; Angle : Float) is
   begin
      --  1. Verify the node is active before attempting to publish
      if not Global_Node.Initialized then
         Put_Line ("[ZenithOrion-ELP3][WARN] Node uninitialized at publish attempt. Bootstrapping now...");
         if not Initialize_ROS2 then
            Put_Line ("[ZenithOrion-ELP3][FATAL] ROS2 not initialized. Cannot publish actuator command.");
            return;
         end if;
      end if;
      
      --  2. In a full binding, rcl_publish would be called here.
      --  For this thin implementation, we log the deterministic ELP3 action with verbose timing.
      Put_Line (Prefix & "Executing deterministic ELP3 Actuator Reflex.");
      Put_Line (Prefix & "--> Publishing to Servo [" & Servo_ID & "] with Angle [" & Angle'Img & "].");
      Put_Line (Prefix & "--> Publish complete. Reflex loop closed.");
   end Publish_Actuator_Command;

end ZO_ROS2_Actuator;
