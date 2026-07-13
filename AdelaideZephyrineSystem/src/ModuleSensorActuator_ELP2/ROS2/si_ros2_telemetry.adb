with Ada.Real_Time; use Ada.Real_Time;
with System;

package body SI_ROS2_Telemetry is

   --  Store the exact time the node system was initialized to calculate uptime.
   Start_Time : Time;

   --  Helper function to generate the formatted verbose prefix with uptime.
   function Prefix return String is
      Now : Time := Clock;
      Span : Time_Span := Now - Start_Time;
      Secs : Duration := To_Duration (Span);
   begin
      --  Format: [Prefix][+Uptime]
      --  Example: [StellaIcarus-ELP2][+0.050s] 
      return "[StellaIcarus-ELP2][+" & Secs'Img & "s] ";
   end Prefix;

   Global_Node : Telemetry_Node;

   function Initialize_ROS2 return Boolean is
      --  1. Create zero-initialized options to prevent garbage memory in C structs
      Init_Opts : aliased rcl_init_options_t := rcl_get_zero_initialized_init_options;
      Node_Opts : aliased rcl_node_options_t;
      Ret       : rcl_ret_t;
      
      --  2. Define the node name and namespace using C-compatible strings
      Node_Name : chars_ptr := New_String ("stellaicarus_telemetry_node");
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

      Put_Line (Prefix & "ROS2 Telemetry Node Initialized successfully on the DDS network.");
      Global_Node.Initialized := True;
      return True;
   end Initialize_ROS2;

   procedure Poll_Telemetry is
   begin
      --  1. Verify the node is active before attempting to poll
      if not Global_Node.Initialized then
         Put_Line ("[StellaIcarus-ELP2][WARN] Node uninitialized at poll attempt. Bootstrapping now...");
         if not Initialize_ROS2 then
            Put_Line ("[StellaIcarus-ELP2][FATAL] ROS2 not initialized. Cannot poll telemetry data.");
            return;
         end if;
      end if;
      
      --  2. In a full binding, rcl_take would be called here to poll the DDS subscription.
      --  For this thin implementation, we log the deterministic ELP2 poll with verbose timing.
      Put_Line (Prefix & "Polling ROS2 Telemetry via native Ada rcl interface.");
      --  Example of data processing:
      --  Put_Line (Prefix & "--> Received [JointState]: Pos 1.2 Rad.");
   end Poll_Telemetry;

end SI_ROS2_Telemetry;
