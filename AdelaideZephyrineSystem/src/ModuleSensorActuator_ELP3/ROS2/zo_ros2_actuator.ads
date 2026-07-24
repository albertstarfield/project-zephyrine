with ROS2_RCL_Bindings; use ROS2_RCL_Bindings;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Text_IO; use Ada.Text_IO;

package ZO_ROS2_Actuator is

   --  ELP3: ZenithOrion Safety-Critical Actuators (1ms consistent timing)
   
   type Actuator_Node is record
      Context : aliased rcl_context_t;
      Node    : aliased rcl_node_t;
      Initialized : Boolean := False;
   end record;

   --  Initialize_ROS2: Initializes the ROS2 node for actuator control.
   function Initialize_ROS2 return Boolean with Pre => True, Post => True;
   --  Publish_Actuator_Command: Publishes a servo command to the ROS2 actuator topic.
   procedure Publish_Actuator_Command (Servo_ID : String; Angle : Float) with Pre => True, Post => True;

end ZO_ROS2_Actuator;
