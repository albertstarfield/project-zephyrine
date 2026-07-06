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

   function Initialize_ROS2 return Boolean;
   procedure Publish_Actuator_Command (Servo_ID : String; Angle : Float);

end ZO_ROS2_Actuator;
