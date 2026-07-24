with ROS2_RCL_Bindings; use ROS2_RCL_Bindings;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with Ada.Text_IO; use Ada.Text_IO;

package SI_ROS2_Telemetry is

   --  ELP2: StellaIcarus Fast-Reflex Telemetry (Sensors)
   
   type Telemetry_Node is record
      Context : aliased rcl_context_t;
      Node    : aliased rcl_node_t;
      Initialized : Boolean := False;
   end record;

   --  Initialize_ROS2: Initializes the ROS2 node and communication infrastructure.
   function Initialize_ROS2 return Boolean with Pre => True, Post => True;
   --  Poll_Telemetry: Polls telemetry data from sensors and publishes to ROS2 topics.
   procedure Poll_Telemetry with Pre => True, Post => True;

end SI_ROS2_Telemetry;
