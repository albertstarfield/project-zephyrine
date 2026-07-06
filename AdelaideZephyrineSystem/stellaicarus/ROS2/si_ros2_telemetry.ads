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

   function Initialize_ROS2 return Boolean;
   procedure Poll_Telemetry;

end SI_ROS2_Telemetry;
