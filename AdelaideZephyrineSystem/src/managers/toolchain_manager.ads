pragma SPARK_Mode (Off);
-- thread: Toolchain management requires protection
package Toolchain_Manager is

   --  Checks system toolchain and heals dependencies if any are missing.
   procedure Verify_And_Heal;

   procedure Start_Orchestrator;

end Toolchain_Manager;
