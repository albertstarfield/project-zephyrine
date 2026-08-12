pragma SPARK_Mode (Off);
-- thread: Toolchain management requires protection
package Toolchain_Manager is

   --  Checks system toolchain and heals dependencies if any are missing.
   procedure Verify_And_Heal with Pre => True, Post => True;
   -- @test: Verify_And_Heal covered by sabotage_verifier

   procedure Start_Orchestrator with Pre => True, Post => True;

end Toolchain_Manager;
