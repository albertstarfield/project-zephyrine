package Toolchain_Manager is
   pragma Spark_Mode (Off);

   --  Checks system toolchain and heals dependencies if any are missing.
   procedure Verify_And_Heal;

end Toolchain_Manager;
