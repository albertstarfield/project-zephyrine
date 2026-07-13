--  ── Master Key Store Implementation ──────────────────────────────────────────
--  SPARK-verified 512-bit key storage implementation.
--  Uses volatile writes to ensure key material is properly cleared.
--  ──────────────────────────────────────────────────────────────────────────────

package body Master_Key_Store
  with SPARK_Mode => On
is

   --  ── Set_Key ───────────────────────────────────────────────────────────────
   procedure Set_Key (K : Key_Type) is
   begin
      Key := K;
      Key_Valid := True;
   end Set_Key;

   --  ── Get_Key ───────────────────────────────────────────────────────────────
   function Get_Key return Key_Type is
   begin
      if Key_Valid then
         return Key;
      else
         return Empty_Key;
      end if;
   end Get_Key;

   --  ── Clear_Key ─────────────────────────────────────────────────────────────
   procedure Clear_Key is
   begin
      --  Volatile write prevents compiler from optimizing away the clear
      Key := (others => 0);
      Key_Valid := False;
   end Clear_Key;

   --  ── Is_Set ────────────────────────────────────────────────────────────────
   function Is_Set return Boolean is
   begin
      return Key_Valid;
   end Is_Set;

   --  ── Get_AES_Part ──────────────────────────────────────────────────────────
   function Get_AES_Part return Key_Type is
      Result : Key_Type := (others => 0);
   begin
      if Key_Valid then
         --  Copy first 32 bytes (indices 1..32) for AES-256
         for I in Key_Index range 1 .. 32 loop
            Result (I) := Key (I);
         end loop;
      end if;
      return Result;
   end Get_AES_Part;

end Master_Key_Store;
