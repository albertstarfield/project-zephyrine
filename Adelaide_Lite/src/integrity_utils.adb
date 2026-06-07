pragma SPARK_Mode (On);
package body Integrity_Utils is

   ---------------------
   -- Calculate_CRC32 --
   ---------------------
   function Calculate_CRC32 (Data : Byte_Array) return Unsigned_32 is
      CRC : Unsigned_32 := 16#FFFF_FFFF#;
   begin
      for I in Data'Range loop
         CRC := CRC xor Unsigned_32 (Data (I));
         for Bit in 1 .. 8 loop
            if (CRC and 1) /= 0 then
               CRC := Shift_Right (CRC, 1) xor 16#EDB8_8320#;
            else
               CRC := Shift_Right (CRC, 1);
            end if;
         end loop;
      end loop;
      return not CRC;
   end Calculate_CRC32;

   ---------------------
   -- Generate_Parity --
   ---------------------
   procedure Generate_Parity (
     Data       : Byte_Array;
     Block_Size : Positive;
     Parity     : in out Byte_Array
   ) is
      Num_Blocks : constant Positive := Data'Length / Block_Size;
      Data_Start : constant Positive := Data'First;
      Par_Start  : constant Positive := Parity'First;
   begin
      --  Initialize parity array to zero
      for I in 0 .. Block_Size - 1 loop
         Parity (Par_Start + I) := 0;
      end loop;

      --  XOR all blocks
      for B_Idx in 0 .. Num_Blocks - 1 loop
         for I in 0 .. Block_Size - 1 loop
            Parity (Par_Start + I) := Parity (Par_Start + I) xor
              Data (Data_Start + B_Idx * Block_Size + I);
         end loop;
      end loop;
   end Generate_Parity;

   ------------------------
   -- Reconstruct_Block --
   ------------------------
   procedure Reconstruct_Block (
     Data          : in out Byte_Array;
     Block_Size    : Positive;
     Corrupt_Index : Positive;
     Parity        : Byte_Array
   ) is
      Num_Blocks    : constant Positive := Data'Length / Block_Size;
      Data_Start    : constant Positive := Data'First;
      Par_Start     : constant Positive := Parity'First;
      Corrupt_Start : constant Positive :=
        Data_Start + (Corrupt_Index - 1) * Block_Size;
   begin
      --  Set corrupt block to parity values initially
      for I in 0 .. Block_Size - 1 loop
         Data (Corrupt_Start + I) := Parity (Par_Start + I);
      end loop;

      --  XOR with all other blocks
      for B_Idx in 0 .. Num_Blocks - 1 loop
         if B_Idx /= Corrupt_Index - 1 then
            for I in 0 .. Block_Size - 1 loop
               Data (Corrupt_Start + I) := Data (Corrupt_Start + I) xor
                 Data (Data_Start + B_Idx * Block_Size + I);
            end loop;
         end if;
      end loop;
   end Reconstruct_Block;

   ----------------
   -- Self_Patch --
   ----------------
   procedure Self_Patch (
     Data          : in out Byte_Array;
     Block_Size    : Positive;
     Expected_CRCs : CRC_Array;
     Parity        : Byte_Array;
     Success       : out Boolean
   ) is
      Num_Blocks    : constant Positive := Data'Length / Block_Size;
      Data_Start    : constant Positive := Data'First;
      Corrupt_Count : Natural := 0;
      Corrupt_Idx   : Positive := 1;
   begin
      Success := True;

      --  Identify corrupted block using CRC
      for B_Idx in 1 .. Num_Blocks loop
         pragma Loop_Invariant (Corrupt_Count <= B_Idx - 1);
         pragma Loop_Invariant (if Corrupt_Count = 1 then Corrupt_Idx <= Num_Blocks);
         declare
            Start_Pos : constant Positive :=
              Data_Start + (B_Idx - 1) * Block_Size;
            End_Pos   : constant Positive := Start_Pos + (Block_Size - 1);
            Actual_CRC : constant Unsigned_32 :=
              Calculate_CRC32 (Data (Start_Pos .. End_Pos));
         begin
            if Actual_CRC /=
              Expected_CRCs (Expected_CRCs'First + (B_Idx - 1))
            then
               Corrupt_Count := Corrupt_Count + 1;
               Corrupt_Idx   := B_Idx;
            end if;
         end;
      end loop;

      if Corrupt_Count = 1 then
         --  Single block corruption, can be fixed with parity
         Reconstruct_Block (Data, Block_Size, Corrupt_Idx, Parity);
      elsif Corrupt_Count > 1 then
         --  Too many corruptions to fix with single parity block
         Success := False;
      end if;
      --  If Corrupt_Count = 0, nothing to do, Success remains True
   end Self_Patch;

   ---------------
   -- Is_Binary --
   ---------------
   function Is_Binary (Data : Byte_Array) return Boolean is
      Non_Printable : Natural := 0;
   begin
      if Data'Length = 0 then
         return False;
      end if;

      for I in Data'Range loop
         pragma Loop_Invariant (Non_Printable <= I - Data'First);

         --  Check for NUL byte
         if Data (I) = 0 then
            return True;
         end if;

         --  Count non-printable characters (heuristic)
         --  ASCII 32-126 are printable, plus CR, LF, TAB
         if not (Data (I) in 32 .. 126 or else
                 Data (I) = 9 or else
                 Data (I) = 10 or else
                 Data (I) = 13)
         then
            Non_Printable := Non_Printable + 1;
         end if;
      end loop;

      --  If more than 30% are non-printable, consider it binary
      --  Uses Unsigned_64 to prevent integer overflow and avoid Float precision/overflow proofs
      return Unsigned_64 (Non_Printable) * 10 > Unsigned_64 (Data'Length) * 3;
   end Is_Binary;

end Integrity_Utils;
