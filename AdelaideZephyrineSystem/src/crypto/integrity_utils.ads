pragma SPARK_Mode (On);
with Interfaces; use Interfaces;

package Integrity_Utils is

   type Byte_Array is array (Positive range <>) of Unsigned_8;
   type CRC_Array is array (Positive range <>) of Unsigned_32;

   --  CRC-32 calculation using the standard IEEE polynomial 0xEDB88320
   function Calculate_CRC32 (Data : Byte_Array) return Unsigned_32 with Pre => True, Post => True;

   --  Generates an XOR parity block for N blocks of size Block_Size
   procedure Generate_Parity (
     Data       : Byte_Array;
     Block_Size : Positive;
     Parity     : in out Byte_Array
   ) with
     Pre => Data'Length > 0 and then
            Block_Size > 0 and then
            Data'Length mod Block_Size = 0 and then
            Parity'Length = Block_Size;

   --  Reconstructs a corrupted block inside Data using the parity block
   procedure Reconstruct_Block (
     Data          : in out Byte_Array;
     Block_Size    : Positive;
     Corrupt_Index : Positive;
     Parity        : Byte_Array
   ) with
     Pre => Data'Length > 0 and then
            Block_Size > 0 and then
            Data'Length mod Block_Size = 0 and then
            Corrupt_Index <= Data'Length / Block_Size and then
            Parity'Length = Block_Size;

   --  Automatically identifies and fixes a corrupted block using CRC checks and parity
   procedure Self_Patch (
     Data          : in out Byte_Array;
     Block_Size    : Positive;
     Expected_CRCs : CRC_Array;
     Parity        : Byte_Array;
     Success       : out Boolean
   ) with
     Pre => Data'Length > 0 and then
            Block_Size > 0 and then
            Data'Length mod Block_Size = 0 and then
            Expected_CRCs'Length = Data'Length / Block_Size and then
            Parity'Length = Block_Size;

   --  Returns True if the data appears to be binary (contains NUL bytes or high non-printable ratio)
   function Is_Binary (Data : Byte_Array) return Boolean with Pre => True, Post => True;

end Integrity_Utils;
