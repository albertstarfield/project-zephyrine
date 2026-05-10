with Interfaces;
with Interfaces.C;
with System;
with Ada.Text_IO;
with Ada.Containers.Vectors;
with Ada.Unchecked_Conversion;

-- FIX #1: The crucial package for pointer arithmetic.
with System.Storage_Elements;

procedure Incremental_Memory_Test is
   use Ada.Text_IO;
   use Interfaces.C;
   
   use type System.Address;
   use type Interfaces.Unsigned_8;
   
   -- Make the pointer math operators and types visible.
   use System.Storage_Elements;

   subtype Byte is Interfaces.Unsigned_8;

   -- Bind to C's malloc and free.
   function C_Malloc (Size : size_t) return System.Address;
   pragma Import (C, C_Malloc, "malloc");

   procedure C_Free (Ptr : System.Address);
   pragma Import (C, C_Free, "free");

   -- The Allocate_Memory procedure is correct and unchanged.
   procedure Allocate_Memory (Size_In_Bytes : size_t; Memory_Block  : out System.Address) is
   begin
      Put ("ATTEMPT: Allocating " & Long_Long_Integer'Image(Long_Long_Integer(Size_In_Bytes)) & " bytes... ");
      Memory_Block := C_Malloc (Size_In_Bytes);
      if Memory_Block = System.Null_Address then
         Put_Line ("FAILURE.");
         Put_Line ("         (This is expected if the system is low on memory.)");
      else
         Put_Line ("OK.");
         Put_Line ("         (Address provided by OS: " & System.Address'Image (Memory_Block) & ")");
      end if;
   end Allocate_Memory;

   -- This is the rewritten, hardened verification procedure.
   procedure Write_And_Verify_Chunk (Memory_Block : System.Address;
                                     Size         : size_t;
                                     Pattern      : Byte;
                                     Success      : out Boolean)
   is
      type Byte_Ptr is access all Byte;
      function To_Byte_Ptr is new Ada.Unchecked_Conversion(System.Address, Byte_Ptr);
      Current_Ptr : Byte_Ptr;
   begin
      Success := True;
      Put ("  -> Verifying chunk at " & System.Address'Image(Memory_Block) & "... ");
      
      -- FIX #1 (continued): Use a Storage_Offset for the loop.
      for I in 0 .. Storage_Offset (Size) - 1 loop
         -- Perform safe, portable address arithmetic.
         Current_Ptr := To_Byte_Ptr (Memory_Block + I);
         Current_Ptr.all := Pattern;
      end loop;

      for I in 0 .. Storage_Offset (Size) - 1 loop
         Current_Ptr := To_Byte_Ptr (Memory_Block + I);
         if Current_Ptr.all /= Pattern then
            Put_Line ("FAILURE!");
            Put_Line ("     Mismatch at offset " & Storage_Offset'Image(I));
            Success := False;
            return;
         end if;
      end loop;

      Put_Line ("OK.");
   end Write_And_Verify_Chunk;

   -------------------------------------------------------------------
   --                        MAIN PROGRAM                           --
   -------------------------------------------------------------------
   
   MB              : constant := 1_048_576;
   Chunk_Size_Bytes: constant size_t := 128 * MB;
   Total_GB_Target : constant := 6;
   Total_MB_Target : constant := Total_GB_Target * 1024;
   Number_Of_Chunks: constant := Total_MB_Target / 128;
   
   package Address_Vectors is new Ada.Containers.Vectors (Index_Type   => Positive,
                                                          Element_Type => System.Address);
   Allocated_Chunks : Address_Vectors.Vector;
   
   New_Chunk_Addr  : System.Address;
   Is_OK           : Boolean;
   
begin
   Put_Line ("--- Starting Incremental Memory Allocation Test ---");
   Put_Line ("Configuration:");
   Put_Line ("  - Chunk Size: " & Long_Long_Integer'Image(Long_Long_Integer(Chunk_Size_Bytes)) & " bytes (128 MB)");
   Put_Line ("  - Total Chunks: " & Integer'Image(Number_Of_Chunks));
   Put_Line ("  - Total Memory: " & Integer'Image(Total_GB_Target) & " GB");
   New_Line;
   
   for I in 1 .. Number_Of_Chunks loop
      Put ("Pass " & Integer'Image(I) & "/" & Integer'Image(Number_Of_Chunks) & ": Allocating 128 MB chunk... ");
      
      Allocate_Memory (Chunk_Size_Bytes, New_Chunk_Addr);
      
      if New_Chunk_Addr = System.Null_Address then
         Put_Line ("  -> OS could not provide memory. Stopping test.");
         exit;
      else
         Allocated_Chunks.Append (New_Chunk_Addr);
         Write_And_Verify_Chunk (New_Chunk_Addr, Chunk_Size_Bytes, Byte(I mod 256), Is_OK);
         
         if not Is_OK then
            Put_Line ("FATAL: Memory integrity check failed. Aborting.");
            exit;
         end if;
         
         Put_Line ("  -> Delaying for 1 second...");
         delay 1.0;
         New_Line;
      end if;
   end loop;
   
   Put_Line ("-------------------------------------------------");
   declare
      Total_Bytes : constant Long_Long_Integer :=
        Long_Long_Integer (Allocated_Chunks.Length) * Long_Long_Integer (Chunk_Size_Bytes);
   begin
      Put_Line ("Total allocated memory: " & Total_Bytes'Image & " bytes.");
   end;
   
   Put_Line ("Now deallocating all chunks...");

   -- FIX #2: Correctly loop through the vector and free each pointer.
   for Ptr of Allocated_Chunks loop
      C_Free (Ptr);
   end loop;
   
   Put_Line ("--- All chunks freed. Test complete. ---");

exception
    when others =>
        Put_Line ("An unexpected error occurred during the test.");
end Incremental_Memory_Test;