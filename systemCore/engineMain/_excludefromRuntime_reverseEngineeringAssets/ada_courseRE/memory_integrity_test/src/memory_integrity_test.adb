with Interfaces;
with Interfaces.C;
with System;
with Ada.Text_IO;

procedure Memory_Integrity_Test is
   use Ada.Text_IO;
   use Interfaces.C;
   
   use type System.Address;
   use type Interfaces.Unsigned_8;

   -- We do NOT need the Address_To_Access_Conversions package.

   -- =================================================================
   -- PART 1: BIND TO C's MALLOC and FREE
   -- =================================================================

   function C_Malloc (Size : size_t) return System.Address;
   pragma Import (C, C_Malloc, "malloc");

   procedure C_Free (Ptr : System.Address);
   pragma Import (C, C_Free, "free");

   -- =================================================================
   -- PART 2: DEFINE OUR ADA HELPER PROCEDURES
   -- =================================================================

   procedure Allocate_Memory (Size_In_Bytes : size_t;
                              Memory_Block  : out System.Address)
   is
   begin
      Put ("ATTEMPT: Allocating ");
      Put (Long_Long_Integer'Image(Long_Long_Integer(Size_In_Bytes)));
      Put_Line (" bytes...");
      
      Memory_Block := C_Malloc (Size_In_Bytes);

      if Memory_Block = System.Null_Address then
         Put_Line ("FAILURE: malloc returned null. Out of memory or invalid size.");
      else
         Put ("SUCCESS: Memory allocated at address: ");
         -- FIX #1: Use the 'Image attribute directly on the Address type.
         Put_Line (System.Address'Image (Memory_Block));
      end if;
   end Allocate_Memory;

   procedure Write_And_Verify_Memory (Memory_Block : System.Address;
                                      Size         : size_t;
                                      Success      : out Boolean)
   is
      subtype Byte is Interfaces.Unsigned_8;
      type Byte_Array is array (size_t range <>) of Byte;
      pragma Pack (Byte_Array);

      Test_Buffer : Byte_Array (1 .. Size) with Address => Memory_Block;
      Test_Pattern : constant Byte := 16#AA#;
   begin
      Success := True;
      Put_Line ("INTEGRITY: Writing test pattern to allocated memory...");
      
      for I in Test_Buffer'Range loop
         Test_Buffer (I) := Test_Pattern;
      end loop;

      Put_Line ("INTEGRITY: Verifying test pattern...");
      for I in Test_Buffer'Range loop
         if Test_Buffer (I) /= Test_Pattern then
            Put ("FAILURE: Mismatch at byte ");
            Put (Long_Long_Integer'Image(Long_Long_Integer(I)));
            Put_Line (". Memory corrupt!");
            Success := False;
            return;
         end if;
      end loop;

      Put_Line ("SUCCESS: Memory integrity check passed.");
   end Write_And_Verify_Memory;

   procedure Deallocate_Memory (Memory_Block : in out System.Address) is
   begin
      Put ("ATTEMPT: Deallocating memory at address: ");
      Put_Line (System.Address'Image (Memory_Block));
      C_Free (Memory_Block);
      
      Memory_Block := System.Null_Address;
      Put_Line ("SUCCESS: Memory freed and pointer nulled.");
   end Deallocate_Memory;

   -- =================================================================
   -- PART 3: THE MAIN PROGRAM LOGIC
   -- =================================================================
   
   My_Memory_Block : System.Address := System.Null_Address;
   Block_Size      : constant size_t := 1024;
   Is_OK           : Boolean;

begin
   Allocate_Memory (Block_Size, My_Memory_Block);

   if My_Memory_Block /= System.Null_Address then
      Write_And_Verify_Memory (My_Memory_Block, Block_Size, Is_OK);

      if Is_OK then
         Deallocate_Memory (My_Memory_Block);

         Put_Line ("--- Main program finished its normal operations. ---");
         New_Line;

         Put_Line ("DANGER: Now attempting to write to a null address...");
         Write_And_Verify_Memory (My_Memory_Block, Block_Size, Is_OK); 
         Put_Line ("...If you see this message, something is very strange.");
      end if;
   end if;

exception
   -- FIX #2: Use the correct, fully qualified name for the exception.
   when Storage_Error =>
      Put_Line ("SUCCESS (in a way): Caught expected error when accessing freed/null memory.");
   when others =>
      Put_Line ("An unexpected, different error occurred.");
end Memory_Integrity_Test;