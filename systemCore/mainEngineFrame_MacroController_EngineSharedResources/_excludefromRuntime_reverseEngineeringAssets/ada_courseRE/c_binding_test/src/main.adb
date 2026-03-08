with Interfaces.C;
with Interfaces.C.Strings;
with Ada.Text_IO;

procedure C_Binding_Test is
   use Ada.Text_IO;
   
   -- FIX #2: Make the operators for C integers visible.
   use type Interfaces.C.int;

   -- =================================================================
   -- PART 1: DECLARE THE ADA "WRAPPER" FOR THE C FUNCTION
   -- =================================================================

   -- FIX #1: Use the correct, full type name for a C string pointer.
   function C_System (Command : Interfaces.C.Strings.chars_ptr) return Interfaces.C.int;
   
   pragma Import
     (Convention    => C,
      Entity        => C_System,
      External_Name => "system");

   -- =================================================================
   -- PART 2: USE THE NEWLY BOUND C FUNCTION IN OUR ADA CODE
   -- =================================================================
begin
   Put_Line ("Attempting to launch Zenity via a direct C 'system' call...");

   declare
      My_Command  : constant String := "zenity --entry --text=""Say Hi from C!""";
      
      -- FIX #3: C_Command cannot be a constant because Free needs to modify it.
      C_Command   : Interfaces.C.Strings.chars_ptr :=
        Interfaces.C.Strings.New_String (My_Command);
        
      Return_Code : Interfaces.C.int;
   begin
      Return_Code := C_System (C_Command);

      -- This comparison now works because of the 'use type' clause.
      if Return_Code = 0 then
         Put_Line ("C 'system' call returned 0 (Success).");
      else
         Put_Line ("C 'system' call returned a non-zero code (Failure).");
      end if;
      
      -- This now works because C_Command is a variable.
      Interfaces.C.Strings.Free (C_Command);
   end;

   Put_Line ("Program finished.");

end C_Binding_Test;
