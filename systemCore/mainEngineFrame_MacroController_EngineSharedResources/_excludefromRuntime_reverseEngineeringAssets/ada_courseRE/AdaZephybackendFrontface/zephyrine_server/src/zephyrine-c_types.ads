with System;
with Interfaces.C;

package Zephyrine.C_Types is
   -- Common opaque pointer type, used as a handle to C objects
   type Opaque_Handle is new System.Address;

   -- A C-style null-terminated string
   subtype C_String is Interfaces.C.char_array;

   -- A pointer to a C_String (char**)
   type C_String_Access is access all C_String;

end Zephyrine.C_Types;