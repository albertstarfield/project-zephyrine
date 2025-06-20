-- We need to import the standard library for Text Input/Output.
-- This is the most fundamental I/O library in Ada.
with Ada.Text_IO;

-- This is the main procedure of our program. The filename should match this name.
procedure Simple_IO is

   -- To avoid typing "Ada.Text_IO." repeatedly, we can bring its contents
   -- into the current scope.
   use Ada.Text_IO;

   -- A special package for reading and writing Integers.
   -- This is a "generic instantiation". We are creating a new package named
   -- "Int_IO" by using the "Integer_IO" template from Ada.Text_IO and telling
   -- it to work specifically with the standard "Integer" type.
   package Int_IO is new Ada.Text_IO.Integer_IO (Integer);

   -- Declare some variables to hold the user's input.
   User_Name : String (1 .. 80); -- A fixed-size buffer to hold the name.
   Last      : Natural;         -- This will store the actual length of the name entered.
   User_Age  : Integer;

begin
   -- The main part of our program starts here.

   Put ("Please enter your name: ");
   -- Get_Line reads a line of text from the user into our buffer.
   -- It also tells us where the last character was placed in the 'Last' variable.
   Get_Line (Item => User_Name, Last => Last);

   Put ("Thank you. Now, please enter your age: ");
   -- We use our specialized Int_IO package to read an integer.
   Int_IO.Get (Item => User_Age);

   -- Print a blank line for better formatting.
   New_Line;

   -- Print a greeting back to the user.
   -- We only print the part of the string that the user actually typed
   -- by using a "slice": User_Name(1 .. Last).
   Put_Line ("Hello, " & User_Name (1 .. Last) & "!");

   -- To print an integer, we use the 'Image attribute. This is the standard
   -- way to convert a value of almost any type into a String.
   Put_Line ("You are" & Integer'Image (User_Age) & " years old.");
   Put_Line ("Next year, you will be" & Integer'Image (User_Age + 1) & ".");
   Put_Line ("Goodbye!");

end Simple_IO;