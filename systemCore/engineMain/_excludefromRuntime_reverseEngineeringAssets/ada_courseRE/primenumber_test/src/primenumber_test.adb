-- We need the standard library for printing to the console.
with Ada.Text_IO;

-- The main procedure for our program.
-- By default, Alire creates a main procedure named after the project.
-- We'll rename the file later to match this name for good practice.
procedure PrimeNumber_Test is

   -- This is a "nested function". It's a helper that is only visible
   -- inside the PrimeNumber_Test procedure. It's a clean way to organize code.
   function Is_Prime (Number : Positive) return Boolean is
   begin
      -- Prime numbers must be greater than 1.
      if Number <= 1 then
         return False;
      end if;

      -- We check for divisors from 2 up to half of the number.
      -- If we find even one divisor, the number is not prime.
      for Divisor in 2 .. Number / 2 loop
         -- The 'rem' operator gives the remainder of a division.
         -- If the remainder is 0, it means we found a perfect divisor.
         if Number rem Divisor = 0 then
            return False; -- Not prime, exit the function immediately.
         end if;
      end loop;

      -- If the loop finishes without finding any divisors, the number is prime.
      return True;
   end Is_Prime;

-- This is where the main program execution begins.
begin
   Ada.Text_IO.Put_Line ("Finding prime numbers up to 1000:");

   -- Loop through every number from 1 to 1000.
   for N in 1 .. 1000 loop

      -- Call our helper function to check if the current number N is prime.
      if Is_Prime (N) then
         -- If it is prime, print it.
         -- Integer'Image is the standard Ada way to convert an integer to a string.
         Ada.Text_IO.Put_Line (Integer'Image (N));
      end if;

   end loop;

end PrimeNumber_Test;
