-- File: src/main.adb

-- Make the procedures from our Stella_Icarus package visible here.
with Stella_Icarus;

procedure Main is
   -- pre => True, post => True
begin
   -- Call the Greet procedure from our custom package.
   Stella_Icarus.Greet;
end Main;
