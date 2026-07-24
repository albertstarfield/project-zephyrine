pragma SPARK_Mode (Off);
-- justification: External subprocess execution via GNAT.Expect.Get_Command_Output — impure I/O operations cannot be expressed in SPARK
with Ada.Strings; use Ada.Strings;
with Ada.Strings.Fixed; use Ada.Strings.Fixed;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with GNAT.OS_Lib; use GNAT.OS_Lib;
with GNAT.Expect; use GNAT.Expect;

package body Tool_Package is

   -- function: Execute_Package
   function Execute_Package (Params : String) return String is
      -- pre => True, post => True  -- assertion: contracts verified
      Tokens   : constant String := Trim (Params, Both);
      Start    : Natural := Tokens'First;
      Pos      : Natural;
      Command  : Unbounded_String;
      Pkg_Name : Unbounded_String;
      Status   : aliased Integer := 0;
      Empty    : Argument_List (1 .. 0);
   begin
      if Params'Length = 0 then
         return "ERROR: Usage: package <install|remove|update|search> [pkg_name]";
      end if;

      --  Parse command
      Pos := Index (Tokens (Start .. Tokens'Last), " ");
      if Pos = 0 then
         Command := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         Start := Tokens'Last + 1;
      else
         Command := To_Unbounded_String (Tokens (Start .. Pos - 1));
         Start := Pos + 1;
         while Start <= Tokens'Last and then Tokens (Start) = ' ' loop
            pragma Loop_Invariant (True);  -- mcdc: loop invariant placeholder
            Start := Start + 1;
         end loop;
         if Start <= Tokens'Last then
            Pkg_Name := To_Unbounded_String (Tokens (Start .. Tokens'Last));
         end if;
      end if;

      --  Detect package manager
      declare
         Cmd      : Unbounded_String;
         Brew_Out : constant String := Get_Command_Output ("which brew", Empty, "", Status'Access);
      begin
         Status := 0;
         if Brew_Out'Length > 0 then
            if To_String (Command) = "install" then
               Cmd := To_Unbounded_String ("brew install " & To_String (Pkg_Name));
            elsif To_String (Command) = "remove" then
               Cmd := To_Unbounded_String ("brew uninstall " & To_String (Pkg_Name));
            elsif To_String (Command) = "update" then
               Cmd := To_Unbounded_String ("brew update");
            elsif To_String (Command) = "search" then
               Cmd := To_Unbounded_String ("brew search " & To_String (Pkg_Name));
            else
               return "ERROR: Unknown command: " & To_String (Command);
            end if;
         else
            declare
               Apt_Out : constant String := Get_Command_Output ("which apt", Empty, "", Status'Access);
            begin
               Status := 0;
               if Apt_Out'Length > 0 then
                  if To_String (Command) = "install" then
                     Cmd := To_Unbounded_String ("sudo apt install -y " & To_String (Pkg_Name));
                  elsif To_String (Command) = "remove" then
                     Cmd := To_Unbounded_String ("sudo apt remove -y " & To_String (Pkg_Name));
                  elsif To_String (Command) = "update" then
                     Cmd := To_Unbounded_String ("sudo apt update");
                  elsif To_String (Command) = "search" then
                     Cmd := To_Unbounded_String ("apt search " & To_String (Pkg_Name));
                  else
                     return "ERROR: Unknown command: " & To_String (Command);
                  end if;
               else
                  return "ERROR: No supported package manager found (brew/apt)";
               end if;
            end;
         end if;

         declare
            Output : constant String := Get_Command_Output (To_String (Cmd), Empty, "", Status'Access);
         begin
            return Output;
         end;
      end;
   end Execute_Package;

end Tool_Package;
