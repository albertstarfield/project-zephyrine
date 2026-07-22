-- Implements the configuration loader.
with Ada.Text_IO;
with Ada.Strings.Fixed;
with Ada.Characters.Handling;

package body Config is
   procedure Load (Into : out App_Config; From_File : String) is
      File    : Ada.Text_IO.File_Type;
      Line    : String (1 .. 256);
      Length  : Natural;
      Default : constant App_Config :=
        (Port              => To_Unbounded_String ("3001"),
         OpenAI_API_Base_URL => To_Unbounded_String ("http://localhost:11434/v1"),
         LLM_API_Key       => To_Unbounded_String ("ollama"),
         DB_Path           => To_Unbounded_String ("./project_zephyrine_chats.db"));
   begin
      Into := Default; -- Start with default values
      Ada.Text_IO.Open (File => File, Mode => Ada.Text_IO.In_File, Name => From_File);

      while not Ada.Text_IO.End_Of_File (File) loop
         Ada.Text_IO.Get_Line (File, Line, Length);
         -- Simple key=value parser
         declare
            Line_Str : constant String := Line (1 .. Length);
            Separator_Pos : constant Natural := Ada.Strings.Fixed.Index (Line_Str, "=");
         begin
            if Separator_Pos > 0 then
               declare
                  Key   : constant String := Ada.Characters.Handling.To_Lower (Ada.Strings.Fixed.Trim (Line_Str (1 .. Separator_Pos - 1), Ada.Strings.Both));
                  Value : constant String := Ada.Strings.Fixed.Trim (Line_Str (Separator_Pos + 1 .. Length), Ada.Strings.Both);
               begin
                  case Key is
                     when "port"                => Into.Port := To_Unbounded_String (Value);
                     when "openai_api_base_url" => Into.OpenAI_API_Base_URL := To_Unbounded_String (Value);
                     when "openai_api_key"      => Into.LLM_API_Key := To_Unbounded_String (Value);
                     when "db_path"             => Into.DB_Path := To_Unbounded_String (Value);
                     when others                => null;
                  end case;
               end;
            end if;
         end;
      end loop;
      Ada.Text_IO.Close (File);
   exception
      when Ada.Text_IO.Name_Error =>
         Ada.Text_IO.Put_Line ("INFO: .env file not found, using default configuration.");
      when others =>
         Ada.Text_IO.Put_Line ("WARNING: Error reading .env file, using default configuration.");
   end Load;
end Config;