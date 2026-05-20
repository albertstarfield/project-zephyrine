with Ada.Text_IO; use Ada.Text_IO;
with Ada.Integer_Text_IO;
with Ada.Float_Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Exceptions;
with Math_Utils;
with Integrity_Utils;
with Interfaces; use Interfaces;

procedure Adelaide_Lite is
   procedure Set_Darwin_Realtime;
   pragma Import (C, Set_Darwin_Realtime, "set_darwin_realtime");

   function Hex_To_Byte (C1, C2 : Character) return Unsigned_8 is
      function Val (C : Character) return Unsigned_8 is
      begin
         case C is
            when '0' .. '9' =>
               return Character'Pos (C) - Character'Pos ('0');
            when 'a' .. 'f' =>
               return Character'Pos (C) - Character'Pos ('a') + 10;
            when 'A' .. 'F' =>
               return Character'Pos (C) - Character'Pos ('A') + 10;
            when others =>
               return 0;
         end case;
      end Val;
   begin
      return Val (C1) * 16 + Val (C2);
   end Hex_To_Byte;

   function Byte_To_Hex (Val : Unsigned_8) return String is
      Hex_Chars : constant String := "0123456789abcdef";
   begin
      return "" & Hex_Chars (Integer (Val / 16) + 1) &
                  Hex_Chars (Integer (Val mod 16) + 1);
   end Byte_To_Hex;

   Command : String (1 .. 100);
   Last    : Natural;
begin
   --  Set soft real-time scheduling policy on macOS/Darwin
   Set_Darwin_Realtime;

   Put_Line ("[+] Adelaide_Lite ready.");
   Flush;

   while not End_Of_File loop
      begin
         Get_Line (Command, Last);
         if Last > 0 then
            declare
               Cmd_Name : constant String := Command (1 .. Last);
            begin
               if Cmd_Name = "similarity" then
                  declare
                     Dim : Integer;
                  begin
                     Ada.Integer_Text_IO.Get (Dim);
                     declare
                        V1 : Math_Utils.Vector (1 .. Dim);
                        V2 : Math_Utils.Vector (1 .. Dim);
                     begin
                        for I in 1 .. Dim loop
                           Ada.Float_Text_IO.Get (V1 (I));
                        end loop;
                        for I in 1 .. Dim loop
                           Ada.Float_Text_IO.Get (V2 (I));
                        end loop;
                        Skip_Line; --  Consume trailing newline

                        declare
                           Sim : constant Float :=
                             Math_Utils.Cosine_Similarity (V1, V2);
                        begin
                           Put_Line ("SIMILARITY: " & Float'Image (Sim));
                           Flush;
                        end;
                     end;
                  end;

               elsif Cmd_Name = "parity_generate" then
                  declare
                     Num_Blocks : Integer;
                     Block_Size : Integer;
                  begin
                     Ada.Integer_Text_IO.Get (Num_Blocks);
                     Ada.Integer_Text_IO.Get (Block_Size);
                     Skip_Line; --  Consume sizes line

                     declare
                        Hex_Data : constant String := Get_Line;
                        Data     : Integrity_Utils.Byte_Array
                          (1 .. Num_Blocks * Block_Size);
                        Parity   : Integrity_Utils.Byte_Array
                          (1 .. Block_Size);
                     begin
                        for B in 1 .. Num_Blocks * Block_Size loop
                           Data (B) := Hex_To_Byte
                             (Hex_Data (Hex_Data'First + (B - 1) * 2),
                              Hex_Data (Hex_Data'First + (B - 1) * 2 + 1));
                        end loop;

                        Integrity_Utils.Generate_Parity
                          (Data, Block_Size, Parity);

                        Put ("PARITY: ");
                        for B in 1 .. Block_Size loop
                           Put (Byte_To_Hex (Parity (B)));
                        end loop;
                        New_Line;
                        Flush;
                     end;
                  end;

               elsif Cmd_Name = "parity_fix" then
                  declare
                     Num_Blocks    : Integer;
                     Block_Size    : Integer;
                     Corrupt_Index : Integer;
                  begin
                     Ada.Integer_Text_IO.Get (Num_Blocks);
                     Ada.Integer_Text_IO.Get (Block_Size);
                     Ada.Integer_Text_IO.Get (Corrupt_Index);
                     Skip_Line; --  Consume sizes line

                     declare
                        Hex_Data   : constant String := Get_Line;
                        Hex_Parity : constant String := Get_Line;
                        Data       : Integrity_Utils.Byte_Array
                          (1 .. Num_Blocks * Block_Size);
                        Parity     : Integrity_Utils.Byte_Array
                          (1 .. Block_Size);
                     begin
                        for B in 1 .. Num_Blocks * Block_Size loop
                           Data (B) := Hex_To_Byte
                             (Hex_Data (Hex_Data'First + (B - 1) * 2),
                              Hex_Data (Hex_Data'First + (B - 1) * 2 + 1));
                        end loop;

                        for B in 1 .. Block_Size loop
                           Parity (B) := Hex_To_Byte
                             (Hex_Parity (Hex_Parity'First + (B - 1) * 2),
                              Hex_Parity (Hex_Parity'First + (B - 1) * 2 + 1));
                        end loop;

                        Integrity_Utils.Reconstruct_Block
                          (Data, Block_Size, Corrupt_Index, Parity);

                        Put ("FIXED: ");
                        for B in 1 .. Num_Blocks * Block_Size loop
                           Put (Byte_To_Hex (Data (B)));
                        end loop;
                        New_Line;
                        Flush;
                     end;
                  end;

               elsif Cmd_Name = "crc32" then
                  declare
                     Hex_Data : constant String := Get_Line;
                     Len      : constant Positive := Hex_Data'Length / 2;
                     Data     : Integrity_Utils.Byte_Array (1 .. Len);
                  begin
                     for B in 1 .. Len loop
                        Data (B) := Hex_To_Byte
                          (Hex_Data (Hex_Data'First + (B - 1) * 2),
                           Hex_Data (Hex_Data'First + (B - 1) * 2 + 1));
                     end loop;

                     declare
                        CRC : constant Unsigned_32 :=
                          Integrity_Utils.Calculate_CRC32 (Data);
                     begin
                        Put_Line ("CRC32: " & Unsigned_32'Image (CRC));
                        Flush;
                     end;
                  end;

               elsif Cmd_Name = "delay" then
                  declare
                     Target_Ms  : Integer;
                     Start_Time : constant Time := Clock;
                  begin
                     Ada.Integer_Text_IO.Get (Target_Ms);
                     Skip_Line;

                     --  Consistent WCEF execution timing delay
                     delay until Start_Time + Milliseconds (Target_Ms);

                     Put_Line ("DELAYED");
                     Flush;
                  end;

               elsif Cmd_Name = "exit" then
                  exit;
               else
                  Put_Line ("ERROR: Unknown command: " & Cmd_Name);
                  Flush;
               end if;
            end;
         end if;
      exception
         when E : others =>
            Put_Line ("ERROR: Exception raised: " &
                      Ada.Exceptions.Exception_Name (E));
            Flush;
            --  Consume the rest of the line if there's any pending data
            begin
               Skip_Line;
            exception
               when others => null;
            end;
      end;
   end loop;
end Adelaide_Lite;
