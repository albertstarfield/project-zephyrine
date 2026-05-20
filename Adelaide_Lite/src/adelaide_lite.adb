with Ada.Text_IO; use Ada.Text_IO;
with Ada.Integer_Text_IO;
with Ada.Float_Text_IO;
with Ada.Real_Time; use Ada.Real_Time;
with Ada.Exceptions;
with Math_Utils;
with Integrity_Utils;
with Interfaces; use Interfaces;

procedure Adelaide_Lite is
   pragma Spark_Mode (Off);
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

   Command : String (1 .. 100);
   Last    : Natural;
begin
   --  Apply Apple Silicon soft real-time thread constraint policy
   Set_Darwin_Realtime;

   Put_Line ("========================================");
   Put_Line ("      Adelaide-Lite CLI Interface       ");
   Put_Line ("========================================");
   Put_Line ("Available commands: similarity, crc, parity, exit");

   loop
      Put ("> ");
      Get_Line (Command, Last);
      exit when Command (1 .. Last) = "exit";

      begin
         if Last >= 10 and then Command (1 .. 10) = "similarity" then
            declare
               Dim : Integer;
            begin
               Put ("Vector dimension: ");
               Ada.Integer_Text_IO.Get (Dim);
               if Dim > 0 then
                  declare
                     V1, V2 : Math_Utils.Vector (1 .. Dim);
                     Res    : Float;
                  begin
                     Put_Line ("Enter Vector 1 elements:");
                     for I in 1 .. Dim loop
                        Ada.Float_Text_IO.Get (V1 (I));
                     end loop;
                     Put_Line ("Enter Vector 2 elements:");
                     for I in 1 .. Dim loop
                        Ada.Float_Text_IO.Get (V2 (I));
                     end loop;

                     Res := Math_Utils.Cosine_Similarity (V1, V2);
                     Put ("Cosine Similarity: ");
                     Ada.Float_Text_IO.Put (Res, Fore => 1, Aft => 4, Exp => 0);
                     New_Line;
                  end;
               end if;
            end;

         elsif Last >= 3 and then Command (1 .. 3) = "crc" then
            declare
               Hex_Data : String (1 .. 256);
               H_Last   : Natural;
            begin
               Put ("Enter data in hex (even length): ");
               Get_Line (Hex_Data, H_Last);
               if H_Last mod 2 = 0 then
                  declare
                     Bytes : Integrity_Utils.Byte_Array (1 .. H_Last / 2);
                     CRC   : Unsigned_32;
                  begin
                     for I in 1 .. H_Last / 2 loop
                        Bytes (I) := Hex_To_Byte (Hex_Data (I * 2 - 1),
                                                 Hex_Data (I * 2));
                     end loop;
                     CRC := Integrity_Utils.Calculate_CRC32 (Bytes);
                     Put_Line ("CRC-32: 16#" & Unsigned_32'Image (CRC) & "#");
                  end;
               else
                  Put_Line ("Error: Hex data must have an even length.");
               end if;
            end;

         elsif Last >= 6 and then Command (1 .. 6) = "parity" then
            declare
               Num_Blocks : Integer;
               Block_Size : Integer;
            begin
               Put ("Number of data blocks: ");
               Ada.Integer_Text_IO.Get (Num_Blocks);
               Put ("Block size (bytes): ");
               Ada.Integer_Text_IO.Get (Block_Size);

               if Num_Blocks > 0 and then Block_Size > 0 then
                  declare
                     Data   : Integrity_Utils.Byte_Array
                       (1 .. Num_Blocks * Block_Size);
                     Parity : Integrity_Utils.Byte_Array (1 .. Block_Size);
                     CRCs   : Integrity_Utils.CRC_Array (1 .. Num_Blocks);
                     Success : Boolean;
                  begin
                     Put_Line ("Enter block data (hex, continuous):");
                     declare
                        Hex_Data : String (1 .. Num_Blocks * Block_Size * 2);
                        H_Last   : Natural;
                     begin
                        Get_Line (Hex_Data, H_Last);
                        if H_Last = Num_Blocks * Block_Size * 2 then
                           for I in 1 .. Num_Blocks * Block_Size loop
                              Data (I) := Hex_To_Byte (Hex_Data (I * 2 - 1),
                                                      Hex_Data (I * 2));
                           end loop;

                           --  Calculate initial state
                           for I in 1 .. Num_Blocks loop
                              declare
                                 Start : constant Positive :=
                                   (I - 1) * Block_Size + 1;
                              begin
                                 CRCs (I) := Integrity_Utils.Calculate_CRC32
                                   (Data (Start .. Start + Block_Size - 1));
                              end;
                           end loop;
                           Integrity_Utils.Generate_Parity
                             (Data, Block_Size, Parity);

                           Put_Line ("Simulate corruption? (y/n)");
                           declare
                              Ans : Character;
                           begin
                              Get (Ans);
                              if Ans = 'y' then
                                 Data (1) := Data (1) xor 16#FF#;
                                 Put_Line ("Corrupted first byte of block 1.");
                              end if;
                           end;

                           --  Self-Patch
                           Integrity_Utils.Self_Patch
                             (Data, Block_Size, CRCs, Parity, Success);

                           if Success then
                              Put_Line ("[+] Self-Patch Successful.");
                           else
                              Put_Line ("[!] Self-Patch Failed.");
                           end if;
                        end if;
                     end;
                  end;
               end if;
            end;

         else
            Put_Line ("Unknown command. Try: similarity, crc, parity, exit");
         end if;

      exception
         when E : others =>
            Put_Line ("Error: " & Ada.Exceptions.Exception_Message (E));
            --  Consume the rest of the line if there's any pending data
            begin
               Skip_Line;
            exception
               when others => null;
            end;
      end;
   end loop;
end Adelaide_Lite;
