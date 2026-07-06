--  ── System Integrity Implementation ──────────────────────────────────────────
--  Cross-platform hardware and binary hash computation.
--  Uses shell commands to gather system identity components.
--  ──────────────────────────────────────────────────────────────────────────────

with Ada.Text_IO;           use Ada.Text_IO;
with Ada.Strings;           use Ada.Strings;
with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;
with Ada.Directories;       use Ada.Directories;
with Ada.Streams.Stream_IO; use Ada.Streams.Stream_IO;
with Interfaces;            use Interfaces;

package body System_Integrity
  with SPARK_Mode => Off
is

   --  ── Platform Detection ────────────────────────────────────────────────────
   --  Using the same approach as adelaide_server.adb for platform detection

   function Is_Linux return Boolean is
      F : File_Type;
      Line : Unbounded_String;
   begin
      begin
         Open (F, In_File, "/etc/os-release");
         while not End_Of_File (F) loop
            Line := To_Unbounded_String (Get_Line (F));
            if Index (Line, "Linux") > 0 or Index (Line, "linux") > 0 then
               Close (F);
               return True;
            end if;
         end loop;
         Close (F);
      exception
         when others =>
            null;
      end;
      return False;
   end Is_Linux;

   function Is_MacOS return Boolean is
   begin
      return not Is_Linux;  --  Simplified: assume macOS if not Linux
   end Is_MacOS;

   --  ── Shell Command Execution ───────────────────────────────────────────────

   function Execute_Command (Cmd : String) return Unbounded_String is
      Result : Unbounded_String;
      F : File_Type;
   begin
      begin
         Open (F, In_File, "/bin/sh -c " & '"' & Cmd & '"');
         while not End_Of_File (F) loop
            Append (Result, Get_Line (F));
            Append (Result, Ascii.LF);
         end loop;
         Close (F);
      exception
         when others =>
            null;
      end;
      return Result;
   end Execute_Command;

   --  ── Hardware Identity Sources ─────────────────────────────────────────────

   function Get_Linux_Hardware_Identity return Unbounded_String is
      Identity : Unbounded_String;
   begin
      --  USB devices
      Append (Identity, Execute_Command ("lsusb 2>/dev/null"));
      --  System info
      Append (Identity, Execute_Command ("lshw -c system 2>/dev/null | head -50"));
      --  PCI devices
      Append (Identity, Execute_Command ("lspci 2>/dev/null | head -30"));
      --  BIOS/Serial
      Append (Identity, Execute_Command ("dmidecode -t system 2>/dev/null | head -20"));
      --  CPU info
      Append (Identity, Execute_Command ("cat /proc/cpuinfo 2>/dev/null | head -20"));
      --  RAM info
      Append (Identity, Execute_Command ("dmidecode -t memory 2>/dev/null | head -20"));
      --  Disk serial
      Append (Identity, Execute_Command ("lsblk -d -o NAME,SERIAL 2>/dev/null"));
      return Identity;
   end Get_Linux_Hardware_Identity;

   function Get_MacOS_Hardware_Identity return Unbounded_String is
      Identity : Unbounded_String;
   begin
      --  USB devices
      Append (Identity, Execute_Command ("system_profiler SPUSBDataType 2>/dev/null | head -50"));
      --  Hardware info
      Append (Identity, Execute_Command ("system_profiler SPHardwareDataType 2>/dev/null"));
      --  PCI devices
      Append (Identity, Execute_Command ("system_profiler SPPCIDataType 2>/dev/null | head -30"));
      --  Hardware tree (IOPlatformSerialNumber, IOPlatformUUID)
      Append (Identity, Execute_Command ("ioreg -l 2>/dev/null | grep -E 'IOPlatformSerialNumber|IOPlatformUUID' | head -10"));
      --  NVMe
      Append (Identity, Execute_Command ("system_profiler SPNVMeDataType 2>/dev/null | head -20"));
      --  CPU info
      Append (Identity, Execute_Command ("sysctl machdep.cpu 2>/dev/null"));
      --  RAM info
      Append (Identity, Execute_Command ("system_profiler SPMemoryDataType 2>/dev/null | head -20"));
      --  Thunderbolt
      Append (Identity, Execute_Command ("system_profiler SPThunderboltDataType 2>/dev/null | head -20"));
      return Identity;
   end Get_MacOS_Hardware_Identity;

   --  ── Binary Integrity Sources ──────────────────────────────────────────────

   function Get_Linux_Binary_Integrity return Unbounded_String is
      Integrity : Unbounded_String;
   begin
      --  Kernel
      Append (Integrity, Execute_Command ("ls -la /boot/*vmlinuz* /boot/*initrd* 2>/dev/null"));
      --  Bootloader
      Append (Integrity, Execute_Command ("ls -la /boot/efi/* 2>/dev/null | head -20"));
      --  Core utils
      Append (Integrity, Execute_Command ("ls -la /bin/* 2>/dev/null | head -30"));
      Append (Integrity, Execute_Command ("ls -la /usr/bin/* 2>/dev/null | head -30"));
      --  Systemd
      Append (Integrity, Execute_Command ("ls -la /etc/systemd/system/* 2>/dev/null | head -30"));
      return Integrity;
   end Get_Linux_Binary_Integrity;

   function Get_MacOS_Binary_Integrity return Unbounded_String is
      Integrity : Unbounded_String;
   begin
      --  Kernel (SIP-protected, scan anyway)
      Append (Integrity, Execute_Command ("ls -la /System/Library/Kernels/* 2>/dev/null | head -10"));
      --  Bootloader
      Append (Integrity, Execute_Command ("ls -la /System/Library/CoreServices/boot.efi 2>/dev/null"));
      --  Homebrew
      Append (Integrity, Execute_Command ("ls -la /usr/local/bin/* 2>/dev/null | head -30"));
      --  LaunchDaemons
      Append (Integrity, Execute_Command ("ls -la /Library/LaunchDaemons/* 2>/dev/null | head -20"));
      --  LaunchAgents
      Append (Integrity, Execute_Command ("ls -la /Library/LaunchAgents/* 2>/dev/null | head -20"));
      --  Kernel Extensions
      Append (Integrity, Execute_Command ("ls -la /Library/Extensions/* 2>/dev/null | head -20"));
      return Integrity;
   end Get_MacOS_Binary_Integrity;

   --  ── SHA-512 Hashing (via OpenSSL) ─────────────────────────────────────────

   function SHA512_Hash (Data : String) return Hash_Type is
      Result : Hash_Type := (others => 0);
      F : File_Type;
      Temp_File : constant String := "/tmp/adelaide_integrity_hash.tmp";
      Cmd : constant String := "echo -n " & '"' & Data & '"' & " | openssl dgst -sha512 -binary > " & Temp_File;
   begin
      --  Write data to temp file and hash it
      begin
         Create (F, Out_File, Temp_File);
         String'Output (F, Data);
         Close (F);
      exception
         when others =>
            null;
      end;

      --  Execute OpenSSL command
      begin
         Execute_Command (Cmd);
      exception
         when others =>
            null;
      end;

      --  Read binary hash
      begin
         Open (F, In_File, Temp_File);
         for I in Hash_Index loop
            begin
               Result (I) := Interfaces.Unsigned_8'Value (Get_Line (F));
            exception
               when others =>
                  null;
            end;
         end loop;
         Close (F);
      exception
         when others =>
            null;
      end;

      --  Clean up temp file
      begin
         Delete_File (Temp_File);
      exception
         when others =>
            null;
      end;

      return Result;
   end SHA512_Hash;

   --  ── Hash Combination ──────────────────────────────────────────────────────

   function Combine_Hashes (Left, Right : Hash_Type) return Hash_Type is
      Combined : Hash_Type := (others => 0);
   begin
      --  Simple concatenation hash: SHA512(Left || Right)
      --  For now, use XOR combination (will be upgraded to proper SHA-512)
      for I in Hash_Index loop
         Combined (I) := Left (I) xor Right (I);
      end loop;
      return Combined;
   end Combine_Hashes;

   --  ── Public Interface ──────────────────────────────────────────────────────

   function Compute_Hardware_Hash return Hash_Type is
      Identity : Unbounded_String;
   begin
      if Is_Linux then
         Identity := Get_Linux_Hardware_Identity;
      elsif Is_MacOS then
         Identity := Get_MacOS_Hardware_Identity;
      else
         return Empty_Hash;
      end if;

      return SHA512_Hash (To_String (Identity));
   end Compute_Hardware_Hash;

   function Compute_Binary_Hash return Hash_Type is
      Integrity : Unbounded_String;
   begin
      if Is_Linux then
         Integrity := Get_Linux_Binary_Integrity;
      elsif Is_MacOS then
         Integrity := Get_MacOS_Binary_Integrity;
      else
         return Empty_Hash;
      end if;

      return SHA512_Hash (To_String (Integrity));
   end Compute_Binary_Hash;

   function Compute_Integrity_Hash return Hash_Type is
      HW_Hash : constant Hash_Type := Compute_Hardware_Hash;
      Bin_Hash : constant Hash_Type := Compute_Binary_Hash;
   begin
      return Combine_Hashes (HW_Hash, Bin_Hash);
   end Compute_Integrity_Hash;

   --  ── String Conversion ─────────────────────────────────────────────────────

   function Hash_To_String (H : Hash_Type) return String is
      Result : String (1 .. 128);
      Hex_Chars : constant String := "0123456789abcdef";
   begin
      for I in Hash_Index loop
         Result ((I - 1) * 2 + 1) := Hex_Chars (Natural (H (I)) / 16 + 1);
         Result ((I - 1) * 2 + 2) := Hex_Chars (Natural (H (I)) mod 16 + 1);
      end loop;
      return Result;
   end Hash_To_String;

   function String_To_Hash (S : String) return Hash_Type is
      Result : Hash_Type := (others => 0);
      Hex_To_Nibble : function (C : Character) return Interfaces.Unsigned_8 is
         (case C is
          when '0' .. '9' => Interfaces.Unsigned_8'Value ("" & C),
          when 'a' .. 'f' => Interfaces.Unsigned_8'Value ("" & C) - 10,
          when 'A' .. 'F' => Interfaces.Unsigned_8'Value ("" & C) - 10,
          when others => 0);
   begin
      if S'Length /= 128 then
         return Empty_Hash;
      end if;

      for I in Hash_Index loop
         Result (I) := Hex_To_Nibble (S ((I - 1) * 2 + 1)) * 16 +
                        Hex_To_Nibble (S ((I - 1) * 2 + 2));
      end loop;
      return Result;
   end String_To_Hash;

end System_Integrity;
