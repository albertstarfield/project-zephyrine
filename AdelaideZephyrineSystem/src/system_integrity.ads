--  ── System Integrity (Cross-Platform) ────────────────────────────────────────
--  Computes hardware and binary hashes for integrity verification.
--  Used to derive hardware-bound encryption keys.
--
--  HARDWARE IDENTITY SOURCES:
--    Linux:  lsusb, lshw, lspci, dmidecode, /proc/cpuinfo, lsblk
--    macOS:  system_profiler, ioreg, sysctl
--
--  BINARY INTEGRITY SOURCES:
--    Linux:  /boot/*, /bin/*, /usr/bin/*, /etc/systemd/system/*
--    macOS:  /System/Library/Kernels/*, /usr/local/bin/*, /Library/LaunchDaemons/*
--
--  SECURITY NOTE:
--    Even SIP-protected paths must be scanned. SIP has zero-day vulnerabilities.
--    Hashing protected files detects tampering even if SIP is bypassed.
--  ──────────────────────────────────────────────────────────────────────────────

with Interfaces; use Interfaces;

package System_Integrity
  with SPARK_Mode => Off  --  Shell commands require SPARK_Mode Off
is
   --  Hash size (SHA-512 = 64 bytes)
   subtype Hash_Index is Positive range 1 .. 64;
   type Hash_Type is array (Hash_Index) of Interfaces.Unsigned_8
     with Pack;

   --  Empty hash (all zeros)
   Empty_Hash : constant Hash_Type := (others => 0);

   --  Compute hardware identity hash from system components
   --  Combines: USB, CPU, RAM, PCI, disk serial, etc.
   function Compute_Hardware_Hash return Hash_Type;

   --  Compute binary integrity hash from critical system files
   --  Combines: kernel, bootloader, core utils, systemd units
   function Compute_Binary_Hash return Hash_Type;

   --  Compute combined integrity hash = SHA512(HW_Hash || Binary_Hash)
   function Compute_Integrity_Hash return Hash_Type;

   --  Get string representation of hash (for logging/debugging)
   function Hash_To_String (H : Hash_Type) return String;

   --  Convert hex string to Hash_Type
   function String_To_Hash (S : String) return Hash_Type;

end System_Integrity;
