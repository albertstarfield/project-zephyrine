-- security_scanner.ads
-- Regex-based security scanner for dangerous code patterns.
-- Native Ada replacement for src/python/security.py.
--
-- Axioms:
--   - ISO/IEC 8652:2012 RM A.4 (String handling)
--   - CWE/SANS Top 25 vulnerability pattern matching
--   - DO-178C MC/DC loop invariants for iteration
--
-- Scans source files for:
--   - Command injection (os.system, subprocess shell=True, os.popen)
--   - Code injection (eval, exec, __import__)
--   - Deserialization risks (pickle, yaml.load)
--   - Hardcoded secrets (password, secret, api_key, token)
--   - SSL verification bypass
--   - SQL injection patterns

pragma SPARK_Mode (Off);
-- Justification: File I/O and pattern matching are impure operations.

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Security_Scanner is

   Max_Issues : constant := 1024;

   type Severity_Level is (CRITICAL, HIGH, MEDIUM, LOW);

   type Security_Issue is record
      Filepath  : Unbounded_String;
      Line_Num  : Natural;
      Severity  : Severity_Level;
      Message   : Unbounded_String;
      Code_Line : Unbounded_String;
   end record;

   type Issue_Array is array (1 .. Max_Issues) of Security_Issue;

   type Scan_Result is record
      Issues : Issue_Array;
      Count  : Natural := 0;
   end record;

   --  Scan a single file for security issues.
   function Scan_File (Filepath : String) return Scan_Result
     with Pre => Filepath'Length > 0, Post => True;

   --  Scan a directory recursively for security issues.
   --  Skips hidden dirs, node_modules, __pycache__, venv, .git.
   function Scan_Directory (Path : String) return Scan_Result
     with Pre => Path'Length > 0, Post => True;

   --  Format a scan result as a human-readable report string.
   function Format_Report (Result : Scan_Result) return String
     with Post => True;

   --  Format a scan result as JSON.
   function Format_JSON (Result : Scan_Result) return String
     with Post => True;

end Security_Scanner;
