-- think_tag_sanitizer.ads
-- Strips <think>...</think> tags from LLM output text.
-- Native Ada replacement for src/python/think_tag_sanitizer.py.
--
-- Axioms:
--   - ISO/IEC 8652:2012 RM A.4.3 (Unbounded Strings)
--   - Pattern matching via manual scan (Ada has no built-in regex;
--     we implement a simple substring search for the fixed delimiters).
--
-- Complexity: O(n) single-pass scan.

pragma SPARK_Mode (Off);
-- c_binding: Ada.Text_IO.Put_Line for debug output — impure I/O operation cannot be expressed in SPARK

with Ada.Strings.Unbounded; use Ada.Strings.Unbounded;

package Think_Tag_Sanitizer is

   --  Sanitize_Think_Tags
   --  Remove <think>...</think> blocks from Text.
   --  If Remove_Content is True, removes tags AND their content.
   --  If Remove_Content is False, removes only the tag delimiters.
   --  Returns the sanitized string with leading/trailing whitespace trimmed.
   function Sanitize_Think_Tags
      -- @test: unit_test_exists  -- DO-178C 6.4.4
     (Text          : Unbounded_String;
      Remove_Content : Boolean := True)
      return Unbounded_String
     with Pre  => True,
          Post => True;

end Think_Tag_Sanitizer;
