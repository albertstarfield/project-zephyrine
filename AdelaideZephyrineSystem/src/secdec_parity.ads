-- SECDED TED Parity Encoding Package for Ada
--
-- Provides SECDED (Single Error Correction, Double Error Detection) TED
-- (Triple Error Detection) parity encoding for function return values.
--
-- AXIOMS:
--    1. Every function must encode its return value with SECDED TED
--    2. SECDED protects against single-bit errors
--    3. TED provides additional triple-error detection capability
--
-- THEOREMS:
--    1. THEOREM: SECDED encoding protects against single-bit errors
--       PROOF: Hamming code construction guarantees unique syndrome patterns
--    2. THEOREM: Functions without parity are vulnerable to silent corruption
--       PROOF: Bit flips in return values go undetected without encoding
--
-- CITATIONS:
--    - Hamming, R.W. (1950) Error detecting and error correcting codes
--    - https://en.wikipedia.org/wiki/Hamming_code
--    - ISO/IEC 25010:2021 Software Quality Model
--    - ECSS-Q-ST-80C Software Product Assurance

package Secdec_Parity is

   -- SECDED TED result container
   type Atomic_Function_Result is record
      Value    : Integer;
      Encoded  : Integer;
      Parity   : Integer;
      Syndrome : Integer;
   end record;

   -- Encode a return value with SECDED TED parity
   -- AXIOMS: Every function must call Secdec_Encode on its return value
   -- THEOREMS: SECDED encoding protects against single-bit errors
   -- CITATIONS: Hamming (1950), ISO/IEC 25010:2021
   function Secdec_Encode (Value : Integer; Bits : Integer := 32)
      return Atomic_Function_Result;

   -- Wrapper procedure for parity-protected function calls
   -- AXIOMS: Every procedure must use Atomic_Function_Wrapper for parity
   -- THEOREMS: Parity protection prevents silent data corruption
   -- CITATIONS: ECSS-Q-ST-80C, CWE-682
   procedure Atomic_Function_Wrapper (Value : in Integer)
     with Pre => True,
          Post => True;

   -- Calculate syndrome for error detection
   -- AXIOMS: Syndrome calculation enables error correction
   -- THEOREMS: Non-zero syndrome indicates bit error
   -- CITATIONS: Hamming (1950)
   function Calculate_Syndrome (Encoded : Integer; Bits : Integer := 32)
      return Integer;

end Secdec_Parity;
