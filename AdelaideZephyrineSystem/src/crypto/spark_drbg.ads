with Interfaces.C; use Interfaces.C;
with Interfaces.C.Strings; use Interfaces.C.Strings;
with System;

package Spark_Drbg
  with SPARK_Mode => On
is
   -- FIPS 140-3 SP 800-90A CTR_DRBG (AES-256) state and logic in SPARK Ada.

   subtype Block_Index is Integer range 1 .. 16;
   type Block_Type is array (Block_Index) of unsigned_char;

   subtype Key_Index is Integer range 1 .. 32;
   type Key_Type is array (Key_Index) of unsigned_char;

   subtype Seed_Index is Integer range 1 .. 48;
   type Seed_Type is array (Seed_Index) of unsigned_char;

   type State_Type is record
      Key            : Key_Type;
      V              : Block_Type;
      Last_Block     : Block_Type;
      Reseed_Counter : Interfaces.Unsigned_64;
      Initialized    : Boolean;
      Last_Valid     : Boolean;
   end record;

   State : State_Type := (Key => [others => 0], 
                          V => [others => 0], 
                          Last_Block => [others => 0],
                          Reseed_Counter => 0,
                          Initialized => False,
                          Last_Valid => False);

   -- C bindings (Imported as procedures to bypass SPARK function limitations on out-parameters)
   procedure C_AES256_ECB_Encrypt (Key : in Key_Type; Plaintext : in Block_Type; Ciphertext : out Block_Type; Result : out int)
     with Import => True, Convention => C, External_Name => "adl_aes256_ecb_encrypt_wrapper";
     
   --  C_Gather_Entropy: C FFI binding to gather entropy from the system.
   procedure C_Gather_Entropy (Buffer : out Seed_Type; Len : in size_t; Result : out int)
     with Import => True, Convention => C, External_Name => "adl_gather_entropy_wrapper";

   --  Instantiate: Initializes the DRBG with entropy and personalization string.
   procedure Instantiate (Success : out Boolean)
     with Global => (In_Out => State), Pre => True, Post => True;

   type Output_Buffer is array (Natural range <>) of unsigned_char;
   
   --  Generate: Generates random bytes using the DRBG.
   procedure Generate (Output : out Output_Buffer; Success : out Boolean)
     with Global => (In_Out => State),
          Pre => Output'Length <= 524288, Post => True;
     
   --  Clear: Clears the DRBG state (zeroizes key and V).
   procedure Clear
     with Global => (Output => State);

   -- C ABI Wrappers
   function Adl_Drbg_Init (Entropy_Bytes : size_t; Pers_String : chars_ptr; Err_Buf : chars_ptr) return int
     with Export => True, Convention => C, External_Name => "adl_drbg_init";

   function Adl_Drbg_Generate (Out_Buf : System.Address; Len : size_t) return int -- FFI: System.Address required for C binding
     with Export => True, Convention => C, External_Name => "adl_drbg_generate";

   --  Adl_Drbg_Clear: C ABI wrapper to clear the DRBG state.
   procedure Adl_Drbg_Clear
     with Export => True, Convention => C, External_Name => "adl_drbg_clear";

end Spark_Drbg;
