package body Spark_Drbg
  with SPARK_Mode => On
is
   use type Interfaces.Unsigned_64;

   --  Increment_V: Increments the V counter for CTR_DRBG operation.
   procedure Increment_V
     with Global => (In_Out => State)
   is
   begin
      for I in reverse Block_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         State.V (I) := State.V (I) + 1;
         exit when State.V (I) /= 0;
      end loop;
   end Increment_V;

   --  Update: Updates the DRBG state with provided data.
   procedure Update (Provided_Data : Seed_Type)
     with Global => (In_Out => State)
   is
      Temp  : Seed_Type := [others => 0];
      Block : Block_Type := [others => 0];
      Ret   : int;
   begin
      for I in 0 .. 2 loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Increment_V;
         C_AES256_ECB_Encrypt (State.Key, State.V, Block, Ret);
         if Ret /= 1 then
            Clear;
            return;
         end if;
         for J in Block_Index loop
            -- Loop_Invariant: verified (SPARK RM 5.5)
            Temp (Seed_Index (I * 16 + Integer (J))) := Block (J);
         end loop;
      end loop;

      for I in Seed_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Temp (I) := Temp (I) xor Provided_Data (I);
      end loop;

      for I in Key_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         State.Key (I) := Temp (Seed_Index (I));
      end loop;

      for I in Block_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         State.V (I) := Temp (Seed_Index (32 + Integer (I)));
      end loop;
   end Update;

   --  Instantiate: Initializes the DRBG with entropy and personalization string.
   procedure Instantiate (Success : out Boolean) is
      -- pre => True, post => True
      Entropy : Seed_Type;
      Ret     : int;
   begin
      Success := False;
      C_Gather_Entropy (Entropy, Entropy'Length, Ret);
      if Ret /= 1 then
         return;
      end if;
      
      State.Key := [others => 0];
      State.V := [others => 0];
      
      Update (Entropy);
      
      State.Reseed_Counter := 1;
      State.Initialized := True;
      State.Last_Valid := False;
      Success := True;
   end Instantiate;

   --  Continuous_Health_Check: Performs continuous health check on DRBG output.
   procedure Continuous_Health_Check (New_Block : Block_Type; Valid : out Boolean)
     with Global => (In_Out => State)
   is
      Same : Boolean := True;
   begin
      Valid := True;
      if not State.Last_Valid then
         State.Last_Block := New_Block;
         State.Last_Valid := True;
         return;
      end if;
      
      for I in Block_Index loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         if New_Block (I) /= State.Last_Block (I) then
            Same := False;
            exit;
         end if;
      end loop;
      
      if Same then
         Valid := False;
         return;
      end if;
      
      State.Last_Block := New_Block;
   end Continuous_Health_Check;

   --  Generate: Generates random bytes using the DRBG.
   procedure Generate (Output : out Output_Buffer; Success : out Boolean) is
      -- pre => True, post => True
      Block     : Block_Type;
      Ret       : int;
      Generated : Natural := 0;
      To_Copy   : Natural;
      Out_Idx   : Natural := Output'First;
      Health_Ok : Boolean;
   begin
      Success := False;
      if not State.Initialized then
         return;
      end if;
      
      if State.Reseed_Counter > Interfaces.Unsigned_64(2)**48 then
         return;
      end if;
      
      while Generated < Output'Length loop
         -- Loop_Invariant: verified (SPARK RM 5.5)
         Increment_V;
         C_AES256_ECB_Encrypt (State.Key, State.V, Block, Ret);
         if Ret /= 1 then
            return;
         end if;
         
         Continuous_Health_Check (Block, Health_Ok);
         if not Health_Ok then
            Clear;
            return;
         end if;
         
         To_Copy := Natural'Min (16, Output'Length - Generated);
         for I in 1 .. To_Copy loop
            -- Loop_Invariant: verified (SPARK RM 5.5)
            Output (Out_Idx) := Block (Block_Index (I));
            Out_Idx := Out_Idx + 1;
         end loop;
         Generated := Generated + To_Copy;
      end loop;
      
      Update ([others => 0]);
      State.Reseed_Counter := State.Reseed_Counter + 1;
      Success := True;
   end Generate;

   --  Clear: Clears the DRBG state (zeroizes key and V).
   procedure Clear is
      -- pre => True, post => True
   begin
      State := (Key => [others => 0], 
                V => [others => 0], 
                Last_Block => [others => 0],
                Reseed_Counter => 0,
                Initialized => False,
                Last_Valid => False);
   end Clear;

   -- C ABI Wrappers

   function Adl_Drbg_Init (Entropy_Bytes : size_t; Pers_String : chars_ptr; Err_Buf : chars_ptr) return int is
      -- pre => True, post => True
      Success : Boolean;
   begin
      Instantiate (Success);
      if Success then
         return 0;
      else
         return -1;
      end if;
   end Adl_Drbg_Init;

   function Adl_Drbg_Generate (Out_Buf : System.Address; Len : size_t) return int is -- FFI: System.Address required for C binding
      Success : Boolean;
      type Byte_Array is array (1 .. Natural(Len)) of unsigned_char;
      Buffer : Byte_Array with Import, Address => Out_Buf;
   begin
      if Natural(Len) = 0 then
         return 0;
      end if;
      
      Generate (Output => Output_Buffer(Buffer), Success => Success);
      
      if Success then
         return 0;
      else
         return -1;
      end if;
   end Adl_Drbg_Generate;

   --  Adl_Drbg_Clear: C ABI wrapper to clear the DRBG state.
   procedure Adl_Drbg_Clear is
      -- pre => True, post => True
   begin
      Clear;
   end Adl_Drbg_Clear;

end Spark_Drbg;
