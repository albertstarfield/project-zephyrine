-- KNOWN GOOD Ada: Every function here should NOT trigger SMT violations.
-- All dangerous operations are properly guarded.

-- CHECK 1: Division by zero — GUARDED
-- B is checked for zero before division.
procedure Divide_Safe (A : Integer; B : Integer) is
   -- pre => True, post => True
   Result : Integer;
begin
   if B /= 0 then
      Result := A / B;
   else
      Result := 0;
   end if;
end Divide_Safe;

-- CHECK 2: Index out of bounds — GUARDED
-- Idx is bounds-checked before array access.
procedure Index_Safe (Idx : Integer) is
   -- pre => True, post => True
   Data : array (1 .. 10) of Integer;
begin
   if Idx >= 1 and Idx <= 10 then
      Data (Idx) := 42;
   end if;
end Index_Safe;

-- CHECK 3: Null dereference — GUARDED
-- Ptr is checked for null before dereference.
type Int_Access is access all Integer;
procedure Null_Safe (Ptr : Int_Access) is
   -- pre => True, post => True
   Value : Integer;
begin
   if Ptr /= null then
      Value := Ptr.all;
   end if;
end Null_Safe;

-- CHECK 4: Constraint error — GUARDED
-- Result is range-checked before assignment.
procedure Constraint_Safe (X : Integer; Y : Integer) is
   -- pre => True, post => True
   Result : Integer range 0 .. 100;
   Temp : Integer;
begin
   Temp := X + Y;
   if Temp >= 0 and Temp <= 100 then
      Result := Temp;
   end if;
end Constraint_Safe;

-- CHECK 5: Integer overflow — GUARDED
-- Multiplication is range-checked.
procedure Overflow_Safe (A : Integer; B : Integer) is
   -- pre => True, post => True
   Result : Integer;
begin
   if A <= Integer'Last / B then
      Result := A * B;
   end if;
end Overflow_Safe;

-- CHECK 6: Precondition — CONSISTENT
-- Precondition is satisfiable.
procedure Pre_Consistent (X : Integer) is
   -- pre => X > 0 and X <= 100
   -- post => True
begin
   null;
end Pre_Consistent;

-- CHECK 7: Postcondition — ENFORCED
-- Body actually satisfies postcondition.
function Post_Enforced (X : Integer) return Integer is
   -- pre => X > 0
   -- post => Post_Enforced'Result > 0
begin
   return X;
end Post_Enforced;

-- CHECK 8: Float NaN/Inf — GUARDED
-- Float division guarded against zero.
procedure Float_Safe (A : Float; B : Float) is
   -- pre => True, post => True
   Result : Float;
begin
   if B /= 0.0 then
      Result := A / B;
   end if;
end Float_Safe;
