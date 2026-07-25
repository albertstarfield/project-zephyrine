-- KNOWN BAD Ada: Every function here SHOULD trigger SMT violations.

-- CHECK 1: Division by zero
-- Denominator B can be 0 — no guard.
procedure Divide_By_Zero (A : Integer; B : Integer) is
   -- pre => True, post => True
   Result : Integer;
begin
   Result := A / B;
end Divide_By_Zero;

-- CHECK 2: Index out of bounds
-- Index variable Idx not bounds-checked before use.
procedure Index_OOB (Idx : Integer) is
   -- pre => True, post => True
   Data : array (1 .. 10) of Integer;
begin
   Data (Idx) := 42;
end Index_OOB;

-- CHECK 3: Null dereference
-- Access type used without null guard.
type Int_Access is access all Integer;
procedure Null_Deref (Ptr : Int_Access) is
   -- pre => True, post => True
   Value : Integer;
begin
   Value := Ptr.all;
end Null_Deref;

-- CHECK 4: Constraint error
-- Arithmetic can exceed range 0..100.
procedure Constraint_Error_Demo (X : Integer; Y : Integer) is
   -- pre => True, post => True
   Result : Integer range 0 .. 100;
begin
   Result := X + Y;
end Constraint_Error_Demo;

-- CHECK 5: Integer overflow
-- Multiplication can exceed Integer'Last.
procedure Overflow_Demo (A : Integer; B : Integer) is
   -- pre => True, post => True
   Result : Integer;
begin
   Result := A * B;
end Overflow_Demo;

-- CHECK 6: Precondition contradiction
-- Pre requires X > 10 AND X < 5 — unreachable.
procedure Pre_Contradiction (X : Integer) is
   -- pre => X > 10 and X < 5
   -- post => True
begin
   null;
end Pre_Contradiction;

-- CHECK 7: Postcondition not enforced
-- Trivial body with postcondition.
function Trivial_With_Post (X : Integer) return Integer is
   -- pre => True
   -- post => Trivial_With_Post'Result > 0
begin
   return 0;
end Trivial_With_Post;

-- CHECK 8: Float NaN/Inf
-- Float division without NaN guard.
procedure Float_NaN_Demo (A : Float; B : Float) is
   -- pre => True, post => True
   Result : Float;
begin
   Result := A / B;
end Float_NaN_Demo;
