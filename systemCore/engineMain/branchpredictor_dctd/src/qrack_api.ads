with Interfaces.C;
with System; -- NEW: Make System visible for System.Address
use Interfaces.C;

package Qrack_Api is

   -- Opaque type for the Qrack simulator instance
   type qsim is private;
   
   --  function init_qsim (num_qubits : int) return qsim;
   --  pragma Import (C, init_qsim, "init_qsim");
   
   -- Gate operations
   procedure X (handle : qsim; qubit_index : int);
   pragma Import (C, X, "X");
   
   procedure RY (handle : qsim; qubit_index : int; angle : double);
   pragma Import (C, RY, "RY");
   
   procedure CNOT (handle : qsim; control_qubit : int; target_qubit : int);
   pragma Import (C, CNOT, "CNOT");
   
   -- Noise injection
   procedure set_depolarizing_error (handle : qsim; qubit_index : int; probability : double);
   pragma Import (C, set_depolarizing_error, "set_depolarizing_error");
   
   -- Measurement
   function M_all (handle : qsim) return unsigned_long; -- Assuming M_all returns a bitmask
   pragma Import (C, M_all, "M_all");
   
   -- Probability retrieval
   -- This assumes get_probs fills a user-provided array.
   type Double_Array is array (size_t) of aliased double;
   procedure get_probs (handle : qsim; probabilities : access Double_Array);
   pragma Import (C, get_probs, "get_probs");
   
private
   
   -- Define qsim as a C pointer (system address)
   type qsim is new System.Address;
   
end Qrack_Api;