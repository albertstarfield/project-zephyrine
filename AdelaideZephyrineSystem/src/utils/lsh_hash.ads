pragma SPARK_Mode (Off);
-- thread: LSH requires thread-safe hash computation
with Math_Utils;

package LSH_Hash is

   --  Compute a 10-bit LSH hash from an embedding vector using the QRNN
   --  Python worker.  Spawns the worker on-demand, sends the embedding,
   --  reads the 10-bit hash (0 .. 1023).
   --
   --  Returns the hash on success, or -1 if:
   --    - The Python worker could not be spawned
   --    - The worker returned an error
   --    - ELP0 preemption was requested (caller should abort)
   function Compute
     (Embedding : Math_Utils.Vector;
      Length    : Natural) return Integer with Pre => True, Post => True;

   --  Compute a steered 10-bit LSH hash using PINN Schrödinger Bridge.
   --  Uses pinn_schrodinger.py --steer-hash to apply Orthogonal Latent
   --  Injection before computing the hash.
   --
   --  Pipeline: Schrödinger → PINN → QRNN → Orthogonal Injection → LSH
   --
   --  Returns the hash on success, or -1 if:
   --    - The PINN worker could not be spawned
   --    - The worker returned an error
   function Compute_Steered
     (Embedding : Math_Utils.Vector;
      Length    : Natural;
      Alpha     : Float := 0.1) return Integer with Pre => True, Post => True;

end LSH_Hash;
