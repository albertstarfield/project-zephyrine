pragma SPARK_Mode (Off);
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
      Length    : Natural) return Integer;

end LSH_Hash;
