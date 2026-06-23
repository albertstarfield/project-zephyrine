pragma SPARK_Mode (On);

package Model_Types is

   type Model_Type is (Qwen_0_8B, Snowball_Enaga_Orchestrator, Qwen_Embedding, MMProj);

   --  ELP levels hierarchy:
   --  ELP0: Background Literature Indexing (Lowest Priority)
   --  ELP1: Active RAG / Memory Retrieval (User Interaction)
   --  ELP2: StellaIcarus Hooks (Deterministic API Logic)
   --  ELP3: ZenithOrion (Deterministic 1ms Pacing Lock - Highest Frequency)
   type ELP_Level is (ELP0, ELP1, ELP2, ELP3);

end Model_Types;
