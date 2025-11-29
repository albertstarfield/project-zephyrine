## DCTD Branch Predictor - Background Generate Implementation Summary

The branch predictor implementation has been moved to the `background_generate` function and now augments the RAG context for the background task.

**Completed Tasks:**

1.  **Branch Prediction Logic Moved**: The branch prediction logic, which was previously in `_direct_generate_logic`, has been moved to the beginning of the `background_generate` function.

2.  **Combined Embedding for Prediction**: The logic now combines the initial prompt of the `background_generate` task with recent interaction history to create a single text block, which is then used to generate a combined embedding for prediction.

3.  **Fuzzy Matching**: The predicted vector is used to find the most similar past interaction, and the text from that interaction is then used to perform a fuzzy search to find relevant information.

4.  **RAG Context Augmentation**: The result of the fuzzy search is appended to the RAG context of the `background_generate` task under the label "FutureInformationPrediction".

The system is now updated to use the branch prediction logic in the low-priority ELP0 background task, as you requested.