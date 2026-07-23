from VectorCompute_Provider import VectorComputeProvider
from loguru import logger
from priority_lock import PriorityQuotaLock
import numpy as np
from CortexConfiguration import *
from typing import Any, Dict, List, Optional, Tuple, Union  # Added Union

class BranchPredictorProvider:
    """
    Orchestrates the branch prediction process by using a vector compute provider.
    """

    def __init__(self, use_gpu: bool = True, priority_lock: PriorityQuotaLock = None):
        logger.info("Initializing Branch Predictor Provider...")
        self.compute_provider = VectorComputeProvider(use_gpu=use_gpu, priority_lock=priority_lock)
        self.priority_lock = priority_lock

    def predict_future_vector(self, temporal_sieve_data: list[dict],
                              current_embedding: np.ndarray,
                              previous_embedding: np.ndarray) -> dict | None:
        """
        Predicts future vector based on trajectory of current vs previous interaction.

        :param temporal_sieve_data: Historical LSH/Time data.
        :param current_embedding: 1024-D vector of user's current input.
        :param previous_embedding: 1024-D vector of the last AI response/User input.
        """
        # 1. Calculate Trajectory Delta
        # What direction did the conversation move in?
        delta_vector = current_embedding - previous_embedding

        # 2. Compress 1024-D Delta to 14 dimensions (Simple Averaging)
        # We reserve 2 dimensions for LSH and Time.
        # 1024 / 14 ≈ 73.1. We slice into 14 chunks.
        reduced_features = []
        chunk_size = 1024 // 14
        for i in range(14):
            start = i * chunk_size
            end = start + chunk_size
            chunk_mean = np.mean(delta_vector[start:end])
            # Normalize via Tanh to keep between -1 and 1, then map to 0.0-1.0
            norm_val = (np.tanh(chunk_mean) + 1) / 2
            reduced_features.append(norm_val)

        # 3. Get latest LSH and Time from sieve data
        latest_sieve = temporal_sieve_data[-1]

        # Normalize 10-bit LSH (0-1023) to 0.0-1.0
        lsh_val = latest_sieve['lsh_hash'] / 1023.0

        # Normalize Time Delta (assuming max relevant delta is 1 hour = 3600s)
        time_val = min(latest_sieve['time_delta_seconds'], 3600) / 3600.0

        # 4. Create the 16-D State Vector
        # [14 dim vector dynamics] + [LSH Context] + [Time Decay]
        state_vector = np.array(reduced_features + [lsh_val, time_val], dtype=np.float32)

        # 5. Run QRNN with this state as a sequence of 1 (or loop if we used full history)
        # Here we treat the trajectory as a single complex step update
        try:
            predicted_vector = self.compute_provider.run_qrnn([state_vector], self.priority_lock)
            return {
                "predicted_vector": predicted_vector,
                "predicted_time_horizon": "+1 step"
            }
        except Exception as e:
            logger.error(f"Prediction logic failed: {e}")
            return None


if __name__ == '__main__':
    # Example Usage
    logger.info("--- BranchPredictorProvider Self-Test ---")
    
    # Simulate temporal sieve data from AdelaideAlbertCortex
    mock_sieve_data = [
        {'combined_feature_16bit': 12345},
        {'combined_feature_16bit': 54321},
        {'combined_feature_16bit': 23456},
        {'combined_feature_16bit': 65432},
    ]
    
    branch_predictor = BranchPredictorProvider(priority_lock=PriorityQuotaLock())
    prediction_result = branch_predictor.predict_future_vector(mock_sieve_data)
    
    if prediction_result:
        logger.info(f"Self-test prediction result vector shape: {prediction_result['predicted_vector'].shape}")
    else:
        logger.error("Self-test failed to get a prediction.")
