from .VectorCompute_Provider import VectorComputeProvider
from loguru import logger
from priority_lock import PriorityQuotaLock

class BranchPredictorProvider:
    """
    Orchestrates the branch prediction process by using a vector compute provider.
    """

    def __init__(self, use_gpu: bool = True, priority_lock: PriorityQuotaLock = None):
        logger.info("Initializing Branch Predictor Provider...")
        self.compute_provider = VectorComputeProvider(use_gpu=use_gpu, priority_lock=priority_lock)
        self.priority_lock = priority_lock

    def predict_future_vector(self, temporal_sieve_data: list[dict]) -> dict | None:
        """
        Predicts a future 1024-dimensional vector based on a sequence of past interactions.

        :param temporal_sieve_data: A list of dictionaries, where each dictionary
                                    contains information about a past interaction,
                                    including the 'combined_feature_16bit'.
        :return: A dictionary containing the 'predicted_vector' and 
                 'predicted_time_horizon', or None if prediction fails.
        """
        if not temporal_sieve_data:
            logger.warning("Prediction skipped: temporal_sieve_data is empty.")
            return None

        # Extract the sequence of 16-bit features for the QRNN
        feature_sequence = [item['combined_feature_16bit'] for item in temporal_sieve_data]

        if not feature_sequence:
            logger.warning("Prediction skipped: no features could be extracted from sieve data.")
            return None
            
        try:
            predicted_vector = self.compute_provider.run_qrnn(feature_sequence, self.priority_lock)
            
            # For now, the time horizon is a fixed placeholder
            predicted_time_horizon = "+5 mins"
            
            logger.info(f"Branch prediction successful. Predicted vector shape: {predicted_vector.shape}")
            
            return {
                "predicted_vector": predicted_vector,
                "predicted_time_horizon": predicted_time_horizon
            }
            
        except Exception as e:
            logger.error(f"An error occurred during branch prediction: {e}")
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
