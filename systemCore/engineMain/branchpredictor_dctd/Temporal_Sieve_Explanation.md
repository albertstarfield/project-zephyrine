The "temporal sieve" is the process of converting a sequence of past interactions into a format that the quantum-inspired branch predictor (`run_qrnn`) can understand. This is implemented in the `_get_temporal_sieve_data` method in `AdelaideAlbertCortex.py`.

Here's a breakdown of how it works:

### 1. Data Retrieval

First, the method retrieves a list of recent interactions from the database. It specifically looks for interactions that have a saved embedding vector, as this is the basis for the content analysis.

### 2. Feature Extraction

For each interaction in the retrieved sequence, the method extracts a single 16-bit integer feature. This feature is a combination of two key pieces of information:

#### a) Content Hash (10 bits)

*   **What it is**: A 10-bit Locality Sensitive Hash (LSH) is calculated from the interaction's embedding vector.
*   **How it's done**: The `calculate_lsh_hash` function in `database.py` takes the high-dimensional embedding vector and projects it onto 10 random hyperplanes. The result of each projection determines one bit of the hash (e.g., positive result = 1, negative = 0).
*   **Purpose**: This hash represents the *content* or *topic* of the interaction in a very compressed format. Similar interactions will have similar LSH hashes.

#### b) Temporal Quantization (6 bits)

*   **What it is**: The time elapsed since the *previous* interaction is calculated and categorized into one of several "time bins".
*   **How it's done**: The time delta is calculated in seconds, and then assigned a 6-bit ID based on these categories:
    *   0-120 seconds
    *   120 seconds - 5 minutes
    *   5 minutes - 1 hour
    *   1 hour - 24 hours
    *   \> 24 hours
*   **Purpose**: This represents the *rhythm* or *cadence* of the conversation. A rapid back-and-forth will have small time bin IDs, while a long pause will have a large one.

### 3. Feature Combination

The 10-bit LSH hash and the 6-bit time bin ID are combined into a single 16-bit integer. This is done by bit-shifting the LSH hash to the left by 6 positions and then adding the time bin ID.

```
Combined Feature (16 bits) = [LSH Hash (10 bits)] [Time Bin ID (6 bits)]
```

### 4. Output

The `_get_temporal_sieve_data` method returns a list of these 16-bit integers. This sequence, which encodes both the "what" and the "when" of the conversation's history, is the final output of the temporal sieve and is ready to be fed into the `run_qrnn` method for prediction.
