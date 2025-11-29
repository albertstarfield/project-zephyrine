I have implemented the preemption mechanism for the quantum simulation in `VectorCompute_Provider.py`.

**Summary of Changes:**

1.  **`CortexThoughts`**: The `_priority_quota_lock` is now passed to the `BranchPredictorProvider` during its initialization.

2.  **`BranchPredictorProvider`**: The provider has been updated to accept the `priority_lock` and pass it down to the `VectorComputeProvider`.

3.  **`VectorComputeProvider`**:
    *   The `__init__` method now accepts the `priority_lock`.
    *   The `run_qrnn` method now acquires the lock with `ELP0` (low priority) before starting the simulation.
    *   The simulation loop now checks for preemption after processing each feature in the input sequence. If an ELP1 task requests the lock, `run_qrnn` will raise a `TaskInterruptedException`.
    *   The lock is released upon completion or interruption of the simulation.
    *   The `qiskit-qrack-provider` has been integrated, and the backend selection logic has been updated to prioritize it when configured.

This implementation ensures that the resource-intensive quantum simulation can be preempted by higher-priority tasks, preventing it from blocking the main application flow.
