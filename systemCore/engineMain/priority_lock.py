# priority_lock.py (New File)

import threading
import time
import subprocess # To store the process handle
from typing import Optional, Tuple
from loguru import logger
import platform
import os

# Priority Levels
ELP0 = 0 # Background Tasks (File Indexer, Reflection)
ELP1 = 1 # Foreground User Requests

class PriorityQuotaLock:
    """
    A lock supporting priority levels (ELP0, ELP1) and an interruption quota.
    ELP1 requests can interrupt ELP0 requests holding the lock, up to a quota limit.
    Manages killing the associated ELP0 worker process upon interruption.
    """
    QUOTA_MAX = 100

    def __init__(self):
        self._condition = threading.Condition(threading.Lock())
        self._is_locked = False
        self._lock_holder_priority: Optional[int] = None
        self._lock_holder_proc: Optional[subprocess.Popen] = None
        self._lock_holder_thread_ident: Optional[int] = None
        self._elp1_interrupt_quota: int = self.QUOTA_MAX
        self._elp1_waiting_count = 0 # How many ELP1 threads are waiting

        logger.info("🚦 PriorityQuotaLock initialized. ELP1 Interrupt Quota: {}", self.QUOTA_MAX)

    def acquire(self, priority: int, proc: Optional[subprocess.Popen] = None, timeout: Optional[float] = None) -> bool:
        """
        Acquires the lock based on priority and quota.
        ELP1 can interrupt ELP0 if quota allows.
        Stores the associated process 'proc' if priority is ELP0.

        Returns:
            bool: True if the lock was acquired, False otherwise (e.g., timeout).
        """
        acquire_start_time = time.monotonic()
        requesting_thread_ident = threading.get_ident()
        log_prefix = f"PQLock|ELP{priority}|Thr{requesting_thread_ident}"
        logger.trace("{}:: Attempting acquire...", log_prefix)

        if priority not in [ELP0, ELP1]:
            raise ValueError("Invalid priority level")
        if priority == ELP0 and proc is not None:
            # This is problematic - proc is only known *after* the lock is acquired
            # and the process is started. We need to set it later.
            logger.warning("{}:: Process passed during acquire for ELP0, should be set later.", log_prefix)
            # raise ValueError("Process should not be provided during acquire for ELP0")

        with self._condition:
            logger.trace("{}:: Acquired internal condition lock.", log_prefix)
            if priority == ELP1:
                self._elp1_waiting_count += 1

            # --- Wait Condition Loop ---
            while True:
                can_interrupt = (priority == ELP1 and
                                 self._is_locked and
                                 self._lock_holder_priority == ELP0 and
                                 self._elp1_interrupt_quota > 0)

                if not self._is_locked:
                    # Lock is free, acquire it
                    self._is_locked = True
                    self._lock_holder_priority = priority
                    self._lock_holder_proc = None # Will be set later for ELP0
                    self._lock_holder_thread_ident = requesting_thread_ident

                    if priority == ELP0:
                         # Reset ELP1 quota when ELP0 gets a chance
                         self._elp1_interrupt_quota = self.QUOTA_MAX
                         logger.info("{}:: Acquired lock (Free). Resetting ELP1 quota to {}.", log_prefix, self.QUOTA_MAX)
                    else: # ELP1 acquired a free lock
                         logger.info("{}:: Acquired lock (Free). Quota remaining: {}", log_prefix, self._elp1_interrupt_quota)

                    if priority == ELP1: self._elp1_waiting_count -= 1
                    return True # Lock acquired

                elif can_interrupt:
                    # ELP1 needs to interrupt ELP0
                    logger.warning("{}:: Attempting IMMEDIATE INTERRUPT (SIGKILL/Force) ELP0 (Holder Thr: {})! Quota: {}->{}",
                                   log_prefix, self._lock_holder_thread_ident,
                                   self._elp1_interrupt_quota, self._elp1_interrupt_quota - 1)

                    # --- Kill the ELP0 worker process FORCEFULLY ---
                    interrupted_proc = self._lock_holder_proc
                    if interrupted_proc and interrupted_proc.poll() is None:
                        pid_to_kill = interrupted_proc.pid
                        logger.warning("{}:: Forcefully terminating ELP0 process PID: {}", log_prefix, pid_to_kill)
                        try:
                            # --- MODIFICATION HERE ---
                            # Directly use kill() or platform equivalent
                            if platform.system() == "Windows":
                                # Windows: Use taskkill with /F for force
                                kill_cmd = ["taskkill", "/PID", str(pid_to_kill), "/F"]
                                kill_proc = subprocess.run(kill_cmd, capture_output=True, text=True, check=False)
                                if kill_proc.returncode != 0:
                                    logger.error("{}:: taskkill /F failed for PID {}: {}", log_prefix, pid_to_kill, kill_proc.stderr.strip())
                                else:
                                    logger.info("{}:: taskkill /F successful for PID {}", log_prefix, pid_to_kill)
                            else:
                                # Unix-like: Use os.kill with SIGKILL
                                import signal as sig # Import signal module
                                os.kill(pid_to_kill, sig.SIGKILL)
                                logger.info("{}:: Sent SIGKILL to PID {}", log_prefix, pid_to_kill)
                            # --- END MODIFICATION ---

                            # Wait briefly to allow OS to clean up, process state might not update instantly
                            interrupted_proc.wait(timeout=1.0)

                        except ProcessLookupError:
                             logger.warning("{}:: ELP0 process PID {} not found during kill attempt (already exited?).", log_prefix, pid_to_kill)
                        except subprocess.TimeoutExpired:
                            logger.error("{}:: Wait after kill timed out for PID {}. Process might be stuck.", log_prefix, pid_to_kill)
                        except Exception as kill_err:
                            logger.error("{}:: Error forcefully killing ELP0 process PID {}: {}", log_prefix, pid_to_kill, kill_err)
                            # Continue? If kill fails, ELP0 might not release lock. Still risky, proceed with lock takeover.
                    # --- End Kill ---

                    # Forcefully take over the lock state (Same as before)
                    self._is_locked = True
                    self._lock_holder_priority = ELP1
                    self._lock_holder_proc = None
                    self._lock_holder_thread_ident = requesting_thread_ident
                    self._elp1_interrupt_quota -= 1

                    logger.info("{}:: FORCE-INTERRUPTED ELP0 and acquired lock. Quota remaining: {}", log_prefix, self._elp1_interrupt_quota)
                    self._condition.notify_all()
                    self._elp1_waiting_count -= 1
                    return True # Lock acquired via interruption

                else:
                    # Cannot acquire now (locked by higher/same priority, or ELP1 quota exhausted)
                    logger.trace("{}:: Lock held by ELP{} (Thr: {}). Waiting...",
                                 log_prefix, self._lock_holder_priority, self._lock_holder_thread_ident)
                    if priority == ELP1 and self._lock_holder_priority == ELP0 and self._elp1_interrupt_quota <= 0:
                        logger.warning("{}:: ELP1 waiting due to EXHAUSTED QUOTA ({} <= 0).", log_prefix, self._elp1_interrupt_quota)

                    # Wait for a signal or timeout
                    wait_start = time.monotonic()
                    signaled = self._condition.wait(timeout=timeout)
                    wait_duration = time.monotonic() - wait_start

                    if not signaled and timeout is not None:
                        logger.warning("{}:: Acquire timed out after {:.2f}s.", log_prefix, wait_duration)
                        if priority == ELP1: self._elp1_waiting_count -= 1
                        return False # Timed out

                    # If signaled, loop continues to re-evaluate conditions
                    logger.trace("{}:: Woke up from wait ({:.2f}s). Re-evaluating conditions...", log_prefix, wait_duration)

            # --- End Wait Condition Loop ---
            # This part should ideally not be reached due to return statements in the loop
            # logger.error("{}:: Exited acquire loop unexpectedly.", log_prefix)
            # if priority == ELP1: self._elp1_waiting_count -= 1
            # return False


    def set_holder_process(self, proc: subprocess.Popen):
        """Stores the Popen object associated with the ELP0 lock holder."""
        with self._condition:
            log_prefix = f"PQLock|SetProc|Thr{threading.get_ident()}"
            if self._is_locked and self._lock_holder_priority == ELP0 and self._lock_holder_thread_ident == threading.get_ident():
                if self._lock_holder_proc is not None and self._lock_holder_proc is not proc:
                     logger.warning("{}:: Overwriting existing process for ELP0 holder.", log_prefix)
                self._lock_holder_proc = proc
                logger.trace("{}:: Associated process PID {} with ELP0 lock holder.", log_prefix, proc.pid if proc else 'None')
            elif self._is_locked and self._lock_holder_priority == ELP1:
                 logger.error("{}:: Attempted to set process for ELP1 holder. Ignoring.", log_prefix)
            elif not self._is_locked:
                 logger.error("{}:: Attempted to set process when lock not held. Ignoring.", log_prefix)
            else: # Lock held by different ELP0 thread?
                 logger.error("{}:: Attempted to set process, but lock held by different thread ({}). Ignoring.", log_prefix, self._lock_holder_thread_ident)


    def release(self):
        """Releases the lock and notifies waiting threads."""
        releasing_thread_ident = threading.get_ident()
        log_prefix = f"PQLock|RLS|Thr{releasing_thread_ident}"
        logger.trace("{}:: Attempting release...", log_prefix)

        with self._condition:
            logger.trace("{}:: Acquired internal condition lock.", log_prefix)
            if not self._is_locked:
                logger.warning("{}:: Lock not held, cannot release.", log_prefix)
                return

            if self._lock_holder_thread_ident != releasing_thread_ident:
                # This might happen if ELP0 was interrupted and killed by ELP1,
                # but the original ELP0 thread eventually tries to release.
                 logger.warning("{}:: Thread attempting release does not match current lock holder ({}). Possible interruption occurred.", log_prefix, self._lock_holder_thread_ident)
                 # Don't actually release if the holder is different, the interruptor already took over.
                 return

            # Clear lock state
            holder_prio = self._lock_holder_priority
            self._is_locked = False
            self._lock_holder_priority = None
            self._lock_holder_proc = None
            self._lock_holder_thread_ident = None

            logger.info("{}:: Released lock (was held by ELP{}). Notifying waiting threads.", log_prefix, holder_prio)
            # Notify potentially waiting threads
            self._condition.notify_all()

    def get_status(self) -> Tuple[bool, Optional[int], int]:
        """Returns (is_locked, holder_priority, elp1_quota)."""
        with self._condition:
            return self._is_locked, self._lock_holder_priority, self._elp1_interrupt_quota