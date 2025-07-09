# priority_lock.py (New File)

import threading
import time
import subprocess  # To store the process handle
from typing import Optional, Tuple
from loguru import logger
import platform
import os

# --- Configuration Import ---
try:
    from CortexConfiguration import AGENTIC_RELAXATION_MODE, AGENTIC_RELAXATION_PRESETS, \
        AGENTIC_RELAXATION_PERIOD_SECONDS
except ImportError:
    # Define fallbacks if config can't be imported (e.g., during testing)
    AGENTIC_RELAXATION_MODE = "Default"
    AGENTIC_RELAXATION_PRESETS = {"default": 0}
    AGENTIC_RELAXATION_PERIOD_SECONDS = 2.0

# Priority Levels
ELP0 = 0  # Background Tasks (File Indexer, Reflection)
ELP1 = 1  # Foreground User Requests


class AgenticRelaxationThread(threading.Thread):
    """
    A thread that implements PWM-style lock acquisition on ELP0 to throttle
    background tasks and manage thermals/power.
    """

    def __init__(self, lock: 'PriorityQuotaLock', duty_cycle_off: float, period_sec: float,
                 stop_event: threading.Event):
        super().__init__(name="AgenticRelaxationThread", daemon=True)
        self.lock = lock
        self.duty_cycle_off = duty_cycle_off  # e.g., 0.3 for 30% off time
        self.period_sec = period_sec
        self.stop_event = stop_event
        self.off_time = self.period_sec * self.duty_cycle_off
        self.on_time = self.period_sec * (1.0 - self.duty_cycle_off)
        logger.info(
            f"AgenticRelaxationThread initialized. Off-Time: {self.off_time:.2f}s, On-Time: {self.on_time:.2f}s per cycle.")

    def run(self):
        logger.info("✅ AgenticRelaxationThread started.")
        while not self.stop_event.is_set():
            try:
                # --- OFF Cycle: Hold the lock as ELP0 ---
                if self.off_time > 0:
                    # Acquire the lock as a low-priority (ELP0) task.
                    # This can be interrupted by an ELP1 task.
                    # We don't register a process, as this thread doesn't have one to kill.
                    was_acquired = self.lock.acquire(priority=ELP0,
                                                     timeout=self.on_time)  # Wait for lock during on-time

                    if was_acquired:
                        try:
                            # Hold the lock for the "off" duration
                            logger.trace(f"Relaxation thread acquired ELP0 lock. Holding for {self.off_time:.2f}s...")
                            self.stop_event.wait(self.off_time)
                        finally:
                            logger.trace("Relaxation thread releasing ELP0 lock.")
                            self.lock.release()
                    else:
                        # Could not acquire lock during on_time, means an ELP0 task is running.
                        # This is fine, we just wait and try again next cycle.
                        logger.trace(
                            f"Relaxation thread could not acquire lock, another ELP0 task is active. Waiting for next cycle.")
                        self.stop_event.wait(self.period_sec)
                else:  # on_time is 100%
                    self.stop_event.wait(self.period_sec)

            except Exception as e:
                logger.error(f"Error in AgenticRelaxationThread loop: {e}")
                # Avoid busy-looping on error
                self.stop_event.wait(5)

        logger.info("🛑 AgenticRelaxationThread has been shut down.")


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
        self._elp1_waiting_count = 0  # How many ELP1 threads are waiting

        # --- NEW: Relaxation Thread ---
        self._relaxation_thread: Optional[AgenticRelaxationThread] = None
        self._relaxation_stop_event = threading.Event()
        self._initialize_relaxation()

        logger.info("🚦 PriorityQuotaLock initialized. ELP1 Interrupt Quota: {}", self.QUOTA_MAX)

    def _initialize_relaxation(self):
        mode = str(AGENTIC_RELAXATION_MODE).lower().replace(" ", "")
        presets = {k.lower(): v for k, v in AGENTIC_RELAXATION_PRESETS.items()}

        duty_cycle_off_percent = 0
        if mode in presets:
            duty_cycle_off_percent = presets[mode]
        else:
            try:
                duty_cycle_off_percent = float(mode)
            except ValueError:
                logger.warning(f"Invalid AGENTIC_RELAXATION_MODE '{AGENTIC_RELAXATION_MODE}'. Defaulting to 0%.")

        duty_cycle_off_percent = max(0, min(100, duty_cycle_off_percent))  # Clamp between 0 and 100

        if duty_cycle_off_percent > 0:
            duty_cycle_float = duty_cycle_off_percent / 100.0
            logger.info(f"Activating AgenticRelaxation with {duty_cycle_off_percent}% off-cycle.")
            self._relaxation_thread = AgenticRelaxationThread(
                lock=self,
                duty_cycle_off=duty_cycle_float,
                period_sec=AGENTIC_RELAXATION_PERIOD_SECONDS,
                stop_event=self._relaxation_stop_event
            )
            self._relaxation_thread.start()
        else:
            logger.info("AgenticRelaxation is disabled (0% off-cycle).")

    def shutdown_relaxation_thread(self):
        if self._relaxation_thread and self._relaxation_thread.is_alive():
            logger.info("Signaling AgenticRelaxationThread to stop...")
            self._relaxation_stop_event.set()
            self._relaxation_thread.join(timeout=AGENTIC_RELAXATION_PERIOD_SECONDS + 1)
            if self._relaxation_thread.is_alive():
                logger.warning("AgenticRelaxationThread did not stop in time.")

    def acquire(self, priority: int, timeout: Optional[float] = None) -> bool:
        acquire_start_time = time.monotonic()
        requesting_thread_ident = threading.get_ident()
        log_prefix = f"PQLock|ACQ|ELP{priority}|Thr{requesting_thread_ident}"  # Changed prefix for acquire
        logger.trace("{}:: Attempting acquire...", log_prefix)

        if priority not in [ELP0, ELP1]:
            raise ValueError("Invalid priority level")

        with self._condition:
            logger.trace(
                "{}:: Acquired internal condition. Current state: locked={}, holder_prio={}, holder_proc_pid={}, quota={}, elp1_wait_cnt={}",
                log_prefix, self._is_locked, self._lock_holder_priority,
                self._lock_holder_proc.pid if self._lock_holder_proc else "None",
                self._elp1_interrupt_quota, self._elp1_waiting_count)

            if priority == ELP1:
                self._elp1_waiting_count += 1
                logger.trace("{}:: Incremented ELP1 waiting count to {}", log_prefix, self._elp1_waiting_count)

            loop_iteration = 0
            while True:
                loop_iteration += 1
                logger.trace("{}:: Loop iteration {}. Checking conditions...", log_prefix, loop_iteration)

                is_locked_val = self._is_locked
                holder_priority_val = self._lock_holder_priority
                interrupt_quota_val = self._elp1_interrupt_quota

                can_interrupt = (priority == ELP1 and
                                 is_locked_val and
                                 holder_priority_val == ELP0 and
                                 interrupt_quota_val > 0)

                logger.trace(
                    "{}:: Conditions eval: priority_is_ELP1={}, is_locked={}, holder_is_ELP0={}, quota_OK={}. ==> can_interrupt={}",
                    log_prefix, (priority == ELP1), is_locked_val, (holder_priority_val == ELP0),
                    (interrupt_quota_val > 0), can_interrupt)

                if not is_locked_val:
                    self._is_locked = True
                    self._lock_holder_priority = priority
                    self._lock_holder_proc = None
                    self._lock_holder_thread_ident = requesting_thread_ident
                    if priority == ELP0:
                        self._elp1_interrupt_quota = self.QUOTA_MAX
                        logger.info("{}:: Acquired lock (was Free). Reset ELP1 quota to {}.", log_prefix,
                                    self.QUOTA_MAX)
                    else:  # ELP1 acquired a free lock
                        logger.info("{}:: Acquired lock (was Free). Quota remaining: {}", log_prefix,
                                    self._elp1_interrupt_quota)
                    if priority == ELP1: self._elp1_waiting_count -= 1
                    return True

                elif can_interrupt:
                    logger.warning("{}:: INTERRUPT PATH: Conditions MET. Holder ELP{} (Proc PID: {}). Quota: {}->{}",
                                   log_prefix, holder_priority_val,
                                   self._lock_holder_proc.pid if self._lock_holder_proc else "None",
                                   interrupt_quota_val, interrupt_quota_val - 1)

                    interrupted_proc = self._lock_holder_proc
                    original_holder_thread_ident = self._lock_holder_thread_ident  # For logging

                    if interrupted_proc and interrupted_proc.poll() is None:
                        pid_to_kill = interrupted_proc.pid
                        logger.warning("{}:: Forcefully terminating ELP0 process PID: {} (was held by Thr: {})",
                                       log_prefix, pid_to_kill, original_holder_thread_ident)
                        try:
                            if platform.system() == "Windows":
                                kill_cmd = ["taskkill", "/PID", str(pid_to_kill), "/F", "/T"]  # Add /T for process tree
                                kill_proc_run = subprocess.run(kill_cmd, capture_output=True, text=True, check=False)
                                if kill_proc_run.returncode != 0:
                                    logger.error("{}:: taskkill failed for PID {}: {}", log_prefix, pid_to_kill,
                                                 kill_proc_run.stderr.strip())
                                else:
                                    logger.info("{}:: taskkill successful for PID {}", log_prefix, pid_to_kill)
                            else:
                                import signal as sig
                                os.kill(pid_to_kill, sig.SIGKILL)  # SIGKILL for forceful termination
                                logger.info("{}:: Sent SIGKILL to PID {}", log_prefix, pid_to_kill)

                            # Brief wait for OS to process the kill signal
                            try:
                                interrupted_proc.wait(timeout=0.5)  # Shorter wait after kill
                                logger.info("{}:: ELP0 process PID {} terminated after kill.", log_prefix, pid_to_kill)
                            except subprocess.TimeoutExpired:
                                logger.warning(
                                    "{}:: Wait after kill timed out for PID {}. Process might be stuck or OS is slow.",
                                    log_prefix, pid_to_kill)
                        except ProcessLookupError:
                            logger.warning("{}:: ELP0 process PID {} not found during kill (already exited?).",
                                           log_prefix, pid_to_kill)
                        except Exception as kill_err:
                            logger.error("{}:: Error killing ELP0 process PID {}: {}", log_prefix, pid_to_kill,
                                         kill_err)
                    else:
                        logger.warning(
                            "{}:: ELP0 lock holder process (PID: {}) not found or already exited. Cannot kill. Taking over lock.",
                            log_prefix, interrupted_proc.pid if interrupted_proc else "UnknownPID")

                    # Take over lock state
                    self._is_locked = True  # Already true, but reaffirm
                    self._lock_holder_priority = ELP1  # Take over with ELP1
                    self._lock_holder_proc = None  # ELP1 tasks don't register a proc this way
                    self._lock_holder_thread_ident = requesting_thread_ident  # New holder
                    self._elp1_interrupt_quota -= 1

                    logger.info(
                        "{}:: INTERRUPTED ELP0 and acquired lock (was held by Thr: {}). Quota remaining: {}. Notifying all.",
                        log_prefix, original_holder_thread_ident, self._elp1_interrupt_quota)
                    self._condition.notify_all()  # Notify any other waiters (though ELP1 took it)
                    if priority == ELP1: self._elp1_waiting_count -= 1  # This ELP1 is no longer waiting
                    return True

                else:  # Cannot acquire now (e.g., ELP0 trying to get lock held by ELP1, or ELP1 quota exhausted for interrupting ELP0)
                    logger.trace(
                        "{}:: WAIT PATH: Lock held by ELP{} (Thr: {}). Quota: {}. Cannot acquire/interrupt now. Waiting...",
                        log_prefix, holder_priority_val, self._lock_holder_thread_ident, interrupt_quota_val)
                    if priority == ELP1 and holder_priority_val == ELP0 and interrupt_quota_val <= 0:
                        logger.warning(
                            "{}:: ELP1 waiting specifically due to EXHAUSTED QUOTA ({} <= 0). Will wait for ELP0 to release.",
                            log_prefix, interrupt_quota_val)

                    wait_for_timeout = timeout  # Use the timeout passed to acquire
                    if loop_iteration > 1 and timeout is not None:  # If we are looping due to spurious wakeups, adjust remaining timeout
                        elapsed_since_acquire_start = time.monotonic() - acquire_start_time
                        wait_for_timeout = max(0, timeout - elapsed_since_acquire_start)
                        if wait_for_timeout == 0:  # Timeout already expired
                            logger.warning("{}:: Acquire timed out while waiting for condition.", log_prefix)
                            if priority == ELP1: self._elp1_waiting_count -= 1
                            return False

                    logger.trace("{}:: Calling condition.wait(timeout={})...", log_prefix, wait_for_timeout)
                    signaled = self._condition.wait(timeout=wait_for_timeout)

                    if not signaled and wait_for_timeout is not None and wait_for_timeout > 0:  # Check if it was a real timeout
                        # Check if timeout was actually reached or if wait_for_timeout was already 0
                        # This check is a bit redundant if the above wait_for_timeout == 0 check is hit.
                        current_elapsed = time.monotonic() - acquire_start_time
                        if timeout is not None and current_elapsed >= timeout:
                            logger.warning("{}:: Acquire timed out after condition.wait(). Total elapsed: {:.2f}s",
                                           log_prefix, current_elapsed)
                            if priority == ELP1: self._elp1_waiting_count -= 1
                            return False

                    logger.trace("{}:: Woke up from condition.wait(). Signaled: {}. Re-evaluating...", log_prefix,
                                 signaled)

    def set_holder_process(self, proc: subprocess.Popen):
        """Stores the Popen object associated with the ELP0 lock holder."""
        with self._condition:
            log_prefix = f"PQLock|SetProc|Thr{threading.get_ident()}"
            if self._is_locked and self._lock_holder_priority == ELP0 and self._lock_holder_thread_ident == threading.get_ident():
                if self._lock_holder_proc is not None and self._lock_holder_proc is not proc:
                    logger.warning("{}:: Overwriting existing process for ELP0 holder.", log_prefix)
                self._lock_holder_proc = proc
                logger.trace("{}:: Associated process PID {} with ELP0 lock holder.", log_prefix,
                             proc.pid if proc else 'None')
            elif self._is_locked and self._lock_holder_priority == ELP1:
                logger.error("{}:: Attempted to set process for ELP1 holder. Ignoring.", log_prefix)
            elif not self._is_locked:
                logger.error("{}:: Attempted to set process when lock not held. Ignoring.", log_prefix)
            else:  # Lock held by different ELP0 thread?
                logger.error("{}:: Attempted to set process, but lock held by different thread ({}). Ignoring.",
                             log_prefix, self._lock_holder_thread_ident)

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
                logger.warning(
                    "{}:: Thread attempting release does not match current lock holder ({}). Possible interruption occurred.",
                    log_prefix, self._lock_holder_thread_ident)
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

    def get_status_extended(self) -> Tuple[bool, Optional[int], int, int]:
        """Returns (is_locked, holder_priority, elp1_quota, elp1_waiting_count)."""
        with self._condition:
            return self._is_locked, self._lock_holder_priority, self._elp1_interrupt_quota, self._elp1_waiting_count