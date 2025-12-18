# priority_lock.py (New File)

import threading
import time
import subprocess  # To store the process handle
from typing import Optional, Tuple
from loguru import logger
import platform
import os
import psutil

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
    Can operate in a fixed duty cycle mode or a dynamic, resource-aware mode.
    """

    def __init__(self, lock: 'PriorityQuotaLock', duty_cycle_off: float, period_sec: float,
                 stop_event: threading.Event, is_dynamic_mode: bool = False):
        super().__init__(name="AgenticRelaxationThread", daemon=True)
        self.lock = lock
        self.initial_duty_cycle_off = duty_cycle_off
        self.period_sec = period_sec
        self.stop_event = stop_event
        self.is_dynamic_mode = is_dynamic_mode
        self.duty_cycle_off = self.initial_duty_cycle_off
        logger.info(f"AgenticRelaxationThread initialized. Dynamic Mode: {self.is_dynamic_mode}")

    def _calculate_dynamic_duty_cycle(self) -> float:
        """Calculates duty cycle based on system load and ELP1 contention."""
        try:
            # Check for high-priority task contention
            _, _, _, elp1_waiting_count = self.lock.get_status_extended()

            # Check for overall system CPU load
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Logic: If any ELP1 is waiting OR system CPU is high, be aggressive with power saving.
            if elp1_waiting_count > 0 or cpu_percent > 80.0:
                logger.warning(f"Dynamic Relaxation: High load detected (ELP1 Waiters: {elp1_waiting_count}, CPU: {cpu_percent:.1f}%). Throttling ELP0 tasks.")
                return 0.98  # 98% off-time (ExtremePowerSaving)
            else:
                # When there's no pressure, don't relax at all.
                logger.trace(f"Dynamic Relaxation: Low load (ELP1 Waiters: {elp1_waiting_count}, CPU: {cpu_percent:.1f}%). Allowing ELP0 tasks.")
                return 0.0  # 0% off-time

        except Exception as e:
            logger.error(f"Error in _calculate_dynamic_duty_cycle: {e}. Defaulting to initial duty cycle.")
            return self.initial_duty_cycle_off

    def run(self):
        logger.info(f"✅ AgenticRelaxationThread started (Dynamic: {self.is_dynamic_mode}).")
        while not self.stop_event.is_set():
            try:
                if self.is_dynamic_mode:
                    self.duty_cycle_off = self._calculate_dynamic_duty_cycle()
                
                off_time = self.period_sec * self.duty_cycle_off
                on_time = self.period_sec * (1.0 - self.duty_cycle_off)

                if off_time > 0:
                    was_acquired = self.lock.acquire(priority=ELP0, timeout=on_time)
                    if was_acquired:
                        try:
                            logger.trace(f"Relaxation thread acquired ELP0 lock. Holding for {off_time:.2f}s...")
                            self.stop_event.wait(off_time)
                        finally:
                            logger.trace("Relaxation thread releasing ELP0 lock.")
                            self.lock.release()
                    else:
                        logger.trace("Relaxation thread could not acquire lock, another ELP0 task is active. Waiting for next cycle.")
                        self.stop_event.wait(self.period_sec)
                else:
                    # If off_time is 0, just wait for the full period before re-evaluating.
                    self.stop_event.wait(self.period_sec)

            except Exception as e:
                logger.error(f"Error in AgenticRelaxationThread loop: {e}")
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
        is_dynamic = False
        
        duty_cycle_off_percent = 0
        if mode in presets:
            preset_value = presets[mode]
            if preset_value == -1: # Our special value for dynamic mode
                is_dynamic = True
                duty_cycle_off_percent = 0 # Start with 0, thread will adjust it
                logger.info("Activating AgenticRelaxation in dynamic 'reservativesharedresources' mode.")
            else:
                duty_cycle_off_percent = preset_value
        else:
            try:
                duty_cycle_off_percent = float(mode)
            except ValueError:
                logger.warning(f"Invalid AGENTIC_RELAXATION_MODE '{AGENTIC_RELAXATION_MODE}'. Defaulting to 0%.")

        duty_cycle_off_percent = max(0, min(100, duty_cycle_off_percent))

        if duty_cycle_off_percent > 0 or is_dynamic:
            duty_cycle_float = duty_cycle_off_percent / 100.0
            if not is_dynamic:
                 logger.info(f"Activating AgenticRelaxation with fixed {duty_cycle_off_percent}% off-cycle.")

            self._relaxation_thread = AgenticRelaxationThread(
                lock=self,
                duty_cycle_off=duty_cycle_float,
                period_sec=AGENTIC_RELAXATION_PERIOD_SECONDS,
                stop_event=self._relaxation_stop_event,
                is_dynamic_mode=is_dynamic
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

    def is_preempted(self, priority: int) -> bool:
        """
        Checks if the current thread has lost the lock to a higher priority task.
        Returns True if the lock is no longer held by this thread/priority.
        """
        current_thread = threading.get_ident()
        with self._condition:
            # If lock is free, we definitely don't hold it -> Preempted/Lost
            if not self._is_locked:
                return True
            
            # If priority doesn't match current holder -> Preempted
            if self._lock_holder_priority != priority:
                return True
            
            # If thread ID doesn't match current holder -> Preempted
            if self._lock_holder_thread_ident != current_thread:
                return True
                
            return False

    def acquire(self, priority: int, timeout: Optional[float] = None) -> bool:
        acquire_start_time = time.monotonic()
        requesting_thread_ident = threading.get_ident()
        log_prefix = f"PQLock|ACQ|ELP{priority}|Thr{requesting_thread_ident}"
        
        if priority not in [ELP0, ELP1]:
            raise ValueError("Invalid priority level")

        with self._condition:
            if priority == ELP1:
                self._elp1_waiting_count += 1

            while True:
                # --- CORE LOGIC: CHECK IF WE CAN ACQUIRE NOW ---
                can_interrupt = (priority == ELP1 and
                                 self._is_locked and
                                 self._lock_holder_priority == ELP0 and
                                 self._elp1_interrupt_quota > 0)

                if not self._is_locked:
                    # LOCK IS FREE: Acquire it.
                    self._is_locked = True
                    self._lock_holder_priority = priority
                    self._lock_holder_proc = None
                    self._lock_holder_thread_ident = requesting_thread_ident
                    if priority == ELP0:
                        self._elp1_interrupt_quota = self.QUOTA_MAX
                        logger.info(f"{log_prefix}:: Acquired lock (was Free). Reset ELP1 quota to {self.QUOTA_MAX}")
                    else: # ELP1 acquired
                        self._elp1_waiting_count -= 1
                        logger.info(f"{log_prefix}:: Acquired lock (was Free). Quota: {self._elp1_interrupt_quota}")
                    return True

                elif can_interrupt:
                    # INTERRUPT PATH: ELP1 takes over from ELP0.
                    # (This logic is complex but correct, no changes needed here)
                    logger.warning(f"{log_prefix}:: INTERRUPT PATH: Holder ELP{self._lock_holder_priority} (PID: {self._lock_holder_proc.pid if self._lock_holder_proc else 'N/A'}). Quota: {self._elp1_interrupt_quota}->{self._elp1_interrupt_quota - 1}")
                    
                    interrupted_proc = self._lock_holder_proc
                    original_holder_thread_ident = self._lock_holder_thread_ident
                    
                    if interrupted_proc and interrupted_proc.poll() is None:
                        # ... (existing process kill logic) ...
                        pid_to_kill = interrupted_proc.pid
                        logger.warning(f"{log_prefix}:: Forcefully terminating ELP0 process PID: {pid_to_kill}")
                        try:
                            if platform.system() == "Windows":
                                subprocess.run(["taskkill", "/PID", str(pid_to_kill), "/F", "/T"], check=False)
                            else:
                                import signal
                                os.kill(pid_to_kill, signal.SIGKILL)
                            interrupted_proc.wait(timeout=0.01)
                        except Exception as e:
                            logger.error(f"{log_prefix}:: Error killing PID {pid_to_kill}: {e}")
                    else:
                        logger.warning(f"{log_prefix}:: ELP0 holder process not found or already exited. Taking over.")

                    self._lock_holder_priority = ELP1
                    self._lock_holder_proc = None
                    self._lock_holder_thread_ident = requesting_thread_ident
                    self._elp1_interrupt_quota -= 1
                    self._elp1_waiting_count -= 1
                    
                    logger.info(f"{log_prefix}:: INTERRUPTED ELP0 (was Thr: {original_holder_thread_ident}). Acquired. Quota: {self._elp1_interrupt_quota}.")
                    self._condition.notify_all()
                    return True

                else:
                    # --- WAIT PATH (This is where the fix goes) ---
                    
                    # Calculate remaining timeout
                    if timeout is not None:
                        elapsed = time.monotonic() - acquire_start_time
                        remaining_timeout = timeout - elapsed
                        if remaining_timeout <= 0:
                            if priority == ELP1: self._elp1_waiting_count -= 1
                            logger.warning(f"{log_prefix}:: Acquire timed out before waiting.")
                            return False
                    else:
                        remaining_timeout = None

                    # --- THE FIX ---
                    # If this is an ELP0 task AND an ELP1 task is waiting,
                    # we do a short, timed wait instead of an indefinite one.
                    # This gives the ELP1 task a window to acquire the lock when it's released.
                    if priority == ELP0 and self._elp1_waiting_count > 0:
                        logger.trace(f"{log_prefix}:: ELP0 Politeness: ELP1 is waiting. Using short timed wait (0.05s).")
                        # Use a very short timeout to "yield"
                        wait_duration = 0.05 
                        if remaining_timeout is not None:
                            wait_duration = min(wait_duration, remaining_timeout)
                        
                        self._condition.wait(timeout=wait_duration)
                        # After waking up, the loop will restart and re-evaluate.
                        
                    else:
                        # Original behavior: ELP1 waits, or ELP0 waits when no ELP1 is present.
                        logger.trace(f"{log_prefix}:: Entering standard wait (timeout={remaining_timeout})...")
                        signaled = self._condition.wait(timeout=remaining_timeout)
                        
                        # Check for timeout after waking
                        if timeout is not None and not signaled:
                            # Re-check elapsed time to be sure it was a timeout
                            if (time.monotonic() - acquire_start_time) >= timeout:
                                if priority == ELP1: self._elp1_waiting_count -= 1
                                logger.warning(f"{log_prefix}:: Acquire timed out after waiting.")
                                return False

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
