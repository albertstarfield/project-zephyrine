# priority_lock.py (New File)

import threading
import time
import subprocess  # To store the process handle
from typing import Optional, Tuple
from loguru import logger
import platform
import os
import ctypes
import psutil

# --- Configuration Import ---
try:
    from CortexConfiguration import *
except ImportError:
    # Define fallbacks if config can't be imported (e.g., during testing)
    AGENTIC_RELAXATION_MODE = "Default"
    AGENTIC_RELAXATION_PRESETS = {"default": 0}
    AGENTIC_RELAXATION_PERIOD_SECONDS = 2.0

# Priority Levels
ELP0 = 0  # Background Tasks (File Indexer, Reflection)
ELP1 = 1  # Foreground User Requests


class SystemStateMonitor:
    @staticmethod
    def get_idle_duration() -> float:
        """Returns the number of seconds the user has been idle (no mouse/keyboard)."""
        system = platform.system()
        try:
            if system == 'Windows':
                class LASTINPUTINFO(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
                lii = LASTINPUTINFO()
                lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
                if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                    return millis / 1000.0
            elif system == 'Darwin': # macOS
                # Use ioreg to get HID idle time (nanoseconds -> seconds)
                cmd = "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF; exit}'"
                result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
                if result.stdout.strip():
                    return int(result.stdout.strip()) / 1_000_000_000
            elif system == 'Linux':
                # Try xprintidle (standard for X11)
                try:
                    result = subprocess.run(["xprintidle"], stdout=subprocess.PIPE, text=True)
                    return float(result.stdout.strip()) / 1000.0
                except FileNotFoundError:
                    # Fallback for headless/Wayland: Assume always active to prevent lockup
                    return 0.0
        except Exception:
            return 0.0 # Fail-safe: Assume active
        return 0.0

    @staticmethod
    def is_plugged_in() -> bool:
        """Returns True if plugged into AC power, False if on Battery."""
        try:
            battery = psutil.sensors_battery()
            # If no battery detected (Desktop), assume plugged in (True)
            return battery.power_plugged if battery else True
        except Exception:
            return True # Fail-safe

    @staticmethod
    def get_resource_load() -> Tuple[float, float]:
        """Returns (cpu_percent, ram_percent)."""
        return psutil.cpu_percent(interval=0.1), psutil.virtual_memory().percent

class AgenticRelaxationThread(threading.Thread):
    """
    A thread that implements PWM-style lock acquisition on ELP0 to throttle
    background tasks and manage thermals/power.
    Can operate in a fixed duty cycle mode or a dynamic, resource-aware mode.
    """

    def __init__(self, lock: 'PriorityQuotaLock', duty_cycle_off: float, period_sec: float,
                 stop_event: threading.Event, dynamic_mode_id: int = 0):
        super().__init__(name="AgenticRelaxationThread", daemon=True)
        self.lock = lock
        self.initial_duty_cycle_off = duty_cycle_off
        self.period_sec = period_sec
        self.stop_event = stop_event
        self.dynamic_mode_id = dynamic_mode_id
        self.duty_cycle_off = self.initial_duty_cycle_off
        logger.info(f"AgenticRelaxationThread initialized. Dynamic Mode: {self.dynamic_mode_id}")

    def _calculate_dynamic_duty_cycle(self) -> float:
        """
        Calculates PWM Duty Cycle (0.0 to 1.0) based on the selected Dynamic Mode.

        Returns:
            1.0 = FULL STOP (Hard Block / Kill Background Tasks).
            0.0 = OPEN (Background Tasks Allowed).

        Fallback:
            If specific sensors fail (e.g., Idle check on Linux), falls back to
            Mode -1 (Basic CPU/RAM safety check).
        """

        # --- Internal Helper for Mode -1 (Safe Harbor) ---
        def _run_mode_minus_one_logic(source_error=None):
            if source_error:
                logger.error(f"Dynamic Mode {self.dynamic_mode_id} failed: {source_error}. Fallback to Mode -1.")

            try:
                # Basic Safety Check: CPU > 90% or RAM > 90% -> Kill ELP0
                cpu, ram = SystemStateMonitor.get_resource_load()
                if cpu > 90.0 or ram > 90.0:
                    logger.warning(f"Dynamic(-1)[Fallback]: Critical Resources (CPU:{cpu}%, RAM:{ram}%). BLOCKING.")
                    return 1.0
                return 0.0
            except Exception as e_fallback:
                logger.error(f"Critical: Fallback Resource Monitor failed: {e_fallback}. Defaulting to safe halt.")
                return 1.0  # Fail-secure: Block background tasks if we can't measure anything.

        try:
            # --- Mode -1: Reservative Shared Resources ---
            if self.dynamic_mode_id == -1:
                return _run_mode_minus_one_logic()

            # --- Gather Sensors for Advanced Modes ---
            # We gather these here inside the try block.
            # If get_idle_duration crashes, we catch it and go to fallback.

            # --- Mode -2: Power Source Based ---
            elif self.dynamic_mode_id == -2:
                if not SystemStateMonitor.is_plugged_in():
                    logger.warning("Dynamic(-2): On Battery. BLOCKING ELP0.")
                    return 1.0
                return 0.0

            # --- Mode -3: Interactivity Prioritization ---
            elif self.dynamic_mode_id == -3:
                idle_sec = SystemStateMonitor.get_idle_duration()
                if idle_sec < 1800:  # 30 mins
                    # User is active -> Block
                    return 1.0
                logger.info(f"Dynamic(-3): System idle ({idle_sec:.0f}s). Releasing ELP0.")
                return 0.0

            # --- Mode -4: Interactivity + Power ---
            elif self.dynamic_mode_id == -4:
                idle_sec = SystemStateMonitor.get_idle_duration()
                plugged_in = SystemStateMonitor.is_plugged_in()

                # Must be Plugged In AND Idle > 30m
                if plugged_in and idle_sec >= 1800:
                    return 0.0

                return 1.0

            # --- Mode -5: Interactivity + Power + Resources (Goldilocks) ---
            elif self.dynamic_mode_id == -5:
                idle_sec = SystemStateMonitor.get_idle_duration()
                plugged_in = SystemStateMonitor.is_plugged_in()
                cpu, ram = SystemStateMonitor.get_resource_load()

                # Strict constraints: Idle > 30m, Plugged In, CPU < 70%, RAM < 80%
                is_idle_enough = idle_sec >= 1800
                is_cpu_safe = cpu < 70.0
                is_ram_safe = ram < 80.0

                if (is_idle_enough and plugged_in and is_cpu_safe and is_ram_safe):
                    logger.info(
                        f"Dynamic(-5): RELEASE | Idle:{idle_sec:.0f}s, AC:{plugged_in}, CPU:{cpu:.1f}%, RAM:{ram:.1f}%")
                    return 0.0
                else:
                    reasons = []
                    if not is_idle_enough: reasons.append(f"Active({idle_sec:.0f}s)")
                    if not plugged_in: reasons.append("Battery")
                    if not is_cpu_safe: reasons.append(f"CPU({cpu:.1f}%)")
                    if not is_ram_safe: reasons.append(f"RAM({ram:.1f}%)")

                    logger.info(f"Dynamic(-5): HARD BLOCK | Blockers: {', '.join(reasons)}")
                    return 1.0

            # Default for unknown negative modes
            return _run_mode_minus_one_logic("Unknown Dynamic Mode ID")

        except Exception as e:
            # Catch-all: If any complex monitor fails, fallback to simple CPU/RAM check
            return _run_mode_minus_one_logic(source_error=e)

    def run(self):
        self.lock.set_relaxation_thread_ident(threading.get_ident())
        logger.info(f"✅ AgenticRelaxationThread started (Dynamic: {self.dynamic_mode_id}).")
        while not self.stop_event.is_set():
            try:
                # 1. Update Duty Cycle
                if self.dynamic_mode_id:
                    self.duty_cycle_off = self._calculate_dynamic_duty_cycle()
                
                # 2. Determine Strategy based on Duty Cycle
                # If >= 0.99, we are in HARD BLOCK mode.
                is_hard_block = self.duty_cycle_off >= 0.99
                
                if is_hard_block:
                    # Strategy: HARD BLOCK
                    # We acquire with ELP1 priority. This tells the Lock to KILL any running ELP0 task immediately.
                    acquire_priority = ELP1 
                    hold_time = self.period_sec # Hold the door shut for the full cycle
                    acquire_timeout = 0.5 # Try briefly to get the lock
                elif self.duty_cycle_off > 0:
                    # Strategy: PWM THROTTLING
                    # We utilize ELP0 to just eat up time slots without killing active tasks
                    acquire_priority = ELP0
                    hold_time = self.period_sec * self.duty_cycle_off
                    acquire_timeout = self.period_sec * (1.0 - self.duty_cycle_off)
                else:
                    # Strategy: FREE RUN (0% off)
                    self.stop_event.wait(self.period_sec)
                    continue

                # 3. Execute Lock Acquisition
                if hold_time > 0:
                    # Try to get the lock
                    was_acquired = self.lock.acquire(priority=acquire_priority, timeout=acquire_timeout)
                    
                    if was_acquired:
                        try:
                            # We have the lock. 
                            # If we are ELP1, we effectively killed any background task and are now blocking the slot.
                            # If we are ELP0, we are just occupying a free slot.
                            logger.trace(f"Relaxation (Prio {acquire_priority}) holding lock for {hold_time:.2f}s (Block Mode: {is_hard_block})")
                            self.stop_event.wait(hold_time)
                        finally:
                            self.lock.release()
                    else:
                        # Could not get lock. 
                        # If Hard Block mode: A real User Request (ELP1) probably holds it. Good.
                        # If PWM mode: A background task (ELP0) holds it.
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
    QUOTA_MAX = 4294967296 #ELP1 Quota preemption (just set to 99999 since there would be idle and that idle would not decrease or reload the quota which cause an softlock thus it's better to set it 2^32)

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
        self._relaxation_thread_ident: Optional[int] = None
        self._initialize_relaxation()

        logger.info("🚦 PriorityQuotaLock initialized. ELP1 Interrupt Quota: {}", self.QUOTA_MAX)

    def _initialize_relaxation(self):
        # Normalize the input string
        mode = str(AGENTIC_RELAXATION_MODE).lower().strip()
        # Normalize the preset keys to match
        presets = {k.lower(): v for k, v in AGENTIC_RELAXATION_PRESETS.items()}
        
        is_dynamic = False
        mode_val = 0 # This will hold either the % (e.g. 50) or the ID (e.g. -5)
        
        if mode in presets:
            preset_value = presets[mode]
            # FIX: Check if less than 0, not just equal to -1
            if preset_value < 0: 
                is_dynamic = True
                mode_val = preset_value # e.g., -5
                logger.info(f"Activating AgenticRelaxation in Dynamic Mode: {mode} ({mode_val})")
            else:
                mode_val = preset_value # e.g., 50
        else:
            # Handle manual numeric input (e.g., "50")
            try:
                val = float(mode)
                if val < 0:
                    logger.warning(f"Negative custom value '{val}' not supported unless defined in presets. Defaulting to 0.")
                    mode_val = 0
                else:
                    mode_val = val
            except ValueError:
                logger.warning(f"Invalid AGENTIC_RELAXATION_MODE '{AGENTIC_RELAXATION_MODE}'. Defaulting to 0%.")
                mode_val = 0

        # Calculate duty cycle (0.0 to 1.0)
        # If dynamic, we start at 0.0 (allow everything) until the thread calculates load
        if is_dynamic:
            duty_cycle_float = 0.0
        else:
            # Clamp static percentages between 0 and 100
            clamped_val = max(0, min(100, mode_val))
            duty_cycle_float = clamped_val / 100.0

        # Start thread if we have a duty cycle OR if it's dynamic
        if duty_cycle_float > 0 or is_dynamic:
            if not is_dynamic:
                logger.info(f"Activating AgenticRelaxation with fixed {mode_val}% off-cycle.")

            self._relaxation_thread = AgenticRelaxationThread(
                lock=self,
                duty_cycle_off=duty_cycle_float,
                period_sec=AGENTIC_RELAXATION_PERIOD_SECONDS,
                stop_event=self._relaxation_stop_event,
                dynamic_mode_id=int(mode_val) if is_dynamic else 0 # Pass the ID (-5) here
            )
            self._relaxation_thread.start()
        else:
            logger.info(f"AgenticRelaxation is disabled (Mode: {mode_val}).")

    def shutdown_relaxation_thread(self):
        if self._relaxation_thread and self._relaxation_thread.is_alive():
            logger.info("Signaling AgenticRelaxationThread to stop...")
            self._relaxation_stop_event.set()
            self._relaxation_thread.join(timeout=AGENTIC_RELAXATION_PERIOD_SECONDS + 1)
            if self._relaxation_thread.is_alive():
                logger.warning("AgenticRelaxationThread did not stop in time.")

    def set_relaxation_thread_ident(self, ident: int):
        self._relaxation_thread_ident = ident

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

            try:
                while True:
                    # -----------------------------------------------------------
                    # 1. PRIORITY STEALING LOGIC (User ELP1 > Relaxation ELP1)
                    # -----------------------------------------------------------
                    # Check if the current holder is specifically the AgenticRelaxationThread
                    holder_is_relaxation = (self._is_locked and
                                            self._relaxation_thread_ident is not None and
                                            self._lock_holder_thread_ident == self._relaxation_thread_ident)

                    # If WE are ELP1 (User) and THEY are Relaxation (Daemon), we steal immediately.
                    if priority == ELP1 and holder_is_relaxation:
                        logger.info(f"{log_prefix}:: STEALING lock from Relaxation Thread to process User Request.")

                        # Overwrite ownership immediately without waiting.
                        self._lock_holder_priority = ELP1
                        self._lock_holder_proc = None  # Relaxation thread has no process to kill
                        self._lock_holder_thread_ident = requesting_thread_ident

                        # Note: We do not decrement quota because we are just swapping "Admin" users.
                        # The Relaxation Thread will fail silently when it tries to release later.
                        return True

                    # -----------------------------------------------------------
                    # 2. STANDARD INTERRUPT LOGIC (ELP1 > ELP0)
                    # -----------------------------------------------------------
                    can_interrupt_elp0 = (priority == ELP1 and
                                          self._is_locked and
                                          self._lock_holder_priority == ELP0 and
                                          self._elp1_interrupt_quota > 0)

                    if not self._is_locked:
                        # STANDARD ACQUIRE: Lock is Free
                        self._is_locked = True
                        self._lock_holder_priority = priority
                        self._lock_holder_proc = None
                        self._lock_holder_thread_ident = requesting_thread_ident

                        if priority == ELP0:
                            self._elp1_interrupt_quota = self.QUOTA_MAX
                            logger.info(
                                f"{log_prefix}:: Acquired lock (was Free). Reset ELP1 quota to {self.QUOTA_MAX}")
                        else:
                            logger.info(f"{log_prefix}:: Acquired lock (was Free). Quota: {self._elp1_interrupt_quota}")
                        return True

                    elif can_interrupt_elp0:
                        # INTERRUPT PATH: Kill the ELP0 worker
                        logger.warning(
                            f"{log_prefix}:: INTERRUPT PATH: Holder ELP{self._lock_holder_priority} (PID: {self._lock_holder_proc.pid if self._lock_holder_proc else 'N/A'}). Quota: {self._elp1_interrupt_quota}->{self._elp1_interrupt_quota - 1}")

                        interrupted_proc = self._lock_holder_proc

                        if interrupted_proc and interrupted_proc.poll() is None:
                            pid_to_kill = interrupted_proc.pid
                            logger.warning(f"{log_prefix}:: Forcefully terminating ELP0 process PID: {pid_to_kill}")
                            try:
                                if platform.system() == "Windows":
                                    subprocess.run(["taskkill", "/PID", str(pid_to_kill), "/F", "/T"], check=False)
                                else:
                                    import signal
                                    os.kill(pid_to_kill, signal.SIGKILL)
                                # Brief wait to ensure signal sends
                                interrupted_proc.wait(timeout=0.01)
                            except Exception as e:
                                logger.error(f"{log_prefix}:: Error killing PID {pid_to_kill}: {e}")
                        else:
                            logger.warning(
                                f"{log_prefix}:: ELP0 holder process not found or already exited. Taking over.")

                        # Take ownership
                        self._lock_holder_priority = ELP1
                        self._lock_holder_proc = None
                        self._lock_holder_thread_ident = requesting_thread_ident
                        self._elp1_interrupt_quota -= 1

                        logger.info(f"{log_prefix}:: INTERRUPTED ELP0. Acquired. Quota: {self._elp1_interrupt_quota}.")
                        self._condition.notify_all()
                        return True

                    else:
                        # -----------------------------------------------------------
                        # 3. WAIT PATH
                        # -----------------------------------------------------------

                        # Calculate remaining timeout
                        remaining_timeout = None
                        if timeout is not None:
                            elapsed = time.monotonic() - acquire_start_time
                            remaining_timeout = timeout - elapsed
                            if remaining_timeout <= 0:
                                logger.warning(f"{log_prefix}:: Acquire timed out before waiting.")
                                return False

                        # ELP0 Politeness: Yield briefly if ELP1 is waiting
                        if priority == ELP0 and self._elp1_waiting_count > 0:
                            logger.trace(f"{log_prefix}:: ELP0 waiting politely (ELP1 active/waiting).")
                            self._condition.wait(timeout=0.05)
                        else:
                            logger.trace(f"{log_prefix}:: Waiting... (timeout={remaining_timeout})")
                            self._condition.wait(timeout=remaining_timeout)

                        # Re-check timeout after wake
                        if timeout is not None and (time.monotonic() - acquire_start_time) >= timeout:
                            logger.warning(f"{log_prefix}:: Acquire timed out after waiting.")
                            return False

            finally:
                if priority == ELP1:
                    self._elp1_waiting_count -= 1

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
