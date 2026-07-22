# priority_lock.py (New File)

import ctypes
import os
import platform
import subprocess  # To store the process handle
import threading
import time
from typing import Optional, Tuple, Dict

import psutil
from loguru import logger

# --- Configuration Import ---
try:
    from CortexConfiguration import *
except ImportError:
    # Define fallbacks if config can't be imported (e.g., during testing)
    AGENTIC_RELAXATION_MODE = "Default"
    AGENTIC_RELAXATION_PRESETS = {"default": 0}
    AGENTIC_RELAXATION_PERIOD_SECONDS = 2.0
    MAX_CONCURRENT_ELP1_TASKS = 1
    MAX_CONCURRENT_ELP0_TASKS = 1

# Priority Levels
ELP0 = 0  # Background Tasks (File Indexer, Reflection)
ELP1 = 1  # Foreground User Requests

# --- NEW: ATOMIC PRIORITY FLAG ---
# This is used for immediate "kill" signaling across threads/processes.
ELP1_ACTIVE_FLAG = False


class SystemStateMonitor:
    @staticmethod
    def get_idle_duration() -> float:
        """Returns the number of seconds the user has been idle (no mouse/keyboard)."""
        system = platform.system()
        try:
            if system == "Windows":

                class LASTINPUTINFO(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

                lii = LASTINPUTINFO()
                lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
                if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
                    millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
                    return millis / 1000.0
            elif system == "Darwin":  # macOS
                # Use ioreg to get HID idle time (nanoseconds -> seconds)
                cmd = "ioreg -c IOHIDSystem | awk '/HIDIdleTime/ {print $NF; exit}'"
                result = subprocess.run(
                    cmd, shell=True, stdout=subprocess.PIPE, text=True
                )
                if result.stdout.strip():
                    return int(result.stdout.strip()) / 1_000_000_000
            elif system == "Linux":
                # Try xprintidle (standard for X11)
                try:
                    result = subprocess.run(
                        ["xprintidle"], stdout=subprocess.PIPE, text=True
                    )
                    return float(result.stdout.strip()) / 1000.0
                except FileNotFoundError:
                    # Fallback for headless/Wayland: Assume always active to prevent lockup
                    return 0.0
        except Exception:
            return 0.0  # Fail-safe: Assume active
        return 0.0

    @staticmethod
    def is_plugged_in() -> bool:
        """Returns True if plugged into AC power, False if on Battery."""
        try:
            battery = psutil.sensors_battery()
            # If no battery detected (Desktop), assume plugged in (True)
            return battery.power_plugged if battery else True
        except Exception:
            return True  # Fail-safe

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

    def __init__(
        self,
        lock: "PriorityQuotaLock",
        duty_cycle_off: float,
        period_sec: float,
        stop_event: threading.Event,
        dynamic_mode_id: int = 0,
    ):
        super().__init__(name="AgenticRelaxationThread", daemon=True)
        self.lock = lock
        self.initial_duty_cycle_off = duty_cycle_off
        self.period_sec = period_sec
        self.stop_event = stop_event
        self.dynamic_mode_id = dynamic_mode_id
        self.duty_cycle_off = self.initial_duty_cycle_off
        self.cycle_count = 0
        logger.info(
            f"AgenticRelaxationThread initialized. Dynamic Mode: {self.dynamic_mode_id}"
        )

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
                logger.error(
                    f"Dynamic Mode {self.dynamic_mode_id} failed: {source_error}. Fallback to Mode -1."
                )

            try:
                # Basic Safety Check: CPU > 90% or RAM > 90% -> Kill ELP0
                cpu, ram = SystemStateMonitor.get_resource_load()
                if cpu > 90.0 or ram > 90.0:
                    logger.warning(
                        f"Dynamic(-1)[Fallback]: Critical Resources (CPU:{cpu}%, RAM:{ram}%). BLOCKING."
                    )
                    return 1.0
                return 0.0
            except Exception as e_fallback:
                logger.error(
                    f"Critical: Fallback Resource Monitor failed: {e_fallback}. Defaulting to safe halt."
                )
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
                logger.info(
                    f"Dynamic(-3): System idle ({idle_sec:.0f}s). Releasing ELP0."
                )
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

                if is_idle_enough and plugged_in and is_cpu_safe and is_ram_safe:
                    logger.info(
                        f"Dynamic(-5): RELEASE | Idle:{idle_sec:.0f}s, AC:{plugged_in}, CPU:{cpu:.1f}%, RAM:{ram:.1f}%"
                    )
                    return 0.0
                else:
                    reasons = []
                    if not is_idle_enough:
                        reasons.append(f"Active({idle_sec:.0f}s)")
                    if not plugged_in:
                        reasons.append("Battery")
                    if not is_cpu_safe:
                        reasons.append(f"CPU({cpu:.1f}%)")
                    if not is_ram_safe:
                        reasons.append(f"RAM({ram:.1f}%)")

                    if self.cycle_count % 100 == 0:
                        logger.info(
                            f"Dynamic(-5): HARD BLOCK | Blockers: {', '.join(reasons)}"
                        )
                    return 1.0

            # Default for unknown negative modes
            return _run_mode_minus_one_logic("Unknown Dynamic Mode ID")

        except Exception as e:
            # Catch-all: If any complex monitor fails, fallback to simple CPU/RAM check
            return _run_mode_minus_one_logic(source_error=e)

    def run(self):
        self.lock.set_relaxation_thread_ident(threading.get_ident())
        logger.info(
            f"✅ AgenticRelaxationThread started (Dynamic: {self.dynamic_mode_id})."
        )

        # NEW: State tracker to maintain an iron grip on the lock across loops
        currently_holding_hard_block = False

        while not self.stop_event.is_set():
            try:
                self.cycle_count += 1
                # 1. Update Duty Cycle
                if self.dynamic_mode_id:
                    self.duty_cycle_off = self._calculate_dynamic_duty_cycle()

                # 2. Determine Strategy based on Duty Cycle
                # If >= 0.99, we are in HARD BLOCK mode.
                is_hard_block = self.duty_cycle_off >= 0.99

                # Transition OUT of hard block: Release the lock if we held it continuously
                if not is_hard_block and currently_holding_hard_block:
                    self.lock.release()
                    currently_holding_hard_block = False

                if is_hard_block:
                    # Strategy: HARD BLOCK - HOLD FIRMLY UNTIL RELEASED
                    if not currently_holding_hard_block:
                        # We acquire with ELP1 priority. This tells the Lock to KILL any running ELP0 task immediately.
                        was_acquired = self.lock.acquire(priority=ELP1, timeout=0.5)
                        if was_acquired:
                            currently_holding_hard_block = True
                    else:
                        # Check if a real User ELP1 task stole the lock from us
                        if self.lock.is_preempted(priority=ELP1):
                            currently_holding_hard_block = False
                            # It was stolen. Don't sleep! Immediately loop to get back in line
                            # so we catch it the millisecond the user drops it.
                            continue

                    # Sleep in small increments to quickly detect if we lost the lock or sensors changed
                    self.stop_event.wait(0.5)

                elif self.duty_cycle_off > 0:
                    # Strategy: PWM THROTTLING
                    # We utilize ELP0 to just eat up time slots without killing active tasks
                    acquire_priority = ELP0
                    hold_time = self.period_sec * self.duty_cycle_off
                    acquire_timeout = self.period_sec * (1.0 - self.duty_cycle_off)

                    was_acquired = self.lock.acquire(
                        priority=acquire_priority, timeout=acquire_timeout
                    )
                    if was_acquired:
                        try:
                            # We have the lock. Occupy the free slot.
                            logger.trace(
                                f"Relaxation (Prio {acquire_priority}) holding lock for {hold_time:.2f}s (Block Mode: False)"
                            )
                            self.stop_event.wait(hold_time)
                        finally:
                            self.lock.release()
                    else:
                        # Could not get lock. A background task (ELP0) holds it.
                        self.stop_event.wait(self.period_sec)
                else:
                    # Strategy: FREE RUN (0% off)
                    self.stop_event.wait(self.period_sec)
                    continue

            except Exception as e:
                logger.error(f"Error in AgenticRelaxationThread loop: {e}")
                self.stop_event.wait(5)

        logger.info("🛑 AgenticRelaxationThread has been shut down.")

        # Ensure we release the lock if the server is shutting down while we hold it
        if currently_holding_hard_block:
            self.lock.release()


class PriorityQuotaLock:
    """
    A multi-slot priority lock (semaphore-like) supporting ELP0 and ELP1 levels.
    ELP1 tasks take precedence and can preempt (kill) running ELP0 tasks if slots are needed.
    """

    def __init__(self):
        self._condition = threading.Condition(threading.Lock())
        
        # Configuration
        self._max_elp1_slots = MAX_CONCURRENT_ELP1_TASKS
        self._max_elp0_slots = MAX_CONCURRENT_ELP0_TASKS
        # Total allowed concurrent binaries
        self._total_slots = max(self._max_elp1_slots, self._max_elp0_slots)
        
        # State tracking
        self._active_tasks: Dict[int, Dict] = {} # ident -> {priority, proc, start_time}
        self._elp1_waiting_count = 0
        self._elp0_waiting_count = 0
        
        # Interruption quota (legacy support)
        self._elp1_interrupt_quota = 18446744073709551616

        # --- Relaxation Thread ---
        self._relaxation_thread: Optional[AgenticRelaxationThread] = None
        self._relaxation_stop_event = threading.Event()
        self._relaxation_thread_ident: Optional[int] = None
        self._initialize_relaxation()

        logger.info(
            "🚦 PriorityQuotaLock (Queued) initialized. Slots: ELP1={}, ELP0={}, Total={}", 
            self._max_elp1_slots, self._max_elp0_slots, self._total_slots
        )

    def _initialize_relaxation(self):
        # Normalize the input string
        mode = str(AGENTIC_RELAXATION_MODE).lower().strip()
        # Normalize the preset keys to match
        presets = {k.lower(): v for k, v in AGENTIC_RELAXATION_PRESETS.items()}

        is_dynamic = False
        mode_val = 0

        if mode in presets:
            preset_value = presets[mode]
            if preset_value < 0:
                is_dynamic = True
                mode_val = preset_value
                logger.info(
                    f"Activating AgenticRelaxation in Dynamic Mode: {mode} ({mode_val})"
                )
            else:
                mode_val = preset_value
        else:
            try:
                val = float(mode)
                if val < 0:
                    mode_val = 0
                else:
                    mode_val = val
            except ValueError:
                mode_val = 0

        if is_dynamic:
            duty_cycle_float = 0.0
        else:
            clamped_val = max(0, min(100, mode_val))
            duty_cycle_float = clamped_val / 100.0

        if duty_cycle_float > 0 or is_dynamic:
            self._relaxation_thread = AgenticRelaxationThread(
                lock=self,
                duty_cycle_off=duty_cycle_float,
                period_sec=AGENTIC_RELAXATION_PERIOD_SECONDS,
                stop_event=self._relaxation_stop_event,
                dynamic_mode_id=int(mode_val) if is_dynamic else 0,
            )
            self._relaxation_thread.start()

    def shutdown_relaxation_thread(self):
        if self._relaxation_thread and self._relaxation_thread.is_alive():
            logger.info("Signaling AgenticRelaxationThread to stop...")
            self._relaxation_stop_event.set()
            self._relaxation_thread.join(timeout=AGENTIC_RELAXATION_PERIOD_SECONDS + 1)

    def set_relaxation_thread_ident(self, ident: int):
        self._relaxation_thread_ident = ident

    def is_preempted(self, priority: int) -> bool:
        """Checks if the current thread's task has been killed/removed from active tasks."""
        current_thread = threading.get_ident()
        with self._condition:
            return current_thread not in self._active_tasks

    def acquire(self, priority: int, timeout: Optional[float] = None) -> bool:
        acquire_start_time = time.monotonic()
        requesting_thread_ident = threading.get_ident()
        log_prefix = f"PQLock|ACQ|ELP{priority}|Thr{requesting_thread_ident}"

        if priority not in [ELP0, ELP1]:
            raise ValueError("Invalid priority level")

        with self._condition:
            if priority == ELP1:
                self._elp1_waiting_count += 1
            else:
                self._elp0_waiting_count += 1

            try:
                while True:
                    # Slot Availability Logic
                    current_elp1_count = sum(1 for t in self._active_tasks.values() if t['priority'] == ELP1)
                    current_elp0_count = sum(1 for t in self._active_tasks.values() if t['priority'] == ELP0)
                    total_active = len(self._active_tasks)

                    if priority == ELP1:
                        if current_elp1_count < self._max_elp1_slots:
                            if total_active >= self._total_slots:
                                elp0_candidates = [tid for tid, t in self._active_tasks.items() if t['priority'] == ELP0]
                                if elp0_candidates:
                                    victim_tid = elp0_candidates[0]
                                    victim = self._active_tasks.pop(victim_tid)
                                    logger.warning(f"{log_prefix}:: PREEMPTING ELP0 task on thread {victim_tid} to free slot for ELP1.")
                                    if victim['proc']:
                                        self._kill_process_tree(victim['proc'].pid)
                            
                            if len(self._active_tasks) < self._total_slots:
                                self._active_tasks[requesting_thread_ident] = {
                                    'priority': ELP1,
                                    'proc': None,
                                    'start_time': time.monotonic()
                                }
                                # Update Atomic Flag
                                global ELP1_ACTIVE_FLAG
                                ELP1_ACTIVE_FLAG = True
                                
                                logger.info(f"{log_prefix}:: Acquired ELP1 slot. Atomic Flag SET. Active: {len(self._active_tasks)}/{self._total_slots}")
                                self._condition.notify_all()
                                return True

                    elif priority == ELP0:
                        if self._elp1_waiting_count == 0:
                            if current_elp0_count < self._max_elp0_slots and total_active < self._total_slots:
                                self._active_tasks[requesting_thread_ident] = {
                                    'priority': ELP0,
                                    'proc': None,
                                    'start_time': time.monotonic()
                                }
                                logger.info(f"{log_prefix}:: Acquired ELP0 slot. Active: {len(self._active_tasks)}/{self._total_slots}")
                                return True

                    # Wait Path
                    remaining_timeout = None
                    if timeout is not None:
                        elapsed = time.monotonic() - acquire_start_time
                        remaining_timeout = timeout - elapsed
                        if remaining_timeout <= 0:
                            return False

                    wait_time = 0.1 if (priority == ELP0 and self._elp1_waiting_count > 0) else remaining_timeout
                    self._condition.wait(timeout=wait_time)

                    if timeout is not None and (time.monotonic() - acquire_start_time) >= timeout:
                        return False

            finally:
                if priority == ELP1:
                    self._elp1_waiting_count -= 1
                else:
                    self._elp0_waiting_count -= 1

    def set_holder_process(self, proc: subprocess.Popen):
        with self._condition:
            tid = threading.get_ident()
            if tid in self._active_tasks:
                self._active_tasks[tid]['proc'] = proc
                logger.trace(f"PQLock|SetProc|Thr{tid}:: Associated PID {proc.pid}")

    def _kill_process_tree(self, pid: int):
        """Recursively kills a process and all its children using psutil."""
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    logger.warning(f"🔪 Killing child process {child.pid} ({child.name()})")
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            logger.warning(f"🔪 Killing parent process {pid}")
            parent.kill()
            psutil.wait_procs([parent] + children, timeout=0.1)
        except psutil.NoSuchProcess:
            logger.warning(f"Process PID {pid} already dead or not found.")
        except Exception as e:
            logger.error(f"Error while killing process tree for PID {pid}: {e}")

    def release(self):
        releasing_thread_ident = threading.get_ident()
        with self._condition:
            if releasing_thread_ident in self._active_tasks:
                task = self._active_tasks.pop(releasing_thread_ident)
                
                # Update Atomic Flag if no more ELP1 tasks
                global ELP1_ACTIVE_FLAG
                has_other_elp1 = any(t['priority'] == ELP1 for t in self._active_tasks.values())
                if not has_other_elp1:
                    ELP1_ACTIVE_FLAG = False
                    logger.debug("PQLock|Flag|Atomic Flag CLEARED (No more ELP1 tasks).")

                logger.info(f"PQLock|RLS|Thr{releasing_thread_ident}:: Released ELP{task['priority']} slot.")
                self._condition.notify_all()
            else:
                logger.warning(f"PQLock|RLS|Thr{releasing_thread_ident}:: Attempted release but no active task found (preempted?).")

    def get_status(self) -> Tuple[bool, Optional[int], int]:
        with self._condition:
            is_locked = len(self._active_tasks) > 0
            holder_prio = ELP1 if any(t['priority'] == ELP1 for t in self._active_tasks.values()) else (ELP0 if is_locked else None)
            return (is_locked, holder_prio, self._elp1_interrupt_quota)

    def get_status_extended(self) -> Tuple[bool, Optional[int], int, int]:
        """Returns (is_locked, holder_priority, elp1_quota, elp1_waiting_count)."""
        with self._condition:
            is_locked = len(self._active_tasks) > 0
            holder_prio = ELP1 if any(t['priority'] == ELP1 for t in self._active_tasks.values()) else (ELP0 if is_locked else None)
            return (
                is_locked,
                holder_prio,
                self._elp1_interrupt_quota,
                self._elp1_waiting_count,
            )

