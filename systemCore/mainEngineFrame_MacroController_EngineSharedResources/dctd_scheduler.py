# dctd_scheduler.py
import threading
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional, List
from loguru import logger
from sqlalchemy.orm import Session

# Local Imports
from database import (
    SessionLocal, 
    ScheduledThoughtTask, 
    get_due_tasks_for_execution,
    _DB_HEALTH_OK_EVENT
)
from CortexConfiguration import (
    ENABLE_DCTD_SCHEDULER,
    DCTD_SCHEDULER_POLL_INTERVAL_SECONDS,
    DCTD_SCHEDULER_MAX_CATCHUP_BATCH_SIZE,
    DCTD_SCHEDULER_CATCHUP_WINDOW_HOURS
)

# We need a reference to the main AI engine to dispatch tasks.
# Since we can't easily import 'cortex_text_interaction' directly without circular deps,
# we will accept it as an argument or set it globally.
_global_cortex_ref = None

def set_scheduler_cortex_ref(cortex_instance):
    global _global_cortex_ref
    _global_cortex_ref = cortex_instance

class DCTDSchedulerThread(threading.Thread):
    """
    The 'Chronos' Daemon.
    Periodically polls for scheduled thoughts that are due.
    Handles immediate execution and offline catch-up.
    """
    def __init__(self, stop_event: threading.Event):
        super().__init__(name="DCTDSchedulerThread", daemon=True)
        self.stop_event = stop_event
        self._last_poll_time = 0.0

    def run(self):
        if not ENABLE_DCTD_SCHEDULER:
            logger.info("⏳ DCTD Scheduler disabled by config. Exiting thread.")
            return

        logger.info("⏳ DCTD Scheduler 'Chronos' thread started (Aggressive Access Mode).")

        # NO WAITING. WE GO STRAIGHT TO THE LOOP.
        while not self.stop_event.is_set():
            try:
                # Attempt to access the DB immediately.
                # If tables aren't ready or DB is locked, this will raise an Exception.
                self._process_due_tasks()
                
                # If successful, wait standard interval
                self.stop_event.wait(timeout=DCTD_SCHEDULER_POLL_INTERVAL_SECONDS)

            except Exception as e:
                # DB failed (Locked, missing tables, etc.). 
                # Log it, wait 5 seconds, and hammer it again.
                logger.warning(f"⏳ Scheduler DB Access Failed (Retrying in 5s): {e}")
                self.stop_event.wait(5.0)

    def _process_due_tasks(self):
        """
        Core logic: Check DB, find due tasks, dispatch to ELP0.
        """
        # 1. Acquire DB Session
        db: Session = SessionLocal()
        try:
            # 2. Poll for tasks
            due_tasks: List[ScheduledThoughtTask] = get_due_tasks_for_execution(
                db, 
                batch_size=DCTD_SCHEDULER_MAX_CATCHUP_BATCH_SIZE
            )

            if not due_tasks:
                return # Nothing to do

            now_utc = datetime.now(timezone.utc)
            
            for task in due_tasks:
                if self.stop_event.is_set(): break

                # 3. Analyze Temporal Drift (Catch-Up Logic)
                # Calculate how late we are
                # Ensure scheduled_time is timezone-aware for math
                scheduled_time_aware = task.scheduled_time
                if scheduled_time_aware.tzinfo is None:
                    scheduled_time_aware = scheduled_time_aware.replace(tzinfo=timezone.utc)
                
                drift_seconds = (now_utc - scheduled_time_aware).total_seconds()
                
                # Check for "Missed Catch-Up" status
                is_catchup = False
                if drift_seconds > (DCTD_SCHEDULER_CATCHUP_WINDOW_HOURS * 3600):
                    # Too old? We might want to skip or mark specially.
                    # For now, we process it but log it as a catchup.
                    logger.warning(f"⏳ Task {task.id} is {drift_seconds/3600:.1f} hours late. Processing as CATCH-UP.")
                    task.status = "MISSED_CATCHUP_EXECUTING"
                    is_catchup = True
                else:
                    task.status = "EXECUTING"
                
                task.execution_attempt_count += 1
                task.execution_actual_time = now_utc
                db.commit() # Save state before executing

                # 4. Dispatch to ELP0 (Async injection)
                # We use asyncio.run_coroutine_threadsafe to inject into the main event loop
                self._dispatch_to_elp0(task, is_catchup)

        finally:
            db.close()

    def _dispatch_to_elp0(self, task_record: ScheduledThoughtTask, is_catchup: bool):
        """
        Injects the task into the main cortex pipeline.
        """
        global _global_cortex_ref
        if not _global_cortex_ref:
            logger.error("⏳ Scheduler cannot dispatch: Cortex reference not set.")
            return

        prompt = task_record.prompt
        if is_catchup:
            prompt = f"[SYSTEM NOTICE: This thought is a recovered memory from {task_record.scheduled_time}. Treat it as a past realization surfacing now.]\n\n{prompt}"

        # We need to find the running event loop to schedule the async task
        # Since this is a thread, we can't just await.
        try:
            # Create a fire-and-forget wrapper that updates the DB on completion
            asyncio.run_coroutine_threadsafe(
                self._async_execute_wrapper(task_record.id, prompt, task_record.source_interaction_id),
                _global_cortex_ref.provider.loop # Assuming provider has a loop ref, or get global loop
            )
            logger.info(f"⏳ Dispatched Task {task_record.id} to ELP0.")
        except Exception as e:
            logger.error(f"⏳ Failed to schedule async task: {e}")
            # We assume the main loop handles it, but if dispatch fails here, we might need DB rollback logic in future versions.

    async def _async_execute_wrapper(self, task_id: int, prompt: str, source_id: Optional[int]):
        """
        The coroutine that actually runs in the main loop.
        It calls background_generate and then updates the task status to COMPLETED.
        """
        global _global_cortex_ref
        
        # New DB session for the async task result update
        db_async = SessionLocal()
        
        try:
            session_id_val = f"scheduled_task_{task_id}_{int(time.time())}"
            
            # CALL ELP0
            await _global_cortex_ref.background_generate(
                db=db_async,
                user_input=prompt,
                session_id=session_id_val,
                classification="scheduled_thought_execution",
                image_b64=None,
                update_interaction_id=source_interaction_id # Link back if needed
            )
            
            # Update Task Status to COMPLETED
            # We need to re-fetch the task because the session is new
            task = db_async.query(ScheduledThoughtTask).filter(ScheduledThoughtTask.id == task_id).first()
            if task:
                task.status = "COMPLETED"
                db_async.commit()
                logger.success(f"⏳ Task {task_id} execution and logging complete.")
                
        except Exception as e:
            logger.error(f"⏳ Task {task_id} execution FAILED: {e}")
            task = db_async.query(ScheduledThoughtTask).filter(ScheduledThoughtTask.id == task_id).first()
            if task:
                task.status = "FAILED"
                task.error_log = str(e)
                db_async.commit()
        finally:
            db_async.close()