
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, create_engine
from alembic import context

# --- Detect Project Root and Add to Path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is alembic/
APP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..')) # This should be engineMain/
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
print(f"Alembic env.py: Added {APP_DIR} to sys.path")

# --- Import Base and Models ---
target_metadata = None
RUNTIME_DATABASE_URL_FROM_DB_PY = None
try:
    # Import Base AND URL from the database module (sibling to alembic dir)
    from database import Base, RUNTIME_DATABASE_URL
    RUNTIME_DATABASE_URL_FROM_DB_PY = RUNTIME_DATABASE_URL
    print(f"Alembic env.py: Imported Base. Using RUNTIME DB URL: {RUNTIME_DATABASE_URL_FROM_DB_PY}")

    # IMPORTANT: Import all models defined in database.py or models.py
    # This ensures they are registered with Base.metadata for autogenerate
    from database import Interaction, AppleScriptAttempt, FileIndex # Adjust if models are elsewhere
    print(f"Alembic env.py: Explicitly imported models.")

    target_metadata = Base.metadata # Assign metadata AFTER models are imported
    print(f"Alembic env.py: Assigned Base.metadata.")

except ImportError as import_err:
    print(f"Alembic env.py: ERROR importing from database.py: {import_err}")
    print(f"Alembic env.py: Check file structure and ensure models are importable.")
except Exception as e:
    print(f"Alembic env.py: UNEXPECTED ERROR during import: {e}")

# --- Alembic Configuration ---
config = context.config

# Ensure target_metadata is loaded before proceeding
if target_metadata is None:
    raise RuntimeError("Alembic env.py: target_metadata not loaded. Check imports and model definitions in database.py.")

# Set the effective URL from the imported variable
effective_db_url = RUNTIME_DATABASE_URL_FROM_DB_PY
if not effective_db_url:
    # Fallback to ini file if import failed, but warn heavily
    print("Alembic env.py: WARNING - Failed to import URL from database.py, falling back to alembic.ini URL.")
    effective_db_url = config.get_main_option("sqlalchemy.url")
    if not effective_db_url:
         raise ValueError("Alembic env.py: Database URL not found in database.py or alembic.ini.")

print(f"Alembic env.py: Configuring Alembic with effective URL: {effective_db_url}")
config.set_main_option("sqlalchemy.url", effective_db_url)

# Interpret the config file for Python logging.
# This line needs to be placed after configuring the URL.
if config.config_file_name is not None:
    try:
        fileConfig(config.config_file_name)
    except Exception as fc_err:
        print(f"Alembic env.py: Error processing logging config from alembic.ini: {fc_err}")


# Other settings from the environment, ensure batch mode for SQLite
is_sqlite = effective_db_url.startswith("sqlite")
print(f"Alembic env.py: Detected SQLite: {is_sqlite}")

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    print("Running migrations offline...")
    if not effective_db_url: raise ValueError("DB URL not set for offline mode.")
    context.configure(
        url=effective_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=is_sqlite # Use batch mode for SQLite offline
    )
    with context.begin_transaction():
        context.run_migrations()
    print("Offline migrations finished.")


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    print("Running migrations online...")
    if not effective_db_url: raise ValueError("DB URL not set for online mode.")

    connectable_args = {}
    # Specific connect args for SQLite in online mode
    if is_sqlite:
        connectable_args["connect_args"] = {"check_same_thread": False, "timeout": 60.0} # Longer timeout for migrations
        print("Applying SQLite connect_args for online migration.")

    # Create engine specifically for migration
    connectable = create_engine(
        effective_db_url,
        poolclass=pool.NullPool, # Avoid pooling for migrations
        **connectable_args
    )
    print("Migration engine created.")
    try:
        with connectable.connect() as connection:
            print("Established connection for online migration.")
            # Configure context, MUST include render_as_batch for SQLite ALTER support
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=is_sqlite, # <<< CRITICAL FOR SQLITE ALTERS >>>
                compare_type=True # Compare column types during autogenerate checks
            )
            print("Beginning transaction and running migrations...")
            try:
                with context.begin_transaction():
                    context.run_migrations()
                print("Online migrations completed successfully.")
            except Exception as mig_err:
                print(f"Alembic online migration execution ERROR: {mig_err}")
                raise # Re-raise to be caught by caller if needed
    except Exception as conn_err:
        print(f"Alembic online database connection ERROR: {conn_err}")
        raise # Re-raise
    finally:
        if 'connectable' in locals() and connectable:
             connectable.dispose()
             print("Migration engine disposed.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

print("Alembic env.py finished.")
