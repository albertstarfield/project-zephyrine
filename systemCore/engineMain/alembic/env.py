
import os
import sys
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool, create_engine
from alembic import context

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_DIR)
print(f"Alembic env.py: Added {PROJECT_DIR} to path")

target_metadata = None # Initialize

try:
    # Import Base and the *runtime* URL from database.py
    from database import Base, RUNTIME_DATABASE_URL
    print(f"Alembic env.py: Imported Base. Using RUNTIME DB URL: {RUNTIME_DATABASE_URL}")

    # Explicitly import models to ensure registration with Base.metadata
    from database import Interaction, AppleScriptAttempt, FileIndex
    print(f"Alembic env.py: Explicitly imported models.")

    target_metadata = Base.metadata # Assign metadata AFTER models are imported
    print(f"Alembic env.py: Assigned Base.metadata.")

except Exception as e:
    print(f"Alembic env.py: ERROR importing from database.py: {e}")
    target_metadata = None
    RUNTIME_DATABASE_URL = None # Clear URL on error too

config = context.config

# Ensure target_metadata is loaded before proceeding
if target_metadata is None:
    raise RuntimeError("Alembic env.py: target_metadata not loaded.")

# Set the effective URL from the imported variable
effective_db_url = RUNTIME_DATABASE_URL
if not effective_db_url:
    raise ValueError("Alembic env.py: Runtime Database URL not found.")

print(f"Alembic env.py: Configuring Alembic with effective URL: {effective_db_url}")
config.set_main_option("sqlalchemy.url", effective_db_url)

if config.config_file_name:
    fileConfig(config.config_file_name)

is_sqlite = effective_db_url.startswith("sqlite")
print(f"Alembic env.py: Detected SQLite: {is_sqlite}")

def run_migrations_offline() -> None:
    # ... (offline mode logic remains the same, uses effective_db_url) ...
    print("Running migrations offline...")
    context.configure(
        url=effective_db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_as_batch=is_sqlite
    )
    with context.begin_transaction():
        context.run_migrations()
    print("Offline migrations finished.")


def run_migrations_online() -> None:
    # ... (online mode logic remains the same, uses effective_db_url) ...
    print("Running migrations online...")
    connectable_args = {}
    if is_sqlite:
        connectable_args["connect_args"] = {"check_same_thread": False, "timeout": 30}
        print("Applying SQLite connect_args for online migration.")

    connectable = create_engine(
        effective_db_url,
        poolclass=pool.NullPool,
        **connectable_args
    )
    try:
        with connectable.connect() as connection:
            print("Established connection for online migration.")
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                render_as_batch=is_sqlite,
                compare_type=True
            )
            print("Beginning transaction and running migrations...")
            try:
                with context.begin_transaction():
                    context.run_migrations()
                print("Online migrations completed successfully.")
            except Exception as mig_err:
                print(f"Alembic online migration execution ERROR: {mig_err}")
                raise
    except Exception as conn_err:
        print(f"Alembic online database connection ERROR: {conn_err}")
        raise
    finally:
        if 'connectable' in locals() and connectable:
             connectable.dispose()
             print("Migration engine disposed.")


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

print("Alembic env.py finished.")
