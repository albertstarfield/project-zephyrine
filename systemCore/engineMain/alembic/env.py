
import os, sys; from logging.config import fileConfig; from sqlalchemy import create_engine, pool; from alembic import context
ENV_PY_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)); ENGINE_MAIN_DIR = os.path.abspath(os.path.join(ENV_PY_SCRIPT_DIR, '..'))
if ENGINE_MAIN_DIR not in sys.path: sys.path.insert(0, ENGINE_MAIN_DIR)
target_metadata = None; EFFECTIVE_DATABASE_URL_FOR_ALEMBIC = None
try:
    from database import Base, Interaction, AppleScriptAttempt, FileIndex
    target_metadata = Base.metadata
    from config import PROJECT_CONFIG_DATABASE_URL
    EFFECTIVE_DATABASE_URL_FOR_ALEMBIC = PROJECT_CONFIG_DATABASE_URL
    print(f"[alembic/env.py] Imported Base.metadata and PROJECT_CONFIG_DATABASE_URL='sqlite:////Users/albertstarfield/Documents/misc/AI/project-zephyrine/systemCore/engineMain/mappedknowledge.db'", file=sys.stderr)
except ImportError as e_import_env: print(f"[alembic/env.py] CRITICAL ERROR importing: {e_import_env}"); raise
alembic_ini_config = context.config
if not EFFECTIVE_DATABASE_URL_FOR_ALEMBIC: raise ValueError("PROJECT_CONFIG_DATABASE_URL from config.py was None. Alembic cannot proceed.")
alembic_ini_config.set_main_option("sqlalchemy.url", EFFECTIVE_DATABASE_URL_FOR_ALEMBIC)
print(f"[alembic/env.py] Alembic context using DB URL: {EFFECTIVE_DATABASE_URL_FOR_ALEMBIC}", file=sys.stderr)
if alembic_ini_config.config_file_name is not None:
    try: fileConfig(alembic_ini_config.config_file_name)
    except Exception as fc_err: print(f"[alembic/env.py] Warn: Error log config from alembic.ini: {fc_err}", file=sys.stderr)
is_sqlite = EFFECTIVE_DATABASE_URL_FOR_ALEMBIC.startswith("sqlite")
def run_migrations_offline() -> None:
    context.configure(url=EFFECTIVE_DATABASE_URL_FOR_ALEMBIC, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"}, render_as_batch=is_sqlite)
    with context.begin_transaction(): context.run_migrations()
def run_migrations_online() -> None:
    connectable_args = {"connect_args": {"check_same_thread": False, "timeout": 60.0}} if is_sqlite else {}
    connectable = create_engine(EFFECTIVE_DATABASE_URL_FOR_ALEMBIC, poolclass=pool.NullPool, **connectable_args) # type: ignore
    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=is_sqlite, compare_type=True)
            with context.begin_transaction(): context.run_migrations()
    finally:
        if connectable: connectable.dispose()
if context.is_offline_mode(): run_migrations_offline()
else: run_migrations_online()
