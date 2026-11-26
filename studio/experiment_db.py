import json
import sqlite3
import time
from pathlib import Path


class ExperimentDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    job_id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    state TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    pid INTEGER,
                    command_json TEXT
                )
                """
            )

    def upsert(self, job_id, state, output_path, config, pid=None, command=None):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments(job_id, created_at, state, output_path, config_json, pid, command_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    state=excluded.state,
                    pid=COALESCE(excluded.pid, experiments.pid),
                    command_json=COALESCE(excluded.command_json, experiments.command_json)
                """,
                (
                    job_id,
                    time.time(),
                    state,
                    str(output_path),
                    json.dumps(config, ensure_ascii=False),
                    pid,
                    json.dumps(command, ensure_ascii=False) if command else None,
                ),
            )

    def update_state(self, job_id, state):
        with self._connect() as conn:
            conn.execute("UPDATE experiments SET state=? WHERE job_id=?", (state, job_id))

    def list_recent(self, limit=20):
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (int(limit),)
            ).fetchall()
        return [dict(row) for row in rows]
