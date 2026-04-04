
import sqlite3

import numpy as np

import json

from datetime import datetime

from pathlib import Path

import threading

from typing import Optional, List, Dict, Tuple, Any

from src.core.config import FACE_DB_PATH
from src.core.utils import logger

DB_PATH = Path(FACE_DB_PATH) / "echora.db"

class Database:
    """
    Manages all persistent storage for ECHORA.

    Stores face profiles, user preferences, and event logs in a SQLite
    database file at database/faces/echora.db.

    Thread-safe — uses a lock to prevent concurrent write corruption.

    Usage:
        db = Database()
        db.init_db()

        db.add_person("Ahmed", embedding_array)

        persons = db.get_all_persons()

        db.set_preference("dominant_hand", "Right")

        hand = db.get_preference("dominant_hand", default="Right")
    """

    def __init__(self):
        """
        Creates the Database object. Does NOT connect yet.
        Call init_db() to create/open the database.
        """

        self._conn: Optional[sqlite3.Connection] = None

        self._lock = threading.Lock()

        self._ready: bool = False

        logger.info("Database created. Call init_db() to open.")

    def init_db(self):
        """
        Opens (or creates) the SQLite database and creates all tables.

        If the database file doesn't exist, SQLite creates it automatically.
        If it already exists, we just open it — existing data is preserved.

        CREATE TABLE IF NOT EXISTS means this is safe to call multiple times
        without destroying existing data.
        """

        DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Opening database at: {DB_PATH}")

        try:
            self._conn = sqlite3.connect(
                str(DB_PATH),
                check_same_thread=False
            )

            self._conn.execute("PRAGMA journal_mode=WAL")

            self._conn.execute("PRAGMA foreign_keys=ON")

            self._conn.row_factory = sqlite3.Row

            self._create_tables()

            self._ready = True
            logger.info("Database opened successfully.")

            n_persons = self._count_rows("persons")
            logger.info(f"  Registered persons: {n_persons}")

        except Exception as e:
            logger.error(f"Failed to open database: {e}")
            raise

    def _create_tables(self):
        """
        Creates all database tables if they don't already exist.

        Called once during init_db(). Safe to call on an existing database —
        IF NOT EXISTS prevents dropping existing data.
        """

        cursor = self._conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                embedding   BLOB    NOT NULL,
                added_at    TEXT    NOT NULL,
                last_seen   TEXT,
                seen_count  INTEGER NOT NULL DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key         TEXT NOT NULL PRIMARY KEY,
                value       TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT    NOT NULL,
                details     TEXT    NOT NULL DEFAULT '{}',
                timestamp   TEXT    NOT NULL
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_persons_name
            ON persons(name)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_log_type
            ON event_log(event_type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_log_timestamp
            ON event_log(timestamp)
        """)

        self._conn.commit()

        logger.debug("Database tables created/verified.")

    def add_person(
        self,
        name:      str,
        embedding: np.ndarray
    ) -> bool:
        """
        Adds or updates a person in the database.

        If the name already exists, updates their embedding (re-registration).
        If the name is new, creates a new row.

        Arguments:
            name:      the person's name — must be unique
            embedding: numpy array of shape (128,) — the face embedding vector

        Returns:
            True if saved successfully, False on error.
        """

        if not self._ready:
            logger.error("Database not initialised. Call init_db() first.")
            return False

        try:
            embedding_bytes = embedding.astype(np.float64).tobytes()

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with self._lock:
                cursor = self._conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO persons
                        (name, embedding, added_at, last_seen, seen_count)
                    VALUES
                        (?, ?, ?, NULL, 0)
                """, (name, embedding_bytes, now))

                self._conn.commit()

            logger.info(f"Person saved: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to save person {name}: {e}")
            return False

    def get_person_by_name(self, name: str) -> Optional[Dict]:
        """
        Retrieves one person's record by name.

        Arguments:
            name: the person's name to look up

        Returns:
            Dictionary with person data, or None if not found.
            {
              "id":         int,
              "name":       str,
              "embedding":  numpy array (128,),
              "added_at":   str,
              "last_seen":  str or None,
              "seen_count": int
            }
        """

        if not self._ready:
            return None

        try:
            cursor = self._conn.cursor()

            cursor.execute(
                "SELECT * FROM persons WHERE name = ?",
                (name,)   # note the comma — (name,) is a tuple, not (name)
            )

            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_person_dict(row)

        except Exception as e:
            logger.error(f"Failed to get person {name}: {e}")
            return None

    def get_all_persons(self) -> List[Dict]:
        """
        Returns all registered persons with their embeddings.

        Called by face_recognition.py at startup to load all known faces
        into memory for comparison.

        Returns:
            List of person dictionaries. Empty list if nobody registered.
        """

        if not self._ready:
            return []

        try:
            cursor = self._conn.cursor()

            cursor.execute("SELECT * FROM persons ORDER BY name")

            rows = cursor.fetchall()

            return [self._row_to_person_dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get all persons: {e}")
            return []

    def update_last_seen(self, name: str):
        """
        Updates the last_seen timestamp and increments seen_count.

        Called every time face_recognition.py successfully identifies someone.
        Enables "Ahmed, last seen 2 days ago" type announcements.

        Arguments:
            name: the identified person's name
        """

        if not self._ready:
            return

        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with self._lock:
                cursor = self._conn.cursor()

                cursor.execute("""
                    UPDATE persons
                    SET last_seen  = ?,
                        seen_count = seen_count + 1
                    WHERE name = ?
                """, (now, name))

                self._conn.commit()

        except Exception as e:
            logger.error(f"Failed to update last_seen for {name}: {e}")

    def delete_person(self, name: str) -> bool:
        """
        Removes a person from the database permanently.

        Arguments:
            name: the person's name to delete

        Returns:
            True if deleted, False if not found or error.
        """

        if not self._ready:
            return False

        try:
            with self._lock:
                cursor = self._conn.cursor()

                cursor.execute(
                    "DELETE FROM persons WHERE name = ?",
                    (name,)
                )

                deleted = cursor.rowcount > 0
                self._conn.commit()

            if deleted:
                logger.info(f"Person deleted: {name}")
            else:
                logger.warning(f"Person not found for deletion: {name}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete person {name}: {e}")
            return False

    def get_person_count(self) -> int:
        """Returns the total number of registered persons."""
        return self._count_rows("persons")

    def _row_to_person_dict(self, row: sqlite3.Row) -> Dict:
        """
        Converts a sqlite3.Row from the persons table to a clean dictionary.

        The key step here is converting the stored bytes back to a numpy array.
        When we stored the embedding, we called .tobytes().
        Now we reverse it with np.frombuffer().

        Arguments:
            row: a sqlite3.Row object from a persons query

        Returns:
            Dictionary with all person fields, embedding as numpy array.
        """

        embedding = np.frombuffer(row["embedding"], dtype=np.float64)

        return {
            "id":         row["id"],
            "name":       row["name"],
            "embedding":  embedding,
            "added_at":   row["added_at"],
            "last_seen":  row["last_seen"],
            "seen_count": row["seen_count"],
        }

    def set_preference(self, key: str, value: Any):
        """
        Saves a user preference to the database.

        Preferences are key-value pairs persisted across restarts.
        Examples:
          set_preference("dominant_hand", "Right")
          set_preference("tts_volume", "0.8")
          set_preference("ocr_language", "en")

        Arguments:
            key:   the setting name (e.g. "dominant_hand")
            value: the setting value — any type, converted to string
        """

        if not self._ready:
            return

        try:
            value_str = str(value)
            now       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with self._lock:
                cursor = self._conn.cursor()

                cursor.execute("""
                    INSERT OR REPLACE INTO preferences (key, value, updated_at)
                    VALUES (?, ?, ?)
                """, (key, value_str, now))

                self._conn.commit()

            logger.debug(f"Preference saved: {key} = {value_str}")

        except Exception as e:
            logger.error(f"Failed to save preference {key}: {e}")

    def get_preference(self, key: str, default: Any = None) -> Optional[str]:
        """
        Reads a user preference from the database.

        Arguments:
            key:     the setting name to look up
            default: value to return if the key doesn't exist

        Returns:
            The stored value as a string, or default if not found.

        Examples:
            hand = db.get_preference("dominant_hand", default="Right")
            vol  = float(db.get_preference("tts_volume", default="0.8"))
        """

        if not self._ready:
            return default

        try:
            cursor = self._conn.cursor()

            cursor.execute(
                "SELECT value FROM preferences WHERE key = ?",
                (key,)
            )

            row = cursor.fetchone()

            if row is None:
                return default

            return row["value"]

        except Exception as e:
            logger.error(f"Failed to get preference {key}: {e}")
            return default

    def get_all_preferences(self) -> Dict[str, str]:
        """
        Returns all stored preferences as a dictionary.

        Used at startup to load all user settings at once.

        Returns:
            Dict like {"dominant_hand": "Right", "tts_volume": "0.8", ...}
        """

        if not self._ready:
            return {}

        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT key, value FROM preferences")
            rows = cursor.fetchall()

            return {row["key"]: row["value"] for row in rows}

        except Exception as e:
            logger.error(f"Failed to get all preferences: {e}")
            return {}

    def log_event(self, event_type: str, details: Dict = None):
        """
        Records a significant event to the log.

        Events are append-only — never modified, only added.
        Useful for: "last time Ahmed was seen", "how many banknotes scanned"

        Arguments:
            event_type: category string e.g. "face_identified", "ocr_read"
            details:    optional dictionary with event-specific data
                        e.g. {"name": "Ahmed", "confidence": 0.92}

        Examples:
            db.log_event("face_identified", {"name": "Ahmed"})
            db.log_event("banknote_scanned", {"denomination": "50 pounds"})
            db.log_event("obstacle_danger",  {"label": "person", "dist_mm": 600})
        """

        if not self._ready:
            return

        try:
            details_str = json.dumps(details or {})
            now         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with self._lock:
                cursor = self._conn.cursor()

                cursor.execute("""
                    INSERT INTO event_log (event_type, details, timestamp)
                    VALUES (?, ?, ?)
                """, (event_type, details_str, now))

                self._conn.commit()

        except Exception as e:
            logger.error(f"Failed to log event {event_type}: {e}")

    def get_recent_events(
        self,
        n:          int           = 20,
        event_type: Optional[str] = None
    ) -> List[Dict]:
        """
        Returns the last N events from the log.

        Arguments:
            n:          how many events to return (default 20)
            event_type: optional filter — only return events of this type
                        e.g. "face_identified"

        Returns:
            List of event dictionaries, newest first.
        """

        if not self._ready:
            return []

        try:
            cursor = self._conn.cursor()

            if event_type:
                cursor.execute("""
                    SELECT * FROM event_log
                    WHERE event_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (event_type, n))
            else:
                cursor.execute("""
                    SELECT * FROM event_log
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (n,))

            rows = cursor.fetchall()

            events = []
            for row in rows:
                events.append({
                    "id":         row["id"],
                    "event_type": row["event_type"],
                    "details":    json.loads(row["details"]),
                    "timestamp":  row["timestamp"],
                })

            return events

        except Exception as e:
            logger.error(f"Failed to get recent events: {e}")
            return []

    def get_last_seen_time(self, name: str) -> Optional[str]:
        """
        Returns when a specific person was last identified, as a string.

        Used for "Ahmed, last seen 2 days ago" type announcements.

        Arguments:
            name: the person's name

        Returns:
            Timestamp string like "2024-01-15 14:30:00" or None.
        """

        person = self.get_person_by_name(name)
        if person is None:
            return None
        return person.get("last_seen")

    def _count_rows(self, table: str) -> int:
        """
        Returns the number of rows in a table.
        Used for diagnostics and logging.
        """

        if not self._ready or self._conn is None:
            return 0

        try:
            cursor = self._conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    def get_stats(self) -> Dict:
        """Returns diagnostic statistics about the database."""

        return {
            "ready":        self._ready,
            "db_path":      str(DB_PATH),
            "persons":      self._count_rows("persons"),
            "preferences":  self._count_rows("preferences"),
            "events":       self._count_rows("event_log"),
        }

    def close(self):
        """
        Closes the database connection cleanly.
        Call this when ECHORA shuts down.
        """

        if self._conn is not None:
            try:
                self._conn.close()
                self._conn = None
                self._ready = False
                logger.info("Database closed.")
            except Exception as e:
                logger.error(f"Error closing database: {e}")

_db: Optional[Database] = None

def init_database() -> Database:
    """
    Initialises the module-level database singleton.
    Call once at startup from control_unit.py.

    Returns the Database instance so callers can use it directly.
    """

    global _db

    if _db is not None:
        logger.debug("Database already initialised.")
        return _db

    _db = Database()
    _db.init_db()
    logger.info("Module-level database ready.")
    return _db

def get_db() -> Optional[Database]:
    """
    Returns the shared database instance.
    Returns None if init_database() has not been called yet.
    """
    return _db

if __name__ == "__main__":

    print("=== ECHORA database.py self-test ===\n")

    db = Database()
    db.init_db()
    print(f"Database opened at: {DB_PATH}\n")

    print("Test 1: Add persons")

    embedding_ahmed = np.random.rand(128).astype(np.float64)
    embedding_sara  = np.random.rand(128).astype(np.float64)

    ok1 = db.add_person("Ahmed", embedding_ahmed)
    ok2 = db.add_person("Sara",  embedding_sara)

    print(f"  Added Ahmed: {ok1}")
    print(f"  Added Sara:  {ok2}")
    assert ok1 and ok2
    print("  PASSED\n")

    print("Test 2: Get person by name")

    person = db.get_person_by_name("Ahmed")
    assert person is not None
    assert person["name"] == "Ahmed"

    assert np.allclose(person["embedding"], embedding_ahmed)
    print(f"  Ahmed found: seen_count={person['seen_count']}")
    print(f"  Embedding matches: {np.allclose(person['embedding'], embedding_ahmed)}")
    print("  PASSED\n")

    print("Test 3: Get all persons")

    all_persons = db.get_all_persons()
    print(f"  Total persons: {len(all_persons)}")
    for p in all_persons:
        print(f"    {p['name']} — added: {p['added_at']}")
    assert len(all_persons) >= 2
    print("  PASSED\n")

    print("Test 4: Update last seen")

    db.update_last_seen("Ahmed")
    db.update_last_seen("Ahmed")

    person = db.get_person_by_name("Ahmed")
    print(f"  Ahmed seen_count: {person['seen_count']} (expected >= 2)")
    print(f"  Last seen: {person['last_seen']}")
    assert person["seen_count"] >= 2
    print("  PASSED\n")

    print("Test 5: Preferences")

    db.set_preference("dominant_hand", "Right")
    db.set_preference("tts_volume",    "0.8")
    db.set_preference("ocr_language",  "en")

    hand = db.get_preference("dominant_hand", default="Left")
    vol  = db.get_preference("tts_volume",    default="1.0")
    lang = db.get_preference("ocr_language",  default="en")
    miss = db.get_preference("nonexistent",   default="fallback")

    print(f"  dominant_hand: {hand}")
    print(f"  tts_volume:    {vol}")
    print(f"  ocr_language:  {lang}")
    print(f"  nonexistent:   {miss} (should be 'fallback')")

    assert hand == "Right"
    assert vol  == "0.8"
    assert miss == "fallback"
    print("  PASSED\n")

    print("Test 6: Event log")

    db.log_event("face_identified", {"name": "Ahmed", "confidence": 0.91})
    db.log_event("banknote_scanned", {"denomination": "50 pounds"})
    db.log_event("obstacle_danger",  {"label": "person", "dist_mm": 600})
    db.log_event("face_identified",  {"name": "Sara",  "confidence": 0.88})

    recent = db.get_recent_events(n=10)
    print(f"  Total events logged: {len(recent)}")
    for e in recent:
        print(f"    [{e['timestamp']}] {e['event_type']}: {e['details']}")

    face_events = db.get_recent_events(n=10, event_type="face_identified")
    print(f"  Face events only: {len(face_events)}")
    assert len(face_events) >= 2
    print("  PASSED\n")

    print("Test 7: Delete person")

    deleted = db.delete_person("Sara")
    assert deleted
    assert db.get_person_by_name("Sara") is None
    print(f"  Sara deleted: {deleted}")
    print(f"  Sara still exists: {db.get_person_by_name('Sara') is not None}")
    print("  PASSED\n")

    print("Test 8: Stats")
    stats = db.get_stats()
    for key, val in stats.items():
        print(f"  {key}: {val}")
    print("  PASSED\n")

    db.delete_person("Ahmed")

    db.close()
    print("=== All tests passed ===")