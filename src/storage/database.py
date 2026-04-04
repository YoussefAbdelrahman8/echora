import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from sqlalchemy import create_engine, Integer, String, LargeBinary, Text
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, Session

from src.core.config import settings
from src.core.utils import logger

DB_PATH = Path(settings.FACE_DB_PATH) / "echora.db"

Base = declarative_base()

class Person(Base):
    __tablename__ = 'persons'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    added_at: Mapped[str] = mapped_column(String, nullable=False)
    last_seen: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    seen_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

class Preference(Base):
    __tablename__ = 'preferences'
    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

class EventLog(Base):
    __tablename__ = 'event_log'
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    details: Mapped[str] = mapped_column(Text, default='{}', nullable=False)
    timestamp: Mapped[str] = mapped_column(String, index=True, nullable=False)


class Database:
    """
    Manages all persistent storage for ECHORA using SQLAlchemy ORM.
    Stores face profiles, user preferences, and event logs.
    """
    def __init__(self):
        self._engine = None
        self._ready = False
        logger.info("Database created. Call init_db() to open.")

    def init_db(self):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Opening database at: {DB_PATH}")

        try:
            self._engine = create_engine(f"sqlite:///{DB_PATH}?check_same_thread=False")
            Base.metadata.create_all(self._engine)
            self._ready = True
            logger.info("Database opened successfully.")
            
            with Session(self._engine) as session:
                n_persons = session.query(Person).count()
                logger.info(f"  Registered persons: {n_persons}")
        except Exception as e:
            logger.error(f"Failed to open database: {e}")
            raise

    def add_person(self, name: str, embedding: np.ndarray) -> bool:
        if not self._ready: return False
        try:
            embedding_bytes = embedding.astype(np.float64).tobytes()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with Session(self._engine) as session:
                person = session.query(Person).filter_by(name=name).first()
                if person:
                    person.embedding = embedding_bytes
                else:
                    person = Person(name=name, embedding=embedding_bytes, added_at=now)
                    session.add(person)
                session.commit()
            logger.info(f"Person saved: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save person {name}: {e}")
            return False

    def get_person_by_name(self, name: str) -> Optional[Dict]:
        if not self._ready: return None
        try:
            with Session(self._engine) as session:
                person = session.query(Person).filter_by(name=name).first()
                if not person: return None
                return self._person_to_dict(person)
        except Exception as e:
            logger.error(f"Failed to get person {name}: {e}")
            return None

    def get_all_persons(self) -> List[Dict]:
        if not self._ready: return []
        try:
            with Session(self._engine) as session:
                persons = session.query(Person).order_by(Person.name).all()
                return [self._person_to_dict(p) for p in persons]
        except Exception as e:
            logger.error(f"Failed to get all persons: {e}")
            return []

    def update_last_seen(self, name: str):
        if not self._ready: return
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with Session(self._engine) as session:
                person = session.query(Person).filter_by(name=name).first()
                if person:
                    person.last_seen = now
                    person.seen_count += 1
                    session.commit()
        except Exception as e:
            logger.error(f"Failed to update last_seen for {name}: {e}")

    def delete_person(self, name: str) -> bool:
        if not self._ready: return False
        try:
            with Session(self._engine) as session:
                person = session.query(Person).filter_by(name=name).first()
                if person:
                    session.delete(person)
                    session.commit()
                    logger.info(f"Person deleted: {name}")
                    return True
                logger.warning(f"Person not found for deletion: {name}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete person {name}: {e}")
            return False

    def get_person_count(self) -> int:
        if not self._ready: return 0
        try:
            with Session(self._engine) as session:
                return session.query(Person).count()
        except Exception:
            return 0

    def _person_to_dict(self, person: Person) -> Dict:
        embedding = np.frombuffer(person.embedding, dtype=np.float64)
        return {
            "id": person.id,
            "name": person.name,
            "embedding": embedding,
            "added_at": person.added_at,
            "last_seen": person.last_seen,
            "seen_count": person.seen_count,
        }

    def set_preference(self, key: str, value: Any):
        if not self._ready: return
        try:
            value_str = str(value)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with Session(self._engine) as session:
                pref = session.query(Preference).filter_by(key=key).first()
                if pref:
                    pref.value = value_str
                    pref.updated_at = now
                else:
                    pref = Preference(key=key, value=value_str, updated_at=now)
                    session.add(pref)
                session.commit()
            logger.debug(f"Preference saved: {key} = {value_str}")
        except Exception as e:
            logger.error(f"Failed to save preference {key}: {e}")

    def get_preference(self, key: str, default: Any = None) -> Any:
        if not self._ready: return default
        try:
            with Session(self._engine) as session:
                pref = session.query(Preference).filter_by(key=key).first()
                return pref.value if pref else default
        except Exception as e:
            logger.error(f"Failed to get preference {key}: {e}")
            return default

    def delete_preference(self, key: str) -> bool:
        if not self._ready: return False
        try:
            with Session(self._engine) as session:
                pref = session.query(Preference).filter_by(key=key).first()
                if pref:
                    session.delete(pref)
                    session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete preference {key}: {e}")
            return False

    def log_event(self, event_type: str, details: Dict = None):
        if not self._ready: return
        try:
            details_str = json.dumps(details) if details else "{}"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with Session(self._engine) as session:
                event = EventLog(event_type=event_type, details=details_str, timestamp=now)
                session.add(event)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to log event {event_type}: {e}")

    def get_events(self, event_type: Optional[str] = None, limit: int = 50, days: int = 7) -> List[Dict]:
        if not self._ready: return []
        try:
            with Session(self._engine) as session:
                query = session.query(EventLog)
                if event_type:
                    query = query.filter_by(event_type=event_type)
                from datetime import timedelta
                cutoff_date_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
                query = query.filter(EventLog.timestamp >= cutoff_date_str)
                events = query.order_by(EventLog.timestamp.desc()).limit(limit).all()
                return [{"id": e.id, "event_type": e.event_type, "details": json.loads(e.details), "timestamp": e.timestamp} for e in events]
        except Exception as e:
            logger.error(f"Failed to get events: {e}")
            return []

    def clear_events(self, days_older_than: int = 30) -> int:
        if not self._ready: return 0
        try:
            from datetime import timedelta
            cutoff_date_str = (datetime.now() - timedelta(days=days_older_than)).strftime("%Y-%m-%d %H:%M:%S")
            with Session(self._engine) as session:
                deleted_count = session.query(EventLog).filter(EventLog.timestamp < cutoff_date_str).delete()
                session.commit()
                if deleted_count > 0:
                    logger.info(f"Cleared {deleted_count} old events.")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to clear events: {e}")
            return 0

    def get_stats(self) -> Dict:
        if not self._ready:
            return {"status": "uninitialized"}
        try:
            with Session(self._engine) as session:
                return {
                    "status": "ready",
                    "persons_count": session.query(Person).count(),
                    "preferences_count": session.query(Preference).count(),
                    "events_count": session.query(EventLog).count(),
                }
        except Exception as e:
            logger.error(f"Failed to get db stats: {e}")
            return {"status": "error"}

    def release(self):
        logger.info("Releasing database connection...")
        if self._engine:
            self._engine.dispose()
        self._ready = False

_db_instance: Optional[Database] = None

def init_database():
    global _db_instance
    if _db_instance is not None: return
    _db_instance = Database()
    _db_instance.init_db()

def get_db() -> Database:
    global _db_instance
    if _db_instance is None:
        init_database()
    return _db_instance