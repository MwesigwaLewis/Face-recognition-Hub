"""Local SQLite storage for multi-angle face embeddings."""

import os
import sqlite3
import cv2
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DATABASE = str(BASE_DIR / "data" / "database.db")


def get_connection():
    os.makedirs(BASE_DIR / "data", exist_ok=True)
    return sqlite3.connect(DATABASE)


def create_table():
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                thumbnail BLOB,
                DateCreated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                DateUpdated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        columns = {row[1] for row in cursor.execute("PRAGMA table_info(faces)").fetchall()}
        if "thumbnail" not in columns:
            cursor.execute("ALTER TABLE faces ADD COLUMN thumbnail BLOB")


def person_exists(name):
    with get_connection() as conn:
        return conn.execute("SELECT 1 FROM faces WHERE name = ?", (name,)).fetchone() is not None


def _pack_embeddings(embeddings):
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr.tobytes()


def _unpack_embeddings(blob):
    arr = np.frombuffer(blob, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    # InsightFace ArcFace embeddings are normally 512-dimensional. The
    # fallback keeps old single-vector records compatible.
    if arr.size % 512 == 0:
        return arr.reshape(-1, 512)
    return arr.reshape(1, -1)


def _pack_thumbnail(image, size=128, quality=72):
    if image is None:
        return None
    try:
        h, w = image.shape[:2]
        if not h or not w:
            return None
        side = min(h, w)
        x = max(0, (w - side) // 2)
        y = max(0, (h - side) // 2)
        crop = image[y:y + side, x:x + side]
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return encoded.tobytes() if ok else None
    except Exception:
        return None


def add_person(name, embedding, thumbnail=None):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO faces (name, embedding, thumbnail) VALUES (?, ?, ?)",
            (name, _pack_embeddings(embedding), _pack_thumbnail(thumbnail)),
        )


def get_person_record(name):
    """Return one stable person record, including all stored embeddings."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, name, embedding, thumbnail FROM faces WHERE name = ?",
            (name,),
        ).fetchone()
    if row is None:
        return None
    pid, person_name, embedding, thumbnail = row
    return {
        "id": int(pid),
        "name": person_name,
        "embeddings": _unpack_embeddings(embedding),
        "thumbnail": thumbnail,
    }


def update_person(name, embedding, thumbnail=None):
    with get_connection() as conn:
        if thumbnail is None:
            conn.execute(
                "UPDATE faces SET embedding = ?, DateUpdated = CURRENT_TIMESTAMP WHERE name = ?",
                (_pack_embeddings(embedding), name),
            )
        else:
            conn.execute(
                "UPDATE faces SET embedding = ?, thumbnail = ?, DateUpdated = CURRENT_TIMESTAMP WHERE name = ?",
                (_pack_embeddings(embedding), _pack_thumbnail(thumbnail), name),
            )


def delete_person(name):
    with get_connection() as conn:
        conn.execute("DELETE FROM faces WHERE name = ?", (name,))


def get_people():
    with get_connection() as conn:
        rows = conn.execute("SELECT name, embedding FROM faces ORDER BY name").fetchall()
    return [(name, _unpack_embeddings(embedding)) for name, embedding in rows]


def get_people_records():
    """Return stable DB ids, embeddings and compact UI thumbnails."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, name, embedding, thumbnail FROM faces ORDER BY name"
        ).fetchall()
    return [
        {
            "id": int(pid),
            "name": name,
            "embeddings": _unpack_embeddings(embedding),
            "thumbnail": thumbnail,
        }
        for pid, name, embedding, thumbnail in rows
    ]


def add_detection_log(camera, name, confidence, snapshot=None):
    """Persist one stabilized surveillance detection. Does not touch face records."""
    with get_connection() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera TEXT NOT NULL,
                name TEXT NOT NULL,
                confidence REAL NOT NULL,
                snapshot BLOB
            )"""
        )
        conn.execute(
            "INSERT INTO detection_logs (camera, name, confidence, snapshot) VALUES (?, ?, ?, ?)",
            (str(camera), str(name), float(confidence), snapshot),
        )

def get_detection_logs(limit=100):
    """Return newest surveillance detections without modifying existing face data."""
    with get_connection() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS detection_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                camera TEXT NOT NULL,
                name TEXT NOT NULL,
                confidence REAL NOT NULL,
                snapshot BLOB
            )"""
        )
        rows = conn.execute(
            "SELECT id, timestamp, camera, name, confidence, snapshot FROM detection_logs ORDER BY id DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "camera": r[2], "name": r[3], "confidence": float(r[4]), "snapshot": r[5]}
        for r in rows
    ]

def clear_detection_logs():
    with get_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS detection_logs")


create_table()
