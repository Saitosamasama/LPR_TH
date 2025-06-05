import sqlite3
from datetime import datetime

DB_PATH = 'vehicle.db'

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            province TEXT,
            driver_name TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plate TEXT,
            province TEXT,
            snapshot TEXT,
            registered INTEGER
        )
    ''')
    conn.commit()
    return conn

def register_vehicle(conn, plate, province=None, driver_name=None):
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO registered_vehicles (plate, province, driver_name) VALUES (?, ?, ?)',
        (plate, province, driver_name)
    )
    conn.commit()


def list_vehicles(conn):
    cursor = conn.cursor()
    cursor.execute('SELECT plate, province, driver_name FROM registered_vehicles')
    return cursor.fetchall()


def is_registered(conn, plate):
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM registered_vehicles WHERE plate=?', (plate,))
    return cursor.fetchone() is not None


def log_detection(conn, plate, province, snapshot, registered):
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO detections (timestamp, plate, province, snapshot, registered) VALUES (?, ?, ?, ?, ?)',
        (datetime.now().isoformat(), plate, province, snapshot, int(registered))
    )
    conn.commit()


def list_detections(conn, limit=100):
    cursor = conn.cursor()
    cursor.execute('SELECT timestamp, plate, province, registered FROM detections ORDER BY id DESC LIMIT ?', (limit,))
    return cursor.fetchall()
