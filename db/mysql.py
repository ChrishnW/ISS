import mysql.connector
from datetime import datetime

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="pass",
    database="cctv"
)

def save_detection(person_id, camera_id):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO detections (person_id, camera_id, detected_at) VALUES (%s,%s,%s)",
        (person_id, camera_id, datetime.now())
    )
    conn.commit()
