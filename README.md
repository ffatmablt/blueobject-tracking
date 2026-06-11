# Blue Object Tracking with OpenCV + PostgreSQL

Real-time blue object detection and tracking from a webcam feed, logged to PostgreSQL. Built from scratch — every problem I hit and how I solved it is documented below.

---

## What It Does

Detects blue objects using HSV color masking, tracks each object with OpenCV's CSRT tracker, and logs it once to PostgreSQL with label, position, and timestamp on first detection. Duplicate DB entries are prevented using a persistent saved_ids set.

---

## How It Works

Webcam → HSV Mask → Morphology → Contours → CSRT Tracker → PostgreSQL

HSV Mask isolates blue pixels (Hue 100–130). HSV is used instead of BGR because it's much less sensitive to lighting changes.

Morphology cleans up the mask — MORPH_CLOSE merges fragmented blobs, MORPH_OPEN removes noise.

CSRT Tracker is initialized on first color detection, then tracks by shape and texture, not color. This means it keeps working even when the object rotates.

PostgreSQL — each object is written to the DB only once, on first appearance.

---

## What Was Added Step by Step

v1 — Basic color detection
Simple HSV mask and contour detection. No tracking, no DB. Objects detected as separate blobs every frame.

v2 — PostgreSQL logging
Added psycopg2 connection and INSERT on each detection. Problem: same object written thousands of times per session.

v3 — ID management and duplicate fix
Added saved_ids = set() outside the loop — once an object is saved, it's never written again. Also fixed object_id being reset every frame.

v4 — CSRT hybrid tracker
Pure color tracking fails when the object rotates (blue surface shrinks). Solution: color mask handles first detection, CSRT takes over afterward. Every 30 frames, color mask re-aligns the CSRT to prevent drift.

---

## Setup

This project was built and tested on Windows using PyCharm.

1. Install PostgreSQL from https://www.postgresql.org/download/ and keep your password handy during setup.

2. Open pgAdmin, go to Servers → PostgreSQL → Databases, right click and create a new database named trackingdb.

3. Install the required packages in PyCharm terminal:

```
pip install opencv-contrib-python numpy psycopg2 python-dotenv
```

4. In main.py, update the database connection with your own credentials:

```python
conn = psycopg2.connect(
    dbname="trackingdb",
    user="postgres",
    password="your_password",
    host="localhost",
    port="5432",
)
```

5. Run main.py. The objects table is created automatically on first run. Press Q or ESC to quit.

To view detections in pgAdmin Query Tool:

```sql
SELECT * FROM objects ORDER BY detected_at DESC;
TRUNCATE TABLE objects RESTART IDENTITY;
```

---

## Problems and Fixes

Same object got a new ID every frame — object_id was inside the loop so it reset on every frame. Fixed by moving it outside and adding saved_ids = set(). Once an object is saved, it is never written again.

Object detected as many small fragments — kernel was too small (5x5) and area threshold too low (300 px). Fixed by using a 15x15 ellipse kernel with MORPH_CLOSE iterations=3 and raising the threshold to 1500 px.

Object lost when rotating or moving fast — pure color tracking fails when the blue surface shrinks on rotation. Fixed with a CSRT hybrid approach: color mask handles first detection, CSRT tracks by shape and texture afterward.

detected_at column not found — table was created before the column was added. Fixed with ALTER TABLE objects ADD COLUMN detected_at TIMESTAMP.

psql not recognized in terminal — PostgreSQL bin folder was not added to PATH during install. Used pgAdmin Query Tool instead.

TrackerCSRT_create not found — standard opencv-python does not include contrib modules. Fixed with pip uninstall opencv-python then pip install opencv-contrib-python.

---

