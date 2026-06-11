import cv2
import numpy as np
import psycopg2
from datetime import datetime

# ─── VERİTABANI BAĞLANTISI ───────────────────────────────────────────────────
conn = psycopg2.connect(
    dbname="trackingdb",
    user="postgres",
    password="Admin1234",
    host="localhost",
    port="5432",
)
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS objects (
        id          SERIAL PRIMARY KEY,
        label       TEXT,
        color       TEXT,
        position_x  INT,
        position_y  INT,
        detected_at TIMESTAMP
    )
""")
conn.commit()


# ─── CSRT TABANLI TRACKER ────────────────────────────────────────────────────
class HybridTracker:
    """
    Renk maskesi  → nesneyi İLK KEZ tespit eder
    CSRT tracker  → sonrasında nesneyi TAKİP eder (renk değişse bile)

    Nesne döndüğünde, hızlı hareket ettiğinde veya kısmen kapandığında
    CSRT şekil/doku bilgisiyle takibi sürdürür.
    """

    def __init__(self, redetect_interval=30, min_area=1500, iou_threshold=0.15):
        self.next_id           = 1
        self.objects           = {}
        # objects[tid] = {
        #   "csrt":    cv2.TrackerCSRT,
        #   "bbox":    (x, y, w, h),
        #   "center":  (cx, cy),
        #   "missing": int,
        #   "label":   str,
        # }
        self.saved_ids         = set()   # program boyunca kalıcı
        self.redetect_interval = redetect_interval  # kaç frame'de bir renk maskesiyle yeniden eşleştir
        self.frame_count       = 0
        self.min_area          = min_area
        self.iou_threshold     = iou_threshold

    # ── Yardımcı ─────────────────────────────────────────────────────────────
    @staticmethod
    def _iou(b1, b2):
        """İki bbox arasındaki IoU (örtüşme oranı)."""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        inter = ix * iy
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0

    @staticmethod
    def _bbox_center(bbox):
        x, y, w, h = bbox
        return x + w // 2, y + h // 2

    # ── Ana güncelleme ────────────────────────────────────────────────────────
    def update(self, frame, color_bboxes):
        """
        frame        : BGR kamera karesi
        color_bboxes : renk maskesinden gelen [(x,y,w,h), ...] listesi
        Döner        : [(cx, cy, label, tid, is_new), ...]
        """
        self.frame_count += 1
        results = []

        # 1) Mevcut CSRT tracker'ları ilerlet
        to_delete = []
        for tid, obj in self.objects.items():
            ok, bbox = obj["csrt"].update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                # Ekran dışına çıktıysa sil
                fh, fw = frame.shape[:2]
                if x < 0 or y < 0 or x+w > fw or y+h > fh:
                    to_delete.append(tid)
                    continue
                obj["bbox"]    = (x, y, w, h)
                obj["center"]  = self._bbox_center((x, y, w, h))
                obj["missing"] = 0
            else:
                obj["missing"] += 1
                if obj["missing"] > 90:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.objects[tid]

        # 2) Renk maskesi tespitlerini mevcut nesnelerle eşleştir
        matched_color = set()

        for tid, obj in self.objects.items():
            best_iou, best_idx = 0, -1
            for i, cb in enumerate(color_bboxes):
                if i in matched_color:
                    continue
                iou = self._iou(obj["bbox"], cb)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            if best_iou >= self.iou_threshold and best_idx >= 0:
                matched_color.add(best_idx)
                # CSRT'yi renk tespiti üzerine yeniden başlat (drift düzelt)
                if self.frame_count % self.redetect_interval == 0:
                    obj["csrt"] = cv2.TrackerCSRT_create()
                    obj["csrt"].init(frame, color_bboxes[best_idx])

        # 3) Eşleşmeyen renk tespitleri → yeni nesne
        for i, (x, y, w, h) in enumerate(color_bboxes):
            if i in matched_color:
                continue
            # Mevcut nesnelerle merkez mesafesine göre kontrol (çakışmayı önle)
            cx, cy = x + w//2, y + h//2
            too_close = False
            for obj in self.objects.values():
                ox, oy = obj["center"]
                if np.hypot(cx - ox, cy - oy) < 150:
                    too_close = True
                    break
            if too_close:
                continue

            tid   = self.next_id
            self.next_id += 1
            csrt  = cv2.TrackerCSRT_create()
            csrt.init(frame, (x, y, w, h))
            self.objects[tid] = {
                "csrt":    csrt,
                "bbox":    (x, y, w, h),
                "center":  (cx, cy),
                "missing": 0,
                "label":   f"blue_object_{tid}",
            }

        # 4) Sonuçları derle
        for tid, obj in self.objects.items():
            cx, cy  = obj["center"]
            label   = obj["label"]
            is_new  = tid not in self.saved_ids
            results.append((cx, cy, label, tid, is_new))

        return results

    def mark_saved(self, tid):
        self.saved_ids.add(tid)

    def is_new(self, tid):
        return tid not in self.saved_ids


# ─── RENK TESPİTİ ────────────────────────────────────────────────────────────
kernel_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,  7))

def detect_blue(frame, min_area=1500):
    """Renk maskesiyle mavi bbox listesi döndür: [(x,y,w,h), ...]"""
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([100, 80, 40]), np.array([130, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_close, iterations=1)
    mask = cv2.dilate(mask, kernel_dilate, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        raw.append(cv2.boundingRect(cnt))

    # Yakın bbox'ları birleştir (NMS)
    used, merged = [False]*len(raw), []
    for i, (x1,y1,w1,h1) in enumerate(raw):
        if used[i]: continue
        used[i] = True
        grp = [(x1,y1,w1,h1)]
        cx1, cy1 = x1+w1//2, y1+h1//2
        for j, (x2,y2,w2,h2) in enumerate(raw):
            if used[j]: continue
            if np.hypot(cx1-(x2+w2//2), cy1-(y2+h2//2)) < 80:
                grp.append((x2,y2,w2,h2)); used[j] = True
        bx  = min(g[0] for g in grp)
        by  = min(g[1] for g in grp)
        bx2 = max(g[0]+g[2] for g in grp)
        by2 = max(g[1]+g[3] for g in grp)
        merged.append((bx, by, bx2-bx, by2-by))

    return merged, mask


# ─── ANA DÖNGÜ ───────────────────────────────────────────────────────────────
cap     = cv2.VideoCapture(0)
tracker = HybridTracker(redetect_interval=30, min_area=1500)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera akışı alınamadı.")
        break

    color_bboxes, mask = detect_blue(frame)
    results = tracker.update(frame, color_bboxes)

    for (cx, cy, label, tid, is_new) in results:
        obj  = tracker.objects.get(tid)
        if obj is None:
            continue
        x, y, w, h = obj["bbox"]

        # ── İlk görüldüğünde DB'ye yaz ───────────────────────────────────────
        if is_new:
            now = datetime.now()
            cur.execute(
                "INSERT INTO objects (label, color, position_x, position_y, detected_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (label, "blue", cx, cy, now),
            )
            conn.commit()
            tracker.mark_saved(tid)
            print(f"[{now.strftime('%H:%M:%S')}] Kaydedildi → {label}  konum=({cx},{cy})")

        # ── Görsel ───────────────────────────────────────────────────────────
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 80, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask",  mask)

    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
        break

# ─── TEMİZLİK ────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
cur.close()
conn.close()
