# blueobject-tracking
blue object-tracking with OpenCV


Bu proje, Python ve OpenCV kullanarak **mavi nesne takibi** yapmayı öğretir.  
Ben bu projeyi sıfırdan öğrenirken adım adım geliştirdim — aynı yerde takılanlara yol göstermesi için detaylı açıklamalar ekledim.

---

## 🎯 Projenin Amacı
Kameradan gelen görüntüde **belirli bir renk (mavi)** nesneyi tespit edip takip etmek.  
Bu sayede hem OpenCV’nin temel mantığını hem de Python’da görüntü işleme adımlarını öğreniyoruz.

---

## 🧠 Neden HSV Kullanıyoruz?
OpenCV’de renkleri algılamak için **HSV (Hue, Saturation, Value)** formatı kullanılır.  
- **Hue (H)** → Renk tonu (örneğin kırmızı, mavi, yeşil).  
- **Saturation (S)** → Rengin canlılığı.  
- **Value (V)** → Parlaklık.  

RGB yerine HSV kullanmamızın nedeni, **ışık değişimlerinden daha az etkilenmesi**.  
Yani ortam biraz karanlık olsa bile mavi tonları daha kararlı şekilde algılanır.

---
cv2.VideoCapture(0) → Kamerayı açar.

cv2.cvtColor(..., cv2.COLOR_BGR2HSV) → Görüntüyü HSV’ye çevirir.

cv2.inRange(...) → Belirlenen aralıktaki mavi pikselleri seçer.

cv2.findContours(...) → Maskedeki beyaz bölgelerin kenarlarını bulur.

cv2.rectangle(...) → Nesnenin etrafına kutu çizer.

Sık Karşılaşılan Sorunlar
Kamera açılmıyor: cv2.VideoCapture(0) yerine 1 veya 2 deneyebilirsin.

Nesne sadece yakında algılanıyor: area > 1000 filtresini küçült (300 gibi).

Mavi ton algılanmıyor: HSV aralığını genişlet (lower_blue = [80, 80, 50] gibi).
