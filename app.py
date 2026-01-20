import streamlit as st
import cv2
import numpy as np
import tempfile
import os

st.set_page_config(page_title="Strobo-App Pro", layout="wide")

st.title("🏃‍♂️ High-Precision Stroboskop")
st.write("Verbesserte Objekterkennung durch Kontur-Analyse und Rauschfilter.")

# Sidebar mit erweiterten Reglern
with st.sidebar:
    st.header("⚙️ Präzisions-Einstellung")
    num_images = st.slider("Anzahl Bilder", 2, 60, 15)
    threshold_val = st.slider("Empfindlichkeit", 5, 100, 25)
    dilation_val = st.slider("Objekt-Stärke (Dilation)", 1, 10, 3, 
                             help="Erhöhen, wenn das Objekt löchrig erscheint.")
    blur_val = st.slider("Rauschunterdrückung", 1, 15, 5, step=2)

uploaded_file = st.file_uploader("Video auswählen", type=["mp4", "mov", "avi", "m4v"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames > 0:
        frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)
        time_interval = (frame_indices[1] - frame_indices[0]) / fps if len(frame_indices) > 1 else 0
        
        st.info(f"⏱️ Zeitabstand: {time_interval:.3f} s")

        if st.button("🚀 Stroboskop-Bild generieren", use_container_width=True):
            progress_bar = st.progress(0)
            
            # Hintergrund-Referenz (Erstes Bild)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
            ret, background = cap.read()
            
            if ret:
                # Wir arbeiten auf einer Kopie des Hintergrunds
                canvas = background.copy().astype(np.float32)
                bg_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(bg_gray, (blur_val, blur_val), 0)

                for i, idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: break

                    # 1. Bild vorbereiten
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_frame = cv2.GaussianBlur(gray_frame, (blur_val, blur_val), 0)

                    # 2. Differenz berechnen
                    diff = cv2.absdiff(bg_gray, gray_frame)
                    _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

                    # 3. Morphologische Verbesserung (Löcher stopfen)
                    kernel = np.ones((dilation_val, dilation_val), np.uint8)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Schließt Lücken
                    mask = cv2.dilate(mask, kernel, iterations=1) # Macht das Objekt "fetter"

                    # 4. Konturen finden und nur das Objekt übernehmen
                    # Wir nutzen die Maske, um die Pixel aus dem Original-Frame zu kopieren
                    mask_inv = cv2.bitwise_not(mask)
                    mask_3d = cv2.merge([mask, mask, mask]).astype(bool)
                    
                    # Nur dort, wo die Maske weiß ist, nehmen wir das neue Frame
                    canvas[mask_3d] = frame[mask_3d]

                    progress_bar.progress((i + 1) / len(frame_indices))

                # Ergebnis anzeigen
                result = np.clip(canvas, 0, 255).astype(np.uint8)
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="Optimiertes Tracking-Ergebnis", use_column_width=True)

                # Download
                _, buffer = cv2.imencode('.jpg', result)
                st.download_button("💾 Bild speichern", buffer.tobytes(), "strobe_pro.jpg", "image/jpeg", use_container_width=True)
            else:
                st.error("Fehler beim Lesen des Videos.")

    cap.release()
    os.unlink(tfile.name)