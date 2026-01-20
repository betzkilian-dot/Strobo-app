import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Seite konfigurieren - Optimiert für mobile Browser
st.set_page_config(
    page_title="Strobo-App iPad", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("📸 Stroboskop-Generator")
st.write("Wähle ein Video aus deiner Mediathek oder nimm eines auf.")

# Sidebar für die Steuerung - gut erreichbar auf dem iPad
with st.sidebar:
    st.header("⚙️ Einstellungen")
    num_images = st.slider("Anzahl Bilder", 2, 60, 15)
    threshold = st.slider("Empfindlichkeit", 5, 100, 30)
    
    st.divider()
    st.info("💡 Tipp: Nutze ein Stativ für beste Ergebnisse!")

# Video Upload - Das iPad bietet hier automatisch 'Fotomediathek' oder 'Datei wählen' an
uploaded_file = st.file_uploader("Video auswählen", type=["mp4", "mov", "avi", "m4v"])

if uploaded_file is not None:
    # Temporäre Datei erstellen
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames > 0:
        # Zeitberechnung
        frame_indices = np.linspace(0, total_frames - 1, num_images, dtype=int)
        step_frames = frame_indices[1] - frame_indices[0] if len(frame_indices) > 1 else 0
        time_interval = step_frames / fps
        
        # Anzeige der Daten im Hauptbereich
        col1, col2 = st.columns(2)
        col1.metric("Bilder", num_images)
        col2.metric("Abstand (s)", f"{time_interval:.3f}")

        if st.button("🚀 Stroboskop-Bild erstellen", use_container_width=True):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Hintergrund-Referenz
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
            ret, first_frame = cap.read()
            
            if ret:
                background = first_frame.copy()
                result_image = first_frame.astype(np.float32)

                for i, idx in enumerate(frame_indices):
                    status_text.text(f"Verarbeite Bild {i+1} von {num_images}...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if not ret: break

                    # Differenz-Analyse
                    diff = cv2.absdiff(background, frame)
                    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
                    mask = cv2.medianBlur(mask, 5) # Rauschunterdrückung
                    
                    mask_3d = cv2.merge([mask, mask, mask]).astype(bool)
                    result_image[mask_3d] = frame[mask_3d]

                    progress_bar.progress((i + 1) / len(frame_indices))

                # Fertiges Bild anzeigen
                final_res = np.clip(result_image, 0, 255).astype(np.uint8)
                final_res_rgb = cv2.cvtColor(final_res, cv2.COLOR_BGR2RGB)
                
                st.image(final_res_rgb, caption="Fertiges Stroboskop-Bild", use_column_width=True)

                # Download für iPad-Dateien-App
                res_bgr = cv2.cvtColor(final_res, cv2.COLOR_RGB2BGR)
                _, buffer = cv2.imencode('.jpg', res_bgr)
                st.download_button(
                    label="💾 Bild auf iPad speichern",
                    data=buffer.tobytes(),
                    file_name="stroboskop_ergebnis.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
            else:
                st.error("Video konnte nicht gelesen werden.")

    cap.release()
    tfile.close()
    os.unlink(tfile.name)
