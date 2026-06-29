import cv2
import time
from ultralytics import YOLO

print("--- Lancement du script ---")
print("Chargement des modèles YOLO...")

# On charge les DEUX cerveaux 
model_face = YOLO("yolov8n-face.pt") 
model_plate = YOLO("yolov8n-plate.pt") 

video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = "data/resultat.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Vidéo trouvée. Appuie sur la touche 'q' pour quitter.")

prev_time = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Fin de la vidéo.")
        break
        
    # ==========================================
    # 1. DÉTECTION ET FLOUTAGE DES VISAGES
    # ==========================================
    results_faces = model_face(frame, stream=True, verbose=False) 
    for r in results_faces:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            roi = frame[y1:y2, x1:x2] 
            try:
                # Flou très fort pour les visages
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
                frame[y1:y2, x1:x2] = blurred_roi
            except Exception:
                pass 

    # ==========================================
    # 2. DÉTECTION ET FLOUTAGE DES PLAQUES
    # ==========================================
    results_plates = model_plate(frame, stream=True, verbose=False) 
    for r in results_plates:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            roi = frame[y1:y2, x1:x2] 
            try:
                # Un flou un peu moins étalé suffit souvent pour les plaques
                blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0) 
                frame[y1:y2, x1:x2] = blurred_roi
            except Exception:
                pass 

    # ==========================================
    # AFFICHAGE ET EXPORT (Inchangé)
    # ==========================================
    current_time = time.time()
    fps_calc = 1 / (current_time - prev_time)
    prev_time = current_time
    
    cv2.putText(frame, f"FPS: {int(fps_calc)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    out.write(frame)

    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
    cv2.imshow("GDPR Anonymizer - Etape Multi-Model", resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Processus terminé !")