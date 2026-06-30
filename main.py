import cv2
import time
from ultralytics import YOLO

print("--- Lancement du script ---")
print("Chargement des modèles YOLO...")

# On charge les DEUX cerveaux 
model_face = YOLO("yolov8n-face.pt") 
model_plate = YOLO("license_plate_detector.pt")

video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

# --- CONFIGURATION DE L'EXPORT  ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# On calcule les dimensions réduites pour l'exporteur (50%)
new_width = int(frame_width * 0.5)
new_height = int(frame_height * 0.5)

output_path = "data/resultat.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# On donne les NOUVELLES dimensions à l'exporteur
out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
# --------------------------------------------

print("Vidéo trouvée. Appuie sur la touche 'q' pour quitter.")

prev_time = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Fin de la vidéo.")
        break
        
    # --- NOUVELLE ASTUCE D'OPTIMISATION ---
    # On réduit la taille de l'image de 50% dès son arrivée.
    # fx=0.5 et fy=0.5 signifient "50% de la largeur" et "50% de la hauteur".
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
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
    # 2. DÉTECTION ET FLOUTAGE DES PLAQUES (AVEC PADDING)
    # ==========================================
    results_plates = model_plate.track(frame, persist=True, stream=True, verbose=False, conf=0.1) 
    
    # On définit notre marge de sécurité en pixels
    padding = 3 
    
    for r in results_plates:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # --- L'ASTUCE DU PADDING ---
            # On agrandit la boîte, mais on utilise max() et min() pour s'assurer 
            # que la boîte agrandie ne sorte pas de l'écran (ce qui ferait planter OpenCV)
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(frame.shape[1], x2 + padding)
            y2 = min(frame.shape[0], y2 + padding)
            
            roi = frame[y1:y2, x1:x2] 
            try:
                # Flou fort pour bien couvrir la zone élargie
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