import cv2
import time
from ultralytics import YOLO

print("--- Lancement du script ---")
print("Chargement du modèle YOLO...")

model = YOLO("yolov8n.pt") 

video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo. Vérifie le chemin !")
    exit()

#  Configuration de l'export vidéo 
# On récupère les propriétés de la vidéo d'origine
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Création du fichier de sortie 'resultat.mp4' dans le dossier data
output_path = "data/resultat.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec vidéo
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print(f"La vidéo anonymisée sera sauvegardée ici : {output_path}")
print("Appuie sur la touche 'q' pour quitter la vidéo.")

# Initialisation du temps pour le calcul des FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Fin de la vidéo.")
        break
        
    results = model(frame, stream=True) 
    
    for r in results:
        boxes = r.boxes 
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            roi = frame[y1:y2, x1:x2] 
            
            try:
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
                frame[y1:y2, x1:x2] = blurred_roi
            except Exception as e:
                pass 

    #  Calcul et affichage des FPS
    current_time = time.time()
    fps_calc = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # On écrit le texte des FPS sur l'image (en vert)
    cv2.putText(frame, f"FPS: {int(fps_calc)}", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Sauvegarde de l'image floutée dans le nouveau fichier
    out.write(frame)

    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
    cv2.imshow("GDPR Anonymizer - Etape Export", resized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
print("Processus terminé. Vérifie le dossier 'data' !")