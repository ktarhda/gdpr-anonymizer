import cv2
from ultralytics import YOLO

# 1. Charger le modèle YOLO (le fichier yolov8n.pt sera téléchargé automatiquement)
print("Chargement du modèle YOLO...")
model = YOLO("yolov8n.pt") 

# Chemin vers la vidéo
video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la vidéo.")
    exit()

print("Appuie sur la touche 'q' pour quitter la vidéo.")

# Boucle de lecture de la vidéo
while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Fin de la vidéo.")
        break
        
    # YOLO analyse l'image
    results = model(frame, stream=True) 
    
    # Logique de floutage
    for r in results:
        boxes = r.boxes # Récupère toutes les boîtes détectées
        
        for box in boxes:
            # 1. Récupérer les coordonnées (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convertir en entiers (pixels)
            
            # 2. Extraire la zone à flouter (ROI)
            # Avec  OpenCV il y'a [y_début:y_fin, x_début:x_fin]
            roi = frame[y1:y2, x1:x2] 
            
            # 3. Appliquer le flou (Les chiffres (99, 99) déterminent la force du flou, ils doivent être impairs)
            # On utilise un try/except pour éviter un crash si la boîte sort de l'écran
            try:
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 0)
                # 4. Remplacer la zone d'origine par la zone floutée
                frame[y1:y2, x1:x2] = blurred_roi
            except Exception as e:
                pass # Si erreur sur les bords de l'image, on ignore

    # --- Redimensionnement pour l'écran (On garde ce qu'on a fait avant) ---
    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
    # Affichage de la vidéo (Note : on affiche 'frame', plus 'annotated_frame')
    cv2.imshow("GDPR Anonymizer - Etape Floutage", resized_frame)
    
    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()