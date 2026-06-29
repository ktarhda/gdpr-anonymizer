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

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Fin de la vidéo.")
        break
        
    # 2. Inférence : YOLO analyse l'image
    # stream=True optimise la mémoire pour les vidéos
    results = model(frame, stream=True) 
    
    # 3. Dessiner les résultats sur l'image
    for r in results:
        # La méthode .plot() de YOLO dessine automatiquement les boîtes de détection
        annotated_frame = r.plot()
        
    # Afficher la vidéo avec les détections
    cv2.imshow("GDPR Anonymizer - Etape Detection", annotated_frame)
    
    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()