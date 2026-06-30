# 🛡️ GDPR Auto-Anonymizer : Real-Time Face & License Plate Blurring

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-yellow.svg)

## 📌 Présentation du projet
Dans le cadre de la conformité RGPD (Règlement Général sur la Protection des Données), l'anonymisation des données visuelles est une obligation légale. 

Ce projet est un pipeline de **Computer Vision** optimisé permettant de détecter et de flouter automatiquement les visages et les plaques d'immatriculation sur des flux vidéos, en gérant les problématiques de Tracking et d'optimisation CPU.

### 🎥 Démonstration
![Demo du floutage]
<video src="data/resultat.mp4" controls="controls" width="100%">
  Ton navigateur ne supporte pas la balise vidéo.
</video>
## 🚀 Fonctionnalités Avancées
- **Architecture Multi-Modèles** : Inférence simultanée de deux modèles YOLOv8 spécialisés (Visages + Plaques).
- **Tracking d'Objets (ByteTrack)** : Suivi temporel des objets pour éliminer le "flickering" (scintillement) sur les vidéos dynamiques.
- **Padding Spatial Dynamique** : Élargissement intelligent des zones de floutage (ROI) pour compenser la latence physique liée aux véhicules rapides.
- **Optimisation CPU (Downscaling)** : Réduction matricielle du flux d'entrée (`cv2.resize`) permettant d'augmenter drastiquement les FPS sur des machines dépourvues de GPU puissant.
- **Export vidéo automatisé** : Recomposition du flux traité avec le codec `mp4v` adapté aux nouvelles dimensions.

## 🛠️ Stack Technique
- **Langage :** Python
- **Computer Vision :** OpenCV (Traitement matriciel, Flou Gaussien, I/O Vidéo)
- **Deep Learning / Object Detection :** YOLOv8 (Ultralytics)

## ⚙️ Installation et Utilisation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/](https://github.com/)[ton-pseudo]/gdpr-auto-anonymizer.git
   cd gdpr-auto-anonymizer