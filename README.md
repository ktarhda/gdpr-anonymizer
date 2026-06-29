# 🛡️ GDPR Auto-Anonymizer : Real-Time Face & License Plate Blurring

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-yellow.svg)

## 📌 Présentation du projet
Dans le cadre de la conformité RGPD (Règlement Général sur la Protection des Données), l'anonymisation des données visuelles est une obligation légale pour la vidéosurveillance et la cartographie publique. 

Ce projet est un outil de **Computer Vision** performant permettant de détecter et de flouter automatiquement les visages et les plaques d'immatriculation sur des flux vidéos ou des images statiques.

### 🎥 Démonstration
![Demo du floutage](video.gif)

## 🚀 Fonctionnalités
- **Détection ultra-rapide** : Utilisation de modèles Deep Learning légers (YOLOv8) pour la détection.
- **Floutage dynamique** : Application d'un filtre Gaussien sur les zones d'intérêt (ROI) avec OpenCV.
- **Export vidéo** : Sauvegarde automatique du flux traité tout en conservant la qualité d'origine.
- **Mesure de performance** : Affichage des FPS en temps réel pour évaluer l'optimisation.

## 🛠️ Stack Technique
- **Langage :** Python
- **Computer Vision :** OpenCV (Traitement matriciel, Flou Gaussien, I/O Vidéo)
- **Deep Learning / Object Detection :** YOLOv8 (Ultralytics)

## ⚙️ Installation et Utilisation

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/](https://github.com/)[ton-pseudo]/gdpr-auto-anonymizer.git
   cd gdpr-auto-anonymizer