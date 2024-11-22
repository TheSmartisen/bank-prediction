# Livrable: Projet de Data Science

Ce projet contient un pipeline complet de traitement de données, d'entraînement de modèles, et d'évaluation pour résoudre un problème spécifique de machine learning.

## Structure du projet

- **data/**
  - `raw/`: Contient les données brutes.
    - `load_data.csv`: Données d'entrée initiales.
  - `processed/`: Contient les données après nettoyage et préparation.
    - `data_clean_for_training.csv`: Données prêtes pour l'entraînement.
    - `data_for_test.csv`: Données pour tester le modèle.
    - `data_without_missing.csv`: Données sans valeurs manquantes.

- **notebooks/**: 
  - `exploration.ipynb`: Analyse exploratoire des données.
  - `preprocessing.ipynb`: Étapes de préparation des données.
  - `training_evaluation_exportation.ipynb`: Entraînement, évaluation et exportation du modèle.

- **outputs/**:
  - `pickle/`: Contient les objets sérialisés.
    - `encoders_config.pkl`: Configuration des encodeurs.
    - `training_data.pkl`: Données d'entraînement sérialisées.
  - `plots/`: Contient les graphiques générés.
    - `exploration/`: Graphiques d'analyse exploratoire.
    - `preprocessing/`: Graphiques liés au prétraitement.
    - `training/`: Graphiques liés à l'entraînement du modèle.

- **scripts/**:
  - `data_preparation.py`: Script pour nettoyer et préparer les données.
  - `model_training.py`: Script pour entraîner et évaluer le modèle.

- **requirements.txt**: Liste des dépendances nécessaires pour exécuter le projet.

## Installation

1. Clonez le dépôt.
2. Installez les dépendances avec :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

1. Lancez les notebooks dans l'ordre pour explorer, préparer, et entraîner les données.
2. Utilisez les scripts pour automatiser le processus :
   - Préparation des données : 
     ```bash
     python scripts/data_preparation.py
     ```
   - Entraînement et évaluation du modèle :
     ```bash
     python scripts/model_training.py
     ```

## Résultats

Les résultats incluent des graphiques d'évaluation du modèle, des fichiers sérialisés pour réutiliser le pipeline, et des données préparées pour de nouvelles prédictions.

## Auteur

- **HEM Patrick** - Développeur et passionné.

## Remarques

Pour toute question ou contribution, merci de contacter le développeur principal.
