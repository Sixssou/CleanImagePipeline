# Clean Image Pipeline

## 🎯 Features

- **Watermark Removal**: Automatiquement supprimer les filigranes des images avec l'IA
- **Background Removal**: Extraire les objets de leur arrière-plan avec transparence
- **Image Enhancement**: Optimiser les images pour l'e-commerce (résolution, taille, qualité)
- **Batch Processing**: Traiter plusieurs images simultanément
- **Git Automation**: Automatiser les commits Git avec GPT-3.5
- **Format Conversion**: Convertir entre différents formats d'image (PNG, JPEG, WebP)

## 🛠 Prérequis

- Python 3.12+
- Conda (pour la gestion de l'environnement)
- Clés API requises :
  - OpenAI API key (pour git_automator)
  - HuggingFace API token (pour florence_vision_client)

## 💻 Installation

1. Créer et activer l'environnement conda :
```bash
conda create -n clean_image python=3.12
conda activate clean_image
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📝 Configuration

1. Configurer les variables d'environnement :
```bash
export OPENAI_API_GIT_AUTOMATOR_KEY='votre-clé-api-openai'
export HUGGINGFACE_TOKEN='votre-token-huggingface'
```

2. Vérifier que les variables sont bien définies :
```bash
echo $OPENAI_API_GIT_AUTOMATOR_KEY
echo $HUGGINGFACE_TOKEN
```

## 📁 Structure du Projet

```
CleanImagePipeline/
├── src/
│   └── clients/
│       └── florence_vision_client.py   # Client pour Florence Vision
├── git_automator.py                    # Automatisation des commits
├── requirements.txt                    # Dépendances du projet
└── README.md                          # Documentation
```

## 🚀 Utilisation

### Git Automator

```bash
python git_automator.py
```
Cet outil va :
- Ajouter les modifications (`git add .`)
- Générer un message de commit avec GPT
- Commiter les changements
- Pousser vers le dépôt distant

### Florence Vision Client

```python
from src.clients.florence_vision_client import FlorenceVisionClient

client = FlorenceVisionClient(
    hf_token='votre-token',
    space_url='url-du-modele'
)
result = client.analyze_image('chemin/vers/image.jpg')
```

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité
3. Committez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier [LICENSE.md](LICENSE.md) pour plus de détails.