# Scripts de test pour l'inpainting

Ce dossier contient des scripts pour tester la fonctionnalité d'inpainting du client `WatermakRemovalClient`.

## Fichiers disponibles

1. **test_inpaint.py** - Script complet pour tester la fonction `inpaint` avec des fichiers d'image et de masque.
2. **test_inpaint_simple.py** - Version simplifiée du script de test.
3. **create_test_mask.py** - Outil pour générer des masques de test à partir d'une image.

## Utilisation

### Création d'un masque de test

```bash
# Depuis la racine du projet
python -m src.clients.test.create_test_mask <chemin_image>

# Options disponibles:
# --rectangle x1 y1 x2 y2 : Créer un masque rectangulaire avec les coordonnées spécifiées
# --circle x y radius : Créer un masque circulaire avec le centre et le rayon spécifiés
# --auto : Créer un masque automatique au centre de l'image (par défaut)
# --output <chemin> : Spécifier un chemin de sortie personnalisé
```

### Test simple de l'inpainting

```bash
# Depuis la racine du projet
python -m src.clients.test.test_inpaint_simple <chemin_image> <chemin_masque>
```

### Test complet de l'inpainting

```bash
# Depuis la racine du projet
python -m src.clients.test.test_inpaint <chemin_image> <chemin_masque>
```

## Workflow de test recommandé

1. Téléchargez ou préparez une image que vous souhaitez tester
2. Créez un masque pour cette image avec `create_test_mask.py`
3. Testez l'inpainting avec `test_inpaint_simple.py`
4. Vérifiez le résultat dans le dossier "output" à la racine du projet

Exemple:
```bash
# Depuis la racine du projet
python -m src.clients.test.create_test_mask chemin/vers/mon_image.jpg
python -m src.clients.test.test_inpaint_simple chemin/vers/mon_image.jpg output/mon_image_mask.jpg
```

## Notes importantes

- Les résultats sont sauvegardés dans le dossier `output` à la racine du projet.
- Les masques doivent être en niveaux de gris (mode 'L' en PIL). Si ce n'est pas le cas, les scripts les convertiront automatiquement.
- Assurez-vous que les variables d'environnement `HF_TOKEN` et `HF_SPACE_WATERMAK_REMOVAL` sont définies dans votre fichier `.env`. 