# dekkr-analyze

Service d'analyse audio pour [DekkR](https://dekkr.app) — détection de tonalité, BPM et structure via [Essentia](https://essentia.upf.edu/).

## Licence

AGPL-3.0 — voir [LICENSE](LICENSE).

## API

```
POST /analyze
Authorization: Bearer {API_KEY}
Content-Type: multipart/form-data

file: <fichier audio>
```

Réponse :
```json
{
  "key": "8B",
  "key_confidence": 0.87,
  "bpm": 128.0,
  "bpm_confidence": 0.95,
  "cue_points": [...],
  "duration": 360.0,
  "analysis_version": 1
}
```

## Installation locale

```bash
pip install -r requirements.txt
API_KEY=votre_cle uvicorn main:app --reload
```

## Déploiement (Railway / Render)

1. Connecter ce repo GitHub
2. Ajouter la variable d'environnement `API_KEY`
3. Déployer — le Dockerfile est utilisé automatiquement
