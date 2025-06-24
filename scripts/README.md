# Répertoire `scripts`

Ce dossier regroupe **tous les utilitaires CLI** du harness d'évaluation *Les Audits-Affaires*.

> ��️ **Langue** : tout est documenté en français pour être cohérent avec le reste du projet.

---

## 1. Script principal : `laal_pipeline.py`

| Fonction | Description |
|----------|-------------|
| `requests` (défaut) | Traite les requêtes en attente dans le dataset **legmlai/laal-requests** puis pousse les scores dans **legmlai/laal-results**. |
| `local <path>` | Upload des résultats déjà calculés depuis un répertoire local contenant `evaluation_summary.json`. |
| `clear-results` | Vide complètement la table des résultats (⚠️ irréversible). |

### 1.1 Variables d'environnement (détection automatique)

Le pipeline vérifie **dynamiquement** ce qui manque et vous demandera les valeurs à la volée ; cependant il est recommandé de tout exporter au préalable :

| Cas | Variables attendues |
|-----|---------------------|
| Tous les cas | `HF_TOKEN` (écriture/lecture datasets) |
| Modèle **OpenAI** | `OPENAI_API_KEY` |
| Modèle **Mistral** | `MISTRAL_API_KEY` |
| Modèle **Claude** | `ANTHROPIC_API_KEY` |
| Modèle **Gemini** | `GOOGLE_API_KEY` |
| Modèle **local** | `MODEL_ENDPOINT` |

Variables optionnelles :
- `HF_TOKEN_SUMMARY_DATASETS` : push des datasets résumés → *legmlai/les-audites-affaires-leadboard*.
- `HF_TOKEN_LEADERBOARD_RESULTS` : mise à jour du leaderboard.

### 1.2 Exemples rapides

```bash
# 1) Évaluer un modèle OpenAI et uploader les résultats
export HF_TOKEN=hf_xxx
export EXTERNAL_PROVIDER=openai
export EXTERNAL_MODEL=gpt-4o
export OPENAI_API_KEY=sk-...
python scripts/laal_pipeline.py requests

# 2) Même chose mais maximum 3 requêtes
python scripts/laal_pipeline.py requests --max 3

# 3) Simulation sans push (utile en local)
python scripts/laal_pipeline.py requests --dry-run

# 4) Upload d'un répertoire local de résultats
python scripts/laal_pipeline.py local ~/resultats/gpt_4o/
```

---

## 2. Autres utilitaires

| Script | Rôle principal |
|--------|----------------|
| `test_external_providers.py` | Vérifie la connectivité aux API externes (clé valide, quota, etc.). |
| `demo_external_providers.py` | Démonstration *offline* des formats attendus. |
| `example_external_evaluation.py` | Évaluation complète d'un petit échantillon pour prise en main. |
| `quick_upload.py` | Détection automatique du dernier dossier de résultats et upload. |
| `upload_results.py` | Upload manuel d'un fichier ou dossier précis. |
| `batch_evaluate_and_upload.py` | Ancien wrapper : traite en boucle les requêtes (désormais remplacé par `laal_pipeline.py`). |

---

## 3. Makefile shortcuts

```bash
make test-providers   # Lance test_external_providers.py
make demo-providers   # Lance demo_external_providers.py
```

---

## 4. FAQ rapide

**Q : Faut-il absolument renseigner toutes les variables ?**  
R : Non. Le pipeline détecte ce qu'il manque et vous le demandera. Néanmoins, en exécution non-interactive (par ex. via Cron ou CI), il est crucial d'exporter toutes les variables nécessaires.

**Q : Où sont stockées les requêtes ?**  
`legmlai/laal-requests` sur le Hub : chaque entrée contient `model_name`, `model_provider`, `request_status`, etc.

**Q : Comment voir le leaderboard ?**  
[LegML.ai – Les Audits-Affaires Leaderboard](https://huggingface.co/spaces/legmlai/les-audites-affaires-leadboard)

---

> ✨ *Bon benchmark !* 