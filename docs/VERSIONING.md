# 🏷️ Versioning automatique

Ce projet utilise un système de versionnage automatique basé sur les **Conventional Commits** et le **Semantic Versioning**.

## Comment ça fonctionne

### 🔄 Processus automatique

1. **Push sur `main`** → Déclenche l'analyse des commits
2. **Analyse des messages** → Détermine le type de version (patch/minor/major)  
3. **Bump automatique** → Met à jour `pyproject.toml`
4. **Création du tag** → Tag Git + Release GitHub
5. **Publication PyPI** → Package disponible automatiquement

### 📝 Format des commits (Conventional Commits)

| Type de commit | Exemple | Impact version |
|----------------|---------|----------------|
| `feat:` | `feat: add new evaluator provider` | **Minor** (1.2.0 → 1.3.0) |
| `fix:` | `fix: resolve API timeout issue` | **Patch** (1.2.0 → 1.2.1) |
| `docs:` | `docs: update README examples` | **Patch** (1.2.0 → 1.2.1) |
| `style:` | `style: format code with black` | **Patch** (1.2.0 → 1.2.1) |
| `refactor:` | `refactor: optimize client initialization` | **Patch** (1.2.0 → 1.2.1) |
| `perf:` | `perf: improve evaluation speed` | **Patch** (1.2.0 → 1.2.1) |
| `test:` | `test: add integration tests` | **Patch** (1.2.0 → 1.2.1) |
| `chore:` | `chore: update dependencies` | **Aucun** |
| **Breaking!** | `feat!: change API interface` | **Major** (1.2.0 → 2.0.0) |
| **BREAKING CHANGE** | Corps du commit contient `BREAKING CHANGE:` | **Major** (1.2.0 → 2.0.0) |

### 🎯 Exemples de commits

```bash
# Nouvelle fonctionnalité (minor version)
git commit -m "feat: add support for Claude evaluator"

# Correction de bug (patch version)  
git commit -m "fix: handle timeout errors gracefully"

# Documentation (patch version)
git commit -m "docs: add evaluator configuration examples"

# Breaking change (major version)
git commit -m "feat!: redesign configuration API"

# Ou avec BREAKING CHANGE dans le corps
git commit -m "feat: update API interface

BREAKING CHANGE: The evaluator configuration now requires provider field"
```

## 🚀 Déclenchement manuel

Vous pouvez aussi déclencher une release manuellement :

1. **Via GitHub Actions** → Onglet "Actions" → "Auto Version & Release" → "Run workflow"
2. **Choisir le type** → patch/minor/major
3. **Confirmer** → La release se fait automatiquement

## 📊 Suivi des versions

- **Tags Git** : `v1.2.3`
- **GitHub Releases** : Changelog automatique
- **PyPI** : Package publié automatiquement
- **CHANGELOG.md** : Mis à jour automatiquement

## 🔧 Configuration

Le système est configuré dans `.github/workflows/auto-version.yml` et analyse :

- **Messages de commits** depuis le dernier tag
- **Format Conventional Commits** 
- **Mots-clés BREAKING CHANGE**
- **Déclenchement manuel** via workflow_dispatch

## 📈 Avantages

✅ **Pas d'oubli** - Versions automatiques  
✅ **Cohérence** - Suit Semantic Versioning  
✅ **Traçabilité** - Changelog automatique  
✅ **Rapidité** - Publication immédiate  
✅ **Fiabilité** - Tests avant publication  

## ⚠️ Bonnes pratiques

1. **Utilisez des commits conventionnels** pour un versionnage précis
2. **Testez localement** avant de pusher sur main
3. **Groupez les changements** en commits logiques
4. **Documentez les breaking changes** clairement
5. **Vérifiez les releases** sur GitHub après publication 