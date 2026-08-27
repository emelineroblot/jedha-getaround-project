#!/usr/bin/env bash
# Deploiement des deux Spaces Hugging Face DEPUIS ce depot Git.
#
# Pourquoi ce script : les Spaces avaient ete edites directement sur Hugging Face,
# et leur contenu avait divergé du code publie sur GitHub. En poussant toujours
# depuis ici, ce que voit un lecteur du depot est exactement ce qui tourne.
#
# Prerequis (une seule fois) :
#   pip install --upgrade huggingface_hub
#   hf auth login
#
# Usage :
#   ./deploy.sh api
#   ./deploy.sh dashboard
#   ./deploy.sh all

set -euo pipefail

HF_USER="EmelineR"
API_SPACE="https://huggingface.co/spaces/${HF_USER}/jedha-getaround-project"
DASHBOARD_SPACE="https://huggingface.co/spaces/${HF_USER}/jedha-getaround-streamlit"

deploy() {
  local prefix="$1" remote="$2" name="$3"

  echo "==> Deploiement de ${name} depuis ${prefix}/"

  if [ ! -d "${prefix}" ]; then
    echo "Erreur : ${prefix} introuvable." >&2
    exit 1
  fi

  if [ "${prefix}" = "deployment/api" ] && [ ! -f "deployment/api/model.pkl" ]; then
    echo "Erreur : model.pkl absent. Lancer d'abord : python src/train_model.py" >&2
    exit 1
  fi

  # git subtree pousse le sous-dossier a la racine du depot distant, ce qui est
  # exactement la structure attendue par un Space Hugging Face.
  git subtree push --prefix="${prefix}" "${remote}" main

  echo "==> ${name} deploye : ${remote}"
}

case "${1:-all}" in
  api)       deploy "deployment/api"       "${API_SPACE}"       "API" ;;
  dashboard) deploy "deployment/dashboard" "${DASHBOARD_SPACE}" "Dashboard" ;;
  all)
    deploy "deployment/api"       "${API_SPACE}"       "API"
    deploy "deployment/dashboard" "${DASHBOARD_SPACE}" "Dashboard"
    ;;
  *)
    echo "Usage : $0 {api|dashboard|all}" >&2
    exit 1
    ;;
esac

echo
echo "Verifier le build : ${API_SPACE}  /  ${DASHBOARD_SPACE}"
