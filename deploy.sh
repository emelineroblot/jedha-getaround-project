#!/usr/bin/env bash
# Deploiement des deux Spaces Hugging Face DEPUIS ce depot Git.
#
# Pourquoi ce script : les Spaces avaient ete edites directement sur Hugging Face,
# et leur contenu avait diverge du code publie sur GitHub. En poussant toujours
# depuis ici, ce que voit un lecteur du depot est exactement ce qui tourne.
#
# Prerequis (une seule fois) :
#   hf auth login --add-to-git-credential
#   ... ou simplement laisser git demander le token au premier push
#       (identifiant = nom d'utilisateur HF, mot de passe = token avec droit write)
#
# Usage :
#   ./deploy.sh api
#   ./deploy.sh dashboard
#   ./deploy.sh all
#   ./deploy.sh all -y     # sans demande de confirmation
#
# /!\ CE SCRIPT FORCE LE PUSH.
# Les Spaces ont ete crees par televersement direct sur Hugging Face : leur
# historique git n'a aucun ancetre commun avec ce depot, et un push normal est
# donc rejete. Le Space etant une CIBLE DE DEPLOIEMENT et non une source de
# verite, on ecrase son historique par le contenu du sous-dossier correspondant.
# Sauvegarder l'etat en ligne avant le premier push :
#   git clone https://huggingface.co/spaces/<user>/<space> sauvegarde/<space>
#   cd sauvegarde/<space> && git lfs pull

set -euo pipefail

HF_USER="EmelineR"
API_SPACE="https://huggingface.co/spaces/${HF_USER}/jedha-getaround-project"
DASHBOARD_SPACE="https://huggingface.co/spaces/${HF_USER}/jedha-getaround-streamlit"

ASSUME_YES=0
for arg in "$@"; do
  [ "${arg}" = "-y" ] && ASSUME_YES=1
done

confirm() {
  [ "${ASSUME_YES}" -eq 1 ] && return 0
  printf 'Ecraser l historique du Space %s ? [o/N] ' "$1"
  read -r answer
  case "${answer}" in
    o|O|y|Y) return 0 ;;
    *) echo "Annule."; return 1 ;;
  esac
}

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

  if ! git diff --quiet HEAD -- "${prefix}"; then
    echo "Erreur : ${prefix} contient des modifications non commitees." >&2
    echo "         Committer avant de deployer, sinon le Space ne refletera pas le depot." >&2
    exit 1
  fi

  confirm "${name}" || return 0

  # On pousse UN SEUL commit, sans parent, contenant l'arbre du sous-dossier.
  #
  # Pourquoi pas `git subtree split` : il reconstruit tout l'historique du
  # sous-dossier, et le hook pre-receive de Hugging Face inspecte CHAQUE commit
  # pousse. Un ancien commit ou model.pkl n'etait pas encore suivi par LFS
  # suffit a faire rejeter le push entier, meme si le tip est correct.
  #
  # Un Space est une cible de deploiement, pas une source de verite : un commit
  # unique qui pointe vers le commit source est plus lisible et plus robuste.
  local tree source_sha commit
  tree="$(git rev-parse "HEAD:${prefix}")"
  source_sha="$(git rev-parse --short HEAD)"
  commit="$(git commit-tree "${tree}"       -m "Deploiement depuis ${prefix} (${source_sha})

Source : https://github.com/emelineroblot/jedha-getaround-project")"
  echo "    arbre ${tree} -> commit ${commit}"

  git push --force "${remote}" "${commit}:refs/heads/main"

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
    echo "Usage : $0 {api|dashboard|all} [-y]" >&2
    exit 1
    ;;
esac

echo
echo "Verifier le build : ${API_SPACE}  /  ${DASHBOARD_SPACE}"
