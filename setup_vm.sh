#!/usr/bin/env bash
# One-shot setup and run for the BITS Virtual Lab (Rocky Linux).
# The lab resets installed packages between sessions, so this reinstalls what
# is needed and then trains the models. Results go to ~/ml_assignment_run so
# the files you committed to GitHub stay untouched.
#
#   bash setup_vm.sh          install packages, train, print metrics
#   bash setup_vm.sh app      the same, then start the Streamlit app locally

set -e
cd "$(dirname "$0")"

if ! command -v git >/dev/null 2>&1; then
    echo "git is not installed. Run:  sudo dnf install git -y"
fi

python3 -m pip install --user --quiet --upgrade pip
python3 -m pip install --user --quiet scikit-learn pandas numpy joblib

python3 train_models.py --output-dir "$HOME/ml_assignment_run"

if [ "$1" = "app" ]; then
    python3 -m pip install --user --quiet streamlit matplotlib
    echo "Open http://localhost:8501 in the lab browser (Ctrl+C to stop)."
    python3 -m streamlit run app.py --server.headless true --server.port 8501
fi
