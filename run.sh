#!/bin/bash

echo "🚀 Setting up Monte Carlo Option Pricing Environment..."

# Création de l'environnement virtuel s'il n'existe pas
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python -m venv venv
fi

# Activation de l'environnement virtuel
echo "Activating virtual environment..."
source venv/bin/activate

# Installation des dépendances
echo "Installing required packages..."
pip install -r requirements.txt

# Vérification de Jupyter
if ! command -v jupyter &> /dev/null; then
    echo "Installing Jupyter..."
    pip install jupyter
fi

# Ajout de l'environnement virtuel comme kernel Jupyter
python -m ipykernel install --user --name=montecarlo-optionpricing --display-name="Monte Carlo Option Pricing"

echo "🔧 Environment setup complete!"
echo "📓 Launching Jupyter notebook..."

# Lancement du notebook
jupyter notebook demo.ipynb

# Pour quitter proprement
deactivate
