import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# Définir le contenu des cellules du notebook
cells = [
    new_markdown_cell("# Démonstration du moteur de tarification d'options\n\nCe notebook présente une démonstration complète du moteur de tarification Monte Carlo, incluant la tarification des options européennes et exotiques, le calcul des Grecs et l'analyse de la convergence."),
    new_markdown_cell("## 1. Configuration et Importations\n\nNous commençons par importer toutes les classes et fonctions nécessaires depuis les fichiers du projet."),
    new_code_cell(
"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Callable, Optional, List

# Les fichiers locaux doivent être dans le même répertoire que le notebook
from pricing import MCPricingEngine, black_scholes_call, black_scholes_put
from exotic_options import ExoticOptionsEngine
from greeks import GreeksEngine
from convergence_analysis import ConvergenceAnalyzer

# Configurez les paramètres d'affichage de matplotlib pour de meilleurs graphiques
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")"""
    ),
    new_markdown_cell("## 2. Définition des paramètres de l'option\n\nNous fixons ici les paramètres financiers standards qui seront utilisés pour les différentes simulations."),
    new_code_cell(
"""# Paramètres standards pour les options
S0 = 100.0   # Prix initial de l'actif
K = 100.0    # Prix d'exercice (Strike)
T = 1.0      # Temps à l'échéance (en années)
r = 0.05     # Taux sans risque
sigma = 0.2  # Volatilité
M = 100000   # Nombre de simulations (trajectoires)
N = 100      # Nombre de pas de temps"""
    ),
    new_markdown_cell("## 3. Tarification des options européennes\n\nNous utilisons le moteur de tarification Monte Carlo et comparons les résultats avec la solution analytique de Black-Scholes pour validation. L'utilisation des **variables antithétiques** est activée par défaut pour une meilleure efficacité."),
    new_code_cell(
"""print("--- Tarification des options européennes ---")

# Initialiser le moteur de tarification
engine = MCPricingEngine(seed=42)

# Prix analytique (Black-Scholes) pour la référence
bs_call_price = black_scholes_call(S0, K, T, r, sigma)
bs_put_price = black_scholes_put(S0, K, T, r, sigma)

print(f"Prix théorique (Black-Scholes) pour un call : {bs_call_price:.6f}")
print(f"Prix théorique (Black-Scholes) pour un put  : {bs_put_price:.6f}")

print("\\n--- Tarification Monte Carlo ---")

# Tarification du call européen avec variables antithétiques
mc_call_price = engine.european_call_mc(S0, K, T, r, sigma, M=M, antithetic=True)
mc_put_price = engine.european_put_mc(S0, K, T, r, sigma, M=M, antithetic=True)

print(f"Prix MC Call (Antithétique) : {mc_call_price:.6f} (Erreur : {abs(mc_call_price - bs_call_price):.6f})")
print(f"Prix MC Put (Antithétique)  : {mc_put_price:.6f} (Erreur : {abs(mc_put_price - bs_put_price):.6f})")"""
    ),
    new_markdown_cell("## 4. Réduction de variance avec les variables de contrôle\n\nCette section illustre l'efficacité des **variables de contrôle** pour améliorer la précision de l'estimation du prix Monte Carlo en réduisant significativement la variance des résultats."),
    new_code_cell(
"""print("--- Amélioration avec les variables de contrôle ---")

# Tarification du call avec variables de contrôle
mc_plain_price, mc_cv_price = engine.european_call_mc(
    S0, K, T, r, sigma, M=M, antithetic=True, control_variate=True
)

print(f"Prix MC sans variable de contrôle : {mc_plain_price:.6f} (Erreur : {abs(mc_plain_price - bs_call_price):.6f})")
print(f"Prix MC avec variable de contrôle : {mc_cv_price:.6f} (Erreur : {abs(mc_cv_price - bs_call_price):.6f})")"""
    ),
    new_markdown_cell("## 5. Tarification des options exotiques\n\nLe moteur est conçu pour gérer des options plus complexes dépendantes de la trajectoire, comme les options asiatiques ou les options barrières."),
    new_code_cell(
"""print("--- Tarification des options exotiques ---")

exotic_engine = ExoticOptionsEngine(seed=42)
M_exotic = 50000 # Moins de simulations pour les options exotiques
N_exotic = 252 # Plus de pas de temps pour des options dépendantes de la trajectoire

# Option asiatique (moyenne arithmétique)
asian_price = exotic_engine.asian_call_mc(S0, K, T, r, sigma, M=M_exotic, N=N_exotic, antithetic=True)
print(f"Prix MC du call asiatique : {asian_price:.6f}")

# Option barrière (Up-and-Out)
B = 120.0 # Niveau de barrière
barrier_price = exotic_engine.barrier_up_out_call_mc(S0, K, B, T, r, sigma, M=M_exotic, N=N_exotic, antithetic=True)
print(f"Prix MC du call barrière : {barrier_price:.6f}")"""
    ),
    new_markdown_cell("## 6. Calcul des Grecs\n\nLes Grecs sont des mesures de sensibilité du prix de l'option aux variations des paramètres sous-jacents. Nous les calculons ici en utilisant la méthode des **différences finies**."),
    new_code_cell(
"""print("--- Calcul des Grecs ---")

greeks_engine = GreeksEngine(seed=42)

# Définir les paramètres pour le calcul des Grecs
params = {'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma, 'M': M}

# Calculer tous les Grecs avec la méthode des différences finies centrales
all_greeks = greeks_engine.compute_all_greeks(
    pricing_func=engine.european_call_mc,
    params=params,
    method='central'
)
print("Grecs (Différences finies) :")
for greek, value in all_greeks.items():
    print(f"  {greek.capitalize()}: {value:.6f}")
    
# Calculer le Delta avec la méthode pathwise (plus précis pour les options lisses)
pathwise_delta = greeks_engine.pathwise_delta(S0, K, T, r, sigma, 'call', M=M)
print(f"\\nDelta (méthode pathwise) : {pathwise_delta:.6f}")"""
    ),
    new_markdown_cell("## 7. Analyse de la convergence\n\nCette dernière section montre comment la précision de l'estimation Monte Carlo s'améliore à mesure que le nombre de simulations augmente, démontrant la loi de convergence $O(1/\\sqrt{M})$."),
    new_code_cell(
"""print("--- Analyse de la convergence Monte Carlo ---")

analyzer = ConvergenceAnalyzer(seed=42)

# Définir le nombre de simulations à tester
M_values = np.logspace(3, 6, 10, dtype=int)
n_trials = 20  # Nombre d'essais par valeur M pour une meilleure statistique

# Prix théorique pour la comparaison
true_price = black_scholes_call(S0, K, T, r, sigma)

# Analyser la convergence
convergence_results = analyzer.monte_carlo_convergence(
    pricing_func=engine.european_call_mc,
    true_price=true_price,
    M_values=M_values,
    n_trials=n_trials,
    S0=S0, K=K, T=T, r=r, sigma=sigma, antithetic=True
)

# Tracer les résultats
fig = analyzer.plot_convergence(convergence_results, true_price, title="Convergence du prix des options européennes")
plt.show(fig)"""
    ),
    new_markdown_cell("## 8. Analyse de la réduction de variance (pour un aperçu rapide)"),
    new_code_cell(
"""print("--- Analyse de la réduction de variance ---")

cv_results = analyzer.control_variates_analysis(
    pricing_func=engine.european_call_mc,
    true_price=true_price,
    M=M,
    n_trials=50,
    S0=S0, K=K, T=T, r=r, sigma=sigma, antithetic=True
)

print(f"Variance MC sans contrôle : {cv_results['plain_variance']:.8f}")
print(f"Variance MC avec contrôle : {cv_results['cv_variance']:.8f}")
print(f"Réduction de variance : {cv_results['variance_reduction']:.2f}x")
print(f"Gain d'efficacité net : {cv_results['efficiency']:.2f}x")"""
    )
]

# Créer un nouveau notebook et ajouter les cellules
notebook = new_notebook(cells=cells)

# Écrire le notebook dans un fichier
with open("demo.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(notebook, f)

print("Le fichier demo.ipynb a été créé avec succès. Vous pouvez maintenant l'ouvrir avec Jupyter Notebook ou JupyterLab.")
