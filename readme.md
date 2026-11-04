# Football Key Person Insurance

Mémoire de M1 — Majeure Actuariat, Université Paris-Dauphine – PSL (2026-2027)

Construction et tarification d'un produit d'assurance personne clé protégeant les clubs de football professionnel contre la perte de valeur marchande de leurs joueurs en cas de blessure grave ou de décès.

## Problématique

> Comment construire et tarifer un produit d'assurance permettant de protéger les clubs de football professionnel contre la perte de valeur marchande de leurs joueurs à la suite d'une blessure grave ou d'un décès ?

## Démarche

Le projet suit quatre phases :

1. **Préparation des données** : construire une base joueur × saison réunissant club, championnat, âge, poste, temps de jeu, historique de blessures et évolution de la valeur marchande.
2. **Modélisation de la fréquence** : définir la notion de blessure grave et estimer, pour chaque joueur, la probabilité annuelle de survenance d'un événement couvert (blessure grave ou décès).
3. **Modélisation de la sévérité** : mesurer la perte de valeur marchande consécutive à une blessure grave entre deux marchés des transferts.
4. **Conception et tarification du produit** : calculer une prime pure et tester différentes structures de garantie (franchise, plafond, taux de couverture, capital maximal).

## Données

Jeu de données public Kaggle : [Football Datasets](https://www.kaggle.com/datasets/xfkzujqjvx97n/football-datasets)


## Installation

```bash
git clone https://github.com/Charlesptrd/football-key-person-insurance
cd football-key-person-insurance

python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Encadrement

- Willy Marais et Alexandre Kebe — Exiom Partners
- Coordination : Tachfine El Alami — Université Paris-Dauphine
