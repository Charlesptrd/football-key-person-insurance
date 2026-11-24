import numpy as np
import pandas as pd

CLES = ["player_id", "season_name"]

COMPTEURS = ["nb_in_group", "nb_on_pitch", "goals", "assists", "own_goals", "subed_in", "subed_out",
             "yellow_cards", "second_yellow_cards", "direct_red_cards", "penalty_goals",
             "goals_conceded", "clean_sheets"]


# Durées moyennes supposées (hypothèses, pas des mesures) pour estimer les minutes
MIN_MATCH_COMPLET = 90      # titulaire non remplacé
MIN_TITULAIRE_REMPLACE = 65   # titulaire sorti en cours de match
MIN_REMPLACANT = 25         # entré en cours de match


def ajouter_temps_de_jeu(agg):
 
    titu = (agg["nb_on_pitch"] - agg["subed_in"]).clip(lower=0)
    sortis = agg["subed_out"].clip(upper=titu)    # titulaires remplacés (au plus titu)
    complets = titu - sortis

    agg["nb_titularisations"] = titu
    agg["nb_matchs_complets"] = complets
    agg["part_titulaire"] = titu / agg["nb_on_pitch"].where(agg["nb_on_pitch"] > 0)
    agg["minutes_estimees"] = (complets * MIN_MATCH_COMPLET + sortis * MIN_TITULAIRE_REMPLACE
                               + agg["subed_in"].clip(upper=agg["nb_on_pitch"]) * MIN_REMPLACANT)
    return agg


def agreger_performances(perf):
    compteurs = [c for c in COMPTEURS if c in perf.columns]
    g = perf.groupby(CLES, sort=True, observed=True)

    agg = g[compteurs].sum(min_count=1)
    agg["nb_competitions"] = g["competition_id"].nunique()
    agg["nb_clubs"] = g["team_id"].nunique()
    agg = agg.reset_index()

    agg = ajouter_temps_de_jeu(agg)

    assert not agg.duplicated(CLES).any()
    for c in compteurs:   # contrôle : la somme globale est conservée
        assert np.isclose(agg[c].sum(), perf[c].sum()), c
    return agg
