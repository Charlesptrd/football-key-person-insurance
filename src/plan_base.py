"""
Plan de la base de données (tables nettoyées + table finale) en page HTML autonome.

Les chiffres sont calculés par le code (aucun chiffre écrit à la main) : lignes, colonnes,
valeurs manquantes, doublons, unicité des clés, couverture des liens entre tables.

Utilisation dans 02_merge.ipynb, après la construction de `base` :

    from src.plan_base import generer_plan_base
    generer_plan_base(etude, base, "../docs/plan_base.html")

Le fichier HTML est autonome (aucune dépendance externe) : à ouvrir dans un navigateur.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# Clé primaire attendue de chaque table (None : pas de clé naturelle, on ne contrôle que les doublons stricts)
CLES = {
    "player_profiles": ["player_id"],
    "team_details": ["club_id"],
    "player_performances": ["player_id", "season_name", "competition_id", "team_id"],
    "player_market_value": ["player_id", "date_valeur"],
    "player_teammates_played_with": ["player_id", "teammate_player_id"],
    "perimetre": ["player_id", "season_name", "team_id", "competition_id"],
    "base": ["player_id", "debut"],
}

# Ce que représente une ligne (uniquement quand c'est certain)
GRAIN = {
    "player_profiles": "un joueur",
    "team_details": "un club",
    "player_performances": "un joueur × saison × compétition × club",
    "player_market_value": "un joueur × date de relevé",
    "player_teammates_played_with": "un joueur × coéquipier",
    "transfer_history": "un transfert",
    "player_injuries": "une blessure",
    "perimetre": "un joueur × saison × club × compétition (15 championnats)",
    "base": "un joueur × demi-saison",
}

GROUPES = {
    "player_profiles": "joueur", "player_performances": "joueur", "player_injuries": "joueur",
    "player_market_value": "joueur", "transfer_history": "joueur",
    "player_national_performances": "joueur", "player_teammates_played_with": "joueur",
    "team_details": "club", "team_competitions_seasons": "club",
    "perimetre": "derivee", "base": "derivee",
}

# (table enfant, colonnes enfant, table parent, colonnes parent)
LIENS = [
    ("player_performances", ["player_id"], "player_profiles", ["player_id"]),
    ("player_injuries", ["player_id"], "player_profiles", ["player_id"]),
    ("player_market_value", ["player_id"], "player_profiles", ["player_id"]),
    ("transfer_history", ["player_id"], "player_profiles", ["player_id"]),
    ("player_national_performances", ["player_id"], "player_profiles", ["player_id"]),
    ("player_teammates_played_with", ["player_id"], "player_profiles", ["player_id"]),
    ("player_teammates_played_with", ["teammate_player_id"], "player_profiles", ["player_id"]),
    ("perimetre", ["player_id"], "player_profiles", ["player_id"]),
    ("player_performances", ["team_id"], "team_details", ["club_id"]),
    ("transfer_history", ["from_team_id"], "team_details", ["club_id"]),
    ("transfer_history", ["to_team_id"], "team_details", ["club_id"]),
    # player_national_performances.team_id retiré : 0 % de correspondance avec team_details, car ce
    # sont vraisemblablement des identifiants de sélections nationales et non de clubs.
    ("team_competitions_seasons", ["club_id"], "team_details", ["club_id"]),
    ("perimetre", ["team_id"], "team_details", ["club_id"]),
    ("base", ["player_id"], "player_profiles", ["player_id"]),
    ("base", ["team_id"], "team_details", ["club_id"]),
    ("base", ["player_id", "season_name"], "player_performances", ["player_id", "season_name"]),
]


def _stats_table(nom, df, cle, colonnes_perf=()):
    n = len(df)
    colonnes = []
    for c in df.columns:
        s = df[c]
        manq = int(s.isna().sum())
        colonnes.append({
            "nom": str(c), "type": str(s.dtype),
            "manquants": manq, "manquants_pct": round(manq / n * 100, 2) if n else 0.0,
            "uniques": int(s.nunique(dropna=True)),
            "source": "performances" if c in colonnes_perf else None,
        })
    cle_ok = [c for c in (cle or []) if c in df.columns]
    return {
        "nom": nom, "lignes": n, "nb_colonnes": df.shape[1],
        "groupe": GROUPES.get(nom, "joueur"),
        "grain": GRAIN.get(nom),
        "cle": cle_ok if cle and len(cle_ok) == len(cle) else None,
        "doublons_stricts": int(df.duplicated().sum()),
        "doublons_cle": int(df.duplicated(cle_ok).sum()) if cle and len(cle_ok) == len(cle) else None,
        "colonnes": colonnes,
    }


def _couverture(enfant, cols_e, parent, cols_p):
    """% de lignes de l'enfant dont la valeur existe chez le parent, et % de clés parent utilisées."""
    e = enfant[cols_e].dropna().astype(object)
    p = parent[cols_p].dropna().astype(object)
    if len(cols_e) == 1:
        ke, kp = pd.Index(e.iloc[:, 0]), pd.Index(p.iloc[:, 0])
    else:
        ke, kp = pd.MultiIndex.from_frame(e), pd.MultiIndex.from_frame(p)
    ok = ke.isin(kp)
    ke_u, kp_u = ke.unique(), kp.unique()
    return {
        "lignes_avec_parent_pct": round(float(ok.mean()) * 100, 2) if len(ok) else None,
        "lignes_orphelines": int((~ok).sum()),
        "parents_utilises_pct": round(float(kp_u.isin(ke_u).mean()) * 100, 2) if len(kp_u) else None,
        "lignes_enfant_non_nulles": len(ke),
    }


def calculer_plan(etude, base, colonnes_perf=()):
    tables = {nom: df for nom, df in etude.items() if isinstance(df, pd.DataFrame)}
    tables["base"] = base
    nb_championnats = (int(tables["perimetre"]["competition_id"].nunique())
                       if "perimetre" in tables and "competition_id" in tables["perimetre"].columns else None)
    stats = [_stats_table(nom, df, CLES.get(nom), colonnes_perf if nom == "base" else ())
             for nom, df in tables.items()]
    if nb_championnats is not None:
        for s in stats:
            if s["nom"] == "perimetre":
                s["grain"] = f"un joueur × saison × club × compétition ({nb_championnats} championnats)"
    liens, ignores = [], []
    for enf, ce, par, cp in LIENS:
        if enf in tables and par in tables and set(ce) <= set(tables[enf].columns) \
                and set(cp) <= set(tables[par].columns):
            liens.append({"enfant": enf, "cols_enfant": ce, "parent": par, "cols_parent": cp,
                          **_couverture(tables[enf], ce, tables[par], cp)})
        else:
            ignores.append(f"{enf}.{ce} -> {par}.{cp}")
    return {"genere_le": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "nb_championnats": nb_championnats,
            "tables": stats, "liens": liens, "liens_ignores": ignores}


def generer_plan_base(etude, base, chemin="plan_base.html", colonnes_perf=None):
    """Calcule les statistiques et écrit la page HTML. Renvoie le dictionnaire des statistiques."""
    if colonnes_perf is None:   # colonnes de base issues de l'agrégation des performances
        from .performances import agreger_performances
        colonnes_perf = set(agreger_performances(etude["player_performances"]).columns) - {"player_id", "season_name"}
        colonnes_perf |= {"nb_clubs_saison"}
    plan = calculer_plan(etude, base, colonnes_perf)
    donnees = json.dumps(plan, ensure_ascii=False, default=str).replace("</", "<\\/")
    html = (Path(__file__).parent / "plan_base_template.html").read_text(encoding="utf-8")
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    chemin.write_text(html.replace("__DONNEES__", donnees), encoding="utf-8")
    if plan["liens_ignores"]:
        print("Liens ignorés (colonnes absentes) :", plan["liens_ignores"])
    print("Plan écrit :", chemin.resolve())
    return plan
