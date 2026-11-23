"""
Table joueur × demi-saison (phase 1 du sujet A1).

Pour chaque joueur du périmètre et chaque demi-saison : club, championnat, âge au début
de la demi-saison et poste.

Convention : si un joueur a eu plusieurs clubs pendant une demi-saison, on retient celui
où il est resté le plus longtemps (en jours) pendant cette demi-saison.

Utilisation (après le nettoyage et la réduction du périmètre) :

    from src.features import preparer_tables, restreindre_perimetre
    from src.demi_saisons import construire_demi_saisons

    etude = restreindre_perimetre(preparer_tables(tables))
    demi_saisons = construire_demi_saisons(etude)
"""

import numpy as np
import pandas as pd

try:                                   # import depuis le dossier du projet : from src.demi_saisons import ...
    from .features import CHAMPIONNATS
except ImportError:                    # import avec src/ dans sys.path : from demi_saisons import ...
    from features import CHAMPIONNATS

# Fin des données : dernière blessure et dernière valeur marchande observées (mi-septembre 2025)
DATE_EXTRACTION = "2025-09-12"


def _bornes_saison(season_name):
    s = season_name.astype(str)
    a_cheval = s.str.contains("/")
    yy = pd.to_numeric(s.str[:2], errors="coerce")
    annee_cheval = np.where(yy >= 50, 1900 + yy, 2000 + yy)
    annee_civile = pd.to_numeric(s.where(~a_cheval), errors="coerce")
    annee = pd.Series(np.where(a_cheval, annee_cheval, annee_civile), index=s.index).astype(int)
    debut_1 = pd.to_datetime(np.where(a_cheval, annee.astype(str) + "-07-01", annee.astype(str) + "-01-01"))
    debut_2 = pd.to_datetime(np.where(a_cheval, (annee + 1).astype(str) + "-01-01", annee.astype(str) + "-07-01"))
    return debut_1, debut_2


def _saisons_possibles(debut):
    annee = np.where(debut.dt.month >= 7, debut.dt.year, debut.dt.year - 1)
    a_cheval = pd.Series([f"{a % 100:02d}/{(a + 1) % 100:02d}" for a in annee], index=debut.index)
    civile = debut.dt.year.astype(str)
    return a_cheval, civile


def _sejours_en_club(transfer_history):
    #Chronologie des clubs de chaque joueur : une ligne par sejour (club, début, fin) en se basant sur les dates des transferts.

    t = transfer_history.dropna(subset=["transfer_date"]).copy()
    t["ordre"] = -np.arange(len(t))          # même date : la ligne la plus haute de la table est la plus récente
    t = t.sort_values(["player_id", "transfer_date", "ordre"])
    t["fin_sejour"] = t.groupby("player_id")["transfer_date"].shift(-1)

    sejours = t[["player_id", "to_team_id", "transfer_date", "fin_sejour"]].rename(
        columns={"to_team_id": "team_id", "transfer_date": "debut_sejour"})
    premiers = t.groupby("player_id").head(1)
    avant_premier = pd.DataFrame({
        "player_id": premiers["player_id"].values, "team_id": premiers["from_team_id"].values,
        "debut_sejour": pd.Timestamp("1900-01-01"), "fin_sejour": premiers["transfer_date"].values})
    sejours = pd.concat([avant_premier, sejours], ignore_index=True)
    sejours["fin_sejour"] = sejours["fin_sejour"].fillna(pd.Timestamp("2100-01-01"))
    return sejours


def _noms_clubs(etude):
    """Nom de chaque club : team_details nettoyée, sinon player_performances, sinon transfer_history."""
    perf, th = etude["player_performances"], etude["transfer_history"]
    sources = [
        etude["team_details"].set_index("club_id")["club_name"],
        perf.drop_duplicates("team_id", keep="last").set_index("team_id")["team_name"],
        th.drop_duplicates("to_team_id").set_index("to_team_id")["to_team_name"],
        th.drop_duplicates("from_team_id").set_index("from_team_id")["from_team_name"],
    ]
    noms = sources[0].astype(object)
    for s in sources[1:]:
        noms = noms.combine_first(s.astype(object))
    return noms.str.replace(r"\s*\(\d+\)$", "", regex=True)


def construire_demi_saisons(etude, date_extraction=DATE_EXTRACTION):
    """

    Une ligne par joueur et par demi-saison couvrant une saison du périmètre :
    - demi-saisons calendaires : 1er juillet - 31 décembre et 1er janvier - 30 juin ;
    - club principal : celui où le joueur a passé le plus de jours pendant la demi-saison,
      d'après les dates de transfer_history (égalité : club d'un championnat étudié, puis le plus récent) ;
    - championnat : celui du club principal cette saison-là, s'il fait partie des championnats étudiés
      (sinon manquant et dans_perimetre = False : club de jeunes, réserve, autre championnat, sans club…) ;
    - âge en années révolues au début de la demi-saison ; poste issu de player_profiles
      (poste unique sur la carrière : la table ne donne pas d'historique des postes) ;
    - part_observee : part de la demi-saison antérieure à la fin des données (1 = entièrement observée).
    """
    per, prof = etude["perimetre"], etude["player_profiles"]

    js = per[["player_id", "season_name"]].drop_duplicates()
    debut_1, debut_2 = _bornes_saison(js["season_name"])
    demis = pd.concat([pd.DataFrame({"player_id": js["player_id"].values, "debut": debut_1}),
                       pd.DataFrame({"player_id": js["player_id"].values, "debut": debut_2})]
                      ).drop_duplicates().reset_index(drop=True)
    demis["fin"] = demis["debut"] + pd.DateOffset(months=6) - pd.Timedelta(days=1)

    # 2. Jours passés dans chaque club pendant chaque demi-saison
    x = demis.merge(_sejours_en_club(etude["transfer_history"]), on="player_id")
    debut_commun = x[["debut", "debut_sejour"]].max(axis=1)
    fin_commune = pd.concat([x["fin"] + pd.Timedelta(days=1), x["fin_sejour"]], axis=1).min(axis=1)
    x["jours"] = (fin_commune - debut_commun).dt.days.clip(lower=0)
    x = x[x["jours"] > 0].groupby(["player_id", "debut", "team_id"], as_index=False)["jours"].sum()

    # 3. Championnat de chaque club par saison (déduit de tous les joueurs du périmètre)
    ligue = (per.groupby(["team_id", "season_name"])["competition_id"]
                .agg(lambda s: s.mode().iloc[0]))




    
    def championnat(team_ids, saisons):
        return pd.Series([ligue.get((t, s)) for t, s in zip(team_ids, saisons)], index=team_ids.index)

    # 4. Club principal : le plus de jours ; égalité -> club d'un championnat étudié, puis le plus récent
    s_cheval, s_civile = _saisons_possibles(x["debut"])
    x["club_etudie"] = championnat(x["team_id"], s_cheval).notna() | championnat(x["team_id"], s_civile).notna()
    nb_clubs = x.groupby(["player_id", "debut"]).size().rename("nb_clubs")
    principal = (x.sort_values(["player_id", "debut", "jours", "club_etudie"], ascending=[True, True, False, False])
                  .drop_duplicates(["player_id", "debut"])
                  .rename(columns={"jours": "jours_club_principal"}))
    res = (demis.merge(principal[["player_id", "debut", "team_id", "jours_club_principal"]],
                       on=["player_id", "debut"], how="left")
                .merge(nb_clubs, on=["player_id", "debut"], how="left"))

    # Repli pour les joueurs sans aucun transfert : club du périmètre s'il est unique sur la saison
    s_cheval, s_civile = _saisons_possibles(res["debut"])
    club_unique = (per.groupby(["player_id", "season_name"])["team_id"]
                      .agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan))
    sans_club = res["team_id"].isna()
    if sans_club.any():
        res["team_id"] = res["team_id"].astype(float)
        res.loc[sans_club, "team_id"] = [
            club_unique.get((p, a), club_unique.get((p, c), np.nan))
            for p, a, c in zip(res.loc[sans_club, "player_id"], s_cheval[sans_club], s_civile[sans_club])]

    # 5. Championnat et saison du club principal
    comp_cheval = championnat(res["team_id"], s_cheval)
    comp_civile = championnat(res["team_id"], s_civile)
    civile = comp_cheval.isna() & comp_civile.notna()
    res["competition_id"] = comp_cheval.fillna(comp_civile)
    res["championnat"] = res["competition_id"].map(CHAMPIONNATS)
    res["season_name"] = np.where(civile, s_civile, s_cheval)
    premier_mois = np.where(civile, 1, 7)
    res["moitie"] = np.where(res["debut"].dt.month == premier_mois, 1, 2)
    res["dans_perimetre"] = res["competition_id"].notna()

    # 6. Âge au début de la demi-saison et poste
    res = res.merge(prof[["player_id", "date_of_birth", "main_position", "sub_position"]], on="player_id", how="left")
    d, n = res["debut"], res["date_of_birth"]
    anniv_pas_passe = (d.dt.month < n.dt.month) | ((d.dt.month == n.dt.month) & (d.dt.day < n.dt.day))
    res["age_debut"] = (d.dt.year - n.dt.year - anniv_pas_passe.astype(int)).astype("Int64")

    # 7. Part de la demi-saison observée avant la fin des données
    fin_donnees = pd.Timestamp(date_extraction)
    duree = (res["fin"] - res["debut"]).dt.days + 1
    observe = (res["fin"].clip(upper=fin_donnees) - res["debut"]).dt.days + 1
    res["part_observee"] = (observe.clip(lower=0) / duree).round(3)

    res["team_id"] = res["team_id"].astype("Int64")
    res["club_name"] = res["team_id"].map(_noms_clubs(etude))
    res["nb_clubs"] = res["nb_clubs"].fillna(1).astype(int)
    res["jours_club_principal"] = res["jours_club_principal"].astype("Int64")
    for c in ["competition_id", "championnat"]:
        res[c] = res[c].astype("category")

    colonnes = ["player_id", "season_name", "moitie", "debut", "fin",
                "team_id", "club_name", "jours_club_principal", "nb_clubs",
                "competition_id", "championnat", "dans_perimetre",
                "age_debut", "main_position", "sub_position", "part_observee"]
    res = res[colonnes].sort_values(["player_id", "debut"]).reset_index(drop=True)

    assert not res.duplicated(["player_id", "debut"]).any()
    return res
