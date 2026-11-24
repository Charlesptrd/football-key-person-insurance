# Préparation des tables du projet.

import numpy as np
import pandas as pd

SUFFIXE_ID = r"\s*\(\d+\)$" 


# =============================================================================
# player_profiles
# =============================================================================

COLONNES_RETIREES_PROFILES = [

    "player_image_url", "outfitter", "social_media_url", "player_agent_id", "player_agent_name",
    "name_in_home_country", "place_of_birth",
    "joined", "contract_expires", "contract_option", "date_of_last_contract_extension",
    "on_loan_from_club_id", "on_loan_from_club_name", "contract_there_expires",
    "second_club_url", "second_club_name", "third_club_url", "third_club_name",
    "fourth_club_url", "fourth_club_name",
]

STATUTS_PAR_CLUB = {123: "retired", 515: "without_club", 2113: "career_break", 75: "unknown", 4023: "unknown"}

COLONNES_FINALES_PROFILES = [
    "player_id", "player_name", "player_slug",
    "date_of_birth", "country_of_birth", "nationality_1", "nationality_2",
    "has_dual_nationality", "is_eu",
    "height", "foot", "main_position", "sub_position",
    "date_of_death", "is_dead", "current_status", "suspected_duplicate",
]


def nettoyer_player_profiles(profiles):
    df = profiles.copy()

    cle_identite = ["player_slug", "date_of_birth", "citizenship"]
    df["suspected_duplicate"] = df.duplicated(cle_identite, keep=False) & df["date_of_birth"].notna()

    df = df.drop(columns=COLONNES_RETIREES_PROFILES)

    df["player_name"] = df["player_name"].str.replace(SUFFIXE_ID, "", regex=True).str.strip()
    manquant = df["player_name"].isna()
    df.loc[manquant, "player_name"] = df.loc[manquant, "player_slug"].str.replace("-", " ").str.title()

    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], format="%Y-%m-%d", errors="coerce")
    date_deces = df["date_of_death"].str.extract(r"^(\d{2}\.\d{2}\.\d{4})")[0]   # « JJ.MM.AAAA (âge) »
    df["date_of_death"] = pd.to_datetime(date_deces, format="%d.%m.%Y")
    df["is_dead"] = df["date_of_death"].notna()

    df["current_status"] = df["current_club_id"].map(STATUTS_PAR_CLUB).fillna("active")
    df.loc[df["is_dead"], "current_status"] = "deceased"
    df = df.drop(columns=["current_club_id", "current_club_name"])

    df.loc[df["height"] < 140, "height"] = np.nan


    citoyennete = df["citizenship"].str.replace("Cookinseln", "Cook Islands")
    nat = citoyennete.str.split(r"\s{2,}", expand=True)
    df["nationality_1"] = nat[0].str.strip()
    df["nationality_2"] = nat[1].str.strip() if 1 in nat.columns else pd.NA
    df["has_dual_nationality"] = df["nationality_2"].notna()
    df = df.drop(columns="citizenship")

    # Poste détaillé (et non post secondaire) (« Defender - Right-Back » -> « Right-Back »)
    poste = df["position"].str.split(" - ", n=1, expand=True)
    df["sub_position"] = poste[1] if 1 in poste.columns else pd.NA
    df.loc[df["main_position"] == "Goalkeeper", "sub_position"] = "Goalkeeper"
    df = df.drop(columns="position")

    df["foot"] = df["foot"].str.strip().str.lower()

    df = df[COLONNES_FINALES_PROFILES].copy()
    for c in ["foot", "main_position", "sub_position", "current_status"]:
        df[c] = df[c].astype("category")

    assert df["player_id"].is_unique
    assert df["height"].dropna().between(140, 215).all()
    assert (df["is_dead"] == df["date_of_death"].notna()).all()
    assert not df["player_name"].str.contains(r"\(\d+\)$", na=False).any()
    return df.reset_index(drop=True)


# =============================================================================
# team_details
# =============================================================================

# Noms tronqués à l'apostrophe par le scraping, corrigés à la main

CORRECTIONS_NOMS_CLUBS = {
    11374: "Jeanne d'Arc de Drancy",
    1286:  "CA Newell's Old Boys",
    17707: "US Jeanne d'Arc Carquefou",
    17665: "Les Genêts d'Anglet",
    23206: "ES Pays d'Uzès",
    2618:  "Olympique d'Alès",
    27490: "C'Chartres Football",
    4153:  "L'Aquila Calcio",
    7882:  "US Raon-l'Étape",
}
MOTIF_CATEGORIE_JEUNES = (r"(?:\bU\s?-?\d{2}\b|\bUnder\s?\d{2}\b|\bSub-?\d{2}\b|\bPrimavera\b|\bJuvenil\b"
                          r"|\bYouth\b|\bAcademy\b|\bReserves?\b|^Jong\s)")
MOTIF_SUFFIXE_RESERVE = r"\s(?:II|III|B|C|2)$"


def nettoyer_team_details(team_details):
    df = team_details.copy()

    df = df.drop(columns=["season_id", "competition_id", "competition_slug", "competition_name",
                          "club_division", "logo_url", "source_url", "_last_modified_at"])

    df["club_name"] = df["club_name"].str.replace(SUFFIXE_ID, "", regex=True).str.strip()
    tronques = df["club_name"].str.endswith("\\")
    assert set(df.loc[tronques, "club_id"]) <= set(CORRECTIONS_NOMS_CLUBS), "nouveau nom tronqué à corriger"
    df["club_name"] = df["club_id"].map(CORRECTIONS_NOMS_CLUBS).fillna(df["club_name"])

    nom_sans_date = df["club_name"].str.replace(r"\s*\(-\s?\d{4}\)$", "", regex=True)
    df["is_youth_or_reserve"] = (nom_sans_date.str.contains(MOTIF_CATEGORIE_JEUNES)
                                 | nom_sans_date.str.contains(MOTIF_SUFFIXE_RESERVE))

    df = df[["club_id", "club_name", "club_slug", "country_name", "is_youth_or_reserve"]].copy()
    df["country_name"] = df["country_name"].astype("category")

    assert df["club_id"].is_unique
    assert not df["club_name"].str.contains(r"\(\d+\)$|\\$").any()
    return df.reset_index(drop=True)


# =============================================================================
# team_competitions_seasons
# =============================================================================

def nettoyer_team_competitions_seasons(team_competitions_seasons, team_details):
    df = team_competitions_seasons.drop_duplicates()   # 73 % de copies exactes

    assert not df.duplicated(["club_id", "season_id"]).any()
    assert df.groupby("club_id")["competition_id"].nunique(dropna=False).eq(1).all(), \
        "la compétition varie selon la saison : revoir la réduction à une ligne par club"

    par_club = (df.sort_values(["club_id", "season_id"])
                  .drop_duplicates("club_id")
                  [["club_id", "competition_id", "competition_name"]]
                  .rename(columns={"competition_id": "current_competition_id",
                                   "competition_name": "current_competition_name"}))

    pays = (par_club.merge(team_details[["club_id", "country_name"]], on="club_id")
                    .dropna(subset=["current_competition_id"])
                    .groupby("current_competition_id")["country_name"]
                    .agg(lambda s: s.mode().iloc[0]))
    par_club["current_competition_country"] = par_club["current_competition_id"].map(pays).astype(object)

    for c in ["current_competition_id", "current_competition_name", "current_competition_country"]:
        par_club[c] = par_club[c].astype("category")

    assert par_club["club_id"].is_unique
    return par_club.reset_index(drop=True)


# =============================================================================
# player_injuries
# =============================================================================

def nettoyer_player_injuries(player_injuries):
    """Nettoie player_injuries : suppression des doublons stricts.

    Conservés tels quels (voir le journal) : dates manquantes, blessures de plus de 365 jours,
    saisons incohérentes. Les dates restent au format texte, comme dans le notebook.
    """
    df = player_injuries.drop_duplicates()
    assert not df.duplicated().any()
    return df.reset_index(drop=True)


# =============================================================================
# player_national_performances
# =============================================================================

def nettoyer_player_national_performances(player_national_performances):
    """Nettoie player_national_performances : doublons stricts, matchs incohérents, colonne vide."""
    df = player_national_performances.drop_duplicates().copy()

    # 0 match mais au moins un but : nombre de matchs mis à manquant
    incoherent = df["matches"].eq(0) & df["goals"].gt(0)
    df["matches"] = df["matches"].astype("Int64")
    df.loc[incoherent, "matches"] = pd.NA

    df = df.drop(columns=["debut"])   # entièrement vide

    assert not (df["matches"].eq(0) & df["goals"].gt(0)).any()
    return df.reset_index(drop=True)

# =============================================================================
# transfer_history
# =============================================================================

def nettoyer_transfer_history(transfer_history):
    """Nettoie transfer_history : doublons stricts, date de transfert convertie en datetime.

    Les noms de club sont gardés : la plupart des clubs des transferts sont absents de team_details.
    """
    df = transfer_history.drop_duplicates().copy()
    df["transfer_date"] = pd.to_datetime(df["transfer_date"], format="%Y-%m-%d", errors="coerce")
    # Une date en 3026 (saison 25/26, sans doute 2026) : mise à manquant, pour que le résultat
    # ne dépende pas de la version de pandas (pandas 2 ne sait pas représenter l'an 3026)
    df.loc[df["transfer_date"].dt.year > 2100, "transfer_date"] = pd.NaT
    assert not df.duplicated().any()
    return df.reset_index(drop=True)

# =============================================================================
# player_market_value
# =============================================================================

def nettoyer_player_market_value(player_market_value):
    df = player_market_value.copy()
    df["date_valeur"] = pd.to_datetime(df["date_unix"], format="%Y-%m-%d", errors="coerce")
    assert df["date_valeur"].notna().all(), "dates non convertibles"
    df = df.drop(columns="date_unix")[["player_id", "date_valeur", "value"]]
    assert not df.duplicated(["player_id", "date_valeur"]).any()
    return df.reset_index(drop=True)

# =============================================================================
# player_performances
# =============================================================================


def nettoyer_player_performances(player_performances):
    df = player_performances.copy()

    # minutes_played : mal scrappée, variable supprimée
    df = df.drop(columns=["minutes_played"])

    df["goals_conceded"] = df["goals_conceded"].astype(float)
    df.loc[df["goals_conceded"] < 0, "goals_conceded"] = np.nan

    assert not df.duplicated(["player_id", "season_name", "competition_id", "team_id"]).any()
    return df.reset_index(drop=True)

# =============================================================================
# player_teammates_played_with
# =============================================================================

def nettoyer_player_teammates_played_with(player_teammates_played_with):
    """player_teammates_played_with : aucun nettoyage décidé, seulement des contrôles.

    Pas de doublon ; les minutes à 0 et à 999 et les coéquipiers sans profil sont conservés.
    Renvoie une copie pour garder la même logique que les autres tables.
    """
    df = player_teammates_played_with.copy()
    assert not df.duplicated(["player_id", "teammate_player_id"]).any()
    return df.reset_index(drop=True)

# =============================================================================
# Toutes les tables d'un coup
# =============================================================================

def preparer_tables(tables, afficher_dim=True):
    propres = {
        "player_profiles": nettoyer_player_profiles(tables["player_profiles"]),
        "team_details": nettoyer_team_details(tables["team_details"]),
        "team_competitions_seasons": nettoyer_team_competitions_seasons(
            tables["team_competitions_seasons"], tables["team_details"]),
        "player_injuries": nettoyer_player_injuries(tables["player_injuries"]),
        "player_national_performances": nettoyer_player_national_performances(
            tables["player_national_performances"]),
        "transfer_history": nettoyer_transfer_history(tables["transfer_history"]),
        "player_market_value": nettoyer_player_market_value(tables["player_market_value"]),
        "player_performances": nettoyer_player_performances(tables["player_performances"]),
        "player_teammates_played_with": nettoyer_player_teammates_played_with(
            tables["player_teammates_played_with"]),
    }
    if afficher_dim:
        for nom, df in propres.items():
            print(f"{nom:<30} {str(tables[nom].shape):>16} -> {df.shape}")
    return propres






# =============================================================================
# Réduction du périmètre : championnats étudiés
# =============================================================================

# Identifiants Transfermarkt : constants dans le temps, contrairement aux noms commerciaux
# (« Liga NOS », « Jupiler Liga »…)
CHAMPIONNATS = {
    "GB1": "Premier League (Angleterre)",
    "ES1": "LaLiga (Espagne)",
    "IT1": "Serie A (Italie)",
    "L1": "Bundesliga (Allemagne)",
    "FR1": "Ligue 1 (France)",
    "BE1": "Jupiler Pro League (Belgique)",
    "PO1": "Liga Portugal (Portugal)",
    "NL1": "Eredivisie (Pays-Bas)",
    "MLS1": "MLS (États-Unis)",
    "SA1": "Saudi Pro League (Arabie saoudite)",
}

TABLES_PAR_JOUEUR = ["player_profiles", "player_injuries", "player_market_value", "transfer_history",
                     "player_national_performances", "player_teammates_played_with"]
TABLES_PAR_CLUB = ["team_details", "team_competitions_seasons"]


def _retirer_categories_inutiles(df):
    """Après un filtre, retire les modalités qui n'apparaissent plus dans les colonnes category."""
    for c in df.select_dtypes("category").columns:
        df[c] = df[c].cat.remove_unused_categories()
    return df


def restreindre_perimetre(propres, championnats=CHAMPIONNATS, afficher=True):
    """Réduit les tables propres aux joueurs ayant évolué dans les championnats étudiés.

    propres : dictionnaire renvoyé par preparer_tables (non modifié).
    championnats : identifiants de compétition à garder (par défaut les 10 de CHAMPIONNATS).

    Règles :
    - un joueur-saison est dans le périmètre si le joueur figure dans l'effectif d'un club
      de l'un de ces championnats cette saison-là (au moins une ligne dans player_performances) ;
    - player_performances : toutes les lignes de ces joueurs-saisons, coupes et compétitions
      européennes comprises (c'est l'exposition au risque de la saison) ;
    - tables par joueur (profils, blessures, valeurs, transferts, sélections, coéquipiers) :
      tout l'historique des joueurs du périmètre (antécédents, valeur avant l'arrivée…) ;
    - team_details et team_competitions_seasons : clubs ayant joué dans ces championnats.

    Renvoie un nouveau dictionnaire, avec en plus la table « perimetre »
    (player_id, season_name, team_id, competition_id) : une ligne par joueur, saison et club
    dans un championnat étudié. C'est la base de la future table joueur-saison.
    """
    ids = list(championnats)
    perf = propres["player_performances"]
    absents = set(ids) - set(perf["competition_id"].unique())
    assert not absents, f"compétitions absentes de player_performances : {absents}"

    perimetre = (perf.loc[perf["competition_id"].isin(ids),
                          ["player_id", "season_name", "team_id", "competition_id"]]
                 .drop_duplicates()
                 .sort_values(["player_id", "season_name", "competition_id"])
                 .reset_index(drop=True))
    joueurs = perimetre["player_id"].unique()
    clubs = perimetre["team_id"].unique()
    joueurs_saisons = perimetre[["player_id", "season_name"]].drop_duplicates()

    reduites = {}

    # Performances : toutes les compétitions des joueurs-saisons du périmètre
    cle_perf = pd.MultiIndex.from_frame(perf[["player_id", "season_name"]])
    garder = cle_perf.isin(pd.MultiIndex.from_frame(joueurs_saisons))
    reduites["player_performances"] = perf.loc[garder].reset_index(drop=True)

    for nom in TABLES_PAR_JOUEUR:
        if nom in propres:
            t = propres[nom]
            reduites[nom] = _retirer_categories_inutiles(t.loc[t["player_id"].isin(joueurs)].reset_index(drop=True))
    for nom in TABLES_PAR_CLUB:
        if nom in propres:
            t = propres[nom]
            reduites[nom] = _retirer_categories_inutiles(t.loc[t["club_id"].isin(clubs)].reset_index(drop=True))

    reduites["perimetre"] = perimetre

    if afficher:
        print(f"Périmètre : {len(joueurs)} joueurs, {len(clubs)} clubs, {len(joueurs_saisons)} joueurs-saisons\n")
        par_champ = perimetre.groupby("competition_id")["player_id"].nunique()
        for cid, libelle in championnats.items():
            print(f"  {libelle:<36} {par_champ.get(cid, 0):>7} joueurs")
        print()
        for nom, t in reduites.items():
            avant = f"{len(propres[nom]):>10}" if nom in propres else " " * 10
            print(f"{nom:<30} {avant} -> {len(t)} lignes")
    return reduites