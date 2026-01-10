import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime
import os

from betting.value import fair_odds, value_percentage
from betting.staking import kelly_stake
from notifications.telegram import send_telegram
LEAGUE_TEAMS = {
    "EPL": [
        "arsenal",
        "aston villa",
        "bournemouth",
        "brentford",
        "brighton",
        "burnley",
        "chelsea",
        "crystal palace",
        "everton",
        "fulham",
        "liverpool",
        "luton",
        "manchester city",
        "manchester united",
        "newcastle",
        "nottingham forest",
        "sheffield united",
        "tottenham",
        "west ham",
        "wolves"
    ],
    "La Liga": [
        "alaves",
        "athletic club",
        "atletico madrid",
        "barcelona",
        "cadiz",
        "celta vigo",
        "getafe",
        "girona",
        "granada",
        "las palmas",
        "mallorca",
        "osasuna",
        "rayo vallecano",
        "real betis",
        "real madrid",
        "real sociedad",
        "sevilla",
        "valencia",
        "villarreal"
    ]
}
# Average goals per match (used in simulation)
avg_goals = 2.7

# Example attack / defense ratings (expand to all teams)
attack = {
    "arsenal": 1.8,
    "chelsea": 1.7,
    "liverpool": 1.9,
    "man city": 2.0
}
defense = {
    "arsenal": 1.1,
    "chelsea": 1.2,
    "liverpool": 1.0,
    "man city": 0.9
}
def normalize_team(name):
    return name.strip().lower()
st.markdown("""
<link rel="manifest" href="/static/manifest.json">
<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/service-worker.js')
}
</script>
""", unsafe_allow_html=True)
EPL_TEAMS = [
    "arsenal",
    "aston villa",
    "bournemouth",
    "brentford",
    "brighton",
    "burnley",
    "chelsea",
    "crystal palace",
    "everton",
    "fulham",
    "liverpool",
    "leeds",
    "manchester city",
    "manchester united",
    "newcastle",
    "nottingham forest",
    "sunderland",
    "tottenham",
    "west ham",
    "wolves"
]

# ----- PAGE CONFIG -----
st.set_page_config(
    page_title="Football Betting Engine",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----- DATA LOAD -----
@st.cache_data
def load_data(file):
    return pd.read_csv(f"data/{file}")
@st.cache_data(ttl=3600)
def load_fixtures():
    return pd.read_csv("data/epl_fixtures_today.csv")
@st.cache_data(ttl=600)
def load_odds():
    return pd.read_csv("data/epl_odds_today.csv")
def is_value_bet(prob, odds, min_ev=0.05):
    ev = (prob * odds) - 1
    return ev >= min_ev, ev
def normalize_team(name):
    return name.lower().strip()

leagues = ["EPL", "LaLiga"]
teams = list(pd.concat([load_data("EPL.csv")['HomeTeam'],
                        load_data("EPL.csv")['AwayTeam']]).unique())

# ----- SETTINGS EXPANDER -----
with st.expander("⚙️ Settings & Inputs"):
    bankroll = st.number_input("Bankroll (£)", min_value=1.0, value=100.0)
    kelly_fraction = st.slider("Kelly Fraction", 0.05, 0.5, 0.25, 0.05)
    min_value = st.slider("Min Value %", 2, 10, 6)

league = st.selectbox("Select League", ["EPL", "La Liga"])

teams = LEAGUE_TEAMS[league]

attack = {team: 1.0 for team in teams}
defense = {team: 1.0 for team in teams}

# ----- LOAD LEAGUE DATA -----
FILE_MAP = {
    "EPL": "EPL.csv",
    "La Liga": "La_Liga.csv"
}
df = load_data(FILE_MAP[league])

# ----- TEAM STRENGTHS -----
def team_strengths(df):
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    avg_goals = df[['FTHG','FTAG']].values.mean()
    attack, defense = {}, {}
    for team in teams:
        scored = df.loc[df.HomeTeam==team,'FTHG'].sum() + df.loc[df.AwayTeam==team,'FTAG'].sum()
        conceded = df.loc[df.HomeTeam==team,'FTAG'].sum() + df.loc[df.AwayTeam==team,'FTHG'].sum()
        matches = (df.HomeTeam==team).sum() + (df.AwayTeam==team).sum()
        attack[team] = (scored/matches)/avg_goals
        defense[team] = (conceded/matches)/avg_goals
    return attack, defense, avg_goals

attack, defense, avg = team_strengths(df)

# ----- SIMULATION -----
def simulate_match(home, away, attack, defense, avg, sims=10000):
    home_xg = (
    attack.get(home, 1.0)
    * defense.get(away, 1.0)
    * avg
    * 1.1
)
    away_xg = (
    attack.get(away, 1.0)
    * defense.get(home, 1.0)
    * avg
)
    results={"H":0,"D":0,"A":0}
    scores={}
    for _ in range(sims):
        hg = poisson.rvs(home_xg)
        ag = poisson.rvs(away_xg)
        if hg>ag: results["H"]+=1
        elif hg==ag: results["D"]+=1
        else: results["A"]+=1
        score=f"{hg}-{ag}"
        scores[score]=scores.get(score,0)+1
    return results, scores, home_xg, away_xg

fixtures = load_fixtures()

predictions = []

for _, row in fixtures.iterrows():
    home = normalize_team(row["home"])
    away = normalize_team(row["away"])

    results, scores, hxg, axg = simulate_match(
        home, away, attack, defense, avg_goals
    )

    predictions.append({
        "home": home,
        "away": away,
        "home_xg": round(hxg, 2),
        "away_xg": round(axg, 2),
        "home_win_prob": results["home"],
        "draw_prob": results["draw"],
        "away_win_prob": results["away"]
    })
    value_bets = []

for _, o in odds_df.iterrows():
    home = normalize_team(o["home"])
    away = normalize_team(o["away"])
    attack = {team.lower(): value for team, value in attack.items()}
    defense = {team.lower(): value for team, value in defense.items()}
    results = {
    "home": home_prob,   # float between 0 and 1
    "draw": draw_prob,   # float between 0 and 1
    "away": away_prob    # float between 0 and 1
    return results, scores, home_xg, away_xg

    match = next(
        (p for p in predictions if p["home"] == home and p["away"] == away),
        None
    )

    if match is None:
        continue

    if o["market"] == "home_win":
        prob = match["home_win_prob"]
    elif o["market"] == "draw":
        prob = match["draw_prob"]
    elif o["market"] == "away_win":
        prob = match["away_win_prob"]
    else:
        continue

    is_value, ev = is_value_bet(prob, o["odds"])

    if is_value:
        value_bets.append({
            "match": f"{home} vs {away}",
            "market": o["market"],
            "odds": o["odds"],
            "ev": round(ev, 3)
        })
        # Sort value bets by EV (expected value), descending
best_bets = sorted(value_bets, key=lambda x: x["ev"], reverse=True)

# Display in Streamlit
st.header("🔥 Best Bets Today (EPL)")
if best_bets:
    st.dataframe(best_bets)
else:
    st.info("No value bets found for today.")
total = sum(results.values())
p_home = results["H"]/total
p_draw = results["D"]/total
p_away = results["A"]/total
if home not in attack or away not in defense:
    print(f"Missing team data: {home} vs {away}")
    # Highlight top 3 bets for emphasis
if best_bets:
    st.subheader("🔥 Top 3 Bets")
    for i, bet in enumerate(best_bets[:3]):
        st.markdown(f"**{i+1}. {bet['match']} — {bet['market']} — Odds: {bet['odds']} — EV: {bet['ev']}**")

# ----- OVER/UNDER & BTTS -----
def over_under_probs(total_xg, line=2.5):
    probs = {i: poisson.pmf(i,total_xg) for i in [0,1,2]}
    under = sum(probs.values())
    over = 1-under
    return over, under

def btts_probs(hxg, axg):
    p_home_0 = poisson.pmf(0,hxg)
    p_away_0 = poisson.pmf(0,axg)
    p_00 = p_home_0*p_away_0
    yes = 1 - p_home_0 - p_away_0 + p_00
    no = 1 - yes
    return yes, no

p_over, p_under = over_under_probs(hxg+axg)
p_btts_yes, p_btts_no = btts_probs(hxg, axg)

# ----- UI METRICS -----
st.title("⚽ Football Betting Engine")
st.subheader(f"{home} vs {away}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Home Win %", f"{p_home*100:.1f}%")
m2.metric("Draw %", f"{p_draw*100:.1f}%")
m3.metric("Away Win %", f"{p_away*100:.1f}%")
m4.metric("Total xG", round(hxg+axg,2))

st.subheader("📊 Outcome Distribution")
chart_df=pd.DataFrame({"Outcome":["Home","Draw","Away"],"Probability":[p_home,p_draw,p_away]})
st.bar_chart(chart_df.set_index("Outcome"))

# ----- INPUT ODDS -----
st.subheader("💰 Enter Bookmaker Odds")
home_odds = st.number_input("Home Odds", min_value=1.01)
draw_odds = st.number_input("Draw Odds", min_value=1.01)
away_odds = st.number_input("Away Odds", min_value=1.01)
over_odds = st.number_input("Over 2.5 Odds", min_value=1.01)
under_odds = st.number_input("Under 2.5 Odds", min_value=1.01)
btts_yes_odds = st.number_input("BTTS Yes Odds", min_value=1.01)
btts_no_odds = st.number_input("BTTS No Odds", min_value=1.01)

# ----- CALCULATE VALUE & STAKES -----
markets = [
    {"market":"Home Win","prob":p_home,"odds":home_odds},
    {"market":"Draw","prob":p_draw,"odds":draw_odds},
    {"market":"Away Win","prob":p_away,"odds":away_odds},
    {"market":"Over 2.5","prob":p_over,"odds":over_odds},
    {"market":"Under 2.5","prob":p_under,"odds":under_odds},
    {"market":"BTTS Yes","prob":p_btts_yes,"odds":btts_yes_odds},
    {"market":"BTTS No","prob":p_btts_no,"odds":btts_no_odds}
]

for m in markets:
    m["fair"]=fair_odds(m["prob"])
    m["value"]=value_percentage(m["odds"], m["fair"])
    m["stake"]=kelly_stake(m["prob"], m["odds"], bankroll, kelly_fraction)

# ----- BEST MARKET -----
best_bet=max([m for m in markets if m["value"]>=min_value and m["stake"]>0], key=lambda x:x["value"], default=None)

if best_bet:
    st.subheader("✅ Best Market")
    st.success(f"{best_bet['market']} @ {best_bet['odds']} | Value +{best_bet['value']}% | Stake £{best_bet['stake']}")
else:
    st.info("No value bets meet criteria")

# ----- TELEGRAM NOTIFICATION -----
if st.button("📲 Send Telegram Notification") and best_bet:
    token="YOUR_BOT_TOKEN"
    chat_id="YOUR_CHAT_ID"
    msg=f"🔥 Today's Best Bet 🔥\n{home} vs {away}\n{best_bet['market']} @ {best_bet['odds']}\nValue +{best_bet['value']}%\nStake £{best_bet['stake']}"
    send_telegram(msg, token, chat_id)
    st.success("Telegram notification sent!")
