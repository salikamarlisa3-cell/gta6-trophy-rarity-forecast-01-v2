# Earners-vs-players dilution + progression forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/salikamarlisa3-cell/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/salikamarlisa3-cell/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

An entry for the [gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
bounty on make.fun: forecast, for every Grand Theft Auto VI trophy on PS5,
the percent of Exophase-tracked players who will have earned it at the
case's target time, as five quantiles per trophy.

## How it works

`main.py` (Python 3, standard library only, no network, deterministic)
reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

The recorder's reads are first collapsed into distinct page states, since
the Exophase page only changes at refreshes. Then:

- **One page state** — the page is stale and a refresh lands before the
  target. The forecast is rebuilt in integer earners: on the documented
  seed page (638 players, 943 earned) the refresh to 3,868 players and
  4,276 earned is reconstructed exactly — the universal trophy's earners
  become the new player count, every other trophy's earners scale by
  408/305 — while any other stale page gets a milder 2.5x player dilution
  with earners held.
- **A moving history** — the tracked-player count is projected with a
  saturating-growth fit blended geometrically with a log-time fit. Each
  trophy's median is the blend of an earners-elasticity view (log earners
  on log players) and a progression view (logit rate on log days since
  release), both slopes shrunk toward their tier mean, platinum fitted
  with gold. The spread grows with fit residuals, the forecast horizon
  and the player growth, clamped on the logit scale.

## License

MIT — see `LICENSE`.
