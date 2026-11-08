# ==============================================================================
# 07_aggregated_profiles.R  -  aggregated xG profiles that can assess players
#
# Cavus & Biecek (2022), Figures 3 and 4, compare teams and players with
# aggregated profiles: take every shot a player took, slide one feature across
# a grid while holding the rest fixed (a ceteris-paribus profile), and average
# the curves. This script rebuilds those figures with the paper's own model and
# code, builds an improved version, and measures which one assesses players
# better, using Messi, Lewandowski and Yilmaz in 2020-21 as the paper did.
#
# What is wrong with the published profiles, and what replaces it:
#
#   1. The model. Figures 3-4 use the random forest trained on ROSE
#      over-sampled data (over_model$model3). It learns on a ~50% goal rate,
#      so its "xG" runs far above the real 10.7%, and it memorises training
#      shots (06: worst held-out AUC of all 24 models, calibration error 0.16).
#      -> 02_model_utils.R's xgboost and its spline logistic regression. The
#      logistic regression is the best calibrated family in 03 (mean decile
#      gap 0.0027 vs 0.0038), with the largest gain in the top two deciles,
#      where most of a striker's shots sit. It is also smooth in distance and
#      angle, so its profiles have no tree steps.
#
#   2. In-sample explanation. The paper over-sampled all seven seasons, trained
#      on 80% of the result and then explained shots from that same data, most
#      of which the model had been trained on. -> Cross-fitting: matches are
#      split into K folds and every shot is explained by the model fitted
#      without its match.
#
#   3. Impossible shots. Distance and angle are both computed from the shot's
#      location, so they cannot vary independently. Holding the angle while
#      moving the distance (what a partial profile does) asks the model about
#      shots nobody can take: from 30 m no spot on the pitch sees the goal at
#      25 degrees. -> Geometry-consistent profiles: move the shot itself and
#      recompute both features from the new location.
#        distance profile: slide each shot along its own line to the goal
#        angle profile:    keep its distance and move it around the arc; only
#                          angles reachable from that distance are used
#
#   4. No uncertainty. 13 Schalke shots get the same unqualified curve as 178
#      Messi shots. -> Match-level bootstrap bands, and a permutation test for
#      "do these two profiles really differ?".
#
#   5. A profile alone cannot assess a player: the model has no player term,
#      so two players' curves differ only through the rest of their shots.
#      -> Expected vs actual goals by distance band and per player-season, with
#      intervals that say whether a gap is finishing or noise.
#
# Reads:  data/splits/train.csv, test.csv (validation_UNSEEN.csv only if
#         USE_VALIDATION). They must carry X and Y: run the updated 01 first.
# Writes: results/07_*.csv and results/figures/07_*.png
#
# Runtime is dominated by the paper's random forest: 500 fully grown trees on
# ~500k over-sampled rows, fitted 1 + K_FOLDS times. Expect roughly an hour
# with PAPER_CROSSFIT = TRUE, about a third of that with FALSE. Memory peaks
# at 7-8 GB while the first forest is built (the trees alone are ~5 GB), so
# close other programs on a 16 GB machine; 8 GB is not enough. Set
# FAST_MODE <- TRUE first to check everything runs (~7 minutes, ~1.5 GB).
#
# Requires 01_build_data.R (updated version) to have run.
# ==============================================================================

FAST_MODE      <- FALSE
FAST_MATCHES   <- 3000    # extra random matches kept in FAST_MODE (focus matches always kept)
USE_VALIDATION <- FALSE   # TRUE adds validation_UNSEEN.csv shots, once models are final
K_FOLDS        <- 5       # cross-fitting folds, grouped by match
PAPER_CROSSFIT <- TRUE    # also cross-fit the paper's forest (K more forest fits)
N_BOOT         <- 500     # match-level bootstrap replicates for profile bands
N_PERM         <- 1000    # permutations per profile-difference test
MIN_SHOTS      <- 30      # player-seasons with at least this many shots are scored

# The paper's examples. Understat labels 2020-21 as season 2020.
FOCUS_SEASON  <- 2020
FOCUS_PLAYERS <- c("Burak Yilmaz" = 8637, "Lionel Messi" = 2097,
                   "Robert Lewandowski" = 227)
FOCUS_MATCH   <- 15298    # Schalke 04 v Bayern Munich, 24 Jan 2021 (Figure 3)

# Grids for the improved profiles. 99.5% of shots are taken inside 39 m. The
# distance grid starts at 4 m because the paper's angle formula (kept as is, it
# is what every model here is trained on) folds back inside 3.66 m of the goal
# centre - half the goal width - so a shot sliding in along its line sees its
# "angle" climb to 90 degrees and fall again. Angles above ~80 degrees almost
# never occur in the data, so the model is guessing there.
DIST_GRID  <- seq(4, 40, by = 0.5)
ANGLE_GRID <- seq(1, 85, by = 1)
# Angle profiles are drawn only where enough of the group's shots can actually
# reach that angle from their own distance.
MIN_SUPPORT_SHOTS <- 10
MIN_SUPPORT_SHARE <- 0.10
DIST_BANDS <- c(0, 6, 12, 18, 24, Inf)    # metres; 6 = edge of the six-yard box

source("02_model_utils.R")
source("paper_setup.R")
library(tidyr)
library(ggplot2)

dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
t_start <- Sys.time()
stamp <- function(...) {
  cat(sprintf("[%5.1f min] ", as.numeric(difftime(Sys.time(), t_start,
                                                  units = "mins"))), ..., "\n",
      sep = "")
}

cat("\n==============================================================\n")
cat("07_aggregated_profiles.R  -  improved aggregated profiles\n")
cat("==============================================================\n")


# ---- 1. Data ------------------------------------------------------------------

# Every shot gets explained, so the analysis runs on a pool rather than on one
# split: train + test (+ validation if USE_VALIDATION). Nothing is tuned here -
# the models are the fixed specifications from 02 and the paper - and
# cross-fitting below guarantees no shot is scored by a model that saw it.

read_pool_part <- function(which) {
  df <- read_split(which)
  if (!all(c("X", "Y") %in% names(df))) {
    stop(basename(SPLITS[[which]]), " has no X / Y columns.\n",
         "Rerun the updated 01_build_data.R: same split, plus the shot location.")
  }
  df$split <- which
  df
}

pool <- bind_rows(lapply(c("train", "test", if (USE_VALIDATION) "validation"),
                         read_pool_part))

# Location in metres from the centre of the goal being attacked. These are the
# same quantities 01 used to build the two spatial features.
paper_angle <- function(dx, dy) {
  abs(atan((7.32 * dx) / (dx^2 + dy^2 - (7.32 / 2)^2)) * 180 / pi)
}
pool <- pool %>% mutate(dx = 105 - X * 105, dy = 34 - Y * 68)

geom_err <- max(abs(sqrt(pool$dx^2 + pool$dy^2) - pool$distanceToGoal),
                abs(paper_angle(pool$dx, pool$dy) - pool$angleToGoal))
if (!is.finite(geom_err) || geom_err > 1e-6) {
  stop("X / Y do not reproduce distanceToGoal / angleToGoal (max error ",
       signif(geom_err, 3), "). The split files are out of sync with 01.")
}

has_understat <- "xg_understat" %in% names(pool) && !all(is.na(pool$xg_understat))
season_label  <- sprintf("%d-%02d", FOCUS_SEASON, (FOCUS_SEASON + 1) %% 100)

# Focus groups: the paper's three players in the focus season, and the two
# teams in the focus match. A shot can sit in two groups (Lewandowski's shots
# at Schalke are Bayern's too), so membership is kept in its own table.
player_rows <- pool %>%
  filter(player_id %in% FOCUS_PLAYERS, season_start == FOCUS_SEASON)
match_rows  <- pool %>% filter(match_id == FOCUS_MATCH)

if (nrow(player_rows) == 0) stop("No focus-player shots in the pool.")
if (nrow(match_rows) == 0)  stop("Match ", FOCUS_MATCH, " is not in the pool.")

team_name <- function(side) {
  col <- if (side == "h") "h_team" else "a_team"
  nm  <- if (col %in% names(match_rows)) unique(match_rows[[col]]) else NA
  if (length(nm) != 1 || is.na(nm)) paste(if (side == "h") "Home" else "Away", "team")
  else nm
}
HOME_TEAM <- team_name("h")
AWAY_TEAM <- team_name("a")

group_map <- bind_rows(
  player_rows %>%
    transmute(row_id, match_id, kind = "player",
              group = names(FOCUS_PLAYERS)[match(player_id, FOCUS_PLAYERS)]),
  match_rows %>%
    transmute(row_id, match_id, kind = "match",
              group = ifelse(as.character(h_a) == "h", HOME_TEAM, AWAY_TEAM))
)
PLAYER_GROUPS <- names(FOCUS_PLAYERS)
TEAM_GROUPS   <- c(HOME_TEAM, AWAY_TEAM)

if ("player" %in% names(player_rows)) {
  found <- player_rows %>% distinct(player_id, player)
  cat("\nFocus players found:",
      paste(sprintf("%s (id %s)", found$player, found$player_id), collapse = ", "),
      "\n")
}

set.seed(SEED)
if (FAST_MODE) {
  others <- setdiff(unique(pool$match_id), group_map$match_id)
  keep <- unique(c(group_map$match_id,
                   others[sample.int(length(others), min(FAST_MATCHES, length(others)))]))
  pool <- pool %>% filter(match_id %in% keep)
  cat(sprintf("\nFAST_MODE: %d matches, %d shots.\n", length(keep), nrow(pool)))
}

cat(sprintf("\nPool: %d shots from %d matches (%s), goal rate %.2f%%\n",
            nrow(pool), n_distinct(pool$match_id),
            paste(unique(pool$split), collapse = " + "), 100 * mean(pool$status)))
print(as.data.frame(group_map %>% count(kind, group, name = "shots")))


# ---- 2. Cross-fitting folds ---------------------------------------------------

# Folds are whole matches, drawn once up front (fit_xgb() reseeds R's RNG, so
# drawing them later would make them depend on how many models were fitted).
# A shot shares game state with the rest of its match, so holding out shots but
# not their matches would still leak.

match_ids <- sort(unique(pool$match_id))
fold_of   <- setNames(sample(rep(seq_len(K_FOLDS), length.out = length(match_ids))),
                      match_ids)
pool$fold <- unname(fold_of[as.character(pool$match_id)])

cat("\nShots per fold:", paste(tabulate(pool$fold, K_FOLDS), collapse = ", "), "\n")


# ---- 3. The two model specifications -------------------------------------------

# The paper's model3: ROSE over-sampling of the training rows, then forester's
# make_ranger() - the paper's own code, patched to run (paper_setup.R) - with
# the arguments forester() passes it. Same seed placement as Rcodes.Rmd.
fit_paper_forest <- function(df) {
  d <- paper_resample(paper_frame(df), "over")
  assert_no_forester_prompt(d)
  set.seed(PAPER_SEED)
  e <- suppressMessages(PAPER$make_ranger(data = d, target = "status",
                                          type = "classification",
                                          fill_na = TRUE))
  list(model = e, rate_trained = mean(d$status), rate_true = mean(df$status),
       predict = function(nd) clip01(paper_predict(e, as.data.frame(nd))))
}

# Ours: fit_xgb() and fit_logreg() from 02, unchanged - the same models 03 and
# 04 evaluate.
#
# glm drops factor levels absent from its training rows, so a held-out match
# with such a level would stop predict(). Collapse them the way 03 and 04 do;
# xgboost's one-hot columns need no such step.
fit_logreg_aligned <- function(df) {
  f   <- fit_logreg(df)
  obs <- observed_levels(df)
  list(model = f$model,
       predict = function(nd) f$predict(align_to_training(nd, obs, verbose = FALSE)))
}


# ---- 4. Models that have seen every shot (how the paper's figures were made) ----

stamp("Fitting the paper's forest on every pool shot (as published) ...")
paper_in <- fit_paper_forest(pool)
stamp("  done. Fitting xgboost on every pool shot (for the memorisation check) ...")
ours_in  <- fit_xgb(pool)
stamp("  done. Fitting logistic regression on every pool shot ...")
logreg_in <- fit_logreg_aligned(pool)
stamp("  done.")


# ---- 5. Profile designs -------------------------------------------------------

# Every profile point is a modified copy of a real shot. Four kinds:
#
#   partial_distance / partial_angle   the paper's: one feature moved across
#       ingredients::ceteris_paribus()'s default grid (101 quantiles of the
#       explainer's data), everything else held. Averaging these per grid value
#       is exactly ingredients::aggregate_profiles(type = "partial").
#   gc_distance   the shot slides along the line from goal centre through its
#       own position; distance AND angle are recomputed from the new spot.
#   gc_angle      the shot keeps its distance and moves around the arc; only
#       angles reachable at that distance are kept.
#
# `feasible` marks points that correspond to a spot on the pitch. The paper's
# profiles average over all points regardless; the improved ones use only
# feasible points.

max_angle_at <- function(d) paper_angle(d, 0)          # straight in front of goal
min_angle_at <- function(d) {                          # widest spot still on the pitch
  dy <- pmin(d, 34)
  a  <- paper_angle(sqrt(pmax(d^2 - dy^2, 0)), dy)
  ifelse(is.nan(a), 0, a)
}
is_feasible <- function(d, a, tol = 1e-9) {
  a <= max_angle_at(d) + tol & a >= min_angle_at(d) - tol
}

n_bad <- sum(!is_feasible(pool$distanceToGoal, pool$angleToGoal, tol = 1e-6))
if (n_bad > 0) warning(n_bad, " observed shots fail the feasibility check.")

ingredients_grid <- function(v, data, n = 101) {
  unique(quantile(data[[v]], probs = seq(0, 1, length.out = n), names = FALSE))
}
PAPER_GRID <- list(
  distanceToGoal = ingredients_grid("distanceToGoal", paper_in$model$data),
  angleToGoal    = ingredients_grid("angleToGoal",    paper_in$model$data)
)

expand_on <- function(base, x) {
  out   <- base[rep(seq_len(nrow(base)), each = length(x)), , drop = FALSE]
  out$x <- rep(x, times = nrow(base))
  out
}

make_design <- function(shots) {
  base <- as.data.frame(shots)[, c("row_id", "fold", FEATURES, "dx", "dy")]

  pd <- expand_on(base, PAPER_GRID$distanceToGoal)
  pd$distanceToGoal <- pd$x
  pd$feasible <- is_feasible(pd$distanceToGoal, pd$angleToGoal)
  pd$profile  <- "partial_distance"

  pa <- expand_on(base, PAPER_GRID$angleToGoal)
  pa$angleToGoal <- pa$x
  pa$feasible <- is_feasible(pa$distanceToGoal, pa$angleToGoal)
  pa$profile  <- "partial_angle"

  gd <- expand_on(base, DIST_GRID)
  theta <- atan2(gd$dy, gd$dx)
  ndx <- gd$x * cos(theta)
  ndy <- gd$x * sin(theta)
  gd$distanceToGoal <- gd$x
  gd$angleToGoal    <- paper_angle(ndx, ndy)
  gd$feasible <- abs(ndy) <= 34 & ndx <= 105
  gd$profile  <- "gc_distance"

  ga <- expand_on(base, ANGLE_GRID)
  ga$angleToGoal <- ga$x
  ga$feasible <- is_feasible(ga$distanceToGoal, ga$angleToGoal)
  ga$profile  <- "gc_angle"

  out <- bind_rows(pd, pa, gd, ga)
  out$angleToGoal[!is.finite(out$angleToGoal)] <- 90
  out
}

focus_shots <- pool %>% filter(row_id %in% group_map$row_id)
design <- make_design(focus_shots)
cp <- design[, c("row_id", "fold", "profile", "x", "feasible")]

cat(sprintf("\nProfile design: %d focus shots x %d grid points = %d rows\n",
            nrow(focus_shots), nrow(design) / nrow(focus_shots), nrow(design)))


# ---- 6. Fit, cross-fit, predict -----------------------------------------------

# Two kinds of prediction per model: one per pool shot (for expected goals)
# and one per profile point (for the curves). In-sample models predict
# everything; each fold's model predicts only its held-out matches.

P <- tibble(row_id = pool$row_id, status = pool$status)
P$paper_insample <- paper_in$predict(pool)
P$ours_insample  <- ours_in$predict(pool)
cp$paper_insample <- paper_in$predict(design)
cp$ours_insample  <- ours_in$predict(design)
P$logreg_insample  <- logreg_in$predict(pool)
cp$logreg_insample <- logreg_in$predict(design)
rm(logreg_in)

# Check that the partial profiles here are the paper's: rerun Figure 3's own
# calls (ingredients::ceteris_paribus + aggregate_profiles) on the home side of
# the focus match and compare with the averages computed from the design.
ing_gap <- tryCatch({
  chk_ids <- group_map$row_id[group_map$group == HOME_TEAM]
  obs <- as.data.frame(focus_shots[focus_shots$row_id %in% chk_ids, FEATURES])
  cpi <- ingredients::ceteris_paribus(paper_in$model, obs,
                                      variables = "distanceToGoal")
  api <- ingredients::aggregate_profiles(cpi, variables = "distanceToGoal",
                                         type = "partial")
  mine <- cp %>%
    filter(profile == "partial_distance", row_id %in% chk_ids) %>%
    group_by(x) %>%
    summarise(ap = mean(paper_insample), .groups = "drop")
  max(abs(api$`_yhat_` - mine$ap[match(round(api$`_x_`, 8), round(mine$x, 8))]))
}, error = function(e) {
  cat("  (ingredients check skipped:", conditionMessage(e), ")\n")
  NA_real_
})
if (!is.na(ing_gap)) {
  cat(sprintf("\nCheck: partial profiles here vs ingredients::aggregate_profiles on %s's shots: max difference %.2g\n",
              HOME_TEAM, ing_gap))
  if (ing_gap > 1e-6) warning("Partial profiles do not match ingredients. Investigate before using them.")
}

rm(paper_in)
invisible(gc())

P$ours_crossfit <- NA_real_
cp$ours_crossfit <- NA_real_
P$logreg_crossfit  <- NA_real_
cp$logreg_crossfit <- NA_real_
if (PAPER_CROSSFIT) {
  P$paper_crossfit    <- NA_real_
  P$paper_crossfit_pc <- NA_real_
  cp$paper_crossfit   <- NA_real_
  cp$paper_crossfit_pc <- NA_real_
}

for (k in seq_len(K_FOLDS)) {
  tr_k  <- pool[pool$fold != k, ]
  te_i  <- which(pool$fold == k)
  cp_i  <- which(design$fold == k)

  stamp(sprintf("Fold %d/%d: xgboost ...", k, K_FOLDS))
  f <- fit_xgb(tr_k)
  P$ours_crossfit[te_i]  <- f$predict(pool[te_i, ])
  if (length(cp_i)) cp$ours_crossfit[cp_i] <- f$predict(design[cp_i, ])
  rm(f)

  stamp(sprintf("Fold %d/%d: logistic regression ...", k, K_FOLDS))
  f <- fit_logreg_aligned(tr_k)
  P$logreg_crossfit[te_i] <- f$predict(pool[te_i, ])
  if (length(cp_i)) cp$logreg_crossfit[cp_i] <- f$predict(design[cp_i, ])
  rm(f)

  if (PAPER_CROSSFIT) {
    stamp(sprintf("Fold %d/%d: the paper's forest ...", k, K_FOLDS))
    f <- fit_paper_forest(tr_k)
    p_te <- f$predict(pool[te_i, ])
    P$paper_crossfit[te_i]    <- p_te
    P$paper_crossfit_pc[te_i] <- prior_correct(p_te, f$rate_trained, f$rate_true)
    if (length(cp_i)) {
      p_cp <- f$predict(design[cp_i, ])
      cp$paper_crossfit[cp_i]    <- p_cp
      cp$paper_crossfit_pc[cp_i] <- prior_correct(p_cp, f$rate_trained, f$rate_true)
    }
    rm(f, p_te)
  }
  rm(tr_k)
  invisible(gc())
}
stamp("All models fitted.")

if (has_understat) P$understat <- pool$xg_understat

MODEL_COLS <- intersect(c("paper_insample", "paper_crossfit", "paper_crossfit_pc",
                          "ours_insample", "ours_crossfit", "logreg_insample",
                          "logreg_crossfit", "understat"), names(P))
MODEL_LABELS <- c(
  paper_insample    = "Paper's model, as published (has seen these shots)",
  paper_crossfit    = "Paper's model, cross-fitted",
  paper_crossfit_pc = "Paper's model, cross-fitted + prior-corrected",
  ours_insample     = "Improved model, in-sample",
  ours_crossfit     = "Improved: cross-fitted xgboost",
  logreg_insample   = "Logistic regression, in-sample",
  logreg_crossfit   = "Improved: cross-fitted logistic regression",
  understat         = "Understat xG (reference, in-sample)"
)


# ---- 7. Shot-level sanity check -----------------------------------------------

# Cross-fitted xgboost should land where 03 put xgboost on test.csv (log loss
# ~0.268), cross-fitted logistic regression near 03's ~0.270, and the
# cross-fitted paper forest where 06 put paper_ranger_over (~0.38). If not,
# something upstream changed.

pool_metrics <- bind_rows(lapply(MODEL_COLS, function(m) {
  ok <- !is.na(P[[m]])
  metrics(P$status[ok], P[[m]][ok]) %>% mutate(model = m)
})) %>%
  select(model, n, auc, brier, log_loss, ece, mean_pred, actual_rate, calib_gap)

cat("\n--- Every pool shot, by model ---\n")
print(as.data.frame(pool_metrics), digits = 4)
write_csv(pool_metrics, "results/07_pool_metrics.csv")


# ---- 8. Expected goals with intervals -----------------------------------------

# Under a model, a player's goal count is a sum of independent Bernoulli shots:
# a Poisson-binomial. Its exact distribution gives a 95% interval for goals and
# a p-value for "scored more / fewer than this model expects". For a calibrated
# model most player-seasons should land inside their interval (not all: real
# finishing skill exists); for a miscalibrated one, few will.

pb_summary <- function(p, g, level = 0.95) {
  if (anyNA(p)) return(c(lo = NA, hi = NA, p_value = NA))
  dens <- 1
  for (q in p) dens <- c(dens * (1 - q), 0) + c(0, dens * q)
  cdf  <- cumsum(dens)
  a    <- (1 - level) / 2
  p_le <- cdf[g + 1]
  p_ge <- 1 - (if (g >= 1) cdf[g] else 0)
  c(lo = which(cdf >= a)[1] - 1, hi = which(cdf >= 1 - a)[1] - 1,
    p_value = min(1, 2 * min(p_le, p_ge)))
}

score_groups <- function(idx_list, df, cols) {
  bind_rows(lapply(names(idx_list), function(key) {
    i <- idx_list[[key]]
    g <- sum(df$status[i])
    bind_rows(lapply(cols, function(m) {
      p <- df[[m]][i]
      s <- pb_summary(p, g)
      tibble(key = key, model = m, shots = length(i), goals = g,
             xg = sum(p), lo = s[["lo"]], hi = s[["hi"]], p_value = s[["p_value"]])
    }))
  }))
}

scored <- pool %>%
  select(row_id, player_id, season_start, situation, any_of("player")) %>%
  left_join(P, by = "row_id")

# 8a. Every player-season with MIN_SHOTS or more shots.
ps_idx <- split(seq_len(nrow(scored)),
                paste(scored$player_id, scored$season_start, sep = "_"))
ps_idx <- ps_idx[lengths(ps_idx) >= MIN_SHOTS]

stamp(sprintf("Scoring %d player-seasons with >= %d shots ...", length(ps_idx), MIN_SHOTS))
player_seasons <- score_groups(ps_idx, scored, MODEL_COLS) %>%
  separate(key, into = c("player_id", "season_start"), sep = "_", convert = TRUE) %>%
  mutate(inside = goals >= lo & goals <= hi)
if ("player" %in% names(scored)) {
  # distinct(), not slice(): xgboost exports its own slice() and masks dplyr's.
  player_seasons <- player_seasons %>%
    left_join(scored %>% select(player_id, player) %>%
                distinct(player_id, .keep_all = TRUE),
              by = "player_id")
}

season_calibration <- player_seasons %>%
  group_by(model) %>%
  summarise(player_seasons = n(),
            total_xg       = sum(xg),
            total_goals    = sum(goals),
            xg_per_goal    = total_xg / total_goals,
            mean_bias      = mean(xg - goals),
            mae            = mean(abs(xg - goals)),
            rmse           = sqrt(mean((xg - goals)^2)),
            correlation    = cor(xg, goals),
            coverage_95    = mean(inside, na.rm = TRUE),
            .groups = "drop") %>%
  mutate(model = factor(model, levels = MODEL_COLS)) %>%
  arrange(model)

cat("\n--- Player-season goal totals vs expected goals ---\n")
cat("    coverage_95 = share of player-seasons whose goals fall inside the\n")
cat("    model's own 95% interval; ~0.9 or above is what a calibrated model gives\n")
print(as.data.frame(season_calibration), digits = 3)
write_csv(player_seasons, "results/07_player_season_xg.csv")
write_csv(season_calibration, "results/07_player_season_calibration.csv")

# 8b. The focus players: all shots, and without penalties (a penalty's spot is
#     fixed by the rules, so it says nothing about chance creation).
focus_idx <- list()
for (pl in PLAYER_GROUPS) {
  ids <- group_map$row_id[group_map$group == pl]
  i   <- which(scored$row_id %in% ids)
  focus_idx[[paste(pl, "all shots", sep = "|")]] <- i
  focus_idx[[paste(pl, "non-penalty", sep = "|")]] <-
    i[as.character(scored$situation[i]) != "Penalty"]
}
assessment <- score_groups(focus_idx, scored, MODEL_COLS) %>%
  separate(key, into = c("player", "shots_used"), sep = "\\|") %>%
  mutate(goals_minus_xg = goals - xg,
         verdict = case_when(is.na(lo)  ~ NA_character_,
                             goals > hi ~ "scored MORE than the model allows",
                             goals < lo ~ "scored FEWER than the model allows",
                             TRUE       ~ "within the model's 95% range"))

cat(sprintf("\n--- Assessment of the focus players, %s ---\n", season_label))
print(as.data.frame(assessment %>%
                      filter(model %in% c("paper_insample", "ours_crossfit",
                                          "logreg_crossfit", "understat")) %>%
                      select(player, shots_used, model, shots, goals, xg, lo, hi, verdict)),
      digits = 3, right = FALSE)
write_csv(assessment, "results/07_player_assessment.csv")


# ---- 9. Expected vs actual, by distance band ----------------------------------

# The most direct test of "assessing a player": for the shots he actually took
# from each distance, how often did he score, and how often did each model say
# he would? Conditional on real shots on both sides, so the comparison is fair.

wilson <- function(x, n, z = 1.96) {
  p <- x / n
  den <- 1 + z^2 / n
  centre <- (p + z^2 / (2 * n)) / den
  half <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / den
  list(lo = pmax(0, centre - half), hi = pmin(1, centre + half))
}

band_labels <- function(b) {
  lab <- paste0(head(b, -1), "-", tail(b, -1), " m")
  lab[is.infinite(tail(b, -1))] <- paste0(b[length(b) - 1], "+ m")
  lab
}

band_cols <- intersect(c("paper_insample", "paper_crossfit_pc", "ours_crossfit",
                         "logreg_crossfit", "understat"), MODEL_COLS)
by_band <- scored %>%
  inner_join(group_map %>% filter(kind == "player") %>% select(row_id, group),
             by = "row_id") %>%
  left_join(pool %>% select(row_id, distanceToGoal), by = "row_id") %>%
  mutate(band = cut(distanceToGoal, DIST_BANDS, right = FALSE,
                    labels = band_labels(DIST_BANDS))) %>%
  group_by(group, band) %>%
  summarise(shots = n(), goals = sum(status),
            across(all_of(band_cols), mean), .groups = "drop") %>%
  mutate(actual = goals / shots,
         actual_lo = wilson(goals, shots)$lo,
         actual_hi = wilson(goals, shots)$hi)

write_csv(by_band, "results/07_expected_vs_actual_by_band.csv")


# ---- 10. Aggregated profiles, with bands --------------------------------------

# For each group, model and profile kind: the per-shot profiles as a matrix
# (shots x grid points), averaged per grid point. The band resamples whole
# matches (single-match groups resample shots), so it reflects how much the
# curve would move with a different sample of the same player's chances.

CP_MODELS <- intersect(c("paper_insample", "paper_crossfit", "paper_crossfit_pc",
                         "ours_insample", "ours_crossfit", "logreg_insample",
                         "logreg_crossfit"), names(cp))
PROFILES  <- c(partial_distance = FALSE, partial_angle = FALSE,
               gc_distance = TRUE, gc_angle = TRUE)   # value = feasible points only

cp_matrix <- function(model, prof, ids) {
  sub <- cp[cp$profile == prof & cp$row_id %in% ids, ]
  xs  <- sort(unique(cp$x[cp$profile == prof]))
  v   <- sub[[model]]
  if (PROFILES[[prof]]) v[!sub$feasible] <- NA
  M <- matrix(NA_real_, length(ids), length(xs))
  M[cbind(match(sub$row_id, ids), match(sub$x, xs))] <- v
  list(M = M, x = xs)
}

col_means <- function(M) {
  m <- colMeans(M, na.rm = TRUE)
  m[is.nan(m)] <- NA
  m
}

group_info <- group_map %>%
  group_by(group, kind) %>%
  summarise(ids = list(row_id), clusters = list(match_id), .groups = "drop")

set.seed(SEED)
stamp("Aggregating profiles and bootstrapping bands ...")
profiles <- list()
for (gi in seq_len(nrow(group_info))) {
  ids <- group_info$ids[[gi]]
  cl  <- if (group_info$kind[gi] == "match") ids else group_info$clusters[[gi]]
  cl_rows <- split(seq_along(ids), cl)
  min_support <- max(MIN_SUPPORT_SHOTS, ceiling(MIN_SUPPORT_SHARE * length(ids)))
  for (m in CP_MODELS) for (prof in names(PROFILES)) {
    cm <- cp_matrix(m, prof, ids)
    boot <- matrix(NA_real_, N_BOOT, length(cm$x))
    for (b in seq_len(N_BOOT)) {
      rows <- unlist(cl_rows[sample.int(length(cl_rows), replace = TRUE)],
                     use.names = FALSE)
      boot[b, ] <- col_means(cm$M[rows, , drop = FALSE])
    }
    support <- colSums(!is.na(cm$M))
    profiles[[length(profiles) + 1]] <- tibble(
      group = group_info$group[gi], kind = group_info$kind[gi],
      model = m, profile = prof, x = cm$x,
      ap = col_means(cm$M),
      lo = apply(boot, 2, quantile, 0.025, na.rm = TRUE, names = FALSE),
      hi = apply(boot, 2, quantile, 0.975, na.rm = TRUE, names = FALSE),
      support = support,
      shown = !PROFILES[[prof]] | support >= min_support
    )
  }
}
profiles <- bind_rows(profiles)
write_csv(profiles, "results/07_profiles.csv")


# ---- 11. Do two profiles really differ? ---------------------------------------

# Permutation test. Statistic: mean absolute gap between the two aggregated
# profiles over the grid. Shuffling group labels across the pooled shots shows
# how big that gap gets by chance with groups of these sizes.

perm_test <- function(model, prof, g1, g2) {
  ids1 <- group_map$row_id[group_map$group == g1]
  ids2 <- group_map$row_id[group_map$group == g2]
  M1 <- cp_matrix(model, prof, ids1)$M
  M2 <- cp_matrix(model, prof, ids2)$M
  need <- function(M) max(MIN_SUPPORT_SHOTS, ceiling(MIN_SUPPORT_SHARE * nrow(M)))
  cols <- if (PROFILES[[prof]]) {
    which(colSums(!is.na(M1)) >= need(M1) & colSums(!is.na(M2)) >= need(M2))
  } else seq_len(ncol(M1))
  M  <- rbind(M1[, cols, drop = FALSE], M2[, cols, drop = FALSE])
  n1 <- nrow(M1)
  stat <- function(i1) {
    mean(abs(col_means(M[i1, , drop = FALSE]) - col_means(M[-i1, , drop = FALSE])),
         na.rm = TRUE)
  }
  obs  <- stat(seq_len(n1))
  perm <- replicate(N_PERM, stat(sample.int(nrow(M), n1)))
  tibble(model = model, profile = prof, group_1 = g1, group_2 = g2,
         n_1 = n1, n_2 = nrow(M2), mean_abs_gap = obs,
         p_value = (1 + sum(perm >= obs - 1e-12)) / (1 + N_PERM))
}

pairs <- c(combn(PLAYER_GROUPS, 2, simplify = FALSE), list(TEAM_GROUPS))
set.seed(SEED)
diff_tests <- bind_rows(lapply(pairs, function(pr) {
  bind_rows(perm_test("paper_insample",  "partial_distance", pr[1], pr[2]),
            perm_test("ours_crossfit",   "gc_distance",      pr[1], pr[2]),
            perm_test("logreg_crossfit", "gc_distance",      pr[1], pr[2]))
}))
cat("\n--- Do the profiles of two groups differ? (permutation tests) ---\n")
print(as.data.frame(diff_tests), digits = 3)
write_csv(diff_tests, "results/07_profile_difference_tests.csv")


# ---- 12. How many profile points are impossible shots? -----------------------

feasibility <- bind_rows(lapply(split(group_map$row_id, group_map$group), function(ids) {
  cp[cp$row_id %in% ids, ] %>%
    group_by(profile) %>%
    summarise(points = n(), impossible = mean(!feasible), .groups = "drop")
}), .id = "group") %>%
  mutate(handled = ifelse(PROFILES[profile],
                          "excluded from the improved profile",
                          "predicted and averaged in, as published"))

cat("\n--- Share of the paper's profile points that are not a spot on the pitch ---\n")
print(as.data.frame(feasibility %>%
                      filter(!PROFILES[profile]) %>%
                      select(group, profile, impossible) %>%
                      pivot_wider(names_from = profile, values_from = impossible)),
      digits = 3)
cat(sprintf(paste0("The improved profiles never use such points: the distance profile drops the %.1f%% of\n",
                   "slides that would leave the pitch sideways, the angle profile keeps only reachable angles.\n"),
            100 * mean(cp$feasible[cp$profile == "gc_distance"] == FALSE)))
write_csv(feasibility, "results/07_profile_feasibility.csv")


# ---- 13. Memorisation ---------------------------------------------------------

# A model that has memorised its training shots gives goals it has seen a
# higher probability, and misses it has seen a lower one, than it would give
# the same shots unseen. The gap between those two shifts is the memorisation
# index: ~0 means the model explains shots the same way whether or not it saw
# them; large means an in-sample profile partly replays what actually happened.

mem_pairs <- list(paper_forest = c("paper_insample",  "paper_crossfit"),
                  ours_xgboost = c("ours_insample",   "ours_crossfit"),
                  ours_logreg  = c("logreg_insample", "logreg_crossfit"))
mem_pairs <- mem_pairs[vapply(mem_pairs, function(p) all(p %in% names(P)), logical(1))]

memorisation <- bind_rows(lapply(names(mem_pairs), function(nm) {
  pr <- mem_pairs[[nm]]
  d  <- P[[pr[1]]] - P[[pr[2]]]
  focus_i <- P$row_id %in% group_map$row_id[group_map$kind == "player"]
  tibble(model = nm,
         shift_goals  = mean(d[P$status == 1]),
         shift_misses = mean(d[P$status == 0]),
         memorisation_index = shift_goals - shift_misses,
         auc_insample = as.numeric(auc(roc(P$status, P[[pr[1]]], quiet = TRUE))),
         auc_crossfit = as.numeric(auc(roc(P$status, P[[pr[2]]], quiet = TRUE))),
         focus_index  = mean(d[focus_i & P$status == 1]) -
                        mean(d[focus_i & P$status == 0]))
}))

cat("\n--- Memorisation: in-sample minus cross-fitted prediction ---\n")
print(as.data.frame(memorisation), digits = 3)
write_csv(memorisation, "results/07_memorisation.csv")


# ---- 14. Figures --------------------------------------------------------------

INK   <- "#0b0b0b"
INK_2 <- "#52514e"
MUTED <- "#898781"
GRID  <- "#e1e0d9"
PLAYER_COLS <- setNames(c("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                          "#e87ba4")[seq_along(PLAYER_GROUPS)], PLAYER_GROUPS)
TEAM_COLS   <- setNames(c("#2a78d6", "#e34948"), TEAM_GROUPS)
PAPER_COL   <- "#e34948"
OURS_COL    <- "#6250d6"
LOGREG_COL  <- "#1baf7a"

theme_07 <- function() {
  theme_minimal(base_size = 12) +
    theme(plot.title       = element_text(face = "bold", colour = INK, size = 14),
          plot.subtitle    = element_text(colour = INK_2, size = 10.5,
                                          margin = margin(b = 8)),
          plot.caption     = element_text(colour = MUTED, size = 9, hjust = 0),
          plot.title.position = "plot",
          plot.caption.position = "plot",
          axis.title       = element_text(colour = INK_2),
          axis.text        = element_text(colour = INK_2),
          panel.grid.major = element_line(colour = GRID, linewidth = 0.3),
          panel.grid.minor = element_blank(),
          strip.text       = element_text(face = "bold", colour = INK, hjust = 0,
                                          size = 10.5),
          legend.position  = "bottom",
          legend.title     = element_blank(),
          legend.text      = element_text(colour = INK_2))
}

save_fig <- function(plot, file, width = 11, height = 6) {
  print(plot)
  ggsave(file.path("results/figures", file), plot, width = width,
         height = height, dpi = 150, bg = "white")
}

ap_at <- function(g, m, prof, at) {
  d <- profiles %>% filter(group == g, model == m, profile == prof, shown, !is.na(ap))
  if (nrow(d) < 2) return(NA_real_)
  approx(d$x, d$ap, xout = at, rule = 2)$y
}
pct <- function(x) sprintf("%.0f%%", 100 * x)

PANEL_PAPER <- "Paper: over-sampled random forest, partial profile,\nexplaining shots it was trained on"
PANEL_OURS  <- "Improved: cross-fitted xgboost, geometry-consistent profile,\n95% match-bootstrap band"
PANEL_LOGREG <- "Improved: cross-fitted logistic regression,\ngeometry-consistent profile, 95% match-bootstrap band"

profile_panels <- function(groups, prof_paper, prof_ours) {
  bind_rows(
    profiles %>% filter(model == "paper_insample", profile == prof_paper,
                        group %in% groups) %>%
      mutate(panel = PANEL_PAPER, lo = NA_real_, hi = NA_real_),
    profiles %>% filter(model == "ours_crossfit", profile == prof_ours,
                        group %in% groups, shown) %>%
      mutate(panel = PANEL_OURS),
    profiles %>% filter(model == "logreg_crossfit", profile == prof_ours,
                        group %in% groups, shown) %>%
      mutate(panel = PANEL_LOGREG)
  ) %>%
    mutate(panel = factor(panel, levels = c(PANEL_PAPER, PANEL_OURS, PANEL_LOGREG)),
           group = factor(group, levels = groups))
}

dist_995 <- quantile(pool$distanceToGoal, 0.995, names = FALSE)
y_top <- function(d) max(c(d$ap, d$hi), na.rm = TRUE) * 1.05

# 14a. Distance profiles of the three players (the paper's Figure 4, left).
fd <- profile_panels(PLAYER_GROUPS, "partial_distance", "gc_distance")
ref_player <- if ("Lionel Messi" %in% PLAYER_GROUPS) "Lionel Messi" else PLAYER_GROUPS[1]
fig_dist <- ggplot(fd, aes(x, ap, colour = group, fill = group)) +
  geom_rect(data = data.frame(panel = factor(PANEL_PAPER, levels = levels(fd$panel))),
            aes(xmin = dist_995, xmax = Inf, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = "#f0efec") +
  geom_text(data = data.frame(panel = factor(PANEL_PAPER, levels = levels(fd$panel))),
            aes(x = dist_995 + 1, y = y_top(fd) * 0.97,
                label = "fewer than 0.5% of shots\nare taken from out here"),
            inherit.aes = FALSE, hjust = 0, vjust = 1, size = 3.2, colour = INK_2) +
  geom_ribbon(data = fd %>% filter(!is.na(lo)), aes(ymin = lo, ymax = hi),
              alpha = 0.18, colour = NA) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ panel, scales = "free_x") +
  scale_colour_manual(values = PLAYER_COLS) +
  scale_fill_manual(values = PLAYER_COLS) +
  scale_y_continuous(labels = function(v) sprintf("%.2f", v),
                     limits = c(0, y_top(fd)), expand = expansion(mult = c(0, 0.02))) +
  labs(title = sprintf("Distance profiles, %s season", season_label),
       subtitle = sprintf(paste0(
         "%s's shots moved to 35 m: the paper's model says %s of them would go in, xgboost %s, logistic regression %s.\n",
         "The paper's curve is still at %s at %.0f m, because it holds each shot's angle fixed while the distance grows."),
         ref_player,
         pct(ap_at(ref_player, "paper_insample", "partial_distance", 35)),
         pct(ap_at(ref_player, "ours_crossfit", "gc_distance", 35)),
         pct(ap_at(ref_player, "logreg_crossfit", "gc_distance", 35)),
         pct(ap_at(ref_player, "paper_insample", "partial_distance", max(PAPER_GRID$distanceToGoal))),
         max(PAPER_GRID$distanceToGoal)),
       x = "Distance to goal (m)", y = "Average predicted chance of a goal",
       caption = "Left: ingredients::aggregate_profiles(type = \"partial\") on the paper's model3, as in Figure 4. Middle and right: each shot slides along its own line to goal; angle recomputed.") +
  theme_07()
save_fig(fig_dist, "07_profiles_distance_players.png", width = 16, height = 6.2)

# 14b. Angle profiles (the paper's Figure 4, right).
# Both panels turn up again below ~5 degrees, as the paper's own Figure 4 does.
# That comes from the data, not the method: Understat logs ~200 shots from
# 26-40 m at angles under 6 degrees (near the corner flag), and they convert at
# 15-28%, far above shots from the same distance at 10 degrees. Most look like
# mislocated events; every model trained on this archive learns them.
fa <- profile_panels(PLAYER_GROUPS, "partial_angle", "gc_angle")
fig_angle <- ggplot(fa, aes(x, ap, colour = group, fill = group)) +
  geom_ribbon(data = fa %>% filter(!is.na(lo)), aes(ymin = lo, ymax = hi),
              alpha = 0.18, colour = NA) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ panel, scales = "free_x") +
  scale_colour_manual(values = PLAYER_COLS) +
  scale_fill_manual(values = PLAYER_COLS) +
  scale_y_continuous(labels = function(v) sprintf("%.2f", v),
                     limits = c(0, y_top(fa)), expand = expansion(mult = c(0, 0.02))) +
  labs(title = sprintf("Angle profiles, %s season", season_label),
       subtitle = paste0(
         "Middle and right: each shot keeps its distance and moves around the arc, and only angles reachable from that distance count\n",
         "(lines stop where fewer than ", MIN_SUPPORT_SHOTS, " shots, or ",
         pct(MIN_SUPPORT_SHARE), " of a player's shots, can reach the angle). ",
         "Left: every shot is pushed to every angle, reachable or not."),
       x = "Angle to goal (degrees)", y = "Average predicted chance of a goal") +
  theme_07()
save_fig(fig_angle, "07_profiles_angle_players.png", width = 16, height = 6.2)

# 14c. Expected vs actual by distance band - the headline for assessing players.
band_series <- c(paper_insample  = "Paper's model (as published)",
                 actual          = "Actual, with 95% interval",
                 ours_crossfit   = "Improved xgboost (cross-fitted)",
                 logreg_crossfit = "Improved logistic regression (cross-fitted)",
                 understat       = "Understat xG (reference)")
band_series <- band_series[names(band_series) %in% c("actual", band_cols)]
band_offset <- c(paper_insample = -0.24, actual = 0, ours_crossfit = 0.12,
                 logreg_crossfit = 0.24, understat = 0.36)
band_cols_plot <- c(paper_insample = PAPER_COL, actual = INK,
                    ours_crossfit = OURS_COL, logreg_crossfit = LOGREG_COL,
                    understat = MUTED)
band_shapes <- c(paper_insample = 16, actual = 15, ours_crossfit = 17,
                 logreg_crossfit = 18, understat = 5)

bb <- by_band %>%
  select(group, band, shots, goals, actual, actual_lo, actual_hi,
         any_of(setdiff(names(band_series), "actual"))) %>%
  pivot_longer(c(actual, any_of(setdiff(names(band_series), "actual"))),
               names_to = "series", values_to = "rate") %>%
  filter(series %in% names(band_series)) %>%
  mutate(xpos = as.integer(band) + band_offset[series],
         series = factor(series, levels = names(band_series)),
         group = factor(group, levels = PLAYER_GROUPS))

player_tot <- assessment %>%
  filter(shots_used == "all shots") %>%
  group_by(player) %>%
  summarise(shots = first(shots), goals = first(goals),
            xg_paper = xg[model == "paper_insample"],
            xg_ours  = xg[model == "ours_crossfit"],
            xg_logreg = xg[model == "logreg_crossfit"], .groups = "drop")
strip_lab <- setNames(sprintf("%s: %d shots, %d goals", player_tot$player,
                              player_tot$shots, player_tot$goals), player_tot$player)

tot_g <- sum(player_tot$goals)
cells <- by_band %>% filter(shots > 0)
n_out <- function(col, dir) {
  if (!col %in% names(cells)) return(NA_integer_)
  if (dir == "above") sum(cells[[col]] > cells$actual_hi) else sum(cells[[col]] < cells$actual_lo)
}
fig_band <- ggplot(bb, aes(xpos, rate, colour = series, shape = series)) +
  geom_errorbar(data = bb %>% filter(series == "actual"),
                aes(ymin = actual_lo, ymax = actual_hi), width = 0.12,
                linewidth = 0.6, show.legend = FALSE) +
  geom_point(size = 2.6) +
  geom_text(data = bb %>% filter(series == "actual"),
            aes(y = -0.035, label = paste0("n=", shots)), colour = MUTED,
            size = 3, show.legend = FALSE) +
  facet_wrap(~ group, nrow = 1, labeller = as_labeller(strip_lab)) +
  scale_x_continuous(breaks = seq_along(levels(by_band$band)),
                     labels = levels(by_band$band)) +
  scale_y_continuous(labels = function(v) pct(v), breaks = seq(0, 1, by = 0.1),
                     limits = c(-0.06, NA)) +
  scale_colour_manual(values = band_cols_plot, labels = band_series, breaks = names(band_series)) +
  scale_shape_manual(values = band_shapes, labels = band_series, breaks = names(band_series)) +
  labs(title = sprintf("How often each player scored from each distance, and how often each model said he would (%s)",
                       season_label),
       subtitle = sprintf(paste0(
         "Across the three players: %d goals. The paper's model expected %.0f, xgboost %.0f, logistic regression %.0f.\n",
         "Of %d player-distance cells, the paper's model sits above the actual 95%% interval in %d and below it in %d; ",
         "xgboost above in %d, below in %d; logistic regression above in %d, below in %d."),
         tot_g, sum(player_tot$xg_paper), sum(player_tot$xg_ours),
         sum(player_tot$xg_logreg), nrow(cells),
         n_out("paper_insample", "above"), n_out("paper_insample", "below"),
         n_out("ours_crossfit", "above"), n_out("ours_crossfit", "below"),
         n_out("logreg_crossfit", "above"), n_out("logreg_crossfit", "below")),
       x = "Distance to goal", y = "Goals per shot") +
  theme_07() +
  theme(axis.text.x = element_text(size = 8.5))
save_fig(fig_band, "07_expected_vs_actual_by_distance.png", width = 13, height = 5.8)

# 14d. The season verdicts: each model's 95% range for a player's goals.
assess_models <- intersect(c("paper_insample", "ours_crossfit", "logreg_crossfit",
                             "understat"), MODEL_COLS)
assess_lab <- c(paper_insample  = "Paper's model\n(as published)",
                ours_crossfit   = "Improved xgboost\n(cross-fitted)",
                logreg_crossfit = "Improved logistic\nregression (cross-fitted)",
                understat       = "Understat xG\n(reference)")
av <- assessment %>%
  filter(shots_used == "all shots", model %in% assess_models) %>%
  mutate(model  = factor(model, levels = rev(assess_models)),
         player = factor(player, levels = PLAYER_GROUPS))
av_goals <- av %>% distinct(player, goals)
verdict_line <- function(pl) {
  a <- av %>% filter(player == pl)
  if (!all(c("paper_insample", "ours_crossfit", "logreg_crossfit") %in% a$model)) return(NULL)
  short <- function(r) ifelse(r$goals > r$hi, "above par",
                       ifelse(r$goals < r$lo, "below par", "in line"))
  p <- a %>% filter(model == "paper_insample")
  o <- a %>% filter(model == "ours_crossfit")
  l <- a %>% filter(model == "logreg_crossfit")
  sprintf("%s scored %d. Paper's model: %.0f expected (95%% range %g-%g), %s. xgboost: %.0f (%g-%g), %s. Logistic regression: %.0f (%g-%g), %s.",
          pl, p$goals, p$xg, p$lo, p$hi, short(p), o$xg, o$lo, o$hi, short(o),
          l$xg, l$lo, l$hi, short(l))
}
verdicts <- unlist(lapply(intersect(c("Lionel Messi", "Robert Lewandowski"), PLAYER_GROUPS),
                          verdict_line))
if (length(verdicts) == 0) verdicts <- verdict_line(PLAYER_GROUPS[1])

fig_assess <- ggplot(av, aes(y = model)) +
  geom_vline(data = av_goals, aes(xintercept = goals), colour = INK, linewidth = 0.8) +
  geom_segment(aes(x = lo, xend = hi, yend = model, colour = model),
               linewidth = 5, alpha = 0.3, lineend = "round") +
  geom_point(aes(x = xg, colour = model, shape = model), size = 3) +
  geom_label(aes(x = xg, label = sprintf("%.0f expected (%g-%g)", xg, lo, hi)),
             vjust = -0.55, size = 3.1, colour = INK_2, fill = "white",
             label.size = 0, label.padding = unit(0.12, "lines")) +
  geom_text(data = av_goals, aes(x = goals, y = Inf, label = paste("scored", goals)),
            hjust = -0.08, vjust = 1.4, colour = INK, size = 3.4, fontface = "bold") +
  facet_wrap(~ player, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = c(paper_insample = PAPER_COL, ours_crossfit = OURS_COL,
                                 logreg_crossfit = LOGREG_COL, understat = MUTED),
                      guide = "none") +
  scale_shape_manual(values = c(paper_insample = 16, ours_crossfit = 17,
                                logreg_crossfit = 18, understat = 5),
                     guide = "none") +
  scale_y_discrete(labels = assess_lab, expand = expansion(add = c(0.6, 1.1))) +
  scale_x_continuous(expand = expansion(mult = c(0.03, 0.12))) +
  expand_limits(x = 0) +
  labs(title = sprintf("What each model concludes about these three players' %s seasons", season_label),
       subtitle = paste0(paste(verdicts, collapse = "\n"),
                         "\nBar: the model's 95% range for the goals these exact shots should produce. ",
                         "Point: its expected goals. Black line: goals actually scored."),
       x = "Goals", y = NULL) +
  theme_07() +
  theme(panel.grid.major.y = element_blank())
save_fig(fig_assess, "07_player_assessment.png", width = 14, height = 6.4)

# 14e. Every player-season: expected vs actual goals.
cal_models <- intersect(c("paper_insample", "paper_crossfit_pc", "ours_crossfit",
                          "logreg_crossfit", "understat"), MODEL_COLS)
cal_lab <- season_calibration %>%
  filter(model %in% cal_models) %>%
  mutate(lab = sprintf("%s\ncoverage %s | xG per goal %.2f | RMSE %.1f goals",
                       MODEL_LABELS[as.character(model)], pct(coverage_95),
                       xg_per_goal, rmse))
cal_lab <- setNames(cal_lab$lab, as.character(cal_lab$model))

pp <- player_seasons %>%
  filter(model %in% cal_models) %>%
  mutate(model = factor(model, levels = cal_models))
lim <- c(0, max(c(pp$xg, pp$goals)) * 1.03)
focus_pp <- pp %>%
  filter(player_id %in% FOCUS_PLAYERS, season_start == FOCUS_SEASON) %>%
  mutate(group = names(FOCUS_PLAYERS)[match(player_id, FOCUS_PLAYERS)],
         # label on whichever side of the point has room
         hj = ifelse(xg > 0.55 * lim[2], 1.12, -0.12))

fig_cal <- ggplot(pp, aes(xg, goals)) +
  geom_abline(slope = 1, intercept = 0, colour = INK_2, linewidth = 0.4) +
  geom_point(colour = MUTED, alpha = 0.25, size = 1) +
  geom_point(data = focus_pp, aes(colour = group), size = 3.2) +
  geom_text(data = focus_pp, aes(colour = group, label = group, hjust = hj),
            vjust = 0.5, size = 3.2, show.legend = FALSE) +
  facet_wrap(~ model, ncol = 3, labeller = as_labeller(cal_lab)) +
  scale_colour_manual(values = PLAYER_COLS) +
  coord_equal(xlim = lim, ylim = lim) +
  labs(title = "Expected vs actual goals for every player-season, 2014-15 to 2020-21",
       subtitle = sprintf(paste0(
         "%d player-seasons with %d+ shots. On the diagonal, the model's expected goals equal the goals scored.\n",
         "Coverage: share of player-seasons whose goals land inside the model's own 95%% interval."),
         n_distinct(paste(pp$player_id, pp$season_start)), MIN_SHOTS),
       x = "Expected goals (sum of the model's xG)", y = "Goals scored") +
  theme_07()
save_fig(fig_cal, "07_player_season_calibration.png", width = 15, height = 10.5)

# 14f. Feasibility: where the profile points sit in (distance, angle) space.
ref_ids <- group_map$row_id[group_map$group == ref_player]
d_seq <- seq(0.3, 85, by = 0.1)
region <- data.frame(x = c(d_seq, rev(d_seq)),
                     y = c(max_angle_at(d_seq), rev(min_angle_at(d_seq))))
PANEL_F1 <- "Paper: distance moves, angle stays fixed"
PANEL_F2 <- "Improved: the shot moves along its own line to goal"
feas_pts <- design %>%
  filter(row_id %in% ref_ids, profile == "partial_distance") %>%
  transmute(x = distanceToGoal, y = angleToGoal,
            what = ifelse(feasible, "A spot on the pitch", "Impossible: no spot on the pitch has this distance and angle"),
            panel = PANEL_F1)
gc_paths <- design %>%
  filter(row_id %in% ref_ids, profile == "gc_distance", feasible) %>%
  transmute(row_id, x = distanceToGoal, y = angleToGoal, panel = PANEL_F2)
obs_ref <- focus_shots %>%
  filter(row_id %in% ref_ids) %>%
  transmute(x = distanceToGoal, y = angleToGoal)
obs_ref <- bind_rows(obs_ref %>% mutate(panel = PANEL_F1),
                     obs_ref %>% mutate(panel = PANEL_F2))
share_imp <- mean(feas_pts$what != "A spot on the pitch")

fig_feas <- ggplot() +
  geom_polygon(data = bind_rows(region %>% mutate(panel = PANEL_F1),
                                region %>% mutate(panel = PANEL_F2)),
               aes(x, y), fill = "#f0efec", colour = NA) +
  geom_point(data = feas_pts, aes(x, y, colour = what), size = 0.35, alpha = 0.6) +
  geom_path(data = gc_paths, aes(x, y, group = row_id), colour = OURS_COL,
            alpha = 0.35, linewidth = 0.35) +
  geom_point(data = obs_ref, aes(x, y), colour = INK, size = 1.1) +
  facet_wrap(~ factor(panel, levels = c(PANEL_F1, PANEL_F2))) +
  scale_colour_manual(values = c("A spot on the pitch" = MUTED,
                                 "Impossible: no spot on the pitch has this distance and angle" = PAPER_COL)) +
  guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  coord_cartesian(xlim = c(0, 85), ylim = c(0, 92), expand = FALSE) +
  labs(title = sprintf("%s of the paper's distance-profile points for %s are shots that cannot exist",
                       pct(share_imp), ref_player),
       subtitle = paste0(
         "Black: his actual shots. Shaded: every (distance, angle) pair a spot on the pitch can produce.\n",
         "The paper's profile asks the model about every point on the left; the improved profile only about the paths on the right."),
       x = "Distance to goal (m)", y = "Angle to goal (degrees)") +
  theme_07()
save_fig(fig_feas, "07_profile_feasibility.png", width = 12, height = 6.2)

# 14g. Memorisation: the same shots, seen vs unseen.
if (length(mem_pairs) >= 2) {
  focus_player_ids <- group_map$row_id[group_map$kind == "player"]
  mem_pts <- bind_rows(lapply(names(mem_pairs), function(nm) {
    pr <- mem_pairs[[nm]]
    P %>% filter(row_id %in% focus_player_ids) %>%
      transmute(cross = .data[[pr[2]]], insample = .data[[pr[1]]],
                outcome = ifelse(status == 1, "Goal", "No goal"), model = nm)
  })) %>%
    mutate(model = factor(model, levels = names(mem_pairs)))
  mem_lab <- memorisation %>%
    mutate(lab = sprintf("%s\nmemorisation index %.2f | AUC seen %.2f, unseen %.2f",
                         c(paper_forest = "Paper's over-sampled random forest",
                           ours_xgboost = "Improved xgboost",
                           ours_logreg  = "Improved logistic regression")[model],
                         memorisation_index, auc_insample, auc_crossfit))
  mem_lab <- setNames(mem_lab$lab, mem_lab$model)
  idx_paper <- memorisation$memorisation_index[memorisation$model == "paper_forest"]
  idx_ours  <- memorisation$memorisation_index[memorisation$model != "paper_forest"]
  mem_title <- if (idx_paper > 3 * max(idx_ours, 0.01)) {
    "The paper's model remembers which of these shots went in"
  } else {
    "Seen vs unseen: how far each model's predictions move once it has trained on the shot"
  }

  fig_mem <- ggplot(mem_pts, aes(cross, insample)) +
    geom_abline(slope = 1, intercept = 0, colour = INK_2, linewidth = 0.4) +
    geom_point(aes(shape = outcome, colour = outcome), size = 1.8, alpha = 0.8) +
    facet_wrap(~ model, labeller = as_labeller(mem_lab)) +
    scale_shape_manual(values = c(Goal = 16, `No goal` = 1)) +
    scale_colour_manual(values = c(Goal = INK, `No goal` = MUTED)) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
    labs(title = mem_title,
         subtitle = paste0(
           "Each point is one of the focus players' ", season_label, " shots. x: prediction from a model that never saw the match.\n",
           "y: prediction from a model trained on it, which is how the paper's figures were made. Off the diagonal = memorised."),
         x = "Cross-fitted prediction (match unseen)",
         y = "In-sample prediction (match seen in training)") +
    theme_07()
  save_fig(fig_mem, "07_memorisation.png", width = 15, height = 6.4)
}

# 14h. The match: Schalke 04 v Bayern Munich (the paper's Figure 3).
fm <- profile_panels(TEAM_GROUPS, "partial_distance", "gc_distance")
p_paper <- diff_tests %>% filter(group_1 == TEAM_GROUPS[1], model == "paper_insample")
p_ours  <- diff_tests %>% filter(group_1 == TEAM_GROUPS[1], model == "ours_crossfit")
p_lr    <- diff_tests %>% filter(group_1 == TEAM_GROUPS[1], model == "logreg_crossfit")
n_team  <- table(group_map$group[group_map$kind == "match"])
team_lab <- function(g) sprintf("%s (%d shots)", g, as.integer(n_team[g]))
fig_match <- ggplot(fm, aes(x, ap, colour = group, fill = group)) +
  geom_ribbon(data = fm %>% filter(!is.na(lo)), aes(ymin = lo, ymax = hi),
              alpha = 0.18, colour = NA) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ panel, scales = "free_x") +
  scale_colour_manual(values = TEAM_COLS, labels = team_lab) +
  scale_fill_manual(values = TEAM_COLS, labels = team_lab) +
  scale_y_continuous(labels = function(v) sprintf("%.2f", v),
                     limits = c(0, y_top(fm)), expand = expansion(mult = c(0, 0.02))) +
  labs(title = sprintf("%s v %s, match %s: distance profiles", HOME_TEAM, AWAY_TEAM, FOCUS_MATCH),
       subtitle = sprintf(paste0(
         "Is the gap between the two teams' curves more than chance? Permutation test p = %.2f for the paper's curves, ",
         "p = %.2f for xgboost, p = %.2f for logistic regression.\nWith this few shots, the band shows how far either curve could move."),
         p_paper$p_value, p_ours$p_value, p_lr$p_value),
       x = "Distance to goal (m)", y = "Average predicted chance of a goal") +
  theme_07()
save_fig(fig_match, "07_match_profiles.png", width = 16, height = 6.2)


# ---- 15. Summary --------------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("How well each model assesses players\n")
cat("------------------------------------------------------------\n")
print(as.data.frame(season_calibration %>%
                      select(model, xg_per_goal, mean_bias, rmse, coverage_95)),
      digits = 3, row.names = FALSE)
cat("\nFocus players (all shots), goals vs expected goals:\n")
print(as.data.frame(assessment %>%
                      filter(shots_used == "all shots",
                             model %in% c("paper_insample", "ours_crossfit",
                                          "logreg_crossfit", "understat")) %>%
                      transmute(player, model, goals, xg = round(xg, 1),
                                range_95 = sprintf("%g-%g", lo, hi), verdict)),
      row.names = FALSE, right = FALSE)

if (FAST_MODE) {
  cat("\nFAST_MODE was on: models trained on a subsample. Set FAST_MODE <- FALSE",
      "for the real run.\n")
}
if (!USE_VALIDATION) {
  cat("\nvalidation_UNSEEN.csv untouched, so the focus players' totals cover",
      "train + test shots only.\nSet USE_VALIDATION <- TRUE once the models",
      "are final for complete seasons.\n")
}
stamp("Done.")
cat("results/07_pool_metrics.csv                 every pool shot, every model\n")
cat("results/07_player_season_calibration.csv    the headline: player-season xG vs goals\n")
cat("results/07_player_season_xg.csv             per player-season, per model\n")
cat("results/07_player_assessment.csv            focus players, with 95% ranges\n")
cat("results/07_expected_vs_actual_by_band.csv   focus players by distance band\n")
cat("results/07_profiles.csv                     every aggregated profile, with bands\n")
cat("results/07_profile_difference_tests.csv     permutation tests\n")
cat("results/07_profile_feasibility.csv          impossible profile points\n")
cat("results/07_memorisation.csv                 in-sample vs cross-fitted\n")
cat("results/figures/07_*.png                    eight figures\n")
