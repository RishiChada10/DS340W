# ==============================================================================
# 01_build_data.R  -  shared data stage
#
# Explainable xG project. Parent paper:
#   Cavus & Biecek (2022), "Explainable expected goal models for performance
#   analysis in football analytics", IEEE DSAA.
#   https://github.com/mcavs/Explainable_xG_model_paper
#
# The authors scraped Understat live with understat_league_season_shots().
# That function no longer works: worldfootballR was archived in Sept 2025 and
# Understat has since changed its site, so every call fails with
# "second argument must be a list". We use load_understat_league_shots()
# instead, which pulls the same shots from worldfootballR's static GitHub
# data repository. No live scraping, all seasons at once, runs in minutes.
#
# This script owns every decision about rows. It downloads, cleans, engineers
# the paper's two spatial features, tags each shot with an era, splits ONCE,
# and writes covariate-drift diagnostics. Nothing downstream re-splits.
#
# Outputs:
#   data/splits/train.csv                 2014-2020, 70%
#   data/splits/test.csv                  2014-2020, 20%
#   data/splits/validation_UNSEEN.csv     2014-2020, 10%
#   data/splits/oot_2021_plus_UNSEEN.csv  2021+, enhancement (a)
#   data/splits/factor_levels.csv         category encoding contract
#   results/01_*.csv                      drift diagnostics
#
# Runtime: minutes on first run (downloads ~140 MB), seconds after that.
# ==============================================================================


# ---- 1. One time setup --------------------------------------------------------
# Uncomment, run once, then comment back out. Leaving these live means
# reinstalling seven packages and rebuilding worldfootballR on every run.

# install.packages(c("remotes", "dplyr", "readr", "tidyr",
#                    "ranger", "xgboost", "pROC", "DALEX"))
# remotes::install_github("JaseZiv/worldfootballR", upgrade = "never")


# ---- 2. Libraries and folders -------------------------------------------------

library(worldfootballR)
library(dplyr)
library(readr)

SEED              <- 2026   # fixes the split; do not change after the first run
PAPER_LAST_SEASON <- 2020   # Understat labels the 2020-21 season as "2020"
P_TRAIN           <- 0.70
P_TEST            <- 0.20   # validation takes the remainder

set.seed(SEED)

dir.create("data/raw_by_league", recursive = TRUE, showWarnings = FALSE)
dir.create("data/splits",        recursive = TRUE, showWarnings = FALSE)
dir.create("results",            recursive = TRUE, showWarnings = FALSE)


# ---- 3. Load the shot data ----------------------------------------------------

leagues <- c("EPL", "La liga", "Serie A", "Bundesliga", "Ligue 1")

cat("Loading", length(leagues), "leagues.\n\n")

for (lg in leagues) {

  tag  <- gsub(" ", "_", lg)
  path <- file.path("data/raw_by_league", paste0(tag, ".csv"))

  if (file.exists(path)) {
    cat("skipping", tag, "(already downloaded)\n")
    next
  }

  cat("loading", tag, "... ")

  result <- tryCatch(
    load_understat_league_shots(league = lg),
    error = function(e) {
      cat("FAILED:", conditionMessage(e), "\n")
      NULL
    }
  )

  if (!is.null(result) && nrow(result) > 0) {
    write_csv(result, path)
    cat(nrow(result), "shots\n")
  }
}


# ---- 4. Combine ---------------------------------------------------------------

files <- list.files("data/raw_by_league", pattern = "\\.csv$", full.names = TRUE)

if (length(files) == 0) stop("Nothing downloaded. Check your internet connection.")

cat("\nCombining", length(files), "files.\n")

dataset <- files %>%
  lapply(read_csv, show_col_types = FALSE) %>%
  bind_rows()

cat("Columns returned:\n")
print(names(dataset))


# ---- 5. Normalise column names ------------------------------------------------

# The loading functions sometimes use friendlier names than the live scraper
# the paper used. Rename back to what the paper's code expects.

renames <- c(
  h_a      = "home_away",
  h_team   = "home_team",
  a_team   = "away_team",
  h_goals  = "home_goals",
  a_goals  = "away_goals"
)

for (target in names(renames)) {
  source_name <- renames[[target]]
  if (!source_name %in% names(dataset)) next
  if (!target %in% names(dataset)) {
    names(dataset)[names(dataset) == source_name] <- target
  } else {
    # The archive switched schema at 2022: earlier seasons fill h_a, later ones
    # fill home_away and leave h_a NA. Without merging, every 2022+ shot has a
    # missing feature and is silently dropped by keep_complete().
    dataset[[target]] <- dplyr::coalesce(as.character(dataset[[target]]),
                                         as.character(dataset[[source_name]]))
    dataset[[source_name]] <- NULL
  }
}

# The same schema switch renamed leagues ("La_liga" -> "La liga"). Standardise
# so per-league summaries don't split one league into two.
if ("league" %in% names(dataset)) {
  dataset$league <- gsub(" ", "_", dataset$league)
}

needed <- c("result", "X", "Y", "h_a", "situation", "shotType",
            "lastAction", "minute", "season", "match_id", "player_id")
missing <- setdiff(needed, names(dataset))
if (length(missing) > 0) {
  stop("Missing expected columns: ", paste(missing, collapse = ", "),
       "\nActual columns: ", paste(names(dataset), collapse = ", "))
}

# Understat ships its own xG per shot. Keep it: it is a production model scored
# on identical rows, so it tells us whether a drop between eras is our model
# ageing or the scoring environment moving. Benchmark only, never a feature.
#
# NAMING: this column is `xg_understat` everywhere in the project.

if ("xG" %in% names(dataset)) {
  dataset <- dataset %>% rename(xg_understat = xG)
} else {
  warning("No xG column in the archive. Drift benchmark will be unavailable.")
  dataset$xg_understat <- NA_real_
}


# ---- 6. Era tagging -----------------------------------------------------------

# The paper covers 2014-15 through 2020-21. The archive runs to the present,
# and those later seasons are what enhancement (a) tests on, so they are
# labelled rather than deleted. The season column may be "2014" or "2014/2015"
# depending on version, so pull the leading year out either way.

dataset <- dataset %>%
  mutate(season_start = suppressWarnings(
    as.numeric(substr(as.character(season), 1, 4))
  ))

if (any(is.na(dataset$season_start))) {
  bad <- unique(dataset$season[is.na(dataset$season_start)])
  stop("Could not parse season for: ", paste(head(bad, 5), collapse = ", "))
}

dataset <- dataset %>%
  filter(season_start >= 2014) %>%
  mutate(era = ifelse(season_start <= PAPER_LAST_SEASON,
                      "paper_2014_2020", "extension_2021_plus"))

if (!"league" %in% names(dataset)) dataset$league <- NA_character_

cat("\nShots by season and era:\n")
print(
  dataset %>%
    count(era, season_start, name = "shots") %>%
    arrange(season_start) %>%
    as.data.frame()
)

write_csv(dataset, "data/raw_data.csv")
cat("\nRaw dataset (2014-present):", nrow(dataset), "rows -> data/raw_data.csv\n")


# ---- 7. Preprocessing ---------------------------------------------------------

# Mirrors the paper's README. Two engineered features:
#
#   distanceToGoal  metres from the shot to the goal centre. Understat gives
#                   X and Y as proportions, so scale to a 105 x 68 pitch first.
#
#   angleToGoal     degrees subtended by the goalmouth from the shot location.
#                   7.32 is the goal width in metres.
#
# Leave the angle formula alone. It is the paper's single-argument atan in
# degrees, not the two-post atan2 variant in collect_understat_shots.py. If you
# carry these CSVs into Python, do not recompute the angle there or the two
# datasets will disagree on a feature that matters.
#
# Own goals are dropped: the shooter is not trying to score, so the pattern has
# nothing to do with what an xG model is estimating.

shot_stats <- dataset %>%
  filter(result != "OwnGoal") %>%
  mutate(
    status = ifelse(result == "Goal", 1, 0),
    distanceToGoal = sqrt((105 - (X * 105))^2 + (34 - (Y * 68))^2),
    angleToGoal = abs(
      atan(
        (7.32 * (105 - (X * 105))) /
          ((105 - (X * 105))^2 + (34 - (Y * 68))^2 - (7.32 / 2)^2)
      ) * 180 / pi
    )
  ) %>%
  mutate(
    h_a        = factor(h_a),
    situation  = factor(situation),
    shotType   = factor(shotType),
    lastAction = factor(lastAction),
    minute     = as.numeric(minute),
    xg_understat = as.numeric(xg_understat)
  )

keep_cols <- c("status", "minute", "h_a", "situation", "shotType", "lastAction",
               "distanceToGoal", "angleToGoal", "league", "season",
               "season_start", "era", "match_id", "result", "player_id",
               "xg_understat")
if ("date" %in% names(shot_stats)) keep_cols <- c(keep_cols, "date")

shot_stats <- shot_stats %>% select(all_of(keep_cols))

write_csv(shot_stats, "data/shot_stats.csv")


# ---- 8. Compare to the published figures --------------------------------------

# Paper era only. Comparing all seasons against the paper's totals would break
# the replication claim.

paper_era <- shot_stats %>% filter(era == "paper_2014_2020")

own_goals <- sum(dataset$result == "OwnGoal" &
                 dataset$season_start <= PAPER_LAST_SEASON, na.rm = TRUE)

cat("\n--- Comparison to the paper (2014-2020 only) ---\n")
cat(sprintf("%-22s %10s %10s\n", "", "ours", "paper"))
cat(sprintf("%-22s %10d %10d\n", "shots (after filter)",
            nrow(paper_era), 315430))
cat(sprintf("%-22s %10d %10d\n", "goals", sum(paper_era$status), 33656))
cat(sprintf("%-22s %10.2f %10.2f\n", "goal rate %",
            100 * mean(paper_era$status), 10.66))
cat(sprintf("%-22s %10d %10d\n", "matches",
            n_distinct(paper_era$match_id), 12655))
cat(sprintf("%-22s %10d %10d\n", "own goals dropped", own_goals, 1012))

cat("\nSmall differences are expected. Understat backfills and corrects\n")
cat("historical data, so the archive today is not identical to 2022.\n")

extension <- shot_stats %>% filter(era == "extension_2021_plus")
cat(sprintf("\nOut-of-time pool (2021+): %d shots, %d matches, goal rate %.2f%%\n",
            nrow(extension), n_distinct(extension$match_id),
            100 * mean(extension$status)))


# ---- 9. Category audit and the encoding contract ------------------------------

# The failure mode specific to out-of-time testing: a `lastAction` or
# `situation` value that appears in 2021-25 but never in training. Prediction
# either errors or silently miscodes it, and schema mismatch gets reported as
# drift. Find out now, before any model exists.
#
# factor_levels.csv is the contract. 02_model_utils.R reads it, so every script
# encodes categories identically no matter which split it happens to load.

cat_vars <- c("h_a", "situation", "shotType", "lastAction")

level_rows <- list()
cat("\n--- Category audit: paper era vs 2021+ ---\n")

for (v in cat_vars) {
  lv_paper <- sort(unique(as.character(paper_era[[v]])))
  lv_ext   <- sort(unique(as.character(extension[[v]])))
  lv_all   <- sort(union(lv_paper, lv_ext))

  new_lv  <- setdiff(lv_ext, lv_paper)
  gone_lv <- setdiff(lv_paper, lv_ext)

  cat(sprintf("%-12s %2d levels in paper era, %2d in 2021+", v,
              length(lv_paper), length(lv_ext)))
  if (length(new_lv))  cat("  | NEW in 2021+: ", paste(new_lv, collapse = ", "))
  if (length(gone_lv)) cat("  | absent from 2021+: ", paste(gone_lv, collapse = ", "))
  cat("\n")

  level_rows[[v]] <- data.frame(
    variable     = v,
    level        = lv_all,
    in_paper_era = lv_all %in% lv_paper,
    in_extension = lv_all %in% lv_ext,
    stringsAsFactors = FALSE
  )

  # Pin the union on both eras. An unused level is harmless; an unseen one
  # is not.
  shot_stats[[v]] <- factor(as.character(shot_stats[[v]]), levels = lv_all)
}

factor_levels <- bind_rows(level_rows)
write_csv(factor_levels, "data/splits/factor_levels.csv")

cat("Encoding contract -> data/splits/factor_levels.csv\n")

paper_era <- shot_stats %>% filter(era == "paper_2014_2020")
extension <- shot_stats %>% filter(era == "extension_2021_plus")


# ---- 10. The one and only split -----------------------------------------------

# 70 / 20 / 10 stratified on `status`, inside the paper era. Everything from
# 2021 on goes out whole as the out-of-time set - there is nothing to tune on
# it, so there is nothing to split.
#
# Why stratified: only about 10.7% of shots are goals. An unstratified split
# can leave the sets with different goal rates, adding noise to every metric.
#
# The paper used a plain 80/20 with no validation set. We use 70/20/10 per the
# assignment.
#
# This split is identical to the one the earlier single-file scripts produced
# under set.seed(2026): same row ordering, same RNG state, same assignment.

paper_era <- paper_era %>% mutate(row_id = row_number())

assign_split <- function(ids) {
  n       <- length(ids)
  ids     <- sample(ids)          # shuffle, so labels in order are a random draw
  n_train <- floor(n * P_TRAIN)
  n_test  <- floor(n * P_TEST)

  data.frame(row_id = ids, split = c(
    rep("train",      n_train),
    rep("test",       n_test),
    rep("validation", n - n_train - n_test)
  ))
}

assignments <- paper_era %>%
  group_by(status) %>%
  group_split() %>%
  lapply(function(grp) assign_split(grp$row_id)) %>%
  bind_rows()

paper_era <- paper_era %>% left_join(assignments, by = "row_id")

train      <- paper_era %>% filter(split == "train")      %>% select(-split)
test       <- paper_era %>% filter(split == "test")       %>% select(-split)
validation <- paper_era %>% filter(split == "validation") %>% select(-split)

# Out-of-time ids continue where the paper era stopped, so row_id stays unique
# across all four files.
oot <- extension %>% mutate(row_id = nrow(paper_era) + row_number())

write_csv(train,      "data/splits/train.csv")
write_csv(test,       "data/splits/test.csv")
write_csv(validation, "data/splits/validation_UNSEEN.csv")
write_csv(oot,        "data/splits/oot_2021_plus_UNSEEN.csv")


# ---- 11. Verify the split -----------------------------------------------------

summary_tbl <- bind_rows(
  tibble(set = "train",       era = "2014-2020", n = nrow(train),
         goals = sum(train$status)),
  tibble(set = "test",        era = "2014-2020", n = nrow(test),
         goals = sum(test$status)),
  tibble(set = "validation",  era = "2014-2020", n = nrow(validation),
         goals = sum(validation$status)),
  tibble(set = "out_of_time", era = "2021+",     n = nrow(oot),
         goals = sum(oot$status))
) %>%
  mutate(
    # Percent of the paper era, which is what the 70/20/10 refers to. The
    # out-of-time set is not a slice of it, so the column is blank there.
    pct_of_paper_era = ifelse(era == "2014-2020",
                              round(100 * n / nrow(paper_era), 1), NA_real_),
    goal_rate        = round(100 * goals / n, 2)
  )

print(as.data.frame(summary_tbl))
write_csv(summary_tbl, "results/01_split_summary.csv")

sets <- list(train = train$row_id, test = test$row_id,
             validation = validation$row_id, oot = oot$row_id)

overlap <- 0
for (i in seq_along(sets)) {
  for (j in seq_along(sets)) {
    if (i < j) overlap <- overlap + length(intersect(sets[[i]], sets[[j]]))
  }
}

cat("\nOverlapping rows between sets:", overlap, "(should be 0)\n")
cat("Paper era accounted for:",
    nrow(train) + nrow(test) + nrow(validation), "of", nrow(paper_era), "\n")
cat("Out-of-time rows:", nrow(oot), "\n")

if (overlap != 0) stop("Splits overlap. Do not proceed.")


# ---- 12. Drift diagnostics (model-free) ---------------------------------------

# Half of enhancement (a) is measurable before a model exists. If the feature
# distributions move between eras that is covariate shift; if the goal rate
# moves that is label shift. Separating them tells you what any later
# performance drop actually means.
#
# The out-of-time numbers below are a result, not a tuning signal. Reading them
# is fine. Choosing a model because of them is not.

cat("\n--- Per-season drift table ---\n")

by_season <- shot_stats %>%
  group_by(era, season_start) %>%
  summarise(
    shots        = n(),
    matches      = n_distinct(match_id),
    goals        = sum(status),
    goal_rate    = round(100 * mean(status), 2),
    mean_dist    = round(mean(distanceToGoal, na.rm = TRUE), 2),
    mean_angle   = round(mean(angleToGoal, na.rm = TRUE), 2),
    pct_penalty  = round(100 * mean(situation == "Penalty", na.rm = TRUE), 2),
    pct_header   = round(100 * mean(shotType == "Head", na.rm = TRUE), 2),
    mean_xg_und  = round(mean(xg_understat, na.rm = TRUE), 4),
    .groups = "drop"
  ) %>%
  mutate(calib_gap = round(mean_xg_und - goal_rate / 100, 4)) %>%
  arrange(season_start)

print(as.data.frame(by_season))
write_csv(by_season, "results/01_drift_by_season.csv")

# calib_gap is Understat's own calibration error for that season: mean predicted
# xG minus the goal rate actually observed. Flat across all seasons means the
# scoring environment is stable and any degradation you see later is your model.
# Widening after 2020 means the target moved underneath everyone, which is a
# finding rather than a failure.

cat("\n--- Season completeness (matches per league-season) ---\n")

coverage <- shot_stats %>%
  group_by(league, season_start) %>%
  summarise(matches = n_distinct(match_id), .groups = "drop") %>%
  group_by(league) %>%
  mutate(pct_of_median = round(100 * matches / median(matches), 1)) %>%
  ungroup()

thin <- coverage %>% filter(pct_of_median < 80)

if (nrow(thin) > 0) {
  cat("Seasons under 80% of their league's median match count:\n")
  print(as.data.frame(thin))
  cat("Likely in-progress or partially archived. Either exclude these from\n")
  cat("drift conclusions or say in the writeup that you kept them.\n")
} else {
  cat("All league-seasons within 80% of their league median. Nothing partial.\n")
}
write_csv(coverage, "results/01_season_coverage.csv")

# Population stability index, paper era as reference, 2021+ as current.
# Rule of thumb: < 0.1 negligible, 0.1-0.25 moderate, > 0.25 major.

psi_numeric <- function(ref, cur, bins = 10) {
  ref <- ref[is.finite(ref)]
  cur <- cur[is.finite(cur)]
  if (length(ref) == 0 || length(cur) == 0) return(NA_real_)

  cuts <- unique(quantile(ref, probs = seq(0, 1, length.out = bins + 1),
                          na.rm = TRUE, names = FALSE))
  if (length(cuts) < 3) return(NA_real_)
  cuts[1]            <- -Inf
  cuts[length(cuts)] <-  Inf

  p_ref <- as.numeric(table(cut(ref, cuts))) / length(ref)
  p_cur <- as.numeric(table(cut(cur, cuts))) / length(cur)

  eps   <- 1e-6
  p_ref <- pmax(p_ref, eps)
  p_cur <- pmax(p_cur, eps)

  sum((p_cur - p_ref) * log(p_cur / p_ref))
}

psi_categorical <- function(ref, cur) {
  lv    <- union(levels(factor(ref)), levels(factor(cur)))
  p_ref <- as.numeric(table(factor(as.character(ref), levels = lv))) / length(ref)
  p_cur <- as.numeric(table(factor(as.character(cur), levels = lv))) / length(cur)

  eps   <- 1e-6
  p_ref <- pmax(p_ref, eps)
  p_cur <- pmax(p_cur, eps)

  sum((p_cur - p_ref) * log(p_cur / p_ref))
}

num_vars <- c("distanceToGoal", "angleToGoal", "minute")

psi_tbl <- bind_rows(
  lapply(num_vars, function(v) {
    tibble(feature = v, type = "numeric",
           psi = psi_numeric(paper_era[[v]], extension[[v]]))
  }),
  lapply(cat_vars, function(v) {
    tibble(feature = v, type = "categorical",
           psi = psi_categorical(paper_era[[v]], extension[[v]]))
  })
) %>%
  mutate(
    psi   = round(psi, 4),
    shift = case_when(
      is.na(psi)  ~ "not computed",
      psi <  0.10 ~ "negligible",
      psi <  0.25 ~ "moderate",
      TRUE        ~ "major"
    )
  )

cat("\n--- Population stability index (paper era -> 2021+) ---\n")
print(as.data.frame(psi_tbl))
write_csv(psi_tbl, "results/01_drift_psi.csv")


# ---- 13. Files written --------------------------------------------------------

cat("\n------------------------------------------------------------\n")
cat("data/shot_stats.csv                      all eras, preprocessed\n")
cat("data/splits/train.csv                    2014-2020, 70%\n")
cat("data/splits/test.csv                     2014-2020, 20%\n")
cat("data/splits/validation_UNSEEN.csv        2014-2020, 10%\n")
cat("data/splits/oot_2021_plus_UNSEEN.csv     2021+, enhancement (a)\n")
cat("data/splits/factor_levels.csv            encoding contract\n")
cat("results/01_split_summary.csv\n")
cat("results/01_drift_by_season.csv\n")
cat("results/01_season_coverage.csv\n")
cat("results/01_drift_psi.csv\n")
cat("------------------------------------------------------------\n")
cat("Train on train.csv. Tune against test.csv only.\n")
cat("Both UNSEEN files stay closed until the models are final.\n")
cat("------------------------------------------------------------\n")

