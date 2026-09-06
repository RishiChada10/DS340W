# ==============================================================================
# Explainable xG model, replication of the input data
#
# Parent paper:
#   Cavus & Biecek (2022), "Explainable expected goal models for performance
#   analysis in football analytics", IEEE DSAA.
#   https://github.com/mcavs/Explainable_xG_model_paper
#
# The authors scraped Understat live with understat_league_season_shots().
# That function no longer works: worldfootballR was archived in Sept 2025 and
# Understat has since changed its site, so every call fails with
# "second argument must be a list".
#
# Instead we use load_understat_league_shots(), which pulls the same shot data
# from worldfootballR's static data repository on GitHub. No live scraping,
# all seasons at once, runs in minutes.
#
# Run top to bottom.
# ==============================================================================


# ---- 1. One time setup --------------------------------------------------------
# Uncomment, run once, then comment back out.

# install.packages(c("remotes", "dplyr", "readr"))
# remotes::install_github("JaseZiv/worldfootballR", upgrade = "never")


# ---- 2. Libraries and folders -------------------------------------------------

library(worldfootballR)
library(dplyr)
library(readr)

set.seed(2026)   # makes the split reproducible

dir.create("data/raw_by_league", recursive = TRUE, showWarnings = FALSE)
dir.create("data/splits",        recursive = TRUE, showWarnings = FALSE)


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
  if (!target %in% names(dataset) && source_name %in% names(dataset)) {
    names(dataset)[names(dataset) == source_name] <- target
  }
}

needed <- c("result", "X", "Y", "h_a", "situation", "shotType",
            "lastAction", "minute", "season", "match_id", "player_id")
missing <- setdiff(needed, names(dataset))
if (length(missing) > 0) {
  stop("Missing expected columns: ", paste(missing, collapse = ", "),
       "\nActual columns: ", paste(names(dataset), collapse = ", "))
}


# ---- 6. Restrict to the paper's window ----------------------------------------

# The paper covers 2014-15 through 2020-21. The loaded data runs to the present,
# so trim it. The season column may be "2014" or "2014/2015" depending on
# version, so pull the leading year out either way.

dataset <- dataset %>%
  mutate(season_start = as.numeric(substr(as.character(season), 1, 4))) %>%
  filter(season_start >= 2014, season_start <= 2020)

# Add a league column if the loaded data lacks one.
if (!"league" %in% names(dataset)) dataset$league <- NA_character_

write_csv(dataset, "data/raw_data.csv")
cat("\nRaw dataset (2014-2020):", nrow(dataset), "rows -> data/raw_data.csv\n")


# ---- 7. Preprocessing ---------------------------------------------------------

# Mirrors the paper's README. Two engineered features:
#
#   distanceToGoal  metres from the shot to the goal centre. Understat gives
#                   X and Y as proportions, so scale to a 105 x 68 pitch first.
#
#   angleToGoal     degrees subtended by the goalmouth from the shot location.
#                   7.32 is the goal width in metres.
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
    minute     = as.numeric(minute)
  ) %>%
  select(
    status, minute, h_a, situation, shotType, lastAction,
    distanceToGoal, angleToGoal, league, season, match_id, result, player_id
  )

write_csv(shot_stats, "data/shot_stats.csv")


# ---- 8. Compare to the published figures --------------------------------------

own_goals <- sum(dataset$result == "OwnGoal", na.rm = TRUE)

cat("\n--- Comparison to the paper ---\n")
cat(sprintf("%-22s %10s %10s\n", "", "ours", "paper"))
cat(sprintf("%-22s %10d %10d\n", "shots (after filter)",
            nrow(shot_stats), 315430))
cat(sprintf("%-22s %10d %10d\n", "goals", sum(shot_stats$status), 33656))
cat(sprintf("%-22s %10.2f %10.2f\n", "goal rate %",
            100 * mean(shot_stats$status), 10.66))
cat(sprintf("%-22s %10d %10d\n", "matches",
            n_distinct(shot_stats$match_id), 12655))
cat(sprintf("%-22s %10d %10d\n", "own goals dropped", own_goals, 1012))

cat("\nSmall differences are expected. Understat backfills and corrects\n")
cat("historical data, so the archive today is not identical to 2022.\n\n")


# ---- 9. Train / test / validation split ---------------------------------------

# 70 / 20 / 10, stratified on `status`.
#
# Why stratified: only about 10.66% of shots are goals. An unstratified split
# can leave the three sets with different goal rates, adding noise to every
# metric. Stratifying holds the balance roughly constant.
#
# The paper used a plain 80/20 train/test with no validation set. We use
# 70/20/10 per the assignment.

p_train <- 0.70
p_test  <- 0.20
# validation takes the remainder

shot_stats <- shot_stats %>% mutate(row_id = row_number())

assign_split <- function(ids) {
  n       <- length(ids)
  ids     <- sample(ids)          # shuffle, so labels in order are a random draw
  n_train <- floor(n * p_train)
  n_test  <- floor(n * p_test)

  data.frame(row_id = ids, split = c(
    rep("train",      n_train),
    rep("test",       n_test),
    rep("validation", n - n_train - n_test)
  ))
}

assignments <- shot_stats %>%
  group_by(status) %>%
  group_split() %>%
  lapply(function(grp) assign_split(grp$row_id)) %>%
  bind_rows()

shot_stats <- shot_stats %>% left_join(assignments, by = "row_id")

train      <- shot_stats %>% filter(split == "train")      %>% select(-split)
test       <- shot_stats %>% filter(split == "test")       %>% select(-split)
validation <- shot_stats %>% filter(split == "validation") %>% select(-split)

write_csv(train,      "data/splits/train.csv")
write_csv(test,       "data/splits/test.csv")
write_csv(validation, "data/splits/validation_UNSEEN.csv")


# ---- 10. Verify the split -----------------------------------------------------

summary_tbl <- bind_rows(
  tibble(set = "train",      n = nrow(train),      goals = sum(train$status)),
  tibble(set = "test",       n = nrow(test),       goals = sum(test$status)),
  tibble(set = "validation", n = nrow(validation), goals = sum(validation$status))
) %>%
  mutate(
    pct_of_total = round(100 * n / nrow(shot_stats), 1),
    goal_rate    = round(100 * goals / n, 2)
  )

print(summary_tbl)

overlap <- length(intersect(train$row_id, test$row_id)) +
           length(intersect(train$row_id, validation$row_id)) +
           length(intersect(test$row_id, validation$row_id))

cat("\nOverlapping rows between sets:", overlap, "(should be 0)\n")
cat("Rows accounted for:",
    nrow(train) + nrow(test) + nrow(validation), "of", nrow(shot_stats), "\n")

cat("\n------------------------------------------------------------\n")
cat("Do not load validation_UNSEEN.csv until the model is finished.\n")
cat("Train on train.csv, tune and debug against test.csv only.\n")
cat("------------------------------------------------------------\n")
