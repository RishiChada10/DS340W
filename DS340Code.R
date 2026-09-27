# ==============================================================================
# Explainable xG model: data replication, model comparison, explanations
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
# Sections 1 to 10 rebuild the paper's data and split it.
# Sections 11 to 17 are our extensions: more model families, calibration
# metrics, DeLong tests, repeated CV, a feature correlation check, and
# DALEX based SHAP explanations (the paper's own explanation library).
#
# Run top to bottom.
# ==============================================================================


# ---- 1. One time setup --------------------------------------------------------
#Uncomment, run once, then comment back out.

install.packages(c("remotes", "dplyr", "readr",
                    "ranger", "xgboost", "pROC", "DALEX"))
remotes::install_github("JaseZiv/worldfootballR", upgrade = "never")


# ---- 2. Libraries and folders -------------------------------------------------

library(worldfootballR)
library(dplyr)
library(readr)
library(ranger)
library(xgboost)
library(pROC)
library(DALEX)

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

# Understat's own xG, kept only as a benchmark in section 14. Never a feature.
if (!"xG" %in% names(dataset)) dataset$xG <- NA_real_

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
    distanceToGoal, angleToGoal, league, season, match_id, result, player_id,
    understat_xG = xG
  ) %>%
  mutate(understat_xG = as.numeric(understat_xG))

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


# ==============================================================================
# EXTENSIONS
# ==============================================================================

# ---- 11. Model setup ----------------------------------------------------------

# Same seven features as the paper. Everything else (ids, league, season,
# Understat's xG) is kept out of the model.

features <- c("minute", "h_a", "situation", "shotType", "lastAction",
              "distanceToGoal", "angleToGoal")

# Lock factor levels to the full dataset so train and test encode identically.
factor_cols <- c("h_a", "situation", "shotType", "lastAction")
lvls <- lapply(factor_cols, function(v) levels(shot_stats[[v]]))
names(lvls) <- factor_cols

prep <- function(df) {
  df <- as.data.frame(df)
  for (v in factor_cols) df[[v]] <- factor(as.character(df[[v]]), levels = lvls[[v]])
  df
}

# Drop any shot missing a model feature (should be very few, if any)
keep_complete <- function(df) df[complete.cases(df[, c("status", features)]), ]

train_m <- keep_complete(prep(train))
test_m  <- keep_complete(prep(test))
cat("\nModel rows: train", nrow(train_m), " test", nrow(test_m), "\n")

# One hot matrix for xgboost
to_matrix <- function(df) {
  mf <- model.frame(~ ., data = df[, features], na.action = na.pass)
  model.matrix(~ . - 1, data = mf)
}

# Each fit function returns a function that maps new data to P(goal).
fit_logreg <- function(df) {
  m <- glm(status ~ ., data = df[, c("status", features)], family = binomial)
  list(model = m,
       predict = function(nd) as.numeric(predict(m, prep(nd), type = "response")))
}

fit_rf <- function(df) {
  d <- df[, c("status", features)]
  d$status <- factor(d$status, levels = c(0, 1))
  m <- ranger(status ~ ., data = d, num.trees = 300, probability = TRUE,
              min.node.size = 50, seed = 2026)
  list(model = m,
       predict = function(nd) predict(m, prep(nd)[, features])$predictions[, "1"])
}

fit_xgb <- function(df) {
  dm <- xgb.DMatrix(to_matrix(df), label = df$status)
  m <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "logloss",
                  eta = 0.05, max_depth = 5, subsample = 0.8,
                  colsample_bytree = 0.8, min_child_weight = 10),
    data = dm, nrounds = 400, verbose = 0
  )
  list(model = m,
       predict = function(nd) predict(m, xgb.DMatrix(to_matrix(prep(nd)))))
}

fitters <- list(
  logistic_regression = fit_logreg,
  random_forest       = fit_rf,
  xgboost             = fit_xgb
)


# ---- 12. Metrics --------------------------------------------------------------

# The paper reports discrimination only. We add calibration (Brier, log loss),
# because an xG value is used as a probability, not just a ranking.

metrics <- function(y, p) {
  p_c <- pmin(pmax(p, 1e-15), 1 - 1e-15)
  pred <- as.integer(p >= 0.5)
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  prec <- ifelse(tp + fp == 0, NA, tp / (tp + fp))
  rec  <- ifelse(tp + fn == 0, NA, tp / (tp + fn))
  tibble(
    auc       = as.numeric(auc(roc(y, p, quiet = TRUE))),
    brier     = mean((p - y)^2),
    log_loss  = -mean(y * log(p_c) + (1 - y) * log(1 - p_c)),
    accuracy  = mean(pred == y),
    precision = prec,
    recall    = rec,
    f1        = ifelse(is.na(prec) | is.na(rec) | prec + rec == 0, NA,
                       2 * prec * rec / (prec + rec))
  )
}


# ---- 13. Feature correlation check --------------------------------------------

# Distance and angle both describe shot location, so they overlap heavily.
# If they do, SHAP can split credit between them, and their importances
# should be read together rather than as two separate effects.

cat("\n--- Spatial feature correlation (train) ---\n")
print(round(cor(train_m[, c("distanceToGoal", "angleToGoal")],
                method = "spearman"), 3))


# ---- 14. Fit on train, evaluate on test ---------------------------------------

cat("\n--- Fitting models on train.csv ---\n")

fits <- list()
preds <- list()
for (nm in names(fitters)) {
  cat("fitting", nm, "... ")
  t0 <- Sys.time()
  fits[[nm]]  <- fitters[[nm]](train_m)
  preds[[nm]] <- fits[[nm]]$predict(test_m)
  cat(round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
}

results <- bind_rows(lapply(names(preds), function(nm)
  metrics(test_m$status, preds[[nm]]) %>% mutate(model = nm)))

# Understat's own xG on the same test shots, as an outside benchmark
if (!all(is.na(test_m$understat_xG))) {
  ok <- !is.na(test_m$understat_xG)
  results <- bind_rows(results,
                       metrics(test_m$status[ok], test_m$understat_xG[ok]) %>%
                         mutate(model = "understat_xG (benchmark)"))
}

results <- results %>% select(model, everything())
cat("\n--- Test set results ---\n")
print(results, width = Inf)

dir.create("results", showWarnings = FALSE)
write_csv(results, "results/test_metrics.csv")


# ---- 15. DeLong tests ---------------------------------------------------------

# The blind football paper we reviewed picked a "best" model even though
# DeLong tests found no significant AUC difference. Here we test every pair
# before claiming one model beats another.

rocs <- lapply(preds, function(p) roc(test_m$status, p, quiet = TRUE))
pairs <- combn(names(rocs), 2, simplify = FALSE)

delong <- bind_rows(lapply(pairs, function(pr) {
  t <- roc.test(rocs[[pr[1]]], rocs[[pr[2]]], method = "delong")
  tibble(model_a = pr[1], model_b = pr[2],
         auc_a = as.numeric(auc(rocs[[pr[1]]])),
         auc_b = as.numeric(auc(rocs[[pr[2]]])),
         p_value = t$p.value)
})) %>%
  mutate(p_holm = p.adjust(p_value, method = "holm"),
         significant = p_holm < 0.05)

cat("\n--- DeLong pairwise AUC tests (Holm adjusted) ---\n")
print(delong, width = Inf)
write_csv(delong, "results/delong_tests.csv")


# ---- 16. Calibration table ----------------------------------------------------

# Bin predicted xG into deciles and compare to the actual goal rate.
# A well calibrated model has mean_pred close to actual_rate in every bin.

calib <- bind_rows(lapply(names(preds), function(nm) {
  tibble(model = nm, p = preds[[nm]], y = test_m$status) %>%
    mutate(bin = ntile(p, 10)) %>%
    group_by(model, bin) %>%
    summarise(n = n(), mean_pred = mean(p), actual_rate = mean(y),
              .groups = "drop")
}))

cat("\n--- Calibration by decile ---\n")
print(calib, n = Inf)
write_csv(calib, "results/calibration_deciles.csv")


# ---- 17. Repeated stratified cross validation ---------------------------------

# One split can flatter or punish a model by chance. Repeat 5 fold CV three
# times on the training set and report the spread. We run it on a stratified
# 50,000 shot sample of train to keep the runtime reasonable; set cv_n to
# nrow(train_m) to use all of it.

cv_n       <- 50000
cv_folds   <- 5
cv_repeats <- 3

cv_data <- train_m %>%
  group_by(status) %>%
  slice_sample(prop = min(1, cv_n / nrow(train_m))) %>%
  ungroup() %>%
  as.data.frame()

cv_rows <- list()
for (r in seq_len(cv_repeats)) {
  # stratified fold labels
  fold <- integer(nrow(cv_data))
  for (s in c(0, 1)) {
    idx <- which(cv_data$status == s)
    fold[idx] <- sample(rep(seq_len(cv_folds), length.out = length(idx)))
  }
  for (k in seq_len(cv_folds)) {
    tr <- cv_data[fold != k, ]
    te <- cv_data[fold == k, ]
    for (nm in names(fitters)) {
      f <- fitters[[nm]](tr)
      cv_rows[[length(cv_rows) + 1]] <-
        metrics(te$status, f$predict(te)) %>%
        mutate(model = nm, repeat_id = r, fold = k)
    }
  }
  cat("finished CV repeat", r, "of", cv_repeats, "\n")
}

cv_results <- bind_rows(cv_rows)
cv_summary <- cv_results %>%
  group_by(model) %>%
  summarise(across(c(auc, brier, log_loss),
                   list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
            .groups = "drop")

cat("\n--- Repeated CV summary (mean and sd over", cv_folds * cv_repeats, "folds) ---\n")
print(cv_summary, width = Inf)
write_csv(cv_results, "results/cv_all_folds.csv")
write_csv(cv_summary, "results/cv_summary.csv")


# ---- 18. Explanations with DALEX ----------------------------------------------

# The paper explains its models with DALEX. We do the same for the xgboost
# model: permutation importance, partial dependence for the two spatial
# features, and SHAP values for a few individual shots.

x_test <- test_m[, features]
explainer <- DALEX::explain(
  model = fits$xgboost$model,
  data  = x_test,
  y     = test_m$status,
  predict_function = function(m, nd) fits$xgboost$predict(nd),
  label = "xgboost",
  verbose = FALSE
)

# Subsample so permutation and profiles finish quickly
imp <- model_parts(explainer, N = 5000, B = 5)
cat("\n--- Permutation importance (xgboost) ---\n")
print(imp)
write_csv(as.data.frame(imp), "results/importance_xgboost.csv")

pdp <- model_profile(explainer,
                     variables = c("distanceToGoal", "angleToGoal"),
                     N = 1000)
write_csv(as.data.frame(pdp$agr_profiles), "results/pdp_xgboost.csv")

# SHAP for a sample of shots: one goal and one miss
shap_ids <- c(which(test_m$status == 1)[1], which(test_m$status == 0)[1])
for (i in shap_ids) {
  sh <- predict_parts(explainer, new_observation = x_test[i, ],
                      type = "shap", B = 10)
  cat("\n--- SHAP for test shot", i, "(status =", test_m$status[i], ") ---\n")
  print(sh)
}

# Plots (open in RStudio's plot pane)
plot(imp)
plot(pdp)

cat("\n------------------------------------------------------------\n")
cat("All results written to the results/ folder.\n")
cat("validation_UNSEEN.csv is still untouched.\n")
cat("------------------------------------------------------------\n")
