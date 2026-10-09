# ==============================================================================
# 02_model_utils.R  -  shared model stage
#
# SOURCED, NOT RUN. `source("02_model_utils.R")` from 03 or 04.
# Defines things; fits nothing, writes nothing, prints nothing on load.
#
# Everything both enhancements need in common lives here, so that the model
# compared across eras in 04 is the same object compared across families in 03.
# Change a hyperparameter here and both analyses move together.
#
# Requires 01_build_data.R to have run at least once.
# ==============================================================================

library(dplyr)
library(readr)
library(ranger)
library(xgboost)
library(pROC)

SEED <- 2026


# ---- Paths --------------------------------------------------------------------

SPLIT_DIR <- "data/splits"
SPLITS <- list(
  train      = file.path(SPLIT_DIR, "train.csv"),
  test       = file.path(SPLIT_DIR, "test.csv"),
  validation = file.path(SPLIT_DIR, "validation_UNSEEN.csv"),
  oot        = file.path(SPLIT_DIR, "oot_2021_plus_UNSEEN.csv")
)
LEVELS_FILE <- file.path(SPLIT_DIR, "factor_levels.csv")


# ---- Features -----------------------------------------------------------------

# The paper's seven. Everything else in the CSVs (ids, league, season, era,
# xg_understat) is metadata or benchmark, never a predictor. In particular
# season and era stay out: a model that can see the year would learn the drift
# instead of being tested on it.

FEATURES    <- c("minute", "h_a", "situation", "shotType", "lastAction",
                 "distanceToGoal", "angleToGoal")
FACTOR_COLS <- c("h_a", "situation", "shotType", "lastAction")


# ---- Encoding contract --------------------------------------------------------

# Levels come from the file 01 wrote, not from whatever happens to appear in the
# split being loaded. That is what makes train, test and the out-of-time set
# encode to identical matrix columns.

load_factor_levels <- function(path = LEVELS_FILE) {
  if (!file.exists(path)) {
    stop("Missing ", path, "\nRun 01_build_data.R first.")
  }
  lv <- read_csv(path, show_col_types = FALSE)
  split(lv$level, lv$variable)
}

FACTOR_LEVELS <- load_factor_levels()

prep <- function(df) {
  df <- as.data.frame(df)
  for (v in FACTOR_COLS) {
    if (!v %in% names(df)) stop("Column missing from data: ", v)
    df[[v]] <- factor(as.character(df[[v]]), levels = FACTOR_LEVELS[[v]])
  }
  df
}

keep_complete <- function(df) {
  df[complete.cases(df[, c("status", FEATURES)]), , drop = FALSE]
}

read_split <- function(which) {
  path <- SPLITS[[which]]
  if (is.null(path)) stop("Unknown split: ", which)
  if (!file.exists(path)) {
    stop("Missing ", path, "\nRun 01_build_data.R first.")
  }
  keep_complete(prep(read_csv(path, show_col_types = FALSE)))
}


# ---- Unseen-level handling ----------------------------------------------------

# A level can be in the contract (because it exists somewhere in the archive)
# and still be absent from the training rows. glm gives it an NA coefficient,
# and any row carrying it predicts NA. Rare, but it is exactly the kind of thing
# that shows up first in the out-of-time set, so handle it explicitly and count
# it rather than discovering NAs in a metrics table.
#
# Unseen levels are collapsed to the most common level in training. Report the
# count in the writeup if it is non-zero.

observed_levels <- function(df) {
  lapply(setNames(FACTOR_COLS, FACTOR_COLS), function(v) {
    tb <- table(droplevels(factor(df[[v]])))
    list(levels = names(tb), modal = names(tb)[which.max(tb)])
  })
}

align_to_training <- function(df, obs, verbose = TRUE) {
  changed <- 0
  for (v in FACTOR_COLS) {
    bad <- !is.na(df[[v]]) & !(as.character(df[[v]]) %in% obs[[v]]$levels)
    if (any(bad)) {
      if (verbose) {
        cat(sprintf("  %s: %d rows carry a level unseen in training (%s) -> %s\n",
                    v, sum(bad),
                    paste(unique(as.character(df[[v]][bad])), collapse = ", "),
                    obs[[v]]$modal))
      }
      df[[v]][bad] <- obs[[v]]$modal
      changed <- changed + sum(bad)
    }
  }
  attr(df, "levels_collapsed") <- changed
  df
}


# ---- Design matrix ------------------------------------------------------------

# One-hot for xgboost. Column set is stable across splits because prep() pinned
# the factor levels first.

to_matrix <- function(df) {
  mf <- model.frame(~ ., data = df[, FEATURES], na.action = na.pass)
  model.matrix(~ . - 1, data = mf)
}


# ---- Model families -----------------------------------------------------------

# Each fitter returns a list with $model and $predict, where $predict maps a
# raw data frame to P(goal). Keeping that interface identical is what lets 03
# loop over families and 04 loop over eras without either knowing about the
# other.
#
# Seeds are explicit in all three so that 03 and 04, run separately from the
# same train.csv, fit the same models. glm is deterministic; ranger takes a
# seed argument; xgboost subsamples rows and columns, so it needs both R's RNG
# and its own seed parameter fixed.

fit_logreg <- function(df) {
  m <- glm(status ~ ., data = df[, c("status", FEATURES)], family = binomial)
  list(
    model   = m,
    predict = function(nd) as.numeric(predict(m, prep(nd), type = "response"))
  )
}

fit_rf <- function(df) {
  d <- df[, c("status", FEATURES)]
  d$status <- factor(d$status, levels = c(0, 1))
  m <- ranger(status ~ ., data = d, num.trees = 300, probability = TRUE,
              min.node.size = 50, seed = SEED)
  list(
    model   = m,
    predict = function(nd) predict(m, prep(nd)[, FEATURES])$predictions[, "1"]
  )
}

fit_xgb <- function(df) {
  set.seed(SEED)
  dm <- xgb.DMatrix(to_matrix(df), label = df$status)
  m <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "logloss",
                  eta = 0.05, max_depth = 5, subsample = 0.8,
                  colsample_bytree = 0.8, min_child_weight = 10, seed = SEED),
    data = dm, nrounds = 400, verbose = 0
  )
  list(
    model   = m,
    predict = function(nd) predict(m, xgb.DMatrix(to_matrix(prep(nd))))
  )
}

FITTERS <- list(
  logistic_regression = fit_logreg,
  random_forest       = fit_rf,
  xgboost             = fit_xgb
)

fit_all <- function(train_df, verbose = TRUE) {
  fits <- list()
  for (nm in names(FITTERS)) {
    if (verbose) cat("fitting", nm, "... ")
    t0 <- Sys.time()
    fits[[nm]] <- FITTERS[[nm]](train_df)
    if (verbose) {
      cat(round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1), "s\n")
    }
  }
  fits
}


# ---- Metrics ------------------------------------------------------------------

# The paper reports discrimination only. We add calibration, because an xG value
# is consumed as a probability, not just a ranking - a model can keep an
# identical AUC across eras while becoming systematically overconfident, which
# is exactly the failure mode enhancement (a) is looking for.
#
# Lead the writeup with auc, brier, log_loss and ece. The threshold metrics
# (accuracy, precision, recall, f1) are near-degenerate here: at a 10.7% base
# rate almost nothing except penalties clears 0.5, so accuracy sits near 89%
# for any model that predicts "no goal" and recall sits near zero. They are
# reported for completeness and because the paper's table has them.

expected_calibration_error <- function(y, p, bins = 10) {
  qs <- unique(quantile(p, probs = seq(0, 1, length.out = bins + 1),
                        na.rm = TRUE, names = FALSE))
  # A constant or near-constant predictor cannot be binned into deciles. That
  # is not an error - the base-rate baseline is exactly such a model - so fall
  # back to a single bin rather than returning NA.
  if (length(qs) < 3) return(abs(mean(p) - mean(y)))
  b <- cut(p, breaks = qs, include.lowest = TRUE)
  parts <- split(seq_along(p), b)
  w   <- vapply(parts, length, integer(1)) / length(p)
  gap <- vapply(parts, function(i) {
    if (length(i) == 0) 0 else abs(mean(p[i]) - mean(y[i]))
  }, numeric(1))
  sum(w * gap, na.rm = TRUE)
}

metrics <- function(y, p, threshold = 0.5, bins = 10) {
  keep <- is.finite(p) & !is.na(y)
  n_dropped <- sum(!keep)
  y <- y[keep]
  p <- p[keep]

  p_c  <- pmin(pmax(p, 1e-15), 1 - 1e-15)
  pred <- as.integer(p >= threshold)

  tp <- sum(pred == 1 & y == 1)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  prec <- if (tp + fp == 0) NA_real_ else tp / (tp + fp)
  rec  <- if (tp + fn == 0) NA_real_ else tp / (tp + fn)

  # pROC errors on a single-class slice. Possible for a thin season or a
  # narrow subgroup, and it would take the whole per-season loop down, so
  # report NA and carry on.
  auc_val <- if (length(unique(y)) < 2) NA_real_
             else as.numeric(auc(roc(y, p, quiet = TRUE)))

  tibble(
    n           = length(y),
    n_dropped   = n_dropped,
    auc         = auc_val,
    brier       = mean((p - y)^2),
    log_loss    = -mean(y * log(p_c) + (1 - y) * log(1 - p_c)),
    ece         = expected_calibration_error(y, p, bins),
    mean_pred   = mean(p),
    actual_rate = mean(y),
    calib_gap   = mean(p) - mean(y),
    threshold   = threshold,
    accuracy    = mean(pred == y),
    precision   = prec,
    recall      = rec,
    f1          = if (is.na(prec) || is.na(rec) || prec + rec == 0) NA_real_
                  else 2 * prec * rec / (prec + rec)
  )
}


# ---- Calibration table --------------------------------------------------------

calibration_table <- function(y, p, bins = 10) {
  tibble(p = p, y = y) %>%
    filter(is.finite(p)) %>%
    mutate(bin = ntile(p, bins)) %>%
    group_by(bin) %>%
    summarise(n = n(), mean_pred = mean(p), actual_rate = mean(y),
              .groups = "drop") %>%
    mutate(gap = mean_pred - actual_rate)
}


# ---- TreeSHAP helper ----------------------------------------------------------

# xgboost computes exact SHAP contributions for every row in one pass
# (predcontrib), which is fast enough to run on whole eras - DALEX's
# predict_parts is per-observation and far too slow for that. 03 uses DALEX for
# the paper-faithful explanation layer; 04 uses this for the era comparison.
#
# The matrix columns are one-hot dummies, so contributions are summed back to
# the parent feature. Column names are unambiguous: "situationOpenPlay" starts
# with "situation", "h_ah" with "h_a", and the numeric features match exactly.

shap_by_feature <- function(xgb_model, df) {
  mat  <- to_matrix(prep(df))
  contrib <- predict(xgb_model, xgb.DMatrix(mat), predcontrib = TRUE)
  colnames(contrib) <- c(colnames(mat), "BIAS")

  out <- vapply(FEATURES, function(f) {
    cols <- grep(paste0("^", f), colnames(contrib))
    if (length(cols) == 0) return(NA_real_)
    mean(abs(rowSums(contrib[, cols, drop = FALSE])))
  }, numeric(1))

  tibble(feature = FEATURES, mean_abs_shap = as.numeric(out))
}

