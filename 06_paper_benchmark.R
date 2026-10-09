# ==============================================================================
# 06_paper_benchmark.R  -  the paper's models vs ours, on the same held-out shots
#
# 05 reproduces the paper's own protocol. This file is the fair comparison:
#
#   - the paper's models (their code, patched to run; see paper_setup.R) and
#     ours (02_model_utils.R) train on the same rows: train.csv
#   - both are scored on the same rows: test.csv, natural 10.7% goal rate,
#     with the same metrics() from 02
#   - for the paper's over/under-sampled variants, resampling touches the
#     training rows only, never test
#   - uncertainty: match-level bootstrap of every model's difference from a
#     reference model (default: the paper's random forest on original data)
#   - optionally, all models scored on the 2021+ out-of-time set too
#
# Over/under-sampled models predict on a ~50% goal prior, so their raw
# probabilities are miscalibrated by construction. Each also gets a
# "_priorcorrected" row with the prior shift undone (paper_setup.R,
# prior_correct). That is their best fair showing on log loss / Brier / ECE.
# AUC is identical either way.
#
# Needs 01_build_data.R to have run. Reads train.csv, test.csv and (if
# SCORE_OOT) the out-of-time file. Never opens validation_UNSEEN.csv.
# Writes results/06_*.csv
#
# Runtime: 12 paper models + our three on 221k training rows, then a bootstrap.
# Expect roughly an hour and several GB of RAM. FAST_MODE <- TRUE first.
# ==============================================================================

FAST_MODE <- FALSE
FAST_N    <- 40000      # stratified training subsample in FAST_MODE (test stays whole)
SAMPLINGS <- c("original", "under", "over")
SCORE_OOT <- TRUE       # also score 2021+; this is enhancement (a)'s set
DROP_SEASONS_FROM <- 2025   # partial season, flagged in 01_season_coverage.csv
N_BOOT    <- 500        # match-level bootstrap replicates
REFERENCE <- "paper_xgboost_original"   # every model is compared against this one
WRITE_PREDICTIONS <- TRUE              # per-shot test predictions, for later plots

source("02_model_utils.R")
source("paper_setup.R")
dir.create("results", showWarnings = FALSE)
t_start <- Sys.time()


# ---- 1. Data ------------------------------------------------------------------

train_m <- read_split("train")
test_m  <- read_split("test")

if (FAST_MODE) {
  train_m <- stratified_subsample(train_m, FAST_N, SEED)
  cat(sprintf("\nFAST_MODE: training on a stratified %d-shot subsample.\n", nrow(train_m)))
}

train_obs <- observed_levels(train_m)
test_m    <- align_to_training(test_m, train_obs, verbose = TRUE)

tr <- paper_frame(train_m)
te <- paper_frame(test_m)
assert_no_forester_prompt(tr)

if (SCORE_OOT) {
  oot_m <- read_split("oot")
  if ("season" %in% names(oot_m)) {
    oot_m <- oot_m[oot_m$season < DROP_SEASONS_FROM, , drop = FALSE]
  }
  oot_m <- align_to_training(oot_m, train_obs, verbose = TRUE)
  oo    <- paper_frame(oot_m)
}

cat(sprintf("\ntrain %d shots (%.2f%% goals) | test %d shots (%.2f%% goals)%s\n",
            nrow(tr), 100 * mean(tr$status), nrow(te), 100 * mean(te$status),
            if (SCORE_OOT) sprintf(" | out-of-time %d shots", nrow(oo)) else ""))

# Everything scored lands in these, one column per model, rows aligned to te/oo.
P_test <- list()
P_oot  <- list()
meta   <- list()

add_model <- function(name, family, algorithm, sampling, prior_corrected,
                      p_test, p_oot = NULL) {
  P_test[[name]] <<- clip01(p_test)
  if (!is.null(p_oot)) P_oot[[name]] <<- clip01(p_oot)
  meta[[name]] <<- tibble(model = name, family = family, algorithm = algorithm,
                          sampling = sampling, prior_corrected = prior_corrected)
}


# ---- 2. The paper's models ----------------------------------------------------

insample_rows <- list()

for (s in SAMPLINGS) {
  cat(sprintf("\n================ paper models, sampling: %s ================\n", s))

  d <- paper_resample(tr, s)   # train rows only
  assert_no_forester_prompt(d)
  rate_trained <- mean(d$status)
  cat(sprintf("training rows after %s-sampling: %d, goal rate %.3f\n",
              s, nrow(d), rate_trained))

  # data_test = te stops forester from splitting again: it trains on all of d
  # and its printed comparison table is on our test set.
  set.seed(PAPER_SEED)
  fm <- PAPER$forester(data = d, target = "status", type = "classification",
                       data_test = te)

  for (slot in names(PAPER_SLOTS)) {
    e <- fm[[slot]]
    algo <- PAPER_SLOTS[[slot]]
    if (is.null(e)) next
    name <- paste("paper", algo, s, sep = "_")

    p_te <- paper_predict(e, te)
    p_oo <- if (SCORE_OOT) paper_predict(e, oo) else NULL
    add_model(name, "paper", algo, s, FALSE, p_te, p_oo)

    if (s != "original") {
      rate_true <- mean(tr$status)
      add_model(paste0(name, "_priorcorrected"), "paper", algo, s, TRUE,
                prior_correct(p_te, rate_trained, rate_true),
                if (SCORE_OOT) prior_correct(p_oo, rate_trained, rate_true))
    }

    # The paper's metric code, two ways, on this exact model object.
    insample_rows[[name]] <- bind_rows(
      tibble(model = name, scored_on = "own_training_rows (how Table III was computed)") %>%
        bind_cols(paper_measures_as_published(e)),
      tibble(model = name, scored_on = "test.csv (held out)") %>%
        bind_cols(paper_measures(e, p_te, te$status))
    )
  }

  rm(fm, d, e)
  invisible(gc())
}


# ---- 3. Our models ------------------------------------------------------------

cat("\n================ our models (02_model_utils.R) ================\n")
fits <- fit_all(train_m)
for (nm in names(fits)) {
  add_model(paste0("ours_", nm), "ours", nm, "original", FALSE,
            fits[[nm]]$predict(test_m),
            if (SCORE_OOT) fits[[nm]]$predict(oot_m))
}
rm(fits)
invisible(gc())

# Understat's own xG: a reference point, not a competitor. Their model was
# fitted on these shots, so it scores in-sample here.
has_understat <- "xg_understat" %in% names(test_m) && !anyNA(test_m$xg_understat)
if (has_understat) {
  add_model("understat_xG_insample", "benchmark", "understat", "n/a", FALSE,
            test_m$xg_understat,
            if (SCORE_OOT && "xg_understat" %in% names(oot_m) && !anyNA(oot_m$xg_understat))
              oot_m$xg_understat)
}

meta <- bind_rows(meta)
if (!REFERENCE %in% names(P_test)) {
  stop("REFERENCE model '", REFERENCE, "' was not fitted. Pick one of:\n  ",
       paste(names(P_test), collapse = "\n  "))
}


# ---- 4. Held-out metrics (shared metrics() from 02) ----------------------------

test_metrics <- bind_rows(lapply(names(P_test), function(nm) {
  metrics(te$status, P_test[[nm]]) %>% mutate(model = nm)
})) %>%
  left_join(meta, by = "model") %>%
  select(model, family, algorithm, sampling, prior_corrected, everything())

write_csv(test_metrics, "results/06_benchmark_test_metrics.csv")

insample <- bind_rows(insample_rows)
write_csv(insample, "results/06_paper_insample_vs_heldout.csv")


# ---- 5. Match-level bootstrap against the reference ----------------------------

# Shots in the same match share a game state, so they are not independent.
# Resampling whole matches keeps that dependence inside each replicate.
# Every model is scored on the same resampled rows, so the differences are
# paired and much tighter than the individual intervals.

fast_auc <- function(y, p) {
  r  <- rank(p)
  n1 <- sum(y == 1)
  n0 <- length(y) - n1
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
col_logloss <- function(y, P) -colMeans(y * log(P) + (1 - y) * log(1 - P))
col_brier   <- function(y, P)  colMeans((P - y)^2)
col_auc     <- function(y, P)  apply(P, 2, fast_auc, y = y)

P  <- do.call(cbind, P_test)
y  <- te$status
ref <- which(colnames(P) == REFERENCE)

if ("match_id" %in% names(test_m)) {
  groups <- split(seq_len(nrow(test_m)), test_m$match_id)
  boot_unit <- "match"
} else {
  warning("test.csv has no match_id; bootstrapping single shots instead.")
  groups <- as.list(seq_len(nrow(test_m)))
  boot_unit <- "shot"
}

n_boot <- if (FAST_MODE) min(N_BOOT, 100) else N_BOOT
cat(sprintf("\nBootstrap: %d replicates over %d %ss ...\n",
            n_boot, length(groups), boot_unit))

point <- list(log_loss = col_logloss(y, P), brier = col_brier(y, P),
              auc = col_auc(y, P))

set.seed(SEED)
draws <- lapply(names(point), function(m) matrix(NA_real_, n_boot, ncol(P),
                                                  dimnames = list(NULL, colnames(P))))
names(draws) <- names(point)
for (b in seq_len(n_boot)) {
  idx <- unlist(groups[sample.int(length(groups), replace = TRUE)], use.names = FALSE)
  yb  <- y[idx]
  Pb  <- P[idx, , drop = FALSE]
  draws$log_loss[b, ] <- col_logloss(yb, Pb)
  draws$brier[b, ]    <- col_brier(yb, Pb)
  draws$auc[b, ]      <- col_auc(yb, Pb)
}

boot <- bind_rows(lapply(names(point), function(m) {
  better_is_lower <- m != "auc"
  diffs <- draws[[m]] - draws[[m]][, ref]
  tibble(
    model        = colnames(P),
    metric       = m,
    estimate     = point[[m]],
    ci_low       = apply(draws[[m]], 2, quantile, 0.025),
    ci_high      = apply(draws[[m]], 2, quantile, 0.975),
    diff_vs_ref  = point[[m]] - point[[m]][ref],
    diff_ci_low  = apply(diffs, 2, quantile, 0.025),
    diff_ci_high = apply(diffs, 2, quantile, 0.975),
    # Share of replicates in which this model beats the reference on this
    # metric. Near 1 or 0 means the ordering is stable; near 0.5 it is noise.
    p_beats_ref  = colMeans(if (better_is_lower) diffs < 0 else diffs > 0)
  )
})) %>%
  mutate(reference = REFERENCE, bootstrap_unit = boot_unit, n_boot = n_boot) %>%
  left_join(meta, by = "model")

write_csv(boot, "results/06_benchmark_bootstrap.csv")


# ---- 6. Out-of-time (2021+) ---------------------------------------------------

if (SCORE_OOT) {
  seasons <- if ("season" %in% names(oot_m)) sort(unique(oot_m$season)) else NULL
  oot_rows <- list()
  for (nm in names(P_oot)) {
    oot_rows[[length(oot_rows) + 1]] <- metrics(oo$status, P_oot[[nm]]) %>%
      mutate(model = nm, season = "all")
    for (sz in seasons) {
      k <- oot_m$season == sz
      oot_rows[[length(oot_rows) + 1]] <- metrics(oo$status[k], P_oot[[nm]][k]) %>%
        mutate(model = nm, season = as.character(sz))
    }
  }
  oot_metrics <- bind_rows(oot_rows) %>%
    left_join(meta, by = "model") %>%
    select(model, season, family, algorithm, sampling, prior_corrected, everything())
  write_csv(oot_metrics, "results/06_benchmark_oot_metrics.csv")
}

if (WRITE_PREDICTIONS) {
  id <- if ("row_id" %in% names(test_m)) test_m$row_id else seq_len(nrow(test_m))
  write_csv(bind_cols(tibble(row_id = id, status = te$status), as_tibble(P)),
            "results/06_benchmark_test_predictions.csv")
}


# ---- 7. Summary ---------------------------------------------------------------

cat("\n--- The paper's metric code on the same models: own training rows vs test.csv ---\n")
print(as.data.frame(
  insample %>%
    select(model, scored_on, auc, log_loss, brier = brier_score) %>%
    mutate(scored_on = ifelse(grepl("^own", scored_on), "train_rows", "test_csv")) %>%
    tidyr::pivot_wider(names_from = scored_on, values_from = c(auc, log_loss, brier))
), digits = 3)

cat(sprintf("\n--- Held out on test.csv, sorted by log loss (reference: %s) ---\n", REFERENCE))
cat("    diff = model - reference; for log loss and Brier, negative is better\n")
summary_tbl <- boot %>%
  select(model, metric, estimate, diff_vs_ref, diff_ci_low, diff_ci_high, p_beats_ref) %>%
  tidyr::pivot_wider(names_from = metric,
                     values_from = c(estimate, diff_vs_ref, diff_ci_low,
                                     diff_ci_high, p_beats_ref)) %>%
  transmute(model,
            log_loss = estimate_log_loss,
            ll_diff  = sprintf("%+.4f [%+.4f, %+.4f]", diff_vs_ref_log_loss,
                               diff_ci_low_log_loss, diff_ci_high_log_loss),
            auc      = estimate_auc,
            auc_diff = sprintf("%+.4f [%+.4f, %+.4f]", diff_vs_ref_auc,
                               diff_ci_low_auc, diff_ci_high_auc),
            brier    = estimate_brier) %>%
  arrange(log_loss)
print(as.data.frame(summary_tbl), digits = 4, right = FALSE)

if (FAST_MODE) {
  cat("\nFAST_MODE was on: all models trained on", nrow(tr),
      "shots. Set FAST_MODE <- FALSE for the real run.\n")
}

cat(sprintf("\nDone in %.1f minutes.\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))
cat("results/06_benchmark_test_metrics.csv      every model on test.csv, metrics() from 02\n")
cat("results/06_paper_insample_vs_heldout.csv   paper's own metrics, train rows vs test\n")
cat("results/06_benchmark_bootstrap.csv         estimates, CIs, paired diffs vs reference\n")
if (SCORE_OOT) cat("results/06_benchmark_oot_metrics.csv       2021+ overall and by season\n")
if (WRITE_PREDICTIONS) cat("results/06_benchmark_test_predictions.csv  per-shot predictions, all models\n")


# ---- 8. Benchmark plots -------------------------------------------------------

# Everything here reads the results/06_*.csv files written above, not objects
# in memory, so once 06 has run this section can be re-run on its own (select
# from here to the end and run) to redraw the figures without refitting.
#
#   06_table3_vs_reproduced.png  the paper's reported Table III next to the
#                                same models on their own training rows and on
#                                test.csv, with our best model for scale
#   06_bootstrap_diff.png        every model minus the reference, with
#                                match-bootstrap 95% intervals
#   06_calibration_gap.png       calibration gap by decile on test.csv, paper
#                                vs ours, Understat dashed for reference
#   06_oot_log_loss.png          log loss by out-of-time season

suppressPackageStartupMessages({
  library(dplyr); library(readr); library(tidyr); library(ggplot2)
})
if (!exists("PAPER_TABLE3")) source("paper_setup.R")
if (!exists("calibration_table")) source("02_model_utils.R")
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

# One colour per algorithm, shared by the paper's and our implementation of
# the same family, so "ranger" and "random_forest" read as the same method.
ALGO_COLS <- c(xgboost = "#2a78d6", ranger = "#eb6834", random_forest = "#eb6834",
               catboost = "#1baf7a", lightgbm = "#eda100",
               logistic_regression = "#e87ba4", understat = "#4a3aa7")

theme_bench <- function() {
  theme_minimal(base_size = 12) +
    theme(legend.position = "bottom",
          panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold"))
}

save_fig <- function(plot, file, width = 9, height = 6) {
  print(plot)
  ggsave(file.path("results/figures", file), plot,
         width = width, height = height, dpi = 150, bg = "white")
}

f_metrics  <- "results/06_benchmark_test_metrics.csv"
f_insample <- "results/06_paper_insample_vs_heldout.csv"
f_boot     <- "results/06_benchmark_bootstrap.csv"
f_preds    <- "results/06_benchmark_test_predictions.csv"
f_oot      <- "results/06_benchmark_oot_metrics.csv"

bench_meta <- read_csv(f_metrics, show_col_types = FALSE)
metric_labels <- c(auc = "AUC (higher is better)",
                   log_loss = "Log loss (lower is better)",
                   brier = "Brier (lower is better)")


# 8a. Table III as published vs reproduced vs held out --------------------------

# The core benchmark: how much of the paper's headline performance survives
# scoring on shots the models never saw. Our best model is the dashed line.
if (file.exists(f_insample)) {
  insample_plot <- read_csv(f_insample, show_col_types = FALSE) %>%
    transmute(model,
              source = ifelse(grepl("^own", scored_on),
                              "Paper models, own training rows",
                              "Paper models, test.csv (held out)"),
              auc, log_loss, brier = brier_score) %>%
    left_join(select(bench_meta, model, algorithm, sampling), by = "model") %>%
    select(-model)

  table3_long <- bind_rows(
    PAPER_TABLE3 %>% rename(algorithm = model) %>%
      mutate(source = "Paper Table III (as published)"),
    insample_plot
  ) %>%
    filter(sampling %in% PAPER_TABLE3$sampling) %>%
    pivot_longer(c(auc, log_loss, brier), names_to = "metric") %>%
    mutate(metric   = factor(metric, levels = names(metric_labels),
                             labels = metric_labels),
           sampling = factor(sampling, levels = c("original", "over"),
                             labels = c("Original data", "Over-sampled")),
           source   = factor(source, levels = c("Paper Table III (as published)",
                                                "Paper models, own training rows",
                                                "Paper models, test.csv (held out)")))

  best_ours <- bench_meta %>%
    filter(family == "ours") %>%
    slice_min(log_loss, n = 1, with_ties = FALSE)
  ours_ref <- best_ours %>%
    select(auc, log_loss, brier) %>%
    pivot_longer(everything(), names_to = "metric") %>%
    mutate(metric = factor(metric, levels = names(metric_labels),
                           labels = metric_labels))

  p_table3 <- ggplot(table3_long, aes(x = value, y = algorithm, colour = source)) +
    geom_vline(data = ours_ref, aes(xintercept = value),
               linetype = "dashed", colour = "grey40") +
    geom_point(size = 3, alpha = 0.9) +
    facet_grid(sampling ~ metric, scales = "free_x") +
    scale_colour_manual(values = c("#4a3aa7", "#eb6834", "#2a78d6")) +
    labs(title    = "The paper's Table III vs the same models on held-out shots",
         subtitle = sprintf("Dashed line = our best model (%s) on test.csv",
                            best_ours$model),
         x = NULL, y = NULL, colour = NULL) +
    theme_bench() +
    guides(colour = guide_legend(nrow = 2))

  save_fig(p_table3, "06_table3_vs_reproduced.png", width = 11, height = 6.5)
}


# 8b. Paired bootstrap differences vs the reference ------------------------------

# Raw over/under-sampled models are left out: their ~50% prior puts their log
# loss and Brier off the scale. Their prior-corrected rows stand in for them,
# and AUC is identical either way.
if (file.exists(f_boot)) {
  boot_plot <- read_csv(f_boot, show_col_types = FALSE) %>%
    filter(sampling %in% c("original", "n/a") | prior_corrected) %>%
    mutate(algorithm = ifelse(family == "benchmark", "understat", algorithm))

  ll_order <- boot_plot %>%
    filter(metric == "log_loss") %>%
    arrange(desc(estimate)) %>%
    pull(model)

  boot_plot <- boot_plot %>%
    mutate(model  = factor(model, levels = ll_order),
           metric = factor(metric, levels = names(metric_labels),
                           labels = metric_labels),
           family = factor(family, levels = c("paper", "ours", "benchmark")))

  p_boot <- ggplot(boot_plot, aes(x = diff_vs_ref, y = model,
                                  colour = algorithm, shape = family)) +
    geom_vline(xintercept = 0, colour = "grey50") +
    geom_errorbarh(aes(xmin = diff_ci_low, xmax = diff_ci_high),
                   height = 0, linewidth = 0.8) +
    geom_point(size = 2.8) +
    facet_wrap(~ metric, scales = "free_x") +
    scale_colour_manual(values = ALGO_COLS) +
    scale_shape_manual(values = c(paper = 16, ours = 17, benchmark = 15)) +
    labs(title    = sprintf("Difference from the reference model (%s)",
                            unique(boot_plot$reference)),
         subtitle = sprintf("Test set, %d %s-level bootstrap replicates, 95%% intervals. Sorted by log loss.",
                            unique(boot_plot$n_boot), unique(boot_plot$bootstrap_unit)),
         x = "Model minus reference", y = NULL,
         colour = "Algorithm", shape = "Family") +
    theme_bench()

  save_fig(p_boot, "06_bootstrap_diff.png", width = 12, height = 7)
}


# 8c. Calibration gap by decile --------------------------------------------------

# Same view as 03_calibration_gap_vs_understat.png, extended to the paper's
# models. One panel per group; Understat dashed in every panel.
if (file.exists(f_preds)) {
  preds_06 <- read_csv(f_preds, show_col_types = FALSE)

  calib_06 <- bind_rows(lapply(setdiff(names(preds_06), c("row_id", "status")),
    function(nm) calibration_table(preds_06$status, preds_06[[nm]]) %>%
      mutate(model = nm))) %>%
    left_join(select(bench_meta, model, family, algorithm, sampling,
                     prior_corrected), by = "model")

  us_calib <- calib_06 %>% filter(family == "benchmark") %>% select(bin, gap)

  calib_06 <- calib_06 %>%
    filter(family != "benchmark",
           sampling == "original" | prior_corrected) %>%
    mutate(panel = case_when(
      family == "ours"       ~ "Ours",
      sampling == "original" ~ "Paper, original data",
      TRUE ~ paste0("Paper, ", sampling, "-sampled (prior-corrected)")),
      panel = factor(panel, levels = c("Paper, original data",
                                       "Paper, under-sampled (prior-corrected)",
                                       "Paper, over-sampled (prior-corrected)",
                                       "Ours")))

  p_calib <- ggplot(calib_06, aes(x = bin, y = gap * 100, colour = algorithm)) +
    geom_hline(yintercept = 0, colour = "grey50") +
    geom_line(data = us_calib, aes(x = bin, y = gap * 100),
              inherit.aes = FALSE, colour = ALGO_COLS[["understat"]],
              linetype = "dashed", linewidth = 0.7) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    facet_wrap(~ panel, nrow = 1, drop = TRUE) +
    scale_x_continuous(breaks = 1:10) +
    scale_colour_manual(values = ALGO_COLS) +
    labs(title    = "Calibration gap by decile on test.csv: the paper's models vs ours",
         subtitle = "Gap = mean predicted - observed; above 0 over-predicts. Dashed = Understat xG (in-sample).",
         x = "Predicted-xG decile", y = "Gap (percentage points)",
         colour = "Algorithm") +
    theme_bench()

  save_fig(p_calib, "06_calibration_gap.png", width = 13, height = 5.5)
}


# 8d. Out-of-time log loss by season ---------------------------------------------

# Does either set of models age better? Prior-corrected resampled models are
# left out to keep the plot readable; they are in the CSV.
if (file.exists(f_oot)) {
  oot_plot <- read_csv(f_oot, show_col_types = FALSE,
                       col_types = cols(season = col_character())) %>%
    filter(season != "all",
           sampling %in% c("original", "n/a")) %>%
    mutate(algorithm = ifelse(family == "benchmark", "understat", algorithm),
           season    = as.integer(season),
           family    = factor(family, levels = c("paper", "ours", "benchmark")))

  p_oot <- ggplot(oot_plot, aes(x = season, y = log_loss, colour = algorithm,
                                linetype = family, group = model)) +
    geom_line(linewidth = 0.8) +
    geom_point(size = 2) +
    scale_x_continuous(breaks = sort(unique(oot_plot$season))) +
    scale_colour_manual(values = ALGO_COLS) +
    scale_linetype_manual(values = c(paper = "solid", ours = "longdash",
                                     benchmark = "dotted")) +
    labs(title    = "Out-of-time log loss by season (trained on 2014-2020)",
         subtitle = "Paper's models on original data vs ours; lower is better",
         x = "Season (start year)", y = "Log loss",
         colour = "Algorithm", linetype = "Family") +
    theme_bench() +
    theme(legend.box = "vertical")

  save_fig(p_oot, "06_oot_log_loss.png", width = 9, height = 6)
}

cat("results/figures/06_*.png                    benchmark plots (section 8)\n")
