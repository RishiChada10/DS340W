# ==============================================================================
# 05_paper_as_published.R  -  the parent paper's models, run the paper's way
#
# Reruns the modelling section of Rcodes.Rmd (Cavus & Biecek 2022) with their
# protocol unchanged, so the output can be set beside their Table III:
#
#   - every paper-era shot (2014-2020), not our train/test split
#   - ROSE over- and under-sampling applied to the FULL data, before any split
#   - forester's own stratified 80/20 split inside each forester() call
#   - set.seed(123) before every random step, as in the Rmd
#   - Table III scored with their model_performancex() on the returned models
#
# Then it scores the same fitted models a second time, on forester's own 20%
# test rows. Same model object, same metric code; only the rows differ. Table
# III appears to have been computed on TRAINING rows (see paper_setup.R,
# paper_measures_as_published). If so, the gap between the two rows of each
# pair below is that effect.
#
# Needs 01_build_data.R to have run. Reads data/shot_stats.csv (paper era only)
# and never opens either UNSEEN file. Writes results/05_*.csv.
#
# Runtime: the full run fits 12 models on up to ~450k rows. Expect over an hour
# and several GB of RAM; the over-sampled random forest is the peak. Set
# FAST_MODE <- TRUE first to check everything runs end to end (minutes).
# ==============================================================================

FAST_MODE <- FALSE
FAST_N    <- 40000                           # stratified subsample in FAST_MODE
SAMPLINGS <- c("original", "under", "over")  # Table III has all three

source("02_model_utils.R")   # only for SEED and the shared libraries
source("paper_setup.R")
dir.create("results", showWarnings = FALSE)
t_start <- Sys.time()


# ---- 1. Data: the paper era, preprocessed the paper's way ---------------------

shot_stats <- read_csv("data/shot_stats.csv", show_col_types = FALSE)
if ("era" %in% names(shot_stats)) {
  shot_stats <- shot_stats %>% filter(era == "paper_2014_2020")
} else {
  shot_stats <- shot_stats %>% filter(season >= 2014, season <= 2020)
}

# The Rmd calls factor() on the paper-era data, so levels are whatever occurs
# there. We do the same here rather than using the shared encoding contract,
# which also carries levels that first appear after 2020.
shot_stats <- shot_stats %>%
  mutate(h_a        = factor(h_a),
         situation  = factor(situation),
         shotType   = factor(shotType),
         lastAction = factor(lastAction),
         minute     = as.numeric(minute))

n_na <- sum(!complete.cases(shot_stats[, PAPER_COLS]))
cat(sprintf("\nPaper era: %d shots, %d goals (%.2f%%), %d rows with an NA feature\n",
            nrow(shot_stats), sum(shot_stats$status),
            100 * mean(shot_stats$status), n_na))
if (n_na > 0) {
  cat("  forester imputes these with k-NN (fill_na = TRUE, its default), as the paper did.\n")
}


# ---- 2. Table II: shots and goals by league -----------------------------------

if (all(c("league", "match_id") %in% names(shot_stats))) {
  table2 <- shot_stats %>%
    group_by(league) %>%
    summarise(nmatch = n_distinct(match_id),
              nshot  = n(),
              mshot  = nshot / nmatch,
              ngoal  = sum(status),
              mgoal  = ngoal / nmatch,
              conv   = 100 * ngoal / nshot,
              .groups = "drop")
  write_csv(table2, "results/05_table2.csv")
  cat("\n--- Table II (paper totals: 12,655 matches, 315,430 shots, 33,656 goals) ---\n")
  print(as.data.frame(table2), digits = 4)
} else {
  cat("\nTable II skipped: shot_stats.csv has no league / match_id column.\n")
}


# ---- 3. Fit the paper's models and score them two ways ------------------------

train_data <- paper_frame(shot_stats)
if (FAST_MODE) {
  train_data <- stratified_subsample(train_data, FAST_N, PAPER_SEED)
  cat(sprintf("\nFAST_MODE: stratified subsample of %d shots.\n", nrow(train_data)))
}
assert_no_forester_prompt(train_data)

perf_rows <- list()
diag_rows <- list()
mae_rows  <- list()

for (s in SAMPLINGS) {
  cat(sprintf("\n================ sampling: %s ================\n", s))

  # The Rmd resamples the full data, then forester splits it 80/20. That order
  # is the paper's, and it is what puts duplicated goals on both sides of the
  # split for "over". Reproduced as is.
  d <- paper_resample(train_data, s)
  assert_no_forester_prompt(d)
  cat(sprintf("rows after %s-sampling: %d, goal rate %.3f\n", s, nrow(d), mean(d$status)))

  # refclass = "" is what the Rmd passes for the under-sampled run. After DS340W
  # PATCH 5 it no longer changes anything; it is kept so the call matches.
  set.seed(PAPER_SEED)
  fm <- PAPER$forester(data   = d,
                       target = "status",
                       type   = "classification",
                       refclass = if (s == "under") "" else NULL)

  test_d <- fm$test_data
  first  <- Filter(Negate(is.null), fm[names(PAPER_SLOTS)])[[1]]
  train_d <- data.frame(status = first$y, first$data)  # forester's 80%, from the explainer

  diag_rows[[s]] <- tibble(
    sampling        = s,
    n_train         = nrow(train_d),
    n_test          = nrow(test_d),
    goal_rate_train = mean(train_d$status),
    goal_rate_test  = mean(test_d$status),
    # Exact copies of a training row among the test rows. Near zero for
    # "original"; for "over" these are the ROSE duplicates (leakage).
    test_rows_duplicating_train = duplicate_share(train_d, test_d)
  )

  for (slot in names(PAPER_SLOTS)) {
    e <- fm[[slot]]
    model <- PAPER_SLOTS[[slot]]
    if (is.null(e)) {
      cat(sprintf("  %-8s not fitted (see forester's reason above)\n", model))
      next
    }

    p_test <- paper_predict(e, test_d)

    perf_rows[[length(perf_rows) + 1]] <- bind_rows(
      tibble(sampling = s, model = model, scored_on = "train_rows_as_published",
             n = length(e$y), goal_rate = mean(e$y)) %>%
        bind_cols(paper_measures_as_published(e)),
      tibble(sampling = s, model = model, scored_on = "forester_test_20pct",
             n = nrow(test_d), goal_rate = mean(test_d$status)) %>%
        bind_cols(paper_measures(e, p_test, test_d$status))
    )

    # Table IV reports MAE for the over-sampled models only, computed on
    # over_model$test_data. The paper's values (2.0, 2.9, 3.1, 3.0) exceed 1,
    # which a median absolute error of probabilities cannot, so the unit is
    # unclear; both raw and x100 are written.
    if (s == "over") {
      mae <- median(abs(p_test - test_d$status))
      mae_rows[[model]] <- tibble(model = model, mae = mae, mae_x100 = 100 * mae)
    }
  }

  rm(fm, d, test_d, train_d, first, e)
  invisible(gc())
}


# ---- 4. Results ---------------------------------------------------------------

table3 <- bind_rows(perf_rows)
diag   <- bind_rows(diag_rows)
write_csv(table3, "results/05_table3_as_published_vs_test.csv")
write_csv(diag,   "results/05_split_diagnostics.csv")
if (length(mae_rows)) write_csv(bind_rows(mae_rows), "results/05_table4_mae.csv")

cat("\n--- forester's split, per sampling variant ---\n")
print(as.data.frame(diag), digits = 3)

# Side by side: the paper's published numbers, our replication of them (same
# code, train rows), and the same models on held-out rows.
side <- table3 %>%
  select(sampling, model, scored_on, auc, log_loss, brier = brier_score) %>%
  tidyr::pivot_wider(names_from = scored_on,
                     values_from = c(auc, log_loss, brier)) %>%
  left_join(PAPER_TABLE3 %>% rename(auc_paper = auc, log_loss_paper = log_loss,
                                    brier_paper = brier),
            by = c("sampling", "model")) %>%
  transmute(sampling, model,
            auc_paper,
            auc_ours_train_rows = auc_train_rows_as_published,
            auc_ours_heldout    = auc_forester_test_20pct,
            log_loss_paper,
            log_loss_ours_train_rows = log_loss_train_rows_as_published,
            log_loss_ours_heldout    = log_loss_forester_test_20pct,
            brier_paper,
            brier_ours_train_rows = brier_train_rows_as_published,
            brier_ours_heldout    = brier_forester_test_20pct) %>%
  arrange(factor(sampling, SAMPLINGS), factor(model, PAPER_SLOTS))
write_csv(side, "results/05_table3_side_by_side.csv")

cat("\n--- Table III: paper vs our replication (train rows) vs same models held out ---\n")
cat("    paper columns are blank for 'under': not transcribed, see PAPER_TABLE3\n")
print(as.data.frame(side), digits = 3)

if (FAST_MODE) {
  cat("\nFAST_MODE was on: models saw", FAST_N, "shots, so numbers are not",
      "comparable with the paper. Set FAST_MODE <- FALSE for the real run.\n")
}

cat(sprintf("\nDone in %.1f minutes.\n",
            as.numeric(difftime(Sys.time(), t_start, units = "mins"))))
cat("results/05_table2.csv\n")
cat("results/05_table3_as_published_vs_test.csv   all nine of their metrics, both scorings\n")
cat("results/05_table3_side_by_side.csv           AUC / log loss / Brier vs the paper\n")
cat("results/05_split_diagnostics.csv             goal rates and duplicate leakage\n")
cat("results/05_table4_mae.csv\n")
