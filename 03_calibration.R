# ==============================================================================
# 03_calibration.R  -  ENHANCEMENT (c)
#   Calibration-focused model-family comparison.
#
# Logistic regression vs random forest vs XGBoost on the paper's seven
# features, compared on discrimination AND calibration, with significance
# testing and repeated CV so that "best model" is a claim we can defend.
#
# The parent paper compares four families via `forester` across three sampling
# schemes and reports log-loss, Brier and MCC. We hand-roll three families
# instead: `forester` is a 2022 API that has since been rewritten, and a
# faithful rerun needs version archaeology. Document that as a reproducibility
# finding. NOTE: this script covers the model-family axis only, not the
# ROSE over/under-sampling axis, so it is not a reproduction of the paper's
# Table III.
#
# Reads:  data/splits/train.csv, data/splits/test.csv
# Writes: results/03_*.csv
# Runtime: ~15-25 min, almost all of it section 6 (45 model fits).
#
# Requires 01_build_data.R to have run.
# ==============================================================================

source("02_model_utils.R")
library(DALEX)

dir.create("results", recursive = TRUE, showWarnings = FALSE)

cat("\n==============================================================\n")
cat("03_calibration.R  -  enhancement (c)\n")
cat("==============================================================\n")


# ---- 1. Data ------------------------------------------------------------------

train_m <- read_split("train")
test_m  <- read_split("test")

cat("\nModel rows: train", nrow(train_m), " test", nrow(test_m), "\n")

# validation_UNSEEN.csv is deliberately not read here. Tuning happens against
# test only.

train_obs <- observed_levels(train_m)
test_m    <- align_to_training(test_m, train_obs, verbose = TRUE)


# ---- 2. Feature correlation ---------------------------------------------------

# Distance and angle both describe shot location, so they overlap heavily. If
# they do, SHAP and permutation importance can split credit between them, and
# the two importances should be read together rather than as separate effects.

cat("\n--- Spatial feature correlation (train, Spearman) ---\n")
spatial_cor <- cor(train_m[, c("distanceToGoal", "angleToGoal")],
                   method = "spearman")
print(round(spatial_cor, 3))

write_csv(
  tibble(pair = "distanceToGoal ~ angleToGoal",
         spearman = round(spatial_cor[1, 2], 4)),
  "results/03_feature_correlation.csv"
)


# ---- 3. Fit and evaluate ------------------------------------------------------

cat("\n--- Fitting on train.csv ---\n")
fits <- fit_all(train_m)

preds <- lapply(fits, function(f) f$predict(test_m))


#bug fixing:
#b <- coef(fits$logistic_regression$model)
#b[is.na(b)]
#table(train_m$lastAction == "Standard", train_m$situation == "Penalty")
#with(train_m, table(situation, lastAction == "Standard"))
#sum(!is.finite(preds$logistic_regression))
#warning is not a real bug, just noise
########

results <- bind_rows(lapply(names(preds), function(nm) {
  metrics(test_m$status, preds[[nm]]) %>% mutate(model = nm)
}))

# Understat's own xG on the same test shots, as an outside benchmark.
# Caveat for the writeup: Understat fitted its model on this data, so it is
# scoring in-sample while ours are not. It is a reference point, not a fair
# competitor.
if (!all(is.na(test_m$xg_understat))) {
  ok <- !is.na(test_m$xg_understat)
  results <- bind_rows(
    results,
    metrics(test_m$status[ok], test_m$xg_understat[ok]) %>%
      mutate(model = "understat_xG (benchmark, in-sample)")
  )
}

results <- results %>% select(model, everything())

cat("\n--- Test set results ---\n")
print(as.data.frame(results))
write_csv(results, "results/03_test_metrics.csv")


# ---- 4. DeLong tests ----------------------------------------------------------

# The blind-football paper we reviewed named a "best" model even though DeLong
# found no significant AUC difference between candidates. Test every pair
# before claiming one family beats another.

rocs  <- lapply(preds, function(p) roc(test_m$status, p, quiet = TRUE))
pairs <- combn(names(rocs), 2, simplify = FALSE)

delong <- bind_rows(lapply(pairs, function(pr) {
  tst <- roc.test(rocs[[pr[1]]], rocs[[pr[2]]], method = "delong")
  tibble(model_a = pr[1], model_b = pr[2],
         auc_a   = as.numeric(auc(rocs[[pr[1]]])),
         auc_b   = as.numeric(auc(rocs[[pr[2]]])),
         p_value = tst$p.value)
})) %>%
  mutate(p_holm      = p.adjust(p_value, method = "holm"),
         significant = p_holm < 0.05)

cat("\n--- DeLong pairwise AUC tests (Holm adjusted) ---\n")
print(as.data.frame(delong))
write_csv(delong, "results/03_delong_tests.csv")

#XGBoost's advantage is statistically significant but practically negligible.
#A logistic regression recovers 99.5% of the achievable discrimination on this feature set.


# ---- 5. Calibration by decile -------------------------------------------------

# Bin predicted xG into deciles and compare to the observed goal rate. A well
# calibrated model has mean_pred close to actual_rate in every bin. This is the
# table that shows what AUC cannot.

calib <- bind_rows(lapply(names(preds), function(nm) {
  calibration_table(test_m$status, preds[[nm]]) %>% mutate(model = nm)
})) %>%
  select(model, everything())

cat("\n--- Calibration by decile ---\n")
print(as.data.frame(calib))
write_csv(calib, "results/03_calibration_deciles.csv")

# Reliability diagram: each point is one decile, mean predicted xG on x and
# observed goal rate on y. Points on the dashed diagonal are perfectly
# calibrated; above it the model under-predicts, below it over-predicts.
# Point size shows the number of shots in the bin (deciles are near-equal).
calib_max <- max(calib$mean_pred, calib$actual_rate)

calib_plot <- ggplot2::ggplot(
    calib, ggplot2::aes(x = mean_pred, y = actual_rate, colour = model)) +
  ggplot2::geom_abline(slope = 1, intercept = 0,
                       linetype = "dashed", colour = "grey50") +
  ggplot2::geom_line(linewidth = 0.8) +
  ggplot2::geom_point(ggplot2::aes(size = n), alpha = 0.8) +
  ggplot2::coord_equal(xlim = c(0, calib_max), ylim = c(0, calib_max)) +
  ggplot2::scale_size_continuous(guide = "none", range = c(1.5, 3.5)) +
  ggplot2::labs(title    = "Calibration by decile (test set)",
                subtitle = "Dashed line = perfect calibration",
                x = "Mean predicted xG", y = "Observed goal rate",
                colour = "Model") +
  ggplot2::theme_minimal(base_size = 13) +
  ggplot2::theme(legend.position = "bottom")

print(calib_plot)
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
ggplot2::ggsave("results/figures/03_calibration_deciles.png", calib_plot,
                width = 7, height = 7.5, dpi = 150, bg = "white")

# What the calibration plot tells us (test set, 6,320 shots per decile).
# gap = mean_pred - actual_rate, so positive means the model over-predicts.
#
# All three families are well calibrated overall. Every decile sits within
# 2.5 percentage points of the diagonal, so a 0.20 xG shot really does go in
# about 20% of the time. Mean absolute gap across deciles: xgboost 0.0038,
# random forest 0.0043, logistic regression 0.0076.
#

# log reg overpredicts early deciles and underpredicts later deciles
#
# RF undercalibrates lowest deciles
#
# All three models over-predict the top decile by 1.5-2 points (about 0.455
# predicted vs 0.436 observed). Every family rates the best chances slightly
# too highly, so this looks like a limit of the seven features rather than of
# any one model.
#
# Combined with the DeLong results above: xgboost's AUC edge over logistic
# regression is negligible, but its calibration edge is larger and systematic.
# If the goal is trustworthy xG values rather than just ranking shots,
# calibration is the stronger argument for xgboost.




# ---- 6. Repeated stratified cross validation ----------------------------------

# One split can flatter or punish a model by chance, so repeat 5-fold CV three
# times and report the spread. If the sd across folds is larger than the gap
# between two families, the ranking between them is noise.
#
# Run on a stratified subsample to keep the runtime sane. Set CV_N to
# nrow(train_m) to use all of it.

CV_N       <- 50000
CV_FOLDS   <- 5
CV_REPEATS <- 3

set.seed(SEED)

cv_data <- train_m %>%
  group_by(status) %>%
  slice_sample(prop = min(1, CV_N / nrow(train_m))) %>%
  ungroup() %>%
  as.data.frame()

cat("\n--- Repeated CV:", CV_FOLDS, "folds x", CV_REPEATS, "repeats on",
    nrow(cv_data), "rows ---\n")

# Draw every repeat's fold labels up front. fit_xgb() calls set.seed(SEED), so
# drawing them inside the loop would give repeats 2 and 3 identical folds.
fold_sets <- lapply(seq_len(CV_REPEATS), function(r) {
  fold <- integer(nrow(cv_data))
  for (s in c(0, 1)) {                       # stratified fold labels
    idx <- which(cv_data$status == s)
    fold[idx] <- sample(rep(seq_len(CV_FOLDS), length.out = length(idx)))
  }
  fold
})

cv_rows <- list()
for (r in seq_len(CV_REPEATS)) {

  fold <- fold_sets[[r]]

  for (k in seq_len(CV_FOLDS)) {
    tr <- cv_data[fold != k, ]
    te <- cv_data[fold == k, ]
    te <- align_to_training(te, observed_levels(tr), verbose = FALSE)

    for (nm in names(FITTERS)) {
      f <- FITTERS[[nm]](tr)
      cv_rows[[length(cv_rows) + 1]] <-
        metrics(te$status, f$predict(te)) %>%
        mutate(model = nm, repeat_id = r, fold = k)
    }
  }
  cat("finished CV repeat", r, "of", CV_REPEATS, "\n")
}

cv_results <- bind_rows(cv_rows)
cv_summary <- cv_results %>%
  group_by(model) %>%
  summarise(across(c(auc, brier, log_loss, ece),
                   list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
            .groups = "drop")

cat("\n--- Repeated CV summary (mean and sd over",
    CV_FOLDS * CV_REPEATS, "folds) ---\n")
print(as.data.frame(cv_summary))
write_csv(cv_results, "results/03_cv_all_folds.csv")
write_csv(cv_summary, "results/03_cv_summary.csv")


# ---- 7. Explanations with DALEX -----------------------------------------------

# The paper explains its models with DALEX, so we use DALEX rather than a SHAP
# substitute: permutation importance, partial dependence for the two spatial
# features, and SHAP for two individual shots. Staying in the paper's own
# explanation library means the report needs no "we substituted X for DALEX"
# caveat.

x_test <- test_m[, FEATURES]

explainer <- DALEX::explain(
  model = fits$xgboost$model,
  data  = x_test,
  y     = test_m$status,
  predict_function = function(m, nd) predict(m, xgb.DMatrix(to_matrix(prep(nd)))),
  label = "xgboost",
  verbose = FALSE
)

imp <- model_parts(explainer, N = 5000, B = 5)
cat("\n--- Permutation importance (xgboost) ---\n")
print(imp)
write_csv(as.data.frame(imp), "results/03_importance_xgboost.csv")

pdp <- model_profile(explainer,
                     variables = c("distanceToGoal", "angleToGoal"),
                     N = 1000)
write_csv(as.data.frame(pdp$agr_profiles), "results/03_pdp_xgboost.csv")

# SHAP for two individual shots: the first goal and the first miss in test.
shap_ids <- c(which(test_m$status == 1)[1], which(test_m$status == 0)[1])
shap_rows <- list()
for (i in shap_ids) {
  sh <- predict_parts(explainer, new_observation = x_test[i, ],
                      type = "shap", B = 10)
  cat("\n--- SHAP for test shot", i, "(status =", test_m$status[i], ") ---\n")
  print(sh)
  shap_rows[[length(shap_rows) + 1]] <-
    as.data.frame(sh) %>% mutate(shot_row = i, status = test_m$status[i])
}
write_csv(bind_rows(shap_rows), "results/03_shap_examples.csv")

# Plots render in RStudio's plot pane; saved to file so run_all.R captures them.
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
try({
  png("results/figures/03_importance_xgboost.png", width = 900, height = 650)
  print(plot(imp)); dev.off()
  png("results/figures/03_pdp_xgboost.png", width = 900, height = 650)
  print(plot(pdp) +
          ggplot2::labs(x = "Feature value (angle in degrees, distance in metres)"))
  dev.off()
}, silent = TRUE)


# ---- 8. Calibration gap: our models vs Understat ------------------------------

# Section 5 shows each family against the diagonal; this puts Understat's xG on
# the same footing. For every decile, gap = mean_pred - actual_rate (positive =
# over-predicts). All four are binned on the same rows (test shots that have an
# Understat value) so the deciles are comparable. Same caveat as section 3:
# Understat is scoring in-sample.

if (!all(is.na(test_m$xg_understat))) {
  ok <- !is.na(test_m$xg_understat)

  gap_preds <- c(lapply(preds, function(p) p[ok]),
                 list(understat_xG = test_m$xg_understat[ok]))

  calib_vs_us <- bind_rows(lapply(names(gap_preds), function(nm) {
    calibration_table(test_m$status[ok], gap_preds[[nm]]) %>% mutate(model = nm)
  })) %>%
    mutate(model = factor(model, levels = names(gap_preds)))

  # Per decile: how much further from the diagonal each of our models sits
  # than Understat does. Negative = ours is better calibrated in that decile.
  us_gap <- calib_vs_us %>%
    filter(model == "understat_xG") %>%
    select(bin, us_gap = gap)

  gap_diff <- calib_vs_us %>%
    filter(model != "understat_xG") %>%
    left_join(us_gap, by = "bin") %>%
    mutate(abs_gap_diff = abs(gap) - abs(us_gap)) %>%
    droplevels()

  gap_summary <- calib_vs_us %>%
    group_by(model) %>%
    summarise(mean_abs_gap = mean(abs(gap)),
              max_abs_gap  = max(abs(gap)), .groups = "drop")

  cat("\n--- Calibration gap vs Understat (same test rows) ---\n")
  print(as.data.frame(gap_summary))
  write_csv(calib_vs_us, "results/03_calibration_vs_understat.csv")
  write_csv(gap_summary, "results/03_calibration_gap_summary.csv")

  model_cols <- c(logistic_regression = "#2a78d6", random_forest = "#eb6834",
                  xgboost = "#1baf7a", understat_xG = "#4a3aa7")

  # Plot 1: signed gap by decile, all four models on one zero line.
  gap_plot <- ggplot2::ggplot(
      calib_vs_us,
      ggplot2::aes(x = bin, y = gap * 100, colour = model,
                   linetype = model == "understat_xG")) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey50") +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::geom_point(size = 2.5) +
    ggplot2::scale_x_continuous(breaks = 1:10) +
    ggplot2::scale_colour_manual(values = model_cols) +
    ggplot2::scale_linetype_manual(values = c(`FALSE` = "solid",
                                              `TRUE` = "dashed"),
                                   guide = "none") +
    ggplot2::labs(title    = "Calibration gap by decile: our models vs Understat",
                  subtitle = "Above 0 = over-predicts, below 0 = under-predicts",
                  x = "Predicted-xG decile", y = "Gap (percentage points)",
                  colour = "Model") +
    ggplot2::theme_minimal(base_size = 13) +
    ggplot2::theme(legend.position = "bottom",
                   panel.grid.minor = ggplot2::element_blank())

  # Plot 2: |gap| difference vs Understat, one bar per model per decile.
  diff_plot <- ggplot2::ggplot(
      gap_diff, ggplot2::aes(x = factor(bin), y = abs_gap_diff * 100,
                             fill = model)) +
    ggplot2::geom_hline(yintercept = 0, colour = "grey50") +
    ggplot2::geom_col(position = ggplot2::position_dodge(width = 0.8),
                      width = 0.75, colour = "white", linewidth = 0.3) +
    ggplot2::scale_fill_manual(values = model_cols) +
    ggplot2::labs(title    = "Calibration error relative to Understat",
                  subtitle = "|gap| ours - |gap| Understat; below 0 = ours is closer to the diagonal",
                  x = "Predicted-xG decile",
                  y = "Difference in |gap| (percentage points)",
                  fill = "Model") +
    ggplot2::theme_minimal(base_size = 13) +
    ggplot2::theme(legend.position = "bottom",
                   panel.grid.major.x = ggplot2::element_blank(),
                   panel.grid.minor   = ggplot2::element_blank())

  print(gap_plot)
  print(diff_plot)
  ggplot2::ggsave("results/figures/03_calibration_gap_vs_understat.png",
                  gap_plot, width = 8, height = 5.5, dpi = 150, bg = "white")
  ggplot2::ggsave("results/figures/03_calibration_gap_diff_vs_understat.png",
                  diff_plot, width = 8, height = 5.5, dpi = 150, bg = "white")
}


cat("\n------------------------------------------------------------\n")
cat("Enhancement (c) complete. results/03_*.csv\n")
cat("Lead the writeup with auc / brier / log_loss / ece.\n")
cat("validation_UNSEEN.csv and the out-of-time set remain untouched.\n")
cat("------------------------------------------------------------\n")
