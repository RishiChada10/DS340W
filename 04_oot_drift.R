# ==============================================================================
# 04_oot_drift.R  -  ENHANCEMENT (a)
#   Out-of-time generalization: train 2014-2020, test 2021-2025, measure drift.
#
# Same three model families as 03, same features, same train.csv, same seeds -
# so the models fitted here are the ones 03 evaluated. One model, several
# evaluation sets, which is what makes the gap between them drift rather than
# a different experiment.
#
# The design question this answers: an xG model is fitted once and then used
# for years. How much does it decay, and is the decay the model ageing or the
# sport moving? Understat's own xG on the same shots separates those two.
#
# Reads:  data/splits/{train,test,oot_2021_plus_UNSEEN}.csv
# Writes: results/04_*.csv
# Runtime: ~5-10 min.
#
# Requires 01_build_data.R to have run.
# ==============================================================================

source("02_model_utils.R")
library(tidyr)

dir.create("results", recursive = TRUE, showWarnings = FALSE)

# Flip to TRUE only once the models are final and nothing further will be
# tuned. Until then the in-time validation set stays closed, same as the
# out-of-time set was until this script.
USE_VALIDATION <- FALSE

cat("\n==============================================================\n")
cat("04_oot_drift.R  -  enhancement (a)\n")
cat("==============================================================\n")


# ---- 1. Data ------------------------------------------------------------------

train_m <- read_split("train")
test_m  <- read_split("test")
oot_m   <- read_split("oot")

cat("\nRows: train", nrow(train_m), " in-time test", nrow(test_m),
    " out-of-time", nrow(oot_m), "\n")
cat("Out-of-time seasons:",
    paste(sort(unique(oot_m$season_start)), collapse = ", "), "\n")

train_obs <- observed_levels(train_m)

cat("\n--- Levels present in evaluation data but not in training ---\n")
test_m <- align_to_training(test_m, train_obs, verbose = TRUE)
oot_m  <- align_to_training(oot_m,  train_obs, verbose = TRUE)

n_collapsed_oot <- attr(oot_m, "levels_collapsed")
if (is.null(n_collapsed_oot) || n_collapsed_oot == 0) {
  cat("None. The 2021+ data uses only categories the model has seen.\n")
} else {
  cat(sprintf("%d out-of-time rows had a category collapsed. Report this.\n",
              n_collapsed_oot))
}

eval_sets <- list(in_time_test = test_m, out_of_time = oot_m)

if (USE_VALIDATION) {
  val_m <- align_to_training(read_split("validation"), train_obs, verbose = FALSE)
  eval_sets <- list(in_time_test = test_m,
                    in_time_validation = val_m,
                    out_of_time = oot_m)
}


# ---- 2. Fit once --------------------------------------------------------------

cat("\n--- Fitting on train.csv (2014-2020 only) ---\n")
fits <- fit_all(train_m)


# ---- 3. Metrics by evaluation set ---------------------------------------------

rows <- list()
for (set_nm in names(eval_sets)) {
  df <- eval_sets[[set_nm]]
  for (mdl in names(fits)) {
    rows[[length(rows) + 1]] <-
      metrics(df$status, fits[[mdl]]$predict(df)) %>%
      mutate(model = mdl, eval_set = set_nm)
  }
  # Understat's own xG scored on the identical rows. This is the control: if
  # our model loses ground between eras and Understat loses the same ground on
  # the same shots, the scoring environment moved rather than our model ageing.
  if (!all(is.na(df$xg_understat))) {
    ok <- !is.na(df$xg_understat)
    rows[[length(rows) + 1]] <-
      metrics(df$status[ok], df$xg_understat[ok]) %>%
      mutate(model = "understat_xG", eval_set = set_nm)
  }
}

era_metrics <- bind_rows(rows) %>% select(model, eval_set, everything())

cat("\n--- Metrics by evaluation set ---\n")
print(as.data.frame(era_metrics))
write_csv(era_metrics, "results/04_era_metrics.csv")


# ---- 4. The drift table -------------------------------------------------------

# The headline of enhancement (a). For each model: performance in time,
# performance out of time, the gap, and how that gap compares to Understat's
# gap over the same shots.
#
# excess_drift is the number to argue from. Near zero means our model decayed
# no faster than a production model maintained by a company - the decay is the
# sport, not the fit. Positive means our model aged worse than the environment
# explains.

wide <- era_metrics %>%
  filter(eval_set %in% c("in_time_test", "out_of_time")) %>%
  select(model, eval_set, auc, brier, log_loss, ece, calib_gap) %>%
  pivot_wider(names_from = eval_set,
                     values_from = c(auc, brier, log_loss, ece, calib_gap))

drift <- wide %>%
  mutate(
    d_auc      = auc_out_of_time      - auc_in_time_test,
    d_brier    = brier_out_of_time    - brier_in_time_test,
    d_log_loss = log_loss_out_of_time - log_loss_in_time_test,
    d_ece      = ece_out_of_time      - ece_in_time_test
  )

und <- drift %>% filter(model == "understat_xG")

if (nrow(und) == 1) {
  drift <- drift %>%
    mutate(
      excess_d_log_loss = d_log_loss - und$d_log_loss,
      excess_d_brier    = d_brier    - und$d_brier,
      excess_d_auc      = d_auc      - und$d_auc
    )
}

drift <- drift %>%
  select(model, starts_with("log_loss_"), starts_with("brier_"),
         starts_with("auc_"), starts_with("ece_"), starts_with("calib_gap_"),
         starts_with("d_"), starts_with("excess_"))

cat("\n--- Drift: in-time test vs out-of-time 2021+ ---\n")
print(as.data.frame(drift))
write_csv(drift, "results/04_drift_table.csv")

cat("\nPositive d_log_loss / d_brier means worse out of time.\n")
cat("excess_* is our drift minus Understat's drift on the same shots.\n")


# ---- 5. Season by season ------------------------------------------------------

# A single before/after number hides whether decay is gradual or a step change
# at one point. Within the paper era only held-out rows are scored, so no
# season is flattered by having been trained on.

season_eval <- bind_rows(
  test_m %>% mutate(eval_source = "in_time_holdout"),
  oot_m  %>% mutate(eval_source = "out_of_time")
)

season_rows <- list()
for (mdl in names(fits)) {
  p <- fits[[mdl]]$predict(season_eval)
  tmp <- season_eval %>% mutate(.p = p)
  for (s in sort(unique(tmp$season_start))) {
    sub <- tmp %>% filter(season_start == s)
    season_rows[[length(season_rows) + 1]] <-
      metrics(sub$status, sub$.p) %>%
      mutate(model = mdl, season_start = s,
             eval_source = sub$eval_source[1])
  }
}

# Understat on the same rows
for (s in sort(unique(season_eval$season_start))) {
  sub <- season_eval %>% filter(season_start == s, !is.na(xg_understat))
  if (nrow(sub) > 0) {
    season_rows[[length(season_rows) + 1]] <-
      metrics(sub$status, sub$xg_understat) %>%
      mutate(model = "understat_xG", season_start = s,
             eval_source = sub$eval_source[1])
  }
}

by_season <- bind_rows(season_rows) %>%
  select(model, season_start, eval_source, n, auc, brier, log_loss, ece,
         mean_pred, actual_rate, calib_gap) %>%
  arrange(model, season_start)

cat("\n--- Per-season performance (held-out rows only) ---\n")
print(as.data.frame(by_season))
write_csv(by_season, "results/04_by_season_metrics.csv")


# ---- 6. Calibration by era ----------------------------------------------------

# Where a model breaks matters as much as whether it breaks. An xG model that
# drifts usually stays sharp on easy chances and gets overconfident on the
# tail, which shows up in the top deciles and not in the AUC.

calib_era <- bind_rows(lapply(names(eval_sets), function(set_nm) {
  df <- eval_sets[[set_nm]]
  bind_rows(lapply(names(fits), function(mdl) {
    calibration_table(df$status, fits[[mdl]]$predict(df)) %>%
      mutate(model = mdl, eval_set = set_nm)
  }))
})) %>%
  select(model, eval_set, everything())

cat("\n--- Calibration by decile and era ---\n")
print(as.data.frame(calib_era))
write_csv(calib_era, "results/04_calibration_by_era.csv")


# ---- 7. Explanation drift -----------------------------------------------------

# The interpretable half of the enhancement, and the part that ties back to the
# parent paper. If mean |SHAP| for distanceToGoal falls between eras while
# situation rises, the model is leaning on different evidence in 2021+ than it
# did in 2014-20 - a finding a coach could act on, rather than a metric that
# got worse for unexplained reasons.
#
# Exact TreeSHAP over whole eras via xgboost predcontrib. DALEX's predict_parts
# is per-observation and far too slow at this scale; 03 uses it for the
# paper-faithful individual explanations.

shap_era <- bind_rows(lapply(names(eval_sets), function(set_nm) {
  shap_by_feature(fits$xgboost$model, eval_sets[[set_nm]]) %>%
    mutate(eval_set = set_nm)
}))

shap_wide <- shap_era %>%
  pivot_wider(names_from = eval_set, values_from = mean_abs_shap) %>%
  mutate(
    share_in_time = in_time_test / sum(in_time_test),
    share_oot     = out_of_time  / sum(out_of_time),
    share_change  = share_oot - share_in_time
  ) %>%
  arrange(desc(abs(share_change)))

cat("\n--- Feature attribution drift (xgboost, mean |SHAP|) ---\n")
print(as.data.frame(shap_wide))
write_csv(shap_wide, "results/04_shap_drift.csv")

cat("\nshare_change is the shift in each feature's slice of total attribution.\n")
cat("Large positive means the model leans on it more in 2021+ than it did\n")
cat("in the paper era.\n")


# ---- 8. Log loss by season plot -----------------------------------------------

# One line per model across every scored season, held-out paper era and
# out-of-time together, so a gradual decay and a step change at 2021 look
# different. Understat's xG is drawn as the benchmark: if our lines rise and
# Understat's rises with them, the environment moved, not our model. The 2025
# season is partial at download time, so its point rests on far fewer shots.

season_plot_data <- by_season %>%
  mutate(model = factor(model,
                        levels = c("logistic_regression", "random_forest",
                                   "xgboost", "understat_xG")))

season_plot <- ggplot2::ggplot(
    season_plot_data,
    ggplot2::aes(x = season_start, y = log_loss, colour = model,
                 linetype = model == "understat_xG")) +
  ggplot2::annotate("rect", xmin = 2020.5, xmax = Inf,
                    ymin = -Inf, ymax = Inf, fill = "grey92") +
  ggplot2::geom_vline(xintercept = 2020.5,
                      linetype = "dashed", colour = "grey50") +
  ggplot2::annotate("text", x = 2020.6, y = Inf,
                    label = "out of time", hjust = 0, vjust = 1.5,
                    colour = "grey40", size = 3.5) +
  ggplot2::geom_line(linewidth = 0.8) +
  ggplot2::geom_point(size = 2) +
  ggplot2::scale_x_continuous(breaks = sort(unique(season_plot_data$season_start))) +
  ggplot2::scale_linetype_manual(values = c(`FALSE` = "solid", `TRUE` = "dashed"),
                                 guide = "none") +
  ggplot2::labs(title    = "Log loss by season",
                subtitle = "Held-out 2014-2020 rows and the 2021+ out-of-time set; lower is better",
                x = "Season (start year)", y = "Log loss", colour = "Model") +
  ggplot2::theme_minimal(base_size = 13) +
  ggplot2::theme(legend.position = "bottom",
                 panel.grid.minor = ggplot2::element_blank())

print(season_plot)
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
ggplot2::ggsave("results/figures/04_log_loss_by_season.png", season_plot,
                width = 9, height = 6, dpi = 150, bg = "white")


cat("\n------------------------------------------------------------\n")
cat("Enhancement (a) complete. results/04_*.csv\n")
if (!USE_VALIDATION) {
  cat("validation_UNSEEN.csv untouched. Set USE_VALIDATION <- TRUE once\n")
  cat("the models are final, for one last confirmation run.\n")
}
cat("------------------------------------------------------------\n")
