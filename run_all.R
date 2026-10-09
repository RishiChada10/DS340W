# ==============================================================================
# run_all.R  -  entry point
#
# Explainable xG: replication of Cavus & Biecek (2022) plus two enhancements.
#
#   01_build_data.R    downloads, cleans, tags eras, splits once, drift
#                      diagnostics                                    [shared]
#   02_model_utils.R   features, encoding, model families, metrics
#                      (sourced by 03 and 04, never run directly)     [shared]
#   03_calibration.R   enhancement (c): calibration-focused model-family
#                      comparison
#   04_oot_drift.R     enhancement (a): out-of-time generalization,
#                      train 2014-2020, test 2021-2025
#
# TO RUN EVERYTHING:
#   setwd() to the folder containing these scripts, then
#   source("run_all.R")
#   or from a terminal:  Rscript run_all.R
#
# FIRST RUN ONLY: uncomment the install block below, run it once, then comment
# it back out. Downloads are cached under data/raw_by_league/, so the first
# pass takes a few minutes longer than later ones.
#
# Expected total runtime after the first download: 25-40 minutes, most of it
# the repeated cross-validation in 03.
#
# All tables land in results/. Nothing outside this folder is touched.
# ==============================================================================


# ---- First run only -----------------------------------------------------------

# install.packages(c("remotes", "dplyr", "readr", "tidyr",
#                    "ranger", "xgboost", "pROC", "DALEX"))
# remotes::install_github("JaseZiv/worldfootballR", upgrade = "never")


# ---- Dependency check ---------------------------------------------------------

required <- c("worldfootballR", "dplyr", "readr", "tidyr",
              "ranger", "xgboost", "pROC", "DALEX")
absent <- required[!vapply(required, requireNamespace, logical(1),
                           quietly = TRUE)]

if (length(absent) > 0) {
  stop("Missing packages: ", paste(absent, collapse = ", "),
       "\nUncomment the install block at the top of run_all.R and run it once.")
}

for (f in c("01_build_data.R", "02_model_utils.R",
            "03_calibration.R", "04_oot_drift.R")) {
  if (!file.exists(f)) {
    stop("Cannot find ", f, "\nsetwd() to the folder holding the scripts.")
  }
}


# ---- Run ----------------------------------------------------------------------

stages <- c("01_build_data.R", "03_calibration.R", "04_oot_drift.R")

started <- Sys.time()
timings <- data.frame(stage = character(), minutes = numeric(),
                      stringsAsFactors = FALSE)

for (s in stages) {
  cat("\n\n##############################################################\n")
  cat("# ", s, "\n")
  cat("##############################################################\n\n")

  t0 <- Sys.time()
  source(s, echo = FALSE)
  mins <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

  timings <- rbind(timings,
                   data.frame(stage = s, minutes = round(mins, 2),
                              stringsAsFactors = FALSE))
  cat("\n[", s, "finished in", round(mins, 1), "min ]\n")
}


# ---- Summary ------------------------------------------------------------------

cat("\n\n##############################################################\n")
cat("# Done in", round(as.numeric(difftime(Sys.time(), started,
                                           units = "mins")), 1), "minutes\n")
cat("##############################################################\n\n")

print(timings, row.names = FALSE)

cat("\nResults written:\n")
res <- sort(list.files("results", pattern = "\\.csv$"))
if (length(res) == 0) {
  cat("  (none found - check the output above for errors)\n")
} else {
  for (f in res) cat("  results/", f, "\n", sep = "")
}

cat("\nKey tables for the report:\n")
cat("  results/01_drift_psi.csv        covariate shift, paper era -> 2021+\n")
cat("  results/03_test_metrics.csv     model-family comparison, enhancement (c)\n")
cat("  results/03_cv_summary.csv       is the ranking stable across folds\n")
cat("  results/04_drift_table.csv      the headline of enhancement (a)\n")
cat("  results/04_shap_drift.csv       what the model leans on, by era\n")

cat("\nStill closed: data/splits/validation_UNSEEN.csv\n")
cat("Set USE_VALIDATION <- TRUE in 04_oot_drift.R once the models are final.\n\n")
