# ==============================================================================
# 08_calibration_detail.R  -  is the calibration gap real, and where is it?
#
# 03 reports one gap per predicted-xG decile, with no uncertainty, over all
# shots pooled. Two questions it cannot answer:
#
#   1. Which decile gaps are bigger than chance? A decile holds ~6,300 test
#      shots, so the observed goal rate in it has a standard error of about
#      0.5 points by decile 9. Each gap gets a match-level bootstrap interval,
#      and each model gets a calibration intercept and slope.
#
#   2. Is the model calibrated within the kinds of shot players differ on?
#      A player's xG is the sum over his own shots, so a model that is right
#      on average but over-rates headers, say, over-rates every target man.
#      07's player assessments inherit exactly these subgroup gaps. Gaps are
#      reported by situation, by body part, and by body part x distance band.
#
# Same sign convention as 03: gap = mean_pred - actual_rate, positive means the
# model over-predicts.
#
# Reads:  data/splits/train.csv, data/splits/test.csv
# Writes: results/08_*.csv, results/figures/08_*.png
# Runtime: ~5 min (refits 02's three families; the bootstrap is vectorised).
#
# Requires 01_build_data.R to have run.
# ==============================================================================

N_BOOT     <- 1000
DIST_BANDS <- c(0, 6, 12, 18, 24, Inf)   # metres, as in 07
# A group with almost no goals has an observed rate pinned at ~0, so the
# bootstrap can only resample the predictions and its interval comes out
# falsely tight (long-range headers: 60 test shots, 0 goals). Such groups are
# listed but not reported.
MIN_GOALS  <- 10

source("02_model_utils.R")
library(tidyr)
library(ggplot2)

dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

cat("\n==============================================================\n")
cat("08_calibration_detail.R  -  calibration uncertainty and subgroups\n")
cat("==============================================================\n")


# ---- 1. Data and predictions --------------------------------------------------

train_m <- read_split("train")
test_m  <- read_split("test")
test_m  <- align_to_training(test_m, observed_levels(train_m), verbose = TRUE)

cat("\n--- Fitting on train.csv ---\n")
fits  <- fit_all(train_m)
preds <- lapply(fits, function(f) f$predict(test_m))

# Understat is added as a reference, on the rows that carry a value (all of
# them in the current archive). Same caveat as 03: it is scoring in-sample.
ok <- rep(TRUE, nrow(test_m))
if ("xg_understat" %in% names(test_m) && !all(is.na(test_m$xg_understat))) {
  ok <- !is.na(test_m$xg_understat)
  preds$understat_xG <- test_m$xg_understat
}
preds  <- lapply(preds, function(p) p[ok])
test_m <- test_m[ok, ]
MODELS <- names(preds)

cat(sprintf("\nScoring %d test shots from %d matches.\n",
            nrow(test_m), n_distinct(test_m$match_id)))


# ---- 2. Match-level bootstrap -------------------------------------------------

# Shots in the same match share a game state, so resample whole matches. Every
# replicate is a vector of match counts; drawing them once lets every model and
# every grouping reuse the same replicates, so their intervals are comparable.
# A group's gap in a replicate is then a ratio of weighted sums, which is a
# matrix product rather than a loop over replicates.

match_f <- factor(test_m$match_id)
set.seed(SEED)
W <- t(rmultinom(N_BOOT, nlevels(match_f), rep(1, nlevels(match_f))))

group_gaps <- function(y, p, g) {
  g  <- droplevels(factor(g))
  N  <- unclass(xtabs(~ match_f + g))
  SY <- unclass(xtabs(y ~ match_f + g))
  SP <- unclass(xtabs(p ~ match_f + g))
  WN <- W %*% N
  boot <- (W %*% SP - W %*% SY) / WN
  boot[WN == 0] <- NA
  n <- colSums(N)
  tibble(group       = colnames(N),
         n           = as.integer(n),
         goals       = as.integer(colSums(SY)),
         mean_pred   = colSums(SP) / n,
         actual_rate = colSums(SY) / n,
         gap         = mean_pred - actual_rate,
         lo          = apply(boot, 2, quantile, 0.025, na.rm = TRUE, names = FALSE),
         hi          = apply(boot, 2, quantile, 0.975, na.rm = TRUE, names = FALSE)) %>%
    mutate(beyond_noise = lo > 0 | hi < 0)
}


# ---- 3. Decile gaps with intervals --------------------------------------------

# Deciles are each model's own (as in 03), fixed on the full test set; only the
# shots inside them are resampled.

decile_gaps <- bind_rows(lapply(MODELS, function(m) {
  group_gaps(test_m$status, preds[[m]], ntile(preds[[m]], 10)) %>%
    mutate(model = m, bin = as.integer(group))
})) %>%
  select(model, bin, n, mean_pred, actual_rate, gap, lo, hi, beyond_noise)

cat("\n--- Decile gaps with 95% match-bootstrap intervals (percentage points) ---\n")
print(as.data.frame(decile_gaps %>%
                      transmute(model, bin, gap = round(100 * gap, 2),
                                lo = round(100 * lo, 2), hi = round(100 * hi, 2),
                                beyond_noise)),
      row.names = FALSE)
write_csv(decile_gaps, "results/08_decile_gaps_ci.csv")


# ---- 4. Calibration intercept and slope ---------------------------------------

# Logistic recalibration of each model's own predictions:
#   intercept: glm(y ~ offset(logit p)). 0 = right on average; below 0 means
#              the model predicts too many goals overall.
#   slope:     glm(y ~ logit p). 1 = spread right; below 1 means predictions
#              are too extreme (high ones too high, low ones too low), above 1
#              too timid.
# Intervals are from the same match bootstrap, so they respect clustering.

logit <- function(p) qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))

cal_fit <- function(y, lp, w = NULL) {
  c(intercept = coef(glm(y ~ offset(lp), family = binomial, weights = w))[[1]],
    slope     = coef(glm(y ~ lp,         family = binomial, weights = w))[[2]])
}

# Per-shot weight in a replicate = how many times its match was drawn. A
# couple of hundred replicates is plenty for a 95% interval on two numbers and
# keeps this to about a minute.
N_BOOT_GLM <- 200
shot_w <- W[seq_len(N_BOOT_GLM), as.integer(match_f), drop = FALSE]

cal_slopes <- bind_rows(lapply(MODELS, function(m) {
  lp  <- logit(preds[[m]])
  est <- cal_fit(test_m$status, lp)
  bt  <- t(vapply(seq_len(N_BOOT_GLM), function(b) {
    suppressWarnings(cal_fit(test_m$status, lp, w = shot_w[b, ]))
  }, numeric(2)))
  tibble(model = m,
         intercept = est[["intercept"]],
         intercept_lo = quantile(bt[, 1], 0.025, names = FALSE),
         intercept_hi = quantile(bt[, 1], 0.975, names = FALSE),
         slope = est[["slope"]],
         slope_lo = quantile(bt[, 2], 0.025, names = FALSE),
         slope_hi = quantile(bt[, 2], 0.975, names = FALSE))
}))

cat("\n--- Calibration intercept (ideal 0) and slope (ideal 1) ---\n")
print(as.data.frame(cal_slopes), digits = 3, row.names = FALSE)
write_csv(cal_slopes, "results/08_calibration_slope.csv")


# ---- 5. Subgroup calibration --------------------------------------------------

# Three cuts:
#   situation      penalties have a near-fixed conversion rate and sit almost
#                  entirely in the top decile, so they can drive decile 10.
#   body part      headers convert far worse than footed shots from the same spot.
#   body x distance, penalties excluded (a penalty is always at 11 m, and would
#                  swamp the foot 6-12 m cell). OtherBodyPart is too rare
#                  (~0.3% of shots) to cut by distance.

band_labels <- function(b) {
  lab <- paste0(head(b, -1), "-", tail(b, -1), " m")
  lab[is.infinite(tail(b, -1))] <- paste0(b[length(b) - 1], "+ m")
  lab
}

body <- case_when(test_m$shotType == "Head" ~ "Header",
                  test_m$shotType %in% c("LeftFoot", "RightFoot") ~ "Foot",
                  TRUE ~ "Other")
band <- cut(test_m$distanceToGoal, DIST_BANDS, right = FALSE,
            labels = band_labels(DIST_BANDS))
non_pen <- as.character(test_m$situation) != "Penalty"

# Factors, so groups keep a meaningful order (distance bands near to far)
# rather than an alphabetical one.
body_band <- factor(paste(body, band),
                    levels = paste(rep(c("Foot", "Header", "Other"),
                                       each = nlevels(band)),
                                   levels(band)))

CUTS <- list(
  "Situation" = list(g = test_m$situation, keep = TRUE),
  "Body part" = list(g = factor(body, levels = c("Foot", "Header", "Other")),
                     keep = TRUE),
  "Body part x distance (no penalties)" =
    list(g = body_band, keep = non_pen & body != "Other")
)

subgroups_all <- bind_rows(lapply(names(CUTS), function(cut_nm) {
  cu <- CUTS[[cut_nm]]
  # Rows outside the cut get group NA, which xtabs drops, so W still lines up
  # with every match.
  g <- cu$g
  g[!rep_len(cu$keep, nrow(test_m))] <- NA
  bind_rows(lapply(MODELS, function(m) {
    group_gaps(test_m$status, preds[[m]], g) %>% mutate(model = m)
  })) %>% mutate(cut = cut_nm)
})) %>%
  select(cut, group, model, n, goals, mean_pred, actual_rate, gap, lo, hi, beyond_noise)

too_few <- subgroups_all %>% filter(goals < MIN_GOALS) %>% distinct(cut, group, n, goals)
if (nrow(too_few) > 0) {
  cat(sprintf("\nNot reported (fewer than %d goals in test):\n", MIN_GOALS))
  print(as.data.frame(too_few), row.names = FALSE)
}
subgroups <- subgroups_all %>% filter(goals >= MIN_GOALS)

cat("\n--- Subgroup gaps (percentage points); * = interval excludes 0 ---\n")
print(as.data.frame(
  subgroups %>%
    transmute(cut, group, model, n,
              gap = sprintf("%+.2f [%+.2f, %+.2f]%s", 100 * gap, 100 * lo,
                            100 * hi, ifelse(beyond_noise, " *", ""))) %>%
    pivot_wider(names_from = model, values_from = gap)),
  row.names = FALSE, right = FALSE)
write_csv(subgroups, "results/08_subgroup_gaps.csv")


# ---- 6. Figures ---------------------------------------------------------------

model_cols   <- c(logistic_regression = "#2a78d6", random_forest = "#eb6834",
                  xgboost = "#1baf7a", understat_xG = "#4a3aa7")
model_shapes <- c(logistic_regression = 16, random_forest = 15,
                  xgboost = 17, understat_xG = 18)

theme_08 <- function() {
  theme_minimal(base_size = 12) +
    theme(plot.title.position = "plot",
          plot.subtitle = element_text(colour = "grey30"),
          panel.grid.minor = element_blank(),
          strip.text = element_text(face = "bold", hjust = 0),
          legend.position = "bottom",
          legend.title = element_blank())
}

# 6a. Decile gaps with intervals, one panel per model; the strip carries the
#     calibration slope and intercept.
strip_lab <- cal_slopes %>%
  mutate(lab = sprintf("%s\nslope %.2f [%.2f, %.2f], intercept %+.3f [%+.3f, %+.3f]",
                       model, slope, slope_lo, slope_hi,
                       intercept, intercept_lo, intercept_hi))
strip_lab <- setNames(strip_lab$lab, strip_lab$model)

dg <- decile_gaps %>% mutate(model = factor(model, levels = MODELS))
fig_dec <- ggplot(dg, aes(bin, 100 * gap, colour = model, shape = model)) +
  geom_hline(yintercept = 0, colour = "grey50") +
  geom_errorbar(aes(ymin = 100 * lo, ymax = 100 * hi), width = 0.25,
                linewidth = 0.5) +
  geom_point(aes(alpha = beyond_noise), size = 2.6) +
  facet_wrap(~ model, ncol = 2, labeller = as_labeller(strip_lab)) +
  scale_x_continuous(breaks = 1:10) +
  scale_colour_manual(values = model_cols, guide = "none") +
  scale_shape_manual(values = model_shapes, guide = "none") +
  scale_alpha_manual(values = c(`FALSE` = 0.45, `TRUE` = 1),
                     labels = c(`FALSE` = "interval includes 0 (within noise)",
                                `TRUE`  = "interval excludes 0")) +
  labs(title = "Calibration gap by decile, with 95% match-bootstrap intervals (test set)",
       subtitle = "Above 0 = over-predicts. Faded points are within sampling noise of perfect calibration.",
       x = "Predicted-xG decile", y = "Gap (percentage points)") +
  theme_08()
print(fig_dec)
ggsave("results/figures/08_decile_gaps_ci.png", fig_dec,
       width = 11, height = 8, dpi = 150, bg = "white")

# 6b. Subgroup gaps: one row per group, models dodged.
sg <- subgroups %>%
  mutate(model = factor(model, levels = rev(MODELS)),
         cut   = factor(cut, levels = names(CUTS)),
         label = sprintf("%s  (n = %s)", group, format(n, big.mark = ",")))
# Order rows within each cut as they were defined (distance bands in order).
sg$label <- factor(sg$label, levels = rev(unique(sg$label)))

fig_sub <- ggplot(sg, aes(100 * gap, label, colour = model, shape = model)) +
  geom_vline(xintercept = 0, colour = "grey50") +
  geom_errorbarh(aes(xmin = 100 * lo, xmax = 100 * hi), height = 0,
                 linewidth = 0.5,
                 position = position_dodge(width = 0.7)) +
  geom_point(size = 2.3, position = position_dodge(width = 0.7)) +
  facet_grid(cut ~ ., scales = "free_y", space = "free_y", switch = "y") +
  scale_colour_manual(values = model_cols, breaks = MODELS) +
  scale_shape_manual(values = model_shapes, breaks = MODELS) +
  labs(title = "Calibration gap by kind of shot, with 95% match-bootstrap intervals (test set)",
       subtitle = sprintf("Right of 0 = the model predicts more goals than were scored. n = test shots in the group. Groups with fewer than %d goals are omitted.", MIN_GOALS),
       x = "Gap: mean predicted xG - observed goal rate (percentage points)", y = NULL) +
  theme_08() +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 0, face = "bold", hjust = 1),
        panel.grid.major.y = element_blank())
print(fig_sub)
ggsave("results/figures/08_subgroup_gaps.png", fig_sub,
       width = 12, height = 10, dpi = 150, bg = "white")


cat("\n------------------------------------------------------------\n")
cat("results/08_decile_gaps_ci.csv      decile gaps with intervals\n")
cat("results/08_calibration_slope.csv   calibration intercept and slope\n")
cat("results/08_subgroup_gaps.csv       gaps by situation, body part, distance\n")
cat("results/figures/08_*.png           two figures\n")
cat("------------------------------------------------------------\n")
