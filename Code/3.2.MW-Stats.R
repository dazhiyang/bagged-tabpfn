#################################################################################
# 3.2.MW-Stats.R — Compute Murphy–Winkler statistics for model outputs
#################################################################################

rm(list = ls(all = TRUE))

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

combo_order <- c("yHxP", "yHxS", "yLxP", "yLxS")
model_order <- c("Raw", "MLR", "KCDE", "XGBoost", "TabPFN")

model_files <- c(
  Raw = "raw.txt",
  MLR = "MLR.txt",
  KCDE = "KCDE.txt",
  XGBoost = "XGBoost.txt",
  TabPFN = "TabPFN.txt"
)

.get.script.dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", args[m[1]]))))
  if (!is.null(sys.frame(1)$ofile)) return(dirname(normalizePath(sys.frame(1)$ofile)))
  getwd()
}

dir0 <- normalizePath(file.path(.get.script.dir(), ".."), mustWork = FALSE)
out_dir <- Sys.getenv("OUTPUT_DIR", file.path(dir0, "Data", "Output"))
out_stats <- Sys.getenv("OUTPUT_STATS", file.path(out_dir, "MW_Stats.txt"))
out_tex <- Sys.getenv("OUTPUT_MW_TEX", file.path(dir0, "tex", "MW_Decomposition.tex"))

read_pred <- function(file_name, model_name) {
  fp <- file.path(out_dir, file_name)
  if (!file.exists(fp)) stop("Missing file: ", fp)
  read.delim(fp, sep = "\t", stringsAsFactors = FALSE) %>%
    transmute(
      Time = .data$Time,
      combo = .data$combo,
      y = as.numeric(.data$y),
      x = as.numeric(.data$x),
      model = model_name
    )
}

KCDE <- function(independent.variable, dependent.variable) {
  x <- independent.variable
  y <- dependent.variable
  id.seq <- order(x)
  est <- rep(NA_real_, length(x))
  est[id.seq] <- stats::ksmooth(
    x,
    y,
    kernel = "normal",
    bandwidth = 10,
    n.points = length(x),
    x.points = x
  )$y
  est
}

all_pred <- bind_rows(lapply(seq_along(model_files), function(i) {
  read_pred(model_files[[i]], names(model_files)[i])
})) %>%
  filter(combo %in% combo_order, model %in% model_order)

decomp <- all_pred %>%
  group_by(model, combo) %>%
  group_modify(~ {
    y_obs <- .x$y
    x_pred <- .x$x

    mse <- mean((x_pred - y_obs)^2, na.rm = TRUE)
    vy <- stats::var(y_obs, na.rm = TRUE)
    vx <- stats::var(x_pred, na.rm = TRUE)
    covar <- stats::cov(y_obs, x_pred, use = "complete.obs")
    ub <- (mean(x_pred, na.rm = TRUE) - mean(y_obs, na.rm = TRUE))^2

    eyx <- KCDE(independent.variable = x_pred, dependent.variable = y_obs)
    cali <- mean((x_pred - eyx)^2, na.rm = TRUE)
    res <- mean((eyx - mean(y_obs, na.rm = TRUE))^2, na.rm = TRUE)

    exy <- KCDE(independent.variable = y_obs, dependent.variable = x_pred)
    t2b <- mean((y_obs - exy)^2, na.rm = TRUE)
    dis <- mean((exy - mean(x_pred, na.rm = TRUE))^2, na.rm = TRUE)

    assoc <- stats::cor(y_obs, x_pred, use = "complete.obs")

    tibble(
      MSE = mse,
      Vx = vx,
      Vy = vy,
      CovXY = covar,
      Uncond_Bias = ub,
      Calibration = cali,
      Resolution = res,
      Type2_Cond_Bias = t2b,
      Discrimination = dis,
      Association = assoc
    )
  }) %>%
  ungroup()

metric_levels <- c(
  "Association",
  "Uncond. bias",
  "Calibration",
  "Resolution",
  "Type-2 cond. bias",
  "Discrimination"
)

stats_long <- decomp %>%
  transmute(
    model,
    combo,
    Association = .data$Association,
    `Uncond. bias` = .data$Uncond_Bias,
    Calibration = .data$Calibration,
    Resolution = .data$Resolution,
    `Type-2 cond. bias` = .data$Type2_Cond_Bias,
    Discrimination = .data$Discrimination
  ) %>%
  pivot_longer(
    cols = c("Association", "Uncond. bias", "Calibration", "Resolution", "Type-2 cond. bias", "Discrimination"),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(metric = factor(metric, levels = metric_levels)) %>%
  mutate(
    value_raw = value,
    value = ifelse(
      metric == "Association",
      value,
      sqrt(pmax(value, 0))
    )
  )

# Rank scores within each (combo, metric) so facet (f) compares models fairly per combo.
stats_long <- stats_long %>%
  group_by(combo, metric) %>%
  mutate(
    direction = ifelse(metric %in% c("Uncond. bias", "Calibration", "Type-2 cond. bias"), "lower_better", "higher_better"),
    score_raw = ifelse(direction == "lower_better", -value, value),
    score = {
      if (dplyr::n_distinct(score_raw[is.finite(score_raw)]) <= 1) {
        rep(0.55, dplyr::n())
      } else {
        scales::rescale(rank(score_raw, ties.method = "average"), to = c(0.18, 1.00))
      }
    }
  ) %>%
  ungroup() %>%
  arrange(combo, metric, factor(model, levels = model_order))

#################################################################################
# LaTeX tables (bias--variance, calibration--refinement, likelihood--base rate)
#################################################################################

combo_cell_latex <- list(
  yHxP = "$y_\\text{H},\\, x_\\text{P}$",
  yHxS = "$y_\\text{H},\\, x_\\text{S}$",
  yLxP = "$y_\\text{L},\\, x_\\text{P}$",
  yLxS = "$y_\\text{L},\\, x_\\text{S}$"
)

product_phrase <- function(mod) {
  if (identical(mod, "Raw")) {
    "the raw products"
  } else {
    paste0("the ", mod, "-corrected products")
  }
}

latex_model_slug <- function(model_name) {
  gsub("[^A-Za-z0-9]", "", model_name)
}

fmt_sq <- function(x) {
  b <- as.integer(round(sqrt(pmax(x, 0, na.rm = TRUE)), 0))
  paste0("$", formatC(b, width = 3, format = "d", flag = " "), "^2$")
}

fmt_cov <- function(covar) {
  b <- as.integer(round(sqrt(abs(covar)), 0))
  pref <- if (!is.finite(covar) || covar >= 0) "" else "-"
  paste0("$", pref, formatC(b, width = 3, format = "d", flag = " "), "^2$")
}

write_mw_decomposition_tex <- function(decomp_tbl, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  con <- file(path, open = "wt", encoding = "UTF-8")
  on.exit(close(con), add = TRUE)
  writeLines("% Auto-generated by Code/3.2.MW-Stats.R (do not edit by hand).", con)
  writeLines("% One table per model: bias--variance, calibration--refinement, and likelihood--base rate (shared MSE, V(X), V(Y)).", con)
  writeLines("% booktabs required; \\small applied inside each table. Label: \\label{tb:decomp-<ModelSlug>} (one per table).", con)
  for (mod in model_order) {
    dm <- decomp_tbl %>% dplyr::filter(.data$model == mod)
    slug <- latex_model_slug(mod)
    writeLines("", con)
    writeLines(paste0("% ----- Model: ", mod, " -----"), con)

    cap_combined <- if (identical(mod, "Raw")) {
      paste0(
        "Results of bias--variance, calibration--refinement, and likelihood--base rate decompositions ",
        "of mean square error of ",
        product_phrase(mod),
        ". ",
        "The results are written as exponentiations, such that all bases have the unit of W\\,m$^{-2}$. ",
        "Var.\\,X $=\\mathbb{V}(X)$; Var.\\,Y $=\\mathbb{V}(Y)$; ",
        "Cov $=\\mathtt{Cov}(X,Y)$; ",
        "Uncond.\\ bias $=[\\mathbb{E}(X)-\\mathbb{E}(Y)]^2$; ",
        "Calibration $=\\mathbb{E}_X[X-\\mathbb{E}(Y|X)]^2$; ",
        "Resolution $=\\mathbb{E}_X[\\mathbb{E}(Y|X)-\\mathbb{E}(Y)]^2$; ",
        "Type-2 cond.\\ bias $=\\mathbb{E}_Y[Y-\\mathbb{E}(X|Y)]^2$; ",
        "Discrimination $=\\mathbb{E}_Y[\\mathbb{E}(X|Y)-\\mathbb{E}(X)]^2$."
      )
    } else {
      paste0("Same as Table~\\ref{tb:decomp-Raw}, but for ", mod, "-corrected retrievals.")
    }

    writeLines("\\begin{table}[!ht]", con)
    writeLines("\\centering", con)
    writeLines(paste0("\\caption{", cap_combined, "}"), con)
    writeLines(paste0("\\label{tb:decomp-", slug, "}"), con)
    writeLines("\\small", con)
    writeLines("\\begin{tabular}{lccccccccc}", con)
    writeLines("\\toprule", con)
    writeLines(paste0(
      " & & \\multicolumn{4}{c}{Bias--variance} & ",
      "\\multicolumn{2}{c}{Calibration--refinement} & ",
      "\\multicolumn{2}{c}{Likelihood--base rate} \\\\"
    ), con)
    writeLines("\\cmidrule(lr){3-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}", con)
    writeLines(
      paste0(
        " & MSE & Var.\\,X & Var.\\,Y & Cov & Uncond.\\ bias & ",
        "Calibration & Resolution & Type-2 cond.\\ bias & Discrimination \\\\"
      ),
      con
    )
    writeLines("\\midrule", con)
    for (cb in combo_order) {
      r <- dm %>% dplyr::filter(.data$combo == cb) %>% dplyr::slice(1)
      if (nrow(r) != 1) next
      combo_tex <- combo_cell_latex[[cb]]
      writeLines(paste0(
        "  ", combo_tex, " & ",
        fmt_sq(r$MSE), " & ",
        fmt_sq(r$Vx), " & ",
        fmt_sq(r$Vy), " & ",
        fmt_cov(r$CovXY), " & ",
        fmt_sq(r$Uncond_Bias), " & ",
        fmt_sq(r$Calibration), " & ",
        fmt_sq(r$Resolution), " & ",
        fmt_sq(r$Type2_Cond_Bias), " & ",
        fmt_sq(r$Discrimination), " \\\\"
      ), con)
    }
    writeLines("\\bottomrule", con)
    writeLines("\\end{tabular}", con)
    writeLines("\\normalsize", con)
    writeLines("\\end{table}", con)
  }
}

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write.table(stats_long, file = out_stats, sep = "\t", quote = FALSE, row.names = FALSE)
write_mw_decomposition_tex(decomp, out_tex)

message("Wrote: ", out_stats)
message("Wrote: ", out_tex)
