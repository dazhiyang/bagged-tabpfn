#################################################################################
# This code is written by Dazhi Yang (a) and Yun Chen (b)
# (a) Department of Electrical Engineering and Automation, Harbin Institute of Technology
# (b) Public Meteorological Service Center, China Meteorological Administration
# emails: yangdazhi.nus@gmail.com, chenyunpku@163.com
#################################################################################

rm(list = ls(all = TRUE))
suppressPackageStartupMessages(library(dplyr))

dir0 <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
out_dir <- file.path(dir0, "Data", "Output")
tex_dir <- file.path(dir0, "tex")
dir.create(tex_dir, recursive = TRUE, showWarnings = FALSE)
fp_tex <- file.path(tex_dir, "baseline_tables.tex")

#################################################################################
# Must match ENS_K in Code/2.5.TabPFN-E.py (TabPFN-C1.txt … TabPFN-C{ENS_K}.txt)
#################################################################################
ENS_K <- 10L

#################################################################################
# Main tables (MBE / RMSE): Raw, MLR, KCDE, XGB, TabPFN (same filenames as 3.2 / 4.2).
#################################################################################
method_outputs_main <- c(
  Raw = "raw.txt",
  MLR = "MLR.txt",
  KCDE = "KCDE.txt",
  XGBoost = "XGBoost.txt",
  TabPFN = "TabPFN.txt"
)

combo_row_order <- c("yHxP", "yHxS", "yLxP", "yLxS")

# Same combo math as Code/3.2.MW-Stats.R (combo_cell_latex)
combo_cell_latex <- list(
  yHxP = "$y_\\text{H},\\, x_\\text{P}$",
  yHxS = "$y_\\text{H},\\, x_\\text{S}$",
  yLxP = "$y_\\text{L},\\, x_\\text{P}$",
  yLxS = "$y_\\text{L},\\, x_\\text{S}$"
)

# Ensemble RMSE table: C1 … C_K then TabPFN-E last (mean of C1 … C_K; Code/2.5.TabPFN-E.py).
ensemble_col_keys <- c(
  stats::setNames(paste0("x_C", seq_len(ENS_K)), paste0("C", seq_len(ENS_K))),
  stats::setNames("x_E", "TabPFN-E")
)
ensemble_headers <- names(ensemble_col_keys)

#################################################################################
# Metrics: x = prediction, y = observation (W·m⁻²)
#################################################################################
summarise_combo <- function(df) {
  df %>%
    group_by(combo) %>%
    summarise(
      MBE = mean(x - y, na.rm = TRUE),
      RMSE = sqrt(mean((x - y)^2, na.rm = TRUE)),
      nMBE = mean(x - y, na.rm = TRUE) / mean(y, na.rm = TRUE) * 100,
      nRMSE = sqrt(mean((x - y)^2, na.rm = TRUE)) / mean(y, na.rm = TRUE) * 100,
      .groups = "drop"
    )
}

fmt_math_int <- function(x) {
  xi <- as.integer(round(x))
  if (xi < 0) sprintf("$%d$", xi) else sprintf("%d", xi)
}

fmt_math_pct <- function(p) {
  if (p < 0) sprintf("$%.1f\\%%$", p) else sprintf("%.1f\\%%", p)
}

# Stacked cell (\\shortstack): primary statistic on top, normalized \\% below — body font (no \\scriptsize).
fmt_pair_cell <- function(val, pct, bold = FALSE) {
  line1 <- fmt_math_int(val)
  line2 <- sprintf("(%s)", fmt_math_pct(pct))
  inner <- sprintf("\\shortstack[c]{%s\\\\%s}", line1, line2)
  wrap_bold(inner, bold)
}

cell <- function(metrics, obs_key, ret_key) {
  co <- paste0(obs_key, ret_key)
  row <- metrics[metrics$combo == co, , drop = FALSE]
  if (nrow(row) != 1L) {
    stop("Missing or duplicate combo: ", co)
  }
  row
}

parse_combo <- function(combo) {
  list(obs = substr(combo, 1L, 2L), ret = substr(combo, 3L, 4L))
}

row_label_for_combo <- function(combo) {
  combo_cell_latex[[combo]]
}

read_pred_file <- function(filename) {
  fp <- file.path(out_dir, filename)
  if (!file.exists(fp)) {
    stop("Missing output file: ", fp)
  }
  read.table(fp, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
}

read_method_metrics_main <- function() {
  metrics <- vector("list", length(method_outputs_main))
  names(metrics) <- names(method_outputs_main)
  for (nm in names(method_outputs_main)) {
    tab <- read_pred_file(method_outputs_main[[nm]])
    metrics[[nm]] <- summarise_combo(tab)
  }
  metrics
}

#################################################################################
# Wide frame: join TabPFN-C1 … TabPFN-C_K on reference grid (first method_outputs_main file).
#################################################################################
build_ensemble_wide <- function() {
  ref <- read_pred_file(method_outputs_main[[1]])
  if (!all(c("Time", "combo", "y") %in% names(ref))) {
    stop("reference output file must contain Time, combo, y")
  }
  out <- ref %>% select(Time, combo, y)

  for (k in seq_len(ENS_K)) {
    fn <- sprintf("TabPFN-C%d.txt", k)
    tab <- read_pred_file(fn)
    xk <- sprintf("x_C%d", k)
    out <- out %>%
      left_join(
        tab %>% transmute(Time, combo, !!xk := x),
        by = c("Time", "combo")
      )
  }
  ccols <- paste0("x_C", seq_len(ENS_K))
  if (any(!ccols %in% names(out))) {
    stop("Missing TabPFN-C prediction columns after join; check TabPFN-C*.txt")
  }
  out$x_E <- rowMeans(as.matrix(out[, ccols]), na.rm = TRUE)

  nas <- sum(is.na(out$x_E)) + sum(as.matrix(is.na(out[, ccols])))
  if (nas > 0L) {
    warning("ensemble wide frame has ", nas, " NA in x_C* columns (check merges)")
  }
  out
}

metrics_from_wide_col <- function(wide, xcol) {
  d <- wide %>% transmute(combo, y, x = .data[[xcol]])
  summarise_combo(d)
}

#################################################################################
# LaTeX tables
#################################################################################
make_table_header <- function(method_names) {
  n <- length(method_names)
  align <- paste0("l", paste(rep("c", n), collapse = ""))
  hdr <- paste0("& ", paste(method_names, collapse = " & "), " \\\\")
  c(
    sprintf("\\begin{tabular}{%s}", align),
    "\\toprule",
    hdr,
    "\\midrule"
  )
}

# Wide table: [Combo | MBE × methods | RMSE × methods]; cmidrules under each block
make_table_header_wide_mbe_rmse <- function(method_names) {
  n <- length(method_names)
  j_lo <- 2L
  j_hi <- 1L + n
  k_lo <- 2L + n
  k_hi <- 1L + 2L * n
  align <- paste0("l", paste(rep("c", 2L * n), collapse = ""))
  hdr_methods <- paste(rep(method_names, 2L), collapse = " & ")
  c(
    sprintf("\\begin{tabular}{%s}", align),
    "\\toprule",
    sprintf(" & \\multicolumn{%d}{c}{MBE} & \\multicolumn{%d}{c}{RMSE} \\\\", n, n),
    sprintf("\\cmidrule(lr){%d-%d}\\cmidrule(lr){%d-%d}", j_lo, j_hi, k_lo, k_hi),
    paste0("& ", hdr_methods, " \\\\"),
    "\\midrule"
  )
}

tabular_footer <- c("\\bottomrule", "\\end{tabular}")
table_env_close <- c("\\normalsize", "\\end{table}", "")

wrap_bold <- function(x, bold) {
  if (bold) sprintf("{\\bfseries\\boldmath %s}", x) else x
}

extract_main_metric_block <- function(metrics_list, combo, type = c("MBE", "RMSE")) {
  type <- match.arg(type)
  if (type == "MBE") {
    vcol <- "MBE"
    pcol <- "nMBE"
  } else {
    vcol <- "RMSE"
    pcol <- "nRMSE"
  }
  p <- parse_combo(combo)
  nmeth <- length(method_outputs_main)
  vals <- numeric(nmeth)
  pcts <- numeric(nmeth)
  for (i in seq_len(nmeth)) {
    nm <- names(method_outputs_main)[i]
    m <- metrics_list[[nm]]
    r <- cell(m, p$obs, p$ret)
    vals[i] <- r[[vcol]][1L]
    pcts[i] <- r[[pcol]][1L]
  }
  list(vals = vals, pcts = pcts)
}

metric_row_main <- function(metrics_list, combo, type = c("MBE", "RMSE")) {
  type <- match.arg(type)
  blk <- extract_main_metric_block(metrics_list, combo, type)
  vals <- blk$vals
  pcts <- blk$pcts
  nmeth <- length(vals)
  ibest <- if (type == "MBE") which.min(abs(vals)) else which.min(vals)
  pieces <- vapply(seq_len(nmeth), function(i) {
    fmt_pair_cell(vals[i], pcts[i], bold = (i == ibest))
  }, character(1))
  sprintf("%s & %s \\\\", row_label_for_combo(combo), paste(pieces, collapse = " & "))
}

table_body_main <- function(metrics_list, type = c("MBE", "RMSE")) {
  type <- match.arg(type)
  vapply(combo_row_order, function(co) metric_row_main(metrics_list, co, type), character(1))
}

# One row per combo: MBE (all methods) then RMSE (all methods); bold best within each half-row
combo_row_main_mbe_rmse <- function(metrics_list, combo) {
  mb <- extract_main_metric_block(metrics_list, combo, "MBE")
  rm <- extract_main_metric_block(metrics_list, combo, "RMSE")
  nmeth <- length(mb$vals)
  ib_m <- which.min(abs(mb$vals))
  ib_r <- which.min(rm$vals)
  cells_mbe <- vapply(seq_len(nmeth), function(i) {
    fmt_pair_cell(mb$vals[i], mb$pcts[i], bold = (i == ib_m))
  }, character(1))
  cells_rmse <- vapply(seq_len(nmeth), function(i) {
    fmt_pair_cell(rm$vals[i], rm$pcts[i], bold = (i == ib_r))
  }, character(1))
  sprintf(
    "%s & %s \\\\",
    row_label_for_combo(combo),
    paste(c(cells_mbe, cells_rmse), collapse = " & ")
  )
}

table_body_main_wide_mbe_rmse <- function(metrics_list) {
  vapply(combo_row_order, function(co) combo_row_main_mbe_rmse(metrics_list, co), character(1))
}

metric_row_ens <- function(metrics_ens_list, combo, type = c("MBE", "RMSE")) {
  type <- match.arg(type)
  if (type == "MBE") {
    vcol <- "MBE"
    pcol <- "nMBE"
    best_idx <- function(v) which.min(abs(v))
  } else {
    vcol <- "RMSE"
    pcol <- "nRMSE"
    best_idx <- function(v) which.min(v)
  }
  p <- parse_combo(combo)
  nmeth <- length(metrics_ens_list)
  vals <- numeric(nmeth)
  pcts <- numeric(nmeth)
  for (i in seq_len(nmeth)) {
    m <- metrics_ens_list[[i]]
    r <- cell(m, p$obs, p$ret)
    vals[i] <- r[[vcol]][1L]
    pcts[i] <- r[[pcol]][1L]
  }
  ibest <- best_idx(vals)
  pieces <- vapply(seq_len(nmeth), function(i) {
    fmt_pair_cell(vals[i], pcts[i], bold = (i == ibest))
  }, character(1))
  sprintf("%s & %s \\\\", row_label_for_combo(combo), paste(pieces, collapse = " & "))
}

table_body_ens <- function(metrics_ens_list, type = c("MBE", "RMSE")) {
  type <- match.arg(type)
  vapply(combo_row_order, function(co) metric_row_ens(metrics_ens_list, co, type), character(1))
}

#################################################################################
# Build metrics
#################################################################################
metrics_main <- read_method_metrics_main()

wide_ens <- build_ensemble_wide()
metrics_ens_list <- vector("list", length(ensemble_col_keys))
names(metrics_ens_list) <- names(ensemble_col_keys)
for (hdr in names(ensemble_col_keys)) {
  xc <- ensemble_col_keys[[hdr]]
  metrics_ens_list[[hdr]] <- metrics_from_wide_col(wide_ens, xc)
}

#################################################################################
# LaTeX headers (escape where needed)
#################################################################################
hdr_main <- names(method_outputs_main)

cap_rmse_ens <- paste0(
  "Columns \\textbf{C1--C", ENS_K,
  "} are bootstrap ensemble members (\\texttt{TabPFN-C\\textit{m}.txt}, Code/2.5.TabPFN-E.py); ",
  "\\textbf{TabPFN-E} (last column) is their row-wise mean. ",
  "Best value per row is bold (lowest RMSE). ",
  "Stacked layout as Table~\\ref{tb:result_main}: W\\,m$^{-2}$ above, normalized \\% below."
)

cap_baseline_main <- paste0(
  "Mean bias error (MBE) and root mean square error (RMSE) of the raw and various corrected retrievals. ",
  "Both W\\,m$^{-2}$ and \\% (in parentheses) metrics are given. ",
  "Best results are in bold."
)

lines_baseline_main <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  paste0("\\caption{", cap_baseline_main, "}"),
  "\\label{tb:result_main}",
  "\\small",
  make_table_header_wide_mbe_rmse(hdr_main),
  table_body_main_wide_mbe_rmse(metrics_main),
  c(tabular_footer, table_env_close)
)

lines_rmse_ens <- c(
  "\\begin{table}[!ht]",
  "\\centering",
  paste0("\\caption{", cap_rmse_ens, "}"),
  "\\label{tb:rmse_ens}",
  "\\small",
  make_table_header(ensemble_headers),
  table_body_ens(metrics_ens_list, "RMSE"),
  c(tabular_footer, table_env_close)
)

lines_doc <- c(
  "% Generated by Code/3.1.overall_metric.R — do not edit by hand.",
  "% Tables: (1) MBE and RMSE on one row per combo (Raw + MLR–TabPFN; MBE columns | RMSE columns)",
  "%         (2) RMSE C1–C_K + TabPFN-E (last).",
  "% \\input{tex/baseline_tables} from your main .tex (adjust path).",
  "% Requires TabPFN-C1.txt … TabPFN-C{K}.txt in Data/Output (K = ENS_K in this file).",
  "% Best values use {\\bfseries\\boldmath ...} so math is visibly bold.",
  "% Label: tb:result_main (combined MBE | RMSE baseline table).",
  "",
  lines_baseline_main,
  lines_rmse_ens
)

writeLines(lines_doc, fp_tex, useBytes = TRUE)