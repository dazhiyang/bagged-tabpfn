#################################################################################
# 4.2.Fig.EnsembleDiag.R — Comprehensive model diagnostics figure
#################################################################################

rm(list = ls(all = TRUE))

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(scattermore)
})

invisible(utils::globalVariables(c("combo_f", "model_f", "nMBE", "nRMSE", "err", "model", "Time", "combo", "y", "x", "metric", "score", "angle")))

#################################################################################
# Parameter block
#################################################################################

base_font_family <- "Times"
text_size_pt <- 8

line_width_grid <- 0.15
line_width_data <- 0.35
line_width_axis <- 0.25

width_mm <- 160
height_mm <- 185

panel_tag_col_mm <- 6
panel_tag_inset_mm <- 0.75

sample_n_per_group <- 500
table_text_size <- text_size_pt * 5 / 14
scatter_pointsize <- 0.05 * 45
kde_n_grid <- 200
scatter_quantile_classes <- 10

wong <- c(
  orange = "#E69F00",
  sky_blue = "#56B4E9",
  blue_green = "#009E73",
  pale_violet = "#CC79A7",
  vermillion = "#D55E00",
  yellow = "#F0E442",
  blue = "#0072B2",
  black = "#000000"
)

viridis_continuous_option <- "viridis"

combo_order <- c("yHxP", "yHxS", "yLxP", "yLxS")
model_order <- c("Raw", "MLR", "KCDE", "XGBoost", "TabPFN")
model_colors <- c(
  wong["orange"],
  wong["sky_blue"],
  wong["blue_green"],
  wong["pale_violet"],
  wong["vermillion"]
)
model_colors <- unname(model_colors)
names(model_colors) <- model_order
combo_shapes <- c(yHxP = 16, yHxS = 17, yLxP = 15, yLxS = 18)
combo_label_expr <- c(
  yHxP = "italic(y)[H]*', '*italic(x)[P]",
  yHxS = "italic(y)[H]*', '*italic(x)[S]",
  yLxP = "italic(y)[L]*', '*italic(x)[P]",
  yLxS = "italic(y)[L]*', '*italic(x)[S]"
)
combo_labeller_parsed <- as_labeller(combo_label_expr, label_parsed)

#################################################################################
# Paths
#################################################################################

.get.script.dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  m <- grep("^--file=", args)
  if (length(m) > 0) return(dirname(normalizePath(sub("^--file=", "", args[m[1]]))))
  if (!is.null(sys.frame(1)$ofile)) return(dirname(normalizePath(sys.frame(1)$ofile)))
  getwd()
}

dir0 <- normalizePath(file.path(.get.script.dir(), ".."), mustWork = FALSE)
out_dir <- Sys.getenv("OUTPUT_DIR", file.path(dir0, "Data", "Output"))
out.fig <- Sys.getenv("OUTPUT_FIG", file.path(dir0, "tex", "Fig2.pdf"))
# Panel (f) reads machine-readable stats from 3.2.MW-Stats.R (same run as MW_Decomposition.tex).
# MW_Stats.txt is still written by 3.2; the .tex file is for tables only, not parsed here.
mw_stats_file <- Sys.getenv("MW_STATS_FILE", file.path(out_dir, "MW_Stats.txt"))

model_files <- c(
  Raw = "raw.txt",
  MLR = "MLR.txt",
  KCDE = "KCDE.txt",
  XGBoost = "XGBoost.txt",
  TabPFN = "TabPFN.txt"
)

#################################################################################
# Theme and helpers
#################################################################################

theme_pub <- function() {
  theme_bw(base_size = text_size_pt, base_family = base_font_family) +
    theme(
      text = element_text(family = base_font_family, size = text_size_pt),
      axis.title = element_text(size = text_size_pt),
      axis.text = element_text(size = text_size_pt),
      legend.title = element_text(size = text_size_pt),
      legend.text = element_text(size = text_size_pt),
      strip.text = element_text(size = text_size_pt, margin = margin(1, 1, 1, 1, "pt")),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      panel.grid.major = element_line(
        colour = grDevices::adjustcolor("black", alpha.f = 0.2),
        linewidth = line_width_grid
      ),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = line_width_axis),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      plot.margin = margin(2, 2, 2, 2, "pt")
    )
}

panel_tag_grob <- function(label) {
  inset <- grid::unit(panel_tag_inset_mm, "mm")
  grid::textGrob(
    label,
    x = grid::unit(1, "npc") - inset,
    y = grid::unit(1, "npc") - inset,
    hjust = 1,
    vjust = 1,
    gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt, fontface = "plain")
  )
}

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

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  dens$z[ii]
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

#################################################################################
# Data assembly
#################################################################################

all_pred <- bind_rows(lapply(seq_along(model_files), function(i) {
  read_pred(model_files[[i]], names(model_files)[i])
}))

all_pred <- all_pred %>%
  filter(combo %in% combo_order, model %in% model_order) %>%
  mutate(
    combo_f = factor(combo, levels = combo_order),
    model_f = factor(model, levels = model_order),
    err = x - y,
    abs_err = abs(err)
  )

metrics <- all_pred %>%
  group_by(model_f, combo_f) %>%
  summarise(
    MBE = mean(err, na.rm = TRUE),
    RMSE = sqrt(mean(err^2, na.rm = TRUE)),
    nMBE = 100 * mean(err, na.rm = TRUE) / mean(y, na.rm = TRUE),
    nRMSE = 100 * sqrt(mean(err^2, na.rm = TRUE)) / mean(y, na.rm = TRUE),
    .groups = "drop"
  )

mbe_best <- metrics %>%
  group_by(combo_f) %>%
  slice_min(order_by = abs(MBE), n = 1, with_ties = FALSE) %>%
  ungroup()

rmse_best <- metrics %>%
  group_by(combo_f) %>%
  slice_min(order_by = RMSE, n = 1, with_ties = FALSE) %>%
  ungroup()

#################################################################################
# (a) MBE table-like heatmap: color by W/m^2, text as %
#################################################################################

p_mbe <- ggplot(metrics, aes(x = combo_f, y = model_f, fill = MBE)) +
  geom_tile(colour = wong["black"], linewidth = line_width_grid * 0.6) +
  geom_tile(
    data = mbe_best,
    aes(x = combo_f, y = model_f),
    inherit.aes = FALSE,
    fill = NA,
    colour = wong["pale_violet"],
    linewidth = line_width_data * 1.1
  ) +
  geom_text(aes(label = sprintf("%.1f%%", nMBE)), family = base_font_family, size = table_text_size) +
  scale_fill_gradient2(
    low = unname(wong["orange"]),
    mid = "white",
    high = unname(wong["sky_blue"]),
    midpoint = 0,
    name = expression(MBE ~ "["*W ~ m^-2*"]")
  ) +
  coord_fixed(ratio = 0.42) +
  scale_x_discrete(labels = parse(text = combo_label_expr[combo_order]), expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Combo", y = "Model") +
  theme_pub() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.margin = margin(-2, 0, 0, 0, "pt"),
    legend.box.margin = margin(4, 0, 0, 0, "pt"),
    legend.box.spacing = unit(0.6, "pt")
  ) +
  guides(fill = guide_colorbar(direction = "horizontal", barheight = unit(2.2, "mm"), barwidth = unit(36, "mm"), title.position = "left"))

#################################################################################
# (b) RMSE table-like heatmap: viridis fill, text as %
#################################################################################

p_rmse <- ggplot(metrics, aes(x = combo_f, y = model_f, fill = RMSE)) +
  geom_tile(colour = wong["black"], linewidth = line_width_grid * 0.6) +
  geom_tile(
    data = rmse_best,
    aes(x = combo_f, y = model_f),
    inherit.aes = FALSE,
    fill = NA,
    colour = wong["pale_violet"],
    linewidth = line_width_data * 1.1
  ) +
  geom_text(aes(label = sprintf("%.1f%%", nRMSE)), family = base_font_family, size = table_text_size) +
  scale_fill_viridis_c(
    option = viridis_continuous_option,
    direction = -1,
    name = expression(RMSE ~ "["*W ~ m^-2*"]"),
    na.value = NA
  ) +
  coord_fixed(ratio = 0.42) +
  scale_x_discrete(labels = parse(text = combo_label_expr[combo_order]), expand = c(0, 0)) +
  scale_y_discrete(expand = c(0, 0)) +
  labs(x = "Combo", y = "Model") +
  theme_pub() +
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 0, hjust = 0.5),
    legend.margin = margin(-2, 0, 0, 0, "pt"),
    legend.box.margin = margin(4, 0, 0, 0, "pt"),
    legend.box.spacing = unit(0.6, "pt")
  ) +
  guides(fill = guide_colorbar(direction = "horizontal", barheight = unit(2.2, "mm"), barwidth = unit(36, "mm"), title.position = "left"))

#################################################################################
# (c) Prediction scatter (scattermore) for yHxP across five models
#################################################################################

set.seed(123)
pred_scatter <- all_pred %>%
  filter(combo == "yHxP") %>%
  mutate(model_f = factor(model, levels = model_order)) %>%
  group_by(model_f) %>%
  mutate(density = get_density(y, x, n = kde_n_grid)) %>%
  ungroup()

scatter_quantiles <- quantile(
  pred_scatter$density,
  probs = seq(0, 1, length.out = scatter_quantile_classes + 1),
  na.rm = TRUE
)
scatter_values <- scales::rescale(scatter_quantiles, to = c(0, 1))
scatter_values <- pmax(0, pmin(1, scatter_values))

p_scatter <- ggplot(pred_scatter, aes(x = y, y = x)) +
  geom_point(alpha = 0, stroke = 0, size = 0) +
  geom_abline(intercept = 0, slope = 1, linewidth = line_width_data, linetype = "dashed", colour = wong["black"]) +
  geom_scattermore(aes(colour = density), pointsize = scatter_pointsize) +
  geom_density_2d(linewidth = line_width_grid, colour = wong["black"]) +
  coord_fixed(xlim = c(0, 1100), ylim = c(0, 1100), expand = FALSE) +
  scale_colour_viridis_c(
    option = viridis_continuous_option,
    direction = 1,
    values = scatter_values,
    name = "Density"
  ) +
  facet_grid(. ~ model_f) +
  labs(
    x = expression("Observed" ~ "[" * W ~ m^-2 * "]"),
    y = expression("Predicted" ~ "[" * W ~ m^-2 * "]")
  ) +
  theme_pub() +
  theme(
    legend.position = "none",
    strip.background = element_rect(fill = "grey95", colour = "black")
  )

#################################################################################
# (d) Marginal error distribution by model (pooled combos)
#################################################################################

p_err <- ggplot(all_pred, aes(x = err, colour = model_f)) +
  geom_density(linewidth = line_width_data, alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = line_width_data, colour = wong["black"]) +
  scale_colour_manual(values = model_colors, name = NULL) +
  labs(
    x = expression("Error (Pred - Obs)" ~ "[" * W ~ m^-2 * "]"),
    y = "Density"
  ) +
  coord_cartesian(xlim = c(-400, 400)) +
  theme_pub() +
  theme(
    legend.position = "bottom",
    legend.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.spacing = unit(0.6, "pt"),
    legend.key.height = unit(text_size_pt * 0.55, "pt"),
    legend.key.width = unit(text_size_pt * 0.9, "pt")
  )

#################################################################################
# (e) Best-model share by combo (lowest |error| at each timestamp)
#################################################################################

best_share <- all_pred %>%
  group_by(combo_f, Time) %>%
  slice_min(order_by = abs_err, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  count(combo_f, model_f, name = "n_best") %>%
  group_by(combo_f) %>%
  mutate(share = 100 * n_best / sum(n_best)) %>%
  ungroup()

# Values underlying panel (e): print to console via a temp file (not saved under Data/).
tmp_panel_e <- tempfile(fileext = ".tsv")
utils::write.table(
  best_share %>%
    transmute(
      combo = as.character(combo_f),
      model = as.character(model_f),
      n_best,
      share_pct = share
    ),
  file = tmp_panel_e,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
message("Panel (e) — best-model counts and share (%) by combo:")
writeLines(readLines(tmp_panel_e))
unlink(tmp_panel_e)

p_share <- ggplot(best_share, aes(x = combo_f, y = share, fill = model_f)) +
  geom_col(width = 0.78, colour = wong["black"], linewidth = line_width_grid * 0.6, alpha = 0.7) +
  scale_fill_manual(values = model_colors, name = NULL) +
  scale_x_discrete(labels = parse(text = combo_label_expr[combo_order])) +
  labs(x = "Combo", y = "% best predictions") +
  theme_pub() +
  theme(
    legend.position = "bottom",
    legend.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.spacing = unit(0.6, "pt"),
    legend.key.height = unit(text_size_pt * 0.55, "pt"),
    legend.key.width = unit(text_size_pt * 0.9, "pt")
  )

#################################################################################
# (f) Polar plots: Murphy–Winkler statistics by combo (model lines)
#################################################################################

if (!file.exists(mw_stats_file)) {
  stop(
    "Missing MW stats file: ", mw_stats_file,
    "\nRun Code/3.2.MW-Stats.R first (writes MW_Stats.txt for this plot and MW_Decomposition.tex for tables)."
  )
}

mw_metric_levels <- c(
  "Association",
  "Uncond. bias",
  "Calibration",
  "Resolution",
  "Type-2 cond. bias",
  "Discrimination"
)

# Short axis text (definitions stay in paper / 3.2 Raw caption); avoids overlap in (f).
mw_metric_label_plot <- c(
  "Association" = "Assoc.",
  "Uncond. bias" = "Uncond.\nbias",
  "Calibration" = "Calib.",
  "Resolution" = "Resol.",
  "Type-2 cond. bias" = "Type-2\ncond.\nbias",
  "Discrimination" = "Discr."
)

mw_plot <- read.delim(mw_stats_file, sep = "\t", stringsAsFactors = FALSE) %>%
  filter(combo %in% combo_order, model %in% model_order, metric %in% mw_metric_levels) %>%
  mutate(
    combo_f = factor(combo, levels = combo_order),
    model_f = factor(model, levels = model_order),
    metric = factor(metric, levels = mw_metric_levels),
    metric_id = as.numeric(metric),
    theta = 2 * pi * (metric_id - 1) / length(mw_metric_levels) + pi / 2,
    x = score * cos(theta),
    y = score * sin(theta)
  )

mw_plot_closed <- mw_plot %>%
  group_by(combo_f, model_f) %>%
  arrange(metric_id, .by_group = TRUE) %>%
  group_modify(~ bind_rows(.x, .x %>% slice(1))) %>%
  ungroup()

radar_rings <- tidyr::expand_grid(
  combo_f = levels(all_pred$combo_f),
  ring = c(0.25, 0.50, 0.75, 1.00),
  theta = seq(0, 2 * pi, length.out = 240)
) %>%
  mutate(
    combo_f = factor(combo_f, levels = levels(all_pred$combo_f)),
    x = ring * cos(theta),
    y = ring * sin(theta)
  )

radar_spokes <- tidyr::expand_grid(
  combo_f = levels(all_pred$combo_f),
  metric = mw_metric_levels
) %>%
  mutate(
    combo_f = factor(combo_f, levels = levels(all_pred$combo_f)),
    metric = factor(metric, levels = mw_metric_levels),
    metric_id = as.numeric(metric),
    theta = 2 * pi * (metric_id - 1) / length(mw_metric_levels) + pi / 2,
    xend = 1.0 * cos(theta),
    yend = 1.0 * sin(theta)
  )

radar_label_r <- 1.26
radar_lim <- 1.32
# Manual nudges (data coords) for label placement on (f); rest use (0, 0).
mw_metric_label_dx <- c(
  "Association" = 0,
  "Uncond. bias" = 0.12,
  "Calibration" = 0,
  "Resolution" = 0,
  "Type-2 cond. bias" = -0.07,
  "Discrimination" = 0
)
mw_metric_label_dy <- c(
  "Association" = -0.06,
  "Uncond. bias" = 0.08,
  "Calibration" = 0,
  "Resolution" = 0.06,
  "Type-2 cond. bias" = -0.15,
  "Discrimination" = 0
)
radar_labels <- radar_spokes %>%
  mutate(
    mchr = as.character(metric),
    x = radar_label_r * cos(theta) + unname(mw_metric_label_dx[mchr]),
    y = radar_label_r * sin(theta) + unname(mw_metric_label_dy[mchr]),
    label = unname(mw_metric_label_plot[mchr])
  ) %>%
  select(-mchr)

p_radar <- ggplot() +
  geom_path(
    data = radar_rings,
    aes(x = x, y = y, group = interaction(combo_f, ring)),
    linewidth = line_width_grid,
    colour = grDevices::adjustcolor(wong["black"], alpha.f = 0.25)
  ) +
  geom_segment(
    data = radar_spokes,
    aes(x = 0, y = 0, xend = 1.02 * xend, yend = 1.02 * yend),
    linewidth = line_width_grid,
    colour = grDevices::adjustcolor(wong["black"], alpha.f = 0.25)
  ) +
  geom_polygon(
    data = mw_plot_closed,
    aes(x = x, y = y, group = model_f, colour = model_f),
    fill = NA,
    linewidth = line_width_data
  ) +
  geom_point(
    data = mw_plot,
    aes(x = x, y = y, colour = model_f),
    size = text_size_pt * 0.16
  ) +
  geom_text(
    data = radar_labels,
    aes(x = x, y = y, label = label),
    family = base_font_family,
    size = text_size_pt / ggplot2::.pt,
    lineheight = 0.85,
    hjust = 0.5,
    vjust = 0.5
  ) +
  scale_colour_manual(values = model_colors, name = NULL) +
  scale_x_continuous(limits = c(-radar_lim, radar_lim), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-radar_lim, radar_lim), expand = c(0, 0)) +
  coord_equal(clip = "off") +
  facet_wrap(~combo_f, nrow = 1, labeller = combo_labeller_parsed) +
  labs(x = NULL, y = NULL) +
  theme_pub() +
  theme(
    axis.text = element_blank(),
    axis.title = element_blank(),
    axis.ticks = element_blank(),
    axis.text.y = element_blank(),
    panel.grid = element_blank(),
    strip.background = element_rect(fill = "grey95", colour = "black"),
    strip.text.x = element_text(size = text_size_pt, margin = margin(1, 1, 1, 1, "pt")),
    legend.position = "bottom",
    legend.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.margin = margin(0, 0, 0, 0, "pt"),
    legend.box.spacing = unit(0.6, "pt"),
    legend.key.height = unit(text_size_pt * 0.55, "pt"),
    legend.key.width = unit(text_size_pt * 0.9, "pt"),
    axis.text.x = element_text(size = text_size_pt - 1)
  )

#################################################################################
# Compose
#################################################################################

row_ab <- gridExtra::arrangeGrob(
  grobs = list(
    panel_tag_grob("(a)"), ggplotGrob(p_mbe),
    panel_tag_grob("(b)"), ggplotGrob(p_rmse)
  ),
  ncol = 4,
  widths = unit(c(panel_tag_col_mm, 1, panel_tag_col_mm, 1), c("mm", "null", "mm", "null"))
)

row_c <- gridExtra::arrangeGrob(
  grobs = list(panel_tag_grob("(c)"), ggplotGrob(p_scatter)),
  ncol = 2,
  widths = unit(c(panel_tag_col_mm, 1), c("mm", "null"))
)

row_de <- gridExtra::arrangeGrob(
  grobs = list(
    panel_tag_grob("(d)"), ggplotGrob(p_err),
    panel_tag_grob("(e)"), ggplotGrob(p_share)
  ),
  ncol = 4,
  widths = unit(c(panel_tag_col_mm, 1, panel_tag_col_mm, 1), c("mm", "null", "mm", "null"))
)

row_f <- gridExtra::arrangeGrob(
  grobs = list(panel_tag_grob("(f)"), ggplotGrob(p_radar)),
  ncol = 2,
  widths = unit(c(panel_tag_col_mm, 1), c("mm", "null"))
)

fig <- gridExtra::arrangeGrob(
  grobs = list(row_ab, row_c, row_de, row_f),
  ncol = 1,
  heights = unit(c(0.82, 0.95, 0.82, 1.05), "null"),
  padding = unit(1.2, "mm")
)

dir.create(dirname(out.fig), recursive = TRUE, showWarnings = FALSE)
ggplot2::ggsave(
  filename = out.fig,
  plot = fig,
  device = grDevices::pdf,
  width = width_mm,
  height = height_mm,
  units = "mm",
  limitsize = FALSE,
  compress = TRUE,
  family = base_font_family
)

message("Wrote: ", out.fig)
