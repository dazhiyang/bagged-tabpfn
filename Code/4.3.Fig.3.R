#################################################################################
# 4.3.Fig.3.R
# (a) ΔRMSE vs unbagged TabPFN (ensemble C1–C10 one color, bagged TabPFN-B another).
# (b)–(c) Code/3.5.attention.py; (d) Code/3.3.bagged.py PCA CSV.
#################################################################################

rm(list = ls(all = TRUE))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(scattermore)
  library(scales)
})
invisible(utils::globalVariables(c(
  "context", "stage", "token", "y_bin", "pc1", "pc2", "layer",
  "from_token", "to_token", "attention", "combo_f", "series", "rmse_wm2_imp"
)))

#################################################################################
# Parameter block (edit here only)
#################################################################################
project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")

pca_file <- file.path(diag_dir, "feature_token_pca_layers_long.csv")
file_attn <- file.path(diag_dir, "attention_feature_layers_long.csv")
baseline_tex <- file.path(fig_dir, "baseline_tables.tex")
fig_out <- file.path(fig_dir, "Fig3.pdf")
n_label_bins <- 7L

base_font_family <- "Times"
text_size_pt <- 8
line_width_axis <- 0.25
width_mm <- 160
height_mm <- 160
# Panel (d) PCA: scattermore point size (larger = bigger pixels). Facets use free x/y scales.
pca_pointsize <- 5

# Panel letters — layout matches Code/4.2.Fig.2.R (tag column + plot column per panel).
panel_tag_col_mm <- 6
panel_tag_inset_mm <- 0.75
# Row 1 plot widths (a), (b), (c) — relative null units; increase middle value to widen (b).
row1_plot_width_rel <- c(1, 1.3, 1.05)
# Panel (a): extra plot margin / symmetric y-expand (ΔRMSE can be − / + vs TabPFN).
panel_a_plot_margin_pt <- margin(20, 1, 20, 1, "pt")
panel_a_y_expand_top_mult <- 0.07

# Wong order from SKILL: 1 orange, 2 sky blue, 3 blue green, 4 pale violet,
# 5 vermillion, 6 yellow, 7 blue, 8 black.
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
# TabPFN tokens only (panels b–d); strings match Code/4.1.Fig.1.R for these codes.
token_label_expr <- c(
  xP = "italic(x)[P]",
  SZA = "italic(Z)",
  lcc = "italic(f)[L]",
  mcc = "italic(f)[M]",
  tcsw = "italic(w)[sn]",
  tcwv = "italic(w)",
  label = "italic(y)[H]"
)
parse_token_labels <- function(x) {
  lab <- unname(token_label_expr[x])
  miss <- is.na(lab)
  if (any(miss)) lab[miss] <- paste0("`", x[miss], "`")
  parse(text = lab)
}

combo_order <- c("yHxP", "yHxS", "yLxP", "yLxS")
combo_label_expr <- c(
  yHxP = "italic(y)[H]*', '*italic(x)[P]",
  yHxS = "italic(y)[H]*', '*italic(x)[S]",
  yLxP = "italic(y)[L]*', '*italic(x)[P]",
  yLxS = "italic(y)[L]*', '*italic(x)[S]"
)
combo_labeller_parsed <- ggplot2::as_labeller(combo_label_expr, label_parsed)

#################################################################################
# Read and format
#################################################################################
pca <- read.csv(pca_file, stringsAsFactors = FALSE)
if (nrow(pca) == 0) {
  stop("feature_token_pca_layers_long.csv is empty. Run Code/3.3.bagged.py first (PCA rows).")
}
attn <- read.csv(file_attn, stringsAsFactors = FALSE)
if (nrow(attn) == 0) {
  stop("attention_feature_layers_long.csv is empty. Run Code/3.5.attention.py first (attention CSV).")
}

feature_token_levels <- c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv", "label")
pca <- pca %>%
  mutate(
    token = trimws(token),
    y_bin = trimws(y_bin),
    context = factor(context, levels = c("full", "b10"), labels = c("Full", "B10")),
    stage = factor(stage, levels = c("Input", "L1", "L2", "L3", "L6", "L9", "L12")),
    y_bin = factor(y_bin, levels = paste0("C", seq_len(n_label_bins)))
  )
tok_chr <- unique(as.character(pca$token))
if (all(tok_chr %in% feature_token_levels)) {
  pca <- pca %>% mutate(token = factor(token, levels = feature_token_levels))
} else if (all(grepl("^token_[0-9]+$", tok_chr))) {
  tok_levels <- tok_chr[order(as.integer(sub("^token_", "", tok_chr)))]
  pca <- pca %>% mutate(token = factor(token, levels = tok_levels))
} else {
  pca <- pca %>% mutate(token = factor(token, levels = sort(tok_chr)))
}

if (all(c("from_group", "to_group") %in% names(attn))) {
  attn <- attn %>%
    dplyr::rename(from_token = from_group, to_token = to_group)
}
if (all(c("from_feature", "to_feature") %in% names(attn))) {
  attn <- attn %>%
    dplyr::rename(from_token = from_feature, to_token = to_feature)
}

layer_levels <- c("L1", "L2", "L3", "L6", "L9", "L12")

token_order <- unique(c(
  attn %>% distinct(from_token) %>% pull(from_token),
  attn %>% distinct(to_token) %>% pull(to_token)
))
token_order <- c(setdiff(token_order, "label"), "label")

attn <- attn %>%
  mutate(
    context = factor(context, levels = c("full", "b10", "delta")),
    layer = factor(layer, levels = layer_levels),
    from_token = factor(from_token, levels = token_order),
    to_token = factor(to_token, levels = token_order)
  )

attn_fb <- attn %>%
  filter(context %in% c("full", "b10"))

attn_delta <- attn %>%
  filter(context %in% c("full", "b10")) %>%
  select(layer, from_token, to_token, context, attention) %>%
  tidyr::pivot_wider(names_from = context, values_from = attention) %>%
  mutate(
    context = "delta",
    attention = b10 - full
  ) %>%
  select(layer, from_token, to_token, context, attention)

attn_plot <- bind_rows(attn_fb, attn_delta) %>%
  mutate(
    context = factor(context, levels = c("full", "b10", "delta"), labels = c("Full", "B10", "B10 - Full"))
  )

delta_lim <- max(abs(attn_delta$attention), na.rm = TRUE)
no_classes <- 10
quantiles <- as.numeric(stats::quantile(attn_fb$attention, probs = seq(0, 1, length.out = no_classes + 1), na.rm = TRUE, type = 8))
quantiles <- sort(unique(quantiles))
quantiles_rescaled <- scales::rescale(quantiles, to = c(0, 1), from = range(attn_fb$attention, na.rm = TRUE))
quantile_cols <- viridisLite::viridis(length(quantiles))

#################################################################################
# Theme
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
      strip.switch.pad.grid = unit(0.4, "pt"),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      panel.grid.major = element_line(
        colour = grDevices::adjustcolor("black", alpha.f = 0.20),
        linewidth = 0.15
      ),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = line_width_axis),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      plot.margin = margin(1, 1, 1, 1, "pt"),
      axis.text.x = element_text(angle = 0),
      strip.background = element_rect(fill = "grey95")
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

#################################################################################
# baseline_tables.tex → ΔRMSE vs TabPFN (W m^-2): ensemble-only data for panel (a)
#################################################################################
cell_first_wm2 <- function(cell) {
  m <- regexpr("\\\\shortstack\\[c\\]\\{", cell, perl = TRUE)
  if (m[[1L]] == -1L) return(NA_real_)
  start <- m[[1L]] + attr(m, "match.length")
  rest <- substr(cell, start, nchar(cell))
  m2 <- regexpr("-?[0-9]+", rest, perl = TRUE)
  if (m2[[1L]] == -1L) return(NA_real_)
  as.numeric(regmatches(rest, m2))
}

combo_from_tex_row <- function(line) {
  m <- regexec("^\\$y_\\\\text\\{([HL])\\},\\\\, x_\\\\text\\{([PS])\\}\\$", line, perl = TRUE)
  g <- regmatches(line, m)[[1L]]
  if (length(g) != 3L) return(NA_character_)
  paste0("y", g[[2L]], "x", g[[3L]])
}

read_baseline_imp_vs_tabpfn_long <- function(tex_path) {
  if (!file.exists(tex_path)) stop("Missing file: ", tex_path)
  txt <- readLines(tex_path, warn = FALSE)
  i_ens <- grep("\\\\label\\{tb:rmse_ens\\}", txt)
  if (length(i_ens) == 0L) stop("baseline_tables.tex: missing \\\\label{tb:rmse_ens}")
  i_ens <- i_ens[[length(i_ens)]]
  mids <- grep("\\midrule", txt, fixed = TRUE)
  bots <- grep("\\bottomrule", txt, fixed = TRUE)
  if (length(mids) < 2L || length(bots) < 2L) {
    stop("baseline_tables.tex: expected at least two \\\\midrule / \\\\bottomrule pairs")
  }
  r_mid <- max(mids[mids < i_ens])
  r_bot <- min(bots[bots > r_mid & bots < i_ens])
  rows_main <- txt[(r_mid + 1L):(r_bot - 1L)]

  mid2 <- min(mids[mids > i_ens])
  r_bot2 <- min(bots[bots > mid2])
  rows_ens <- txt[(mid2 + 1L):(r_bot2 - 1L)]

  ref_tabpfn_rmse <- setNames(rep(NA_real_, length(combo_order)), combo_order)
  for (ln in rows_main) {
    if (!grepl("^\\$y_\\\\text\\{", ln)) next
    co <- combo_from_tex_row(ln)
    if (!(co %in% combo_order)) stop("Unexpected combo label in baseline main table row: ", ln)
    parts <- strsplit(ln, " & ", fixed = TRUE)[[1L]]
    if (length(parts) < 11L) stop("Unexpected main table row: ", ln)
    ref_tabpfn_rmse[co] <- cell_first_wm2(parts[[11L]])
  }
  ens_k <- seq_len(10L)
  series_levels <- c(paste0("B", ens_k), "Bagged")
  out <- list()
  for (ln in rows_ens) {
    if (!grepl("^\\$y_\\\\text\\{", ln)) next
    co <- combo_from_tex_row(ln)
    if (!(co %in% combo_order)) stop("Unexpected combo label in ensemble table row: ", ln)
    parts <- strsplit(ln, " & ", fixed = TRUE)[[1L]]
    if (length(parts) < 12L) stop("Unexpected ensemble table row: ", ln)
    ref <- unname(ref_tabpfn_rmse[co])
    if (!is.finite(ref)) stop("Missing TabPFN RMSE for combo ", co)
    ens <- vapply(parts[seq_len(length(series_levels)) + 1L], cell_first_wm2, numeric(1L))
    for (k in ens_k) {
      dm <- ref - ens[[k]]
      out[[length(out) + 1L]] <- data.frame(
        combo = co,
        series = paste0("B", k),
        rmse_wm2_imp = dm,
        stringsAsFactors = FALSE
      )
    }
    dm <- ref - ens[[11L]]
    out[[length(out) + 1L]] <- data.frame(
      combo = co,
      series = "Bagged",
      rmse_wm2_imp = dm,
      stringsAsFactors = FALSE
    )
  }
  bind_rows(out) %>%
    mutate(
      combo_f = factor(combo, levels = combo_order),
      series = factor(series, levels = series_levels)
    )
}

panel_a_long <- read_baseline_imp_vs_tabpfn_long(baseline_tex)
bar_levels <- c(paste0("B", seq_len(10L)), "Bagged")
bar_cols <- c(rep(unname(wong["sky_blue"]), 10L), unname(wong["orange"]))
names(bar_cols) <- bar_levels

#################################################################################
# Plot — panel (a): ΔRMSE vs unbagged TabPFN (facet stripes match 4.2.Fig.2.R)
#################################################################################
p_rmse <- ggplot(panel_a_long, aes(x = series, y = rmse_wm2_imp, fill = series)) +
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = line_width_axis,
    colour = grDevices::adjustcolor(unname(wong["black"]), alpha.f = 0.35)
  ) +
  geom_col(width = 0.82, colour = wong["black"], linewidth = 0.12) +
  facet_wrap(
    ~combo_f,
    ncol = 1L,
    labeller = combo_labeller_parsed,
    strip.position = "right"
  ) +
  scale_y_continuous(
    expand = ggplot2::expansion(mult = c(panel_a_y_expand_top_mult, panel_a_y_expand_top_mult))
  ) +
  scale_fill_manual(values = bar_cols, name = NULL) +
  scale_x_discrete(drop = FALSE) +
  labs(
    x = NULL,
    y = expression(Delta ~ RMSE ~ "(vs TabPFN)" ~ "[" * W ~ m^-2 * "]")
  ) +
  theme_pub() +
  theme(
    legend.position = "none",
    plot.margin = panel_a_plot_margin_pt,
    axis.text.x = element_text(angle = 55, hjust = 1, vjust = 1, size = text_size_pt * 0.85),
    # Vertical strip labels (same sense as row facets for layer in panel b).
    strip.text.y.right = element_text(angle = -90, hjust = 0.5, vjust = 0.5, size = text_size_pt),
    strip.placement = "outside"
  )

#################################################################################
# Plot — panel (b): attention heatmaps (Full / B10)
#################################################################################
p_base <- ggplot(attn_fb %>% mutate(context = factor(context, levels = c("full", "b10"), labels = c("Full", "B10"))), aes(x = to_token, y = from_token, fill = attention)) +
  geom_tile(colour = wong["black"], linewidth = 0.06) +
  coord_fixed() +
  facet_grid(layer ~ context, drop = FALSE) +
  scale_fill_gradientn(
    colours = quantile_cols,
    values = quantiles_rescaled,
    breaks = quantiles,
    name = "Attention"
  ) +
  scale_x_discrete(labels = parse_token_labels) +
  scale_y_discrete(labels = parse_token_labels) +
  labs(x = "To token", y = "From token", title = NULL) +
  theme_pub() +
  theme(legend.position = "none")

#################################################################################
# Plot — panel (c): delta attention (B10 - Full)
#################################################################################
p_delta <- ggplot(attn_plot %>% filter(context == "B10 - Full"), aes(x = to_token, y = from_token, fill = attention)) +
  geom_tile(colour = wong["black"], linewidth = 0.06) +
  coord_fixed() +
  facet_grid(layer ~ ., drop = FALSE) +
  scale_fill_gradient2(
    low = wong["orange"],
    mid = "white",
    high = wong["sky_blue"],
    midpoint = 0,
    limits = c(-delta_lim, delta_lim),
    name = "Delta"
  ) +
  scale_x_discrete(labels = parse_token_labels) +
  scale_y_discrete(labels = parse_token_labels) +
  labs(x = "To token", y = "From token", title = NULL) +
  theme_pub() +
  theme(
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 4, "pt")
  ) +
  guides(
    fill = guide_colorbar(
      direction = "vertical",
      barheight = grid::unit(42, "mm"),
      barwidth = grid::unit(2.5, "mm"),
      title.position = "top",
      title.hjust = 0.5
    )
  )

#################################################################################
# Plot — panel (d): token PCA (PC1 vs PC2), colour = input feature (token)
# Uses scattermore for speed.
#################################################################################
feat_token_levels <- c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv")
feat_cols <- c(
  xP = unname(wong["orange"]),
  SZA = unname(wong["sky_blue"]),
  lcc = unname(wong["blue_green"]),
  mcc = unname(wong["pale_violet"]),
  tcsw = unname(wong["vermillion"]),
  tcwv = unname(wong["yellow"])
)
feat_tokens_plot <- intersect(feat_token_levels, unique(as.character(pca$token)))
feat_cols_plot <- feat_cols[feat_tokens_plot]

p_pca <- ggplot() +
  facet_grid(
    context ~ stage,
    drop = FALSE,
    scales = "free",
    labeller = labeller(context = label_value, stage = label_value)
  )

for (ft in feat_tokens_plot) {
  p_pca <- p_pca +
    scattermore::geom_scattermore(
      data = pca %>% filter(as.character(token) == ft),
      mapping = aes(x = pc1, y = pc2),
      color = unname(feat_cols[[ft]]),
      pointsize = pca_pointsize,
      alpha = 0.35,
      pixels = c(700, 700),
      inherit.aes = FALSE
    )
}

legend_df <- data.frame(
  token = factor(feat_tokens_plot, levels = feat_tokens_plot),
  pc1 = 0,
  pc2 = 0
)

p_pca <- p_pca +
  geom_point(data = legend_df, aes(x = pc1, y = pc2, colour = token), alpha = 1, size = 0.01, inherit.aes = FALSE) +
  scale_colour_manual(
    values = feat_cols_plot,
    breaks = feat_tokens_plot,
    labels = parse_token_labels,
    name = NULL
  ) +
  labs(x = "PC1", y = "PC2") +
  theme_pub() +
  theme(
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 0, "pt"),
    axis.text = element_blank()
  ) +
  guides(colour = guide_legend(override.aes = list(size = 3.2, alpha = 1), ncol = 1))

g_rmse <- ggplotGrob(p_rmse)
g_a <- ggplotGrob(p_base)
g_b <- ggplotGrob(p_delta)
g_c <- ggplotGrob(p_pca)

# Same arrangement pattern as 4.2.Fig.2.R: each panel is [tag mm column | plot null].
row_abc <- gridExtra::arrangeGrob(
  grobs = list(
    panel_tag_grob("(a)"), g_rmse,
    panel_tag_grob("(b)"), g_a,
    panel_tag_grob("(c)"), g_b
  ),
  ncol = 6L,
  widths = grid::unit(
    c(
      panel_tag_col_mm, row1_plot_width_rel[[1L]],
      panel_tag_col_mm, row1_plot_width_rel[[2L]],
      panel_tag_col_mm, row1_plot_width_rel[[3L]]
    ),
    c("mm", "null", "mm", "null", "mm", "null")
  )
)

row_d <- gridExtra::arrangeGrob(
  grobs = list(panel_tag_grob("(d)"), g_c),
  ncol = 2L,
  widths = grid::unit(c(panel_tag_col_mm, 1), c("mm", "null"))
)

fig <- gridExtra::arrangeGrob(
  grobs = list(row_abc, row_d),
  ncol = 1L,
  heights = grid::unit(c(1, 0.5), "null"),
  padding = grid::unit(1.2, "mm")
)

#################################################################################
# Save figure (PDF)
#################################################################################
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(
  fig_out,
  fig,
  device = grDevices::pdf,
  width = width_mm,
  height = height_mm,
  units = "mm",
  limitsize = FALSE,
  compress = TRUE,
  family = base_font_family
)
message("Wrote: ", fig_out)
