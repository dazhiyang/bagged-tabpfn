#################################################################################
# 4.3.Fig.3.R — Fig 3 PDF (tex/Fig3.pdf).
# (a) ΔRMSE vs TabPFN for B1–B10 and TabPFN-B (baseline_tables.tex). (b) Attention + Δ attention
#     heatmaps (one ggplot, ggnewscale; facets Full / B10 / B10−Full × layers L3–L12). (c) Token PCA
#     (PC1 vs PC2) via ggh4x::facet_grid2(..., independent = "all") for facet_grid-style strips and
#     per-panel free axes.
# Layout: patchwork — top row (a)|(b) with top_row_widths, bottom row (c) full width.
#################################################################################

rm(list = ls(all = TRUE))

if (!requireNamespace("ggnewscale", quietly = TRUE)) {
  stop("Install ggnewscale: install.packages(\"ggnewscale\")")
}
if (!requireNamespace("ggh4x", quietly = TRUE)) {
  stop("Install ggh4x (PCA panel strips + fully free scales): install.packages(\"ggh4x\")")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(ggnewscale)
  library(ggh4x)
  library(patchwork)
  library(scattermore)
})
invisible(utils::globalVariables(c("combo_f", "series", "rmse_wm2_imp")))

project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")
file_attn <- file.path(diag_dir, "attention_feature_layers_long.csv")
pca_file <- file.path(diag_dir, "feature_token_pca_layers_long.csv")
baseline_tex <- file.path(fig_dir, "baseline_tables.tex")
fig_out <- file.path(fig_dir, "Fig3.pdf")
fig_width_mm <- 160
fig_height_mm <- 125
n_label_bins <- 3L
pca_pointsize <- 5
# When fig_height_mm shrinks, **mm** / **pt** in themes & guides stay the same length on paper —
# only panel drawing areas shrink → whitespace looks “stuck”. coord_fixed() letterboxing still
# applies; Δ colorbar below uses **lines** so it scales with theme text (not fixed mm).
rel_heights_bcd <- c(1.6, 1)
top_row_widths <- c(4, 7)
# Panel (a): baseline_tables.tex → ΔRMSE vs TabPFN (W m^-2).
panel_a_margin_v_pt <- max(2, round(fig_height_mm * 0.055))
panel_a_plot_margin_pt <- margin(panel_a_margin_v_pt, 1, panel_a_margin_v_pt, 1, "pt")
panel_a_y_expand_top_mult <- 0.07
# BC Delta bar: ggplot puts the guide in its own row under the axis title; theme_bw axis.title.x
# bottom margin + legend.margin top create a visible gap under "To token". Patchwork stacks BC
# directly on panel (c) with little external gap, so the bar can look nearer PCA than the axis title.
bc_axis_title_x_margin <- margin(t = 0, r = 0, b = 0, l = 0, "pt")
bc_legend_margin <- margin(t = 0, r = 0, b = 0, l = 4, "pt")
pca_plot_margin <- margin(0, 1, 1, 1, "pt")
# Δ legend on panel (b): vertical bar; heights in **lines** (see header comment).
delta_legend_barheight <- 30
delta_legend_barwidth <- 2.2

base_font_family <- "Times"
text_size_pt <- 8
line_width_axis <- 0.25

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

attn <- read.csv(file_attn, stringsAsFactors = FALSE)
if (nrow(attn) == 0) stop("attention CSV empty: ", file_attn)

if (all(c("from_group", "to_group") %in% names(attn))) {
  attn <- attn %>% dplyr::rename(from_token = from_group, to_token = to_group)
}
if (all(c("from_feature", "to_feature") %in% names(attn))) {
  attn <- attn %>% dplyr::rename(from_token = from_feature, to_token = to_feature)
}

layer_levels <- c("L1", "L2", "L3", "L6", "L9", "L12")
stage_levels_bc <- c("L3", "L6", "L9", "L12")
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

attn_fb <- attn %>% filter(context %in% c("full", "b10"))

attn_delta <- attn %>%
  filter(context %in% c("full", "b10")) %>%
  select(layer, from_token, to_token, context, attention) %>%
  tidyr::pivot_wider(names_from = context, values_from = attention) %>%
  mutate(context = "delta", attention = b10 - full) %>%
  select(layer, from_token, to_token, context, attention)

attn_plot <- bind_rows(attn_fb, attn_delta) %>%
  mutate(
    context = factor(context, levels = c("full", "b10", "delta"), labels = c("Full", "B10", "B10 - Full"))
  )

pca <- read.csv(pca_file, stringsAsFactors = FALSE)
if (nrow(pca) == 0) stop("PCA CSV empty: ", pca_file)

pca_feature_levels <- c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv", "label")
pca <- pca %>%
  mutate(
    token = trimws(token),
    y_bin = trimws(y_bin),
    context = factor(context, levels = c("full", "b10"), labels = c("Full", "B10")),
    stage = factor(stage, levels = c("Input", "L1", "L2", "L3", "L6", "L9", "L12")),
    y_bin = factor(y_bin, levels = paste0("C", seq_len(n_label_bins)))
  )
tok_chr <- unique(as.character(pca$token))
if (all(tok_chr %in% pca_feature_levels)) {
  pca <- pca %>% mutate(token = factor(token, levels = pca_feature_levels))
} else if (all(grepl("^token_[0-9]+$", tok_chr))) {
  tok_levels <- tok_chr[order(as.integer(sub("^token_", "", tok_chr)))]
  pca <- pca %>% mutate(token = factor(token, levels = tok_levels))
} else {
  pca <- pca %>% mutate(token = factor(token, levels = sort(tok_chr)))
}

pca_legend_tokens <- c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv")
feat_cols <- c(
  xP = unname(wong["orange"]),
  SZA = unname(wong["sky_blue"]),
  lcc = unname(wong["blue_green"]),
  mcc = unname(wong["pale_violet"]),
  tcsw = unname(wong["vermillion"]),
  tcwv = unname(wong["yellow"])
)
feat_tokens_plot <- intersect(pca_legend_tokens, unique(as.character(pca$token)))
feat_cols_plot <- feat_cols[feat_tokens_plot]

delta_lim <- max(abs(attn_delta$attention), na.rm = TRUE)
no_classes <- 10
quantiles <- as.numeric(stats::quantile(attn_fb$attention, probs = seq(0, 1, length.out = no_classes + 1), na.rm = TRUE, type = 8))
quantiles <- sort(unique(quantiles))
quantiles_rescaled <- scales::rescale(quantiles, to = c(0, 1), from = range(attn_fb$attention, na.rm = TRUE))
quantile_cols <- viridisLite::viridis(length(quantiles))

row_levels <- c("Full", "B10", "B10 - Full")
df_b <- attn_fb %>%
  filter(as.character(layer) %in% stage_levels_bc) %>%
  mutate(
    row_lab = factor(
      ifelse(as.character(context) == "full", "Full", "B10"),
      levels = row_levels
    ),
    stage_col = factor(as.character(layer), levels = stage_levels_bc)
  )
df_c <- attn_plot %>%
  filter(as.character(context) == "B10 - Full", as.character(layer) %in% stage_levels_bc) %>%
  mutate(
    row_lab = factor("B10 - Full", levels = row_levels),
    stage_col = factor(as.character(layer), levels = stage_levels_bc)
  )

theme_pub <- function() {
  theme_bw(base_size = text_size_pt, base_family = base_font_family) +
    theme(
      text = element_text(family = base_font_family, size = text_size_pt),
      axis.title = element_text(size = text_size_pt),
      axis.text = element_text(size = text_size_pt),
      legend.title = element_text(size = text_size_pt),
      legend.text = element_text(size = text_size_pt),
      strip.text = element_text(size = text_size_pt, margin = margin(1, 1, 1, 1, "pt")),
      strip.switch.pad.grid = grid::unit(0.4, "pt"),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      panel.grid.major = element_line(
        colour = grDevices::adjustcolor("black", alpha.f = 0.20),
        linewidth = 0.15
      ),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = line_width_axis),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "white", colour = NA),
      plot.margin = margin(1, 1, 1, 1, "pt"),
      axis.text.x = element_text(angle = 0),
      strip.background = element_rect(fill = "grey95"),
      strip.placement = "outside",
      # Default tag placement uses plot margins → extra vertical strip + patchwork reads as row gap.
      plot.tag.location = "topleft",
      plot.tag = element_text(
        family = base_font_family,
        size = text_size_pt,
        face = "plain",
        margin = margin(0, 0, 0, 0, "pt"),
        lineheight = 1
      )
    )
}

#################################################################################
# Panel (a) — ΔRMSE vs TabPFN from baseline_tables.tex (ensemble table: B1–B10 + bagged column → label TabPFN-B)
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
  series_levels <- c(paste0("B", ens_k), "TabPFN-B")
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
      series = "TabPFN-B",
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
bar_levels <- c(paste0("B", seq_len(10L)), "TabPFN-B")
bar_cols <- c(rep(unname(wong["sky_blue"]), 10L), unname(wong["orange"]))
names(bar_cols) <- bar_levels

panel_a <- ggplot(panel_a_long, aes(x = series, y = rmse_wm2_imp, fill = series)) +
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
    y = expression(Delta ~ RMSE ~ "(vs TabPFN)" ~ "[" * W ~ m^-2 * "]"),
    tag = "(a)"
  ) +
  theme_pub() +
  theme(
    legend.position = "none",
    plot.margin = panel_a_plot_margin_pt,
    axis.text.x = element_text(angle = 55, hjust = 1, vjust = 1, size = text_size_pt * 0.85),
    strip.text.y.right = element_text(angle = -90, hjust = 0.5, vjust = 0.5, size = text_size_pt),
    strip.placement = "outside"
  )

p_pca <- ggplot() +
  ggh4x::facet_grid2(
    rows = vars(context),
    cols = vars(stage),
    scales = "free",
    independent = "all",
    drop = FALSE,
    labeller = labeller(context = label_value, stage = label_value)
  )

for (ft in feat_tokens_plot) {
  p_pca <- p_pca +
    scattermore::geom_scattermore(
      data = pca %>% filter(as.character(token) == ft),
      mapping = aes(x = pc1, y = pc2),
      color = unname(feat_cols[[ft]]),
      pointsize = pca_pointsize,
      alpha = 0.3,
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
  labs(x = "PC1", y = "PC2", tag = "(c)") +
  theme_pub() +
  theme(
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 0, "pt"),
    axis.ticks = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    plot.margin = pca_plot_margin
  ) +
  guides(colour = guide_legend(override.aes = list(size = 3.2, alpha = 1), ncol = 1))

# One plot: layer strips on top, row_lab strips on right (facet_grid default).
fig_bc <- ggplot() +
  facet_grid(rows = vars(row_lab), cols = vars(stage_col), drop = FALSE) +
  geom_tile(
    data = df_b,
    aes(x = to_token, y = from_token, fill = attention),
    colour = wong["black"],
    linewidth = 0.06
  ) +
  scale_x_discrete(labels = parse_token_labels, expand = ggplot2::expansion(mult = c(0.02, 0.02))) +
  scale_y_discrete(labels = parse_token_labels, expand = ggplot2::expansion(mult = c(0.02, 0.02))) +
  scale_fill_gradientn(
    colours = quantile_cols,
    values = quantiles_rescaled,
    breaks = quantiles,
    name = "Attention",
    guide = "none"
  ) +
  ggnewscale::new_scale_fill() +
  geom_tile(
    data = df_c,
    aes(x = to_token, y = from_token, fill = attention),
    colour = wong["black"],
    linewidth = 0.06
  ) +
  scale_fill_gradient2(
    low = wong["orange"],
    mid = "white",
    high = wong["sky_blue"],
    midpoint = 0,
    limits = c(-delta_lim, delta_lim),
    name = "Delta",
    guide = guide_colorbar(
      direction = "vertical",
      barheight = grid::unit(delta_legend_barheight, "mm"),
      barwidth = grid::unit(delta_legend_barwidth, "mm"),
      title.position = "top",
      title.hjust = 0.5
    )
  ) +
  coord_fixed(expand = FALSE) +
  labs(x = "To token", y = "From token", title = NULL, tag = "(b)") +
  theme_pub() +
  theme(
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = bc_legend_margin,
    axis.title.x = element_text(margin = bc_axis_title_x_margin),
    plot.margin = margin(1, 1, 0, 1, "pt")
  )

# patchwork: plot_layout(widths=...) on (a|BC)/PCA often collapses to ~50:50. Using wrap_plots()
# design allocates one grid cell per character — 3×"A" vs 7×"B" enforces top_row_widths literally.
pw_top_cols <- sum(top_row_widths)
pw_design <- paste0(
  paste0(strrep("A", top_row_widths[[1L]]), strrep("B", top_row_widths[[2L]])),
  "\n",
  strrep("C", pw_top_cols)
)
fig_all <- patchwork::wrap_plots(A = panel_a, B = fig_bc, C = p_pca, design = pw_design) +
  patchwork::plot_layout(heights = grid::unit(rel_heights_bcd, "null"), guides = "keep")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggplot2::ggsave(
  fig_out,
  fig_all,
  device = grDevices::pdf,
  width = fig_width_mm,
  height = fig_height_mm,
  units = "mm",
  limitsize = FALSE,
  compress = TRUE,
  family = base_font_family
)
message("Wrote: ", fig_out)
