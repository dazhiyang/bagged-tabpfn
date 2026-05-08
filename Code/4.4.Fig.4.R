#################################################################################
# 4.4.Fig.4.R
# Combined diagnostics from Code/3.4.attention.py CSV outputs:
# (a) Feature attention (Full / B10), (b) Delta attention (B10 - Full),
# (c) Feature-token PCA (PC1 vs PC2; points colored by input feature).
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
invisible(utils::globalVariables(c("context", "stage", "token", "y_bin", "pc1", "pc2", "layer", "from_token", "to_token", "attention")))

#################################################################################
# Parameter block (edit here only)
#################################################################################
project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")

pca_file <- file.path(diag_dir, "feature_token_pca_layers_long.csv")
file_attn <- file.path(diag_dir, "attention_feature_layers_long.csv")
fig_out <- file.path(fig_dir, "Fig4.pdf")
n_label_bins <- 7L

base_font_family <- "Times"
text_size_pt <- 8
line_width_axis <- 0.25
width_mm <- 160
height_mm <- 170
# Panel (c) PCA: upper limit on PC1; scattermore point size (larger = bigger pixels).
pca_pc1_xmax <- 25
pca_pointsize <- 5

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
token_label_expr <- c(
  xP = "italic(x)[P]",
  SZA = "italic(Z)",
  lcc = "'lcc'",
  mcc = "'mcc'",
  tcsw = "'tcsw'",
  tcwv = "'tcwv'",
  label = "italic(y)[H]"
)

#################################################################################
# Read and format
#################################################################################
pca <- read.csv(pca_file, stringsAsFactors = FALSE)
if (nrow(pca) == 0) {
  stop("feature_token_pca_layers_long.csv is empty. Run Code/3.4.attention.py first.")
}
attn <- read.csv(file_attn, stringsAsFactors = FALSE)
if (nrow(attn) == 0) {
  stop("attention_feature_layers_long.csv is empty. Run Code/3.4.attention.py first.")
}

feature_token_levels <- c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv", "label")
pca <- pca %>%
  mutate(
    token = trimws(token),
    y_bin = trimws(y_bin),
    context = factor(context, levels = c("full", "b10"), labels = c("Full", "B10")),
    stage = factor(stage, levels = c("Input", "L3", "L6", "L9", "L12")),
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

layer_levels <- c("L3", "L6", "L9", "L12")

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
# Plot — panel (a): attention heatmaps (Full / B10)
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
  scale_x_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  scale_y_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  labs(x = "To token", y = "From token", title = NULL) +
  theme_pub() +
  theme(legend.position = "none")

#################################################################################
# Plot — panel (b): delta attention (B10 - Full)
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
  scale_x_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  scale_y_discrete(labels = function(x) parse(text = token_label_expr[x])) +
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
# Plot — panel (c): token PCA (PC1 vs PC2), colour = input feature (token)
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
    labels = function(x) parse(text = token_label_expr[x]),
    name = NULL
  ) +
  coord_cartesian(xlim = c(NA, pca_pc1_xmax)) +
  labs(x = "PC1", y = "PC2") +
  theme_pub() +
  theme(
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 0, "pt")
  ) +
  guides(colour = guide_legend(override.aes = list(size = 3.2, alpha = 1), ncol = 1))

g_a <- ggplotGrob(p_base)
g_b <- ggplotGrob(p_delta)
g_c <- ggplotGrob(p_pca)

panel_a <- gridExtra::arrangeGrob(
  grobs = list(
    grid::textGrob("(a)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt)),
    g_a
  ),
  ncol = 1,
  heights = grid::unit.c(
    grid::unit(2.5, "mm"),
    grid::unit(1, "null")
  )
)

panel_b <- gridExtra::arrangeGrob(
  grobs = list(
    grid::textGrob("(b)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt)),
    g_b
  ),
  ncol = 1,
  heights = grid::unit.c(
    grid::unit(2.5, "mm"),
    grid::unit(1, "null")
  )
)

panel_c <- gridExtra::arrangeGrob(
  grobs = list(
    grid::textGrob("(c)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt)),
    g_c
  ),
  ncol = 1,
  heights = grid::unit.c(
    grid::unit(2.5, "mm"),
    grid::unit(1, "null")
  )
)

# Mosaic: aab / aab / ccc / ccc — (a) 2×2 left, (b) right strip, (c) full width bottom.
fig <- gridExtra::arrangeGrob(
  grobs = list(panel_a, panel_b, panel_c),
  layout_matrix = rbind(
    c(1L, 1L, 2L),
    c(1L, 1L, 2L),
    c(1L, 1L, 2L),
    c(3L, 3L, 3L),
    c(3L, 3L, 3L)
  ),
  heights = grid::unit(c(1, 1, 1, 1, 1), "null"),
  widths = grid::unit(c(1, 1, 1), "null")
)

#################################################################################
# Save figure (PDF)
#################################################################################
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(fig_out, fig, device = grDevices::pdf, width = width_mm, height = height_mm, units = "mm", limitsize = FALSE, family = base_font_family)
message("Wrote: ", fig_out)
