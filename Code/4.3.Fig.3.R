#################################################################################
# 4.3.Fig.TabPFN-Diagnostics.R
# Hard-coded yHxP diagnostics plots from Code/3.3.TabPFN-Diagnostics.py outputs.
#################################################################################

rm(list = ls(all = TRUE))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(scattermore)
  library(gridExtra)
  library(grid)
})
invisible(utils::globalVariables(c("from_token", "to_token", "attention", "attention_quantile", "layer", "context", "context_lab", "pc1", "pc2", "y_bin", "space", "fisher_ratio", "dim", "label")))

#################################################################################
# Parameter block (edit here only)
#################################################################################
base_font_family <- "Times"
text_size_pt <- 8
line_width_data <- 0.35
line_width_grid <- 0.15
line_width_axis <- 0.25

width_mm <- 180
height_mm <- 110

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
continuous_color_note <- "Panel (a): quantile coloring in [0,1]. Panel (b): raw delta with Wong orange-white-skyblue diverging scale."
token_label_expr <- c(
  xP = "italic(x)[P]",
  SZA = "'sza'",
  lcc = "'lcc'",
  mcc = "'mcc'",
  tcsw = "'tcsw'",
  tcwv = "'tcwv'",
  label = "italic(y)[H]"
)

project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")

file_attn_layer_long <- file.path(diag_dir, "attention_layers_long.csv")
fig_attn_layer_combined <- file.path(fig_dir, "Fig_Attn_LayerFacet_Combined.pdf")
file_metrics <- file.path(diag_dir, "metrics_all_yHxP.csv")
file_pred_compare <- file.path(diag_dir, "pred_context_compare_all_yHxP.csv")
file_embed_pca <- file.path(diag_dir, "embedding_pca_per_test.csv")
file_embed_space <- file.path(diag_dir, "embedding_space_metrics.csv")
fig_story <- file.path(fig_dir, "Fig3.pdf")

#################################################################################
# Shared helpers
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
        linewidth = line_width_grid
      ),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = line_width_axis),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      plot.margin = margin(1, 1, 1, 1, "pt")
    )
}

read_attn_long <- function(file_path) {
  attn <- read.csv(file_path, stringsAsFactors = FALSE)
  attn %>%
    mutate(
      layer = factor(layer, levels = c("L3", "L6", "L9", "L12", "L15")), # nolint: object_usage_linter
      context = factor(context, levels = c("full", "cap2000", "delta_cap2000_minus_full")), # nolint: object_usage_linter
      from_token = factor(from_token, levels = c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv", "label")), # nolint: object_usage_linter
      to_token = factor(to_token, levels = c("xP", "SZA", "lcc", "mcc", "tcsw", "tcwv", "label")) # nolint: object_usage_linter
    )
}

plot_attn_heatmap <- function(df_long, legend_title) {
  ggplot(df_long, aes(x = to_token, y = from_token, fill = attention_quantile)) + # nolint: object_usage_linter
    geom_tile(colour = wong["black"], linewidth = line_width_grid * 0.4) +
    scale_fill_viridis_c(
      option = viridis_continuous_option,
      limits = c(0, 1),
      name = legend_title
    ) +
    coord_fixed() +
    scale_x_discrete(labels = function(x) parse(text = token_label_expr[x])) +
    scale_y_discrete(labels = function(x) parse(text = token_label_expr[x])) +
    labs(x = "To token", y = "From token") +
    theme_pub() +
    theme(
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      legend.position = "right",
      legend.box.spacing = grid::unit(0, "pt"),
      legend.margin = margin(0, 0, 0, 0, "pt")
    ) +
    guides(fill = guide_colorbar(barheight = grid::unit(24, "mm"), barwidth = grid::unit(2.4, "mm"), title.position = "top"))
}

#################################################################################
# Combined layer-specific attention figure
#################################################################################
attn <- read_attn_long(file_attn_layer_long)

attn_top <- attn %>%
  filter(context %in% c("full", "cap2000")) %>%
  group_by(context) %>%
  mutate(attention_quantile = (rank(attention, ties.method = "average") - 0.5) / n()) %>% # nolint: object_usage_linter
  ungroup() %>%
  mutate(context_lab = factor(context, levels = c("full", "cap2000"), labels = c("Full", "Small")))

attn_delta <- attn %>%
  filter(context == "delta_cap2000_minus_full")
delta_lim <- max(abs(attn_delta$attention), na.rm = TRUE)

p_top <- plot_attn_heatmap(attn_top, "Attention quantile") +
  facet_grid(context_lab ~ layer) +
  labs(x = "To token", y = "From token") +
  theme(
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, -4, 0, -40, "pt")
  )

p_bottom <- ggplot(attn_delta, aes(x = to_token, y = from_token, fill = attention)) + # nolint: object_usage_linter
  geom_tile(colour = wong["black"], linewidth = line_width_grid * 0.4) +
  scale_fill_gradient2(
    low = unname(wong["orange"]),
    mid = "white",
    high = unname(wong["sky_blue"]),
    midpoint = 0,
    limits = c(-delta_lim, delta_lim),
    oob = scales::squish,
    name = "Delta attention"
  ) +
  coord_fixed() +
  scale_x_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  scale_y_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  facet_grid(. ~ layer) +
  labs(x = "To token", y = "From token") +
  theme_pub() +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    legend.position = "right",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 10, "pt")
  ) +
  guides(fill = guide_colorbar(barheight = grid::unit(24, "mm"), barwidth = grid::unit(2.4, "mm"), title.position = "top"))

g_top <- ggplotGrob(p_top)
g_bottom <- ggplotGrob(p_bottom)
shared_widths <- grid::unit.pmax(g_top$widths, g_bottom$widths)
g_top$widths <- shared_widths
g_bottom$widths <- shared_widths

tag_a <- grid::textGrob("(a)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt, fontface = "plain"))
tag_b <- grid::textGrob("(b)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt, fontface = "plain"))

fig_combined <- gridExtra::arrangeGrob(
  grobs = list(tag_a, g_top, tag_b, g_bottom),
  ncol = 1,
  heights = grid::unit.c(
    grid::unit(2.5, "mm"),
    grid::unit(1, "null"),
    grid::unit(2.5, "mm"),
    grid::unit(0.6, "null")
  )
)

#################################################################################
# Additional panels (c)-(d): mechanism-level embedding view
#################################################################################
embed_pca <- read.csv(file_embed_pca, stringsAsFactors = FALSE) %>%
  mutate(
    context_lab = factor(context, levels = c("full", "cap2000"), labels = c("Full", "Small")),
    y_bin = factor(y_bin, levels = c("low", "mid", "high"), labels = c("Low y", "Mid y", "High y"))
  )

p_c <- ggplot(embed_pca, aes(x = pc1, y = pc2, colour = y_bin)) +
  geom_scattermore(pointsize = 3, alpha = 0.7) +
  facet_grid(. ~ context_lab) +
  scale_colour_manual(
    values = c("Low y" = unname(wong["orange"]), "Mid y" = unname(wong["sky_blue"]), "High y" = unname(wong["blue_green"])),
    name = NULL
  ) +
  labs(
    x = "PC1 of embedding space",
    y = "PC2 of embedding space"
  ) +
  theme_pub() +
  theme(legend.position = "bottom")

embed_space <- read.csv(file_embed_space, stringsAsFactors = FALSE) %>%
  mutate(
    label = dplyr::case_when(
      space == "raw_inputs" ~ "Raw input (6D)",
      context == "full" ~ "Embedding Full (512D)",
      context == "cap2000" ~ "Embedding Small (512D)",
      TRUE ~ "Other"
    ),
    label = factor(label, levels = c("Raw input (6D)", "Embedding Full (512D)", "Embedding Small (512D)"))
  )

p_d <- ggplot(embed_space, aes(x = label, y = fisher_ratio, fill = label)) +
  geom_col(width = 0.64, colour = wong["black"], linewidth = line_width_grid * 0.6) +
  geom_text(aes(label = sprintf("%.3f", fisher_ratio)), vjust = -0.45, family = base_font_family, size = text_size_pt * 0.24) +
  scale_fill_manual(values = unname(c(wong["yellow"], wong["orange"], wong["sky_blue"])), guide = "none") +
  labs(
    x = NULL,
    y = "Class separability (Fisher ratio)"
  ) +
  theme_pub() +
  theme(axis.text.x = element_text(angle = 10, hjust = 1))

panel_c <- gridExtra::arrangeGrob(
  grobs = list(
    grid::textGrob("(c)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt, fontface = "plain")),
    ggplotGrob(p_c)
  ),
  ncol = 1,
  heights = grid::unit.c(grid::unit(2.5, "mm"), grid::unit(1, "null"))
)
panel_d <- gridExtra::arrangeGrob(
  grobs = list(
    grid::textGrob("(d)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt, fontface = "plain")),
    ggplotGrob(p_d)
  ),
  ncol = 1,
  heights = grid::unit.c(grid::unit(2.5, "mm"), grid::unit(1, "null"))
)
row_cd <- gridExtra::arrangeGrob(
  grobs = list(panel_c, panel_d),
  ncol = 2,
  widths = grid::unit(c(1, 1), "null")
)

fig_story_all <- gridExtra::arrangeGrob(
  grobs = list(fig_combined, row_cd),
  ncol = 1,
  heights = grid::unit(c(1.55, 1), "null")
)

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(fig_attn_layer_combined, fig_combined, device = grDevices::pdf, width = width_mm, height = height_mm, units = "mm", limitsize = FALSE, family = base_font_family)
ggsave(fig_story, fig_story_all, device = grDevices::pdf, width = width_mm, height = 175, units = "mm", limitsize = FALSE, family = base_font_family)

message("Wrote: ", fig_attn_layer_combined)
message("Wrote: ", fig_story)
message("Color note: ", continuous_color_note)
