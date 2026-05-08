#################################################################################
# 4.4.Fig.4.R
# Quick attention-matrix plots from Code/3.4.attention.py outputs.
#################################################################################

rm(list = ls(all = TRUE))

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})
invisible(utils::globalVariables(c("context", "layer", "from_token", "to_token", "attention")))

#################################################################################
# Parameter block (edit here only)
#################################################################################
project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")

file_attn <- file.path(diag_dir, "attention_feature_layers_long.csv")
fig_out <- file.path(fig_dir, "Fig4.pdf")

base_font_family <- "Times"
text_size_pt <- 8
line_width_data <- 0.35
line_width_grid <- 0.15
line_width_axis <- 0.25
width_mm <- 180
height_mm <- 95

wong <- c(
  orange = "#E69F00",
  sky_blue = "#56B4E9",
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
attn <- read.csv(file_attn, stringsAsFactors = FALSE)
if (nrow(attn) == 0) {
  stop("attention_feature_layers_long.csv is empty. Run Code/3.4.attention.py first.")
}

if (all(c("from_group", "to_group") %in% names(attn))) {
  attn <- attn %>%
    dplyr::rename(from_token = from_group, to_token = to_group)
}
if (all(c("from_feature", "to_feature") %in% names(attn))) {
  attn <- attn %>%
    dplyr::rename(from_token = from_feature, to_token = to_feature)
}

layer_levels <- attn %>%
  distinct(layer) %>%
  mutate(layer_num = as.integer(gsub("^L", "", layer))) %>%
  arrange(layer_num) %>%
  pull(layer)

token_order <- unique(c(
  attn %>% distinct(from_token) %>% pull(from_token),
  attn %>% distinct(to_token) %>% pull(to_token)
))

# Keep label as the final token for display order.
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
# Plot
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
      plot.margin = margin(1, 1, 1, 1, "pt"),
      axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      strip.background = element_rect(fill = "grey95")
    )
}

p_base <- ggplot(attn_fb %>% mutate(context = factor(context, levels = c("full", "b10"), labels = c("Full", "B10"))), aes(x = to_token, y = from_token, fill = attention)) +
  geom_tile(colour = wong["black"], linewidth = line_width_grid * 0.4) +
  coord_fixed() +
  facet_grid(context ~ layer) +
  scale_fill_gradientn(
    colours = quantile_cols,
    values = quantiles_rescaled,
    breaks = quantiles,
    name = "Attention"
  ) +
  scale_x_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  scale_y_discrete(labels = function(x) parse(text = token_label_expr[x])) +
  labs(x = "To token", y = "From token") +
  theme_pub() +
  theme(legend.position = "none")

p_delta <- ggplot(attn_plot %>% filter(context == "B10 - Full"), aes(x = to_token, y = from_token, fill = attention)) +
  geom_tile(colour = wong["black"], linewidth = line_width_grid * 0.4) +
  coord_fixed() +
  facet_grid(. ~ layer) +
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
  labs(x = "To token", y = "From token") +
  theme_pub() +
  theme(
    legend.position = "bottom",
    legend.box.spacing = grid::unit(0, "pt"),
    legend.margin = margin(0, 0, 0, 0, "pt")
  ) +
  guides(fill = guide_colorbar(direction = "horizontal", barheight = grid::unit(2.2, "mm"), barwidth = grid::unit(36, "mm"), title.position = "left"))

g1 <- ggplotGrob(p_base)
g2 <- ggplotGrob(p_delta)
shared_widths <- grid::unit.pmax(g1$widths, g2$widths)
g1$widths <- shared_widths
g2$widths <- shared_widths

fig <- gridExtra::arrangeGrob(
  grobs = list(
    grid::textGrob("(a)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt)),
    g1,
    grid::textGrob("(b)", x = 0, hjust = 0, gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt)),
    g2
  ),
  ncol = 1,
  heights = grid::unit.c(
    grid::unit(2.5, "mm"),
    grid::unit(1, "null"),
    grid::unit(2.5, "mm"),
    grid::unit(0.72, "null")
  )
)

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggsave(fig_out, fig, device = grDevices::pdf, width = width_mm, height = height_mm, units = "mm", limitsize = FALSE, family = base_font_family)
message("Wrote: ", fig_out)
