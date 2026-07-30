#################################################################################
# This code is co-authored by:
# - Dazhi Yang (yangdazhi.nus@gmail.com)
#   School of Electrical Engineering and Automation,
#   Harbin Institute of Technology (HIT)
# - Yun Chen (PowerPuffYun) (chenyunpku@163.com)
#   Public Meteorological Service Center,
#   China Meteorological Administration (CMA)
#################################################################################

#################################################################################
# 4.5.Fig.S2.R — SI token PCA for remaining TabPFN-B members
# (B1–B6, B8, B9; B7 and B10 are in Fig. 3(c); no Full).
# Facet grid: rows = members, columns = stages (Input, L1, L2, L3, L6, L9, L12).
# Same Wong token colours / scattermore style as Fig. 3(c).
# Input: Data/Output/Diag/feature_token_pca_layers_members_long.csv (Code/3.7).
# Output: tex/FigS2.png
#################################################################################

rm(list = ls(all = TRUE))

# Avoid accidental Rplots.pdf under Rscript.
if (!interactive()) {
  grDevices::pdf(NULL)
  on.exit(
    {
      while (grDevices::dev.cur() > 1L) grDevices::dev.off()
    },
    add = TRUE
  )
}

if (!requireNamespace("ggh4x", quietly = TRUE)) {
  stop("Install ggh4x: install.packages(\"ggh4x\")")
}
if (!requireNamespace("scattermore", quietly = TRUE)) {
  stop("Install scattermore: install.packages(\"scattermore\")")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(ggh4x)
  library(scattermore)
})

#################################################################################
# Parameter block
#################################################################################

base_font_family <- "Times"
text_size_pt <- 10
line_width_axis <- 0.25

# Single-column SI; 8 remaining members × 7 stages.
width_mm <- 140
height_mm <- 140

pca_pointsize <- 4
pca_alpha <- 0.3
pca_pixels <- c(400L, 400L)

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

# Continuous colour unused here (discrete Wong tokens); keep for publication block.
viridis_continuous_option <- "viridis"
# Equal-count quantile breaks for continuous colour when used (finite values only).

stage_levels <- c("Input", "L1", "L2", "L3", "L6", "L9", "L12")
# Members not already shown in Fig. 3(c).
member_order <- paste0("b", c(1:6, 8, 9))

token_label_expr <- c(
  xP = "italic(x)[P]",
  SZA = "italic(Z)",
  lcc = "italic(f)[L]",
  mcc = "italic(f)[M]",
  tcsw = "italic(w)[sn]",
  tcwv = "italic(w)"
)

feat_cols <- c(
  xP = unname(wong["orange"]),
  SZA = unname(wong["sky_blue"]),
  lcc = unname(wong["blue_green"]),
  mcc = unname(wong["pale_violet"]),
  tcsw = unname(wong["vermillion"]),
  tcwv = unname(wong["yellow"])
)

#################################################################################
# Paths
#################################################################################

project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")
pca_file <- Sys.getenv(
  "PCA_MEMBERS_CSV",
  file.path(diag_dir, "feature_token_pca_layers_members_long.csv")
)
fig_out <- Sys.getenv("OUTPUT_FIG", file.path(fig_dir, "FigS2.png"))
# Raster export DPI (PNG); unused for vector PDF.
fig_dpi <- 300L

parse_token_labels <- function(x) {
  lab <- unname(token_label_expr[as.character(x)])
  miss <- is.na(lab)
  if (any(miss)) lab[miss] <- paste0("`", as.character(x)[miss], "`")
  parse(text = lab)
}

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
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = line_width_axis),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA),
      axis.ticks = element_line(linewidth = line_width_axis),
      strip.background = element_rect(fill = "grey95"),
      strip.placement = "outside",
      legend.key = element_rect(fill = "transparent", colour = NA)
    )
}

#################################################################################
# Data
#################################################################################

stopifnot(file.exists(pca_file))
pca <- read.csv(pca_file, stringsAsFactors = FALSE)
if (nrow(pca) == 0L) stop("PCA CSV empty: ", pca_file)

pca <- pca %>%
  mutate(
    token = trimws(.data$token),
    y_bin = trimws(.data$y_bin),
    context = as.character(.data$context),
    stage = as.character(.data$stage)
  ) %>%
  filter(
    .data$context %in% member_order,
    .data$stage %in% stage_levels,
    .data$token %in% names(feat_cols)
  )
if (nrow(pca) == 0L) stop("No remaining-member / stage / attribute-token rows in: ", pca_file)

feat_tokens_plot <- intersect(names(feat_cols), unique(pca$token))
feat_cols_plot <- feat_cols[feat_tokens_plot]

pca <- pca %>%
  mutate(
    member = factor(.data$context, levels = member_order),
    member_lab = factor(
      toupper(as.character(.data$member)),
      levels = toupper(member_order)
    ),
    stage = factor(.data$stage, levels = stage_levels),
    token = factor(.data$token, levels = feat_tokens_plot)
  )

#################################################################################
# Plot
#################################################################################

p <- ggplot() +
  ggh4x::facet_grid2(
    rows = vars(member_lab),
    cols = vars(stage),
    scales = "free",
    independent = "all",
    drop = FALSE,
    labeller = labeller(member_lab = label_value, stage = label_value)
  )

for (ft in feat_tokens_plot) {
  p <- p +
    scattermore::geom_scattermore(
      data = pca %>% filter(as.character(.data$token) == ft),
      mapping = aes(x = .data$pc1, y = .data$pc2),
      color = unname(feat_cols[[ft]]),
      pointsize = pca_pointsize,
      alpha = pca_alpha,
      pixels = pca_pixels,
      inherit.aes = FALSE
    )
}

legend_df <- data.frame(
  token = factor(feat_tokens_plot, levels = feat_tokens_plot),
  pc1 = 0,
  pc2 = 0
)

p <- p +
  geom_point(
    data = legend_df,
    aes(x = .data$pc1, y = .data$pc2, colour = .data$token),
    alpha = 1,
    size = 0.01,
    inherit.aes = FALSE
  ) +
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
    axis.ticks = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    strip.text.y.right = element_text(
      angle = -90,
      hjust = 0.5,
      vjust = 0.5,
      size = text_size_pt
    ),
    strip.placement = "outside",
    plot.margin = margin(1, 1, 1, 1, "pt")
  ) +
  guides(colour = guide_legend(override.aes = list(size = 3.2, alpha = 1), ncol = 1))

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
# Close the null PDF device (opened above) so ggsave does not inherit a PDF stream.
while (grDevices::dev.cur() > 1L) grDevices::dev.off()
# Force a real PNG (ragg); device="png" previously wrote PDF bytes into *.png here.
ggplot2::ggsave(
  filename = fig_out,
  plot = p,
  device = ragg::agg_png,
  width = width_mm,
  height = height_mm,
  units = "mm",
  dpi = fig_dpi,
  background = "white",
  limitsize = FALSE
)

message("Wrote: ", fig_out)
stopifnot(identical(readBin(fig_out, "raw", 8L), as.raw(c(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a))))
message("Verified PNG signature: ", fig_out)
