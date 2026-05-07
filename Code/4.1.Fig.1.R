#################################################################################
# 4.1.Fig.1.R — Composite manuscript figure (Site_Adaptation)
#   Top    : yHxP scatter + marginal histograms via ggExtra::ggMarginal
#            (dummy geom_point layer required for ggMarginal + geom_scattermore).
#   Bottom : SHAP beeswarm from Code/1.2.XAI.py (Data/xai_yHxP.txt)
#
# Stack scatter + SHAP with gridExtra::grid.arrange(grobs = multiplot, layout_matrix = lay).
#
# Follows .cursor/rules/SKILL.mdc (scientific-publication-plotter): vector PDF,
# single parameter block, one Times text size, no plot titles, Wong discrete,
# viridis-family continuous color with percentile (equal-rank) encoding.
#################################################################################

rm(list = ls(all = TRUE))

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(grid)
  library(gridExtra)
  library(ggExtra)
  library(scattermore)
})

invisible(utils::globalVariables(c("density", "density_pct", "feat_u")))

#################################################################################
# --- Parameter block (edit here) ----------------------------------------------
#################################################################################

base_font_family <- "Times"
text_size_pt <- 8

# Stroke widths (ggplot2 linewidth ≈ mm context via theme; keep thin grids vs data)
line_width_grid <- 0.15
line_width_data <- 0.35
line_width_axis <- 0.25

scatter_pointsize <- 0.05 * 45 # geom_scattermore pointsize scale (matches Fig.1.R spirit)
shap_point_size <- text_size_pt * 0.22

# Panel (b): SHAP x-axis span and outer margins (top, right, bottom, left, pt).
shap_x_lim <- c(-0.28, 0.28)
shap_plot_margin_pt <- margin(27, 0, 9, 5, "pt")

# Irradiance scatter limits (W·m⁻²); square canvas → coord_fixed.
scatter_lim_wm2 <- c(0, 1100)
kde_n_grid <- 200

hist_binwidth_wm2 <- 50
hist_fill_grey <- "grey85"

# Wong palette (Bang Wong), order fixed — use only for discrete accents (≤8).
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

# Continuous colour: viridis family only (skill default).
viridis_continuous_option <- "viridis"

# Journal geometry — PDF page size (mm); custom one-row composite.
width_mm <- 160
height_mm <- 70

# ggMarginal: ratio main:marginal strips (see ?ggMarginal — larger ⇒ relatively smaller marginals).
ggmarginal_size <- 5

# grid.arrange row heights (null units): scatter+marginals vs SHAP — ratio 1 : 0.8.
scatter_row_rel <- 1
shap_row_rel <- 0.85

# Panel-letter column: needs ~≥5 mm at text_size_pt bold or glyphs clip; keep modest so labels stay near plots.
panel_tag_col_mm <- 6
panel_tag_inset_mm <- 0.75

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
data.path <- Sys.getenv("DATA_TXT", file.path(dir0, "Data", "arranged15min.txt"))
shap.path <- Sys.getenv("SHAP_LONG_TXT", file.path(dir0, "Data", "xai_yHxP.txt"))
out.fig <- Sys.getenv("OUTPUT_FIG", file.path(dir0, "tex", "Fig1.pdf"))

# Fig.1.R panel (a): high-accuracy observation yH vs NSMC retrieval xP.
obs_col <- "yH"
ret_col <- "xP"

#################################################################################
# Theme & helpers
#################################################################################

theme_pub <- function() {
  theme_bw(base_size = text_size_pt, base_family = base_font_family) +
    theme(
      text = element_text(family = base_font_family, size = text_size_pt),
      axis.title = element_text(size = text_size_pt),
      axis.text = element_text(size = text_size_pt),
      legend.title = element_text(size = text_size_pt),
      legend.text = element_text(size = text_size_pt),
      strip.text = element_text(size = text_size_pt),
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      plot.tag = element_text(
        face = "bold",
        family = base_font_family,
        size = text_size_pt
      ),
      plot.tag.position = c(0.02, 0.98),
      plot.margin = margin(1, 2, 1, 2, "pt"),
      legend.position = "none",
      panel.grid.major = element_line(
        colour = grDevices::adjustcolor("black", alpha.f = 0.22),
        linewidth = line_width_grid
      ),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(colour = "black", linewidth = line_width_axis),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background = element_rect(fill = "transparent", colour = NA)
    )
}

theme_pub_legend_bottom <- function() {
  theme_pub() %+replace%
    theme(
      legend.position = "bottom",
      legend.key.height = unit(text_size_pt * 0.75, "pt"),
      legend.key.width = unit(text_size_pt * 3.5, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.margin = margin(-4, 0, -2, 0)
    )
}

get_density <- function(x, y, ...) {
  dens <- MASS::kde2d(x, y, ...)
  ix <- findInterval(x, dens$x)
  iy <- findInterval(y, dens$y)
  ii <- cbind(ix, iy)
  dens$z[ii]
}

#################################################################################
# Data
#################################################################################

stopifnot(file.exists(data.path), file.exists(shap.path))

data <- read.table(data.path, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
data <- tibble::as_tibble(data)

shap.raw <- read.delim(shap.path, sep = "\t", stringsAsFactors = FALSE, comment.char = "")
shap.df <- tibble::as_tibble(shap.raw) %>%
  filter(is.finite(shap_value), is.finite(feature_value_scaled))

combo_expect <- paste0(obs_col, ret_col)
if ("combo" %in% names(shap.df)) {
  shap.df <- shap.df %>% filter(as.character(.data$combo) == combo_expect)
}

if (nrow(shap.df) == 0L) stop("No finite rows in SHAP file (after combo filter): ", shap.path)

# Within-feature percentile rank → viridis_c avoids one hue dominating skewed covariates (SKILL).
shap.df <- shap.df %>%
  group_by(.data$feature) %>%
  mutate(feat_u = dplyr::percent_rank(.data$feature_value_scaled)) %>%
  ungroup()

rank.tbl <- shap.df %>%
  group_by(.data$feature) %>%
  summarise(mabs = mean(abs(.data$shap_value), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(.data$mabs))

shap.df$feature <- factor(shap.df$feature, levels = rev(rank.tbl$feature))

# Panel (b): tick label for covariate xP → plotmath italic(x)[P] (pass parse(text=…) via scale, not a labels fn).
feat_ord <- levels(shap.df$feature)
feat_label_txt <- dplyr::case_when(
  feat_ord == "xP" ~ "italic(x)[P]",
  feat_ord == "SZA" ~ "'sza'",
  feat_ord == "W" ~ "'ws'",
  TRUE ~ paste0("`", feat_ord, "`")
)
shap.df$feature_disp <- factor(
  dplyr::case_when(
    as.character(shap.df$feature) == "xP" ~ "italic(x)[P]",
    as.character(shap.df$feature) == "SZA" ~ "'sza'",
    as.character(shap.df$feature) == "W" ~ "'ws'",
    TRUE ~ paste0("`", as.character(shap.df$feature), "`")
  ),
  levels = feat_label_txt
)

#################################################################################
# Scatter: yHxP — ggMarginal histogram strips (needs a geom_point layer for ggMarginal)
#################################################################################

stopifnot(obs_col %in% names(data), ret_col %in% names(data))

x <- dplyr::pull(data, ret_col)
y <- dplyr::pull(data, obs_col)
data.tmp <- tibble(retrieval = x, observation = y) %>%
  filter(is.finite(retrieval), is.finite(observation)) %>%
  mutate(
    density = get_density(retrieval, observation, n = kde_n_grid),
    density_pct = dplyr::percent_rank(density)
  )

hist_outline <- grDevices::adjustcolor(unname(wong["black"]), alpha.f = 0.35)

marg_hist_aes <- list(
  binwidth = hist_binwidth_wm2,
  fill = hist_fill_grey,
  colour = hist_outline,
  linewidth = line_width_grid
)

pt <- ggplot(data.tmp, aes(x = retrieval, y = observation)) +
  geom_point(alpha = 0, stroke = 0, size = 0) +
  geom_scattermore(aes(colour = density_pct), pointsize = scatter_pointsize) +
  ggplot2::scale_colour_viridis_c(
    option = viridis_continuous_option,
    name = "Density\n(percentile)",
    na.value = NA,
    guide = "none"
  ) +
  geom_abline(
    linewidth = line_width_data,
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    colour = wong["black"]
  ) +
  geom_density_2d(linewidth = line_width_grid, colour = wong["orange"]) +
  scale_x_continuous(limits = scatter_lim_wm2, expand = c(0, 0)) +
  scale_y_continuous(limits = scatter_lim_wm2, expand = c(0, 0)) +
  coord_fixed(ratio = 1, clip = "off") +
  xlab(as.expression(bquote(
    italic(.(substr(ret_col, 1, 1)))[.(substr(ret_col, 2, 2))] ~ " [W m"^-2 * "]"
  ))) +
  ylab(as.expression(bquote(
    italic(.(substr(obs_col, 1, 1)))[.(substr(obs_col, 2, 2))] ~ " [W m"^-2 * "]"
  ))) +
  theme_pub() +
  theme(plot.margin = margin(4, 4, 4, 4, "pt"))

p.top <- ggExtra::ggMarginal(
  pt,
  type = "histogram",
  margins = "both",
  size = ggmarginal_size,
  fill = hist_fill_grey,
  xparams = marg_hist_aes,
  yparams = marg_hist_aes
)

#################################################################################
# SHAP (continuous colour → viridis only)
#################################################################################

p.shap <- ggplot(shap.df, aes(x = shap_value, y = feature_disp, colour = feat_u)) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = line_width_data,
    colour = grDevices::adjustcolor(unname(wong["black"]), alpha.f = 0.45)
  ) +
  geom_point(
    alpha = 0.72,
    size = shap_point_size,
    position = position_jitter(height = 0.18, width = 0)
  ) +
  scale_y_discrete(breaks = feat_label_txt, labels = parse(text = feat_label_txt)) +
  scale_x_continuous(limits = shap_x_lim, expand = c(0, 0)) +
  ggplot2::scale_colour_viridis_c(
    option = viridis_continuous_option,
    limits = c(0, 1),
    name = NULL,
    guide = guide_colourbar(
      direction = "vertical",
      barwidth = unit(text_size_pt * 0.55, "pt"),
      barheight = unit(text_size_pt * 12, "pt"),
      title.position = "right",
      title.hjust = 0.5
    )
  ) +
  labs(x = "SHAP value", y = NULL) +
  theme_pub() +
  theme(
    panel.grid.major.y = element_blank(),
    legend.position = "right",
    plot.margin = shap_plot_margin_pt
  )

#################################################################################
# Compose — gridExtra::grid.arrange(grobs = multiplot, layout_matrix = lay)
# Column 1: shared panel letters (a)/(b); column 2: plots (ggplotGrobs).
# draw = FALSE avoids plotting to the screen/RStudio device (default TRUE opens a viewer).
#################################################################################

panel_tag_grob <- function(label) {
  inset <- grid::unit(panel_tag_inset_mm, "mm")
  grid::textGrob(
    label,
    x = grid::unit(1, "npc") - inset,
    y = grid::unit(1, "npc") - inset,
    hjust = 1,
    vjust = 1,
    gp = grid::gpar(fontfamily = base_font_family, fontsize = text_size_pt, fontface = "bold")
  )
}

multiplot <- list(
  panel_tag_grob("(a)"),
  p.top,
  panel_tag_grob("(b)"),
  ggplot2::ggplotGrob(p.shap)
)
lay <- rbind(c(1L, 2L, 3L, 4L))

fig <- gridExtra::grid.arrange(
  grobs = multiplot,
  layout_matrix = lay,
  widths = grid::unit(c(panel_tag_col_mm, 1, panel_tag_col_mm, 1), c("mm", "null", "mm", "null")),
  heights = grid::unit(1, "null"),
  padding = grid::unit(1.5, "mm"),
  draw = FALSE
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
