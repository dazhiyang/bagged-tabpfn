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
# 4.4.Fig.S1.R — SI attention heatmaps for TabPFN-B members B1–B10
# (no Full / no delta). Facet grid: rows = members, columns = layers L3/L6/L9/L12.
# Input: Data/Output/Diag/attention_feature_layers_members_long.csv (Code/3.6).
# Output: tex/FigS1.pdf
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

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(scales)
})

#################################################################################
# Parameter block
#################################################################################

base_font_family <- "Times"
text_size_pt <- 8
line_width_grid <- 0.06
line_width_axis <- 0.25

# Single-column SI; height sized for 10 member rows (≈ Fig. 3(b) row pitch × 10).
width_mm <- 85
height_mm <- 160

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

# Continuous attention fill: viridis + equal-count quantile breaks (finite values).
viridis_continuous_option <- "viridis"
n_quantile_classes <- 10L

layer_levels <- c("L3", "L6", "L9", "L12")
member_order <- paste0("b", 1:10)

token_label_expr <- c(
  xP = "italic(x)[P]",
  SZA = "italic(Z)",
  lcc = "italic(f)[L]",
  mcc = "italic(f)[M]",
  tcsw = "italic(w)[sn]",
  tcwv = "italic(w)",
  label = "italic(y)[H]"
)

#################################################################################
# Paths
#################################################################################

project_path <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
diag_dir <- file.path(project_path, "Data", "Output", "Diag")
fig_dir <- file.path(project_path, "tex")
file_attn <- Sys.getenv(
  "ATTN_MEMBERS_CSV",
  file.path(diag_dir, "attention_feature_layers_members_long.csv")
)
fig_out <- Sys.getenv("OUTPUT_FIG", file.path(fig_dir, "FigS1.pdf"))

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

stopifnot(file.exists(file_attn))
attn <- read.csv(file_attn, stringsAsFactors = FALSE)
if (nrow(attn) == 0L) stop("attention CSV empty: ", file_attn)

if (all(c("from_feature", "to_feature") %in% names(attn))) {
  attn <- attn %>% rename(from_token = from_feature, to_token = to_feature)
}

attn <- attn %>%
  filter(
    as.character(.data$context) %in% member_order,
    as.character(.data$layer) %in% layer_levels
  )
if (nrow(attn) == 0L) stop("No B1–B10 / L3–L12 rows in: ", file_attn)

token_order <- unique(c(
  as.character(attn$from_token),
  as.character(attn$to_token)
))
token_order <- c(setdiff(token_order, "label"), "label")

attn <- attn %>%
  mutate(
    member = factor(as.character(.data$context), levels = member_order),
    layer = factor(as.character(.data$layer), levels = layer_levels),
    from_token = factor(.data$from_token, levels = token_order),
    to_token = factor(.data$to_token, levels = token_order),
    member_lab = factor(
      toupper(as.character(.data$member)),
      levels = toupper(member_order)
    )
  )

# Shared viridis scale across all panels (quantile breaks from finite attentions).
att_vals <- attn$attention[is.finite(attn$attention)]
quantiles <- as.numeric(stats::quantile(
  att_vals,
  probs = seq(0, 1, length.out = n_quantile_classes + 1L),
  na.rm = TRUE,
  type = 8
))
quantiles <- sort(unique(quantiles))
quantiles_rescaled <- scales::rescale(
  quantiles,
  to = c(0, 1),
  from = range(att_vals, na.rm = TRUE)
)
quantile_cols <- viridisLite::viridis(
  length(quantiles),
  option = viridis_continuous_option
)

#################################################################################
# Plot
#################################################################################

p <- ggplot(attn, aes(x = to_token, y = from_token, fill = attention)) +
  facet_grid(rows = vars(member_lab), cols = vars(layer), drop = FALSE) +
  geom_tile(colour = unname(wong["black"]), linewidth = line_width_grid) +
  scale_x_discrete(
    labels = parse_token_labels,
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_y_discrete(
    labels = parse_token_labels,
    expand = expansion(mult = c(0.02, 0.02))
  ) +
  scale_fill_gradientn(
    colours = quantile_cols,
    values = quantiles_rescaled,
    breaks = quantiles,
    name = "Attention",
    guide = "none"
  ) +
  coord_fixed(expand = FALSE) +
  labs(x = "To token", y = "From token") +
  theme_pub() +
  theme(
    legend.position = "none",
    # Same as ggplot2 default / Fig. 3(b) right-strip orientation.
    strip.text.y.right = element_text(
      angle = -90,
      hjust = 0.5,
      vjust = 0.5,
      size = text_size_pt
    ),
    strip.placement = "outside",
    axis.text.x = element_text(angle = 0, size = text_size_pt),
    axis.text.y = element_text(size = text_size_pt),
    plot.margin = margin(1, 1, 1, 1, "pt")
  )

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
ggplot2::ggsave(
  filename = fig_out,
  plot = p,
  device = grDevices::pdf,
  width = width_mm,
  height = height_mm,
  units = "mm",
  limitsize = FALSE,
  compress = TRUE,
  family = base_font_family
)

message("Wrote: ", fig_out)
