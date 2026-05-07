#################################################################################
# This code is written by Dazhi Yang (a) and Yun Chen (b)
# (a) Department of Electrical Engineering and Automation, Harbin Institute of Technology
# (b) Public Meteorological Service Center, China Meteorological Administration
# emails: yangdazhi.nus@gmail.com, chenyunpku@163.com
#################################################################################
# Raw retrieval vs observation (no bias correction).
# Output matches 2.1.MLR.R / 2.2.KCDE.R: test year only, columns Time, combo, y, x (W·m⁻²).
#################################################################################

rm(list = ls(all = TRUE))
libs <- c("dplyr", "lubridate")
invisible(lapply(libs, library, character.only = TRUE))

#################################################################################
# Inputs
#################################################################################
dir0 <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
ret <- c("xP", "xS")
obs <- c("yH", "yL")
train_year <- 2024L # same chronology as 2.1 / 2.2; raw output uses test year only
test_year <- 2025L

out_dir <- file.path(dir0, "Data", "Output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(out_dir, "raw.txt")

#################################################################################
# Load data (irradiance in file; work in clear-sky index then scale back, as in 2.1)
#################################################################################
fp <- file.path(dir0, "Data", "arranged15min.txt")
data <- read.table(fp, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
data <- tibble(data) %>%
  mutate(Time = lubridate::ymd_hms(Time, tz = "UTC")) %>%
  mutate(across(c(yH, yL, xP, xS), ~ .x / Ghc))

data.te <- data %>% filter(year(Time) == test_year)

#################################################################################
# Raw pairs: x = retrieval, y = observation (same evaluation period as MLR/KCDE)
#################################################################################
blocks <- vector("list", length(obs) * length(ret))
for (i in seq_along(obs)) {
  for (j in seq_along(ret)) {
    x <- data.te[[ret[j]]] * data.te$Ghc
    y <- data.te[[obs[i]]] * data.te$Ghc
    blocks[[(i - 1L) * length(ret) + j]] <- tibble(
      Time = format(as.POSIXct(data.te$Time, tz = "UTC"), "%Y-%m-%d %H:%M:%S", tz = "UTC"),
      combo = paste0(obs[i], ret[j]),
      y = round(y, 2),
      x = round(x, 2)
    )
  }
}

out <- bind_rows(blocks)

write.table(
  out,
  file = out_file,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
