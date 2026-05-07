#################################################################################
# This code is written by Dazhi Yang (a) and Yun Chen (b)
# (a) Department of Electrical Engineering and Automation, Harbin Institute of Technology
# (b) Public Meteorological Service Center, China Meteorological Administration
# emails: yangdazhi.nus@gmail.com, chenyunpku@163.com 
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
train_year <- 2024L
test_year <- 2025L

out_dir <- file.path(dir0, "Data", "Output")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
out_file <- file.path(out_dir, "MLR.txt")

# Extra predictors beyond xP/xS (explicit list; keep in sync with Code/2.3.XGBoost.py, 2.4.TabPFN.py).
cov_use <- c("SZA", "lcc", "mcc", "tcsw", "tcwv")

#################################################################################
# Load data (irradiance); convert to clear-sky index
#################################################################################
fp <- file.path(dir0, "Data", "arranged15min.txt")
data <- read.table(fp, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
data <- tibble(data) %>%
  mutate(Time = lubridate::ymd_hms(Time, tz = "UTC")) %>%
  mutate(across(c(yH, yL, xP, xS), ~ .x / Ghc)) %>%
  # μ₀ = cos(solar zenith angle); column name stays "SZA". KCDE (2.2) keeps zenith in ° for von Mises kernel.
  mutate(SZA = cos(SZA * pi / 180))

data.tr <- data %>% filter(year(Time) == train_year)
data.te <- data %>% filter(year(Time) == test_year)

#################################################################################
# multivariate linear model for correction
#################################################################################
blocks <- vector("list", length(obs) * length(ret))
for (i in seq_along(obs)) {
  for (j in seq_along(ret)) {
    # multivariate linear regression
    formula <- as.formula(paste(obs[i], "~", paste(c(ret[j], cov_use), collapse = " + ")))
    model <- lm(formula, data = data.tr) # fit the model
    x <- predict(model, newdata = data.te) * data.te$Ghc
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
