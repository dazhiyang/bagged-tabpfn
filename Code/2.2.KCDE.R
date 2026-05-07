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
out_file <- file.path(out_dir, "KCDE.txt")

# Extra predictors beyond xP/xS (explicit list; keep in sync with Code/2.1.MLR.R, 2.3, 2.4).
cov_use <- c("SZA", "lcc", "mcc", "tcsw", "tcwv")

#################################################################################
# main function for KCDE
#################################################################################
KCDE <- function(y, x, x.new = x, id.G = 1, id.Z = 2, other.circ = NA)
{
  if(id.G!=1)
    stop("The first explanatory variable must be NWP GHI or clear-sky index")
  if(!is.na(id.Z) & id.Z != 2)
    stop("The second explanatory variable must be zenith angle")
  #univariate x
  if (is.null(dim(x)) | is.vector(x)) {
    bw <- KernSmooth::dpill(x, y)
    id.seq <- order(x.new)
    est <- rep(NA, length(x.new))
    est[id.seq] <- stats::ksmooth(x, y, kernel = "normal", bandwidth = bw, n.points = length(x.new), x.points = x.new)$y
  }else if (is.matrix(x)) {
    if (ncol(x) == 1) {
      bw <- KernSmooth::dpill(x, y)
      id.seq <- order(x.new)
      est <- rep(NA, length(x.new))
      est[id.seq] <- stats::ksmooth(x, y, kernel = "normal", bandwidth = bw, n.points = length(x.new), x.points = x.new)$y
    }
    else {
      #multivariate x
      bw <- bw.select(y, x, other.circ)
      est <- AMK(y, x, x.new, bw, other.circ)
    }
  }
  return(est)
}

#################################################################################
# bandwidth selection
#################################################################################
#this is inherited from the "kernplus" package
get.dpill <- function(cov, y) {
  bw <- KernSmooth::dpill(cov, y)
  if (is.nan(bw)) {
    par <- 0.06
    while (is.nan(bw)) {
      bw <- KernSmooth::dpill(cov, y, proptrun = par)
      par <- par + 0.01
    }
  }
  return(bw)
}

bw.select <- function(y, x, other.circ = NA) {
  x <- rand.add(x)
  
  bw <- array(NA, ncol(x))
  bw[1] <- get.dpill(x[,1], y) #bandwidth for G
  bw[2] <- circular::bw.nrd.circular(circular::circular(x[,2], units = "degrees"), kappa.est = "trigmoments") #bandwidth for Z
  if(ncol(x)>2)
  {
    x.remain <- as.matrix(x[,-c(1:2)])
    if(is.na(other.circ))
    {
      bw[-c(1:2)] <- sapply(1:ncol(x.remain), function(p) get.dpill(x.remain[, p], y))
    }else{
      x.Gau <- as.matrix(x[,-c(1:2, other.circ)]) #remaining variables with Gaussian kernel
      x.vM <- as.matrix(x[,other.circ]) #remaining variables with von Mises kernel
      bw[-c(1:2, other.circ)] <- sapply(1:ncol(x.Gau), function(p) get.dpill(x.Gau[, p], y))
      bw[other.circ] <- sapply(1:ncol(x.vM), function(p) circular::bw.nrd.circular(circular::circular(x.vM[,p], units = "degrees"), kappa.est = "trigmoments"))
    }
  }
  return(bw)
}

#################################################################################
# bandwidth selection
#################################################################################
#this is inherited from the "kernplus" package
get.diff <- function(X.tr, x.TS) {
  X.tr <- as.matrix(X.tr)
  n.TR <- dim(X.tr)[1]
  q.TR <- dim(X.tr)[2]
  x.TS <- matrix(x.TS, 1, q.TR)
  oneV <- matrix(1, n.TR, 1)
  diff <- X.tr - (oneV %*% x.TS)
  return(diff)
}

AMK <- function(y, x, x.new, bw, other.circ) {
  X.tr <- as.matrix(x)
  X.ts <- as.matrix(x.new)
  y.tr <- as.matrix(y)
  p <- ncol(X.tr)
  n.tr <- nrow(X.tr)
  n.ts <- nrow(X.ts)
  h <- bw
  
  cat("Estimating (%)\n")
  cat("0")
  est <- rep(NA, n.ts)
  cnt <- 0
  for (i in 1:n.ts) {
    cnt <- cnt + 1
    if (cnt/n.ts >= 0.1) {
      cat(".")
      cnt <- 0
    }
    
    diff <- get.diff(X.tr, X.ts[i, ])
    
    if (p == 2) {
      # Bivariate kernel regression
      dir <- circular::circular(diff[, 2], units = "degrees")
      
      kappa.j <- matrix(NA, n.tr, 2)
      kappa.j[, 1] <- stats::dnorm(diff[, 1]/h[1])/h[1]
      kappa.j[, 2] <- circular::dvonmises(dir, circular::circular(0), h[2])
      kappa <- kappa.j[, 1] * kappa.j[, 2]
      yhat <- sum(y.tr * kappa/sum(kappa))
    } else if (p > 2) {
      # Additive multivariate kernel method
      id.remain <- 1:p
      id.remain <- id.remain[-c(1:2)]
      dir <- circular::circular(diff[, 2], units = "degrees")
      kappa <- rep(0, n.tr)
      yhat <- rep(NA, (p - 2))
      
      kappa.G <- stats::dnorm(diff[, 1]/h[1])/h[1]
      kappa.Z <- circular::dvonmises(dir, circular::circular(0), h[2])
      for (j in 3:p) {
        if(j %in% other.circ)
        {
          kappa.j <- circular::dvonmises(dir, circular::circular(0), h[id.remain[j - 2]])
        }else{
          kappa.j <- stats::dnorm(diff[, id.remain[j - 2]]/h[id.remain[j - 2]])/h[id.remain[j - 2]]
        }
        kappa <- (kappa.G * kappa.Z * kappa.j)
        yhat[j - 2] <- sum(y.tr * kappa/sum(kappa))
      }
    }
    est[i] <- mean(yhat)
  }
  cat("100\n")
  
  return(est)
}

#################################################################################
# functions to add noise to the random variable, since some random variables give interger values
#################################################################################
#all these functions are inherited from the "kernplus" package
decimalplaces <- function(num) {
  if ((num%%1) != 0)
    nchar(strsplit(sub("0+$", "", as.character(num)), ".", fixed = TRUE)[[1]][[2]]) else return(0)
}

rand.add <- function(df) {
  n.obs <- nrow(df)
  n.sp <- 1000
  if (nrow(df) < n.sp)
    n.sp <- n.obs
  num.decimal <- sapply(1:ncol(df), function(x) {
    max(sapply(df[1:n.sp, x], decimalplaces))
  })
  id.add <- which(num.decimal < 3)
  if (length(id.add) > 0) {
    df.new <- df
    num.add <- sapply(id.add, function(id.col) {
      inc.seed()
      set.seed(local(.seed, envir = kp.env))
      rng <- 10^(-num.decimal[id.col])/2
      return(stats::runif(n.obs, -rng, rng))
    })
    local(.seed <- 0, envir = kp.env)
    df.new[, id.add] <- df.new[, id.add] + num.add
    return(df.new)
  } else return(df)
}

kp.env <- new.env(parent = environment())
local(.seed <- 0, envir = kp.env)
local(.n.reg <- 2, envir = kp.env)

inc.seed <- function() {
  local(.seed <- .seed + 1, envir = kp.env)
}

#################################################################################
# Load data (clear-sky index); same irradiance scaling as 2.1.MLR.R; SZA stays zenith in ° (von Mises). MLR uses cos(SZA°) after scaling.; SZA remains zenith in ° (circular kernel). MLR replaces SZA with cos(SZA°).
#################################################################################
fp <- file.path(dir0, "Data", "arranged15min.txt")
data <- read.table(fp, sep = "\t", header = TRUE, stringsAsFactors = FALSE)
data <- tibble(data) %>%
  mutate(Time = lubridate::ymd_hms(Time, tz = "UTC")) %>%
  mutate(across(c(yH, yL, xP, xS), ~ .x / Ghc))

data.tr <- data %>% filter(year(Time) == train_year)
data.te <- data %>% filter(year(Time) == test_year)

#################################################################################
# Kernel conditional density estimation (KCDE) bias correction
# Covariates: retrieval + cov_use (same columns as 2.1; here SZA stays zenith in ° for von Mises).
#################################################################################
blocks <- vector("list", length(obs) * length(ret))
for (i in seq_along(obs)) {
  for (j in seq_along(ret)) {
    vars <- c(ret[j], cov_use)
    pred <- KCDE(
      y = pull(data.tr, obs[i]),
      x = data.matrix(data.tr[, vars]),
      x.new = data.matrix(data.te[, vars]),
      id.G = 1,
      id.Z = 2
    )
    x <- pred * data.te$Ghc
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
