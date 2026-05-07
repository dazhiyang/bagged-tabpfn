#################################################################################
# This code is written by Dazhi Yang
# Department of Electrical Engineering and Automation
# Harbin Institute of Technology
# emails: yangdazhi.nus@gmail.com
#################################################################################

#Clear all workspace
rm(list = ls(all = TRUE))
# load necessary packages 
library(dplyr)
library(lubridate)
library(ncdf4)
library(SolarData)
library(doSNOW)
library(insol)

# NOTE:
# if you don't know how to install the "SolarData" package, see https://github.com/dazhiyang/SolarData

#################################################################################
# Inputs and functions
#################################################################################
dir.data <- "/Volumes/Macintosh Research/Data/Fuyu FY" # directory for all raw data
dir0 <- "/Users/seryangd/Library/CloudStorage/Dropbox/Working papers/Site_Adaptation"
agg <- 15 # ground data aggregation interval in minute
lat <- 47.7957 # latitude of QIQ
lon <- 124.4852 # longitude of QIQ
alt <- 170 # altitude of QIQ

#################################################################################
# solar positioning
#################################################################################
# Tm: POSIXct, UTC time
# lat: numeric, latitude in degrees -90 to 90
# lon: numeric, longitude in degrees -180 to 180
# alt: numeric, altitude in m
calZen <- function(Tm, lat, lon, alt = 0)
{
  libs <- c("insol")
  invisible(lapply(libs, library, character.only = TRUE))
  
  jd <- insol::JD(Tm)
  sunv <- insol::sunvector(jd, lat, lon, timezone = 0)
  azi <- round(insol::sunpos(sunv)[,1],3) # azimuth of the sun
  zen <- round(insol::sunpos(sunv)[,2],3) # zenith angle
  doy <- insol::daydoy(Tm)
  da <- (2 * pi / 365) * (doy - 1)
  re = 1.000110+0.034221*cos(da)+0.001280*sin(da)+0.00719*cos(2*da)+0.000077*sin(2*da)
  Io = round(1361.1*re,3) # extraterrestrial direct normal irradiance
  #Ioh = round(1361.1*re*cos(radians(zen))) # horizontal extraterrestrial irradiance
  #Ioh <- ifelse(zen>=90, 0, Ioh)
  
  solpos = list(zen, azi, Io)
  names(solpos) = c("zenith", "azimuth", "Io")
  solpos
}

#################################################################################
# K tests (see requirement 1.5)
#################################################################################
# data: tibble, data to be QCed
# alt: numeric, site altitude (aka elevation)
K.tests <- function(data, alt = 0)
{
  libs <- c("dplyr", "insol")
  invisible(lapply(libs, library, character.only = TRUE))

  # check input
  if (class(data)[1] != "tbl_df")
    stop("data must be a tibble, see 'as_tibble'.")
  if (!identical(names(data), c("Time", "GHI", "DIF", "DNI", "SZA", "AZI", "ETR")))
    stop("data columns must be named and contain 'Time','GHI','DIF','DNI','SZA','AZI','ETR'.")
  
  # compute K's
  data <- data %>%
    mutate(Kn = DNI / ETR, K = DIF / GHI, Kt = GHI / (ETR * cos(radians(SZA))))
  
  # Kn < Kt
  data <- data %>%
    mutate(flagKnKt = NA) %>%
    mutate(flagKnKt = ifelse(GHI > 50 & Kn < Kt, 0, flagKnKt)) %>%
    mutate(flagKnKt = ifelse(GHI > 50 & Kn >= Kt, 1, flagKnKt))
  
  # Kn < (1100+0.03*Elev)/ETR
  data <- data %>%
    mutate(flagKn = NA) %>%
    mutate(flagKn = ifelse(GHI > 50 & Kn < (1100+0.03*alt)/ETR, 0, flagKn)) %>%
    mutate(flagKn = ifelse(GHI > 50 & Kn >= (1100+0.03*alt)/ETR, 1, flagKn))
  
  # K_t<1.35
  data <- data %>%
    mutate(flagKt = NA) %>%
    mutate(flagKt = ifelse(GHI > 50 & Kt < 1.35, 0, flagKt)) %>%
    mutate(flagKt = ifelse(GHI > 50 & Kt >= 1.35, 1, flagKt))
  
  # K<1.05
  data <- data %>%
    mutate(flagKlowSZA = NA) %>%
    mutate(flagKlowSZA = ifelse(SZA < 75 & GHI > 50 & K < 1.05, 0, flagKlowSZA)) %>%
    mutate(flagKlowSZA = ifelse(SZA < 75 & GHI > 50 & K >= 1.05, 1, flagKlowSZA))
  
  # K<1.1
  data <- data %>%
    mutate(flagKhighSZA = NA) %>%
    mutate(flagKhighSZA = ifelse(SZA > 75 & GHI > 50 & K < 1.1, 0, flagKhighSZA)) %>%
    mutate(flagKhighSZA = ifelse(SZA > 75 & GHI > 50 & K >= 1.1, 1, flagKhighSZA))
  
  # K < 0.96
  data <- data %>%
    mutate(flagKKt = NA) %>%
    mutate(flagKKt = ifelse(Kt > 0.6 & GHI > 150 & SZA < 85 & K < 0.96, 0, flagKKt)) %>%
    mutate(flagKKt = ifelse(Kt > 0.6 & GHI > 150 & SZA < 85 & K >= 0.96, 1, flagKKt))
  
  # output the flags
  data %>% dplyr::select(one_of("Time", "flagKnKt", "flagKn", "flagKt", "flagKlowSZA", "flagKhighSZA", "flagKKt"))
}

#################################################################################
# BSRN three component tests (aka closure tests) (see requirement 1.6)
#################################################################################
# data: tibble, data to be QCed
closr.tests <- function(data)
{
  libs <- c("dplyr")
  invisible(lapply(libs, library, character.only = TRUE))
  
  # check input
  if (class(data)[1] != "tbl_df")
    stop("data must be a tibble, see 'as_tibble'.")
  if (!identical(names(data), c("Time", "GHI", "DIF", "DNI", "SZA", "AZI", "ETR")))
    stop("data columns must be named and contain 'Time','GHI','DIF','DNI','SZA','AZI','ETR'.")
  
  closr <- abs(data$GHI/(data$DNI*cos(radians(data$SZA))+data$DIF)-1)
  
  # low-zenith closure
  data <- data %>%
    mutate(flag3lowSZA = NA) %>%
    mutate(flag3lowSZA = ifelse(SZA <= 75 & GHI > 50 & closr <= 0.08, 0, flag3lowSZA)) %>%
    mutate(flag3lowSZA = ifelse(SZA <= 75 & GHI > 50 & closr > 0.08, 1, flag3lowSZA))
  
  # high-zenith closure
  data <- data %>%
    mutate(flag3highSZA = NA) %>%
    mutate(flag3highSZA = ifelse(SZA > 75 & GHI > 50 & closr <= 0.15, 0, flag3highSZA)) %>%
    mutate(flag3highSZA = ifelse(SZA > 75 & GHI > 50 & closr > 0.15, 1, flag3highSZA))
  
  # output the flags
  data %>% dplyr::select(one_of("Time", "flag3lowSZA", "flag3highSZA"))
}

#################################################################################
# ERL tests, aka, extremely-rare limit tests (see requirement 1.7)
#################################################################################
# data: tibble, data to be QCed
ERL.tests <- function(data)
{
  libs <- c("dplyr", "insol")
  invisible(lapply(libs, library, character.only = TRUE))
  
  # check input
  if (class(data)[1] != "tbl_df")
    stop("data must be a tibble, see 'as_tibble'.")
  if (!identical(names(data), c("Time", "GHI", "DIF", "DNI", "SZA", "AZI", "ETR")))
    stop("data columns must be named and contain 'Time','GHI','DIF','DNI','SZA','AZI','ETR'.")
  
  # ERL GHI test
  data <- data %>%
    mutate(flagERLGHI = NA) %>%
    mutate(flagERLGHI = ifelse(GHI >= -2 & GHI <= 1.2*ETR*(cos(radians(SZA)))^1.2+50, 0, flagERLGHI)) %>%
    mutate(flagERLGHI = ifelse(GHI < -2 | GHI > 1.2*ETR*(cos(radians(SZA)))^1.2+50, 1, flagERLGHI))
  
  # ERL DIF test
  data <- data %>%
    mutate(flagERLDIF = NA) %>%
    mutate(flagERLDIF = ifelse(DIF >= -2 & DIF <= 0.75*ETR*(cos(radians(SZA)))^1.2+30, 0, flagERLDIF)) %>%
    mutate(flagERLDIF = ifelse(DIF < -2 | DIF > 0.75*ETR*(cos(radians(SZA)))^1.2+30, 1, flagERLDIF))
  
  # ERL DNI test
  data <- data %>%
    mutate(flagERLDNI = NA) %>%
    mutate(flagERLDNI = ifelse(DNI >= -2 & DNI <= 0.95*ETR*(cos(radians(SZA)))^0.2+10, 0, flagERLDNI)) %>%
    mutate(flagERLDNI = ifelse(DNI < -2 | DNI > 0.95*ETR*(cos(radians(SZA)))^0.2+10, 1, flagERLDNI))
  
  # output the flags
  data %>% dplyr::select(one_of("Time", "flagERLGHI", "flagERLDIF", "flagERLDNI"))
}

#################################################################################
# tracker off tests (see requirement 1.9)
#################################################################################
# data: tibble, data to be QCed
tracker.off.tests <- function(data)
{
  libs <- c("dplyr", "insol")
  invisible(lapply(libs, library, character.only = TRUE))
  
  # check input
  if (class(data)[1] != "tbl_df")
    stop("data must be a tibble, see 'as_tibble'.")
  if (!identical(names(data), c("Time", "GHI", "DIF", "DNI", "SZA", "AZI", "ETR")))
    stop("data columns must be named and contain 'Time','GHI','DIF','DNI','SZA','AZI','ETR'.")
  
  data <- data %>%
    mutate(GHI_clear = 0.8*ETR*cos(radians(SZA))) %>%
    mutate(DIF_clear = 0.165*GHI_clear) %>%
    mutate(DNI_clear = (GHI_clear-DIF_clear)/cos(radians(SZA))) %>%
    mutate(cond1 = (GHI_clear-GHI)/(GHI_clear+GHI) < 0.2) %>%
    mutate(cond2 = (DNI_clear-DNI)/(DNI_clear+DNI) >0.95) %>%
    mutate(cond3 = SZA <= 85) %>%
    mutate(flagTracker = NA) %>%
    mutate(flagTracker = ifelse(cond1 & cond2 & cond3, 1, flagTracker))
  
  # output the flags
  data %>% dplyr::select(one_of("Time", "flagTracker"))
}

#################################################################################
# QIQ observations (yH)
#################################################################################
setwd(file.path(dir.data, "QIQ")) # directory for BSRN QIQ station data
files.qiq <- dir()
files.qiq <- files.qiq[order(paste0(substr(files.qiq,6,7), substr(files.qiq, 4,5)))] #sort files according to months

# use a loop to read the BSRN station-to-archive files
qiq <- NULL # create an empty object to pile data
pb <- txtProgressBar(max = length(files.qiq), style = 3) # progress bar
for(i in seq_along(files.qiq))
{
  tmp <- BSRN.read(files.qiq[i], directory = file.path(dir.data, "QIQ"), use.qc = FALSE, use.agg = FALSE)
  # select only the shortwave variables and time
  tmp <- tmp %>%
    rename(GHI = dw_solar, DNI = direct_n, DIF = diffuse) %>% 
    dplyr::select(one_of("Time", "GHI", "DIF", "DNI"))

  # append monthly data into one big tibble
  qiq <- bind_rows(qiq, tmp)
  
  setTxtProgressBar(pb,i)
}
close(pb)

# solar positioning
solpos <- calZen(qiq$Time-30, lat = lat, lon = lon, alt = alt) # "-30" to calculate the zenith in the middle of an interval
qiq <- qiq %>%
  mutate(SZA = solpos$zenith, AZI = solpos$azimuth, ETR = solpos$Io) %>%
  filter(SZA <= 85)

nrow(qiq)

# do QC tests and flag the samples
K_tests_flags <- K.tests(qiq, alt)
closr_tests_flags <- closr.tests(qiq)
ERL_tests_flags <- ERL.tests(qiq)
tracker_off_tests_flags <- tracker.off.tests(qiq)

# join the QC flags with data
qiq <- qiq %>%
  left_join(., K_tests_flags, by = "Time") %>%
  left_join(., closr_tests_flags, by = "Time") %>%
  left_join(., ERL_tests_flags, by = "Time") %>%
  left_join(., tracker_off_tests_flags, by = "Time") 

# remove the all flagged data points
flagged <- rowSums(as.matrix(qiq %>% dplyr::select(., matches("flag"))), na.rm = TRUE)
nrow(qiq)
length(which(flagged>=1))
length(which(flagged>=1))/nrow(qiq)*100 # compute flagged percentage
qiq <- qiq %>%
  filter(flagged == 0)

#################################################################################
# McClear data for clear-sky GHI
#################################################################################
setwd(file.path(dir.data, "McClear")) # directory for McClear data
files.mc <- dir()

McClear <- NULL # create an empty object to pile data
pb <- txtProgressBar(max = length(files.mc), style = 3) # progress bar
for(i in seq_along(files.mc))
{
  # read one file at a time
  tmp <- tibble(read.table(files.mc[i], skip = 37, header = FALSE, sep = ";"))
  # arrange McClear from csv to tibble
  tmp <- tmp %>%
    tidyr::separate(V1, into = c("start_time", "end_time"), sep = "/") %>%
    mutate(
      start_time = ymd_hms(start_time),
      end_time = ymd_hms(end_time)
    ) %>%
    rename(Time = end_time, TOA = V2, Ghc = V3, Bhc = V4, Dhc = V5, Bnc = V6) %>%
    dplyr::select(one_of("Time", "Ghc")) %>%
    mutate(Ghc = Ghc * 60) # McClear unit conversion from Wh/m2 to W/m2
  # append the tmp tibble into the McClear
  McClear <- McClear %>%
    bind_rows(., tmp)

  setTxtProgressBar(pb,i)
}
close(pb)

# combine QIQ data with McClear
qiq <- qiq %>%
  left_join(., McClear, by = "Time")

# aggregate the data into 15 min, to match the resolution of FY-4B
# it should be noted that according to pg 435 of Gueyarmd (2009), the best BSRN GHI measurement should use the sum of diffuse and direct components instead of directly use GHI values measured by the pyranometer.
qiq <- qiq %>%
  mutate(sum = DIF + cos(radians(SZA)) * DNI) %>%
  dplyr::select(one_of("Time", "sum", "SZA", "Ghc")) %>%
  mutate(Time = ceiling_date(Time, "15 min")) %>% # ceilinged to the time stamps
  rename(yH = sum) # yH stands for measurement with high accuracy (there will be a yL below, for low-accuracy measurements)

# count for intervals with insufficient data points for aggregation, i.e., < 7
n.point.in.each.interval <- array(0, length(unique(qiq$Time)))
non.empty.interval <- match(unique(qiq$Time[-which(is.na(qiq$yH))]), unique(qiq$Time))
n.point.in.each.interval[non.empty.interval] <- rle(as.numeric(qiq$Time[-which(is.na(qiq$yH))]))$length
bad.interval <- unique(qiq$Time)[which(n.point.in.each.interval < 7)][-1]
remove <- which(qiq$Time %in% bad.interval)
qiq[remove, c(2:3)] <- NA
qiq <- qiq %>%
  filter(complete.cases(.)) %>%
  group_by(Time) %>%
  summarise_all(., mean) %>%
  ungroup()

#################################################################################
# Get the auxiliary weather variables from ERA5 reanalysis
#################################################################################
setwd(file.path(dir.data, "ERA5"))
files.era5 <- dir(recursive = TRUE)

# find out the collocated pixel
ncin <- nc_open(files.era5[1]) # open the first nc, and all the collocated indexes will be the same for all subsequent files
collocate.lon.index <- which.min(abs(lon-ncvar_get(ncin, "longitude")))
collocate.lat.index <- which.min(abs(lat-ncvar_get(ncin, "latitude")))
nc_close(ncin) # close nc

# set up parallel processing
# number of cores used for parallel
cl <- makeCluster(8) 
registerDoSNOW(cl)

pb <- txtProgressBar(max = length(files.era5), style = 3) # progress bar
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
#foreach execution for timestamp
time1h <- foreach(j = seq_along(files.era5), .combine = 'c', .options.snow = opts, .packages = c("ncdf4", "lubridate")) %dopar% {
  ncin <- nc_open(files.era5[j]) # open nc
  # get time stamps
  time <- as.POSIXct("1970-01-01 00:00:00", tz = "UTC") + ncvar_get(ncin, "valid_time")
  nc_close(ncin) # close nc
  
  time
}
close(pb)

#foreach execution for variables
pb <- txtProgressBar(max = length(files.era5), style = 3) # progress bar
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
VAR <- foreach(j = seq_along(files.era5), .combine = 'rbind', .options.snow = opts, .packages = c("ncdf4")) %dopar% {
  # open nc files and get the parameters
  ncin <- nc_open(files.era5[j])
  # get variables
  #variables <- names(ncin$var)[3:length(names(ncin$var))]
  variables <- c("u10", "v10", "d2m", "t2m", "sp", "hcc", "lcc", "mcc", "asn", "rsn", "sd", "tcsw", "fal", "tco3", "tcwv")
  value <- sapply(seq_along(variables), function(x) ncvar_get(ncin, variables[x])[collocate.lon.index, collocate.lat.index, ])
  colnames(value) <- variables
  nc_close(ncin) # close nc
  
  value
}
close(pb)
stopCluster(cl)

# make ERA5 tibble 
era5 <- tibble(Time = time1h) %>%
  bind_cols(., as_tibble(VAR)) %>% 
  # wind speed, W [m/s], round to three decimal places
  mutate(W = round(sqrt(u10^2+v10^2),3)) %>% 
  # surface pressure, sp [Pa], need to be converted to mbar, by dividing 100
  mutate(sp = round(sp/100, 3)) %>%
  # surface temperature, t2m [K], need to be converted to celsius
  mutate(t2m = round(t2m - 273.15, 3)) %>%
  # dew point temperature, d2m [K], need to be converted to celsius
  mutate(d2m = round(d2m - 273.15, 3)) %>%
  # compute relative humidity from t2m and d2m, [%]
  mutate(rh = 10^(7.591386*(d2m/(d2m+240.7263)-t2m/(t2m+240.7263)))) %>%
  mutate(rh = rh*100) %>% # convert to percetage
  mutate(rh = ifelse(rh > 100, 100, rh)) %>%
  # snow density [kg/m3], convert to g/m3
  mutate(rsn = rsn/1000) %>%
  # Total column snow water, [kg/m2], remove negative and extreme values
  mutate(tcsw = ifelse(tcsw < 0, 0, tcsw)) %>%
  mutate(tcsw = ifelse(tcsw > 1.2, 1.2, tcsw)) %>%
  # total column ozone, [kg/m2], covert to Dobson unit
  mutate(tco3 = round(tco3/2.1415e-5, 3)) %>%
  # remove unused variables
  dplyr::select(-one_of("u10", "v10", "d2m"))


# combine QIQ and ERA5 data
qiq <- qiq %>%
  left_join(., era5, by = "Time") %>%
  mutate(across(t2m:rh, ~ zoo::na.approx(.x, na.rm = FALSE, rule = 2)))

#################################################################################
# CMA observations (yL)
#################################################################################
setwd(file.path(dir.data, "CMA")) # directory for BSRN QIQ station data
files.cma <- dir(recursive = TRUE)

cma <- NULL # create an empty object to pile data
pb <- txtProgressBar(max = length(files.cma), style = 3) # progress bar
for(i in seq_along(files.cma))
{
  # read the csv file
  tmp <- tibble(read.table(files.cma[i], skip = 1, header = TRUE, sep = " ")) 
  # arrange for date time and clean the tibble 
  # CMA data is in local time, which needs to be changed to UTC
  # 台站资料地方时：在进行气象观测、记录台站资料时，使用地方时可以更准确地反映当地自然现象发生的实际时间顺序。比如，记录日出、日落、气象要素变化等时间，使用地方时能直观体现出这些现象与当地地理位置的关系，有助于研究本地的气象规律和地理环境特征。
  # After a quick communication with Haizhi Qiu from Fuyu, this file is actually stamped with local time, which leads Beijing time (UTC+8) by 18 minutes. 
  tmp <- tmp %>%
    mutate(date = paste(Year, Mon, Day, sep = "-")) %>%
    mutate(time = paste(Hour, Min, sep = ":")) %>%
    mutate(Time = ymd_hm(paste(date, time)) - 8*3600 - 18*60) %>%
    rename(GHI = V14311) %>%
    mutate(GHI = ifelse(GHI == 999999, NA, GHI)) # CMA files missing values with 999999

  cma <- cma %>%
    bind_rows(., tmp)
  
  setTxtProgressBar(pb,i)
}
close(pb)

# aggregate the data into 15 min, to match the resolution of FY-4B
cma <- cma %>%
  dplyr::select(one_of("Time", "GHI")) %>%
  mutate(Time = ceiling_date(Time, "15 min")) %>% # ceilinged to the time stamps
  rename(yL = GHI) # yl stands for measurement with low accuracy 

# count for intervals with insufficient data points for aggregation, i.e., < 7
n.point.in.each.interval <- array(0, length(unique(cma$Time)))
non.empty.interval <- match(unique(cma$Time[-which(is.na(cma$yL))]), unique(cma$Time))
n.point.in.each.interval[non.empty.interval] <- rle(as.numeric(cma$Time[-which(is.na(cma$yL))]))$length
bad.interval <- unique(cma$Time)[which(n.point.in.each.interval < 7)][-1]
remove <- which(cma$Time %in% bad.interval)
cma[remove, 2] <- NA
cma <- cma %>%
  filter(complete.cases(.)) %>%
  group_by(Time) %>%
  summarise_all(., mean) %>%
  ungroup()
  
#################################################################################
# NSMC retrievals (xP)
#################################################################################
setwd(file.path(dir.data, "NSMC4B")) # directory for FY-4B retrieval data
files.nsmc <- dir(recursive = TRUE)
# get time stamps of FY-4B data
time <- ceiling_date(ymd_hms(sapply(strsplit(files.nsmc, split='_', fixed=TRUE), function(x) x[11])), "15 min")
# reorder files according to time
files.nsmc <- files.nsmc[order(time)]

# find out the collocated pixel
ncin <- nc_open(files.nsmc[1]) # open the first nc, and all the collocated indexes will be the same for all subsequent files
collocate.lon.index <- which.min(abs(lon-ncvar_get(ncin, "longitude")))
collocate.lat.index <- which.min(abs(lat-ncvar_get(ncin, "latitude")))
nc_close(ncin) # close nc

# use parallel computing to retrieve GHI values from NetCDF files
cl <- makeCluster(6) # number of cores used for parallel
registerDoSNOW(cl)
#foreach execution
pb <- txtProgressBar(max = length(files.nsmc), style = 3) # progress bar
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
ghi <- foreach(j = seq_along(files.nsmc), .combine = 'c', .options.snow = opts, .packages = c("ncdf4")) %dopar% {
  ncin <- nc_open(files.nsmc[j]) # open nc
  tmp <- ncvar_get(ncin, "SSI")
  nc_close(ncin) # close nc
  value <- tmp[collocate.lon.index, collocate.lat.index]
  value
}
close(pb)
stopCluster(cl)

# construct the FY-4B tibble (i.e., data frame)
nsmc <- tibble(Time = time, xP = ghi) # xP represents the physically retrieved data

# do some simple data processing, remove some fill values
nsmc <- nsmc %>%
  mutate(xP = ifelse(xP > 1500, 0, xP)) %>% # this valid max value is specified in the nc files
  mutate(xP = ifelse(xP < 0, NA, xP)) # same for this valid min value

#################################################################################
# Heliosat-2 retrievals (xS)
#################################################################################
setwd(file.path(dir.data, "Helio4B")) # directory for FY-4B retrieval data
files.helio <- dir(recursive = TRUE)
# get time stamps of HelioFY-4B data (+15 min, because Huang's file is stamped at the beginning of the scanning interval)
time <- ymd_hm(substr(files.helio, 8, 17)) + 15*60
# reorder files according to time
files.helio <- files.helio[order(time)]

# find out the collocated pixel
ncin <- nc_open(files.helio[1]) # open the first nc, and all the collocated indexes will be the same for all subsequent files
collocate.lon.index <- which.min(abs(lon-ncvar_get(ncin, "longitude")))
collocate.lat.index <- which.min(abs(lat-ncvar_get(ncin, "latitude")))
nc_close(ncin) # close nc

# use parallel computing to retrieve GHI values from NetCDF files
cl <- makeCluster(6) # number of cores used for parallel
registerDoSNOW(cl)
#foreach execution
pb <- txtProgressBar(max = length(files.helio), style = 3) # progress bar
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)
ghi <- foreach(j = seq_along(files.helio), .combine = 'c', .options.snow = opts, .packages = c("ncdf4")) %dopar% {
  ncin <- nc_open(files.helio[j]) # open nc
  tmp <- ncvar_get(ncin, "GHI")
  nc_close(ncin) # close nc
  value <- tmp[collocate.lon.index, collocate.lat.index]
  value
}
close(pb)
stopCluster(cl)

# construct the FY-4B tibble (i.e., data frame)
helio <- tibble(Time = time, xS = ghi) # xP represents the physically retrieved data

#################################################################################
# Combine the four datasets and make some simple plots to check quality
#################################################################################
# combine observations from QIQ and CMA, as well as the retrievals
joined <- qiq %>%
  left_join(., cma, by = "Time") %>%
  left_join(., nsmc, by = "Time") %>%
  left_join(., helio, by = "Time")

# Round for file size: 2 dp for most columns. ERA5 forecast albedo (fal) lives in ~0.15-0.55;
# rounding it to 2 dp collapses it to O(10-40) levels per year and breaks KernSmooth::dpill in KCDE.
num_cols <- names(joined)[vapply(joined, is.numeric, logical(1L))]
num_cols <- setdiff(num_cols, "fal")

data <- joined %>%
  mutate(
    across(all_of(num_cols), ~ round(.x, digits = 2)),
    fal = round(fal, digits = 6)
  ) %>%
  filter(., complete.cases(.)) %>%
  # Export March–December only (exclude Jan–Feb)
  filter(month(Time) >= 4L, month(Time) <= 12L) %>%
  relocate(yL, xP, xS, .after = yH) %>%
  mutate(Time = format(as.POSIXct(Time, tz = "UTC"), "%Y-%m-%d %H:%M:%S", tz = "UTC"))

cor(data$yH, data$yL)
cor(data$xP, data$xS)

# save data into txt file
setwd(file.path(dir0, "Data"))
write.table(data, file = "arranged15min.txt", quote = FALSE, sep = "\t")


# # a plot to show the agreement of yH and yL
# get_density <- function(x, y, ...){
#   dens <- MASS::kde2d(x, y, ...)
#   ix <- findInterval(x, dens$x)
#   iy <- findInterval(y, dens$y)
#   ii <- cbind(ix, iy)
#   return(dens$z[ii])
# }

# data <- tibble(read.table(file = "arranged15min.txt", sep = "\t", header = TRUE))
# data.plot1 <- data %>%
#   dplyr::select(one_of("yH", "yL")) %>%
#   mutate(density = get_density(yH, yL, n=200))
# # Discrete classes with quantile scale
# no_classes <- 10
# quantiles <- quantile(data.plot1$density, probs = seq(0, 1, length.out = no_classes + 1))
# quantiles <- scales::rescale(quantiles, to = c(0,1))
# 
# p1 <-  ggplot(data.plot1) +
#   geom_scattermore(aes(x = yH, y = yL, color = density), pointsize = point.size*45) +
#   viridis::scale_color_viridis(name ="density", direction = 1, option = "B", values = quantiles) +
#   geom_abline(linewidth = line.size*2, intercept = 0, slope = 1, linetype = "dashed", color = "#2E8B57") +
#   #geom_hex(aes(x = yH, y = yL, fill = ..count..), bins = 100) +
#   #viridis::scale_fill_viridis(name = "Count", direction = 1, option = "B", na.value = 'transparent', trans = scales::pseudo_log_trans(sigma = 0.001)) +
#   scale_x_continuous(name = expression(paste("QIQ observation, ", italic(y)[H], " [W ", m^-2, "]")), expand = c(0,0), limits = c(0, 1100)) +
#   scale_y_continuous(name = expression(paste("CMA observation, ",italic(y)[L], " [W ", m^-2, "]")), expand = c(0,0), limits = c(0, 1100)) +
#   theme_bw() +
#   theme(plot.margin = unit(c(0.1,0.2,0,0.2), "lines"), panel.spacing = unit(0.05, "lines"), plot.background = element_rect(fill = "transparent", color = NA), text = element_text(family = "Times", size = plot.size), strip.text.x = element_text(margin = margin(0.05,0,0.05,0, "lines"), size = plot.size), strip.text.y = element_text(margin = margin(0,0.05,0,0.05, "lines"), size = plot.size), axis.title = element_text(size = plot.size), axis.text = element_text(size = plot.size), legend.position = "none", legend.text = element_text(family = "Times", size = plot.size, color = "black"), legend.title = element_text(family = "Times", size = plot.size, color = "black"), legend.key.height = unit(1, "lines"), legend.key.width = unit(0.9, "lines"), legend.box.margin = unit(c(-0.7,0,0,0), "lines"), legend.background = element_rect(fill = "transparent", colour = "transparent"), legend.key = element_rect(fill = "transparent"), panel.background = element_rect(fill = "transparent", colour = "transparent"))
# 
# p1
# 
# # another plot to show the agreement of xP and xS
# data.plot2 <- data %>%
#   dplyr::select(one_of("xP", "xS")) %>%
#   mutate(density = get_density(xP, xS, n=200))
# # Discrete classes with quantile scale
# no_classes <- 10
# quantiles <- quantile(data.plot2$density, probs = seq(0, 1, length.out = no_classes + 1))
# quantiles <- scales::rescale(quantiles, to = c(0,1))
# 
# p2 <-  ggplot(data.plot2) +
#   geom_scattermore(aes(x = xP, y = xS, color = density), pointsize = point.size*45) +
#   viridis::scale_color_viridis(name ="density", direction = 1, option = "B", values = quantiles) +
#   geom_abline(linewidth = line.size*2, intercept = 0, slope = 1, linetype = "dashed", color = "#2E8B57") +
#   #geom_hex(aes(x = yH, y = yL, fill = ..count..), bins = 100) +
#   #viridis::scale_fill_viridis(name = "Count", direction = 1, option = "B", na.value = 'transparent', trans = scales::pseudo_log_trans(sigma = 0.001)) +
#   scale_x_continuous(name = expression(paste("NSMC retrieval, ", italic(x)[P], " [W ", m^-2, "]")), expand = c(0,0), limits = c(0, 1100)) +
#   scale_y_continuous(name = expression(paste("Heliosat-2 retrieval, ",italic(x)[S], " [W ", m^-2, "]")), expand = c(0,0), limits = c(0, 1100)) +
#   theme_bw() +
#   theme(plot.margin = unit(c(0.1,0.1,0,0.3), "lines"), panel.spacing = unit(0.05, "lines"), plot.background = element_rect(fill = "transparent", color = NA), text = element_text(family = "Times", size = plot.size), strip.text.x = element_text(margin = margin(0.05,0,0.05,0, "lines"), size = plot.size), strip.text.y = element_text(margin = margin(0,0.05,0,0.05, "lines"), size = plot.size), axis.title = element_text(size = plot.size), axis.text = element_text(size = plot.size), legend.position = "none", legend.text = element_text(family = "Times", size = plot.size, color = "black"), legend.title = element_text(family = "Times", size = plot.size, color = "black"), legend.key.height = unit(1, "lines"), legend.key.width = unit(0.9, "lines"), legend.box.margin = unit(c(-0.7,0,0,0), "lines"), legend.background = element_rect(fill = "transparent", colour = "transparent"), legend.key = element_rect(fill = "transparent"), panel.background = element_rect(fill = "transparent", colour = "transparent"))
# 
# p2
# 
# p <- ggpubr::ggarrange(p1, p2, ncol = 2, align = "h", labels = c("(a)", "(b)"), widths = c(1, 1), font.label = list(size = plot.size, color = "black", face = "plain", family = "Times"))

# setwd(file.path(dir0, "Revision 1"))
# ggsave(filename = "yHyLxPxS.pdf", plot = p, width = 120, height = 55, unit = "mm")

