library(ggplot2)
library(scales)
library(grid)
library(hdf5r)

seq2char <- function(dat, aas = 1:40) {
  return(apply(dat, 1, function(d) paste(aas[d], collapse=",")))
}
charge2int <- function(dat) {
  return(apply(dat, 1, function(d) which.max(d)))
}
readHDF5 <- function(file, sparse=F, custom_data = NULL, nlosses = c(""), 
                     parse = c('important fragment' = F, 'raw data' = T, 'pred data' = T, 'raw mz' = T, 'pred mz' = T, 'updated mz' = T)) {
  dh5 <- h5file(file, mode="r")
  cat("parsing meta data\n")

  meta <- data.frame(	raw_file = as.character(dh5[["rawfile"]][]),
                      scan_number = dh5[["scan_number"]][],
                      ce = dh5[["collision_energy_normed"]][],
                      ce_calib = dh5[["collision_energy_aligned_normed"]][],
                      sequence = seq2char(t(dh5[['sequence_integer']][,])),
                      precursor_charge = charge2int(t(dh5[["precursor_charge_onehot"]][,])),
                      stringsAsFactors = F
  )
  additionals <- c("method","collision_energy_aligned", "collision_energy_aligned_normed", "score", "spectral_angle", "reverse", "se_rank", 'ce_normed','tmt','mass_analyzer')
  additionals <- c(custom_data, additionals)
  for (additional in additionals) {
    tmp <- NULL
    try(tmp <- dh5[[additional]][], silent=T)
    if (!is.null(tmp)) {
      if (class(tmp) == "matrix") {
        meta[,additional] <- tmp[,1]
      } else {
        meta[,additional] <- tmp
      }
    } else {
      try(tmp <- dh5[[additional]][,], silent=T)
      if (!is.null(tmp)) {
        if (class(tmp) == "matrix") {
          meta[,additional] <- tmp[,1]
        } else {
          meta[,additional] <- tmp
        }
      }else{
      message("Could not find: ", additional)
      }
    }
  }
  dist_info <- NULL
  try(dist_info <- dh5["ph_dist_infos"][], silent=T)
  if (!is.null(dist_info)) {
    meta[, "ph_dist"] = dist_info[,1]
    meta[, "unique_phs"] = dist_info[,2]
    meta[, "common_phs"] = dist_info[,3]
  } else {
    message("Could not find: ", "ph_dist_infos")
  }
  try(dist_info <- dh5["ph_dist_infos_truth"][], silent=T)
  if (!is.null(dist_info)) {
    meta[, "ph_dist_truth"] = dist_info[,1]
    meta[, "unique_phs_truth"] = dist_info[,2]
    meta[, "common_phs_truth"] = dist_info[,3]
  } else {
    message("Could not find: ", "ph_dist_infos_truth")
  }
  
  if (length(grep("reverse", colnames(meta))) > 0) {
    meta$reverse <- as.logical(meta$reverse)
  } else {
    meta$reverse <- F
  }
  
  meta$sequence <- as.character(meta$sequence)
  seq_list = strsplit(meta$sequence, split = ",")
  meta$len <- lapply(seq_list, length)
  meta$UID <- 1:dim(meta)[1]
  meta$PCM <- paste(meta$sequence, meta$precursor_charge, meta$method, meta$ce, sep="|")
  
  meta$nterm <- sapply(seq_list, function(x) x[1])
  meta$cterm <- sapply(seq_list, function(x, ...) x[length(x)])
  
  fnum <- 1:29
  ftypes <- c("y", "b")
  zs <- 1:3
  ion_numbers <- rep(fnum, each=length(zs) * length(ftypes) * length(nlosses))
  ion_types <- rep(rep(ftypes, each=length(zs) * length(nlosses)), length(fnum))
  neutral_losses <- rep(rep(nlosses, each=length(zs), length(fnum) * length(ftypes)))
  ion_charges <- rep(zs, length(fnum) * length(ftypes) * length(nlosses))
  cn <- paste(ion_types, ion_numbers, neutral_losses, "_", ion_charges, sep="")
  
  ret <- list()
  ret[["meta"]] <- meta
  
  if (!sparse) {
    if(parse['important fragment']){
      cat("parsing important fragments\n")
      important_fragment <- NULL
      try(important_fragment <- dh5["important_fragment"][,], silent=T)
      if (!is.null(important_fragment)) {
        colnames(important_fragment) <- cn
        rownames(important_fragment) <- meta$UID
        n_col = ncol(important_fragment)
        #browser()
        important_fragment = as.logical(important_fragment)
        important_fragment = matrix(important_fragment, ncol = n_col, byrow = F)
      }
      ret[["important_fragment"]] <- important_fragment
    }
    
    if(parse['raw data']){
      cat("parsing raw data\n")
      raw <- t(dh5[["intensities_raw"]][,])
      colnames(raw) <- cn
      rownames(raw) <- meta$UID
      ret[["raw"]] <- raw
    }
    
    if(parse['pred data']){
      cat("parsing pred data\n")
      pred <- NULL
      try(pred <- t(dh5[["intensities_pred"]][,]))
      if (!is.null(pred)) {
        colnames(pred) <- cn
        rownames(pred) <- meta$UID
      }
      ret[["pred"]] <- pred
    }
    
    #if(parse['pred mz']){
     # cat("parsing pred mz data\n")
     # pred_mz <- NULL
     # try(pred_mz <- t(dh5[["masses_pred"]][,]), silent=T)
     # if (!is.null(pred_mz)) {
       # colnames(pred_mz) <- cn
       # rownames(pred_mz) <- meta$UID
     # }
     # ret[["pred_mz"]] <- pred_mz
    #}
    
    #if(parse['raw mz']){
     # cat("parsing raw mz data\n")
     # raw_mz <- NULL
     # try(raw_mz <- t(dh5[["masses_raw"]][,]), silent=T)
     # if (!is.null(raw_mz)) {
      #  colnames(raw_mz) <- cn
      #  rownames(raw_mz) <- meta$UID
      #}
     # ret[["raw_mz"]] <- raw_mz
    #}
    
    #if(parse['updated mz']){
     # cat("parsing updated mz data\n")
     # update_mz <- NULL
     # try(update_mz <- t(dh5[["masses_pred_update"]][,]), silent=T)
     # if (!is.null(update_mz)) {
       # colnames(update_mz) <- cn
       # rownames(update_mz) <- meta$UID
      #}
     # ret[["update_mz"]] <- update_mz
    #}
    
  }
  
  h5close(dh5)
  message("Successfully parsed hdf5")
  return(ret)
}

GeomSplitViolin <- ggproto("GeomSplitViolin", GeomViolin, 
                           draw_group = function(self, data, ..., draw_quantiles = NULL) {
                             data <- transform(data, xminv = x - violinwidth * (x - xmin), xmaxv = x + violinwidth * (xmax - x))
                             grp <- data[1, "group"]
                             newdata <- plyr::arrange(transform(data, x = if (grp %% 2 == 1) xminv else xmaxv), if (grp %% 2 == 1) y else -y)
                             newdata <- rbind(newdata[1, ], newdata, newdata[nrow(newdata), ], newdata[1, ])
                             newdata[c(1, nrow(newdata) - 1, nrow(newdata)), "x"] <- round(newdata[1, "x"])
                             
                             if (length(draw_quantiles) > 0 & !scales::zero_range(range(data$y))) {
                               stopifnot(all(draw_quantiles >= 0), all(draw_quantiles <=
                                                                         1))
                               quantiles <- ggplot2:::create_quantile_segment_frame(data, draw_quantiles)
                               aesthetics <- data[rep(1, nrow(quantiles)), setdiff(names(data), c("x", "y")), drop = FALSE]
                               aesthetics$alpha <- rep(1, nrow(quantiles))
                               both <- cbind(quantiles, aesthetics)
                               quantile_grob <- GeomPath$draw_panel(both, ...)
                               ggplot2:::ggname("geom_split_violin", grid::grobTree(GeomPolygon$draw_panel(newdata, ...), quantile_grob))
                             }
                             else {
                               ggplot2:::ggname("geom_split_violin", GeomPolygon$draw_panel(newdata, ...))
                             }
                           })

geom_split_violin <- function(mapping = NULL, data = NULL, stat = "ydensity", position = "identity", ..., 
                              draw_quantiles = NULL, trim = TRUE, scale = "area", na.rm = FALSE, 
                              show.legend = NA, inherit.aes = TRUE) {
  layer(data = data, mapping = mapping, stat = stat, geom = GeomSplitViolin, 
        position = position, show.legend = show.legend, inherit.aes = inherit.aes, 
        params = list(trim = trim, scale = scale, draw_quantiles = draw_quantiles, na.rm = na.rm, ...))
}


jaccAlphaBinary = function(pred, exp){
  # add cases so that table will always be 2x2
  tp = sum(pred==1 & exp==1)
  fp = sum(pred==1 & exp==0)
  return(fp/(tp+fp))
}
jaccAlphaBinaryRow = function(pred, exp){
  tp = rowSums(pred==1 & exp==1)
  message('tp alpha')
  fp = rowSums(pred==1 & exp==0)
  message('fp alpha')
  return(fp/(tp+fp))
}

jaccBetaBinary = function(pred, exp){
  tp = sum(pred==1 & exp==1)
  fn = sum(pred==0 & exp==1)
  return(fn/(tp+fn))
}
jaccBetaBinaryRow = function(pred, exp){
  message('beta started')
  tp = rowSums(pred==1 & exp==1)
  message('tp beta')
  fn = rowSums(pred==0 & exp==1)
  return(fn/(tp+fn))
}


addJaccError = function(hdf5_loaded){
  predNZ = hdf5_loaded$pred>0
  rawNZ = hdf5_loaded$raw>0
  jacc_alpha = jaccAlphaBinaryRow(predNZ, rawNZ)
  message('jacc_alpha')
  jacc_beta = jaccBetaBinaryRow(predNZ, rawNZ)
  message('jacc_beta')
  hdf5_loaded$meta$jacc_alpha = jacc_alpha
  hdf5_loaded$meta$jacc_beta = jacc_beta
  return(hdf5_loaded)
}

