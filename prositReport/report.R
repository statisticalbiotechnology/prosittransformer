install.packages("optparse")
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("gridExtra")
install.packages("stringr")
install.packages("scales")
install.packages("hdf5r")
install.packages("plyr")
install.packages("feather")

library(feather)

list.of.packages = c("methods", "optparse", "dplyr", "tidyr", "ggplot2", "grid", "gridExtra", "stringr", "scales")
suppressWarnings({packages_check = sapply(list.of.packages, require, character.only=TRUE, warn.conflicts = FALSE)})
if (!all(packages_check)) {
  stop("These packages are needed but were not installed:\n", 
       paste(names(packages_check)[!packages_check], collapse = ", "))
}
source("utils.R")
# define functions
aggregate_by = function(df, agg_var=""){
  if(agg_var == ""){
    df_agg = df %>% group_by(data_set)
  } else {
    df_agg = df %>% group_by(.dots = list('data_set', agg_var))
  }
  df_agg = df_agg %>% summarise(Mean_jacc_alpha=round(mean(jacc_alpha), 4),
                                Mean_jacc_beta=round(mean(jacc_beta), 4),
                                Mean_sa=round(mean(spectral_angle), 4), 
                                log_count=log10(n()))
  
  df_temp = df_agg %>% as.data.frame %>% nest(log_count, Mean_jacc_alpha, Mean_jacc_beta, Mean_sa, .key = 'value_col') %>% 
    spread(key = data_set, value = value_col)
  
  # reshaping to long format by test and ho columns:
  # replace NULL entries with proper dimensioned NA values
  ho_null = sapply(df_temp$ho, is.null)
  df_temp$ho[ho_null] = list(tibble(Mean_jacc_alpha=NA, Mean_jacc_beta=NA,Mean_sa=NA, log_count=NA))
  test_null = sapply(df_temp$test, is.null)
  df_temp$test[test_null] = list(tibble(Mean_jacc_alpha=NA, Mean_jacc_beta=NA, Mean_sa=NA, log_count=NA))
  df_agg_table = df_temp %>% unnest(test, ho, .sep = '_')
  df_agg_table$test_log_count = as.integer(10^df_agg_table$test_log_count)
  df_agg_table$ho_log_count = as.integer(10^df_agg_table$ho_log_count)
  
  if(agg_var==""){
    ret = list(df_agg_table)
  } else{
    ret  = list()
    ret[[agg_var]] = df_agg_table
  }
  return(ret)
}

makePlots = function(
  df,
  out_csv, 
  pool_filter=NULL, 
  #aggregate_vars = c("","ce","mass_analyzer","len", "ce_calib_binned", "precursor_charge", "nterm", "cterm","method","pool")
  aggregate_vars = c("","ce","len", "ce_calib_binned", "precursor_charge", "nterm", "cterm","method","pool")
  ){
  label = "All"
  if (!is.null(pool_filter)){
    df = df %>% filter(pool == pool_filter)
    label = pool_filter
  }
  # aggregate for all psms
  aggregated_data = lapply(aggregate_vars, function(char){
    listed = aggregate_by(df = df, agg_var = char)
    return(listed)
  })
  aggregated_data = unlist(aggregated_data, recursive = FALSE)
  
  # Plot the data
  for(agg_var in names(aggregated_data)){
    cat("x")
    if(agg_var=="") agg_var = "All" #aes_strings is not accepting agg_var if it is empty
    
    # violin plots
    # geom_split_violin is not working properly with aes_ or aes_string. 
    # To make the dynamic selection possible the data itself is use
    print(ggplot(df, aes(x=df[,agg_var], y=spectral_angle, fill = data_set)) + 
            geom_split_violin() + labs(title = label) + xlab(agg_var) + scale_y_continuous(limits = c(0,1)))
    print(ggplot(df, aes(x=df[,agg_var], y=jacc_alpha, fill = data_set)) + 
            geom_split_violin() + labs(title = label) + xlab(agg_var) + scale_y_continuous(limits = c(0,1)))
    print(ggplot(df, aes(x=df[,agg_var], y=jacc_beta, fill = data_set)) + 
            geom_split_violin() + labs(title = label) + xlab(agg_var) + scale_y_continuous(limits = c(0,1)))
    # bar plots
    print(ggplot(df, aes_string(x=agg_var, y="spectral_angle", fill="data_set")) + 
            geom_bar(position = "dodge", stat = "summary", fun.y = "mean") + 
            labs(title = label) + scale_y_continuous(limits = c(0,1)))
    print(ggplot(df, aes_string(x=agg_var, y="jacc_alpha", fill="data_set")) + 
            geom_bar(position = "dodge", stat = "summary", fun.y = "mean") + 
            labs(title = label) + scale_y_continuous(limits = c(0,1)))
    print(ggplot(df, aes_string(x=agg_var, y="jacc_beta", fill="data_set")) + 
            geom_bar(position = "dodge", stat = "summary", fun.y = "mean") + 
            labs(title = label) + scale_y_continuous(limits = c(0,1)))
    print(ggplot(df, aes_string(x=agg_var, y="counts", fill="data_set")) + 
            geom_bar(position = "dodge", stat = "summary", fun.y = "sum") + 
            labs(title = label))
    # table plot
    grid.newpage()
    grid.text(label, x = 0.1, y=0.95)
    subdir = file.path(out_csv, label)
    dir.create(subdir, showWarnings = FALSE)
    csv_name = file.path(subdir, paste0("aggregation_", agg_var, ".csv"))
    if(agg_var=="All") agg_var = 1
    df_to_save = aggregated_data[[agg_var]] %>% as.data.frame()
    grid.table(df_to_save, rows=NULL)
    write.csv(df_to_save, csv_name, row.names = FALSE)
  }
}

## Script starts here ##
# add parser
option_list = list(
  make_option(c("-v", "--val_file"), type="character", default=NULL,
              help="validation dataset file name", metavar="character"),
  make_option(c("-o", "--ho_file"), type="character", default=NULL,
              help="holdout dataset file name", metavar="character"),
  make_option(c("-d", "--out_dir"), type="character", default=NULL,
              help="Directory path of the report", metavar="character"),
  make_option(c("-n", "--out_name"), type="character", default="report.pdf",
              help="output file name [default= %default]", metavar="character"),
 make_option(c("-f", "--out_df_name"), type="character", default="dataframe.feather",
              help="output file name [default= %default]", metavar="character"),
  make_option(c("-c", "--out_csv"), type="character", default="report_csvs/",
              help="output folder for report csvs: [default= %default]", metavar="character")
)
opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)
if (is.null(opt$val_file)){
  print_help(opt_parser)
  stop("--val_file must be supplied.", call.=FALSE)
}
if (is.null(opt$ho_file)){
  print_help(opt_parser)
  stop("--ho_file must be supplied.", call.=FALSE)
}
if (is.null(opt$out_dir)){
  print_help(opt_parser)
  stop("--out_dir must be supplied.", call.=FALSE)
}
out = file.path(opt$out_dir, opt$out_name)
out_df = file.path(opt$out_dir, opt$out_df_name)
out_csvs = file.path(opt$out_dir, opt$out_csv)


message("Loading validation data")
test_data = readHDF5(opt$val_file)
message('file Read')
test_data = addJaccError(test_data)
test_data$meta$data_set = "test"
test_data$pred = NULL
test_data$raw = NULL

message("Loading holdout data")
ho_data = readHDF5(opt$ho_file)
ho_data = addJaccError(ho_data)
ho_data$meta$data_set = "ho"
ho_data$pred = NULL
ho_data$raw = NULL

invisible(gc(verbose = FALSE))
message("Combine data")
combined_df = rbind.data.frame(ho_data$meta, test_data$meta)

#write_feather(combined_df, "/sdd/PrositRscript/report/dataframe.feather")

combined_df$data_set = factor(combined_df$data_set, levels = c("test", "ho"))
combined_df$nterm = factor(combined_df$nterm, sort(as.integer(unique(combined_df$nterm))))
combined_df$cterm = factor(combined_df$cterm, sort(as.integer(unique(combined_df$cterm))))
ce= round(combined_df$ce, digits = 2)
combined_df$ce = factor(ce, levels = sort(unique(ce)))
ce_calib_binned = round(combined_df$ce_calib, digits = 2)
combined_df$ce_calib_binned = factor(ce_calib_binned, levels = sort(unique(ce_calib_binned)))
combined_df$precursor_charge = factor(combined_df$precursor_charge, levels = sort(unique(combined_df$precursor_charge))) 
combined_df$cterm = factor(combined_df$cterm, levels = sort(unique(combined_df$cterm)))
combined_df$method <- factor(combined_df$method, levels = sort(unique(combined_df$method)))
if(class(combined_df$len)=='list') combined_df$len = unlist(combined_df$len)

combined_df$len = factor(combined_df$len, levels = sort(unique(combined_df$len)))
#combined_df$phospho = factor(combined_df$phospho, levels= sort(unique(combined_df$phospho)))
#combined_df$mass_analyzer= factor(combined_df$mass_analyzer, levels= sort(unique(combined_df$mass_analyzer)))

combined_df$All = ""
combined_df$counts = 1

message("Prepare pool strings")
pool = str_split(str_split_fixed(combined_df$raw_file, pattern = "-", n = 3)[,2], "_")
pool = sapply(pool, FUN = function(vec){
  stopp = length(vec) - 3 # last 3 numbers are removed
  paste(vec[1:stopp], collapse="_")
})
combined_df$pool = pool

write_feather(combined_df, out_df)

dir.create(out_csvs, showWarnings = FALSE)

message("Start plotting to pdf:")
{
  pdf(file = out, height=9, width=15)
  makePlots(combined_df, out_csv = out_csvs)
  for (pool in unique(combined_df$pool)){
    cat("\n")
    message("Plotting pool: ", pool)
    makePlots(combined_df, pool, out_csv = out_csvs)
  }
  invisible(dev.off())
}
