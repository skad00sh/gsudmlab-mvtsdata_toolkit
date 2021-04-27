# Function to detect outliers using standard deviation
Outliers_StdDev <- function(df_list, distance_threshold = 3){
  
  df_mean <- mean(df_list[!is.na(df_list)])
  df_sd <- sd(df_list)
  upper_div <- df_mean + (distance_threshold * df_sd)
  lower_div <- df_mean - (distance_threshold * df_sd)
  
  outliers <- c(df_list[df_list > upper_div], df_list[df_list < lower_div])
  return(sort(outliers))
}

# Function to detect outliers using IQR method
Outliers_IQR <- function(df_list){
  return(sort(boxplot.stats(df_list[!is.na(df_list)])$out))
}