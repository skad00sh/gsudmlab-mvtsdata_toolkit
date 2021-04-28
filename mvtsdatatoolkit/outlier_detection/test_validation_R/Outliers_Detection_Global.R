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
  df_list <- df_list[!is.na(df_list)]
  Q1 <- quantile(df_list, 0.25)
  Q3 <- quantile(df_list, 0.75)
  IQR <- Q3 - Q1
  lower_div <- Q1 - 1.5 * IQR
  upper_div <- Q3 + 1.5 * IQR
  
  outliers <- c(df_list[df_list > upper_div], df_list[df_list < lower_div])
  return (sort(outliers))
  #return(sort(boxplot.stats(df_list[!is.na(df_list)])$out))
}