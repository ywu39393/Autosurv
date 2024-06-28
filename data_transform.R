setwd("/cloud-home/U1039935")
# improt data
genesets <- readRDS("/cloud-home/U1039935/genesets.RDS")

# Initialize an empty list to store individual data frames
dat_list <- list()

# Loop through all 13 datasets in genesets
for (i in 1:13) {
  # Create a data frame for each dataset
  dat <- as.data.frame(genesets[[i]])
  colnames(dat) <- 'gene'
  
  #get pathway name
  path_name <- names(genesets[i])
  
  # Add a new column with the dataset name and value 1
  dat[[path_name]] <- 1
  
  # Store this data frame in our list
  dat_list[[i]] <- dat
}

# Combine all data frames using a full join
library(dplyr)
final_dat <- dat_list[[1]]
for (i in 2:13) {
  final_dat <- full_join(final_dat, dat_list[[i]], by = 'gene')%>% distinct()
}

# Replace NA with 0
final_dat[is.na(final_dat)] <- 0

#test if gene are all unique
anyDuplicated(final_dat$gene) == 0

write.csv(final_dat, "pathway_mask.csv", row.names = FALSE)

dat <- read.csv('pathway_mask.csv')



