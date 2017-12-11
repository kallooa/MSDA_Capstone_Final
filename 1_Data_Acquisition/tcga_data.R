#tcga data
library(RCurl)
library(jsonlite)
library(gtools)

cohort_api_endpoint = "http://digitalslidearchive.emory.edu:8080/api/v1/tcga/cohort?limit=500&sort=name&sortdir=1"
cohort_data = fromJSON(cohort_api_endpoint)
cohort_names = cohort_data$data$tcga$cohort
list_of_cohorts = cohort_data$data$`_id`

image_folders_df = data.frame(folderId=NA, collection=NA)

for (cohort_index in 1:length(list_of_cohorts)) {
  cohort_image_url_endpoint = paste0("http://digitalslidearchive.emory.edu:8080/api/v1/tcga/cohort/", list_of_cohorts[cohort_index], "/images?limit=500000&sort=name&sortdir=1")
  cohort_images = fromJSON(cohort_image_url_endpoint)
  cohort_image_folders = data.frame(cohort_images$data$folderId)
  cohort_image_folders$collection = cohort_names[cohort_index]
  names(cohort_image_folders)[1] = "folderId"
  image_folders_df = rbind(image_folders_df, cohort_image_folders)
}

image_folders_df = image_folders_df[complete.cases(image_folders_df$folderId),]
row.names(image_folders_df) = NULL

temp_url = paste0("http://digitalslidearchive.emory.edu:8080/api/v1/item?folderId=",image_folders_df$folderId[1],"&limit=500&sort=lowerName&sortdir=1")
temp_data = fromJSON(temp_url)
image_data = data.frame(image_id = temp_data$`_id`, image_name=temp_data$name, image_folderId=temp_data$folderId, collection=image_folders_df$collection[1], stringsAsFactors = FALSE)

for (image_index in 2:length(image_folders_df$folderId)) {
  image_data_endpoint = paste0("http://digitalslidearchive.emory.edu:8080/api/v1/item?folderId=",image_folders_df$folderId[image_index],"&limit=500&sort=lowerName&sortdir=1")
  temp_data = fromJSON(image_data_endpoint)
  temp_df = data.frame(image_id = temp_data$`_id`, image_name=temp_data$name, image_folderId=temp_data$folderId, collection=image_folders_df$collection[image_index], stringsAsFactors = FALSE)
  image_data = rbind(image_data, temp_df)
  print(paste0(image_index, " / ", length(image_folders_df$folderId)))
}

