library(stringr)

drive = "D:/brca_thumbnails/"
data_folder = paste0(drive, "data/")
train_dir = paste0(data_folder, "train/")
val_dir = paste0(data_folder, "val/")
test_dir = paste0(data_folder, "test/")

#proportions of 20 million total
train_proportion = 2000/3164
val_proportion = 464/3164
test_proportion = 700/3164

files = list.files("D:/brca_thumbnails/brca/", recursive = TRUE, full.names = TRUE, pattern = "jpg")
files = data.frame(files, stringsAsFactors = FALSE)

save(files, file="D:/scripts/Data_Organization/Slide_Level_Data_Organization/brca.RData")

#load("G:/all_imgs.RData")

set.seed(1)

train = data.frame(sample(files$files, size=train_proportion*nrow(files)), stringsAsFactors = FALSE)
names(train) = "train"
remainder_files = data.frame(files$files[!(files$files %in% train$train)], stringsAsFactors = FALSE)

val = data.frame(sample(remainder_files[,1], size=val_proportion*nrow(files)), stringsAsFactors = FALSE)
names(val) = "val"
names(remainder_files) = "files"

remainder_files = data.frame(files = remainder_files$files[!(remainder_files$files %in% val$val)], stringsAsFactors = FALSE)

test = data.frame(test = sample(remainder_files$files, size=test_proportion*nrow(files)), stringsAsFactors = FALSE)
names(test) = "test"

######Set up for binary DX/PM

train$class = str_extract(train$train, pattern="DX|(T|B|M)S")
train = train[!is.na(train$class),]
train$class[train$class!="DX"] = "PM"
val$class = str_extract(val$val, pattern="DX|(T|B|M)S")
val =  val[!is.na(val$class),]
val$class[val$class!="DX"] = "PM"
test$class = str_extract(test$test, pattern="DX|(T|B|M)S")
test =  test[!is.na(test$class),]
test$class[test$class!="DX"] = "PM"


unique_classes = unique(test$class)
if(!dir.exists(data_folder)) {
  dir.create(data_folder)
}
if(!dir.exists(train_dir)) {
  dir.create(train_dir)
}
if (!dir.exists(val_dir)) {
  dir.create(val_dir)
}
if (!dir.exists(test_dir)) {
  dir.create(test_dir)
}

for(current_dir in c(train_dir, val_dir, test_dir)) {
  for (current_class in unique_classes) {
    current_folder = paste0(current_dir, current_class)
    if(!dir.exists(current_folder)) {
      dir.create(current_folder)
    }
  }
}

for (i in 1:nrow(train)) {
  to_name = paste0(train_dir, train$class[i], "/", basename(train$train[i]))
  file.copy(from=train$train[i], to=to_name)
  print(i)
}

for (i in 1:nrow(val)) {
  to_name = paste0(val_dir, val$class[i], "/", basename(val$val[i]))
  file.copy(from=val$val[i], to=to_name)
  print(i)
}

for (i in 1:nrow(test)) {
  to_name = paste0(test_dir, test$class[i], "/", basename(test$test[i]))
  file.copy(from=test$test[i], to=to_name)
  print(i)
}




all_thumbnails_dir = "D:/brca_thumbnails/data/test_all/"
all_thumbs_df = data.frame(filename=list.files(all_thumbnails_dir, recursive = TRUE, pattern="jpg"), stringsAsFactors = FALSE)
all_thumbs_df$full_path = list.files(all_thumbnails_dir, recursive = TRUE, full.names = TRUE, pattern="jpg")
all_thumbs_df$class = unlist(lapply(all_thumbs_df$filename, function(x) {strsplit(x, split="/")[[1]][1]}))
all_thumbs_df$section = str_extract(all_thumbs_df$filename, pattern="DX|(T|B|M)S")
all_thumbs_df$section[all_thumbs_df$section!="DX"] = "PM"
all_thumbs_df = all_thumbs_df[!is.na(all_thumbs_df$section),]
write.csv(all_thumbs_df, paste0(all_thumbnails_dir, "thumbnail_data.csv"), row.names = FALSE)

for (i in 1:nrow(all_thumbs_df)) {
  from_file = all_thumbs_df$full_path[i]
  to_file = paste0("E:/brca_thumbnails/", "all/", basename(all_thumbs_df$full_path[i]))
  file.copy(from=from_file, to=to_file)
}




###Set up for 27 class thumbnail classification
require(RCurl)
temp = getURL("https://raw.github.com/gist/984691/fb8e0483b093caa871444db162ed11210a1bac5b/Stratified.R")
source(textConnection(temp))


drive = "E:/"
data_folder = paste0(drive, "data_thumbnails/")
train_dir = paste0(data_folder, "train/")
val_dir = paste0(data_folder, "val/")
test_dir = paste0(data_folder, "test/")
train_proportion = 0.75
val_proportion = 0.05
test_proportion = 0.20

#all_thumbs_df_bk = all_thumbs_df
all_thumbs_df = all_thumbs_df[all_thumbs_df$class!="all",]
classes_27 = list.dirs("E:/data/train", recursive=FALSE, full.names = FALSE)
all_thumbs_df = all_thumbs_df[(all_thumbs_df$class %in% classes_27),]

set.seed(1)

train_27 = stratified(all_thumbs_df, 3, train_proportion)[,c(1:4)]#all_thumbs_df[sample(train_proportion*nrow(all_thumbs_df)),]
#names(train_27) = "train"
remainder_files_27 = all_thumbs_df[!(all_thumbs_df$full_path %in% train_27$full_path),]

val_27 = stratified(remainder_files_27, 3, nrow(all_thumbs_df)/nrow(remainder_files_27)*val_proportion)[,c(1:4)]#remainder_files_27[sample(val_proportion*nrow(all_thumbs_df)),]
#names(val_27) = "val"
#names(remainder_files_27) = "files"

remainder_files_27 = remainder_files_27[!(remainder_files_27$full_path %in% val_27$full_path),]

test_27 = remainder_files_27#data.frame(test = sample(remainder_files_27$files, size=test_proportion*nrow(all_thumbs_df)), stringsAsFactors = FALSE)
#names(test) = "test"

# train$class = str_extract(train$train, pattern="DX|(T|B|M)S")
# train = train[!is.na(train$class),]
# train$class[train$class!="DX"] = "PM"
# val$class = str_extract(val$val, pattern="DX|(T|B|M)S")
# val =  val[!is.na(val$class),]
# val$class[val$class!="DX"] = "PM"
# test$class = str_extract(test$test, pattern="DX|(T|B|M)S")
# test =  test[!is.na(test$class),]
# test$class[test$class!="DX"] = "PM"


unique_classes = as.character(unique(train_27$class))
if(!dir.exists(data_folder)) {
  dir.create(data_folder)
}
if(!dir.exists(train_dir)) {
  dir.create(train_dir)
}
if (!dir.exists(val_dir)) {
  dir.create(val_dir)
}
if (!dir.exists(test_dir)) {
  dir.create(test_dir)
}

for(current_dir in c(train_dir, val_dir, test_dir)) {
  for (current_class in unique_classes) {
    current_folder = paste0(current_dir, current_class)
    if(!dir.exists(current_folder)) {
      dir.create(current_folder)
    }
  }
}

resized_thumbnail_dir = "E:/brca_thumbnails/all/"

for (i in 1:nrow(train_27)) {
  from_file = paste0(resized_thumbnail_dir, basename(train_27$full_path[i]))
  to_name = paste0(train_dir, train_27$class[i], "/", basename(train_27$full_path[i]))
  file.copy(from=from_file, to=to_name)
  print(i)
}

for (i in 1:nrow(val_27)) {
  from_file = paste0(resized_thumbnail_dir, basename(val_27$full_path[i]))
  to_name = paste0(val_dir, val_27$class[i], "/", basename(val_27$full_path[i]))
  file.copy(from=from_file, to=to_name)
  print(i)
}

for (i in 1:nrow(test_27)) {
  from_file = paste0(resized_thumbnail_dir, basename(test_27$full_path[i]))
  to_name = paste0(test_dir, test_27$class[i], "/", basename(test_27$full_path[i]))
  file.copy(from=from_file, to=to_name)
  print(i)
}










