library(stringr)

drive = "G:/"
data_folder = paste0(drive, "data/")
train_dir = paste0(data_folder, "train/")
val_dir = paste0(data_folder, "val/")
test_dir = paste0(data_folder, "test/")

#proportions of 20 million total
train_proportion = 0.2
val_proportion = 0.05
test_proportion = 0.05

files = list.files("G:/tcga_imgs/tiles/", recursive = TRUE, full.names = TRUE, pattern = "jpg")
files = data.frame(files, stringsAsFactors = FALSE)

save(files, file="G:/all_imgs.RData")

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

train$class = str_extract(train$train, pattern="\\/\\/.+[a-z]{1,4}\\/")
train$class = str_extract(train$class, pattern="[a-z]{1,4}")
val$class = str_extract(val$val, pattern="\\/\\/.+[a-z]{1,4}\\/")
val$class = str_extract(val$class, pattern="[a-z]{1,4}")
test$class = str_extract(test$test, pattern="\\/\\/.+[a-z]{1,4}\\/")
test$class = str_extract(test$class, pattern="[a-z]{1,4}")

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
  file.rename(from=train$train[i], to=to_name)
  print(i)
}

for (i in 1:nrow(val)) {
  to_name = paste0(val_dir, val$class[i], "/", basename(val$val[i]))
  file.rename(from=val$val[i], to=to_name)
  print(i)
}

for (i in 1:nrow(test)) {
  to_name = paste0(test_dir, test$class[i], "/", basename(test$test[i]))
  file.rename(from=test$test[i], to=to_name)
  print(i)
}

