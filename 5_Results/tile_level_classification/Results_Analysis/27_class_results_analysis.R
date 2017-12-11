library(gtools)
library(psych)

dir_loc = "/Users/aadi/Downloads/results/"

results_dirs = dir(dir_loc, pattern="27")
#results_dirs = results_dirs[c(3,12)] #don't want to run for all, just for new ones
model_acc = data.frame(matrix(0, nrow=length(results_dirs), ncol=7))
names(model_acc) = c("model_name", "accuracy", "F_score_macro", "Recall_macro", "Precision_macro", "Kappa", "Weighted_Kappa")

for (i in 1:length(results_dirs)) {
  
  #extract data from csvs
  this_dir = results_dirs[i]
  files_in_dir = list.files(paste0(dir_loc, this_dir))
  df_name = paste0(gsub(this_dir, pattern='27_', replacement=''), "_df")
  eval(parse(text=paste0(df_name, "= read.csv(file=paste0(dir_loc, this_dir, '/', files_in_dir[1]), stringsAsFactors=FALSE)")))
  for (csv in files_in_dir[2:length(files_in_dir)]) {
    print(paste0(this_dir, " / ", csv))
    this_csv = read.csv(file=paste0(dir_loc, this_dir, '/', csv))
    eval(parse(text=paste0(df_name, "= smartbind(", df_name, ", this_csv)")))
  }

  #organize true and predicted classes
  eval(parse(text=paste0(df_name, "$true_class = unlist(lapply(", df_name, "$filenames, function(x) {strsplit(as.character(x), split='/')[[1]][4]}))")))
  eval(parse(text=paste0(df_name, "$predicted_idx = unlist(apply(", df_name, "[,c(3:29)], 1, function(x) {which.max(x)}))")))
  eval(parse(text=paste0("histopath_classes = unique(", df_name, "$true_class)")))
  histopath_classes = histopath_classes[!is.na(histopath_classes)]
  histopath_classes = sort(histopath_classes)
  eval(parse(text=paste(df_name, "$predicted_class = unlist(lapply(", df_name, "$predicted_idx, function(x) {histopath_classes[x]}))")))
  
  #calculate accuracy
  eval(parse(text=paste0("cm_", df_name, " = as.matrix(table(True=", df_name, "$true_class, Predicted=", df_name, "$predicted_class))")))
  model_acc$model_name[i] = df_name
  eval(parse(text=paste0("model_acc$accuracy[i] = sum(diag(cm_", df_name, "))/sum(cm_", df_name, ")")))
  eval(parse(text=paste0("model_acc$Weighted_Kappa[i] = psych::cohen.kappa(cm_", df_name, ")[2][[1]]")))
  eval(parse(text=paste0("n = sum(cm_", df_name, ")")))
  eval(parse(text=paste0("nc = nrow(cm_", df_name, ")")))
  eval(parse(text=paste0("diag = diag(cm_", df_name, ")")))
  eval(parse(text=paste0("rowsums = apply(cm_", df_name, ", 1, sum)")))
  eval(parse(text=paste0("colsums = apply(cm_", df_name, ", 2, sum)")))
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  accuracy = sum(diag) / n
  precision = diag / colsums
  recall = diag / rowsums
  f1 = 2 * precision * recall / (precision + recall)
  macroPrecision = mean(precision)
  macroRecall = mean(recall[recall<1])
  macroF1 = mean(f1)
  expAccuracy = sum(p*q)
  kappa = (accuracy - expAccuracy) / (1 - expAccuracy)
  model_acc$F_score_macro[i] = macroF1
  model_acc$Recall_macro[i] = macroRecall
  model_acc$Precision_macro[i] = macroPrecision
  model_acc$Kappa[i] = kappa

  eval(parse(text=paste0("data = ", df_name, "[, c(30, 32)]")))
  names(data) = c("Actual", "Predicted") 
  
  #compute frequency of actual categories
  actual = as.data.frame(table(data$Actual))
  names(actual) = c("Actual","ActualFreq")
  
  #build confusion matrix
  confusion = as.data.frame(table(data$Actual, data$Predicted))
  names(confusion) = c("Actual","Predicted","Freq")
  
  #calculate percentage of test cases based on actual frequency
  confusion = merge(confusion, actual, by=c("Actual"))
  confusion$Percent = confusion$Freq/confusion$ActualFreq*100
  
  #render plot
  # we use three different layers
  # first we draw tiles and fill color based on percentage of test cases
  tile <- ggplot() +
    geom_tile(aes(x=Actual, y=Predicted,fill=Percent),data=confusion, color="black",size=0.1) +
    labs(x="Actual",y="Predicted")
  tile = tile + 
    geom_text(aes(x=Actual,y=Predicted, label=sprintf("%.1f", Percent)),data=confusion, size=3, colour="black") +
    scale_fill_gradient(low="white",high="lightblue")
  
  # lastly we draw diagonal tiles. We use alpha = 0 so as not to hide previous layers but use size=0.3 to highlight border
  tile = tile + 
    geom_tile(aes(x=Actual,y=Predicted),data=subset(confusion, as.character(Actual)==as.character(Predicted)), color="black",size=0.3, fill="black", alpha=0) 
  
  #render
  #tile
  
  ggsave(filename=paste0("/Users/aadi/Google Drive/School/MS Data Analytics/Master's Project/", df_name,".png"), plot=tile, units="in", width=10, height=5, device="png")

  
}



# cm_10
# n = sum(cm_10) # number of instances
# nc = nrow(cm_10) # number of classes
# diag = diag(cm_10) # number of correctly classified instances per class
# rowsums = apply(cm_10, 1, sum) # number of instances per class
# colsums = apply(cm_10, 2, sum) # number of predictions per class
# p = rowsums / n # distribution of instances over the actual classes
# q = colsums / n # distribution of instances over the predicted classes
# accuracy = sum(diag) / n
# precision = diag / colsums
# recall = diag / rowsums
# f1 = 2 * precision * recall / (precision + recall)
# data.frame(precision, recall, f1)
# macroPrecision = mean(precision)
# macroRecall = mean(recall)
# macroF1 = mean(f1)
# expAccuracy = sum(p*q)
# kappa = (accuracy - expAccuracy) / (1 - expAccuracy)



##training history
training_files = list.files("/Users/aadi/Downloads/results/training_history/important_models/")
val_acc_df = data.frame(matrix(0, nrow=200, ncol=length(training_files)))

for (i in 1:length(training_files)) {
  this_csv = read.csv(paste0("/Users/aadi/Downloads/results/training_history/important_models/", training_files[i]), stringsAsFactors = FALSE)
  val_acc_df[,i] = this_csv$val_acc
  names(val_acc_df)[i] = training_files[i]
}
val_acc_df[101:200,1] = NA
val_acc_df[201:400,] = NA
val_acc_df$`27_class_inception_color_10.csv` = c(val_acc_df$`27_class_inception_color_9.csv`[1:200], val_acc_df$`27_class_inception_color_10.csv`[1:200])
names(val_acc_df) = c("de novo architecture, 10CL, 411FCN, adadelta", "Inception v3, 100000FCN, adadelta+adagrad", "Inception v3, 1000FCN, adadelta", "Inception v3, 10000FCN, adadelta", "Inception v3, 100000FCN, adadelta", "VGG19, 10000FCN, adadelta")
val_acc_df = val_acc_df[,c(1,6,3,4,5,2)]

library(ggplot2)
library(reshape2)
val_acc_df2 = melt(val_acc_df)
val_acc_df2$Epoch = rep(seq(1:400), length(training_files))
val_acc_df3 = val_acc_df2[complete.cases(val_acc_df2$value),]
names(val_acc_df3) = c("Model", "Validation_Accuracy", "Epoch")

ggplot(data=val_acc_df3, aes(x=Epoch, y=Validation_Accuracy, col=Model)) + geom_line() + labs(title="Training History: Validation Accuracy vs Epoch") + theme(plot.title = element_text(hjust = 0.5))




