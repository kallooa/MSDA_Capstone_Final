library(gtools)
library(psych)

dir_loc = "/Users/aadi/Downloads/results/slide_level/"

results_dirs = dir(dir_loc, pattern="27")
#results_dirs = results_dirs[c(3,12)] #don't want to run for all, just for new ones
model_acc = data.frame(matrix(0, nrow=length(results_dirs), ncol=9))
names(model_acc) = c("model_name", "accuracy", "Recall_macro", "Precision_macro", "Kappa", "Weighted_Kappa", "AUC", "Sensitivity", "Specificity")

#load(file="/Users/aadi/Google Drive/School/MS Data Analytics/Master's Project/brca1.RData")
thumb_all = read.csv("/Users/aadi/Downloads/results/slide_level/slide_classes.csv", stringsAsFactors = FALSE)
#thumb_all$basenames = basename(thumb_all$full_path)
#thumb_all = thumb_all[!duplicated(thumb_all$basenames),]

#temp = thumb_all[,c(3,5)]
#merged = merge(x=temp, y=thumbnails_brca_color_1.h5_df, by.x="basenames", by.y="basenames")
#merged = merged[!duplicated(merged$basenames),]


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
  eval(parse(text=paste0(df_name, " = ", df_name, "[!nchar(", df_name, "$filename)<5,]")))
  eval(parse(text=paste0(df_name, "$filenames = gsub(", df_name, "$filenames, pattern='\\\\\\\\', replacement='/')")))
  eval(parse(text=paste0(df_name, "$basenames = basename(", df_name, "$filenames)")))
  eval(parse(text=paste0(df_name, " = merge(x=", df_name, ", y=thumb_all[,c(2:3)], by='basenames')")))
  eval(parse(text=paste0(df_name, " = ", df_name, "[!duplicated(", df_name, "$basenames),]")))
  eval(parse(text=paste0(df_name, " = ", df_name, "[!(", df_name, "$class %in% c('stad', 'hnsc', 'thca', 'pcpg')),]")))
  
  #organize true and predicted classes
  eval(parse(text=paste0(df_name, "$true_class = ", df_name, "$class")))
  eval(parse(text=paste0(df_name, "$predicted_idx = unlist(apply(", df_name, "[,c(4:30)], 1, function(x) {which.max(x)}))")))
  eval(parse(text=paste0("histopath_classes = unique(", df_name, "$true_class)")))
  histopath_classes = histopath_classes[!is.na(histopath_classes)]
  histopath_classes = sort(histopath_classes)
  eval(parse(text=paste0(df_name, "$predicted_class = unlist(lapply(", df_name, "$predicted_idx, function(x) {histopath_classes[x]}))")))
  eval(parse(text=paste0(df_name, "$predicted_prob = sapply(1:nrow(", df_name, "), function(x){", df_name, "[x, 3+", df_name, "$predicted_idx[x]]})")))
  #f1 = sapply(1:nrow(class_inception_color_13.h5_df), function(x){class_inception_color_13.h5_df[x, 2+class_inception_color_13.h5_df$predicted_idx[x]]})
  eval(parse(text=paste0(df_name, "$predicted_binary = ifelse(", df_name, "$true_class==", df_name, "$predicted_class, 1, 0)")))
  eval(parse(text=paste0("model_acc[i, 7:9] = fun.auc(", df_name, "$predicted_prob,", df_name, "$predicted_binary)[1:3]")))  
  
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
  #model_acc$F_score_macro[i] = macroF1
  model_acc$Recall_macro[i] = macroRecall
  model_acc$Precision_macro[i] = macroPrecision
  model_acc$Kappa[i] = kappa

  eval(parse(text=paste0("data = ", df_name, "[, c(32, 34)]")))
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
  
  eval(parse(text=paste0("roc_auc = fun.auc.ggplot(", df_name, "$predicted_prob,", df_name, "$predicted_binary, '')")))  
  
  ggsave(filename=paste0("/Users/aadi/Google Drive/School/MS Data Analytics/Master's Project/", df_name,"_ROCAUC.png"), plot=roc_auc, units="in", width=10, height=5, device="png")
  
  
}
View(model_acc)


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






library(ROCR)
fun.auc <- function(pred,obs){
  # Run the ROCR functions for AUC calculation
  ROC_perf <- performance(prediction(pred,obs),"tpr","fpr")
  ROC_sens <- performance(prediction(pred,obs),"sens","spec")
  ROC_err <- performance(prediction(pred, labels=obs),"err")
  ROC_auc <- performance(prediction(pred,obs),"auc")
  # AUC value
  AUC <- ROC_auc@y.values[[1]] # AUC
  # Mean sensitivity across all cutoffs
  x.Sens <- mean(as.data.frame(ROC_sens@y.values)[,1])
  # Mean specificity across all cutoffs
  x.Spec <- mean(as.data.frame(ROC_sens@x.values)[,1])
  # Sens-Spec table to estimate threshold cutoffs
  SS <- data.frame(SENS=as.data.frame(ROC_sens@y.values)[,1],SPEC=as.data.frame(ROC_sens@x.values)[,1])
  # Threshold cutoff with min difference between Sens and Spec
  SS_min_dif <- ROC_perf@alpha.values[[1]][which.min(abs(SS$SENS-SS$SPEC))]
  # Threshold cutoff with max sum of Sens and Spec
  SS_max_sum <- ROC_perf@alpha.values[[1]][which.max(rowSums(SS[c("SENS","SPEC")]))]
  # Min error rate
  Min_Err <- min(ROC_err@y.values[[1]])
  # Threshold cutoff resulting in min error rate
  Min_Err_Cut <- ROC_err@x.values[[1]][which(ROC_err@y.values[[1]]==Min_Err)][1]
  # Kick out the values
  round(cbind(AUC,x.Sens,x.Spec,SS_min_dif,SS_max_sum,Min_Err,Min_Err_Cut),3)
}


fun.auc.ggplot <- function(pred, obs, title) {
  # pred = predicted values
  # obs = observed values (truth)
  # title = plot title
  
  # Run the AUC calculations
  ROC_perf <- performance(prediction(pred,obs),"tpr","fpr")
  ROC_sens <- performance(prediction(pred,obs),"sens","spec")
  ROC_auc <- performance(prediction(pred,obs),"auc")
  
  # Make plot data
  plotdat <- data.frame(FP=ROC_perf@x.values[[1]],TP=ROC_perf@y.values[[1]],CUT=ROC_perf@alpha.values[[1]],POINT=NA)
  plotdat[unlist(lapply(seq(0,1,0.1),function(x){which.min(abs(plotdat$CUT-x))})),"POINT seq(0,1,0.1)"]
          
          # Plot the curve
          ggplot(plotdat, aes(x=FP,y=TP)) + 
          geom_abline(intercept=0,slope=1) +
          geom_line(lwd=1) + 
          geom_point(data=plotdat[!is.na(plotdat$POINT),], aes(x=FP,y=TP,fill=POINT), pch=21, size=3) +
          geom_text(data=plotdat[!is.na(plotdat$POINT),], aes(x=FP,y=TP,fill=POINT), label=seq(1,0,-0.1), hjust=1, vjust=0) +
          scale_fill_gradientn("Threhsold Cutoff",colours=rainbow(14)[1:11]) +
          scale_x_continuous("False Positive Rate", limits=c(0,1)) +
          scale_y_continuous("True Positive Rate", limits=c(0,1)) +
          geom_polygon(aes(x=X,y=Y), data=data.frame(X=c(0.7,1,1,0.7),Y=c(0,0,0.29,0.29)), fill="white") +
          annotate("text",x=0.97,y=0.25,label=paste("n_correct = ",sum(obs==1),sep=""),hjust=1) +
          annotate("text",x=0.97,y=0.20,label=paste("n_incorrect = ",sum(obs==0),sep=""),hjust=1) +
          annotate("text",x=0.97,y=0.15,label=paste("AUC = ",round(ROC_auc@y.values[[1]],digits=4),sep=""),hjust=1) +
          annotate("text",x=0.97,y=0.10,label=paste("Sens = ",round(mean(as.data.frame(ROC_sens@y.values)[,1]),digits=4),sep=""),hjust=1) +
          annotate("text",x=0.97,y=0.05,label=paste("Spec = ",round(mean(as.data.frame(ROC_sens@x.values)[,1]),digits=4),sep=""),hjust=1) +
          theme(legend.position="none", plot.title=element_text(vjust=2)) +
          ggtitle(title)
}

f1 = sapply(1:nrow(class_inception_color_13.h5_df), function(x){class_inception_color_13.h5_df[x, 2+class_inception_color_13.h5_df$predicted_idx[x]]})
pred_status = ifelse(class_inception_color_13.h5_df$true_class==class_inception_color_13.h5_df$predicted_class, 1, 0)


fun.auc(f1, pred_status)[1:3]
fun.auc.ggplot(f1, pred_status, "My AUC Plot")




