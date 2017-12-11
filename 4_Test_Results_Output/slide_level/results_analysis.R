library(stringr)

thumb_results = read.csv("D:/results/slide_level/thumbnails_brca_color_1.h5/brca_dxpm_all_1_results.csv", stringsAsFactors = FALSE)
if (thumb_results[1,1]==0) {
  thumb_results = thumb_results[,-1]
}


thumb_results$true_class = str_extract(thumb_results$filename, pattern="DX|(B|M|T)S")
thumb_results$true_class[thumb_results$true_class!="DX"] = "PM"


thumb_results$predicted = ""


thumb_results$predicted_idx = apply(thumb_results[,2:3], 1, function(x){which.max(x)})
thumb_results$predicted[thumb_results$predicted_idx==1] = "DX"
thumb_results$predicted[thumb_results$predicted_idx==2] = "PM"
thumb_results$predicted_prob = apply(thumb_results[,2:3], 1, max)

prop.table(table(actual=thumb_results$true_class, predicted=thumb_results$predicted))
