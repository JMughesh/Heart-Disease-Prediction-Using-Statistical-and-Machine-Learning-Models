# Load necessary libraries
library(readxl)
library(caret)
library(car)
library(glmnet)
library(randomForest)
library(e1071)
library(arm)
library(pROC)

#load the dataset
heart <- read_excel("C:/Users/mughe/OneDrive/Desktop/projects/heart disease/heart.xlsx")

#inspect the data
str(heart)     # Check the structure of the dataset
sum(is.na(heart))     # Check for missing values

#EDA
#to understand the distribution and characteristics of the variables.
library(Hmisc)
describe(heart)
library(psych)
corr.test(x = heart$Weight, y = heart$BMI, use = "pairwise", method = "pearson")

# Convert categorical variables to factors
heart[] <- lapply(heart, function(x) if(is.character(x)) as.factor(x) else x)

# Define features (X) and target (y)
heart$Heart_Disease <- as.factor(heart$Heart_Disease)  # Ensure target is a factor

# Check for multicollinearity using VIF
vif_values <- vif(glm(Heart_Disease ~ ., data = heart, family = binomial))
print(vif_values)

# Split dataset into training (80%) and testing (20%) sets
set.seed(42)
train_index <- createDataPartition(heart$Heart_Disease, p = 0.8, list = FALSE)
train_data <- heart[train_index, ]
test_data <- heart[-train_index, ]

#EDA
#box plot
boxplot(heart$General_Health, main = "boxplot of General Health")
boxplot(heart$Exercise, main = "boxplot of exercise")
boxplot(heart$Depression, main = "boxplot of Depression")
boxplot(heart$Weight, main = "boxplot of weight")
boxplot(heart$BMI, main = "boxplot of BMI")

#QQ plot
qqnorm(heart$BMI, main = "QQ-plot of BMI")
qqnorm(heart$Weight, main = "QQ-plot of Weight")

# Function to evaluate models
evaluate_model <- function(model, test_data, type = "response") {
  pred_probs <- predict(model, test_data, type = type)
  pred_labels <- ifelse(pred_probs > 0.5, "Yes", "No")
  pred_labels <- factor(pred_labels, levels = levels(test_data$Heart_Disease))
  
  conf_matrix <- confusionMatrix(pred_labels, test_data$Heart_Disease)
  roc_curve <- roc(test_data$Heart_Disease, as.numeric(pred_probs))
  
  return(list(RSquare = ifelse("r.squared" %in% names(summary(model)), summary(model)$r.squared, NA), 
              Accuracy = conf_matrix$overall["Accuracy"], 
              ConfusionMatrix = conf_matrix, 
              ROC = roc_curve))
}

# Logistic Regression
log_model <- glm(Heart_Disease ~ ., data = train_data, family = binomial)
log_results <- evaluate_model(log_model, test_data)

# Stepwise Regression
step_model <- step(glm(Heart_Disease ~ ., data = train_data, family = binomial), 
                   direction = "both", trace = 1)  # trace = 1 to show steps
step_results <- evaluate_model(step_model, test_data)

# Decision Tree
dt_model <- rpart(Heart_Disease ~ ., data = train_data, method = "class")
dt_pred <- predict(dt_model, test_data, type = "class")
dt_conf_matrix <- confusionMatrix(dt_pred, test_data$Heart_Disease)
library(rpart)
library(rpart.plot)
rpart.plot(dt_model, main="Decision Tree Structure")

# Random Forest
rf_model <- randomForest(Heart_Disease ~ ., data = train_data)
rf_pred <- predict(rf_model, test_data)
rf_conf_matrix <- confusionMatrix(rf_pred, test_data$Heart_Disease)

# Bayesian Generalized Linear Model
bayesian_model <- bayesglm(Heart_Disease ~ ., data = train_data, family = binomial)
bayesian_results <- evaluate_model(bayesian_model, test_data)

# Support Vector Machine (SVM)
svm_model <- svm(Heart_Disease ~ ., data = train_data, probability = TRUE)
svm_pred <- predict(svm_model, test_data, probability = TRUE)
svm_probs <- attr(svm_pred, "probabilities")[,2]
svm_labels <- ifelse(svm_probs > 0.5, "Yes", "No")
svm_labels <- factor(svm_labels, levels = levels(test_data$Heart_Disease))
svm_conf_matrix <- confusionMatrix(svm_labels, test_data$Heart_Disease)

# Calculate ROC curves and AUC values
log_roc <- log_results$ROC
log_auc <- auc(log_roc)
step_roc <- step_results$ROC
step_auc <- auc(step_roc)
dt_roc <- roc(test_data$Heart_Disease, as.numeric(predict(dt_model, test_data, type = "prob")[,2]))
dt_auc <- auc(dt_roc)
rf_roc <- roc(test_data$Heart_Disease, as.numeric(predict(rf_model, test_data, type = "prob")[,2]))
rf_auc <- auc(rf_roc)
bayesian_roc <- bayesian_results$ROC
bayesian_auc <- auc(bayesian_roc)
svm_roc <- roc(test_data$Heart_Disease, svm_probs)
svm_auc <- auc(svm_roc)

# Compare ROC Curves with AUC in Legend (Reduced Size)
plot(log_roc, col = "blue", main = "ROC Curves for Models", lwd = 2, legacy.axes = TRUE, xlab = "False Positive Rate", ylab = "True Positive Rate")
plot(step_roc, col = "cyan", add = TRUE, lwd = 2)
plot(dt_roc, col = "red", add = TRUE, lwd = 2)
plot(rf_roc, col = "green", add = TRUE, lwd = 2)
plot(bayesian_roc, col = "purple", add = TRUE, lwd = 2)
plot(svm_roc, col = "orange", add = TRUE, lwd = 2)



# Add small, floating legend in bottom-left corner
legend("topleft", 
       legend = c(
         paste("Logistic (AUC =", round(log_auc, 3), ")"),
         paste("Stepwise (AUC =", round(step_auc, 3), ")"),
         paste("DT (AUC =", round(dt_auc, 3), ")"),
         paste("RF (AUC =", round(rf_auc, 3), ")"),
         paste("Bayesian (AUC =", round(bayesian_auc, 3), ")"),
         paste("SVM (AUC =", round(svm_auc, 3), ")")
       ), 
       col = c("blue","cyan", "red", "green", "purple", "orange"), 
       pch = 19,
       cex = 0.6,
       x.intersp = 0.5)

       # Print results
       list(Logistic = log_results,Stepwise = step_results, DecisionTree = dt_conf_matrix, RandomForest = rf_conf_matrix,
            Bayesian = bayesian_results, SVM = svm_conf_matrix)
       
       # Extract accuracy values from the results
       accuracies <- c(
         Logistic = log_results$Accuracy,
         Stepwise = step_results$Accuracy,
         DecisionTree = dt_conf_matrix$overall["Accuracy"],
         RandomForest = rf_conf_matrix$overall["Accuracy"],
         Bayesian = bayesian_results$Accuracy,
         SVM = svm_conf_matrix$overall["Accuracy"])
       
       # Extract AUC values (already calculated in your code)
       auc_values <- c(Logistic = log_auc,
                       Stepwise = step_auc,
                       DecisionTree = dt_auc,
                       RandomForest = rf_auc,
                       Bayesian = bayesian_auc,
                       SVM = svm_auc)
       print(auc_values)
       
       # Create a data frame to summarize model performance
       model_performance <- data.frame(
         Model = names(accuracies),
         Accuracy = accuracies,
         AUC = auc_values)
       
       # Print the performance table
       print("Model Performance Summary:")
       print(model_performance)
       
       # Identify the best model based on Accuracy
       best_accuracy_model <- model_performance$Model[which.max(model_performance$Accuracy)]
       best_accuracy_value <- max(model_performance$Accuracy)
       
       # Identify the best model based on AUC
       best_auc_model <- model_performance$Model[which.max(model_performance$AUC)]
       best_auc_value <- max(model_performance$AUC)
       
       # Print the best models
       cat("\nBest Model by Accuracy:\n")
       cat(sprintf("%s (Accuracy = %.3f)\n", best_accuracy_model, best_accuracy_value))
       
       cat("\nBest Model by AUC:\n")
       cat(sprintf("%s (AUC = %.3f)\n", best_auc_model, best_auc_value))
       
       