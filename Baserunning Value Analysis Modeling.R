# Streamlined Baseball Analysis - Separate Target-Specific Models
# Enhanced to include coefficients alongside importance values

# ---- ENHANCED CROSS-MODEL COMPARISON WITH COEFFICIENTS ----
generate_cross_model_comparison <- function(results) {
  if (length(results) < 2) {
    cat("⚠️ Need at least 2 models for cross-comparison\n")
    return(NULL)
  }
  
  cat("\n", rep("🔍", 40), "\n")
  cat("CROSS-MODEL FEATURE COMPARISON WITH COEFFICIENTS\n")
  cat(rep("🔍", 40), "\n\n")
  
  # Get all features from both models
  target_names <- names(results)
  all_features <- list()
  
  for (target in target_names) {
    if (!is.null(results[[target]]$importance)) {
      all_features[[target]] <- results[[target]]$importance
    }
  }
  
  if (length(all_features) < 2) {
    cat("⚠️ Insufficient feature data for comparison\n")
    return(NULL)
  }
  
  # Find common features
  feature_lists <- lapply(all_features, function(x) x$Feature)
  common_features <- Reduce(intersect, feature_lists)
  
  cat("🤝 Features common to both models:", length(common_features), "\n")
  
  if (length(common_features) > 0) {
    # Create comparison table
    target1 <- target_names[1]
    target2 <- target_names[2]
    
    imp1 <- all_features[[target1]]
    imp2 <- all_features[[target2]]
    
    comparison_data <- data.frame(
      Feature = common_features,
      RSP_Importance = imp1$Avg_Importance[match(common_features, imp1$Feature)],
      RPG_Importance = imp2$Avg_Importance[match(common_features, imp2$Feature)]
    )
    
    # Add coefficient information if available
    if ("linear_Coefficient" %in% names(imp1)) {
      comparison_data$RSP_Coefficient <- imp1$linear_Coefficient[match(common_features, imp1$Feature)]
      comparison_data$RSP_p_Value <- imp1$linear_p_Value[match(common_features, imp1$Feature)]
    }
    
    if ("linear_Coefficient" %in% names(imp2)) {
      comparison_data$RPG_Coefficient <- imp2$linear_Coefficient[match(common_features, imp2$Feature)]
      comparison_data$RPG_p_Value <- imp2$linear_p_Value[match(common_features, imp2$Feature)]
    }
    
    comparison_data$Avg_Importance <- (comparison_data$RSP_Importance + comparison_data$RPG_Importance) / 2
    comparison_data$Importance_Diff <- abs(comparison_data$RSP_Importance - comparison_data$RPG_Importance)
    
    # Sort by average importance
    comparison_data <- comparison_data[order(-comparison_data$Avg_Importance), ]
    
    # Display with coefficients if available
    if ("RSP_Coefficient" %in% names(comparison_data) && "RPG_Coefficient" %in% names(comparison_data)) {
      cat("\n📊 TOP 15 COMMON FEATURES WITH COEFFICIENTS:\n")
      cat(sprintf("%-35s %8s %8s %10s %10s %8s %8s\n", 
                  "Feature", "RSP_Imp", "RPG_Imp", "RSP_Coef", "RPG_Coef", "RSP_p", "RPG_p"))
      cat(rep("─", 95), "\n")
      
      for (i in 1:min(15, nrow(comparison_data))) {
        cat(sprintf("%-35s %8.3f %8.3f %10.4f %10.4f %8.3f %8.3f\n",
                    substr(comparison_data$Feature[i], 1, 35),
                    comparison_data$RSP_Importance[i],
                    comparison_data$RPG_Importance[i],
                    ifelse(is.na(comparison_data$RSP_Coefficient[i]), 0, comparison_data$RSP_Coefficient[i]),
                    ifelse(is.na(comparison_data$RPG_Coefficient[i]), 0, comparison_data$RPG_Coefficient[i]),
                    ifelse(is.na(comparison_data$RSP_p_Value[i]), 1, comparison_data$RSP_p_Value[i]),
                    ifelse(is.na(comparison_data$RPG_p_Value[i]), 1, comparison_data$RPG_p_Value[i])))
      }
    } else {
      # Original importance-only display
      cat("\n📊 TOP 15 COMMON FEATURES COMPARISON:\n")
      cat(sprintf("%-45s %12s %12s %12s %8s\n", "Feature", "RSP", "RPG", "Average", "Diff"))
      cat(rep("-", 90), "\n")
      
      for (i in 1:min(15, nrow(comparison_data))) {
        indicator <- ifelse(comparison_data$RSP_Importance[i] > comparison_data$RPG_Importance[i], "→RSP", "→RPG")
        
        cat(sprintf("%-45s %12.4f %12.4f %12.4f %8.4f %s\n", 
                    substr(comparison_data$Feature[i], 1, 45),
                    comparison_data$RSP_Importance[i],
                    comparison_data$RPG_Importance[i],
                    comparison_data$Avg_Importance[i],
                    comparison_data$Importance_Diff[i],
                    indicator))
      }
    }
    
    # Effect size analysis for coefficients
    if ("RSP_Coefficient" %in% names(comparison_data)) {
      cat("\n🔍 COEFFICIENT EFFECT SIZE ANALYSIS:\n")
      cat(rep("-", 90), "\n")
      
      # Features with largest positive effects
      positive_rsp <- comparison_data[!is.na(comparison_data$RSP_Coefficient) & comparison_data$RSP_Coefficient > 0, ]
      positive_rpg <- comparison_data[!is.na(comparison_data$RPG_Coefficient) & comparison_data$RPG_Coefficient > 0, ]
      
      if (nrow(positive_rsp) > 0) {
        positive_rsp <- positive_rsp[order(-positive_rsp$RSP_Coefficient), ]
        cat("📈 Largest positive effects for RSP:\n")
        for (i in 1:min(3, nrow(positive_rsp))) {
          cat(sprintf("   • %-40s (%.4f)\n", positive_rsp$Feature[i], positive_rsp$RSP_Coefficient[i]))
        }
      }
      
      if (nrow(positive_rpg) > 0) {
        positive_rpg <- positive_rpg[order(-positive_rpg$RPG_Coefficient), ]
        cat("\n⚾ Largest positive effects for RPG:\n")
        for (i in 1:min(3, nrow(positive_rpg))) {
          cat(sprintf("   • %-40s (%.4f)\n", positive_rpg$Feature[i], positive_rpg$RPG_Coefficient[i]))
        }
      }
      
      # Features with largest negative effects
      negative_rsp <- comparison_data[!is.na(comparison_data$RSP_Coefficient) & comparison_data$RSP_Coefficient < 0, ]
      negative_rpg <- comparison_data[!is.na(comparison_data$RPG_Coefficient) & comparison_data$RPG_Coefficient < 0, ]
      
      if (nrow(negative_rsp) > 0) {
        negative_rsp <- negative_rsp[order(negative_rsp$RSP_Coefficient), ]
        cat("\n📉 Largest negative effects for RSP:\n")
        for (i in 1:min(3, nrow(negative_rsp))) {
          cat(sprintf("   • %-40s (%.4f)\n", negative_rsp$Feature[i], negative_rsp$RSP_Coefficient[i]))
        }
      }
      
      if (nrow(negative_rpg) > 0) {
        negative_rpg <- negative_rpg[order(negative_rpg$RPG_Coefficient), ]
        cat("\n📉 Largest negative effects for RPG:\n")
        for (i in 1:min(3, nrow(negative_rpg))) {
          cat(sprintf("   • %-40s (%.4f)\n", negative_rpg$Feature[i], negative_rpg$RPG_Coefficient[i]))
        }
      }
    }
    
    # Identify features that are much more important for one model vs another
    cat("\n🔍 FEATURE SPECIALIZATION ANALYSIS:\n")
    cat(rep("-", 90), "\n")
    
    # Features more important for RSP
    rsp_specialized <- comparison_data[comparison_data$RSP_Importance > comparison_data$RPG_Importance + 0.1, ]
    if (nrow(rsp_specialized) > 0) {
      cat("📈 Features MORE important for Run Scoring Percentage:\n")
      for (i in 1:min(5, nrow(rsp_specialized))) {
        cat(sprintf("   • %-40s (RSP: %.3f vs RPG: %.3f)\n", 
                    rsp_specialized$Feature[i],
                    rsp_specialized$RSP_Importance[i],
                    rsp_specialized$RPG_Importance[i]))
      }
    }
    
    # Features more important for RPG
    rpg_specialized <- comparison_data[comparison_data$RPG_Importance > comparison_data$RSP_Importance + 0.1, ]
    if (nrow(rpg_specialized) > 0) {
      cat("\n⚾ Features MORE important for Runs Per Game:\n")
      for (i in 1:min(5, nrow(rpg_specialized))) {
        cat(sprintf("   • %-40s (RPG: %.3f vs RSP: %.3f)\n", 
                    rpg_specialized$Feature[i],
                    rpg_specialized$RPG_Importance[i],
                    rpg_specialized$RSP_Importance[i]))
      }
    }
    
    # Most consistent features (similar importance across both)
    consistent_features <- comparison_data[comparison_data$Importance_Diff < 0.05 & comparison_data$Avg_Importance > 0.1, ]
    if (nrow(consistent_features) > 0) {
      cat("\n🎯 UNIVERSALLY IMPORTANT Features (similar importance for both):\n")
      consistent_features <- consistent_features[order(-consistent_features$Avg_Importance), ]
      for (i in 1:min(5, nrow(consistent_features))) {
        cat(sprintf("   • %-40s (Avg: %.3f, Diff: %.3f)\n", 
                    consistent_features$Feature[i],
                    consistent_features$Avg_Importance[i],
                    consistent_features$Importance_Diff[i]))
      }
    }
    
    return(comparison_data)
  }
  
  return(NULL)
}# Streamlined Baseball Analysis - Separate Target-Specific Models
# Enhanced to include coefficients alongside importance values

# ---- PACKAGE LOADING ----
required_packages <- c("dplyr", "randomForest", "xgboost", "ggplot2", "caret", "glmnet", "broom")

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# ---- DATA LOADING AND CLEANING ----
file_path <- "C:/Users/henbe/Desktop/Projects/Baserunning Analysis/Baserunning CSVs 7.21.25/baserunning_stats_2016_2024_expanded.csv"

if (!file.exists(file_path)) {
  stop("CSV file not found at: ", file_path)
}

df <- read.csv(file_path, stringsAsFactors = FALSE)
cat("Data loaded. Dimensions:", nrow(df), "x", ncol(df), "\n")

# Clean and convert to numeric
df[] <- lapply(df, function(col) {
  if (is.character(col)) col <- gsub("[,%]", "", col)
  suppressWarnings(as.numeric(col))
})

# Remove rows with all NAs and impute remaining NAs with median
df <- df[rowSums(!is.na(df)) > 0, ]
df[] <- lapply(df, function(col) {
  if (is.numeric(col)) col[is.na(col)] <- median(col, na.rm = TRUE)
  col
})

# ---- ADVANCED FEATURE SELECTION WITH BORUTA ----
load_advanced_packages <- function() {
  advanced_packages <- c("Boruta", "car", "corrplot")
  for (pkg in advanced_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

load_advanced_packages()

# Enhanced feature selection with multiple methods
prepare_data_for_target <- function(df, target_name, exclude_circular = TRUE, use_boruta = TRUE) {
  if (!target_name %in% names(df)) {
    stop("Target variable not found: ", target_name)
  }
  
  cat("\n🎯 Advanced feature selection for", target_name, "\n")
  
  # Define circular predictors to exclude
  circular_predictors <- c("Run_Scoring_Percentage", "Runs_Per_Game")
  
  if (exclude_circular) {
    exclude_vars <- circular_predictors[circular_predictors != target_name]
    df_clean <- df[, !names(df) %in% exclude_vars, drop = FALSE]
    cat("🚫 Removed circular predictor(s):", paste(exclude_vars, collapse = ", "), "\n")
  } else {
    df_clean <- df
  }
  
  # Keep only numeric columns
  numeric_cols <- sapply(df_clean, is.numeric)
  df_clean <- df_clean[, numeric_cols, drop = FALSE]
  
  # Step 1: Remove zero/low variance features (protect Year)
  cat("📊 Checking variance for", ncol(df_clean), "features...\n")
  variance_scores <- sapply(df_clean, function(x) {
    if (is.numeric(x)) {
      return(var(x, na.rm = TRUE))
    } else {
      return(0)
    }
  })
  
  # Debug: show variance scores
  cat("📈 Variance summary - Min:", round(min(variance_scores, na.rm = TRUE), 6), 
      "Max:", round(max(variance_scores, na.rm = TRUE), 2), "\n")
  
  variance_check <- variance_scores > 0.001
  variance_check[is.na(variance_check)] <- FALSE
  
  # Protect Year feature
  if ("Year" %in% names(df_clean) && !variance_check["Year"]) {
    variance_check["Year"] <- TRUE
    cat("🛡️ Protected Year from variance filtering\n")
  }
  
  # Ensure we're only selecting columns that exist
  valid_cols <- names(variance_check)[variance_check & names(variance_check) %in% names(df_clean)]
  
  if (length(valid_cols) == 0) {
    cat("⚠️ No features passed variance check, keeping all numeric columns\n")
    valid_cols <- names(df_clean)[sapply(df_clean, is.numeric)]
  }
  
  df_clean <- df_clean[, valid_cols, drop = FALSE]
  cat("✅ Kept", ncol(df_clean), "features after variance filtering\n")
  
  # Step 2: Remove highly correlated features (protect Year)
  if (ncol(df_clean) > 2) {
    cat("🔍 Checking correlations for", ncol(df_clean), "features...\n")
    
    # Handle any remaining non-numeric columns
    numeric_cols <- sapply(df_clean, is.numeric)
    df_clean <- df_clean[, numeric_cols, drop = FALSE]
    
    if (ncol(df_clean) > 1) {
      cor_matrix <- cor(df_clean, use = "complete.obs")
      cor_matrix[is.na(cor_matrix)] <- 0
      
      # Find highly correlated features
      high_cor <- tryCatch({
        caret::findCorrelation(cor_matrix, cutoff = 0.95, names = TRUE)
      }, error = function(e) {
        cat("⚠️ Correlation analysis failed:", e$message, "\n")
        character(0)
      })
      
      # Protect Year
      if ("Year" %in% high_cor) {
        high_cor <- high_cor[high_cor != "Year"]
        cat("🛡️ Protected Year from correlation removal\n")
      }
      
      if (length(high_cor) > 0) {
        df_clean <- df_clean[, !names(df_clean) %in% high_cor, drop = FALSE]
        cat("🗑️ Removed", length(high_cor), "highly correlated features\n")
      } else {
        cat("✅ No highly correlated features found\n")
      }
    }
  }
  
  # Step 3: VIF filtering (protect Year)
  df_clean <- vif_filter_with_year(df_clean, target_name)
  
  # Step 4: Boruta feature selection (if enabled)
  if (use_boruta && ncol(df_clean) > 2) {
    df_clean <- run_boruta_selection_with_year(df_clean, target_name)
  }
  
  cat("✅ Final feature count for", target_name, ":", ncol(df_clean) - 1, "\n")
  return(df_clean)
}

# VIF filtering with Year protection
vif_filter_with_year <- function(df, target_name) {
  if (!target_name %in% names(df)) return(df)
  
  complete_rows <- complete.cases(df)
  if (sum(complete_rows) == 0) return(df)
  
  df_complete <- df[complete_rows, ]
  if (nrow(df_complete) < ncol(df_complete)) return(df)
  
  tryCatch({
    feature_cols <- names(df_complete)[!names(df_complete) %in% c(target_name, "Year")]
    
    if (length(feature_cols) > 1) {
      formula_str <- paste(target_name, "~", paste(feature_cols, collapse = " + "))
      temp_model <- lm(as.formula(formula_str), data = df_complete)
      
      if (length(feature_cols) > 1) {
        vif_values <- car::vif(temp_model)
        high_vif <- names(vif_values[vif_values > 10])
        
        if (length(high_vif) > 0) {
          df <- df[, !(names(df) %in% high_vif)]
          cat("🔧 Removed", length(high_vif), "high VIF features\n")
        }
      }
    }
    
    # Always preserve Year if it exists
    if ("Year" %in% names(df_complete) && !"Year" %in% names(df)) {
      df$Year <- df_complete$Year[1:nrow(df)]
      cat("🛡️ Year column re-added after VIF filtering\n")
    }
    
    return(df)
  }, error = function(e) {
    cat("⚠️ VIF filtering failed:", e$message, "- using original data\n")
    return(df)
  })
}

# Boruta feature selection with Year protection
run_boruta_selection_with_year <- function(df, target_name) {
  if (!target_name %in% names(df)) {
    cat("⚠️ Target variable not found for Boruta selection\n")
    return(df)
  }
  
  complete_target <- !is.na(df[[target_name]])
  if (sum(complete_target) == 0) {
    cat("⚠️ No complete cases for target variable\n")
    return(df)
  }
  
  df_clean <- df[complete_target, ]
  valid_cols <- sapply(df_clean, function(x) is.numeric(x) && var(x, na.rm = TRUE) > 0)
  
  # Always include Year if it exists
  if ("Year" %in% names(df_clean) && !valid_cols["Year"]) {
    valid_cols["Year"] <- TRUE
    cat("🛡️ Year included in Boruta despite low variance\n")
  }
  
  df_clean <- df_clean[, valid_cols, drop = FALSE]
  
  if (!target_name %in% names(df_clean)) {
    cat("⚠️ Target variable removed during cleaning\n")
    return(df_clean)
  }
  
  tryCatch({
    cat("🔍 Running Boruta feature selection...\n")
    
    boruta_result <- Boruta(as.formula(paste(target_name, "~ .")), 
                            data = df_clean, doTrace = 0, maxRuns = 100)
    final_decision <- getSelectedAttributes(boruta_result, withTentative = TRUE)
    
    # Check Year status
    if ("Year" %in% names(df_clean)) {
      year_status <- boruta_result$finalDecision["Year"]
      cat("📅 Year feature Boruta status:", as.character(year_status), "\n")
      
      # Force include Year if not selected
      if (!"Year" %in% final_decision && "Year" %in% names(df_clean)) {
        final_decision <- c(final_decision, "Year")
        cat("🛡️ Year manually preserved for temporal modeling\n")
      }
    }
    
    if (length(final_decision) == 0) {
      cat("⚠️ No features selected by Boruta, keeping all\n")
      return(df_clean)
    }
    
    cat("✅ Boruta selected", length(final_decision), "features\n")
    return(df_clean[, c(final_decision, target_name), drop = FALSE])
    
  }, error = function(e) {
    cat("⚠️ Boruta failed:", e$message, "- using correlation-filtered data\n")
    return(df_clean)
  })
}

# ---- ENHANCED STACKED ENSEMBLE ----
stack_models <- function(x_train, y_train, x_test, y_test, target_name) {
  cat("\n🏗️ Building stacked ensemble for", target_name, "...\n")
  
  set.seed(123)
  n <- nrow(x_train)
  
  # Create stratified folds for cross-validation
  folds <- caret::createFolds(y_train, k = 5)
  
  # Initialize out-of-fold prediction matrix
  oof_preds <- data.frame(
    linear = rep(NA, n),
    rf = rep(NA, n),
    xgb = rep(NA, n),
    ridge = rep(NA, n)
  )
  
  # Generate out-of-fold predictions
  cat("📊 Generating out-of-fold predictions...\n")
  for (i in seq_along(folds)) {
    fold <- folds[[i]]
    train_idx <- setdiff(1:n, fold)
    
    # Linear regression
    tryCatch({
      train_df <- data.frame(x_train[train_idx, , drop = FALSE])
      val_df <- data.frame(x_train[fold, , drop = FALSE])
      colnames(val_df) <- colnames(train_df)
      
      lm_fit <- lm(y_train[train_idx] ~ ., data = train_df)
      oof_preds$linear[fold] <- predict(lm_fit, newdata = val_df)
    }, error = function(e) {
      cat("⚠️ Linear model failed in fold", i, "\n")
    })
    
    # Ridge regression (with better error handling)
    tryCatch({
      if (!require(glmnet, quietly = TRUE)) {
        stop("glmnet package not available for Ridge regression")
      }
      ridge_fit <- cv.glmnet(x_train[train_idx, , drop = FALSE], y_train[train_idx], 
                             alpha = 0, nfolds = 3)
      oof_preds$ridge[fold] <- as.vector(predict(ridge_fit, 
                                                 x_train[fold, , drop = FALSE], 
                                                 s = "lambda.min"))
    }, error = function(e) {
      cat("⚠️ Ridge model failed in fold", i, ":", e$message, "\n")
    })
    
    # Random forest
    tryCatch({
      rf_fit <- randomForest(x = x_train[train_idx, , drop = FALSE], 
                             y = y_train[train_idx], ntree = 200)
      oof_preds$rf[fold] <- predict(rf_fit, newdata = x_train[fold, , drop = FALSE])
    }, error = function(e) {
      cat("⚠️ Random Forest failed in fold", i, "\n")
    })
    
    # XGBoost
    tryCatch({
      dtrain <- xgb.DMatrix(data = x_train[train_idx, , drop = FALSE], 
                            label = y_train[train_idx])
      dval <- xgb.DMatrix(data = x_train[fold, , drop = FALSE])
      
      # Quick parameter search for each fold
      xgb_fit <- xgboost(
        data = dtrain,
        nrounds = 100,
        max_depth = 6,
        eta = 0.1,
        objective = "reg:squarederror",
        verbose = 0
      )
      
      oof_preds$xgb[fold] <- predict(xgb_fit, newdata = dval)
    }, error = function(e) {
      cat("⚠️ XGBoost failed in fold", i, "\n")
    })
  }
  
  # Remove columns with too many NAs
  valid_models <- sapply(oof_preds, function(x) sum(!is.na(x)) > n * 0.7)
  oof_preds <- oof_preds[, valid_models, drop = FALSE]
  
  if (ncol(oof_preds) == 0) {
    cat("❌ No valid out-of-fold predictions generated\n")
    return(NULL)
  }
  
  cat("✅ Generated OOF predictions for", ncol(oof_preds), "models\n")
  
  # Train meta-model on out-of-fold predictions
  meta_data <- cbind(oof_preds, y = y_train)
  complete_rows <- complete.cases(meta_data)
  meta_data <- meta_data[complete_rows, ]
  
  if (nrow(meta_data) < 10) {
    cat("❌ Insufficient complete cases for meta-model\n")
    return(NULL)
  }
  
  # Fit meta-model (regularized linear regression)
  meta_model <- NULL
  tryCatch({
    if (!require(glmnet, quietly = TRUE)) {
      # Fallback to simple linear regression if glmnet not available
      meta_model <- lm(y ~ ., data = meta_data)
      cat("✅ Meta-model trained with linear regression (glmnet not available)\n")
    } else {
      meta_model <- cv.glmnet(as.matrix(meta_data[, -ncol(meta_data)]), 
                              meta_data$y, alpha = 0.5, nfolds = 5)  # Elastic net
      cat("✅ Meta-model trained with elastic net regression\n")
    }
  }, error = function(e) {
    cat("⚠️ Elastic net failed, trying simple linear regression:", e$message, "\n")
    tryCatch({
      meta_model <<- lm(y ~ ., data = meta_data)
      cat("✅ Meta-model trained with linear regression fallback\n")
    }, error = function(e2) {
      cat("❌ All meta-model approaches failed:", e2$message, "\n")
      return(NULL)
    })
  })
  
  # If meta-model training failed completely, return NULL
  if (is.null(meta_model)) {
    cat("❌ Meta-model training failed completely\n")
    return(NULL)
  }
  
  # Retrain base learners on full training data
  cat("🔄 Retraining base learners on full data...\n")
  base_models <- list()
  test_preds <- data.frame(row.names = 1:nrow(x_test))
  
  # Retrain each model that had valid OOF predictions
  if ("linear" %in% names(oof_preds)) {
    tryCatch({
      train_df_full <- data.frame(x_train)
      test_df_full <- data.frame(x_test)
      colnames(test_df_full) <- colnames(train_df_full)
      
      lm_full <- lm(y_train ~ ., data = train_df_full)
      test_preds$linear <- predict(lm_full, newdata = test_df_full)
      base_models$linear <- lm_full
    }, error = function(e) {
      cat("⚠️ Failed to retrain linear model\n")
    })
  }
  
  if ("ridge" %in% names(oof_preds)) {
    tryCatch({
      if (!require(glmnet, quietly = TRUE)) {
        stop("glmnet package not available")
      }
      ridge_full <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 5)
      test_preds$ridge <- as.vector(predict(ridge_full, x_test, s = "lambda.min"))
      base_models$ridge <- ridge_full
    }, error = function(e) {
      cat("⚠️ Failed to retrain ridge model:", e$message, "\n")
    })
  }
  
  if ("rf" %in% names(oof_preds)) {
    tryCatch({
      rf_full <- randomForest(x = x_train, y = y_train, ntree = 200)
      test_preds$rf <- predict(rf_full, newdata = x_test)
      base_models$rf <- rf_full
    }, error = function(e) {
      cat("⚠️ Failed to retrain random forest\n")
    })
  }
  
  if ("xgb" %in% names(oof_preds)) {
    tryCatch({
      dtrain_full <- xgb.DMatrix(data = x_train, label = y_train)
      dtest_full <- xgb.DMatrix(data = x_test)
      
      xgb_full <- xgboost(
        data = dtrain_full,
        nrounds = 100,
        max_depth = 6,
        eta = 0.1,
        objective = "reg:squarederror",
        verbose = 0
      )
      
      test_preds$xgb <- predict(xgb_full, newdata = dtest_full)
      base_models$xgb <- xgb_full
    }, error = function(e) {
      cat("⚠️ Failed to retrain XGBoost\n")
    })
  }
  
  # Get final stacked predictions
  valid_test_preds <- test_preds[, names(test_preds) %in% names(oof_preds), drop = FALSE]
  
  if (ncol(valid_test_preds) == 0) {
    cat("❌ No valid test predictions for ensemble\n")
    return(NULL)
  }
  
  # Handle missing predictions by using mean
  for (col in names(valid_test_preds)) {
    na_idx <- is.na(valid_test_preds[[col]])
    if (any(na_idx)) {
      valid_test_preds[na_idx, col] <- mean(valid_test_preds[[col]], na.rm = TRUE)
    }
  }
  
  # Generate final predictions based on meta-model type
  stacked_pred <- tryCatch({
    if (inherits(meta_model, "cv.glmnet")) {
      # Use glmnet prediction
      as.vector(predict(meta_model, as.matrix(valid_test_preds), s = "lambda.min"))
    } else {
      # Use linear model prediction
      predict(meta_model, newdata = valid_test_preds)
    }
  }, error = function(e) {
    cat("❌ Final prediction failed:", e$message, "\n")
    # Fallback to simple average
    rowMeans(as.matrix(valid_test_preds), na.rm = TRUE)
  })
  
  # Calculate metrics
  rmse <- sqrt(mean((stacked_pred - y_test)^2, na.rm = TRUE))
  r2 <- cor(stacked_pred, y_test, use = "complete.obs")^2
  mae <- mean(abs(stacked_pred - y_test), na.rm = TRUE)
  
  metrics <- data.frame(
    RMSE = rmse,
    R2 = r2,
    MAE = mae
  )
  
  cat("🎯 Stacked ensemble metrics - RMSE:", round(rmse, 4), "R²:", round(r2, 4), "\n")
  
  return(list(
    predictions = stacked_pred,
    metrics = metrics,
    meta_model = meta_model,
    base_models = base_models,
    oof_predictions = oof_preds
  ))
}

fit_models <- function(x_train, y_train, x_test, y_test) {
  results <- list()
  
  # Linear Model
  tryCatch({
    lm_model <- lm(y_train ~ ., data = data.frame(x_train))
    lm_pred <- predict(lm_model, data.frame(x_test))
    results$linear <- list(pred = lm_pred, model = lm_model)
  }, error = function(e) {
    cat("Linear model failed:", e$message, "\n")
  })
  
  # Random Forest
  tryCatch({
    rf_model <- randomForest(x = x_train, y = y_train, ntree = 300)
    rf_pred <- predict(rf_model, x_test)
    results$rf <- list(pred = rf_pred, model = rf_model)
  }, error = function(e) {
    cat("Random Forest failed:", e$message, "\n")
  })
  
  # XGBoost
  tryCatch({
    # Simple parameter grid
    ctrl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
    xgb_grid <- expand.grid(
      nrounds = c(100, 200),
      max_depth = c(3, 6),
      eta = c(0.1, 0.3),
      gamma = 0,
      colsample_bytree = 0.8,
      min_child_weight = 1,
      subsample = 0.8
    )
    
    xgb_model <- train(
      x = x_train, y = y_train,
      method = "xgbTree",
      tuneGrid = xgb_grid,
      trControl = ctrl,
      verbose = FALSE
    )
    
    xgb_pred <- predict(xgb_model, x_test)
    results$xgb <- list(pred = xgb_pred, model = xgb_model)
  }, error = function(e) {
    cat("XGBoost failed:", e$message, "\n")
  })
  
  return(results)
}

# ---- EVALUATION FUNCTIONS ----
evaluate_predictions <- function(predictions, actual, model_names = names(predictions)) {
  results <- data.frame(
    Model = model_names,
    RMSE = sapply(predictions, function(pred) sqrt(mean((actual - pred)^2, na.rm = TRUE))),
    R2 = sapply(predictions, function(pred) cor(actual, pred, use = "complete.obs")^2),
    MAE = sapply(predictions, function(pred) mean(abs(actual - pred), na.rm = TRUE))
  )
  
  results <- results[order(results$RMSE), ]
  return(results)
}

# ---- DIAGNOSTIC FUNCTIONS ----
plot_diagnostics <- function(predictions, actual, model_name) {
  if (!require(ggplot2, quietly = TRUE)) return(NULL)
  
  residuals <- actual - predictions
  
  p1 <- ggplot(data.frame(fitted = predictions, residuals = residuals), 
               aes(x = fitted, y = residuals)) +
    geom_point(alpha = 0.6) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    geom_smooth(se = FALSE, color = "blue") +
    labs(title = paste("Residuals vs Fitted -", model_name),
         x = "Fitted Values", y = "Residuals") +
    theme_minimal()
  
  p2 <- ggplot(data.frame(actual = actual, predicted = predictions), 
               aes(x = actual, y = predicted)) +
    geom_point(alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = paste("Actual vs Predicted -", model_name),
         x = "Actual", y = "Predicted") +
    theme_minimal()
  
  return(list(residuals = p1, actual_pred = p2))
}

assess_model_quality <- function(predictions, actual, model_name) {
  residuals <- actual - predictions
  mean_residual <- mean(residuals, na.rm = TRUE)
  
  # Normality test (if sample size is reasonable)
  normality_p <- NA
  if (length(residuals) >= 3 && length(residuals) <= 5000) {
    tryCatch({
      normality_p <- shapiro.test(residuals)$p.value
    }, error = function(e) {
      normality_p <<- NA
    })
  }
  
  cat("\n", model_name, "Quality Assessment:\n")
  cat("- Bias (mean residual):", round(mean_residual, 4), 
      ifelse(abs(mean_residual) < 0.01, "✓", "⚠"), "\n")
  
  if (!is.na(normality_p)) {
    cat("- Normality (p-value):", round(normality_p, 4), 
        ifelse(normality_p > 0.05, "✓", "⚠"), "\n")
  }
  
  # Overall assessment
  quality_checks <- sum(c(
    abs(mean_residual) < 0.01,
    !is.na(normality_p) && normality_p > 0.05
  ), na.rm = TRUE)
  
  overall <- c("POOR", "FAIR", "GOOD")[quality_checks + 1]
  cat("- Overall Quality:", overall, "\n")
}

# ---- ENHANCED FEATURE IMPORTANCE FUNCTIONS ----
get_feature_importance <- function(model_results, target_name) {
  importance_list <- list()
  
  for (model_name in names(model_results)) {
    model <- model_results[[model_name]]$model
    
    if (model_name == "linear") {
      # Extract both importance and coefficients with full statistical info
      model_summary <- summary(model)
      coefs <- model_summary$coefficients
      
      # Remove intercept if present
      if ("(Intercept)" %in% rownames(coefs)) {
        coefs <- coefs[-1, , drop = FALSE]
      }
      
      # Get confidence intervals with error handling
      conf_int <- tryCatch({
        ci <- confint(model)
        if ("(Intercept)" %in% rownames(ci)) {
          ci <- ci[-1, , drop = FALSE]
        }
        ci
      }, error = function(e) {
        # If confint fails, create manual CI using std errors
        se <- coefs[, "Std. Error"]
        if ("(Intercept)" %in% names(se)) {
          se <- se[-which(names(se) == "(Intercept)")]
        }
        est <- coefs[, "Estimate"]
        if ("(Intercept)" %in% names(est)) {
          est <- est[-which(names(est) == "(Intercept)")]
        }
        
        ci_manual <- matrix(nrow = length(est), ncol = 2)
        ci_manual[, 1] <- est - 1.96 * se  # Lower CI
        ci_manual[, 2] <- est + 1.96 * se  # Upper CI
        rownames(ci_manual) <- names(est)
        colnames(ci_manual) <- c("2.5 %", "97.5 %")
        ci_manual
      })
      
      # Ensure confidence intervals match coefficients
      feature_names <- rownames(coefs)
      ci_lower <- rep(NA, length(feature_names))
      ci_upper <- rep(NA, length(feature_names))
      
      if (!is.null(conf_int) && nrow(conf_int) > 0) {
        matched_indices <- match(feature_names, rownames(conf_int))
        valid_matches <- !is.na(matched_indices)
        ci_lower[valid_matches] <- conf_int[matched_indices[valid_matches], 1]
        ci_upper[valid_matches] <- conf_int[matched_indices[valid_matches], 2]
      }
      
      importance_list[[model_name]] <- data.frame(
        Feature = feature_names,
        Importance = abs(coefs[, "Estimate"]),  # Use absolute coefficient as importance
        Coefficient = coefs[, "Estimate"],
        Std_Error = coefs[, "Std. Error"],
        t_Value = coefs[, "t value"],
        p_Value = coefs[, "Pr(>|t|)"],
        CI_Lower = ci_lower,
        CI_Upper = ci_upper,
        Significant = coefs[, "Pr(>|t|)"] < 0.05,
        stringsAsFactors = FALSE
      )
    } else if (model_name == "rf") {
      imp <- importance(model)[, 1, drop = FALSE]
      importance_list[[model_name]] <- data.frame(
        Feature = rownames(imp),
        Importance = imp[, 1],
        Coefficient = NA,
        Std_Error = NA,
        t_Value = NA,
        p_Value = NA,
        CI_Lower = NA,
        CI_Upper = NA,
        Significant = NA,
        stringsAsFactors = FALSE
      )
    } else if (model_name == "xgb") {
      imp <- xgb.importance(model = model$finalModel)
      if (nrow(imp) > 0) {
        importance_list[[model_name]] <- data.frame(
          Feature = imp$Feature,
          Importance = imp$Gain,
          Coefficient = NA,
          Std_Error = NA,
          t_Value = NA,
          p_Value = NA,
          CI_Lower = NA,
          CI_Upper = NA,
          Significant = NA,
          stringsAsFactors = FALSE
        )
      }
    }
  }
  
  return(importance_list)
}

# Enhanced display function for coefficients and importance
display_coefficients_and_importance <- function(agg_importance, model_results, target_name) {
  if (is.null(agg_importance)) {
    cat("Top 10 Most Important Features:\n")
    cat("No feature importance data available\n")
    return(NULL)
  }
  
  # Check if we have coefficient data
  has_coefficients <- "linear_Coefficient" %in% names(agg_importance)
  
  if (has_coefficients) {
    # Display ALL features with coefficients and importance
    total_features <- nrow(agg_importance)
    cat("🎯 TOP", total_features, "FEATURES WITH COEFFICIENTS & IMPORTANCE:\n")
    cat(sprintf("%-35s %12s %12s %12s %12s %12s %12s\n", 
                "Feature", "Importance", "Coefficient", "Std_Error", "p_Value", "CI_Lower", "CI_Upper"))
    cat(rep("- ", 100), "\n")
    
    # Display ALL features with full statistical info
    for (i in 1:nrow(agg_importance)) {
      # Create significance markers
      significance_marker <- ""
      if (!is.na(agg_importance$linear_p_Value[i])) {
        if (agg_importance$linear_p_Value[i] < 0.001) {
          significance_marker <- "***"
        } else if (agg_importance$linear_p_Value[i] < 0.01) {
          significance_marker <- "**"
        } else if (agg_importance$linear_p_Value[i] < 0.05) {
          significance_marker <- "*"
        }
      }
      
      cat(sprintf("%-35s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %s\n",
                  substr(agg_importance$Feature[i], 1, 35),
                  ifelse(is.na(agg_importance$Avg_Importance[i]), 0, agg_importance$Avg_Importance[i]),
                  ifelse(is.na(agg_importance$linear_Coefficient[i]), 0, agg_importance$linear_Coefficient[i]),
                  ifelse(is.na(agg_importance$linear_Std_Error[i]), 0, agg_importance$linear_Std_Error[i]),
                  ifelse(is.na(agg_importance$linear_p_Value[i]), 1, agg_importance$linear_p_Value[i]),
                  ifelse(is.na(agg_importance$linear_CI_Lower[i]), 0, agg_importance$linear_CI_Lower[i]),
                  ifelse(is.na(agg_importance$linear_CI_Upper[i]), 0, agg_importance$linear_CI_Upper[i]),
                  significance_marker))
    }
    
    cat("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 ' ' 1\n")
    
    # Statistical Summary
    significant_features <- sum(agg_importance$linear_Significant, na.rm = TRUE)
    total_features_with_p <- sum(!is.na(agg_importance$linear_Significant))
    positive_effects <- sum(agg_importance$linear_Coefficient > 0, na.rm = TRUE)
    negative_effects <- sum(agg_importance$linear_Coefficient < 0, na.rm = TRUE)
    
    cat("\n📈 STATISTICAL SUMMARY:\n")
    cat("Statistically significant features (p < 0.05):", significant_features, "out of", total_features_with_p, "\n")
    cat("Positive effects:", positive_effects, "| Negative effects:", negative_effects, "\n")
    
    # Largest effects
    if (nrow(agg_importance) > 0) {
      max_positive_idx <- which.max(agg_importance$linear_Coefficient)
      max_negative_idx <- which.min(agg_importance$linear_Coefficient)
      
      if (length(max_positive_idx) > 0 && !is.na(agg_importance$linear_Coefficient[max_positive_idx])) {
        cat("Largest positive effect:", agg_importance$Feature[max_positive_idx], 
            "(", round(agg_importance$linear_Coefficient[max_positive_idx], 4), ")\n")
      }
      
      if (length(max_negative_idx) > 0 && !is.na(agg_importance$linear_Coefficient[max_negative_idx])) {
        cat("Largest negative effect:", agg_importance$Feature[max_negative_idx], 
            "(", round(agg_importance$linear_Coefficient[max_negative_idx], 4), ")\n")
      }
    }
    
    # Model Summary from linear regression
    if ("linear" %in% names(model_results)) {
      linear_model <- model_results$linear$model
      model_summary <- summary(linear_model)
      
      cat("\n📊 LINEAR MODEL SUMMARY:\n")
      cat("R-squared:", round(model_summary$r.squared, 4), "\n")
      cat("Adjusted R-squared:", round(model_summary$adj.r.squared, 4), "\n")
      cat("F-statistic:", round(model_summary$fstatistic[1], 2), 
          "on", model_summary$fstatistic[2], "and", model_summary$fstatistic[3], "DF\n")
      
      # Model Equation (first 5 terms)
      coef_data <- agg_importance[!is.na(agg_importance$linear_Coefficient), ]
      if (nrow(coef_data) > 0) {
        cat("\n🧮 MODEL EQUATION (top 5 terms):\n")
        intercept <- coef(linear_model)[1]
        cat(target_name, "=", round(intercept, 4))
        
        for (i in 1:min(5, nrow(coef_data))) {
          coef_val <- coef_data$linear_Coefficient[i]
          if (!is.na(coef_val)) {
            sign_char <- ifelse(coef_val >= 0, " + ", " - ")
            cat(sign_char, round(abs(coef_val), 4), "*", coef_data$Feature[i], sep = "")
          }
        }
        
        if (nrow(coef_data) > 5) {
          cat(" + ... (", nrow(coef_data) - 5, " more terms)")
        }
        cat("\n")
      }
    }
    
  } else {
    # Fallback to original importance-only display
    cat("Top 10 Most Important Features:\n")
    print(head(agg_importance[, c("Feature", "Avg_Importance")], 10))
  }
  
  return(agg_importance)
}

aggregate_importance <- function(importance_list) {
  if (length(importance_list) == 0) return(NULL)
  
  # Normalize importance scores for each model
  for (i in seq_along(importance_list)) {
    max_imp <- max(importance_list[[i]]$Importance, na.rm = TRUE)
    if (max_imp > 0) {
      importance_list[[i]]$Importance <- importance_list[[i]]$Importance / max_imp
    }
  }
  
  # Merge all importance scores
  all_features <- unique(unlist(lapply(importance_list, function(x) x$Feature)))
  
  agg_df <- data.frame(Feature = all_features, stringsAsFactors = FALSE)
  
  for (model_name in names(importance_list)) {
    model_imp <- importance_list[[model_name]]
    agg_df[[paste0(model_name, "_Importance")]] <- model_imp$Importance[match(agg_df$Feature, model_imp$Feature)]
    
    # Add coefficient information for linear models
    if (!all(is.na(model_imp$Coefficient))) {
      agg_df[[paste0(model_name, "_Coefficient")]] <- model_imp$Coefficient[match(agg_df$Feature, model_imp$Feature)]
      agg_df[[paste0(model_name, "_Std_Error")]] <- model_imp$Std_Error[match(agg_df$Feature, model_imp$Feature)]
      agg_df[[paste0(model_name, "_p_Value")]] <- model_imp$p_Value[match(agg_df$Feature, model_imp$Feature)]
      agg_df[[paste0(model_name, "_CI_Lower")]] <- model_imp$CI_Lower[match(agg_df$Feature, model_imp$Feature)]
      agg_df[[paste0(model_name, "_CI_Upper")]] <- model_imp$CI_Upper[match(agg_df$Feature, model_imp$Feature)]
      agg_df[[paste0(model_name, "_Significant")]] <- model_imp$Significant[match(agg_df$Feature, model_imp$Feature)]
    }
  }
  
  # Calculate average importance
  importance_cols <- names(agg_df)[grepl("_Importance$", names(agg_df))]
  agg_df$Avg_Importance <- rowMeans(agg_df[, importance_cols, drop = FALSE], na.rm = TRUE)
  
  # Sort by average importance
  agg_df <- agg_df[order(-agg_df$Avg_Importance), ]
  
  return(agg_df)
}

# Enhanced display function for coefficients and importance
display_coefficients_and_importance <- function(agg_results, model_results, target_name, top_n = 20) {
  cat("\n📊 COMPREHENSIVE FEATURE ANALYSIS FOR", toupper(target_name), "\n")
  cat(rep("=", 80), "\n")
  
  if (is.null(agg_results) || nrow(agg_results) == 0) {
    cat("No feature analysis results available\n")
    return(NULL)
  }
  
  # Display top features with full statistical information
  display_data <- head(agg_results, top_n)
  
  # Check if we have linear model coefficients
  has_linear <- "linear_Coefficient" %in% names(display_data)
  
  if (has_linear) {
    cat("\n🎯 TOP", min(top_n, nrow(display_data)), "FEATURES WITH COEFFICIENTS & IMPORTANCE:\n")
    cat(sprintf("%-35s %12s %12s %12s %12s %12s %8s\n", 
                "Feature", "Importance", "Coefficient", "Std_Error", "p_Value", "CI_Lower", "CI_Upper"))
    cat(rep("-", 110), "\n")
    
    for (i in 1:nrow(display_data)) {
      significance_marker <- ""
      if (!is.na(display_data$linear_p_Value[i])) {
        if (display_data$linear_p_Value[i] < 0.001) {
          significance_marker <- "***"
        } else if (display_data$linear_p_Value[i] < 0.01) {
          significance_marker <- "**"
        } else if (display_data$linear_p_Value[i] < 0.05) {
          significance_marker <- "*"
        } else if (display_data$linear_p_Value[i] < 0.1) {
          significance_marker <- "."
        }
      }
      
      cat(sprintf("%-35s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %s\n",
                  substr(display_data$Feature[i], 1, 35),
                  ifelse(is.na(display_data$Avg_Importance[i]), 0, display_data$Avg_Importance[i]),
                  ifelse(is.na(display_data$linear_Coefficient[i]), 0, display_data$linear_Coefficient[i]),
                  ifelse(is.na(display_data$linear_Std_Error[i]), 0, display_data$linear_Std_Error[i]),
                  ifelse(is.na(display_data$linear_p_Value[i]), 1, display_data$linear_p_Value[i]),
                  ifelse(is.na(display_data$linear_CI_Lower[i]), 0, display_data$linear_CI_Lower[i]),
                  ifelse(is.na(display_data$linear_CI_Upper[i]), 0, display_data$linear_CI_Upper[i]),
                  significance_marker))
    }
    
    cat("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
    
    # Statistical summary
    significant_features <- sum(display_data$linear_Significant, na.rm = TRUE)
    total_features <- sum(!is.na(display_data$linear_Significant))
    
    cat("\n📈 STATISTICAL SUMMARY:\n")
    cat("Statistically significant features (p < 0.05):", significant_features, "out of", total_features, "\n")
    
    # Effect size summary
    positive_effects <- sum(display_data$linear_Coefficient > 0, na.rm = TRUE)
    negative_effects <- sum(display_data$linear_Coefficient < 0, na.rm = TRUE)
    
    cat("Positive effects:", positive_effects, "| Negative effects:", negative_effects, "\n")
    
    # Largest effects
    if (nrow(display_data) > 0) {
      max_positive_idx <- which.max(display_data$linear_Coefficient)
      max_negative_idx <- which.min(display_data$linear_Coefficient)
      
      if (length(max_positive_idx) > 0 && !is.na(display_data$linear_Coefficient[max_positive_idx])) {
        cat("Largest positive effect:", display_data$Feature[max_positive_idx], 
            "(", round(display_data$linear_Coefficient[max_positive_idx], 4), ")\n")
      }
      
      if (length(max_negative_idx) > 0 && !is.na(display_data$linear_Coefficient[max_negative_idx])) {
        cat("Largest negative effect:", display_data$Feature[max_negative_idx], 
            "(", round(display_data$linear_Coefficient[max_negative_idx], 4), ")\n")
      }
    }
    
  } else {
    cat("\n🎯 TOP", min(top_n, nrow(display_data)), "FEATURES BY IMPORTANCE:\n")
    cat(sprintf("%-45s %15s\n", "Feature", "Avg_Importance"))
    cat(rep("-", 65), "\n")
    
    for (i in 1:nrow(display_data)) {
      cat(sprintf("%-45s %15.4f\n",
                  substr(display_data$Feature[i], 1, 45),
                  ifelse(is.na(display_data$Avg_Importance[i]), 0, display_data$Avg_Importance[i])))
    }
  }
  
  # Model-specific information
  if ("linear" %in% names(model_results)) {
    linear_model <- model_results$linear$model
    model_summary <- summary(linear_model)
    
    cat("\n📊 LINEAR MODEL SUMMARY:\n")
    cat("R-squared:", round(model_summary$r.squared, 4), "\n")
    cat("Adjusted R-squared:", round(model_summary$adj.r.squared, 4), "\n")
    cat("F-statistic:", round(model_summary$fstatistic[1], 2), 
        "on", model_summary$fstatistic[2], "and", model_summary$fstatistic[3], "DF\n")
    
    # Model equation
    coef_data <- display_data[!is.na(display_data$linear_Coefficient), ]
    if (nrow(coef_data) > 0) {
      cat("\n🧮 MODEL EQUATION:\n")
      intercept <- coef(linear_model)[1]
      cat(target_name, "=", round(intercept, 4))
      
      for (i in 1:min(5, nrow(coef_data))) {  # Show first 5 terms
        coef_val <- coef_data$linear_Coefficient[i]
        if (!is.na(coef_val)) {
          sign_char <- ifelse(coef_val >= 0, " + ", " - ")
          cat(sign_char, round(abs(coef_val), 4), "*", coef_data$Feature[i], sep = "")
        }
      }
      
      if (nrow(coef_data) > 5) {
        cat(" + ... (", nrow(coef_data) - 5, " more terms)")
      }
      cat("\n")
    }
  }
  
  return(display_data)
}

# ---- YEAR ANALYSIS FUNCTIONS ----
analyze_year_impact <- function(df_with_year, df_without_year, target_name) {
  if (!"Year" %in% names(df_with_year)) {
    cat("Year feature not found\n")
    return(NULL)
  }
  
  # Quick model comparison
  set.seed(123)
  train_idx <- sample(1:nrow(df_with_year), 0.8 * nrow(df_with_year))
  
  # With Year
  train_with <- df_with_year[train_idx, ]
  test_with <- df_with_year[-train_idx, ]
  x_train_with <- as.matrix(train_with[, !names(train_with) %in% target_name])
  x_test_with <- as.matrix(test_with[, !names(test_with) %in% target_name])
  y_train <- train_with[[target_name]]
  y_test <- test_with[[target_name]]
  
  # Without Year
  train_without <- df_without_year[train_idx, ]
  test_without <- df_without_year[-train_idx, ]
  x_train_without <- as.matrix(train_without[, !names(train_without) %in% target_name])
  x_test_without <- as.matrix(test_without[, !names(test_without) %in% target_name])
  
  # Fit simple models for comparison
  lm_with <- lm(y_train ~ ., data = data.frame(x_train_with))
  lm_without <- lm(y_train ~ ., data = data.frame(x_train_without))
  
  pred_with <- predict(lm_with, data.frame(x_test_with))
  pred_without <- predict(lm_without, data.frame(x_test_without))
  
  rmse_with <- sqrt(mean((y_test - pred_with)^2, na.rm = TRUE))
  rmse_without <- sqrt(mean((y_test - pred_without)^2, na.rm = TRUE))
  
  improvement <- ((rmse_without - rmse_with) / rmse_without) * 100
  
  cat("\n=== YEAR FEATURE IMPACT FOR", toupper(target_name), "===\n")
  cat("RMSE without Year:", round(rmse_without, 4), "\n")
  cat("RMSE with Year:", round(rmse_with, 4), "\n")
  cat("Improvement:", round(improvement, 2), "%\n")
  
  # Extract Year coefficient if available
  if ("Year" %in% names(coef(lm_with))) {
    year_coef <- coef(lm_with)["Year"]
    year_summary <- summary(lm_with)
    year_p_value <- year_summary$coefficients["Year", "Pr(>|t|)"]
    
    cat("Year coefficient:", round(year_coef, 6), "\n")
    cat("Year p-value:", round(year_p_value, 6), "\n")
  }
  
  if (improvement > 2) {
    cat("✅ Year provides SUBSTANTIAL improvement\n")
  } else if (improvement > 0.5) {
    cat("👍 Year provides MODERATE improvement\n")
  } else if (improvement > 0) {
    cat("📊 Year provides SLIGHT improvement\n")
  } else {
    cat("❌ Year does not improve performance\n")
  }
  
  return(list(
    rmse_with = rmse_with,
    rmse_without = rmse_without,
    improvement = improvement
  ))
}

temporal_trend_analysis <- function(df, target_name) {
  if (!"Year" %in% names(df) || !target_name %in% names(df)) {
    return(NULL)
  }
  
  # Basic temporal summary
  temporal_summary <- df %>%
    group_by(Year) %>%
    summarise(
      Mean = mean(.data[[target_name]], na.rm = TRUE),
      SD = sd(.data[[target_name]], na.rm = TRUE),
      Count = n(),
      .groups = 'drop'
    )
  
  # Linear trend test
  trend_model <- lm(df[[target_name]] ~ df$Year)
  trend_summary <- summary(trend_model)
  
  cat("\n=== TEMPORAL TRENDS FOR", toupper(target_name), "===\n")
  print(temporal_summary)
  
  cat("\nTrend Analysis:\n")
  cat("Slope (change per year):", round(coef(trend_model)[2], 6), "\n")
  cat("P-value:", round(trend_summary$coefficients[2,4], 6), "\n")
  
  if (trend_summary$coefficients[2,4] < 0.05) {
    direction <- ifelse(coef(trend_model)[2] > 0, "INCREASING", "DECREASING")
    cat("📈 Significant", direction, "trend detected\n")
  } else {
    cat("📊 No significant temporal trend\n")
  }
  
  return(list(
    summary = temporal_summary,
    trend_model = trend_model,
    significant = trend_summary$coefficients[2,4] < 0.05
  ))
}

# ---- MAIN ANALYSIS EXECUTION ----
main_analysis <- function() {
  
  # Check for target variables
  targets <- c("Run_Scoring_Percentage", "Runs_Per_Game")
  available_targets <- targets[targets %in% names(df)]
  
  if (length(available_targets) == 0) {
    stop("No target variables found in data")
  }
  
  cat("Found targets:", paste(available_targets, collapse = ", "), "\n\n")
  
  results <- list()
  
  for (target in available_targets) {
    cat("==================== ANALYZING", toupper(target), "====================\n\n")
    
    # Prepare data (excluding circular predictors, with advanced feature selection)
    target_data <- prepare_data_for_target(df, target, exclude_circular = TRUE, use_boruta = TRUE)
    target_data_no_year <- target_data[, names(target_data) != "Year", drop = FALSE]
    
    # Train-test split
    set.seed(123)
    train_idx <- sample(1:nrow(target_data), 0.8 * nrow(target_data))
    
    train_data <- target_data[train_idx, ]
    test_data <- target_data[-train_idx, ]
    
    x_train <- as.matrix(train_data[, !names(train_data) %in% target])
    x_test <- as.matrix(test_data[, !names(test_data) %in% target])
    y_train <- train_data[[target]]
    y_test <- test_data[[target]]
    
    # Store feature names
    feature_names <- colnames(x_train)
    
    # Fit models
    cat("Fitting models...\n")
    model_results <- fit_models(x_train, y_train, x_test, y_test)
    
    if (length(model_results) == 0) {
      cat("No models successfully fitted for", target, "\n")
      next
    }
    
    # Extract predictions
    predictions <- lapply(model_results, function(x) x$pred)
    
    # Evaluate individual models
    cat("\nModel Performance:\n")
    performance <- evaluate_predictions(predictions, y_test)
    print(performance)
    
    # Build stacked ensemble
    stacked_result <- stack_models(x_train, y_train, x_test, y_test, target)
    
    if (!is.null(stacked_result)) {
      cat("\n🏆 STACKED ENSEMBLE RESULTS:\n")
      print(stacked_result$metrics)
      
      # Compare with best individual model
      if (nrow(performance) > 0) {
        best_individual_rmse <- performance$RMSE[1]
        stacked_rmse <- stacked_result$metrics$RMSE
        improvement <- ((best_individual_rmse - stacked_rmse) / best_individual_rmse) * 100
        
        cat("📈 Stacked ensemble improvement:", round(improvement, 2), "% over best individual\n")
        
        # Add stacked predictions to the prediction list
        predictions$stacked <- stacked_result$predictions
        
        # Update performance comparison
        performance_with_stack <- evaluate_predictions(predictions, y_test)
        cat("\n📊 UPDATED PERFORMANCE RANKING (including ensemble):\n")
        print(performance_with_stack)
      }
    }
    
    # Model quality assessment
    cat("\nModel Quality Assessment:")
    for (model_name in names(predictions)) {
      assess_model_quality(predictions[[model_name]], y_test, paste(target, model_name))
    }
    
    # Generate diagnostic plots for best model
    if (require(ggplot2, quietly = TRUE) && nrow(performance) > 0) {
      best_model <- performance$Model[1]
      if (best_model %in% names(predictions)) {
        cat("\nGenerating diagnostic plots for best model:", best_model, "\n")
        plots <- plot_diagnostics(predictions[[best_model]], y_test, 
                                  paste(target, best_model))
        if (!is.null(plots)) {
          print(plots$residuals)
          print(plots$actual_pred)
        }
      }
    }
    
    # Enhanced feature importance and coefficient analysis
    cat("\nFeature Importance Analysis:\n")
    importance_results <- get_feature_importance(model_results, target)
    agg_importance <- aggregate_importance(importance_results)
    
    if (!is.null(agg_importance)) {
      # Force display of ALL features - bypass any other display functions
      if ("linear_Coefficient" %in% names(agg_importance)) {
        # Display ALL features directly here - don't call any other functions
        total_features <- nrow(agg_importance)
        
        cat("\n📊 ALL", total_features, "FEATURES WITH COMPLETE COEFFICIENT ANALYSIS:\n")
        cat(sprintf("%-35s %12s %12s %12s %12s %12s %12s\n", 
                    "Feature", "Importance", "Coefficient", "Std_Error", "p_Value", "CI_Lower", "CI_Upper"))
        cat(rep("=", 130), "\n")
        
        # Display EVERY SINGLE feature without exception
        for (feature_idx in 1:total_features) {
          # Get significance marker
          sig_marker <- ""
          p_val <- agg_importance$linear_p_Value[feature_idx]
          if (!is.na(p_val)) {
            if (p_val < 0.001) sig_marker <- "***"
            else if (p_val < 0.01) sig_marker <- "**"
            else if (p_val < 0.05) sig_marker <- "*"
            else if (p_val < 0.1) sig_marker <- "."
          }
          
          cat(sprintf("%-35s %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f %s\n",
                      substr(agg_importance$Feature[feature_idx], 1, 35),
                      ifelse(is.na(agg_importance$Avg_Importance[feature_idx]), 0, agg_importance$Avg_Importance[feature_idx]),
                      ifelse(is.na(agg_importance$linear_Coefficient[feature_idx]), 0, agg_importance$linear_Coefficient[feature_idx]),
                      ifelse(is.na(agg_importance$linear_Std_Error[feature_idx]), 0, agg_importance$linear_Std_Error[feature_idx]),
                      ifelse(is.na(p_val), 1, p_val),
                      ifelse(is.na(agg_importance$linear_CI_Lower[feature_idx]), 0, agg_importance$linear_CI_Lower[feature_idx]),
                      ifelse(is.na(agg_importance$linear_CI_Upper[feature_idx]), 0, agg_importance$linear_CI_Upper[feature_idx]),
                      sig_marker))
        }
        
        cat("\nSignificance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")
        
        # Summary statistics
        sig_count <- sum(agg_importance$linear_Significant, na.rm = TRUE)
        total_with_p <- sum(!is.na(agg_importance$linear_Significant))
        pos_count <- sum(agg_importance$linear_Coefficient > 0, na.rm = TRUE)
        neg_count <- sum(agg_importance$linear_Coefficient < 0, na.rm = TRUE)
        
        cat("\n📈 COMPLETE STATISTICAL SUMMARY:\n")
        cat("Total features analyzed:", total_features, "\n")
        cat("Statistically significant features (p < 0.05):", sig_count, "out of", total_with_p, "\n")
        cat("Positive effects:", pos_count, "| Negative effects:", neg_count, "\n")
        
        # Find largest effects
        max_pos_idx <- which.max(agg_importance$linear_Coefficient)
        max_neg_idx <- which.min(agg_importance$linear_Coefficient)
        
        if (length(max_pos_idx) > 0 && !is.na(agg_importance$linear_Coefficient[max_pos_idx])) {
          cat("Largest positive effect:", agg_importance$Feature[max_pos_idx], 
              "(", round(agg_importance$linear_Coefficient[max_pos_idx], 4), ")\n")
        }
        
        if (length(max_neg_idx) > 0 && !is.na(agg_importance$linear_Coefficient[max_neg_idx])) {
          cat("Largest negative effect:", agg_importance$Feature[max_neg_idx], 
              "(", round(agg_importance$linear_Coefficient[max_neg_idx], 4), ")\n")
        }
        
        # Model summary
        if ("linear" %in% names(model_results)) {
          linear_model <- model_results$linear$model
          model_summary <- summary(linear_model)
          
          cat("\n📊 LINEAR MODEL SUMMARY:\n")
          cat("R-squared:", round(model_summary$r.squared, 4), "\n")
          cat("Adjusted R-squared:", round(model_summary$adj.r.squared, 4), "\n")
          cat("F-statistic:", round(model_summary$fstatistic[1], 2), 
              "on", model_summary$fstatistic[2], "and", model_summary$fstatistic[3], "DF\n")
          
          # Model equation - FIXED: use 'target' variable instead of 'target_name'
          coef_data <- agg_importance[!is.na(agg_importance$linear_Coefficient), ]
          if (nrow(coef_data) > 0) {
            cat("\n🧮 MODEL EQUATION (first 5 terms):\n")
            intercept <- coef(linear_model)[1]
            cat(target, "=", round(intercept, 4))  # FIXED: use 'target' not 'target_name'
            
            for (eq_i in 1:min(5, nrow(coef_data))) {
              coef_val <- coef_data$linear_Coefficient[eq_i]
              if (!is.na(coef_val)) {
                sign_char <- ifelse(coef_val >= 0, " + ", " - ")
                cat(sign_char, round(abs(coef_val), 4), "*", coef_data$Feature[eq_i], sep = "")
              }
            }
            
            if (nrow(coef_data) > 5) {
              cat(" + ... (", nrow(coef_data) - 5, " more terms)")
            }
            cat("\n")
          }
        }
        
        cat("\n✅ VERIFICATION: Successfully displayed ALL", total_features, "features with coefficients\n")
        
      } else {
        # Fallback for non-linear models - use the original display function
        cat("Top 10 Most Important Features:\n")
        print(head(agg_importance[, c("Feature", "Avg_Importance")], 10))
      }
    }
    
    # Highlight Year feature
    if (!is.null(agg_importance) && "Year" %in% agg_importance$Feature) {
      year_rank <- which(agg_importance$Feature == "Year")
      year_importance <- agg_importance$Avg_Importance[year_rank]
      cat("\n📅 YEAR FEATURE:\n")
      cat("Rank:", year_rank, "out of", nrow(agg_importance), "\n")
      cat("Importance Score:", round(year_importance, 4), "\n")
      
      if ("linear_Coefficient" %in% names(agg_importance)) {
        year_coef <- agg_importance$linear_Coefficient[year_rank]
        year_p_val <- agg_importance$linear_p_Value[year_rank]
        if (!is.na(year_coef)) {
          cat("Year Coefficient:", round(year_coef, 6), "\n")
          cat("Year p-value:", round(year_p_val, 6), "\n")
        }
      }
    }
    
    # Year impact analysis
    year_impact <- analyze_year_impact(target_data, target_data_no_year, target)
    
    # Temporal trend analysis
    temporal_results <- temporal_trend_analysis(target_data, target)
    
    # Store results (include stacked ensemble if available)
    results[[target]] <- list(
      performance = if (!is.null(stacked_result)) performance_with_stack else performance,
      importance = agg_importance,
      year_impact = year_impact,
      temporal = temporal_results,
      stacked_ensemble = stacked_result,
      model_objects = model_results,
      best_model = if (!is.null(stacked_result) && exists("performance_with_stack")) {
        performance_with_stack$Model[1]
      } else {
        performance$Model[1]
      },
      best_rmse = if (!is.null(stacked_result) && exists("performance_with_stack")) {
        performance_with_stack$RMSE[1]
      } else {
        performance$RMSE[1]
      }
    )
    
    cat("\n" , rep("=", 80), "\n\n")
  }
  
  return(results)
}

# ---- ENHANCED CROSS-MODEL COMPARISON WITH COEFFICIENTS ----
generate_cross_model_comparison <- function(results) {
  if (length(results) < 2) {
    cat("⚠️ Need at least 2 models for cross-comparison\n")
    return(NULL)
  }
  
  cat("\n", rep("🔍", 40), "\n")
  cat("CROSS-MODEL FEATURE COMPARISON WITH COEFFICIENTS\n")
  cat(rep("🔍", 40), "\n\n")
  
  # Get all features from both models
  target_names <- names(results)
  all_features <- list()
  
  for (target in target_names) {
    if (!is.null(results[[target]]$importance_coefficients)) {
      all_features[[target]] <- results[[target]]$importance_coefficients
    }
  }
  
  if (length(all_features) < 2) {
    cat("⚠️ Insufficient feature data for comparison\n")
    return(NULL)
  }
  
  # Find common features
  feature_lists <- lapply(all_features, function(x) x$Feature)
  common_features <- Reduce(intersect, feature_lists)
  
  cat("🤝 Features common to both models:", length(common_features), "\n")
  
  if (length(common_features) > 0) {
    # Create comparison table
    target1 <- target_names[1]
    target2 <- target_names[2]
    
    imp1 <- all_features[[target1]]
    imp2 <- all_features[[target2]]
    
    comparison_data <- data.frame(
      Feature = common_features,
      RSP_Importance = imp1$Avg_Importance[match(common_features, imp1$Feature)],
      RPG_Importance = imp2$Avg_Importance[match(common_features, imp2$Feature)]
    )
    
    # Add coefficient information if available
    if ("linear_Coefficient" %in% names(imp1)) {
      comparison_data$RSP_Coefficient <- imp1$linear_Coefficient[match(common_features, imp1$Feature)]
      comparison_data$RSP_p_Value <- imp1$linear_p_Value[match(common_features, imp1$Feature)]
    }
    
    if ("linear_Coefficient" %in% names(imp2)) {
      comparison_data$RPG_Coefficient <- imp2$linear_Coefficient[match(common_features, imp2$Feature)]
      comparison_data$RPG_p_Value <- imp2$linear_p_Value[match(common_features, imp2$Feature)]
    }
    
    comparison_data$Avg_Importance <- (comparison_data$RSP_Importance + comparison_data$RPG_Importance) / 2
    comparison_data$Importance_Diff <- abs(comparison_data$RSP_Importance - comparison_data$RPG_Importance)
    
    # Sort by average importance
    comparison_data <- comparison_data[order(-comparison_data$Avg_Importance), ]
    
    cat("\n📊 TOP 15 COMMON FEATURES WITH COEFFICIENTS:\n")
    if ("RSP_Coefficient" %in% names(comparison_data) && "RPG_Coefficient" %in% names(comparison_data)) {
      cat(sprintf("%-35s %8s %8s %10s %10s %8s %8s\n", 
                  "Feature", "RSP_Imp", "RPG_Imp", "RSP_Coef", "RPG_Coef", "RSP_p", "RPG_p"))
      cat(rep("-", 100), "\n")
      
      for (i in 1:min(15, nrow(comparison_data))) {
        cat(sprintf("%-35s %8.3f %8.3f %10.4f %10.4f %8.3f %8.3f\n",
                    substr(comparison_data$Feature[i], 1, 35),
                    comparison_data$RSP_Importance[i],
                    comparison_data$RPG_Importance[i],
                    ifelse(is.na(comparison_data$RSP_Coefficient[i]), 0, comparison_data$RSP_Coefficient[i]),
                    ifelse(is.na(comparison_data$RPG_Coefficient[i]), 0, comparison_data$RPG_Coefficient[i]),
                    ifelse(is.na(comparison_data$RSP_p_Value[i]), 1, comparison_data$RSP_p_Value[i]),
                    ifelse(is.na(comparison_data$RPG_p_Value[i]), 1, comparison_data$RPG_p_Value[i])))
      }
    } else {
      cat(sprintf("%-35s %12s %12s %12s %8s\n", 
                  "Feature", "RSP", "RPG", "Average", "Diff"))
      cat(rep("-", 75), "\n")
      
      for (i in 1:min(15, nrow(comparison_data))) {
        indicator <- ifelse(comparison_data$RSP_Importance[i] > comparison_data$RPG_Importance[i], "→RSP", "→RPG")
        
        cat(sprintf("%-35s %12.4f %12.4f %12.4f %8.4f %s\n", 
                    substr(comparison_data$Feature[i], 1, 35),
                    comparison_data$RSP_Importance[i],
                    comparison_data$RPG_Importance[i],
                    comparison_data$Avg_Importance[i],
                    comparison_data$Importance_Diff[i],
                    indicator))
      }
    }
    
    return(comparison_data)
  }
  
  return(NULL)
}

generate_summary <- function(results) {
  cat("\n#################### FINAL SUMMARY ####################\n\n")
  
  for (target in names(results)) {
    result <- results[[target]]
    
    cat("🎯", toupper(target), "RESULTS:\n")
    # Show which model is actually being used
    if (!is.null(result$ensemble_used) && result$ensemble_used) {
      cat("🏆 BEST Model: STACKED ENSEMBLE (RMSE:", round(result$best_rmse, 4), ")\n")
      cat("   ✅ Ensemble BEAT individual models\n")
      cat("   📊 Best individual was:", result$individual_best, "(RMSE:", round(result$individual_rmse, 4), ")\n")
      cat("   📈 Ensemble improvement:", round(((result$individual_rmse - result$best_rmse) / result$individual_rmse) * 100, 2), "%\n")
    } else {
      cat("🏆 BEST Model:", result$best_model, "(RMSE:", round(result$best_rmse, 4), ")\n")
      if (!is.null(result$stacked_ensemble)) {
        ensemble_rmse <- result$stacked_ensemble$metrics$RMSE
        cat("   ⚠️ Individual model BEAT ensemble\n")
        cat("   📊 Ensemble RMSE:", round(ensemble_rmse, 4), 
            "(", round(((ensemble_rmse - result$best_rmse) / result$best_rmse) * 100, 2), "% worse)\n")
      } else {
        cat("   📊 Individual model used (no ensemble available)\n")
      }
    }
    
    if (!is.null(result$importance) && nrow(result$importance) > 0) {
      top_features <- head(result$importance$Feature, 3)
      cat("🏆 Top 3 Predictors:", paste(top_features, collapse = ", "), "\n")
      
      # Show coefficient information for top features if available
      if ("linear_Coefficient" %in% names(result$importance)) {
        cat("📊 Top Feature Coefficients:\n")
        for (i in 1:min(3, nrow(result$importance))) {
          feat_name <- result$importance$Feature[i]
          feat_coef <- result$importance$linear_Coefficient[i]
          feat_p <- result$importance$linear_p_Value[i]
          significance <- if (!is.na(feat_p) && feat_p < 0.05) "***" else ""
          
          if (!is.na(feat_coef)) {
            cat("   •", feat_name, ":", round(feat_coef, 4), significance, "\n")
          }
        }
      }
      
      # Show complete ranking summary
      total_features <- nrow(result$importance)
      high_importance <- sum(result$importance$Avg_Importance > 0.3, na.rm = TRUE)
      moderate_importance <- sum(result$importance$Avg_Importance > 0.1 & result$importance$Avg_Importance <= 0.3, na.rm = TRUE)
      low_importance <- sum(result$importance$Avg_Importance <= 0.1, na.rm = TRUE)
      
      cat("📊 Feature Distribution:", high_importance, "high,", moderate_importance, "moderate,", low_importance, "low importance\n")
      
      # Statistical significance summary
      if ("linear_Significant" %in% names(result$importance)) {
        significant_count <- sum(result$importance$linear_Significant, na.rm = TRUE)
        total_with_p <- sum(!is.na(result$importance$linear_Significant))
        cat("📈 Statistically significant features:", significant_count, "out of", total_with_p, "\n")
      }
    }
    
    if (!is.null(result$year_impact)) {
      cat("Year Impact:", round(result$year_impact$improvement, 2), "% improvement\n")
    }
    
    if (!is.null(result$temporal) && result$temporal$significant) {
      cat("⚠ Significant temporal trend detected\n")
    }
    
    cat("\n")
  }
  
  cat("Key Enhancements Made:\n")
  cat("✅ Removed circular predictors (RSP ↔ RPG)\n")
  cat("✅ Advanced feature selection (Correlation + VIF + Boruta)\n")
  cat("✅ Sophisticated stacked ensemble with cross-validation\n")
  cat("✅ Separate models for each target\n")
  cat("✅ Complete coefficient analysis with statistical significance\n")
  cat("✅ Comprehensive Year feature analysis\n")
  cat("✅ Multiple model algorithms (Linear, Ridge, RF, XGBoost)\n")
  cat("✅ Enhanced coefficient display with confidence intervals\n")
}

# ---- RUN ANALYSIS WITH BETTER ERROR HANDLING ----
cat("🚀 Starting enhanced baseball analysis with coefficients...\n\n")

# Initialize results
analysis_results <- list()

# Run analysis with comprehensive error handling
tryCatch({
  analysis_results <- main_analysis()
  
  if (length(analysis_results) > 0) {
    cat("✅ Analysis completed successfully!\n\n")
    generate_summary(analysis_results)
    
    # Generate cross-model comparison if we have multiple targets
    if (length(analysis_results) > 1) {
      cross_comparison <- generate_cross_model_comparison(analysis_results)
    }
  } else {
    cat("⚠️ No results generated - check data and target variables\n")
  }
  
}, error = function(e) {
  cat("❌ Critical error in main analysis:", e$message, "\n")
  cat("📊 Available columns in data:", paste(names(df)[1:min(20, ncol(df))], collapse = ", "), "\n")
  cat("📏 Data dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
  
  # Check for target variables
  targets <- c("Run_Scoring_Percentage", "Runs_Per_Game")
  found_targets <- targets[targets %in% names(df)]
  
  if (length(found_targets) > 0) {
    cat("🎯 Target variables found:", paste(found_targets, collapse = ", "), "\n")
  } else {
    cat("❌ No target variables found in data\n")
    cat("🔍 Possible target-like columns:", 
        paste(grep("run|score|game", names(df), ignore.case = TRUE, value = TRUE), collapse = ", "), "\n")
  }
})

cat("\n#################### ANALYSIS COMPLETE ####################\n")
  
  