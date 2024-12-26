# first: generate the dataframe from pca.py file

df <- read.csv("normalized_pca_data_FINAL.csv", header = TRUE, sep = ",")
params <- df[,c(2:4)]
df <- df[, -c(1:4)]



# principal component analysis
res <- prcomp(df)

# get percentage of explained varience
prop_explained <- res$sdev^2 / sum(res$sdev^2)
new_df <- data.frame(prop_explained, cumsum(prop_explained))
write.csv(new_df, file = "pca_results2.csv", row.names = FALSE)

# loadings
loadings <- res$rotation
write.csv(loadings, "loadings_results2.csv", row.names = FALSE)

# scores
scores = res$x
scores_with_params <- cbind(params, scores)
write.csv(scores, "scores_results2.csv", row.names = FALSE)
write.csv(scores_with_params, "scores_with_params_results2.csv", row.names = FALSE)


# na značilkah
df <- read.csv("features_FINAL.csv", header = TRUE, sep = ',')
params <- df[,c(2:4)]
df <- df[, -c(1:4)]

df <- scale(df)
write.csv(df, "scaled_features.csv", row.names = FALSE)

res <- prcomp(df)

# get percentage of explained varience
prop_explained <- res$sdev^2 / sum(res$sdev^2)
new_df <- data.frame(prop_explained, cumsum(prop_explained))
write.csv(new_df, file = "pca_results2_znacilke.csv", row.names = FALSE)

# loadings
loadings <- res$rotation
write.csv(loadings, "loadings_results2_znacilke.csv", row.names = FALSE)


# scores
scores = res$x
scores_with_params <- cbind(params, scores)
write.csv(scores, "scores_results2_znacilke.csv", row.names = FALSE)
write.csv(scores_with_params, "scores_with_params_results2_znacilke.csv", row.names = FALSE)
