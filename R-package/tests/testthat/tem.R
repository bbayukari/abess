library(abess)
library(testthat)
library(ordinalNet)
test_that("abess (ordinal) works", {
  
  n <- 2000
  p <- 10
  support.size <- 5
  dataset <- generate.data(n, p, support.size,
                           family = "ordinal",class.num = 3, seed = 123)
  true_beta_idx <- which(dataset[["beta"]]!=0)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "ordinal",
    tune.type = "cv", 
    support.size = support.size,
    always.include = true_beta_idx,
    #max.splicing.iter = 1
    )
  ordinalNet_fit <- ordinalNet(dataset[["x"]][,true_beta_idx],
                               as.factor(dataset[["y"]]),
                               family = "cumulative",
                               link = "logit",
                               parallelTerms = TRUE,
                               nonparallelTerms = FALSE
                               )
  
  fit_beta_idx <- as.vector(which(abess_fit[["beta"]][[as.character(support.size)]][,1]!=0))
  #true_beta_idx  
  #fit_beta_idx
  coef(ordinalNet_fit)
  dataset[["beta"]][true_beta_idx]
  abess_fit[["beta"]][[as.character(support.size)]][fit_beta_idx,1]
})