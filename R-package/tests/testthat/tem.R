library(abess)
library(testthat)
#library(ordinalNet)
test_that("abess (ordinal) works", {
  
  n <- 1000
  p <- 5
  support.size <- 2
  dataset <- generate.data(n, p, support.size,
                           family = "ordinal",class.num = 4, seed = 1)
  true_beta_idx <- which(dataset[["beta"]]!=0)
  
  abess_fit <- abess(
    dataset[["x"]],
    dataset[["y"]],
    family = "ordinal",
    tune.type = "cv", 
    support.size = support.size,
    #always.include = true_beta_idx,
    num.threads = 1,
    newton.thresh = 1e-10,
    max.newton.iter = 200
    #max.splicing.iter = 1
    )
 #  ordinalNet_fit <- ordinalNet(dataset[["x"]][,true_beta_idx],
 #                               as.factor(dataset[["y"]]),
 #                               family = "cumulative",
 #                               link = "logit",
 #                               parallelTerms = TRUE,
 #                               nonparallelTerms = FALSE
 #                               )
  
  fit_beta_idx <- as.vector(which(abess_fit[["beta"]][[as.character(support.size)]][,1]!=0))
  #true_beta_idx  
  #fit_beta_idx
  #coef(ordinalNet_fit)
  print(true_beta_idx)
  print(dataset[["beta"]][true_beta_idx])
  print(abess_fit[["beta"]][[as.character(support.size)]][fit_beta_idx,1])
  print(extract(abess_fit))
})