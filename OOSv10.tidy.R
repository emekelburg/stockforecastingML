# Script to generate predictions based on different methods
#  how to call from console:

# Rscript OOSv10.tidy.R
#--file="Dissertation/usa.df.analysis.rda"
#--method=NN5
#--rcum=rolling
#--yvar=ret_exc_lead1m
#--paramadj=pca
#--window=120
#--gap=0
#--prefix=USA

# Rscript OOSv10.tidy.R --file="Dissertation/usa.df.analysis.rda" --method=NN5 --rcum=rolling --yvar=ret_exc_lead1m --paramadj=pca --window=120 --gap=0 --prefix=USAvi --varimp=T

#Rscript OOSv10.tidy.R --file="usa.3m.exp.df.rda" --method=SCFMEAN --rcum=rolling --yvar=y --paramadj=allvars --window=120 --gap=2 --prefix=USA3m
#Rscript OOSv10.tidy.R --file="usa.3m.exp.df.rda" --method=SCFMEAN --rcum=rolling --yvar=y --paramadj=allvars --window=240 --gap=2 --prefix=USA3m

# Rscript OOSv10.tidy.R --file="Dissertation/usa.df.analysis.rda" --method=NN5 --rcum=rolling --yvar=ret_exc_lead1m --paramadj=pca --window=120 --gap=0 --prefix=USA

# Rscript OOSv10.tidy.R --file=test.usa.ts.df.rda --method=LASSO --rcum=rolling --yvar=ret_exc_lead1m --paramadj=pca --window=120 --gap=0 --prefix=USA
# Rscript OOSv10.tidy.R --file="Dissertation/usa.df.analysis.rda" --method=SCF-MEAN --rcum=rolling --yvar=ret_exc_lead1m --paramadj=allvars --window=120 --gap=0 --prefix=USA
# export R_PROGRESSR_ENABLE=TRUE

suppressPackageStartupMessages(library(optparse))
suppressPackageStartupMessages(library(tidymodels))
suppressPackageStartupMessages(library(furrr)) # parallel execution of models on multiple cores (extends futures packages)
suppressPackageStartupMessages(library(progressr)) # progress bar for parallel execution
#suppressPackageStartupMessages(library(tensorflow))
#suppressPackageStartupMessages(library(keras))
suppressPackageStartupMessages(library(ANN2)) # neural nets
suppressPackageStartupMessages(library(glmnet))
suppressPackageStartupMessages(library(ranger))
#suppressPackageStartupMessages(library(lightgbm))

# set execution plan for futures packages - influences parallel execution
#   default is to use all available cores (retrieved with parallelly::availableCores())
#plan(sequential)  # for sequential execution
#plan(multisession, workers = 2) # for multi-processing with specific number of cores
plan(multicore, workers = parallelly::availableCores() - 2) # leave some cores for regular ops
#plan(multicore, workers = parallelly::availableCores()) # leave some cores for regular ops

# set format of the progress bar#
#handlers("txtprogressbar")
handlers("debug")


#############################
#      Data Prep            #
#############################

func.load.dataset = function(i.filename) {
  model.df = load(i.filename)
  l.dataset = get(model.df)
  return (l.dataset)
}

# function to create recipe based on selected parameter adjustments (pca, pls, none)
func.data.prep <- function(i.slice.data, i.paramadj, ...){
  # recipe for data preprocessing
  l.recipe <-
    recipe(y ~ ., data = i.slice.data %>% analysis ) %>%
    update_role(date, new_role = "ID")  %>%  # remove date from the list of regressors
    step_filter_missing(all_predictors(), threshold = 0) %>% # remove all predictor columns with missing values
    #step_YeoJohnson(all_predictors()) %>%#all_numeric()) %>%
    step_zv(all_predictors()) %>%  # remove zero variance columns before normalizing
    step_normalize(all_predictors())

    # add PCA adjustment if desired
    if (i.paramadj == "pca") {
      l.recipe <- l.recipe %>% step_pca(all_numeric_predictors(), threshold=.99)#num_comp = 90)
    }

    # add cppls() adjustment if desired  # note - this requires package mixOmics
    if (i.paramadj == "pls") {
      l.recipe <- l.recipe %>% step_pls(all_numeric_predictors(), outcome="y", num_comp = 90)
    }

    l.recipe <- l.recipe %>% prep() # train the recipe based on the input data

    return(l.recipe)
}




#############################
#      Model Functions      #
#############################


# simple averaged combination forecast of all individual predictors
func.fit.scf <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {

  df.train <- i.recipe %>% bake(new_data=NULL)
  df.test  <- i.recipe %>% bake(new_data=i.data %>% assessment)

  # check if the test data has missing data (this is not caught by recipe/bake
  #   because the recipe is only build with the training data)
  missing_cols_test <- df.test %>%  select_if(~any(is.na(.))) %>% colnames
  # remove missing cols from both datasets
  if(length(missing_cols_test > 0)) {
    df.train <- df.train %>% select( -any_of(missing_cols_test) )
    df.test <- df.test %>% select( -any_of(missing_cols_test) )
  }

  # split the dataset into X and Y separately
  train.features <- df.train %>% select(-c(y, date))
  test.features <- df.test %>% select(-c(y, date))

  train.labels <- df.train %>% select(y)
#  test.labels <- df.test %>% select(y)

  train.labels.scaled <- scale(train.labels)

  train.data.combined <- cbind(train.features, train.labels.scaled)

  # collect scale stats for y-variable, so we can unscale prediction later
  y.scale.sd = attr(train.labels.scaled, 'scaled:scale')[c("y")]
  y.scale.mean = attr(train.labels.scaled, 'scaled:center')[c("y")]

  # create empty list for our predictions and variable importances
  df.indiv.predictions <- c()
  l.var.importance <- c()

  # run regression for each data point in the training data set and output fit
  for (column in colnames(train.features)){
    l.f = as.formula(paste("y~",column))
    l.model = lm(l.f, data = train.data.combined)
    indiv.prediction = as.numeric(predict(l.model, newdata = test.features))
    df.indiv.predictions = append( df.indiv.predictions, indiv.prediction)
    l.var.importance[column] = summary(l.model)$adj.r.squared
  }

  l.method = i.xargs$method

  # take a simple average of the predictions and return individual R^2 as importance vector
  #prediction = mean(df.indiv.predictions)
  prediction = l.method(df.indiv.predictions)

  prediction <- prediction * y.scale.sd + y.scale.mean

  return ( list( prediction = prediction, var.importance = l.var.importance, extra.info = list() ) )

}

func.fit.scf.mean  <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  if (is.null(i.xargs)) {
    l.xargs <- list()
    l.xargs$method <- mean
  } else {
    l.xargs$method <- mean
  }
  return(func.fit.scf(i.data, i.recipe, i.xargs=l.xargs,  i.var.imp=i.var.imp,...))
}

func.fit.scf.median  <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  if (is.null(i.xargs)) {
    l.xargs <- list()
    l.xargs$method <- median
  } else {
    l.xargs$method <- median
  }
  return(func.fit.scf(i.data, i.recipe, i.xargs=l.xargs,  i.var.imp=i.var.imp,...))
}

func.fit.scf.trimmedmean  <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  func.trimmedmean <- function(x){
    return ( mean (x,trim=0.05) )
  }
  if (is.null(i.xargs)) {
    l.xargs <- list()
    l.xargs$method <- func.trimmedmean
  } else {
    l.xargs$method <- func.trimmedmean
  }
  return(func.fit.scf(i.data, i.recipe, i.xargs=l.xargs,  i.var.imp=i.var.imp,...))
}


# tuned glmnet model definition - requires i.xargs$alpha parameter (set by wrappers below)
func.fit.glmnet <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {

  df.train <- i.recipe %>% bake(new_data=NULL)
  df.test  <- i.recipe %>% bake(new_data=i.data %>% assessment)

  # check if the test data has missing data (this is not caught by recipe/bake
  #   because the recipe is only build with the training data)
  missing_cols_test <- df.test %>%  select_if(~any(is.na(.))) %>% colnames
  # remove missing cols from both datasets
  if(length(missing_cols_test > 0)) {
    df.train <- df.train %>% select( -any_of(missing_cols_test) )
    df.test <- df.test %>% select( -any_of(missing_cols_test) )
  }

  # split the dataset into X and Y separately
  train.features <- df.train %>% select(-c(y, date))
  test.features <- df.test %>% select(-c(y, date))

  train.labels <- df.train %>% select(y)
#  test.labels <- df.test %>% select(y)

  train.labels.scaled <- scale(train.labels)

  # collect scale stats for y-variable, so we can unscale prediction later
  y.scale.sd = attr(train.labels.scaled, 'scaled:scale')[c("y")]
  y.scale.mean = attr(train.labels.scaled, 'scaled:center')[c("y")]

  l.alpha = i.xargs$alpha

  # fit GLMNET Lasso Model
  lambdas <- 10^seq(5, -5, by = -.1)
  # fit GLMNET Lasso Model
  lasso.model <- cv.glmnet(as.matrix(train.features), train.labels.scaled, alpha = l.alpha, standardize = F, lambda = lambdas, nfolds = 10, parallel=TRUE)
  # take the mean of the min MSE and 1SE (lowest lambda that is < 1se above min) - this avoids overfitting
  lambda.choice <- mean(lasso.model$lambda.min, lasso.model$lambda.1se)
  prediction <- as.numeric(predict(lasso.model, s=lambda.choice, newx = as.matrix(test.features)))

  prediction <- prediction * y.scale.sd + y.scale.mean

  var.imp.vector = c()

  if ( i.var.imp == TRUE ){
    #coefList <- coef(lasso.model, s='lambda.1se')
    coefList <- coef(lasso.model, s=lambda.choice )
    coefList <- data.frame(coefList@Dimnames[[1]][coefList@i+1],coefList@x)
    names(coefList) <- c('var','val')
    # remove intercept
    var.importance = coefList[-1,]
    # extract a named vector for Gains field
    var.imp.vector = var.importance$val
    names(var.imp.vector) = var.importance$var
  } 

  return ( list( prediction = prediction, var.importance = var.imp.vector, extra.info = list() ) )

}

func.fit.lasso  <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  # alpha is 1 for lasso
  if (is.null(i.xargs)) {
    l.xargs <- list()
    l.xargs$alpha <- 1
  } else {
    l.xargs$alpha <- 1
  }
  return(func.fit.glmnet(i.data, i.recipe, i.xargs=l.xargs,  i.var.imp=i.var.imp,...))
}

func.fit.ridge  <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  # alpha is 1 for lasso
  if (is.null(i.xargs)) {
    l.xargs <- list()
    l.xargs$alpha <- 0
  } else {
    l.xargs$alpha <- 0
  }
  return(func.fit.glmnet(i.data, i.recipe, i.xargs=l.xargs, i.var.imp,...))
}

func.fit.enet  <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  # alpha is 1 for lasso
  if (is.null(i.xargs)) {
    l.xargs <- list()
    l.xargs$alpha <- 0.5
  } else {
    l.xargs$alpha <- 0.5
  }
  return(func.fit.glmnet(i.data, i.recipe, i.xargs=l.xargs, i.var.imp,...))
}


func.fit.model.NN5 <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {

  l.layers <- c(32,16,8,4,2)
  df.train <- i.recipe %>% bake(new_data=NULL)
  df.test  <- i.recipe %>% bake(new_data=i.data %>% assessment)

  # check if the test data has missing data (this is not caught by recipe/bake
  #   because the recipe is only build with the training data)
  missing_cols_test <- df.test %>%  select_if(~any(is.na(.))) %>% colnames
  # remove missing cols from both datasets
  if(length(missing_cols_test > 0)) {
    df.train <- df.train %>% select( -any_of(missing_cols_test) )
    df.test <- df.test %>% select( -any_of(missing_cols_test) )
  }

  # split the dataset into X and Y separately
  train.features <- df.train %>% select(-c(y, date))
  test.features <- df.test %>% select(-c(y, date))

  train.labels <- df.train %>% select(y)
#  test.labels <- df.test %>% select(y)

  train.labels.scaled <- scale(train.labels)
#  test.labels.scaled <- scale(test.labels, center = attr(train.labels.scaled, 'scaled:center'), scale = attr(train.labels.scaled, 'scaled:scale'))
#  test.y.scaled = scale(df.test.labels, center = attr(train.data.scaled, 'scaled:center'), scale = attr(train.data.scaled, 'scaled:scale'))

  # collect scale stats for y-variable, so we can unscale prediction later
  y.scale.sd = attr(train.labels.scaled, 'scaled:scale')[c("y")]
  y.scale.mean = attr(train.labels.scaled, 'scaled:center')[c("y")]

  NN <- ANN2::neuralnetwork(X = train.features, y = train.labels.scaled, regression=TRUE,
                            standardize = FALSE, hidden.layers = l.layers, # c(32),
                            loss.type = 'absolute', optim.type = 'sgd', sgd.momentum = 0.9,
                            activ.functions = 'relu', learn.rates = 0.0001,
                            batch.size=as.integer(nrow(train.features)*.9), val.prop = 0.1,
                            n.epochs=1000, rmsprop.decay=0.999,
                            verbose = FALSE)

  prediction <- as.numeric(predict(NN, newdata = test.features))

  # re-scale prediction for later comparison
  prediction <- prediction * y.scale.sd + y.scale.mean

  return ( list( prediction = prediction, var.importance = c(), extra.info = list() ) )

}


func.fit.model.randomforest <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {

  l.trees <- 1000

  df.train <- i.recipe %>% bake(new_data=NULL)
  df.test  <- i.recipe %>% bake(new_data=i.data %>% assessment)

  # check if the test data has missing data (this is not caught by recipe/bake
  #   because the recipe is only build with the training data)
  missing_cols_test <- df.test %>%  select_if(~any(is.na(.))) %>% colnames
  # remove missing cols from both datasets
  if(length(missing_cols_test > 0)) {
    df.train <- df.train %>% select( -any_of(missing_cols_test) )
    df.test <- df.test %>% select( -any_of(missing_cols_test) )
  }

  # split the dataset into X and Y separately
  train.features <- df.train %>% select(-c(y, date))
  test.features <- df.test %>% select(-c(y, date))

  train.labels <- df.train %>% select(y)
#  test.labels <- df.test %>% select(y)

  train.labels.scaled <- scale(train.labels)

  train.data.combined <- cbind(train.features, train.labels.scaled)

  # collect scale stats for y-variable, so we can unscale prediction later
  y.scale.sd = attr(train.labels.scaled, 'scaled:scale')[c("y")]
  y.scale.mean = attr(train.labels.scaled, 'scaled:center')[c("y")]

  model <- ranger(y~., data = train.data.combined, importance = 'permutation', num.trees=l.trees)#, regularization.factor=.99)

  prediction <- predict(model, test.features)$pred
  prediction <- prediction * y.scale.sd + y.scale.mean

  return ( list( prediction = prediction, var.importance = c(), extra.info = list() ) )

}


func.fit.lightgbm <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {

  df.train <- i.recipe %>% bake(new_data=NULL)
  df.test  <- i.recipe %>% bake(new_data=i.data %>% assessment)

  # check if the test data has missing data (this is not caught by recipe/bake
  #   because the recipe is only build with the training data)
  missing_cols_test <- df.test %>%  select_if(~any(is.na(.))) %>% colnames
  # remove missing cols from both datasets
  if(length(missing_cols_test > 0)) {
    df.train <- df.train %>% select( -any_of(missing_cols_test) )
    df.test <- df.test %>% select( -any_of(missing_cols_test) )
  }

  # split the dataset into X and Y separately
  train.features <- df.train %>% select(-c(y, date))
  test.features <- df.test %>% select(-c(y, date))

  train.labels <- df.train %>% select(y)
  test.labels <- df.test %>% select(y)

  train.labels.scaled <- scale(train.labels)

  # collect scale stats for y-variable, so we can unscale prediction later
  y.scale.sd = attr(train.labels.scaled, 'scaled:scale')[c("y")]
  y.scale.mean = attr(train.labels.scaled, 'scaled:center')[c("y")]

  test.labels.scaled <- (test.labels - y.scale.mean ) / y.scale.sd

  # prepare datasets for lightGBM
  dtrain = lgb.Dataset(as.matrix(train.features), label = as.matrix(train.labels.scaled))
#  dtest = lgb.Dataset.create.valid(dtrain, as.matrix(test.features), label = as.matrix(test.labels.scaled))

  light_gbm_tuned <- lgb.train(
    params = list(
      objective = "regression",
      metric = "l2",
      max_depth = 10,
      num_leaves = 20,
      num_iterations = 50,
      #early_stopping_rounds= 5,
      learning_rate = .25
      #feature_fraction = .9
      ),
  #  valids = list(test = dtest),
    data = dtrain,
    verbose=-1 #turn off info and warning messages
  )

  prediction <- predict(light_gbm_tuned, as.matrix(test.features))
  prediction <- prediction * y.scale.sd + y.scale.mean

  return ( list( prediction = prediction, var.importance = c(), extra.info = list() ) )

}


func.fit.NN5.keras <- function(i.data, i.recipe, i.xargs=NULL, i.var.imp = FALSE, ...) {
  df.train <- i.recipe %>% bake(new_data=NULL)
  df.test  <- i.recipe %>% bake(new_data=i.data %>% assessment)

  # check if the test data has missing data (this is not caught by recipe/bake
  #   because the recipe is only build with the training data)
  missing_cols_test <- df.test %>%  select_if(~any(is.na(.))) %>% colnames
  # remove missing cols from both datasets
  if(length(missing_cols_test > 0)) {
    df.train <- df.train %>% select( -any_of(missing_cols_test) )
    df.test <- df.test %>% select( -any_of(missing_cols_test) )
  }

  # split the dataset into X and Y separately
  train.features <- df.train %>% select(-c(y, date))
  test.features <- df.test %>% select(-c(y, date))

  train.labels <- df.train %>% select(y)
#  test.labels <- df.test %>% select(y)

  train.labels.scaled <- scale(train.labels)
  y.scale.sd = attr(train.labels.scaled, 'scaled:scale')[c("y")]
  y.scale.mean = attr(train.labels.scaled, 'scaled:center')[c("y")]

# build more complex multi-layer model
  multilevel.model <- keras_model_sequential() %>%
    #normalizer() %>%   #not required because data was already normalized by recipe
    layer_dense(32, activation = 'relu') %>%
    layer_dense(16, activation = 'relu') %>%
    layer_dense(8, activation = 'relu') %>%
    layer_dense(4, activation = 'relu') %>%
    layer_dense(2, activation = 'relu') %>%
    layer_dense(1)

  multilevel.model %>% compile(
    loss = 'mean_squared_error',
    optimizer = optimizer_sgd(
        learning_rate = 0.0001,  #0.01
        momentum = 0.9, # 0.9
        decay = 0.999 )  # 0.999
    )

  multilevel.history <- multilevel.model %>% fit(
    as.matrix(train.features),
    as.matrix(train.labels),
    validation_split = 0.2,
    verbose = 0,
    epochs = 1000,
    # callbacks
    callbacks = list(
      callback_early_stopping(monitor = "val_loss",
                                        min_delta = 0.0001,
                                        patience = 100,
                                        restore_best_weights = TRUE,
                                        verbose = 0)
    )
  )

  prediction <- as.numeric(predict(multilevel.model, as.matrix(test.features)))

  # re-scale prediction for later comparison
  prediction <- prediction * y.scale.sd + y.scale.mean

  return ( list( prediction = prediction, var.importance = c(), extra.info = list() ) )

}


#############################
#      Parameter Handling   #
#############################

option_list = list(
  make_option(c("-f", "--file"), type="character", default=NULL,
              help="dataset file name", metavar="character"),
  make_option(c("-m", "--method"), type="character", default=NULL,
              help="method to be applied [e.g. NN5, randomforest, lasso]", metavar="character"),
  make_option(c("-r", "--rcum"), type="character", default="rolling",
              help="data set step split method [rolling or recursive], default [%default]", metavar="character"),
  make_option(c("-y", "--yvar"), type="character", default=NULL,
              help="column name of the outcome variable", metavar="character"),
  make_option(c("-p", "--paramadj"), type="character", default=NULL,
              help="parameter adjustment method [allvars, pca, pls]", metavar="character"),
  make_option(c("-w", "--window"), type="integer", default=NULL,
              help="size of the training window", metavar="number"),
  make_option(c("-g", "--gap"), type="integer", default=NULL,
              help="size of the lag between training and test data for multi-period returns", metavar="number"),
  make_option(c("-x", "--prefix"), type="character", default=NULL,
              help="prefix for output file", metavar="character"),
  make_option(c("-v", "--varimp"), type="character", default=NULL,
              help="generate variable importance (T/F) default F", metavar="character")

);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if (is.null(opt$file)){
  print_help(opt_parser)
  stop("Supply input data file with -f argument.", call.=FALSE)
}

if (is.null(opt$method)){
  print_help(opt_parser)
  stop("Supply method  with -m argument.", call.=FALSE)
}

if (is.null(opt$rcum)){
  print_help(opt_parser)
  stop("Supply data step split method with -r argument (either rolling or recursive).", call.=FALSE)
}

if (is.null(opt$yvar)){
  print_help(opt_parser)
  stop("Supply outcome variable with -y argument.", call.=FALSE)
}

if (is.null(opt$window)){
  print_help(opt_parser)
  stop("Supply training window size with -w argument.", call.=FALSE)
}

if (is.null(opt$gap)){
  opt$gap = 0
}

if (is.null(opt$paramadj)){
  opt$paramadj = "allvars"
}

if (is.null(opt$varimp)){
  opt$varimp = F
} else {
  if(opt$varimp == "T"){
    opt$varimp = T
  }
}


#############################
#      Execution Script     #
#############################
#--file="Dissertation/usa.df.analysis.rda"
#--method=NN5
#--rcum=rolling
#--yvar=ret_exc_lead1m
#--paramadj=pca
#--window=120
#--gap=0

i.data <- as_tibble(func.load.dataset(opt$file)) # load("Dissertation/can.dataset.rda")
#i.data$date = as.Date(as.character(i.data$date),format="%Y%m%d")  # data format required for tibble date column

# select DV - monthly returns
 i.data$y = i.data %>% pull(opt$yvar)

# remove all other DVs from the data frame
 i.data = i.data %>% select(-any_of(c("ret_geom_3m","ret_geom_12m","ret","ret_exc","ret_exc_lead1m", opt$yvar)))

# select appropriate function based on parameter
l.func.ref <- tibble(parameter = c( "NN5",
                                    "NN5KERAS",
                                    "LASSO",
                                    "RIDGE",
                                    "ENET",
                                    "RF",
                                    #"SCF",
                                    "SCFMEAN",
                                    "SCFMEDIAN",
                                    "SCFTRIMMEDMEAN",
                                    "LIGHTGBM"),
                     value    = c(  func.fit.model.NN5,
                                    func.fit.NN5.keras,
                                    func.fit.lasso,
                                    func.fit.ridge,
                                    func.fit.enet,
                                    func.fit.model.randomforest,
                                    #func.fit.scf,
                                    func.fit.scf.mean,
                                    func.fit.scf.median,
                                    func.fit.scf.trimmedmean,
                                    func.fit.lightgbm))
l.func <- ( l.func.ref %>% filter (parameter == opt$method) %>% pull(value) ) [[1]]

# determine if rolling or recursive method is desired
l.rcum.ref <- tibble(parameter=c("rolling", "recursive"),
                     value   = c(FALSE, TRUE))
l.rcum <- l.rcum.ref %>% filter (parameter == opt$rcum) %>% pull(value)

# rolling split function resampling
roll_rs <- rolling_origin(
  i.data,
  initial = opt$window,
  assess = 1,
  lag = opt$gap, # no lag between analysis and assessment set requiredtest
  cumulative = l.rcum
  )

l.splits = roll_rs$splits #[1:10]  #reduce size of splits list for testing

print(paste("[*] Starting: ", paste(opt$prefix,opt$method,opt$rcum,opt$yvar,opt$paramadj,opt$window,opt$gap,"rda",sep="."),sep=""))

# trigger parallel execution with progress bar
with_progress({
  p <- progressor(steps = length(l.splits))

  l.results <- future_map(.x=l.splits, i.func=l.func, i.paramadj = opt$paramadj,
    .f=function(i.split, i.func, i.paramadj) {
      p() # iterate progress bar

      # preprocess data based on parameter
      l.recipe = func.data.prep(i.split, i.paramadj)

      # execute method and generate prediction
      res.pred.varimp = i.func(i.split, i.recipe = l.recipe, i.var.imp=opt$varimp)

      # prepare return data
      date = (i.split  %>% assessment)$date
      prediction = res.pred.varimp$prediction
      mean.train.data =  as.numeric ( i.split %>% analysis %>% dplyr::summarize(Mean = mean(y, na.rm=TRUE)))
      test.value = (i.split %>% assessment)$y
      return (list( prediction = data.frame(date,test.value, mean.train.data, prediction),
                    var.importance = res.pred.varimp$var.importance,
                    extra.info = res.pred.varimp$extra.info))
    },
    .options = furrr_options(seed = 1234)
  )
}, enable = TRUE)

## format output as data frame.options (list of lists by default)
#df.result <- data.frame(do.call(rbind, l.results)) %>%
#  unnest(cols = everything())

l.predictions <- l.results %>% map(1)
# combine the results into one big data frame
df.results <- Reduce(function(dtf1, dtf2) merge(dtf1, dtf2, all = TRUE), l.predictions)

# extract all var.importance matrices and bind them together in one big list
l.var.importance <- ( l.results %>% map(2) ) %>% dplyr::bind_rows()

# do the same with any extra info provided by the models
# extra info can be a deep structured list, can't generically assume flat table
l.extra.info <- l.results %>% map(3)

e.result <- list( values = df.results, var.importance = l.var.importance, extra.info = l.extra.info )

# save file with correct filename
l.filename = paste(opt$prefix,opt$method,opt$rcum,opt$yvar,opt$paramadj,opt$window,opt$gap,"rda",sep=".")
save(e.result, file = l.filename)
print(paste("[+] Completed: ", paste(opt$prefix,opt$method,opt$rcum,opt$yvar,opt$paramadj,opt$window,opt$gap,"rda",sep="."),sep=""))
