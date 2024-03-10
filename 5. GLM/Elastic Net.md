```R
# The easiest way to get recipes is to install all of tidymodels:
# install.packages("tidymodels")
install.packages("multilevelmod")
options(encoding = 'UTF-8')
#Loading all the necessary packages
if (!require("caret")) install.packages("caret")
if (!require("recipes")) install.packages("recipes")
if (!require("visreg")) install.packages("visreg")
if (!require("MASS")) install.packages("MASS")
if (!require("glmnet")) install.packages("glmnet")
if (!require("jtools")) install.packages("jtools")
if (!require("scales")) install.packages("scales")
if (!require("forcats")) install.packages("forcats")
if (!require("stringr")) install.packages("stringr")
if (!require("poissonreg")) install.packages("poissonreg")



require("caret")
require("recipes")
require("visreg")
require("MASS")
require("glmnet")
require("jtools")
require("scales")
require("forcats")
require("stringr")
require("arrow")
require("forcats")
require("doParallel")
require("yardstick")
require("parsnip")
require("workflows")
require("poissonreg")
require("rsample")
require("tune")
require("yardstick")

options(repr.plot.width = 8, repr.plot.height = 6, repr.plot.res = 150);
```

    Installation du package dans 'C:/Users/Florian/Documents/R/win-library/4.1'
    (car 'lib' n'est pas spécifié)
    
    

    package 'multilevelmod' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\Florian\AppData\Local\Temp\Rtmp4GxSpH\downloaded_packages
    

    Le chargement a nécessité le package : caret
    
    Le chargement a nécessité le package : ggplot2
    
    Le chargement a nécessité le package : lattice
    
    Le chargement a nécessité le package : recipes
    
    Le chargement a nécessité le package : dplyr
    
    
    Attachement du package : 'dplyr'
    
    
    Les objets suivants sont masqués depuis 'package:stats':
    
        filter, lag
    
    
    Les objets suivants sont masqués depuis 'package:base':
    
        intersect, setdiff, setequal, union
    
    
    
    Attachement du package : 'recipes'
    
    
    L'objet suivant est masqué depuis 'package:stats':
    
        step
    
    
    Le chargement a nécessité le package : visreg
    
    Le chargement a nécessité le package : MASS
    
    
    Attachement du package : 'MASS'
    
    
    L'objet suivant est masqué depuis 'package:dplyr':
    
        select
    
    
    Le chargement a nécessité le package : glmnet
    
    Le chargement a nécessité le package : Matrix
    
    Loaded glmnet 4.1-3
    
    Le chargement a nécessité le package : jtools
    
    Le chargement a nécessité le package : scales
    
    Le chargement a nécessité le package : forcats
    
    Le chargement a nécessité le package : stringr
    
    
    Attachement du package : 'stringr'
    
    
    L'objet suivant est masqué depuis 'package:recipes':
    
        fixed
    
    
    Le chargement a nécessité le package : poissonreg
    
    Le chargement a nécessité le package : parsnip
    
    Le chargement a nécessité le package : arrow
    
    
    Attachement du package : 'arrow'
    
    
    L'objet suivant est masqué depuis 'package:utils':
    
        timestamp
    
    
    Le chargement a nécessité le package : doParallel
    
    Le chargement a nécessité le package : foreach
    
    Le chargement a nécessité le package : iterators
    
    Le chargement a nécessité le package : parallel
    
    Le chargement a nécessité le package : yardstick
    
    For binary classification, the first factor level is assumed to be the event.
    Use the argument `event_level = "second"` to alter this as needed.
    
    
    Attachement du package : 'yardstick'
    
    
    L'objet suivant est masqué depuis 'package:jtools':
    
        get_weights
    
    
    Les objets suivants sont masqués depuis 'package:caret':
    
        precision, recall, sensitivity, specificity
    
    
    Le chargement a nécessité le package : workflows
    
    Le chargement a nécessité le package : rsample
    
    Le chargement a nécessité le package : tune
    
    


```R
dataset = read_parquet(file = "../data/dataset.parquet")

set.seed(21)
in_training = createDataPartition(dataset$ClaimNb, times = 1, p = 0.8, list = FALSE)
training_set = dataset[in_training, ]
testing_set = dataset[-in_training, ]
```

# Lasso 


```R
ptn=Sys.time()
x = model.matrix(ClaimNb ~ 0 + Power  * Region + Power*Brand + Power*Gas +  Region* Brand + Region* Gas + Brand*Gas,
                 data=training_set)
set.seed(542)
folds = createFolds(training_set$ClaimNb, 5, list=FALSE)

set.seed(58)
m.lasso.0.cv = cv.glmnet(x, y = training_set$ClaimNb, 
                         offset = log(training_set$Exposure),
       family = "poisson",
       alpha = 1, #LASSO = 1, Ridge = 0,
       nfolds = 5,
       foldid = folds,
       maxit=10^4,
       nlambda = 25)


ptn_1 = Sys.time() - ptn
ptn_1
```


    Time difference of 6.335381 mins



```R
plot(m.lasso.0.cv)
```


    
![png](Elastic%20Net_files/Elastic%20Net_4_0.png)
    



```R
coef(m.lasso.0.cv, s = "lambda.min")
```


    274 x 1 sparse Matrix of class "dgCMatrix"
                                                                                s1
    (Intercept)                                                      -2.5394767393
    Powerd                                                           -0.1033242799
    Powere                                                            .           
    Powerf                                                            .           
    Powerg                                                            .           
    Powerh                                                            .           
    Poweri                                                            .           
    Powerj                                                            .           
    Powerk                                                            .           
    Powerl                                                            .           
    Powerm                                                            .           
    Powern                                                            .           
    Powero                                                            .           
    RegionBasse-Normandie                                             .           
    RegionBretagne                                                    .           
    RegionCentre                                                     -0.0409454592
    RegionHaute-Normandie                                             .           
    RegionIle-de-France                                               0.1752546086
    RegionLimousin                                                    0.0093840912
    RegionNord-Pas-de-Calais                                          0.0455362903
    RegionPays-de-la-Loire                                            .           
    RegionPoitou-Charentes                                            .           
    BrandJapanese (except Nissan) or Korean                          -0.1364607840
    BrandMercedes, Chrysler or BMW                                    .           
    BrandOpel, General Motors or Ford                                 0.0159814997
    Brandother                                                        .           
    BrandRenault, Nissan or Citroen                                   .           
    BrandVolkswagen, Audi, Skoda or Seat                              .           
    GasRegular                                                       -0.0478934072
    Powere:RegionBasse-Normandie                                      .           
    Powerf:RegionBasse-Normandie                                      .           
    Powerg:RegionBasse-Normandie                                      .           
    Powerh:RegionBasse-Normandie                                      .           
    Poweri:RegionBasse-Normandie                                      .           
    Powerj:RegionBasse-Normandie                                      .           
    Powerk:RegionBasse-Normandie                                      .           
    Powerl:RegionBasse-Normandie                                      .           
    Powerm:RegionBasse-Normandie                                      0.4479234685
    Powern:RegionBasse-Normandie                                      .           
    Powero:RegionBasse-Normandie                                      .           
    Powere:RegionBretagne                                             .           
    Powerf:RegionBretagne                                             .           
    Powerg:RegionBretagne                                            -0.0733383587
    Powerh:RegionBretagne                                             .           
    Poweri:RegionBretagne                                             0.0467020130
    Powerj:RegionBretagne                                             0.0130839942
    Powerk:RegionBretagne                                             0.1340007764
    Powerl:RegionBretagne                                             .           
    Powerm:RegionBretagne                                             .           
    Powern:RegionBretagne                                             0.1088958164
    Powero:RegionBretagne                                             .           
    Powere:RegionCentre                                               .           
    Powerf:RegionCentre                                               .           
    Powerg:RegionCentre                                               .           
    Powerh:RegionCentre                                               .           
    Poweri:RegionCentre                                               .           
    Powerj:RegionCentre                                               .           
    Powerk:RegionCentre                                               .           
    Powerl:RegionCentre                                              -0.0713801368
    Powerm:RegionCentre                                               .           
    Powern:RegionCentre                                               0.0324150018
    Powero:RegionCentre                                               .           
    Powere:RegionHaute-Normandie                                      .           
    Powerf:RegionHaute-Normandie                                      .           
    Powerg:RegionHaute-Normandie                                      .           
    Powerh:RegionHaute-Normandie                                      .           
    Poweri:RegionHaute-Normandie                                      .           
    Powerj:RegionHaute-Normandie                                     -0.0300365589
    Powerk:RegionHaute-Normandie                                      0.0605117177
    Powerl:RegionHaute-Normandie                                      .           
    Powerm:RegionHaute-Normandie                                      .           
    Powern:RegionHaute-Normandie                                      .           
    Powero:RegionHaute-Normandie                                      .           
    Powere:RegionIle-de-France                                        .           
    Powerf:RegionIle-de-France                                        .           
    Powerg:RegionIle-de-France                                       -0.0227732494
    Powerh:RegionIle-de-France                                        .           
    Poweri:RegionIle-de-France                                        .           
    Powerj:RegionIle-de-France                                        .           
    Powerk:RegionIle-de-France                                        0.0481364720
    Powerl:RegionIle-de-France                                        .           
    Powerm:RegionIle-de-France                                        .           
    Powern:RegionIle-de-France                                        .           
    Powero:RegionIle-de-France                                        .           
    Powere:RegionLimousin                                             0.2608401749
    Powerf:RegionLimousin                                             .           
    Powerg:RegionLimousin                                             .           
    Powerh:RegionLimousin                                             0.2021079974
    Poweri:RegionLimousin                                             .           
    Powerj:RegionLimousin                                             .           
    Powerk:RegionLimousin                                             .           
    Powerl:RegionLimousin                                             0.1383863510
    Powerm:RegionLimousin                                             .           
    Powern:RegionLimousin                                             1.5620413323
    Powero:RegionLimousin                                             .           
    Powere:RegionNord-Pas-de-Calais                                   .           
    Powerf:RegionNord-Pas-de-Calais                                   0.0189618631
    Powerg:RegionNord-Pas-de-Calais                                   .           
    Powerh:RegionNord-Pas-de-Calais                                   .           
    Poweri:RegionNord-Pas-de-Calais                                   .           
    Powerj:RegionNord-Pas-de-Calais                                   .           
    Powerk:RegionNord-Pas-de-Calais                                   .           
    Powerl:RegionNord-Pas-de-Calais                                   .           
    Powerm:RegionNord-Pas-de-Calais                                   .           
    Powern:RegionNord-Pas-de-Calais                                   .           
    Powero:RegionNord-Pas-de-Calais                                   .           
    Powere:RegionPays-de-la-Loire                                     .           
    Powerf:RegionPays-de-la-Loire                                     .           
    Powerg:RegionPays-de-la-Loire                                     .           
    Powerh:RegionPays-de-la-Loire                                     .           
    Poweri:RegionPays-de-la-Loire                                     .           
    Powerj:RegionPays-de-la-Loire                                     .           
    Powerk:RegionPays-de-la-Loire                                     .           
    Powerl:RegionPays-de-la-Loire                                     .           
    Powerm:RegionPays-de-la-Loire                                     0.3450769607
    Powern:RegionPays-de-la-Loire                                    -0.3456796351
    Powero:RegionPays-de-la-Loire                                     .           
    Powere:RegionPoitou-Charentes                                     .           
    Powerf:RegionPoitou-Charentes                                     .           
    Powerg:RegionPoitou-Charentes                                     .           
    Powerh:RegionPoitou-Charentes                                    -0.0242924158
    Poweri:RegionPoitou-Charentes                                     .           
    Powerj:RegionPoitou-Charentes                                     0.0013611703
    Powerk:RegionPoitou-Charentes                                     .           
    Powerl:RegionPoitou-Charentes                                     .           
    Powerm:RegionPoitou-Charentes                                     .           
    Powern:RegionPoitou-Charentes                                     .           
    Powero:RegionPoitou-Charentes                                     .           
    Powere:BrandJapanese (except Nissan) or Korean                   -0.1858136423
    Powerf:BrandJapanese (except Nissan) or Korean                    .           
    Powerg:BrandJapanese (except Nissan) or Korean                   -0.0360671896
    Powerh:BrandJapanese (except Nissan) or Korean                    .           
    Poweri:BrandJapanese (except Nissan) or Korean                    .           
    Powerj:BrandJapanese (except Nissan) or Korean                    .           
    Powerk:BrandJapanese (except Nissan) or Korean                    .           
    Powerl:BrandJapanese (except Nissan) or Korean                    .           
    Powerm:BrandJapanese (except Nissan) or Korean                    .           
    Powern:BrandJapanese (except Nissan) or Korean                    .           
    Powero:BrandJapanese (except Nissan) or Korean                    .           
    Powere:BrandMercedes, Chrysler or BMW                             .           
    Powerf:BrandMercedes, Chrysler or BMW                             0.2016839858
    Powerg:BrandMercedes, Chrysler or BMW                             0.0041495492
    Powerh:BrandMercedes, Chrysler or BMW                             0.0775130552
    Poweri:BrandMercedes, Chrysler or BMW                             .           
    Powerj:BrandMercedes, Chrysler or BMW                             .           
    Powerk:BrandMercedes, Chrysler or BMW                            -0.0834248330
    Powerl:BrandMercedes, Chrysler or BMW                            -0.0408418679
    Powerm:BrandMercedes, Chrysler or BMW                             .           
    Powern:BrandMercedes, Chrysler or BMW                            -0.0044918949
    Powero:BrandMercedes, Chrysler or BMW                             .           
    Powere:BrandOpel, General Motors or Ford                          0.0128431207
    Powerf:BrandOpel, General Motors or Ford                          .           
    Powerg:BrandOpel, General Motors or Ford                          .           
    Powerh:BrandOpel, General Motors or Ford                          .           
    Poweri:BrandOpel, General Motors or Ford                          .           
    Powerj:BrandOpel, General Motors or Ford                          0.1475029614
    Powerk:BrandOpel, General Motors or Ford                          .           
    Powerl:BrandOpel, General Motors or Ford                          .           
    Powerm:BrandOpel, General Motors or Ford                          .           
    Powern:BrandOpel, General Motors or Ford                          .           
    Powero:BrandOpel, General Motors or Ford                          0.8131944587
    Powere:Brandother                                                -0.0079737540
    Powerf:Brandother                                                 .           
    Powerg:Brandother                                                 .           
    Powerh:Brandother                                                -0.0627668306
    Poweri:Brandother                                                 .           
    Powerj:Brandother                                                 .           
    Powerk:Brandother                                                 0.3503050904
    Powerl:Brandother                                                 0.0901656283
    Powerm:Brandother                                                 .           
    Powern:Brandother                                                 .           
    Powero:Brandother                                                 .           
    Powere:BrandRenault, Nissan or Citroen                            .           
    Powerf:BrandRenault, Nissan or Citroen                            .           
    Powerg:BrandRenault, Nissan or Citroen                           -0.0593211013
    Powerh:BrandRenault, Nissan or Citroen                            .           
    Poweri:BrandRenault, Nissan or Citroen                            0.0505803657
    Powerj:BrandRenault, Nissan or Citroen                            .           
    Powerk:BrandRenault, Nissan or Citroen                            .           
    Powerl:BrandRenault, Nissan or Citroen                            0.3115151204
    Powerm:BrandRenault, Nissan or Citroen                            0.0420458709
    Powern:BrandRenault, Nissan or Citroen                            0.1861175160
    Powero:BrandRenault, Nissan or Citroen                            .           
    Powere:BrandVolkswagen, Audi, Skoda or Seat                       0.0741928826
    Powerf:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powerg:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powerh:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Poweri:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powerj:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powerk:BrandVolkswagen, Audi, Skoda or Seat                       0.1664306480
    Powerl:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powerm:BrandVolkswagen, Audi, Skoda or Seat                      -0.2089844796
    Powern:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powero:BrandVolkswagen, Audi, Skoda or Seat                       .           
    Powere:GasRegular                                                -0.0696922929
    Powerf:GasRegular                                                -0.1008120428
    Powerg:GasRegular                                                 .           
    Powerh:GasRegular                                                 .           
    Poweri:GasRegular                                                 .           
    Powerj:GasRegular                                                 0.0284692227
    Powerk:GasRegular                                                 .           
    Powerl:GasRegular                                                 .           
    Powerm:GasRegular                                                 .           
    Powern:GasRegular                                                 .           
    Powero:GasRegular                                                 .           
    RegionBasse-Normandie:BrandJapanese (except Nissan) or Korean    -0.1095517558
    RegionBretagne:BrandJapanese (except Nissan) or Korean            0.0804593406
    RegionCentre:BrandJapanese (except Nissan) or Korean              .           
    RegionHaute-Normandie:BrandJapanese (except Nissan) or Korean     .           
    RegionIle-de-France:BrandJapanese (except Nissan) or Korean       .           
    RegionLimousin:BrandJapanese (except Nissan) or Korean            .           
    RegionNord-Pas-de-Calais:BrandJapanese (except Nissan) or Korean  .           
    RegionPays-de-la-Loire:BrandJapanese (except Nissan) or Korean    .           
    RegionPoitou-Charentes:BrandJapanese (except Nissan) or Korean    .           
    RegionBasse-Normandie:BrandMercedes, Chrysler or BMW             -0.0707676581
    RegionBretagne:BrandMercedes, Chrysler or BMW                    -0.1440561921
    RegionCentre:BrandMercedes, Chrysler or BMW                       .           
    RegionHaute-Normandie:BrandMercedes, Chrysler or BMW              .           
    RegionIle-de-France:BrandMercedes, Chrysler or BMW                0.0293028541
    RegionLimousin:BrandMercedes, Chrysler or BMW                     0.2781129868
    RegionNord-Pas-de-Calais:BrandMercedes, Chrysler or BMW           .           
    RegionPays-de-la-Loire:BrandMercedes, Chrysler or BMW             .           
    RegionPoitou-Charentes:BrandMercedes, Chrysler or BMW             .           
    RegionBasse-Normandie:BrandOpel, General Motors or Ford           .           
    RegionBretagne:BrandOpel, General Motors or Ford                  .           
    RegionCentre:BrandOpel, General Motors or Ford                    .           
    RegionHaute-Normandie:BrandOpel, General Motors or Ford           .           
    RegionIle-de-France:BrandOpel, General Motors or Ford             0.0701375283
    RegionLimousin:BrandOpel, General Motors or Ford                  .           
    RegionNord-Pas-de-Calais:BrandOpel, General Motors or Ford        0.0066780381
    RegionPays-de-la-Loire:BrandOpel, General Motors or Ford          .           
    RegionPoitou-Charentes:BrandOpel, General Motors or Ford          .           
    RegionBasse-Normandie:Brandother                                  .           
    RegionBretagne:Brandother                                         .           
    RegionCentre:Brandother                                          -0.0009409526
    RegionHaute-Normandie:Brandother                                 -0.0649813226
    RegionIle-de-France:Brandother                                    0.1477005052
    RegionLimousin:Brandother                                         0.1871616319
    RegionNord-Pas-de-Calais:Brandother                               0.1539171473
    RegionPays-de-la-Loire:Brandother                                 .           
    RegionPoitou-Charentes:Brandother                                 .           
    RegionBasse-Normandie:BrandRenault, Nissan or Citroen             .           
    RegionBretagne:BrandRenault, Nissan or Citroen                   -0.0435360869
    RegionCentre:BrandRenault, Nissan or Citroen                     -0.1516769073
    RegionHaute-Normandie:BrandRenault, Nissan or Citroen             .           
    RegionIle-de-France:BrandRenault, Nissan or Citroen               .           
    RegionLimousin:BrandRenault, Nissan or Citroen                    0.1235088848
    RegionNord-Pas-de-Calais:BrandRenault, Nissan or Citroen          0.0678884501
    RegionPays-de-la-Loire:BrandRenault, Nissan or Citroen            .           
    RegionPoitou-Charentes:BrandRenault, Nissan or Citroen            .           
    RegionBasse-Normandie:BrandVolkswagen, Audi, Skoda or Seat        .           
    RegionBretagne:BrandVolkswagen, Audi, Skoda or Seat               .           
    RegionCentre:BrandVolkswagen, Audi, Skoda or Seat                 .           
    RegionHaute-Normandie:BrandVolkswagen, Audi, Skoda or Seat        .           
    RegionIle-de-France:BrandVolkswagen, Audi, Skoda or Seat          0.0858737158
    RegionLimousin:BrandVolkswagen, Audi, Skoda or Seat               0.0526893515
    RegionNord-Pas-de-Calais:BrandVolkswagen, Audi, Skoda or Seat     0.1920193083
    RegionPays-de-la-Loire:BrandVolkswagen, Audi, Skoda or Seat       0.0462851416
    RegionPoitou-Charentes:BrandVolkswagen, Audi, Skoda or Seat       .           
    RegionBasse-Normandie:GasRegular                                  .           
    RegionBretagne:GasRegular                                        -0.0807079635
    RegionCentre:GasRegular                                           .           
    RegionHaute-Normandie:GasRegular                                  .           
    RegionIle-de-France:GasRegular                                    .           
    RegionLimousin:GasRegular                                         0.0040850621
    RegionNord-Pas-de-Calais:GasRegular                               .           
    RegionPays-de-la-Loire:GasRegular                                -0.0017936288
    RegionPoitou-Charentes:GasRegular                                 .           
    BrandJapanese (except Nissan) or Korean:GasRegular                .           
    BrandMercedes, Chrysler or BMW:GasRegular                         .           
    BrandOpel, General Motors or Ford:GasRegular                      .           
    Brandother:GasRegular                                            -0.1191474871
    BrandRenault, Nissan or Citroen:GasRegular                       -0.0083848564
    BrandVolkswagen, Audi, Skoda or Seat:GasRegular                   .           


# Ridge


```R
ptn=Sys.time()
set.seed(58)
m.ridge.0.cv = cv.glmnet(x, y = training_set$ClaimNb, offset = log(training_set$Exposure),
       family = "poisson",
       alpha = 0, #LASSO = 1, Ridge = 0,
       nfolds = 5,
       foldid = folds,
       maxit = 10^3,
       nlambda = 25)

ptn_1 = Sys.time() - ptn
ptn_1
```


    Time difference of 4.243747 mins



```R
plot(m.ridge.0.cv)
```


    
![png](Elastic%20Net_files/Elastic%20Net_8_0.png)
    


# Elastic Net


```R
ptn=Sys.time()
set.seed(58)
m.elasticnet.0.cv = cv.glmnet(x, y = training_set$ClaimNb, offset = log(training_set$Exposure),
       family = "poisson",
       alpha = 0.5, #LASSO = 1, Ridge = 0,
       nfolds = 5,
       foldid = folds,
       maxit = 10^3,
       nlambda = 25)

ptn_1 = Sys.time() - ptn
ptn_1
```


    Time difference of 6.538676 mins



```R
plot(m.elasticnet.0.cv)
```


    
![png](Elastic%20Net_files/Elastic%20Net_11_0.png)
    


# Comparison with GLM


```R
x_test = model.matrix(ClaimNb ~ 0 + Power * Region + Power * Brand + Power * Gas +
    Region * Brand + Region * Gas + Brand * Gas, data = testing_set)


2 * (sum(dpois(x = testing_set$ClaimNb, lambda = testing_set$ClaimNb, log = TRUE)) -
    sum(dpois(x = testing_set$ClaimNb, lambda = predict(m.lasso.0.cv, newx = x_test,
        newoffset = log(testing_set$Exposure), s = m.lasso.0.cv$lambda.min, type = "response"),
        log = TRUE)))
```


20769.3114660243



```R
2 * (sum(dpois(x = testing_set$ClaimNb, lambda = testing_set$ClaimNb, log = TRUE)) -
    sum(dpois(x = testing_set$ClaimNb, lambda = predict(m.ridge.0.cv, newx = x_test,
        newoffset = log(testing_set$Exposure), s = m.ridge.0.cv$lambda.min, type = "response"),
        log = TRUE)))
```


20769.927705197



```R
2 * (sum(dpois(x = testing_set$ClaimNb, lambda = testing_set$ClaimNb, log = TRUE)) -
    sum(dpois(x = testing_set$ClaimNb, lambda = predict(m.elasticnet.0.cv, newx = x_test,
        newoffset = log(testing_set$Exposure), s = m.elasticnet.0.cv$lambda.min,
        type = "response"), log = TRUE)))
```


20769.5785835634


# Experiment

Let us construct the factors a bit differently. We will only use the variables DriverAge and CarAge for illustration here.


```R
for (age in (18:99)){
    training_set[paste0("DriverAge_", age)] = 1*(training_set$DriverAge <= age)
    testing_set[paste0("DriverAge_", age)] = 1*(testing_set$DriverAge <= age)
}

for (vehage in (0:25)){
    training_set[paste0("CarAge_", vehage)] = 1*(training_set$CarAge <= vehage)
    testing_set[paste0("CarAge_", vehage)] = 1*(testing_set$CarAge <= vehage)
}

```


```R
ptn=Sys.time()
model_exp_x = model.matrix(ClaimNb ~ 0 + DriverAge_18 + DriverAge_19 + DriverAge_20 + DriverAge_21 + DriverAge_22 + 
                 DriverAge_23 + DriverAge_24 + DriverAge_25 + DriverAge_26 + DriverAge_27 + DriverAge_28 + 
                 DriverAge_29 + DriverAge_30 + DriverAge_31 + DriverAge_32 + DriverAge_33 + DriverAge_34 + 
                 DriverAge_35 + DriverAge_36 + DriverAge_37 + DriverAge_38 + DriverAge_39 + DriverAge_40 + 
                 DriverAge_41 + DriverAge_42 + DriverAge_43 + DriverAge_44 + DriverAge_45 + DriverAge_46 + 
                 DriverAge_47 + DriverAge_48 + DriverAge_49 + DriverAge_50 + DriverAge_51 + DriverAge_52 + 
                 DriverAge_53 + DriverAge_54 + DriverAge_55 + DriverAge_56 + DriverAge_57 + DriverAge_58 + 
                 DriverAge_59 + DriverAge_60 + DriverAge_61 + DriverAge_62 + DriverAge_63 + DriverAge_64 + 
                 DriverAge_65 + DriverAge_66 + DriverAge_67 + DriverAge_68 + DriverAge_69 + DriverAge_70 + 
                 DriverAge_71 + DriverAge_72 + DriverAge_73 + DriverAge_74 + DriverAge_75 + DriverAge_76 + 
                 DriverAge_77 + DriverAge_78 + DriverAge_79 + DriverAge_80 + DriverAge_81 + DriverAge_82 + 
                 DriverAge_83 + DriverAge_84 + DriverAge_85 + DriverAge_86 + DriverAge_87 + DriverAge_88 + 
                 DriverAge_89 + DriverAge_90 + DriverAge_91 + DriverAge_92 + DriverAge_93 + DriverAge_94 + 
                 DriverAge_95 + DriverAge_96 + DriverAge_97 + DriverAge_98 + DriverAge_99 + 
                 CarAge_0 + CarAge_1 + CarAge_2 + CarAge_3 + CarAge_4 + CarAge_5 + CarAge_6 + 
                 CarAge_7 + CarAge_8 + CarAge_9 + CarAge_10 + CarAge_11 + CarAge_12 + 
                 CarAge_13 + CarAge_14 + CarAge_15 + CarAge_16 + CarAge_17 + CarAge_18 + 
                 CarAge_19 + CarAge_20 + CarAge_21 + CarAge_22 + CarAge_23 + CarAge_24 + CarAge_25,
                 data=training_set)

```


```R
set.seed(542)
folds = createFolds(training_set$ClaimNb, 5, list=FALSE)

set.seed(58)
m.lasso.1.cv = cv.glmnet(  model_exp_x, y = training_set$ClaimNb, 
                           offset = log(training_set$Exposure),
                           family = "poisson",
                           alpha = 1, #LASSO = 1, Ridge = 0,
                           nfolds = 5,
                           foldid = folds,
                           maxit=10^4,
                           nlambda = 25)


ptn_1 = Sys.time() - ptn
ptn_1
```


    Time difference of 1.80776 mins



```R
m.lasso.1.cv
```


    
    Call:  cv.glmnet(x = model_exp_x, y = training_set$ClaimNb, offset = log(training_set$Exposure),      nfolds = 5, foldid = folds, family = "poisson", alpha = 1,      maxit = 10^4, nlambda = 25) 
    
    Measure: Poisson Deviance 
    
          Lambda Index Measure       SE Nonzero
    min 0.000379     9  0.2538 0.001867      29
    1se 0.005559     2  0.2553 0.001888       5



```R
coef(m.lasso.1.cv, s = "lambda.min")
```


    109 x 1 sparse Matrix of class "dgCMatrix"
                            s1
    (Intercept)  -3.1281701957
    DriverAge_18  0.1057046756
    DriverAge_19  0.2334929393
    DriverAge_20  0.3401326182
    DriverAge_21  0.0020254687
    DriverAge_22  0.1904254512
    DriverAge_23  0.1091952209
    DriverAge_24  0.1222833992
    DriverAge_25  .           
    DriverAge_26  0.2335448741
    DriverAge_27  .           
    DriverAge_28  0.0158713088
    DriverAge_29  0.0897545434
    DriverAge_30  .           
    DriverAge_31  0.0340601774
    DriverAge_32  .           
    DriverAge_33  .           
    DriverAge_34  .           
    DriverAge_35  .           
    DriverAge_36  .           
    DriverAge_37  .           
    DriverAge_38  .           
    DriverAge_39  .           
    DriverAge_40  .           
    DriverAge_41 -0.0187640582
    DriverAge_42 -0.0356267177
    DriverAge_43 -0.0006757569
    DriverAge_44  .           
    DriverAge_45  .           
    DriverAge_46  .           
    DriverAge_47  .           
    DriverAge_48  .           
    DriverAge_49  .           
    DriverAge_50  .           
    DriverAge_51  0.0152449699
    DriverAge_52  .           
    DriverAge_53  0.0473036621
    DriverAge_54  0.0854077776
    DriverAge_55  .           
    DriverAge_56  .           
    DriverAge_57  .           
    DriverAge_58  .           
    DriverAge_59  .           
    DriverAge_60  .           
    DriverAge_61  .           
    DriverAge_62  0.0459254860
    DriverAge_63  0.0213810652
    DriverAge_64  .           
    DriverAge_65  .           
    DriverAge_66  .           
    DriverAge_67  .           
    DriverAge_68  .           
    DriverAge_69  .           
    DriverAge_70  .           
    DriverAge_71  .           
    DriverAge_72  .           
    DriverAge_73  .           
    DriverAge_74 -0.0079226705
    DriverAge_75  .           
    DriverAge_76  .           
    DriverAge_77  .           
    DriverAge_78  .           
    DriverAge_79  .           
    DriverAge_80  .           
    DriverAge_81  .           
    DriverAge_82  .           
    DriverAge_83  .           
    DriverAge_84  .           
    DriverAge_85 -0.0193009723
    DriverAge_86  .           
    DriverAge_87  .           
    DriverAge_88 -0.0875766416
    DriverAge_89  .           
    DriverAge_90  .           
    DriverAge_91  0.0658694661
    DriverAge_92  .           
    DriverAge_93  .           
    DriverAge_94  .           
    DriverAge_95  .           
    DriverAge_96  .           
    DriverAge_97  .           
    DriverAge_98  .           
    DriverAge_99  .           
    CarAge_0      .           
    CarAge_1      .           
    CarAge_2      .           
    CarAge_3      .           
    CarAge_4      .           
    CarAge_5      .           
    CarAge_6      .           
    CarAge_7      .           
    CarAge_8      .           
    CarAge_9      .           
    CarAge_10     0.0171396393
    CarAge_11     0.0026479357
    CarAge_12     0.1348012753
    CarAge_13     .           
    CarAge_14     0.0531308774
    CarAge_15     .           
    CarAge_16     .           
    CarAge_17     0.1456152254
    CarAge_18     .           
    CarAge_19     .           
    CarAge_20     0.0065982939
    CarAge_21     .           
    CarAge_22     .           
    CarAge_23     .           
    CarAge_24     .           
    CarAge_25     .           



```R
plotdata = expand.grid(DriverAge = 18:99, CarAge = 0:25, Exposure = 1)


for (age in (18:99)){
    plotdata[paste0("DriverAge_", age)] = 1*(plotdata$DriverAge <= age)
}
for (vehage in (0:25)){
    plotdata[paste0("CarAge_", vehage)] = 1*(plotdata$CarAge <= vehage)
}

plotdata['prediction'] = predict(m.lasso.1.cv, 
                                 as.matrix(subset(plotdata, select = -c(DriverAge, CarAge, Exposure))),
                                 newoffset = 0, 
                                 type="response", 
                                 s = m.lasso.1.cv$lambda.min)

```


```R
require(ggplot2)
ggplot(plotdata %>% group_by(DriverAge) %>% summarise(prediction = mean(prediction)), 
       aes(x=DriverAge, y=prediction)) + geom_point() + geom_line() + theme_bw()
```


    
![png](Elastic%20Net_files/Elastic%20Net_23_0.png)
    



```R
ggplot(plotdata %>% group_by(CarAge) %>% summarise(prediction = mean(prediction)), 
       aes(x=CarAge, y=prediction)) + geom_point() + geom_line() + theme_bw()
```


    
![png](Elastic%20Net_files/Elastic%20Net_24_0.png)
    

