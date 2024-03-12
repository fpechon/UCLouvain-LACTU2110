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
    	C:\Users\Florian\AppData\Local\Temp\RtmpQh1clH\downloaded_packages
    

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


    Time difference of 6.412889 mins



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


    Time difference of 3.835233 mins



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


    Time difference of 6.278916 mins



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
driver_age_lst = c()
for (age in (18:99)){
    training_set[paste0("DriverAge_", age)] = 1*(training_set$DriverAge <= age)
    testing_set[paste0("DriverAge_", age)] = 1*(testing_set$DriverAge <= age)
    driver_age_lst = c(driver_age_lst, paste0("DriverAge_", age))
}

car_age_lst = c()
for (vehage in (0:25)){
    training_set[paste0("CarAge_", vehage)] = 1*(training_set$CarAge <= vehage)
    testing_set[paste0("CarAge_", vehage)] = 1*(testing_set$CarAge <= vehage)
    car_age_lst = c(car_age_lst, paste0("CarAge_", vehage))
}

density_lst = c()
for (density in unique(quantile(training_set$Density, seq(0.01,0.99,0.01)))){
    training_set[paste0("Density_", density)] = 1*(training_set$Density <= density)
    testing_set[paste0("Density_", density)] = 1*(testing_set$Density <= density)
    density_lst = c(density_lst, paste0("Density_", density))
}

power_lst = c()
levels = levels(ordered(training_set$Power))
for (power in 1:length(unique(ordered(training_set$Power)))){
    training_set[paste0("Power_", power)] = 1*(ordered(training_set$Power) <= levels[power])
    testing_set[paste0("Power_", power)] = 1*(ordered(testing_set$Power) <= levels[power])
    power_lst = c(power_lst, paste0("Power_", power))
}

lst_vars = paste(paste(driver_age_lst, collapse=" + "), 
                 paste(car_age_lst, collapse=" + "), 
                 paste(density_lst, collapse=" + "), 
                 paste(power_lst, collapse=" + "), 
                 sep = " + ")
```


```R
ptn=Sys.time()
model_exp_x = model.matrix(as.formula(paste("ClaimNb ~ 0  + ", 
                                            lst_vars, 
                                            sep="")),
                           data=training_set)

head(model_exp_x)
```


<table class="dataframe">
<caption>A matrix: 6 × 217 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>DriverAge_18</th><th scope=col>DriverAge_19</th><th scope=col>DriverAge_20</th><th scope=col>DriverAge_21</th><th scope=col>DriverAge_22</th><th scope=col>DriverAge_23</th><th scope=col>DriverAge_24</th><th scope=col>DriverAge_25</th><th scope=col>DriverAge_26</th><th scope=col>DriverAge_27</th><th scope=col>...</th><th scope=col>Power_3</th><th scope=col>Power_4</th><th scope=col>Power_5</th><th scope=col>Power_6</th><th scope=col>Power_7</th><th scope=col>Power_8</th><th scope=col>Power_9</th><th scope=col>Power_10</th><th scope=col>Power_11</th><th scope=col>Power_12</th></tr>
</thead>
<tbody>
	<tr><th scope=row>2</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>...</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>...</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>4</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>...</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>5</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>...</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>6</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>...</td><td>0</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>7</th><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>...</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr>
</tbody>
</table>




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


    Time difference of 7.383376 mins



```R
m.lasso.1.cv
```


    
    Call:  cv.glmnet(x = model_exp_x, y = training_set$ClaimNb, offset = log(training_set$Exposure),      nfolds = 5, foldid = folds, family = "poisson", alpha = 1,      maxit = 10^4, nlambda = 25) 
    
    Measure: Poisson Deviance 
    
          Lambda Index Measure       SE Nonzero
    min 0.000379     9  0.2523 0.001820      49
    1se 0.003787     3  0.2538 0.001851      10



```R
coef(m.lasso.1.cv, s = "lambda.min")
```


    218 x 1 sparse Matrix of class "dgCMatrix"
                             s1
    (Intercept)   -2.6881967110
    DriverAge_18   0.1230383584
    DriverAge_19   0.2425459500
    DriverAge_20   0.3468576743
    DriverAge_21   0.0081852836
    DriverAge_22   0.1919444916
    DriverAge_23   0.1115035954
    DriverAge_24   0.1334220842
    DriverAge_25   .           
    DriverAge_26   0.2385793758
    DriverAge_27   .           
    DriverAge_28   0.0063266606
    DriverAge_29   0.0884926007
    DriverAge_30   .           
    DriverAge_31   0.0330756231
    DriverAge_32   .           
    DriverAge_33   .           
    DriverAge_34   .           
    DriverAge_35   .           
    DriverAge_36   .           
    DriverAge_37   .           
    DriverAge_38   .           
    DriverAge_39   .           
    DriverAge_40   .           
    DriverAge_41  -0.0139173282
    DriverAge_42  -0.0452490010
    DriverAge_43  -0.0048463049
    DriverAge_44   .           
    DriverAge_45   .           
    DriverAge_46   .           
    DriverAge_47   .           
    DriverAge_48   .           
    DriverAge_49   .           
    DriverAge_50   .           
    DriverAge_51   0.0142488466
    DriverAge_52   .           
    DriverAge_53   0.0473966312
    DriverAge_54   0.0867331457
    DriverAge_55   .           
    DriverAge_56   .           
    DriverAge_57   .           
    DriverAge_58   .           
    DriverAge_59   .           
    DriverAge_60   .           
    DriverAge_61   .           
    DriverAge_62   0.0212773459
    DriverAge_63   0.0152644944
    DriverAge_64   .           
    DriverAge_65   .           
    DriverAge_66   .           
    DriverAge_67   .           
    DriverAge_68   .           
    DriverAge_69   .           
    DriverAge_70   .           
    DriverAge_71   .           
    DriverAge_72   .           
    DriverAge_73   .           
    DriverAge_74  -0.0203627755
    DriverAge_75   .           
    DriverAge_76   .           
    DriverAge_77   .           
    DriverAge_78   .           
    DriverAge_79   .           
    DriverAge_80   .           
    DriverAge_81   .           
    DriverAge_82  -0.0028710836
    DriverAge_83   .           
    DriverAge_84   .           
    DriverAge_85  -0.0453411877
    DriverAge_86   .           
    DriverAge_87   .           
    DriverAge_88  -0.1156017796
    DriverAge_89   .           
    DriverAge_90   .           
    DriverAge_91   0.1600934560
    DriverAge_92   .           
    DriverAge_93   .           
    DriverAge_94   .           
    DriverAge_95   .           
    DriverAge_96   .           
    DriverAge_97   .           
    DriverAge_98   .           
    DriverAge_99   .           
    CarAge_0       .           
    CarAge_1       .           
    CarAge_2       .           
    CarAge_3      -0.0412367870
    CarAge_4       .           
    CarAge_5       .           
    CarAge_6       .           
    CarAge_7       .           
    CarAge_8       .           
    CarAge_9       .           
    CarAge_10      .           
    CarAge_11      .           
    CarAge_12      0.1249058005
    CarAge_13      .           
    CarAge_14      0.0553587798
    CarAge_15      .           
    CarAge_16      .           
    CarAge_17      0.1305474560
    CarAge_18      .           
    CarAge_19      .           
    CarAge_20      .           
    CarAge_21      0.0034407096
    CarAge_22      .           
    CarAge_23      .           
    CarAge_24      .           
    CarAge_25      .           
    Density_10     0.0250248791
    Density_11     .           
    Density_13     .           
    Density_15     .           
    Density_17     .           
    Density_19    -0.1046584112
    Density_22     .           
    Density_24     .           
    Density_26     .           
    Density_28     .           
    Density_30     .           
    Density_32     .           
    Density_34     .           
    Density_37     .           
    Density_40    -0.0691068227
    Density_43     .           
    Density_44     .           
    Density_48     .           
    Density_50     .           
    Density_51     .           
    Density_55     .           
    Density_57    -0.0355121844
    Density_60    -0.0015025919
    Density_64     .           
    Density_67     .           
    Density_73     .           
    Density_79     .           
    Density_83     .           
    Density_88     .           
    Density_91     .           
    Density_95     .           
    Density_102    .           
    Density_105    .           
    Density_110   -0.0202492526
    Density_117    .           
    Density_125    .           
    Density_133    .           
    Density_142    .           
    Density_149    .           
    Density_159    .           
    Density_169    .           
    Density_182    .           
    Density_192    .           
    Density_204   -0.0527680685
    Density_218    .           
    Density_229    .           
    Density_243    .           
    Density_264    .           
    Density_282    .           
    Density_288    .           
    Density_298    .           
    Density_329    .           
    Density_359    .           
    Density_394   -0.0714800408
    Density_405   -0.0320320494
    Density_408    .           
    Density_451    .           
    Density_473    .           
    Density_503    .           
    Density_557    .           
    Density_612    .           
    Density_644    .           
    Density_713    .           
    Density_741   -0.0078728044
    Density_796   -0.0944469716
    Density_851   -0.0152366937
    Density_957    .           
    Density_1052  -0.0013819343
    Density_1055   .           
    Density_1165   .           
    Density_1284   .           
    Density_1313   .           
    Density_1329   .           
    Density_1410   .           
    Density_1500   .           
    Density_1765   .           
    Density_1943  -0.0730819011
    Density_2103   .           
    Density_2411   .           
    Density_2663   .           
    Density_2906   .           
    Density_3060   .           
    Density_3368   .           
    Density_3541   .           
    Density_3866   .           
    Density_4087   .           
    Density_4116   .           
    Density_4128   .           
    Density_4348   .           
    Density_4496   .           
    Density_5376   .           
    Density_6257   .           
    Density_6864   .           
    Density_8023  -0.0009601506
    Density_10477  .           
    Density_17140 -0.0458907286
    Density_27000  .           
    Power_1       -0.1273518546
    Power_2        .           
    Power_3        .           
    Power_4       -0.0463346564
    Power_5       -0.0241069746
    Power_6       -0.0130266934
    Power_7        .           
    Power_8        .           
    Power_9        .           
    Power_10       .           
    Power_11       .           
    Power_12       .           



```R
plotdata = expand.grid(DriverAge = 18:99, 
                       CarAge = 0:25, 
                       Density = unique(quantile(training_set$Density, seq(0.01,0.99,0.01))),
                       Exposure = 1)


for (age in (18:99)){
    plotdata[paste0("DriverAge_", age)] = 1*(plotdata$DriverAge <= age)
}
for (vehage in (0:25)){
    plotdata[paste0("CarAge_", vehage)] = 1*(plotdata$CarAge <= vehage)
}
for (density in unique(sort(plotdata$Density))){
    plotdata[paste0("Density_", density)] = 1*(plotdata$Density <= density)
}

plotdata['prediction'] = predict(m.lasso.1.cv, 
                                 as.matrix(subset(plotdata, select = -c(DriverAge, CarAge,Density, Exposure))),
                                 newoffset = 0, 
                                 type="response", 
                                 s = m.lasso.1.cv$lambda.min)

```


    Error in predict.glmnet(object$glmnet.fit, newx, s = lambda, ...): The number of variables in newx must be 217
    Traceback:
    

    1. predict(m.lasso.1.cv, as.matrix(subset(plotdata, select = -c(DriverAge, 
     .     CarAge, Density, Exposure))), newoffset = 0, type = "response", 
     .     s = m.lasso.1.cv$lambda.min)

    2. predict.cv.glmnet(m.lasso.1.cv, as.matrix(subset(plotdata, select = -c(DriverAge, 
     .     CarAge, Density, Exposure))), newoffset = 0, type = "response", 
     .     s = m.lasso.1.cv$lambda.min)

    3. predict(object$glmnet.fit, newx, s = lambda, ...)

    4. predict.fishnet(object$glmnet.fit, newx, s = lambda, ...)

    5. NextMethod("predict")

    6. predict.glmnet(object$glmnet.fit, newx, s = lambda, ...)

    7. stop(paste0("The number of variables in newx must be ", p))



```R
require(ggplot2)
ggplot(plotdata %>% group_by(DriverAge) %>% summarise(prediction = mean(prediction)), 
       aes(x=DriverAge, y=prediction)) + geom_point() + geom_line() + theme_bw() + 
        scale_y_continuous(labels = scales::label_percent(accuracy = 0.02))
```


```R
ggplot(plotdata %>% group_by(CarAge) %>% summarise(prediction = mean(prediction)), 
       aes(x=CarAge, y=prediction)) + geom_point() + geom_line() + theme_bw()+
scale_y_continuous(labels = scales::label_percent(accuracy = 0.02))
```


```R
ggplot(plotdata %>% group_by(Density) %>% summarise(prediction = mean(prediction)), 
       aes(x=Density, y=prediction)) + geom_point() + geom_line() + theme_bw()+
scale_y_continuous(labels = scales::label_percent(accuracy = 0.02))
```


```R
s = coef(m.lasso.1.cv, s = "lambda.min")
driver_age_breaks = c()
car_age_breaks = c()
density_breaks = c()
for (col in names(which(s[,1] != 0))){
    if (grepl("DriverAge_", col)){
         driver_age_breaks = c(driver_age_breaks, as.numeric(unlist(strsplit(col, "_"))[2]))
    }
    if (grepl("CarAge_", col)){
         car_age_breaks = c(car_age_breaks, as.numeric(unlist(strsplit(col, "_"))[2]))
    }
    if (grepl("Density_", col)){
         density_breaks = c(density_breaks, as.numeric(unlist(strsplit(col, "_"))[2]))
    }
}

# Define a data preprocessing with these breaks

data_prep = recipe(ClaimNb ~ DriverAge + CarAge + Power + Gas + Region + Brand + Density + Exposure, data = training_set) %>%
    step_relevel(Power, ref_level = "d") %>%
    step_relevel(Gas, ref_level = "Regular") %>%
    step_relevel(Region, ref_level = "Centre") %>%
    step_relevel(Brand, ref_level = "Renault, Nissan or Citroen") %>%
    step_mutate(DriverAge = cut(DriverAge, breaks = c(-Inf, driver_age_breaks, Inf))) %>%
    step_mutate(CarAge = cut(CarAge, breaks = c(-Inf, car_age_breaks, Inf))) %>%
    step_mutate(Density = cut(Density, breaks = c(-Inf, density_breaks, Inf))) %>%
    step_mutate(Power = forcats::fct_collapse(Power, 
                                                 "d-e-f" = c("d", "e", "f"),
                                                 "j-k-l-m-n-o" = c("j", "k", "l", "m", "n", "o")
                                                )) %>%
    prep()
```


```R
m_glm = glm(ClaimNb ~ offset(log(Exposure)) + Power + Brand + Gas + Region + DriverAge + CarAge + Density, 
   data = data_prep %>% bake(training_set),
   family=poisson(link=log))
summary(m_glm)
```


```R
2 * (sum(dpois(x = testing_set$ClaimNb, lambda = testing_set$ClaimNb,
    log = TRUE)) - sum(dpois(x = testing_set$ClaimNb, lambda = predict(m_glm, data_prep %>% bake(testing_set), type="response"), log=TRUE)))
```
