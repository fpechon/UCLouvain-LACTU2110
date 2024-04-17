# Generalized Additive Models

## Loading the data and the packages

First, the packages


```R
require("CASdatasets") #Not needed if use of dataset.parquet
require("mgcv")
require("caret")
require("plyr")
require("ggplot2")
require("gridExtra")
if (!require("parallel")) install.packages("parallel")
if (!require("mgcViz")) install.packages("mgcViz")
require("parallel")
require("mgcViz")
require("arrow")
require("tidymodels")

```

    Le chargement a nécessité le package : CASdatasets
    
    Le chargement a nécessité le package : xts
    
    Le chargement a nécessité le package : zoo
    
    
    Attachement du package : 'zoo'
    
    
    Les objets suivants sont masqués depuis 'package:base':
    
        as.Date, as.Date.numeric
    
    
    Le chargement a nécessité le package : sp
    
    Le chargement a nécessité le package : mgcv
    
    Le chargement a nécessité le package : nlme
    
    This is mgcv 1.8-42. For overview type 'help("mgcv-package")'.
    
    Le chargement a nécessité le package : caret
    
    Le chargement a nécessité le package : ggplot2
    
    Le chargement a nécessité le package : lattice
    
    Le chargement a nécessité le package : plyr
    
    Le chargement a nécessité le package : gridExtra
    
    Le chargement a nécessité le package : parallel
    
    Le chargement a nécessité le package : mgcViz
    
    Le chargement a nécessité le package : qgam
    
    Registered S3 method overwritten by 'GGally':
      method from   
      +.gg   ggplot2
    
    Registered S3 method overwritten by 'mgcViz':
      method from  
      +.gg   GGally
    
    
    Attachement du package : 'mgcViz'
    
    
    L'objet suivant est masqué depuis 'package:lattice':
    
        qq
    
    
    Les objets suivants sont masqués depuis 'package:stats':
    
        qqline, qqnorm, qqplot
    
    
    Le chargement a nécessité le package : arrow
    
    
    Attachement du package : 'arrow'
    
    
    L'objet suivant est masqué depuis 'package:utils':
    
        timestamp
    
    
    Le chargement a nécessité le package : tidymodels
    
    -- [1mAttaching packages[22m ------------------------------------------------------------------------------ tidymodels 1.0.0 --
    
    [32mv[39m [34mbroom       [39m 1.0.4      [32mv[39m [34mrsample     [39m 1.1.1 
    [32mv[39m [34mdials       [39m 1.2.0      [32mv[39m [34mtibble      [39m 3.2.1 
    [32mv[39m [34mdplyr       [39m 1.1.1      [32mv[39m [34mtidyr       [39m 1.3.0 
    [32mv[39m [34minfer       [39m 1.0.4      [32mv[39m [34mtune        [39m 1.1.1 
    [32mv[39m [34mmodeldata   [39m 1.1.0      [32mv[39m [34mworkflows   [39m 1.1.3 
    [32mv[39m [34mparsnip     [39m 1.1.0      [32mv[39m [34mworkflowsets[39m 1.0.1 
    [32mv[39m [34mpurrr       [39m 1.0.1      [32mv[39m [34myardstick   [39m 1.1.0 
    [32mv[39m [34mrecipes     [39m 1.0.10     
    
    -- [1mConflicts[22m --------------------------------------------------------------------------------- tidymodels_conflicts() --
    [31mx[39m [34mdplyr[39m::[32marrange()[39m         masks [34mplyr[39m::arrange()
    [31mx[39m [34mrecipes[39m::[32mcheck()[39m         masks [34mqgam[39m::check()
    [31mx[39m [34mdplyr[39m::[32mcollapse()[39m        masks [34mnlme[39m::collapse()
    [31mx[39m [34mdplyr[39m::[32mcombine()[39m         masks [34mgridExtra[39m::combine()
    [31mx[39m [34mpurrr[39m::[32mcompact()[39m         masks [34mplyr[39m::compact()
    [31mx[39m [34mdplyr[39m::[32mcount()[39m           masks [34mplyr[39m::count()
    [31mx[39m [34mdplyr[39m::[32mdesc()[39m            masks [34mplyr[39m::desc()
    [31mx[39m [34mpurrr[39m::[32mdiscard()[39m         masks [34mscales[39m::discard()
    [31mx[39m [34mdplyr[39m::[32mfailwith()[39m        masks [34mplyr[39m::failwith()
    [31mx[39m [34mdplyr[39m::[32mfilter()[39m          masks [34mstats[39m::filter()
    [31mx[39m [34mdplyr[39m::[32mfirst()[39m           masks [34mxts[39m::first()
    [31mx[39m [34mdplyr[39m::[32mid()[39m              masks [34mplyr[39m::id()
    [31mx[39m [34mdplyr[39m::[32mlag()[39m             masks [34mstats[39m::lag()
    [31mx[39m [34mdplyr[39m::[32mlast()[39m            masks [34mxts[39m::last()
    [31mx[39m [34mpurrr[39m::[32mlift()[39m            masks [34mcaret[39m::lift()
    [31mx[39m [34mdplyr[39m::[32mmutate()[39m          masks [34mplyr[39m::mutate()
    [31mx[39m [34myardstick[39m::[32mprecision()[39m   masks [34mcaret[39m::precision()
    [31mx[39m [34myardstick[39m::[32mrecall()[39m      masks [34mcaret[39m::recall()
    [31mx[39m [34mdplyr[39m::[32mrename()[39m          masks [34mplyr[39m::rename()
    [31mx[39m [34myardstick[39m::[32msensitivity()[39m masks [34mcaret[39m::sensitivity()
    [31mx[39m [34myardstick[39m::[32mspecificity()[39m masks [34mcaret[39m::specificity()
    [31mx[39m [34mrecipes[39m::[32mstep()[39m          masks [34mstats[39m::step()
    [31mx[39m [34mdplyr[39m::[32msummarise()[39m       masks [34mplyr[39m::summarise()
    [31mx[39m [34mdplyr[39m::[32msummarize()[39m       masks [34mplyr[39m::summarize()
    [34m*[39m Use suppressPackageStartupMessages() to eliminate package startup messages
    
    

then, the data


```R
dataset = read_parquet(file = "../data/dataset.parquet")
```

Checking that the data is loaded.


```R
str(dataset)
```

    'data.frame':	410864 obs. of  10 variables:
     $ PolicyID : Factor w/ 413169 levels "1","2","3","4",..: 1 2 3 4 5 6 7 8 9 10 ...
     $ ClaimNb  : int  0 0 0 0 0 0 0 0 0 0 ...
     $ Exposure : num  0.09 0.84 0.52 0.45 0.15 0.75 0.81 0.05 0.76 0.34 ...
     $ Power    : Factor w/ 12 levels "d","e","f","g",..: 4 4 3 3 4 4 1 1 1 6 ...
     $ CarAge   : int  0 0 2 2 0 0 1 0 9 0 ...
     $ DriverAge: int  46 46 38 38 41 41 27 27 23 44 ...
     $ Brand    : Factor w/ 7 levels "Fiat","Japanese (except Nissan) or Korean",..: 2 2 2 2 2 2 2 2 1 2 ...
     $ Gas      : Factor w/ 2 levels "Diesel","Regular": 1 1 2 2 1 1 2 2 2 2 ...
     $ Region   : Factor w/ 10 levels "Aquitaine","Basse-Normandie",..: 1 1 8 8 9 9 1 1 8 6 ...
     $ Density  : int  76 76 3003 3003 60 60 695 695 7887 27000 ...
    

# Outline of this session.

- Illustration of the backfitting algorithm
- Use of mgcv package
- When using ‘manual backfitting’ can be useful


# Illustration of the backfitting algorithm

## First iteration

- First we start with a Poisson regression with only an intercept.


```R
autofit=dataset #Copy the data

#Model with only an intercept
require(mgcv) # Load package if not loaded yet.
fit0<-gam(ClaimNb~1, data=autofit, family=poisson(), offset=log(Exposure))

autofit$fit0=fit0$fitted.values
head(autofit$fit0)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>0.0063083008562306</li><li>0.0588774746581522</li><li>0.0364479605026657</li><li>0.031541504281153</li><li>0.0105138347603843</li><li>0.0525691738019216</li></ol>



- We fit a model with the discrete variables. (e.g. model from the GLM session, or any other model for the illustration)


```R
fit1<-gam(ClaimNb ~ offset(log(Exposure)) + Power  * Region +  Brand + Gas,
         data = autofit,
         family=poisson(link = log))
autofit$fit1 = fit1$fitted.values
```

- Let us now consider a continuous covariate: CarAge


```R
require(plyr)
mm <- ddply(autofit, .(CarAge), summarise, totalExposure = sum(Exposure), 
                totalClaimObs=sum(ClaimNb), totalClaimExp=sum(fit1))    
head(mm)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>CarAge</th><th scope=col>totalExposure</th><th scope=col>totalClaimObs</th><th scope=col>totalClaimExp</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td> 8710.903</td><td> 618</td><td> 621.9297</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>18137.929</td><td>1311</td><td>1290.4246</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>17347.019</td><td>1234</td><td>1230.6921</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>15818.469</td><td>1101</td><td>1120.1672</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>14966.334</td><td>1086</td><td>1057.8357</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>14445.505</td><td>1019</td><td>1022.4781</td></tr>
</tbody>
</table>




```R
fit2<-gam(totalClaimObs ~ s(CarAge), 
              offset=log(totalClaimExp), 
              family=poisson(), 
              data=mm)
```

- Let us visualize the estimated function.


```R
require(visreg)
visreg(fit2, xvar = "CarAge", gg = TRUE, scale = "response") + ylim(c(0.25, 1.2)) +
    ylab("Multiplicative Effect")
```

    Le chargement a nécessité le package : visreg
    
    


    
![png](GAMs_files/GAMs_16_1.png)
    


The new prediction of the claim frequency is now given by the old one times the correction due to CarAge.


```R
autofit$fit2<-autofit$fit1*predict(fit2, newdata=autofit, type="response")
```

The total number of predicted claim remains unchanged:


```R
c(sum(autofit$fit1), sum(autofit$fit2))
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>16127.0008725337</li><li>16126.9999999942</li></ol>



- Let us now consider the other continuous covariate: DriverAge


```R
mm <- ddply(autofit, .(DriverAge), summarise, totalExposure = sum(Exposure), totalClaimObs = sum(ClaimNb),
    totalClaimExp = sum(fit2))
head(mm)
```


<table class="dataframe">
<caption>A data.frame: 6 × 4</caption>
<thead>
	<tr><th></th><th scope=col>DriverAge</th><th scope=col>totalExposure</th><th scope=col>totalClaimObs</th><th scope=col>totalClaimExp</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>18</td><td> 148.4220</td><td> 46</td><td>  9.951695</td></tr>
	<tr><th scope=row>2</th><td>19</td><td> 636.9296</td><td>171</td><td> 42.610896</td></tr>
	<tr><th scope=row>3</th><td>20</td><td>1020.8744</td><td>216</td><td> 69.527420</td></tr>
	<tr><th scope=row>4</th><td>21</td><td>1277.9390</td><td>207</td><td> 87.265481</td></tr>
	<tr><th scope=row>5</th><td>22</td><td>1562.9872</td><td>239</td><td>107.852781</td></tr>
	<tr><th scope=row>6</th><td>23</td><td>1830.1058</td><td>230</td><td>128.854915</td></tr>
</tbody>
</table>




```R
fit3 <- gam(totalClaimObs ~ s(DriverAge), offset = log(totalClaimExp), family = poisson(),
    data = mm)
```


```R
visreg(fit3, xvar = "DriverAge", gg = TRUE, scale = "response") + ylim(c(0, 5)) +
    ylab("Multiplicative Effect") + scale_x_continuous(name = "Age of Driver", limits = c(18,
    99), breaks = c(18, seq(20, 95, 5), 99))
```


    
![png](GAMs_files/GAMs_24_0.png)
    


The new prediction of the claim frequency is now given by the old one times the correction due to DriverAge.


```R
autofit$fit3 <- autofit$fit2 * predict(fit3, newdata = autofit, type = "response")
```

The total expected number of claims remains unchanged.


```R
c(sum(autofit$fit2), sum(autofit$fit3))
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>16126.9999999942</li><li>16127</li></ol>



Let us compute the log-likelihood


```R
LL0 = sum(dpois(x = autofit$ClaimNb, lambda = autofit$fit0, log = TRUE))
LLi = sum(dpois(x = autofit$ClaimNb, lambda = autofit$fit3, log = TRUE))
c(LL0, LLi)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>-68153.7399700476</li><li>-67309.9644762711</li></ol>



## Further iterations

Let us now iterate, and fit again the discrete variables, then CarAge, then DriverAge, and let us stop when the log-likelihood change is smaller than some small epsilon. When we fit the model, everything that has been fitted before and is unrelated to the current variable is put in the offset. To be sure that the algorithm stops, we also put a maximum of 20 iterations...




```R
epsilon = 1e-08
i = 0
fit_it_discr = list(fit1)
fit_it_CarAge = list(fit2)
fit_it_DriverAge = list(fit3)

while (abs(LL0/LLi - 1) > epsilon & (i < 20)) {
    i = i + 1
    LL0 = LLi
    # Discrete variables
    autofit$logoffset = predict(fit_it_CarAge[[i]], newdata = autofit) + predict(fit_it_DriverAge[[i]],
        newdata = autofit) + log(autofit$Exposure)
    fit_it_discr[[i + 1]] <- gam(ClaimNb ~ Power * Region + Brand + Gas, 
                                 autofit, family = poisson(), offset = logoffset)

    # CarAge
    autofit$logoffset = predict(fit_it_discr[[i + 1]], newdata = autofit) + predict(fit_it_DriverAge[[i]],
        newdata = autofit) + log(autofit$Exposure)
    mm <- ddply(autofit, .(CarAge), summarise, totalClaimObs = sum(ClaimNb), totalClaimExp = sum(exp(logoffset)))
    fit_it_CarAge[[i + 1]] <- gam(totalClaimObs ~ s(CarAge), offset = log(totalClaimExp),
        family = poisson(), data = mm)

    # DriverAge
    autofit$logoffset = predict(fit_it_discr[[i + 1]], newdata = autofit) + predict(fit_it_CarAge[[i +
        1]], newdata = autofit) + log(autofit$Exposure)
    mm <- ddply(autofit, .(DriverAge), summarise, totalClaimObs = sum(ClaimNb), totalClaimExp = sum(exp(logoffset)))
    fit_it_DriverAge[[i + 1]] <- gam(totalClaimObs ~ s(DriverAge), offset = log(totalClaimExp),
        family = poisson(), data = mm)
    ## Compute the new estimates

    autofit$currentfit = predict(fit_it_discr[[i + 1]], newdata = autofit, type = "response") *
        predict(fit_it_CarAge[[i + 1]], newdata = autofit, type = "response") * predict(fit_it_DriverAge[[i +
        1]], newdata = autofit, type = "response") * (autofit$Exposure)

    LLi = sum(dpois(x = autofit$ClaimNb, lambda = autofit$currentfit, log = TRUE))
    print(c(i, LL0, LLi))
}
```

    [1]      1.00 -67309.96 -67299.70
    [1]      2.00 -67299.70 -67299.33
    [1]      3.00 -67299.33 -67299.31
    [1]      4.00 -67299.31 -67299.31
    [1]      5.00 -67299.31 -67299.31
    

## Results

Let us now see the betas at each iteration.

### Discrete variables



```R
res_discr = matrix(NA, ncol = 127, nrow = i + 1)
colnames(res_discr) = names(fit_it_discr[[1]]$coefficients)
res_discr[1, ] = fit_it_discr[[1]]$coefficients
res_discr[2, ] = fit_it_discr[[2]]$coefficients
res_discr[3, ] = fit_it_discr[[3]]$coefficients
res_discr[4, ] = fit_it_discr[[4]]$coefficients
res_discr[5, ] = fit_it_discr[[5]]$coefficients
res_discr[6, ] = fit_it_discr[[6]]$coefficients
```

For instance, the 9 first variables:


```R
require("gridExtra")
p1 = lapply(2:10, function(i) {
    ggplot() + geom_point(aes(y = res_discr[, i], x = 1:6)) + xlab("Iteration") +
        ylab("beta") + ggtitle(names(fit_it_discr[[1]]$coefficients)[i]) + scale_x_continuous(breaks = 1:6)
})
do.call(grid.arrange, p1)
```


    
![png](GAMs_files/GAMs_36_0.png)
    


### CarAge


```R
CarAge = matrix(NA, ncol = 6, nrow = 26)
CarAge[, 1] = predict(fit_it_CarAge[[1]], data.frame(CarAge = seq(from = 0, to = 25,
    by = 1)), type = "response")
CarAge[, 2] = predict(fit_it_CarAge[[2]], data.frame(CarAge = seq(from = 0, to = 25,
    by = 1)), type = "response")
CarAge[, 3] = predict(fit_it_CarAge[[3]], data.frame(CarAge = seq(from = 0, to = 25,
    by = 1)), type = "response")
CarAge[, 4] = predict(fit_it_CarAge[[4]], data.frame(CarAge = seq(from = 0, to = 25,
    by = 1)), type = "response")
CarAge[, 5] = predict(fit_it_CarAge[[5]], data.frame(CarAge = seq(from = 0, to = 25,
    by = 1)), type = "response")
CarAge[, 6] = predict(fit_it_CarAge[[6]], data.frame(CarAge = seq(from = 0, to = 25,
    by = 1)), type = "response")

x = as.data.frame(CarAge)
names(x) = sapply(1:6, function(i) {
    paste("it", i)
})
x = stack(as.data.frame(x))
names(x)[2] = "Iteration"

ggplot(x) + geom_line(aes(x = rep(0:25, 6), y = values, color = Iteration)) + xlab("Age of the Car") +
    ylab("Multiplicative Effect")
```


    
![png](GAMs_files/GAMs_38_0.png)
    


### DriverAge



```R
DriverAge = matrix(NA, ncol = 6, nrow = 82)
DriverAge[, 1] = predict(fit_it_DriverAge[[1]], data.frame(DriverAge = seq(from = 18,
    to = 99, by = 1)), type = "response")
DriverAge[, 2] = predict(fit_it_DriverAge[[2]], data.frame(DriverAge = seq(from = 18,
    to = 99, by = 1)), type = "response")
DriverAge[, 3] = predict(fit_it_DriverAge[[3]], data.frame(DriverAge = seq(from = 18,
    to = 99, by = 1)), type = "response")
DriverAge[, 4] = predict(fit_it_DriverAge[[4]], data.frame(DriverAge = seq(from = 18,
    to = 99, by = 1)), type = "response")
DriverAge[, 5] = predict(fit_it_DriverAge[[5]], data.frame(DriverAge = seq(from = 18,
    to = 99, by = 1)), type = "response")
DriverAge[, 6] = predict(fit_it_DriverAge[[6]], data.frame(DriverAge = seq(from = 18,
    to = 99, by = 1)), type = "response")

x = as.data.frame(DriverAge)
names(x) = sapply(1:6, function(i) {
    paste("it", i)
})
x = stack(as.data.frame(x))
names(x)[2] = "Iteration"

ggplot(x) + geom_line(aes(x = rep(18:99, 6), y = values, color = Iteration)) + xlab("Age of the Driver") +
    ylab("Multiplicative Effect")
```


    
![png](GAMs_files/GAMs_40_0.png)
    


## Comparison with GAM


Let us now compare with the GAM directly


```R
m0_gam = gam(ClaimNb ~ offset(log(Exposure)) + Power * Region + Brand + Gas + s(DriverAge) + s(CarAge), 
             data = autofit,
             family = poisson(link = log))

ggplot() + geom_point(aes(x = autofit$currentfit, y = m0_gam$fitted.values)) + xlab("Manual backfitting") +
    ylab("GAM from mgcv")
```


    
![png](GAMs_files/GAMs_42_0.png)
    


# Use of the mgcv package


First, let us retrieve the training and testing set we used before (in the GLM session).




```R
set.seed(21)
in_training = createDataPartition(dataset$ClaimNb, times = 1, p = 0.8, list = FALSE)
training_set = dataset[in_training, ]
testing_set = dataset[-in_training, ]
```

The gam function works very similarly to the glm function. The continuous covariate have to be specified using for instance the function s(.). Interaction with respect to a discrete variable can be done by specifying the variable in the ‘by’ argument (see below).

## First try with gam


Let us start with the model we created during the GLM session. We will replace the continuous variables by splines.


```R
rec <- recipe(ClaimNb ~ DriverAge + CarAge + Power + Gas + Region + Brand + Exposure, 
              data = training_set) %>% # Which columns do we need ?
    step_relevel(Power, ref_level = "d") %>%
    step_relevel(Gas, ref_level = "Regular") %>%
    step_relevel(Region, ref_level = "Centre") %>%
    step_relevel(Brand, ref_level = "Renault, Nissan or Citroen") %>%
    step_mutate(Brand = forcats::fct_collapse(Brand, A = c("Fiat", "Mercedes, Chrysler or BMW", 
                                                     "Opel, General Motors or Ford",
                                                     "other", 
                                                     "Volkswagen, Audi, Skoda or Seat"))) %>%
    step_mutate(Power = forcats::fct_collapse(Power, 
                                                 "e-f-g-h" = c("e", "f", "g", "h"),
                                                 "i-j-k-l-m" = c("i", "j", "k", "l", "m"),
                                                 "n-o" = c("n", "o")
                                                )) %>%
    step_mutate(Region = forcats::fct_collapse(Region, 
                                                 "A" = c("Pays-de-la-Loire", "Poitou-Charentes", "Aquitaine")
                                                )) %>%
    prep()


# Same as above..
ptn_0 = Sys.time()
m0_gam = gam(ClaimNb ~ offset(log(Exposure)) + Power * Region + Brand +
    Gas + s(DriverAge) + s(CarAge), data = bake(rec, training_set),
    family = poisson(link = log))
print(Sys.time() - ptn_0)
```

    Time difference of 37.88866 secs
    

## Comparison with bam

We see that the computational time is already long, especially if we wanted to use cross-validation. There is also the function bam, which is optimized for very large datasets and allows parallel computing.


```R
require(parallel)
cl = makeCluster(detectCores() - 1)  # Number of cores to use, for parallel computing.
ptn_0 = Sys.time()
m0_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power * Region + Brand +
    Gas+ s(DriverAge) + s(CarAge), data = bake(rec, training_set),
    family = poisson(link = log), cluster = cl)
stopCluster(cl)
print(Sys.time() - ptn_0)
```

    Time difference of 9.079048 secs
    

We can see the fitted function using plot,


```R
par(mfrow = c(1, 2))
plot(m0_bam, trans = exp, scale = 0, shade = TRUE)
```


    
![png](GAMs_files/GAMs_52_0.png)
    


Since 2020, the package mgcViz simplifies greatly the creation of visuals of GAMs. (Vignette available)


```R
require(mgcViz)
viz <- getViz(m0_bam)
print(plot(viz, allTerms = T), pages = 1)
```


    
![png](GAMs_files/GAMs_54_0.png)
    


## Bivariate function



```R
cl = makeCluster(detectCores()-1) # Number of cores to use
m1_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + te(DriverAge, CarAge), # or replace te(DriverAge, CarAge) by ti(DriverAge) + ti(CarAge) + ti(DriverAge, CarAge)
         data = bake(rec, training_set),
         family=poisson(link = log),
         cluster = cl)
stopCluster(cl)
m1_bam
```


    
    Family: poisson 
    Link function: log 
    
    Formula:
    ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + 
        te(DriverAge, CarAge)
    
    Estimated degrees of freedom:
    14.6  total = 28.56 
    
    fREML score: 577639.5     



```R
cl = makeCluster(detectCores()-1) # Number of cores to use
m1_bam_b = bam(ClaimNb ~ offset(log(Exposure)) + offset(log(Exposure)) + Power + Region + Brand + Gas + s(DriverAge, CarAge),
         data = bake(rec, training_set),
         family=poisson(link = log),
         cluster = cl)
stopCluster(cl)
m1_bam_b
```


    
    Family: poisson 
    Link function: log 
    
    Formula:
    ClaimNb ~ offset(log(Exposure)) + offset(log(Exposure)) + Power + 
        Region + Brand + Gas + s(DriverAge, CarAge)
    
    Estimated degrees of freedom:
    23  total = 37.03 
    
    fREML score: 577751.8     


To choose between te and s when adding bivariate functions, Wood (2017) recommends the following:

- ”Tensor product, te Invariant to linear rescaling of covariates, but not to rotation of covariate space. Good for smooth interactions of quantities measured in different units, or where very different degrees of smoothness appropriate relative to different covariates. Computationally inexpensive, provided TPRS bases are not used as marginal bases. Apart from scale invariance, not much supporting theory.

- TPRS, s(…,bs=“tp”) Invariant to rotation of covariate space (isotropic), but not to rescaling of covariates. Good for smooth interactions of quantities measured in same units, such as spatial co-ordinates, where isotropy is appropriate. Computational cost can be high as it increases with square of number of data (can be avoided by approximation). ”

We can visualize the interactions:


```R
vis.gam(m1_bam, view=c("DriverAge", "CarAge"),  plot.type = 'contour')
```


    
![png](GAMs_files/GAMs_59_0.png)
    



```R
vis.gam(m1_bam_b, view=c("DriverAge", "CarAge"),  plot.type = 'contour')
```


    
![png](GAMs_files/GAMs_60_0.png)
    


We can compute the log-likelihood


```R
logLik.gam(m0_bam)
```


    'log Lik.' -54016.41 (df=46.06032)



```R
logLik.gam(m1_bam)
```


    'log Lik.' -54092.17 (df=29.20055)



```R
logLik.gam(m1_bam_b)
```


    'log Lik.' -54013.76 (df=37.25777)


# Interaction between a continuous and a discrete variable

To include an interaction with a discrete variable, we can use the by argument. For example, between CarAge and Gas:


```R
cl = makeCluster(detectCores() - 1)  # Number of cores to use
m2_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas  + s(DriverAge) + s(CarAge, by = Gas), 
             data = bake(rec, training_set),,
    family = poisson(link = log), cluster = cl)
stopCluster(cl)
summary(m2_bam)
```


    
    Family: poisson 
    Link function: log 
    
    Formula:
    ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + 
        s(DriverAge) + s(CarAge, by = Gas)
    
    Parametric coefficients:
                                            Estimate Std. Error  z value Pr(>|z|)
    (Intercept)                             -2.93627    0.02785 -105.417  < 2e-16
    Powere-f-g-h                             0.09236    0.02628    3.515  0.00044
    Poweri-j-k-l-m                           0.22509    0.03457    6.510 7.50e-11
    Powern-o                                 0.29049    0.11133    2.609  0.00907
    RegionA                                  0.16598    0.02383    6.967 3.25e-12
    RegionBasse-Normandie                    0.08938    0.05392    1.658  0.09735
    RegionBretagne                           0.05259    0.02959    1.777  0.07550
    RegionHaute-Normandie                    0.12242    0.07690    1.592  0.11137
    RegionIle-de-France                      0.37357    0.02856   13.079  < 2e-16
    RegionLimousin                           0.42306    0.07697    5.497 3.87e-08
    RegionNord-Pas-de-Calais                 0.26994    0.03963    6.811 9.70e-12
    BrandA                                   0.08657    0.02001    4.326 1.52e-05
    BrandJapanese (except Nissan) or Korean -0.19997    0.03236   -6.180 6.39e-10
    GasDiesel                                0.12495    0.01907    6.552 5.66e-11
                                               
    (Intercept)                             ***
    Powere-f-g-h                            ***
    Poweri-j-k-l-m                          ***
    Powern-o                                ** 
    RegionA                                 ***
    RegionBasse-Normandie                   .  
    RegionBretagne                          .  
    RegionHaute-Normandie                      
    RegionIle-de-France                     ***
    RegionLimousin                          ***
    RegionNord-Pas-de-Calais                ***
    BrandA                                  ***
    BrandJapanese (except Nissan) or Korean ***
    GasDiesel                               ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    Approximate significance of smooth terms:
                           edf Ref.df Chi.sq p-value    
    s(DriverAge)         7.906  8.554 982.34 < 2e-16 ***
    s(CarAge):GasRegular 3.856  4.795  75.24 < 2e-16 ***
    s(CarAge):GasDiesel  1.842  2.333  13.07 0.00225 ** 
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    R-sq.(adj) =  0.00728   Deviance explained =  2.9%
    fREML = 5.7861e+05  Scale est. = 1         n = 328692


When we now plot the functions, we obtain two functions for CarAge.


```R
b <- getViz(m2_bam)
gridPrint(plot(sm(b, 2)) + theme_bw() +l_ciPoly()+ l_fitLine(colour = "red"),
          plot(sm(b, 3)) + theme_bw() +l_ciPoly()+ l_fitLine(colour = "red"), ncol=2)
```


    
![png](GAMs_files/GAMs_69_0.png)
    


We can test if the interaction improves our model (but does it improve the predictible power of our model ?).


```R
anova(m0_bam, m2_bam, test = "Chisq")
```


<table class="dataframe">
<caption>A anova: 2 × 5</caption>
<thead>
	<tr><th></th><th scope=col>Resid. Df</th><th scope=col>Resid. Dev</th><th scope=col>Df</th><th scope=col>Deviance</th><th scope=col>Pr(&gt;Chi)</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>328644.2</td><td>83026.14</td><td>       NA</td><td>       NA</td><td>       NA</td></tr>
	<tr><th scope=row>2</th><td>328661.2</td><td>83035.39</td><td>-17.00991</td><td>-9.251538</td><td>0.9322877</td></tr>
</tbody>
</table>




```R
cl = makeCluster(detectCores() - 1)  # Number of cores to use
m3_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + s(DriverAge, by = Gas) + s(CarAge), 
             data = bake(rec, training_set),
    family = poisson(link = log), cluster = cl)
stopCluster(cl)
anova(m0_bam, m3_bam, test = "Chisq")
```


<table class="dataframe">
<caption>A anova: 2 × 5</caption>
<thead>
	<tr><th></th><th scope=col>Resid. Df</th><th scope=col>Resid. Dev</th><th scope=col>Df</th><th scope=col>Deviance</th><th scope=col>Pr(&gt;Chi)</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>328644.2</td><td>83026.14</td><td>       NA</td><td>       NA</td><td>       NA</td></tr>
	<tr><th scope=row>2</th><td>328656.7</td><td>83031.03</td><td>-12.54417</td><td>-4.888129</td><td>0.9712308</td></tr>
</tbody>
</table>




```R
par(mfrow = c(1, 2))
plot(m3_bam, shade = TRUE, trans = exp, scale = -1, select = 1)
plot(m3_bam, shade = TRUE, trans = exp, scale = -1, select = 2)
```


    
![png](GAMs_files/GAMs_73_0.png)
    


Or with mgcViz:


```R
b <- getViz(m3_bam)
gridPrint(plot(sm(b, 1)) + theme_bw() +l_ciPoly()+ l_fitLine(colour = "red"),
          plot(sm(b, 2)) + theme_bw() +l_ciPoly()+ l_fitLine(colour = "red"), ncol=2)
```


    
![png](GAMs_files/GAMs_75_0.png)
    


# Cross-validation

We can also use cross-validation to check whether or not to include this variable. First we need to create the folds, let’s say 5.




```R
require(caret)
set.seed(41)
folds = createFolds(training_set$ClaimNb, k = 5)
res0 = lapply(folds, function(X) {
    cl = makeCluster(detectCores() - 1)  # Number of cores to use
    m3_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + s(DriverAge) + s(CarAge),
        data = bake(rec, training_set[-X, ]), family = poisson(link = log), cluster = cl)
    stopCluster(cl)
    pred = predict(m3_bam, bake(rec, training_set[X, ]), type = "response")
    sum(dpois(x = bake(rec, training_set[X, ])$ClaimNb, lambda = pred, log = TRUE))
    # sum(-pred +
    # training_set[X,]$ClaimNb*log(pred)-log(factorial(training_set[X,]$ClaimNb)))
})

res3 = lapply(folds, function(X) {
    cl = makeCluster(detectCores() - 1)  # Number of cores to use
    m3_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + s(DriverAge, by = Gas) +
        s(CarAge), 
                 data = bake(rec, training_set[-X, ]), family = poisson(link = log), cluster = cl)
    stopCluster(cl)
    pred = predict(m3_bam, bake(rec, training_set[X, ]), type = "response")
    sum(dpois(x = bake(rec, training_set[X, ])$ClaimNb, lambda = pred, log = TRUE))
    # sum(-pred +
    # training_set[X,]$ClaimNb*log(pred)-log(factorial(training_set[X,]$ClaimNb)))
})

cbind(unlist(res0), unlist(res3))
```


<table class="dataframe">
<caption>A matrix: 5 × 2 of type dbl</caption>
<tbody>
	<tr><th scope=row>Fold1</th><td>-10963.91</td><td>-10965.52</td></tr>
	<tr><th scope=row>Fold2</th><td>-10512.25</td><td>-10511.47</td></tr>
	<tr><th scope=row>Fold3</th><td>-10875.86</td><td>-10879.43</td></tr>
	<tr><th scope=row>Fold4</th><td>-10901.66</td><td>-10900.27</td></tr>
	<tr><th scope=row>Fold5</th><td>-10803.40</td><td>-10802.31</td></tr>
</tbody>
</table>




```R
# Average on 5 folds
apply(cbind(unlist(res0), unlist(res3)), 2, mean)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>-10811.4175425836</li><li>-10811.8007204507</li></ol>



There is no improvement with the interaction.


```R
res4 = lapply(folds, function(X) {
    cl = makeCluster(detectCores() - 1)  # Number of cores to use
    m3_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power + Region + Brand + Gas + s(DriverAge) + s(CarAge, by = Power), 
                 data = bake(rec, training_set[-X, ]), 
                 family = poisson(link = log),
        cluster = cl)
    stopCluster(cl)
    pred = predict(m3_bam, bake(rec, training_set[X, ]), type = "response")
    sum(dpois(x = bake(rec, training_set[X, ])$ClaimNb, lambda = pred, log = TRUE))
    # sum(-pred +
    # training_set[X,]$ClaimNb*log(pred)-log(factorial(training_set[X,]$ClaimNb)))
})
apply(cbind(unlist(res0), unlist(res3), unlist(res4)), 2, mean)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>-10811.4175425836</li><li>-10811.8007204507</li><li>-10812.4874932714</li></ol>



We conclude here, we did not find any further interactions. We can compute the deviance on the validation set


```R
2 * (sum(dpois(x = bake(rec, testing_set)$ClaimNb, lambda = bake(rec, testing_set)$ClaimNb, log = TRUE)) -
    sum(dpois(x = bake(rec, testing_set)$ClaimNb, lambda = predict(m0_bam, bake(rec, testing_set), offset = bake(rec, testing_set)$Exposure,
        type = "response"), log = TRUE)))
```


20525.0908437418


# Optimizing the number of nodes

We can also optimize the number of nodes by cross-validation.


```R
# First understand what it changes
default_choice = bam(ClaimNb ~ offset(log(Exposure)) + Power * Region + Brand +
    Gas+ s(DriverAge), data = bake(rec, training_set),
    family = poisson(link = log))
plot(default_choice)
```


    
![png](GAMs_files/GAMs_84_0.png)
    



```R
gam.check(default_choice)
```

    
    Method: fREML   Optimizer: perf newton
    full convergence after 3 iterations.
    Gradient range [0.006536233,0.006536233]
    (score 577557.4 & scale 1).
    Hessian positive definite, eigenvalue range [3.426133,3.426133].
    Model rank =  44 / 44 
    
    Basis dimension (k) checking results. Low p-value (k-index<1) may
    indicate that k is too low, especially if edf is close to k'.
    
                   k'  edf k-index p-value
    s(DriverAge) 9.00 7.88     0.9    0.31
    


```R
choose_nodes = bam(ClaimNb ~ offset(log(Exposure)) + Power * Region + Brand +
    Gas+ s(DriverAge, k=15), data = bake(rec, training_set),
    family = poisson(link = log))
plot(choose_nodes)
```


```R
res5 = lapply(folds, function(X) {
    cl = makeCluster(detectCores() - 1)  # Number of cores to use
    m3_bam = bam(ClaimNb ~ offset(log(Exposure)) + Power * Region + Brand +
                 Gas+ s(DriverAge, k=15) + s(CarAge), data = bake(rec, training_set[-X,]), 
                 family = poisson(link = log),
                 cluster = cl)
    stopCluster(cl)
    pred = predict(m3_bam, bake(rec, training_set[X, ]), type = "response")
    sum(dpois(x = bake(rec, training_set[X, ])$ClaimNb, lambda = pred, log = TRUE))
    # sum(-pred +
    # training_set[X,]$ClaimNb*log(pred)-log(factorial(training_set[X,]$ClaimNb)))
})
apply(cbind(unlist(res0), unlist(res5)), 2, mean)
```

# Comparison with best GLM model


```R
rec_glm = recipe(ClaimNb ~ DriverAge + CarAge + Power + Gas + Region + Brand + Exposure, data = training_set) %>% # Which columns do we need ?
    step_relevel(Power, ref_level = "d") %>%
    step_relevel(Gas, ref_level = "Regular") %>%
    step_relevel(Region, ref_level = "Centre") %>%
    step_relevel(Brand, ref_level = "Renault, Nissan or Citroen") %>%
    step_mutate(Brand = forcats::fct_collapse(Brand, A = c("Fiat", "Mercedes, Chrysler or BMW", 
                                                     "Opel, General Motors or Ford",
                                                     "other", 
                                                     "Volkswagen, Audi, Skoda or Seat"))) %>%
    step_mutate(Power = forcats::fct_collapse(Power, 
                                                 "e-f-g-h" = c("e", "f", "g", "h"),
                                                 "i-j-k-l-m" = c("i", "j", "k", "l", "m"),
                                                 "n-o" = c("n", "o")
                                                )) %>%
    step_mutate(Region = forcats::fct_collapse(Region, 
                                                 "A" = c("Pays-de-la-Loire", "Poitou-Charentes", "Aquitaine")
                                                )) %>%
    prep()


m_glm = gam(ClaimNb ~ offset(log(Exposure))+ poly(DriverAge, 7) + poly(CarAge, 2) + Power + Gas + Region + Brand, 
    data = bake(rec_glm, training_set), family = poisson())
```


```R
testing_set$GLM_pred = predict(m_glm, bake(rec_glm, testing_set), type="response")
testing_set$GAM_pred = predict(m0_bam, bake(rec, testing_set), type="response")
head(testing_set[,c("GLM_pred", "GAM_pred")], n=5)
```


```R
ggplot(testing_set) + geom_point(aes(x=GLM_pred, y=GAM_pred))+ylab("GAM")+xlab("GLM")+geom_abline(slope=1, intercept=0, color="red")+
  scale_x_continuous(labels = scales::percent_format(accuracy = 0.01))+
  scale_y_continuous(labels = scales::percent_format(accuracy = 0.01))
```

However, the total amount of expected claims are still close.


```R
sum(testing_set$GLM_pred) #GLM
```


```R
sum(testing_set$GAM_pred) #GAM
```
