---
title: "Disease mapping with CARs in NIMBLE: A case study with the Scottish lip cancer data"
author: "Garyfallos Konstantinoudis"
output:
  html_document:
    toc: true
    toc_float: true
bibliography: biblio.bib
---


<style type="text/css">
body{ /* Normal  */
      font-size: 14px;
  }
h1.title {
  font-size: 30px;
  color: black;
  font-weight: bold;
}
h1 { /* Header 1 */
    font-size: 25px;
  color: black;
  font-weight: bold;
}
h2 { /* Header 2 */
    font-size: 20px;
  color: black;
  font-weight: bold;
}
h3 { /* Header 3 */
    font-size: 15px;
  color: black;
  font-weight: bold;
}
code.r{ /* Code block */
    font-size: 14px;
}
pre, code {
    color: 	#1B0F0E;
  }
</style>



\pagenumbering{gobble} 
\pagenumbering{arabic} 


```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, message=FALSE, warning=FALSE, fig.align = "center")
```

In this document we will perform disease mapping and spatial regression using `NIMBLE` [@nimble]. We will fit a series of disease mapping models with unstructured and structured random effects (ICAR, BYM and BYM2). The data can be downloaded by \href{https://geodacenter.github.io/data-and-lab/scotlip/}{geodacener} and includes lip cancer cases (CANCER) per ward in Scotland during 1975-1980 [@clayton1987]. The dataset also includes the district number (DISTRICT), name (NAME), code (CODE), the population per ward (POP) and the expected number of cases (CEXP) [@lawson1999disease].


# Install and load packages 

This practical requires the following packages to be installed and attached: `sf`, `dplyr`, `tidyverse`,  `nimble`, `coda`, `spdep`, `patchwork`, `GGally`, `ggmcmc` and `INLA`. 

* To install the entire suite of packages, we can use:
```{r eval = FALSE,  results="hide"}
#install.packages(c("sf", "dplyr", "tidyverse", "nimble", "coda", "spdep", "patchwork", "GGally", "ggmcmc"), dependencies = TRUE, repos = "http://cran.r-project.org")
```

* For `INLA`, you can use 
```{r eval = FALSE,  results="hide"}
# install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)
```

* Then, load the needed packages:
```{r eval = TRUE, results="hide", message=FALSE, warning=FALSE}

library(sf)           # Simple feature for R
library(dplyr)        # A package for data manipulation
library(tidyverse)    # A package for data manipulation and plotting


library(nimble)       # A package for performing MCMC in R

library(coda)         # A package for summarizing and plotting of MCMC output 
                      # and diagnostic tests
library(spdep)        # A package to calculate neighbors

library(patchwork)    # A package to combine plots
library(GGally)       # A package to do pairplots
library(ggmcmc)       # MCMC diagnostics with ggplot

library(INLA)         # A package for quick Bayesian inference (here used to calculate
                      # the scale for BYM2)

library(kableExtra)   # packages for formatting tables in rmarkdown
```



# Import and explore the Scottish lip cancer data

* Import the health data
```{r eval=TRUE}
ScotLip <- read_sf("ScotLip.shp")
ScotLip
```

* Calculate the SIRs (standarized incidence ratios) and add them to the data frame `ScotLip`
```{r eval=TRUE}
ScotLip$crudeSIR <- ScotLip$CANCER/ScotLip$CEXP
```

* Some plots for the crude SIRs
```{r eval=TRUE, out.width="60%"}
ggplot() + geom_boxplot(data = ScotLip, aes(y = crudeSIR)) + theme_bw() + theme(text = element_text(size = 12))-> p1
ggplot() + geom_sf(data = ScotLip, aes(fill = crudeSIR), col = "NA") + scale_fill_viridis_c() + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 12)) -> p2
(p1|p2)
```


# Obtaining the posterior relative risks (SIRs)

## Model with an unstructured random effect

### Model specification

The SIRs will be smoothed using the Poisson-logNormal model. The inference is done with `NIMBLE` called through R. 
In particular, let each ward $i$ be indexed by  the integers $1, 2,...,N$. The model is as follows:
\[
\begin{eqnarray}
O_{i}|\lambda_{i}  & \sim & \text{Poisson}(\lambda_{i}E_{i} )  \\
\log(\lambda_{i}) & = & \alpha + \theta_{i}  \\
\theta_{i} & \sim & N(0,1/\tau_{\theta})
\end{eqnarray}
\]

where $O_i$ is the observed number of scottish lip cancer cases, $E_{i}$ the expected, $\alpha$ is an intercept term denoting the average log SIR, $\theta_{i}$ is a random intercept and $\tau_{\theta}$ a precision (reciprocal of the variance) term that controls the magnitude of $\theta_{i}$.

### Code

* We will first write the model in nimble:
```{r eval=TRUE}
UnstrCode <- nimbleCode(
  {
    for (i in 1:N){
      
      O[i] ~ dpois(mu[i])                                # Poisson for observed counts 
      log(mu[i]) <- log(E[i]) + alpha + theta[i] 
      
      theta[i] ~ dnorm(0, tau = tau.theta) 			         # area-specific unstr random effects
      SIR[i] <- exp(alpha + theta[i])		                 # area-specific SIR
      resSIR[i] <- exp(theta[i]) 			                   # area-specific residual SIR
      e[i] <- (O[i]-mu[i])/sqrt(mu[i])     	             # residuals      
    } 
    
    # Priors:
    alpha ~ dflat()                                      # Unif(-inf, +inf)
    overallSIR <- exp(alpha)                             # overall SIR across study region
    
    tau.theta ~ dgamma(1, 0.01)                          # prior for the precision
    sigma2.theta <- 1/tau.theta                          # variance of theta
  }
)
```


* Create data object as required for `NIMBLE.` Here we define the data and its constants. In our case the data is the observed number of lip cancer cases, whereas the constants the total number of wards and the expected number of cases per ward:
```{r eval=TRUE}
# Obtain the number of wards
N <- dim(ScotLip)[1] 

# Format the data for NIMBLE in a list
ScotLipdata = list(
                 O = ScotLip$CANCER         # observed nb of cases
)

ScotLipConsts <-list(
                 N = N,                     # nb of wards   
                 E = ScotLip$CEXP           # expected number of cases
                   
)
  
```

* What are the parameters to be initialised? Create a list with two elements (each a list) with different initial values for the parameters:
```{r eval=TRUE}
# Initialise the unknown parameters, 2 chains
inits <- list(
  list(alpha=0.01, tau.theta=10, theta = rep(0.01,times=N)),  # chain 1
  list(alpha=0.5, tau.theta=1, theta = rep(-0.01,times=N)))   # chain 2
```


* Set the parameters that will be monitored:
```{r eval=TRUE}
# Monitored parameters
params <- c("sigma2.theta", "overallSIR", "resSIR", "SIR", "e", "mu", "alpha")
```

Note that the parameters that are not set, will NOT be monitored!

* Specify the MCMC setting:
```{r eval=TRUE}
# MCMC setting
ni <- 50000  # nb iterations 
nt <- 10     # thinning interval
nb <- 10000   # nb iterations as burn-in 
nc <- 2      # nb chains
```

The burn-in should be long enough to discard the initial part of the Markov chains that have not yet converged to the stationary distribution.

* Run the MCMC simulations calling Nimble from R using the function `nimbleMCMC()`
```{r eval=TRUE}
UnstrCodesamples <- nimbleMCMC(code = UnstrCode,
                      data = ScotLipdata,
                      constants = ScotLipConsts, 
                      inits = inits,
                      monitors = params,
                      niter = ni,
                      nburnin = nb,
                      thin = nt, 
                      nchains = nc, 
                      setSeed = 9, 
                      progressBar = FALSE,      # if true then you can monitor the progress
                      samplesAsCodaMCMC = TRUE, 
                      summary = TRUE, 
                      WAIC = TRUE
                      )
```


### Check convergence

Note that specifying `samplesAsCodaMCMC = TRUE`, we can use the functionality of `coda` package to run diagnostics. 


* *The Gelman-Rubin diagnostic (Rhat)*
The Gelman-Rubin diagnostic evaluates MCMC convergence by analyzing the difference between multiple Markov chains [@gelman1992]. The convergence is assessed by comparing the estimated between-chains and within-chain variances for each model parameter. Large differences between these variances indicate nonconvergence.
Indeed, from the summary statistics, we saw displayed the *potential scale reduction factor (psrf)* or *Rhat*.  
When the scale reduction factor is high (perhaps greater than 1.1), then we should run our chains out longer to improve convergence to the stationary distribution.

We will use the function `gelman.diag()` from the package `coda` to calculate the *Rhat* and get an idea of which random variables have already converged:
```{r eval=TRUE}
GR.diag <- gelman.diag(UnstrCodesamples$samples, multivariate = FALSE)
all(GR.diag$psrf[,"Point est."] < 1.1) 
which(GR.diag$psrf[,"Point est."] > 1.1) 
```
 
There is some evidence that most random variables of interest have converged. Lets assess it more. We can do traceplots and autocorrelation plots using the `coda` package:

* Check the overall SIR: 
```{r eval=TRUE, figure.width=7, fig.height=6, fig.align="center"}  
unstr_ggmcmc <- ggs(UnstrCodesamples$samples) 

unstr_ggmcmc %>% filter(Parameter == "overallSIR") %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12)) -> p1

unstr_ggmcmc %>% filter(Parameter == "overallSIR") %>% 
  ggs_autocorrelation + theme_bw() + theme(text = element_text(size = 12)) -> p2

p1/p2
```	

* Check the $1/\tau_{\theta}$: 
```{r eval=TRUE, figure.width=7, fig.height=6, fig.align="center"}  
unstr_ggmcmc %>% filter(Parameter == "sigma2.theta") %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12)) -> p1

unstr_ggmcmc %>% filter(Parameter == "sigma2.theta") %>% 
  ggs_autocorrelation + theme_bw() + theme(text = element_text(size = 12)) -> p2

p1/p2
```	

* Check the SIR of a random area: 
```{r eval=TRUE, figure.width=7, fig.height=6, fig.align="center"}  
set.seed(9)
ra <- sample(1:N, size = 1)

unstr_ggmcmc %>% filter(Parameter == paste0("SIR[", ra, "]")) %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12)) -> p1

unstr_ggmcmc %>% filter(Parameter == paste0("SIR[", ra, "]")) %>% 
  ggs_autocorrelation + theme_bw() + theme(text = element_text(size = 12)) -> p2

p1/p2
```	

### Extract results

Once we are more or less sure that the chain has reached convergence we can extract the results

* Summarize posteriors from `UnstrCodesamples`:
```{r eval=TRUE, results="hide"}
head(UnstrCodesamples$summary$chain1, digits = 3)
head(UnstrCodesamples$summary$chain2, digits = 3)
head(UnstrCodesamples$summary$all.chains, digits = 3)
# also
UnstrCodesamples$summary$chain2[c(1, 2, 3, 7),]
# or
UnstrCodesamples$summary$chain2["sigma2.theta",]

```

* We can obtain the posterior distribution of the random variable of interest, say `sigma2.theta`:
```{r eval=TRUE, fig.align="center"} 

unstr_ggmcmc %>% filter(Parameter == "sigma2.theta") %>% 
                 ggs_histogram() + theme_bw() + theme(text = element_text(size = 12)) -> p1

unstr_ggmcmc %>% filter(Parameter == "sigma2.theta") %>% 
                 ggs_density() + theme_bw() + theme(text = element_text(size = 12)) -> p2

p1/p2
  
```	


* To map the smoothed SIRs in `R` we extract the posterior mean of the SIRs and add it on the shapefile. 
```{r eval=TRUE} 
ScotLip$unstr_SIR <- UnstrCodesamples$summary$all.chains[paste0("SIR[", 1:N, "]"), "Median"]
```	

* Using `ggplot2` to produce a map of the smoothed SIRs
```{r eval=TRUE, out.width="70%"} 

ggplot() + geom_sf(data = ScotLip, aes(fill = unstr_SIR), col = NA) + theme_bw() + 
                   scale_fill_viridis_c(limits = c(0,5)) + 
                   theme(axis.text = element_blank(), 
                         text = element_text(size = 12))
```	

* We can also do a boxplot to assess the level of global smoothing
```{r eval=TRUE, out.width= "50%"} 
dat.box = data.frame(SMR = c(ScotLip$crudeSIR, ScotLip$unstr_SIR), 
                     type = c(rep("Crude", times = N), rep("SIR_unstr", times = N)))
ggplot() + geom_boxplot(data = dat.box, aes(x = type, y = SMR, fill = type), alpha = .4) + 
           theme_bw() + theme(text = element_text(size = 14))
```	
* We can also get the posterior probability that the spatial SIR per area is larger than 1:
```{r eval=TRUE, , out.width= "60%", fig.align="center"} 

postProb <- sapply(paste0("SIR[", 1:N, "]"), 
                   function(X) mean(UnstrCodesamples$samples$chain1[,X]>1))
ScotLip$unstr_postProb <- postProb

ggplot() + geom_sf(data = ScotLip, aes(fill = postProb), col = NA) + theme_bw() + 
                   scale_fill_viridis_c(limits = c(0,1)) + 
                   theme(axis.text = element_blank(), 
                         text = element_text(size = 12)) -> p1

ggplot() + geom_boxplot(data = ScotLip, aes(y = postProb)) + 
                        scale_fill_viridis_d(alpha = .5) + theme_bw() + xlim(c(-1,1)) + 
                        theme(axis.text.x = element_blank(), 
                              text = element_text(size = 12)) -> p2

(p1|p2) + plot_annotation(title = 'Posterior probability that Pr(SIR>1)')

```	


## Model with an ICAR random effect

We will now fit an ICAR model [@besag1991]. Note that an ICAR prior models strong spatial dependence. The model is as follows:
\[
\begin{eqnarray}
O_{i}|\lambda_{i}  & \sim & \text{Poisson}(\lambda_{i}E_{i} )  \\
\log(\lambda_{i}) & = & \alpha + \phi_{i}  \\
\phi_{i} & \sim & \text{ICAR}({\bf W}, 1/\tau_{\phi})
\end{eqnarray}
\]

where in this case the random effects are denoted with $\phi$ and model strong spatial dependence. The elements of the adjacency matrix ${\bf W}$ are defined as:
\begin{equation}
w_{ij} = 
\begin{cases} 
1 & \text{if } j \in \partial_i  \\
0         & \text{otherwise}
\end{cases}
\end{equation}

where $\partial_i$ is the set of neighbors of $j$.

* Create the list of neighbors based on the shapefile
```{r echo=TRUE, eval=TRUE,}
Wards_nb <- poly2nb(pl = ScotLip)
nbWB_A <- nb2WB(nb = Wards_nb)
```	


* Specify the model. The priors of the model are kept the same. This is not a good practice and in general priors should be motivated by the research question in hand. Nevertheless, the purpose of this tutorial is show how these models are coded in `NIMBLE` and not to answer any research question specific for the scottish lip cancer data. 
```{r eval=TRUE}


ICARCode <- nimbleCode(
  {
    for (i in 1:N){
      
      O[i] ~ dpois(mu[i])                                # Poisson for observed counts 
      log(mu[i]) <- log(E[i]) + alpha + phi[i]
      
      SIR[i] <- exp(alpha + phi[i])		                   # area-specific SIR
      resSIR[i] <- exp(phi[i]) 			                     # area-specific residual SIR
      e[i] <- (O[i]-mu[i])/sqrt(mu[i])     	             # residuals      
    } 
    
    # ICAR
    phi[1:N] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N], tau.phi, zero_mean = 1) 
    # the zero_mean is to impose sum to zero constrains in case you have an intercept in the model
    
    # Priors:
    alpha ~ dflat()                                      # vague uniform prior
    overallSIR <- exp(alpha)                             # overall SIR across study region
    
    tau.phi ~ dgamma(1, 0.01)                            # precision of the ICAR component
    sigma2.phi <- 1/tau.phi                              # variance of the ICAR component
    
  }
  
)

```


* Format the data as previously, now we need to specify the characteristics of the adjacency matrix in the constants. Note that L is the total number of neighbors, N as before the total number of wards, adj is an index showing the neighboring areas, num shows how many neighbors each area has, weights is a vector of 1s with length L. Notice the the `zero_mean = 1` imposes sum to zero constraints. This is important in case you include an intercept in the model to bypass the identifiability issues. If you dont have an intercept in the model then you should remove the sum to zero constraints. In theory, given a flat prior on the intercept both approaches should be identical. 
```{r echo=TRUE, eval=TRUE}

# Obtain the number of wards
N <- dim(ScotLip)[1] 

# Format the data for NIMBLE in a list
ScotLipdata = list(
                 O = ScotLip$CANCER                 # observed nb of cases
)        
        
ScotLipConsts <-list(        
                 N = N,                             # nb of wards   
                 E = ScotLip$CEXP,                  # expected number of cases
                 
                 # and the elements of the neighboring matrix:
                 L = length(nbWB_A$weights),        
                 adj = nbWB_A$adj,                             
                 num = nbWB_A$num,
                 weights = nbWB_A$weights
                   
)

```

* Initial values (as before):
```{r eval=TRUE}
# Initialise the unknown parameters, 2 chains
inits <- list(
  list(alpha=0.01, 
       tau.phi=10, phi = rep(0.01,times=N)),  # chain 1
  list(alpha=0.5, 
       tau.phi=1, phi = rep(-0.01,times=N))   # chain 2
  )
```

* Set the parameters that will be monitored:
```{r eval=TRUE}
# Monitored parameters
params <- c("sigma2.phi", "overallSIR", "resSIR", "SIR", "e", "mu", "alpha")
```

* Specify the MCMC setting (identical as before):
```{r eval=TRUE}
# MCMC setting
ni <- 50000  # nb iterations 
nt <- 10     # thinning interval
nb <- 10000   # nb iterations as burn-in 
nc <- 2      # nb chains
```

* and run the sampler
```{r eval=TRUE}
ICARCodesamples <- nimbleMCMC(code = ICARCode,
                               data = ScotLipdata,
                               constants = ScotLipConsts, 
                               inits = inits,
                               monitors = params,
                               niter = ni,
                               nburnin = nb,
                               thin = nt, 
                               nchains = nc, 
                               setSeed = 9, 
                               progressBar = FALSE,     
                               samplesAsCodaMCMC = TRUE, 
                               summary = TRUE, 
                               WAIC = TRUE
)
```


### Check convergence

We can check convergence similar as before:

```{r eval=TRUE}
GR.diag <- gelman.diag(ICARCodesamples$samples, multivariate = FALSE)
all(GR.diag$psrf[,"Point est."] < 1.1) 
which(GR.diag$psrf[,"Point est."] > 1.1) 
```

Note that you can get an overall plot for the G-R diagnostic using the `ggmcmc` package. Lets select specific variables because it will look messy, say `overallSIR`, `sigma2.phi`, `alpha` and `resSIR[1:3]`

```{r eval=TRUE, out.width = "50%", fig.align="center"}
ICAR_ggmcmc <- ggs(ICARCodesamples$samples) 

ICAR_ggmcmc %>% filter(Parameter == c("overallSIR", "sigma2.phi", 
                                      "alpha", paste0("resSIR[", 1:3, "]"))) %>% 
                ggs_Rhat() + xlab("R_hat") + theme_bw() + theme(text = element_text(size = 16))

```

* Check the overall SIR: 
```{r eval=TRUE, figure.width=7, fig.height=6, fig.align="center"}  
ICAR_ggmcmc %>% filter(Parameter == "overallSIR") %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12)) -> p1

ICAR_ggmcmc %>% filter(Parameter == "overallSIR") %>% 
  ggs_autocorrelation + theme_bw() + theme(text = element_text(size = 12)) -> p2

p1/p2
```	
Of course more checks are needed to make sure that the random variables of interest have converged, but the initial impression is positive. 

For comparison purposes, I will attached the ICAR SIR on the initial `shp`
```{r eval=TRUE} 
ScotLip$ICAR_SIR <- ICARCodesamples$summary$all.chains[paste0("SIR[", 1:N, "]"), "Median"]
```	

## The BYM model

The BYM model is probably the most popular model for disease mapping. It is basically a combination of the ICAR model and a model with an unstructured component, see [@besag1991] for more details:
\[
\begin{eqnarray}
O_{i}|\lambda_{i}  & \sim & \text{Poisson}(\lambda_{i}E_{i} )  \\
\log(\lambda_{i}) & = & \alpha + \theta_{i} + \phi_{i}  \\
\theta_{i} & \sim & N(0,1/\tau_{\theta}) \\
\phi_{i} & \sim & \text{ICAR}({\bf W}, 1/\tau_{\phi})
\end{eqnarray}
\]

The code is as follows:
```{r eval=TRUE}
BYMCode <- nimbleCode(
  {
    for (i in 1:N){
      
      O[i] ~ dpois(mu[i])                                  # Poisson for observed counts 
      log(mu[i]) <- log(E[i]) + alpha + theta[i] + phi[i]
      
      theta[i] ~ dnorm(0, tau = tau.theta) 			           # area-specific RE
      SIR[i] <- exp(alpha + theta[i] + phi[i])		         # area-specific SIR
      resSIR[i] <- exp(theta[i] + phi[i]) 			           # area-specific residual SIR
      e[i] <- (O[i]-mu[i])/sqrt(mu[i])     	               # residuals      
    } 
    
    # ICAR
    phi[1:N] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N], tau.phi, zero_mean = 1)
    
    # Priors:
    alpha ~ dflat()                                       # vague uniform prior
    overallSIR <- exp(alpha)                              # overall SIR across study region
    
    tau.theta ~ dgamma(1, 0.01)                           # prior for the precision of theta
    sigma2.theta <- 1/tau.theta                           # variance of theta
    
    tau.phi ~ dgamma(1, 0.01)                             # prior for the precision of phi
    sigma2.phi <- 1/tau.phi                               # variance of phi
    
  }
  
)
```	

In a similar way as done before you need to specify the data. The data that we are using is identical as for the ICAR so we will skip this. 

* Initial values:
```{r eval=TRUE}
inits <- list(
  list(alpha=0.01, 
       tau.theta=10, theta = rep(0.01,times=N), 
       tau.phi=10, phi = rep(0.01,times=N)),  # chain 1
  list(alpha=0.5, 
       tau.theta=1, theta = rep(-0.01,times=N), 
       tau.phi=1, phi = rep(-0.01,times=N))   # chain 2
)
```	

* Parameters to monitor:
```{r eval=TRUE}
# Monitored parameters
params <- c("sigma2.phi", "sigma2.theta", "overallSIR", "resSIR", "SIR", "e", "mu", "alpha")
```

* Run the code:

```{r eval=TRUE}
BYMCodesamples <- nimbleMCMC(code = BYMCode,
                              data = ScotLipdata,
                              constants = ScotLipConsts, 
                              inits = inits,
                              monitors = params,
                              niter = ni,
                              nburnin = nb,
                              thin = nt, 
                              nchains = nc, 
                              setSeed = 9, 
                              progressBar = FALSE,     
                              samplesAsCodaMCMC = TRUE, 
                              summary = TRUE, 
                              WAIC = TRUE
)
```


### Check convergence

* Check the G-R diagnostic
```{r eval=TRUE}
GR.diag <- gelman.diag(BYMCodesamples$samples, multivariate = FALSE)
all(GR.diag$psrf[,"Point est."] < 1.1) 
which(GR.diag$psrf[,"Point est."] > 1.1) 
```

* Check `sigma2.theta`
```{r eval=TRUE, figure.width=7, fig.height=6, fig.align="center"}

BYM_ggmcmc <- ggs(BYMCodesamples$samples) 
BYM_ggmcmc %>% filter(Parameter == "sigma2.theta") %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12)) -> p1

BYM_ggmcmc %>% filter(Parameter == "sigma2.theta") %>% 
  ggs_autocorrelation + theme_bw() + theme(text = element_text(size = 12)) -> p2

p1/p2

```	
Seems that we need to tune values given for the MCMC setting to get convergence for `sigma2.theta` (not done here). 

and attach the BYM SIR on the initial `shp`
```{r eval=TRUE} 
ScotLip$BYM_SIR <- BYMCodesamples$summary$all.chains[paste0("SIR[", 1:N, "]"), "Median"]
```	



## The BYM2 model

BYM2 is a reparametrization of BYM model which has the following features: 1) It resolves the identifiability issues of the BYM model 2) The hyperparameters of the field are almost orthogonal, facilitating clear interpretation and making prior assignment intuitive, 3) The spatially structured random effect is scaled and thus the hyperprior used in one graph have the same interpretation in any other graph (for instance the hyperpriors for the graph of scottland will have same interpretation as the hyperpriors for the graph of Greece only if the spatially structured random effect is scaled). For more information see @riebler2016 and @freni2018.
\[
\begin{eqnarray}
O_{i}|\lambda_{i}  & \sim & \text{Poisson}(\lambda_{i}E_{i} )  \\
\log(\lambda_{i}) & = & \alpha + b_{i}  \\
b_{i} & = & \frac{1}{\tau_b}\big(\theta\sqrt{1-\rho} + \phi\sqrt{\rho}\big)\\
\theta_{i} & \sim & N(0,1/\tau_{\theta}) \\
\phi_{i} & \sim & \text{ICAR}({\bf W^*}, 1)
\end{eqnarray}
\]


The code is as follows:
```{r eval=TRUE}
BYM2Code <- nimbleCode(
  {
    for (i in 1:N){
      
      O[i] ~ dpois(mu[i])                                # Poisson likelihood for observed counts 
      log(mu[i]) <- log(E[i]) + alpha + b[i]
      
      b[i] <- (1/sqrt(tau.b))*(sqrt((1-rho))*theta[i] + sqrt(rho/scale)*phi[i])
      
      theta[i] ~ dnorm(0, tau = tau.theta) 			         # area-specific RE
      SIR[i] <- exp(alpha + b[i])	            	         # area-specific SIR
      resSIR[i] <- exp(b[i]) 			                       # area-specific residual SIR
      e[i] <- (O[i]-mu[i])/sqrt(mu[i])     	             # residuals      
    } 
    
    # ICAR
    phi[1:N] ~ dcar_normal(adj[1:L], weights[1:L], num[1:N], tau = 1, zero_mean = 1) # its scaled so tau = 1
    
    # Priors:
    # intercept
    alpha ~ dflat()                                      # vague uniform prior
    overallSIR <- exp(alpha)                             # overall SIR across study region
    
    # precision parameter of the reparametrization
    tau.b ~ dgamma(1, 0.01)                              # prior for the precision of b
    sigma2.b <- 1/tau.b                                  # the variance of b
    
    # precision parameter of theta
    tau.theta ~ dgamma(1, 0.01)                          # prior for the precision of theta
    sigma2.theta <- 1/tau.theta                          # the variance of theta
    
    # mixing parameter
    rho ~ dbeta(1, 1)                                    # prior for the mixing parameter
  }
  
)
```	

* We need to calculate the scale:
```{r eval=TRUE}
W.scale <- nb2mat(Wards_nb, zero.policy = TRUE, style = "B")           # Wards_nb is the spatial object created using the shapefile and the spdep package
W.scale <- -W.scale
diag(W.scale) <- abs(apply(W.scale, 1, sum))
# solve(W.scale) # this should not work since by definition the matrix is singular

Q = inla.scale.model(W.scale, constr=list(A=matrix(1, nrow=1, ncol=nrow(W.scale)), e=0))
scale = exp((1/nrow(W.scale))*sum(log(1/diag(Q))))
```	

* Add the scale in the constants (the data file remains the same):
```{r eval=TRUE}
LipCancerConsts <-list(
  N = N,                    
  E = ScotLip$CEXP,                      
  
  # elements of neighboring matrix
  adj = nbWB_A$adj, 
  weights = nbWB_A$weights, 
  num = nbWB_A$num, 
  L = length(nbWB_A$weights), 
  
  # scale
  scale = scale
)
```	

* Set the initials:
```{r eval=TRUE}
inits <- list(
  list(alpha = 0.01, 
       tau.theta = 10, 
       theta = rep(0.01, times = N), 
       phi = rep(0.01, times = N), 
       tau.b = 10, 
       rho = .2),                        # chain 1
  list(alpha = 0.5, 
       tau.theta = 1, 
       theta = rep(-0.01 ,times = N), 
       phi = rep(-0.01, times = N), 
       tau.b = 1, 
       rho = .7)                         # chain 2
)
```	


* Set the parameters to monitor:
```{r eval=TRUE}
params <- c("sigma2.b", "sigma2.theta", "rho", "overallSIR", "resSIR", "SIR", "e", "mu", "alpha")
```

* Run the BYM2
```{r eval = TRUE}
BYM2Codesamples <- nimbleMCMC(code = BYM2Code,
                              data = ScotLipdata,
                              constants = LipCancerConsts, 
                              inits = inits,
                              monitors = params,
                              niter = ni,
                              nburnin = nb,
                              thin = nt, 
                              nchains = nc, 
                              setSeed = 9, 
                              progressBar = FALSE,     
                              samplesAsCodaMCMC = TRUE, 
                              summary = TRUE, 
                              WAIC = TRUE
)
```

### Check convergence

* Check the G-R diagnostic
```{r eval=TRUE}
GR.diag <- gelman.diag(BYM2Codesamples$samples, multivariate = FALSE)
all(GR.diag$psrf[,"Point est."] < 1.1) 
which(GR.diag$psrf[,"Point est."] > 1.1) 
```


attached the SIR on the initial `shp`
```{r eval=TRUE} 
ScotLip$BYM2_SIR <- BYM2Codesamples$summary$all.chains[paste0("SIR[", 1:N, "]"), "Median"]
```	

# Comparing results from the different models

In this section we will compare the output of the three different models. In this comparison we will focus on SIRs, exceedance probabilities and the different intercepts. 

## Smoothed SIRs

First, I will take some random samples of the SIRs and check if we are fine with convergence (keep in mind the the G-R test provided evidence towards convergence):
```{r eval=FALSE}

set.seed(9)
samples <- sample(x = 1:N, size = 5)
unstr_ggmcmc %>% filter(Parameter == paste0("SIR[", samples, "]")) %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12))

samples <- sample(x = 1:N, size = 5)
ICAR_ggmcmc %>% filter(Parameter == paste0("SIR[", samples, "]")) %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12))

samples <- sample(x = 1:N, size = 5)
BYM_ggmcmc %>% filter(Parameter == paste0("SIR[", samples, "]")) %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12))

BYM2_ggmcmc <- ggs(BYM2Codesamples$samples)
samples <- sample(x = 1:N, size = 5)
BYM2_ggmcmc %>% filter(Parameter == paste0("SIR[", samples, "]")) %>% 
  ggs_traceplot + theme_bw() + theme(text = element_text(size = 12))

```

Now we can start comparing
```{r}
ggplot() + geom_sf(data = ScotLip, aes(fill = crudeSIR), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 6.6)) +  theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p1

ggplot() + geom_sf(data = ScotLip, aes(fill = unstr_SIR), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 6.6)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p2

ggplot() + geom_sf(data = ScotLip, aes(fill = ICAR_SIR), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 6.6)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p3

ggplot() + geom_sf(data = ScotLip, aes(fill = BYM_SIR), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 6.6)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p4

ggplot() + geom_sf(data = ScotLip, aes(fill = BYM2_SIR), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 6.6)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p5

(p1|p2|p3)/(p4|p5)

```

It is clear that there is some level of smoothness induced in the SIR using the different hierarchical Bayesian models. For the model with just the unstructured term, as expected we have a global but not local smoothing, whereas for the ICAR and the rest of the models, it is clear that the smoothing was also performed locally, using the adjacency structure. We can also assess the boxplots of the SIRs:

```{r out.width = '60%'}
data.frame(model = c(rep("Crude", N), rep("Unstr", N), 
                     rep("ICAR", N), rep("BYM", N), rep("BYM2", N)),
                     SIR =c(ScotLip$crudeSIR, ScotLip$unstr_SIR, ScotLip$ICAR_SIR, 
                            ScotLip$BYM_SIR, ScotLip$BYM2_SIR)) %>% 
  
  ggplot() + geom_boxplot(aes(x = model, y = SIR)) + theme_bw() + 
             theme(text = element_text(size = 14))
```


* Lets do some pairplots too:
```{r}
suppressMessages(
  print(
ScotLip %>% select(crudeSIR, unstr_SIR, ICAR_SIR, BYM_SIR, BYM2_SIR) %>% 
            as.data.frame() %>% 
            select(-geometry) %>% 

  ggpairs(lower = list(continuous = wrap("points", size=.8))) + 
  theme_bw() + theme(text = element_text(size = 14))
  )
)
```


* We can also compare the posterior probabilities:

```{r }

ScotLip$ICAR_postProb <- sapply(paste0("SIR[", 1:N, "]"), 
                                function(X) mean(ICARCodesamples$samples$chain1[,X]>1))
ScotLip$BYM_postProb <- sapply(paste0("SIR[", 1:N, "]"), 
                               function(X) mean(BYMCodesamples$samples$chain1[,X]>1))
ScotLip$BYM2_postProb <- sapply(paste0("SIR[", 1:N, "]"), 
                                function(X) mean(BYM2Codesamples$samples$chain1[,X]>1))

ggplot() + geom_sf(data = ScotLip, aes(fill = unstr_postProb), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 1)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p1

ggplot() + geom_sf(data = ScotLip, aes(fill = ICAR_postProb), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 1)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p2

ggplot() + geom_sf(data = ScotLip, aes(fill = BYM_postProb), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 1)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p3

ggplot() + geom_sf(data = ScotLip, aes(fill = BYM2_postProb), col = "NA") + 
           scale_fill_viridis_c(limits = c(0, 1)) + theme_bw() + 
           theme(axis.text = element_blank(), text = element_text(size = 10)) -> p4

(p1|p2)/(p3|p4)

```

Note that we can also calculate exceedance probabilities in the sampler using the  `step()` function. For example in the BYM2, inside the for loop under the calculation of e, we can add this: `proba.resRR[i] <- step(resRR[i]-1)`. Make sure you monitor `proba.resRR` by specifying it in the `params`. After the sampler works we can take its posterior mean and get the posterior probability.  

* For completeness I am also showing a table of the intercept and hyperparameters. Note that any comparison is conditional on convergence. More effort should be put to examine convergence and to tune the MCMC setting. The table presents the median posterior of the random variables together with the 95% credibility regions. 

```{r echo = FALSE}

CrI <- function(X) paste0(round(X[,1], digits = 2), " (", round(X[,2], digits = 2), ", ", round(X[,3], digits = 2), ")")

tab <- data.frame(matrix(nrow = 6, ncol = 4))
colnames(tab) <- c("Unstr", "ICAR", "BYM", "BYM2")
rownames(tab) <- c("alpha", "sigma2.theta", "sigma2.phi", "sigma2.b", "rho", "WAIC")

tab[c("alpha", "sigma2.theta"), "Unstr"] <- 
  CrI(UnstrCodesamples$summary$all.chains[c("alpha", "sigma2.theta"), c("Median", "95%CI_low", "95%CI_upp")])

tab[c("alpha", "sigma2.phi"), "ICAR"] <- 
  CrI(ICARCodesamples$summary$all.chains[c("alpha", "sigma2.phi"), c("Median", "95%CI_low", "95%CI_upp")])

tab[c("alpha", "sigma2.theta", "sigma2.phi"), "BYM"] <- 
  CrI(BYMCodesamples$summary$all.chains[c("alpha", "sigma2.theta", "sigma2.phi"), c("Median", "95%CI_low", "95%CI_upp")])

tab[c("alpha", "sigma2.theta", "sigma2.b", "rho"), "BYM2"] <- 
  CrI(BYM2Codesamples$summary$all.chains[c("alpha", "sigma2.theta", "sigma2.b", "rho"), c("Median", "95%CI_low", "95%CI_upp")])
tab[c("sigma2.phi"), "BYM2"] <- 1

tab["WAIC",] <- round(c(UnstrCodesamples$WAIC, ICARCodesamples$WAIC, BYMCodesamples$WAIC, BYM2Codesamples$WAIC), digits = 2)

options(knitr.kable.NA = '-')
rownames(tab) <- c("$\\alpha$", "$\\sigma^2_{\\theta}$", "$\\sigma^2_{\\phi}$", "$\\sigma^2_{b}$", "$\\rho$", "WAIC")
knitr::kable(tab, caption = "Median and 95%CrI for the intercept and hyperparameters of the models. The last row shows the WAIC for model comparison.") %>% 
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center")

```

Some interesting features for the table above:

1. The uncertainty of the intercept is larger for the model with the unstructured random effect.
2. $\sigma^2_{\theta}$ is almost negligible for BYM and BYM2.
3. $\rho$ is 0.71 for the BYM2 implying that large part of the variation is due to the structured spatial effect (but of course the uncertainty is large).
4. The Watanabe–Akaike information criterion or WAIC [@watanabe2013] suggests that the best fit model is the ICAR (the smaller the WAIC the better the fit). 


# Conclusion
The purpose of this tutorial is to show how to fit spatial models using `NIMBLE` [@nimble]. Nevertheless, the tutorial does not provide an exhaustive list of spatial model that can be fit in `NIMBLE`. Examples of other models include the proper model [@cressie1989], the Leroux model [@leroux2000], see also @best2005 and @lee2011 for simulation studies and relevant comparisons. Additional sources for code include the tutorial by Lawson [@lawson2020nimble] and also chapter 9 of \href{https://r-nimble.org/html_manual/cha-welcome-nimble.html}{the NIMBLE manual}.


# References

