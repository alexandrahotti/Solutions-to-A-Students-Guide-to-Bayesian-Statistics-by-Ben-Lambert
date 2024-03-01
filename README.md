# Solutions-to-Problems-in-Bayesian-Statistics
This repository contains my solutions to the assignments in the book: "A Student’s Guide to Bayesian Statistics" by Ben Lambert. I will update the repository with my solutions continuously.

Each chapter of the book has its corresponding folder in this repository. These solutions consist of Python code as well as pdfs. 


## Content

### An introduction to Bayesian inference

#### Chapter 2 - The subjective worlds of Frequentist and Bayesian statistics
The code for this section can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/tree/master/2%20-%20The%20subjective%20worlds%20of%20Frequentist/2.3%20-%20Model%20choice/Q%202.3.1%20-%202.3.2)
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/2%20-%20The%20subjective%20worlds%20of%20Frequentist/2.3%20-%20Model%20choice/solutions_chapter_2.pdf)


#### Chapter 3 - Probability - the nuts and bolts of Bayesian inference
The code for this section can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/tree/master/3%20-%20Probability%20-%20the%20nuts%20and%20bolts%20of%20Bayesian%20inference/3.8%20-%20Breast%20cancer%20revisited)
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/3%20-%20Probability%20-%20the%20nuts%20and%20bolts%20of%20Bayesian%20inference/solutions_chap_3.pdf)


### Understanding the Bayesian formula

#### Chapter 4 - Likelihoods
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/4%20-%20Likelihood/solutions_chapter_4.pdf)

##### Excerpt of some results
<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/4%20-%20Likelihood/4.1%20-%20Blog%20blues/Poisson%20model/MLE%20estimate%20-%20Evaluate%20model/results/generated_time_between_beer_visits.png' width="31%" height="31%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/4%20-%20Likelihood/4.1%20-%20Blog%20blues/Poisson%20model/MLE%20estimate%20-%20Mean%20visit%20rate/results/likelihood_as_function_of_rate_between_first_time_visits_blog.png' width="35%" height="35%"
 />
  

#### Chapter 5 - Priors
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/5%20-%20Prior/solutions_chapter_5.pdf)


#### Chapter 6 - The devil is in the denominator
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/6%20-%20The%20devil%20is%20in%20the%20denominator/solutions_chapter_6.pdf)

##### Excerpt of some results
<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/6%20-%20The%20devil%20is%20in%20the%20denominator/results/NB_posterior.png' width="47%" height="47%"
 />

#### Chapter 7 - The posterior - The goal of Bayesian inference

### Analytic Bayesian methods

#### Chapter 8 - Distributions

##### Excerpt of some results
<p float="left" align='center'>
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/8%20-%20Distributions/Election%20Models/results/Multinomial%20Likelihood%20less%20data/Dirichlet(10%2C10%2C10)%20prior/multinomial_likelihood.png' width="47%" height="47%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/8%20-%20Distributions/Election%20Models/results/Multinomial%20Likelihood%20less%20data/Dirichlet(10%2C10%2C10)%20prior/prior_dir_10_10_10.png' width="47%" height="47%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/8%20-%20Distributions/Election%20Models/results/Multinomial%20Likelihood%20less%20data/Dirichlet(10%2C10%2C10)%20prior/posterior_dir_10_10_10.png' width="57%" height="57%"
 />


#### Chapter 9 - Conjugate priors
##### Excerpt of some results
<p float="left" align='center'>
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/9%20-%20Conjugate%20Priors/Lyme%20disease/results/posterior%20predictive/posterior_predictive_estimation.png' width="47%" height="47%"
 />


#### Chapter 10 - Evaluation of model fit and hypothesis testing

#### Chapter 11 - Making Bayesian analysis objective?

### Computational Bayes

#### Chapter 12 - Leaving conjugates behind: Markov chain Monte Carlo

#### Chapter 13 - Metropolis Hastings
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/solutions_chapter_13.pdf)


##### Modeling presence of Borrelia amongst Ticks

###### Symmetric Kernel - Random Walk Metropolis
Using a Binomial likelihood, a Beta prior and an symmetric Normal jumping kernel.

<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_1_Borrelia/results/symmetric_jumping_kernel/rmw_100_chains_burn_in.png' width="47%" height="47%"
 />

###### Assymmetric Kernel - Metropolis Hastings
Using a Beta-Binomial likelihood, a Gamma prior and an assymmetric log-Normal jumping kernel.
<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_1_Borrelia/results/assymetric_jumping_kernel/joint_prior.png' width="47%" height="47%"
 /> <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_1_Borrelia/results/assymetric_jumping_kernel/posterior_alpha_beta_joint.png' width="47%" height="47%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_1_Borrelia/results/assymetric_jumping_kernel/chains.png' width="47%" height="47%"
 />

##### Modeling Mosquito Death Rate
Using a Poisson Likelihood, a Gamma prior, a Beta Prior, a log-Normal jumping kernel and a beta jumping kernel.
<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_3_Malarial_mosquitoes/results/likelihood.png' width="47%" height="47%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_3_Malarial_mosquitoes/results/chain%204/posterior.png' width="47%" height="47%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_3_Malarial_mosquitoes/results/chain%204/psi_post.png' width="47%" height="47%"
 /><img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/13%20-%20Random%20Walk%20Metropolis/13_3_Malarial_mosquitoes/results/chain%204/mu_post.png' width="47%" height="47%"
 />
  
  
  
#### Chapter 14 - Gibbs Sampling
The report can be found: [HERE](https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/14%20-%20Gibbs%20Sampling/solutions_chapter_14.pdf)

##### The sensitivity and specificity of a test for a disease - Gibbs Sampling

<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/14%20-%20Gibbs%20Sampling/14_1_The%20sensitivity_and_specificity_of_a_test_for_a_disease/results/Q_14_1_7/C_posterior.png' width="47%" height="47%"
 />

##### Coal mining disasters in the UK - Gibbs Sampling
Using Gibbs sampling to estimate the point in time when legislative and societal changes caused a reduction in coal mining disasters in
the UK. The number of disasters per year pre and post legislations were modeled using Poisson Likelihoods: Possion(lambda_1), Possion(lambda_2) with Gamma priors. The point in time when the new legislations were enacted is called n.

<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/14%20-%20Gibbs%20Sampling/14_2_%20Coal_mining_disasters_in_the_UK/results/disasters_yr.png' width="47%" height="47%"
 />
  <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/14%20-%20Gibbs%20Sampling/14_2_%20Coal_mining_disasters_in_the_UK/results/n_posterior_gibbs.png' width="47%" height="47%"
 />
    <img src='https://github.com/alexandrahotti/Solutions-to-Problems-in-Bayesian-Statistics/blob/master/14%20-%20Gibbs%20Sampling/14_2_%20Coal_mining_disasters_in_the_UK/results/lambda1_2_posterior_gibbs.png' width="47%" height="47%"
 />



#### Chapter 15 - Hamiltonian Monte Carlo

#### Chapter 16 - Stan

### Hierarchical models and regression

#### Chapter 17 - Hierarchical models

#### Chapter 18 - Linear regression models

#### Chapter 19 - Generalized  linear models and other animals















