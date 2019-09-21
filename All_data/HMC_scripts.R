
## HMC for a single step
HMC <- function (current_q, U, grad_U, epsilon, L, aSigma){
  q = current_q
  p = rnorm(length(q),0,aSigma)  # independent standard normal variates
  current_p = p
  
  # Make a half step for momentum at the beginning
  
  p = p - epsilon * grad_U(q) / 2
  
  # Alternate full steps for position and momentum
  
  for (i in 1:L)
  {
    # Make a full step for the position
    
    q = q + epsilon * p
    
    # Make a full step for the momentum, except at end of trajectory
    
    if (i!=L) p = p - epsilon * grad_U(q)
  }
  
  # Make a half step for momentum at the end.
  
  p = p - epsilon * grad_U(q) / 2
  
  # Negate momentum at end of trajectory to make the proposal symmetric
  
  p = -p
  
  # Evaluate potential and kinetic energies at start and end of trajectory
  
  current_U = U(current_q)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q)
  proposed_K = sum(p^2) / 2
  
  #   print(current_U-proposed_U)
  #   print(current_K-proposed_K)
  
  # Accept or reject the state at end of trajectory, returning either
  # the position at the end of the trajectory or the initial position
  r <- exp(current_U-proposed_U+current_K-proposed_K)
  # print(r)
  if (runif(1) < r)
  {
    return (q)  # accept
  }
  else
  {
    return (current_q)  # reject
  }
}

## Gradient of the potential with respect to x
fGradSimpleX <- function(x,y){
  aGrad <- -0.5 *exp(1/2 * (-(-20 + x) * (1.38889 * (-20 + x) - 
                                            2.22222 * (-5 + y)) - (-2.22222 * (-20 + x) + 
                                                                     5.55556 * (-5 + y)) * (-5 + y)) + 
                       1/2 * ((-20 + x) * (1.38889 * (-20 + x) - 
                                             2.22222 * (-5 + y)) + (-2.22222 * (-20 + x) + 
                                                                      5.55556 * (-5 + y)) * (-5 + y))) *(-2.77778 * (-20 + x) + 4.44444 * (-5 + y))
  return(aGrad)
}

## Gradient of the potential with respect to y
fGradSimpleY <- function(x,y){
  aGrad <- -0.5 * exp(1/2 * (-(-20 + x) * (1.38889 * (-20 + x) - 
                                             2.22222 * (-5 + y)) - (-2.22222 * (-20 + x) + 
                                                                      5.55556 * (-5 + y)) * (-5 + y)) + 
                        1/2 * ((-20 + x) * (1.38889 * (-20 + x) - 
                                              2.22222 * (-5 + y)) + (-2.22222 * (-20 + x) + 
                                                                       5.55556 * (-5 + y)) * (-5 + y))) * (4.44444 * (-20 + x) - 
                                                                                                             11.1111 * (-5 + y))
  
  return(aGrad)
}

## Gradient of U with respect to both coordinates
grad_U <- function(aQ){
  aGradX <- fGradSimpleX(aQ[[1]],aQ[[2]])
  aGrady <- fGradSimpleY(aQ[[1]],aQ[[2]])
  return(c(aGradX,aGrady))
}

## Potential function given by the -ve log of the posterior
U <- function(aQ){
  x <- aQ[[1]]
  y <- aQ[[2]]
  aU <- -log(0.265258 * exp(
    1/2 * (-(-20 + x) * (1.38889 * (-20 + x) - 
                       2.22222 * (-5 + y)) - (-2.22222 * (-20 + x) + 
                                              5.55556 * (-5 + y)) * (-5 + y))))
  return(aU) 
}

## Simulate a number of HMC steps 
fHMCAllSteps <- function(nIterations,start_q,U, grad_U, epsilon, L, aSigma){
  
  mSamples <- matrix(nrow=nIterations,ncol=2)
  mSamples[1,] <- start_q
  current_q <- start_q
  for (i in 1:nIterations){
    current_q <- HMC(current_q, U, grad_U, epsilon, L, aSigma)
    mSamples[i,] <- current_q
  }
  return(mSamples)
}

## Try out the function
lSamples <- as.data.frame(fHMCAllSteps(100,c(22,6),U,grad_U,0.18,10,0.18))
ggplot(lSamples,aes(x=V1,y=V2)) + geom_path()

mean(lSamples$V2)



HMC_keep <- function (current_q, U, grad_U, epsilon, L, aSigma){
  q = current_q
  p = rnorm(length(q),0,aSigma)  # independent standard normal variates
  current_p = p
  
  # Make a half step for momentum at the beginning
  
  p = p - epsilon * grad_U(q) / 2
  
  # Alternate full steps for position and momentum
  
  lPosition <- matrix(nrow = (L+1),ncol = 2)
  lPosition[1,] <- q
  
  for (i in 1:L)
  {
    
    # Make a full step for the position
    
    q = q + epsilon * p
    lPosition[(i+1),] <- q
    
    # Make a full step for the momentum, except at end of trajectory
    
    if (i!=L) p = p - epsilon * grad_U(q)
  }
  
  # Make a half step for momentum at the end.
  
  p = p - epsilon * grad_U(q) / 2
  
  # Negate momentum at end of trajectory to make the proposal symmetric
  
  p = -p
  
  # Evaluate potential and kinetic energies at start and end of trajectory
  
  current_U = U(current_q)
  current_K = sum(current_p^2) / 2
  proposed_U = U(q)
  proposed_K = sum(p^2) / 2
  
  #   print(current_U-proposed_U)
  #   print(current_K-proposed_K)
  
  # Accept or reject the state at end of trajectory, returning either
  # the position at the end of the trajectory or the initial position
  r <- exp(current_U-proposed_U+current_K-proposed_K)
  # print(r)
  if (runif(1) < 5)
  {
    return (list(q=q,pos=lPosition))  # accept
  }
  else
  {
    return (list(q=current_q,pos=lPosition))  # reject
  }
}

nReplicates <- 100
nStep <- 100
mAll <- matrix(ncol = nReplicates,nrow = nStep)
for(i in 1:nReplicates){
  lTest <- HMC_keep(c(20,5), U, grad_U, 0.18, nStep, 0.18)
  lTemp <- lTest$pos[,1]
  aLen <- length(lTemp)
  mAll[,i] <- lTemp[1:(aLen-1)]
}

library(reshape2)
mAll <- melt(mAll)
library(ggplot2)
ggplot(mAll,aes(x=Var1,colour=as.factor(Var2),y=value)) + geom_path() + theme(legend.position="none") +
  ylab('mu_t') + xlab('number of steps')


par(mfrow=c(1,4))
lTest <- HMC_keep(c(20,5), U, grad_U, 0.18, 10, 0.18)
plot(lTest$pos,type='l',xlab='mu_t',ylab='mu_c',main='L=10') 
lTest <- HMC_keep(c(20,5), U, grad_U, 0.18, 20, 0.18)
plot(lTest$pos,type='l',xlab='mu_t',ylab='mu_c',main='L=20') 
lTest <- HMC_keep(c(20,5), U, grad_U, 0.18, 50, 0.18)
plot(lTest$pos,type='l',xlab='mu_t',ylab='mu_c',main='L=50') 
lTest <- HMC_keep(c(20,5), U, grad_U, 0.18, 100, 0.18)
plot(lTest$pos,type='l',xlab='mu_t',ylab='mu_c',main='L=100') 
