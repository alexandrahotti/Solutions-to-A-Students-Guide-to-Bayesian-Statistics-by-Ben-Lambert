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

## Keeps all the positions of the particle over time
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
  if (runif(1) < r)
  {
    return (list(q=q,pos=lPosition))  # accept
  }
  else
  {
    return (list(q=current_q,pos=lPosition))  # reject
  }
}

