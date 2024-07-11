source("singleStepMethods.R")



adaptRandomizedHMCStep <- function(fun,q0,v0,J.fixed=NULL,
                                   H=1.0,R=4L,
                                   mu=pi,
                                   energy.tol=0.05,
                                   last.eval=NULL,
                                   maxc=12,
                                   I0.prob=0.99,
                                   plot.stats=FALSE){
  
  # randomized number of integration steps
  if(is.null(J.fixed)){
    J <- round(rexp(1,rate=1.0/mu)/H)
  } else {
    J <- J.fixed
  }
  
  # storage
  Cs <- numeric(J)
  If <- numeric(J)
  Ib <- numeric(J)
  lpf <- numeric(J)
  lpb <- numeric(J)
  
  q.left <- q0
  v.left <- v0
  # make these in outermost scope
  q <- q0
  v <- v0
  
  # collect stats
  q.min <- q0
  q.max <- q0
  
  if(is.null(last.eval)){
    ev0 <- fun(q0)
    n.eval <- 1
  } else {
    ev0 <- last.eval
  }
  ev.init <- ev0
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  ev.left <- ev0
  
  if(plot.stats){
    q.big.step <- q0
    q.intermediate <- q0
    t.trace <- 0
    H.trace <- H0
  }
  
  for(i in 1:J){ # loop over "big" integration steps
    I.forw <- maxc
    
    
    for( c in 0:maxc){ # loop over integration accuracies
      if(plot.stats) q.all <- c()
      q <- q.left
      v <- v.left
      ev <- ev.left
      nstep <- R*2^c # number of integration steps
      Hs <- numeric(nstep+1)
      Hs[1] <- -ev.left$lp + 0.5*sum(v.left^2)
      h <- H/nstep
      
      for(s in 1:nstep){ # leap frog integration
        vh <- v + 0.5*h*ev$grad
        q <- q + h*vh
        ev <- fun(q)
        n.eval <- n.eval+1
        v <- vh + 0.5*h*ev$grad
        Hs[s+1] <- -ev$lp + 0.5*sum(v^2)
        if(plot.stats) q.all <- cbind(q.all,q)
      }
      
      energy.err <- max(Hs)-min(Hs)
      
      if(energy.err<energy.tol){ # stop if integration error is sufficiently small 
        I.forw <- c # this is I(z) in notes 
        break
      }
    } # loop over integration accuracies
    
    C <- rng.disc.exp(1,I=I.forw,pI=I0.prob) # sample integration accuracy
    Cs[i] <- C
    If[i] <- I.forw
    lpf[i] <- lp.disc.exp(C,I=I.forw,pI=I0.prob) # forward probability of integration accuracy
    
    if(C>I.forw){ # simulated C was higher than I, need to integrate once more
      if(plot.stats) q.all <- c()
      q <- q.left
      v <- v.left
      ev <- ev.left
      nstep <- R*2^C
      Hs <- numeric(nstep+1)
      Hs[1] <- -ev.left$lp + 0.5*sum(v.left^2)
      h <- H/nstep
      
      for(s in 1:nstep){
        vh <- v + 0.5*h*ev$grad
        q <- q + h*vh
        ev <- fun(q)
        n.eval <- n.eval+1
        v <- vh + 0.5*h*ev$grad
        Hs[s+1] <- -ev$lp + 0.5*sum(v^2)
        if(plot.stats) q.all <- cbind(q.all,q)
      }
    } 
    q.right <- q
    v.right <- v
    ev.right <- ev
    if(plot.stats){
      H.trace <- c(H.trace,Hs[2:length(Hs)])
      t.trace <- c(t.trace,t.trace[length(t.trace)] + h*(1:nstep))
      q.big.step <- cbind(q.big.step,q.right)
      q.intermediate <- cbind(q.intermediate,q.all)
    } else {
      H.trace <- NULL
      t.trace <- NULL
      q.big.step <- NULL
      q.intermediate <- NULL
    }
    
    # now work out backward GIST prob
    if(C==0){
      # due to reversibility of the error estimate, there is no need to do further computations here
      I.back <- 0L
    } else {
      # need to check every lower integration accuracy
      if(C==I.forw){
        I.back <- I.forw
        max.test <- I.forw-1L # only need to check these due to reversibility
      } else {
        I.back <- maxc
        max.test <- maxc
      }
      for(c in 0:max.test){
        q <- q.right
        v <- -v.right
        ev <- ev.right
        nstep <- R*2^c
        Hs <- numeric(nstep+1)
        Hs[1] <- -ev.right$lp + 0.5*sum(v.right^2)
        h <- H/nstep
        
        for(s in 1:nstep){
          vh <- v + 0.5*h*ev$grad
          q <- q + h*vh
          ev <- fun(q)
          n.eval <- n.eval+1
          v <- vh + 0.5*h*ev$grad
          Hs[s+1] <- -ev$lp + 0.5*sum(v^2)
        }
        energy.err <- max(Hs)-min(Hs)
        if(energy.err<energy.tol){
          I.back <- c
          break
        }
      }
    } # reverse integration switch
    
    Ib[i] <- I.back # backward I(z)
    lpb[i] <- lp.disc.exp(C,I=I.back,pI=I0.prob) # backward probability of c
    
    
    
    # prepare for next step
    q.left <- q.right
    v.left <- v.right
    ev.left <- ev.right
    
    # collect stats
    q.min <- pmin(q.min,q.right)
    q.max <- pmax(q.max,q.right)
    
  } # loop over big steps
  
  H1 <- -ev.right$lp + 0.5*sum(v.right^2)
  
  lalph <- H0 - H1 + sum(lpb) - sum(lpf) # accept prob
  
  if(runif(1)< exp(min(0,lalph))){
    q.new <- q.right
    v.new <- v.right
    ev.new <- ev.right
    acc <- 1.0
  } else {
    q.new <- q0
    v.new <- -v0
    ev.new <- ev.init
    acc <- 0.0
  }
  return(list(q.new=q.new,v.new=v.new,lalph=lalph,J=J,
              H=H,Cs=Cs,H0=H0,H1=H1,If=If,Ib=Ib,q0=q0,v0=v0,acc=acc,
              q.min=q.min,q.max=q.max,n.eval=n.eval,
              H.trace=H.trace,
              t.trace=t.trace,
              q.big.step=q.big.step,
              q.intermediate=q.intermediate))
}
