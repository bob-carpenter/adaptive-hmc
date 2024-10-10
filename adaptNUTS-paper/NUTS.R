

source("singleStepMethods.R")


NUTS <- function(fun,q0,n.iter=2000,
                 H0=0.3, # step size
                 M=11L, # max number of doublings
                 step.size.rand.scale=0.2, 
                 within.orbit.randomized=TRUE){
  d <- length(q0)
  # storage for mcmc samples
  q.samples <- matrix(0.0,d,n.iter+1)
  q.samples[,1] <- q0
  # storage for diagnostics info
  diagnostics <- matrix(0.0,n.iter,12)
  
  H <- H0
  
  # storage for orbits
  qs <- matrix(0.0,d,2^M)
  vs <- matrix(0.0,d,2^M)
  Hs <- numeric(2^M)
  Is <- numeric(2^M)
  
  # U-turn indicators
  indicator.U.turn <- function(l,r,zero.ind){
    if(l==r) return(FALSE)
    ql <- qs[,l+zero.ind]
    vl <- vs[,l+zero.ind]
    qr <- qs[,r+zero.ind]
    vr <- vs[,r+zero.ind]
    cf <- sum(vr*(qr-ql))
    cb <- sum(vl*(qr-ql))
    return(min(cf,cb) < -1.0e-14)
  }
  
  
  indicator.sub.U.turn <- function(l,r,zero.ind){
    if(l==r){ 
      return(FALSE)
    } else{
      m <- floor((l+r)/2)
      return(indicator.sub.U.turn(l,m,zero.ind) || 
               indicator.sub.U.turn(m+1L,r,zero.ind) ||
               indicator.U.turn(l,r,zero.ind))
    }
  }
  
  # see paper for details
  build.orbit <- function(Bs,I0,nSubStep,HH,qq0,vv0){
    
    # number of NUTS iterations
    last.nuts.iter <- M
    orbit.good <- TRUE
    
    
    qs[,I0] <<- qq0
    vs[,I0] <<- vv0 
    f0 <- fun(qs[,I0])
    n.eval <<- n.eval+1L
    Hs[I0] <<- -f0$lp + 0.5*sum(vs[,I0]^2)
    
    
    a <- 0L
    b <- 0L
    J <- 0L
    fl <- f0
    fr <- f0
    
    n.I0 <- 0L
    
    
    
    
    for(i in 1:M){ # NUTS iteration loop
      
      nstep <- 2L^(i-1)
      tt <- (-1L)^Bs[i]*nstep
      at <- a + tt
      bt <- b + tt
      
      if(Bs[i]>0.5){ # integrate backward
        qq <- qs[,I0+a] # last integrated so far
        vv <- -vs[,I0+a]
        ff <- fl
        for(j in 1:nstep){
          
          
          h <- HH[I0+a-j]/nSubStep
          
          for(k in 1:nSubStep){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <<- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
          }
          Hc <- -ff$lp + 0.5*sum(vv^2)
          qs[,I0+a-j] <<- qq
          vs[,I0+a-j] <<- -vv
          Hs[I0+a-j] <<- Hc
          
        } # integration steps within suborbit
        
        fl <- ff
        I.new <- a - (nstep:1)
        
      } else { # integrate forward
        qq <- qs[,I0+b] # last integrated so far
        vv <- vs[,I0+b]
        ff <- fr
        
        for(j in 1:nstep){
          
          h <- HH[I0+b+j-1]/nSubStep
          for(k in 1:nSubStep){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <<- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
          }
          
          
          Hc <- -ff$lp + 0.5*sum(vv^2)
          qs[,I0+b+j] <<- qq
          vs[,I0+b+j] <<- vv
          Hs[I0+b+j] <<- Hc
          
        } # integration steps within suborbit
        fr <- ff
        I.new <-  (b+1):(b+nstep)
        
      } # forward/backward if
      
      
      Uturn <- indicator.U.turn(a,b,I0)
      subUturn <- indicator.sub.U.turn(at,bt,I0)
      
      aj <- min(a,at)
      bj <- max(b,bt)
      
      if(Uturn || subUturn){
        last.nuts.iter <- i
        break
      }
      
      
      wts <- -Hs[(aj:bj)+I0]
      
      
      if(! all(is.finite(wts)) ||
         ! all(is.finite(qs[,I0 + (at:bt)])) ||
         ! all(is.finite(vs[,I0 + (at:bt)]))){
        print("numerical problems, rejecting")
        orbit.good <- FALSE
        break
      }
      
      a <- min(a,at)
      b <- max(b,bt)
      
      
    } # NUTS iterations
    
    return(list(a=a,b=b,aj=aj,bj=bj,last.nuts.iter=last.nuts.iter))
  }
  
  
  
  for(iter in 1:n.iter){ # main MCMC iterations loop
    if(iter %% 1000 == 0) message(paste0("iteration # ",iter))
    
    # sample directions
    Bs <- sample(c(0,1),size=M,replace=TRUE,prob=c(0.5,0.5))
    # sample step sizes
    if(within.orbit.randomized){
      HH <- runif(2^M,min=H*(1-step.size.rand.scale),max=H*(1+step.size.rand.scale))
    } else {
      HH <- rep(runif(1,min=H*(1-step.size.rand.scale),max=H*(1+step.size.rand.scale)),2^M)
    }
    # alread know maximum extent of orbit
    nleft <- sum(Bs*2L^(0:(M-1)))
    nright <- sum((1-Bs)*2L^(0:(M-1)))
    
    I0 <- nleft+1L
    
    # initial state
    qq0 <- q.samples[,iter]
    vv0 <- rnorm(d)
    n.eval <- 0L
    
    # build NUTS orbit
    bo.out <- build.orbit(Bs,I0,1L,HH,qq0,vv0)
    
    
    # weights for multinomial sampling
    wts <- -Hs[(bo.out$a:bo.out$b)+I0] 
    wts <- exp(wts-max(wts))
    wts <- wts/sum(wts)
    
    # sample new state
    if(all(is.finite(wts))){
      J <- sample(bo.out$a:bo.out$b,size=1,replace=TRUE,prob = wts)
    } else {
      print("numerical problems")
      J <- 0
    }
    
    qstar <- qs[,I0+J]
    
    
    # store progress
    q.samples[,iter+1] <- qstar
    diagnostics[iter,1:5] <- c(J,bo.out$last.nuts.iter,n.eval,H,
                               sum(HH[I0+(bo.out$a:(bo.out$b-1))]))
    
    
  } # main MCMC iteration loop
  colnames(diagnostics) <- c("J","NUTSIter",
                             "neval","H","orbitLength","d6","d7","d8","d9","d10","d11","d12")
  return(list(samples=q.samples,diagnostics=diagnostics,method=0))
  
} # adaptNUTS function

#ret <- NUTS(lp.std.gauss,rnorm(2),n.iter = 10000,H0=0.1,M=11)
