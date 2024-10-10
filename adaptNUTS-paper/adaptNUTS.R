source("singleStepMethods.R")


adaptNUTS <- function(fun,q0,n.iter=2000,n.warmup=1000,
                      H0=0.3, # initial step size
                      energy.tol=0.05, # energy error tolerance
                      I0.target=0.8, # target for P(step-reduction=0)
                      M=11L, # max NUTS iterations
                      maxC=10L,
                      PI=0.9999999,
                      step.size.rand.scale=0.2,
                      within.orbit.randomized=TRUE){
  d <- length(q0)
  q.samples <- matrix(0.0,d,n.iter+1)
  q.samples[,1] <- q0
  orbit.min <- matrix(0.0,d,n.iter)
  orbit.max <- matrix(0.0,d,n.iter)
  
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
  
  build.orbit <- function(Bs,I0,nSubStep,HH,qq0,vv0){
    
    #h <- HH/nSubStep
    
    
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
    
    Hsub <- numeric(2^maxC+1)
    
    
    
    
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
          
          #print(c(I0+a-j,nSubStep))
          
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
          
          #print(c(I0+b+j-1,nSubStep))
          
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
      
      # reject orbit if numerical problems occurred
      if(! all(is.finite(wts)) ||
         ! all(is.finite(qs[,I0 + (at:bt)])) ||
         ! all(is.finite(vs[,I0 + (at:bt)]))){
        print("numerical problems, rejecting")
        orbit.good <- TRUE
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
    
    
    
    # step reduction loop
    II <- maxC
    for(c in 0:maxC){
      nSubStep <- 2L^c
      bo.out <- build.orbit(Bs,I0,nSubStep,HH,qq0,vv0)
      energy.err <- max(Hs[(bo.out$aj:bo.out$bj)+I0]) - min(Hs[(bo.out$aj:bo.out$bj)+I0])
      if(is.finite(energy.err) && energy.err < energy.tol){
        II <- c
        break
      }      
    } # step reduction loop
    
    cc <- sample(c(II,II+1),size=1,replace=TRUE,prob=c(PI,1-PI))
    #print(c(II,cc))
    if(cc>II){
      nSubStep <- 2L^cc
      bo.out <- build.orbit(Bs,I0,nSubStep,HH,qq0,vv0)
    }
    # multinomial weights
    wts <- -Hs[(bo.out$a:bo.out$b)+I0] 
    wts <- exp(wts-max(wts))
    wts <- wts/sum(wts)
    
    # sample new state
    J <- sample(bo.out$a:bo.out$b,size=1,replace=TRUE,prob = wts)
    
    a <- bo.out$a
    b <- bo.out$b
    
    # compute Bstar
    atil <- a
    btil <- b
    Bstar <- Bs
    for(i in log2(b-a+1):1){
      m <- floor(0.5*(atil+btil))
      if(J<=m){
        Bstar[i] <- 0
        btil <- m
      } else {
        Bstar[i] <- 1
        atil <- m+1
      }
    }
    
    nleft <- sum(Bstar*2L^(0:(M-1)))
    I0star <- nleft+1
    
    
    qstar <- qs[,I0+J]
    vstar <- vs[,I0+J]
    alph <- 1.0
    II.back <- -1
    # work out how many reverse orbits we need to try
    if(abs(J)>0.5 && cc>0.5){
      if(cc==II){
        maxTry <- II-1
        II.back <- II
      } else {
        maxTry <- II+1
        II.back <- maxC
      }
      
      # compute reverse orbit
      for(c in 0:maxTry){
        nSubStep <- 2L^c
        bo.out <- build.orbit(Bstar,I0star,nSubStep,HH,qstar,vstar)
        energy.err <- max(Hs[(bo.out$aj:bo.out$bj)+I0star]) - min(Hs[(bo.out$aj:bo.out$bj)+I0star])
        #print(energy.err)
        if(is.finite(energy.err) && energy.err < energy.tol){
          II.back <- c
          break
        }      
      } # step reduction loop
      
      if(II.back != II){
        if(II.back+1==cc){
          alph <- (1-PI)/PI
          
        } else {
          alph <- 0
        }
      }
      if(runif(1)>alph){
        # reject
        qstar <- q.samples[,iter]
      }
    } # only check backward if J>0
    
    # store progress
    q.samples[,iter+1] <- qstar
    diagnostics[iter,1:10] <- c(J,bo.out$last.nuts.iter,II,cc,II.back,alph,n.eval,H,
                               mean(HH[I0+a:(b-1)]*2^(-cc)),
                               sum(HH[I0+a:(b-1)]))
    
    # dual averaging tuning
    if(iter <= n.warmup){
      eta.H <- iter^(-0.85)
      ave.H <- sum(I0.target-as.integer(diagnostics[1:iter,3]==0))/(iter+10)
      step.H <- log(H0) - 0.2*sqrt(iter)*ave.H
      H <- exp(eta.H*step.H + (1.0-eta.H)*log(H))
    }
    
    
  } # main MCMC iteration loop
  colnames(diagnostics) <- c("J","NUTSIter","I","c","Ib","alpha",
                             "neval","H","meanh","orbitLength","d11","d12")
  return(list(samples=q.samples,diagnostics=diagnostics,method=1))
  
} # adaptNUTS function

#ret <- adaptNUTS(lp.std.gauss,rnorm(10),n.iter = 10000,n.warmup=5000,H0=0.5,M=11,step.size.rand.scale=1.5)
