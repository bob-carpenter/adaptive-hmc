source("singleStepMethods.R")




dynStepSizeNUTS2 <- function(fun,q0,v0,H=0.5,R=4L,M=11L,
                             energy.tol=0.05,
                             probI=0.99,
                             maxC=10L,
                             debug=FALSE){
  
  
  
  f0 <- fun(q0) # not counted /inherited from previous iteration
  d <- length(q0)
  n.eval <- 0L
  fl <- f0
  fr <- f0
  a <- 0L
  b <- 0L
  qs <- matrix(q0,d,1)
  vs <- matrix(v0,d,1)
  Hs <- -f0$lp + 0.5*sum(v0^2)
  zero.ind <- 1L
  
  cc.f <- numeric()
  cc.b <- numeric()
  II.f <- numeric()
  II.b <- numeric()
  
  
  indicator.U.turn <- function(l,r){
    if(l==r) return(FALSE)
    ql <- qs[,l+zero.ind]
    vl <- vs[,l+zero.ind]
    qr <- qs[,r+zero.ind]
    vr <- vs[,r+zero.ind]
    cf <- sum(vr*(qr-ql))
    cb <- sum(vl*(qr-ql))
    return(min(cf,cb) < -1.0e-14)
  }
  indicator.sub.U.turn <- function(l,r){
    if(l==r){ 
      return(FALSE)
    } else{
      m <- floor((l+r)/2)
      return(indicator.sub.U.turn(l,m) || 
               indicator.sub.U.turn(m+1L,r) ||
               indicator.U.turn(l,r))
    }
  }
  
  last.iter <- M
  J <- 0
  
  for(i in 1:M){
    Bi <- sample(x=c(0L,1L),size=1,replace=TRUE,prob=c(0.5,0.5))
    nstep <- 2L^(i-1)
    tt <- (-1L)^Bi*nstep
    at <- a + tt
    bt <- b + tt
    
    if(Bi<0.5){
      # integrate forward in time 
      qc <- qs[,ncol(qs)]
      vc <- vs[,ncol(vs)]
      fc <- fr
      Hc <- Hs[length(Hs)]
      for(j in 1:nstep){
        II <- maxC
        for(c in 0:maxC){
          nsub <- R*(2L)^c
          h <- H/nsub
          Hsub <- numeric(nsub+1) 
          qq <- qc
          vv <- vc
          ff <- fc
          Hsub[1] <- Hc
          for(k in 1:nsub){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
            Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
          }
          energy.err <- max(Hsub)-min(Hsub)
          if(energy.err < energy.tol){
            II <- c
            break
          }
        } # precision iteration loop
        
        # simulate precsion parameter
        cc <- rng.disc.exp(1,I=II,pI=probI)
        
        if(cc>II){ # need to simulate with even higher precision
          nsub <- R*(2L)^cc
          h <- H/nsub
          qq <- qc
          vv <- vc
          ff <- fc
          for(k in 1:nsub){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
          }
        }
        cc.f <- c(cc.f,cc)
        II.f <- c(II.f,II)
        
        qc <- qq
        vc <- vv
        Hc <- -ff$lp + 0.5*sum(vv^2)
        fc <- ff
        qs <- cbind(qs,qq)
        vs <- cbind(vs,vv)
        Hs <- c(Hs,Hc)
      }
      fr <- ff
      I.new <- (b+1):(b+nstep)
    } else {
      # integrate backward in time
      qc <- qs[,1]
      vc <- -vs[,1]
      fc <- fl
      Hc <- Hs[1]
      for(j in 1:nstep){
        II <- maxC
        for(c in 0:maxC){
          nsub <- R*(2L)^c
          h <- H/nsub
          Hsub <- numeric(nsub+1) 
          qq <- qc
          vv <- vc
          ff <- fc
          Hsub[1] <- Hc
          for(k in 1:nsub){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
            Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
          }
          energy.err <- max(Hsub)-min(Hsub)
          if(energy.err < energy.tol){
            II <- c
            break
          }
        } # precision iteration loop
        
        # simulate precsion parameter
        cc <- rng.disc.exp(1,I=II,pI=probI)
        
        if(cc>II){ # need to simulate with even higher precision
          nsub <- R*(2L)^cc
          h <- H/nsub
          qq <- qc
          vv <- vc
          ff <- fc
          for(k in 1:nsub){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
          }
        }
        cc.b <- c(cc.b,cc)
        II.b <- c(II.b,II)
        
        qc <- qq
        vc <- vv
        Hc <- -ff$lp + 0.5*sum(vv^2)
        fc <- ff
        qs <- cbind(qq,qs)
        vs <- cbind(-vv,vs)
        Hs <- c(Hc,Hs)
        zero.ind <- zero.ind + 1L
      }
      I.new <- a - (nstep:1)
      fl <- ff
    }
    
    
    Uturn <- indicator.U.turn(a,b)
    subUturn <- indicator.sub.U.turn(at,bt)
    
    if(Uturn || subUturn){
      last.iter <- i
      break
    }
    
    # algorithm 3 of Durmus et alt
    wts <- exp(-Hs-max(-Hs))
    pib <- wts/sum(wts[zero.ind+I.new])
    
    j.prop <- sample(x=I.new,size=1,replace = TRUE,prob=pib[zero.ind+I.new])
    
    alph.t <- min(1.0,1.0/sum(pib[zero.ind+(a:b)]))
    if(runif(1) < alph.t) J <- j.prop
    
    a <- min(a,at)
    b <- max(b,bt)
  }
  
  # compute adaptive step size accept probability
  denom <- 0.0
  numer <- 0.0
  II.rev <- numeric(abs(J)) # only for diagnostics
  if(J>0.5){
    denom <- sum(lp.disc.exp(cc.f[1:J],II.f[1:J],probI))
    numer <- 0.0
    for(s in 1:J){ # loop over all transitions between current state and proposal
      # determine how 
      maxC.back <- -1L
      II.back <- -1L
      if(cc.f[s]==0 && II.f[s]==0){
        # due to reversibility of the error estimate, we do not need to check further
        II.back <- 0L
      } else if((cc.f[s] == II.f[s]) && II.f[s] < maxC ) {
        # here we know I(R \circ z[s]) <= I(z[s-1])
        maxC.back <- II.f[s] - 1L
        II.back <- II.f[s]
      } else {
        maxC.back <- maxC
        II.back <- maxC
      }
      
      if(maxC.back > -0.5){
        fc <- fun(qs[,zero.ind+s]) # do not count this one as it has already been calculated
        for(c in 0:maxC.back){
          nsub <- R*(2L)^c
          h <- H/nsub
          qq <- qs[,zero.ind+s]
          vv <- -vs[,zero.ind+s] # here we are integrating backward in time
          ff <- fc
          Hsub <- numeric(nsub+1)
          Hsub[1] <- Hs[zero.ind + s]
          for(k in 1:nsub){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
            Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
          }
          energy.err <- max(Hsub)-min(Hsub)
          if(energy.err < energy.tol){
            II.back <- c
            break
          }
        } # precision loop
        
      } # calculation of I.back non.trivial
      numer <- numer + lp.disc.exp(cc.f[s],II.back,probI)
      II.rev[s] <- II.back
    } # loop over steps between current state and proposal
    
    
    
  } else if(J < -0.5){
    denom <- sum(lp.disc.exp(cc.b[1:(-J)],II.b[1:(-J)],probI))
    numer <- 0.0
    for(s in 1:(-J)){ # loop over all transitions between current state and proposal
      # determine how 
      maxC.back <- -1L
      II.back <- -1L
      if(cc.b[s]==0 && II.b[s]==0){
        # due to reversibility of the error estimate, we do not need to check further
        II.back <- 0L
      } else if((cc.b[s] == II.b[s]) && II.b[s] < maxC ) {
        # here we know I(R \circ z[s]) <= I(z[s-1])
        maxC.back <- II.b[s] - 1L
        II.back <- II.b[s]
      } else {
        maxC.back <- maxC
        II.back <- maxC
      }
      
      if(maxC.back > -0.5){
        fc <- fun(qs[,zero.ind-s]) # do not count this one as it has already been calculated
        for(c in 0:maxC.back){
          nsub <- R*(2L)^c
          h <- H/nsub
          qq <- qs[,zero.ind-s]
          vv <- vs[,zero.ind-s] # here we are integrating forward in time
          ff <- fc
          Hsub <- numeric(nsub+1)
          Hsub[1] <- Hs[zero.ind - s]
          for(k in 1:nsub){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval + 1L
            vv <- vh + 0.5*h*ff$grad
            Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
          }
          energy.err <- max(Hsub)-min(Hsub)
          if(energy.err < energy.tol){
            II.back <- c
            break
          }
        } # precision loop
        
      } # calculation of I.back non.trivial
      numer <- numer + lp.disc.exp(cc.b[s],II.back,probI)
      II.rev[s] <- II.back
    
    } # loop over steps between current state and proposal
  } # forward/backward if
  if(debug) print(paste0("J before accept/reject: ",J)) 
    
    
  alph <- exp(min(0.0,numer-denom))
  if(runif(1)>alph){
    #reject due to adaptive step sizes
    J <- 0L
  } 
  

  
  
  wts <- exp(-Hs[zero.ind+a:b] - (max(-Hs[zero.ind+a:b])))
  wts <- wts/sum(wts)
  
  
  if(debug){
    print(paste0("last iter: ",last.iter))
    print(paste0("J : ",J))
    print(paste0("wts ess: ",1.0/(length(wts)*sum(wts^2))))
    print("I/C forw")
    print(rbind(II.f,cc.f))
    print("I/C back")
    print(rbind(II.b,cc.b))
    
    print("I.rev")
    print(II.rev)
    
    print(paste0("numer = ",numer))
    print(paste0("denom = ",denom))
   
    par(mfrow=c(1,2))
    plot(qs[1,],qs[2,],type="l")
    points(qs[1,],qs[2,],col="black")
    points(qs[1,zero.ind+a:b],qs[2,zero.ind+a:b],col="red")
    points(qs[1,zero.ind],qs[2,zero.ind],col="green")
    
    plot((1:length(Hs))-zero.ind,Hs,col="black")
    points(a:b,Hs[zero.ind+a:b],col="red")
    points(0,Hs[zero.ind],col="green")
    
  }
  
  
  
  return(list(q.new=qs[,zero.ind+J],v.new=vs[,zero.ind+J],
              J=J,
              maxc=max(c(cc.f,cc.b)),
              minc=min(c(cc.f,cc.b)),
              last.iter=last.iter,
              alph=alph,
              n.eval=n.eval,
              wts.ess=1.0/(length(wts)*sum(wts^2)),
              minq=apply(qs,1,min),
              maxq=apply(qs,1,max)))
  
}

if(1==0){
  n.iter <- 50000
  fun <- lp.funnel
  qs <- matrix(0.0,2,n.iter)
  q <- rnorm(2)
  wts.ess <- numeric(n.iter)
  last.iter <- numeric(n.iter)
  min.c <- numeric(n.iter)
  max.c <- numeric(n.iter)
  
  for(iter in 1:n.iter){
    out <- dynStepSizeNUTS2(fun,q,rnorm(2),R=4L,H=0.2)
    q <- out$q.new
    qs[,iter] <- q
    wts.ess[iter] <- out$wts.ess
    last.iter[iter] <- out$last.iter
    min.c[iter] <- out$minc
    max.c[iter] <- out$maxc
  }
}
