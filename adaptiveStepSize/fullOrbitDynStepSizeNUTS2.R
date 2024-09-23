source("singleStepMethods.R")




fullOrbitDynStepSizeNUTS2 <- function(fun,q0,v0,H=0.5,Cmax=10L,M=11L,energy.tol=0.05,probI=0.99,debug=FALSE){
  
  
  buildOrbit <- function(fun,q0,v0,c,H,M,Bis){
    f0 <- fun(q0) # not counted as eval
    d <- length(q0)
    
    R <- 2L^c
    h <- H/R
    fl <- f0
    fr <- f0
    a <- 0L
    b <- 0L
    qs <- matrix(q0,d,1)
    vs <- matrix(v0,d,1)
    Hs <- -f0$lp + 0.5*sum(v0^2)
    zero.ind <- 1L
    n.eval <- 0L
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
      Bi <- Bis[i] 
      nstep <- 2L^(i-1)
      tt <- (-1L)^Bi*nstep
      at <- a + tt
      bt <- b + tt
      
      if(Bi<0.5){
        # integrate forward in time 
        qq <- qs[,ncol(qs)]
        vv <- vs[,ncol(vs)]
        ff <- fr
        for(j in 1:nstep){
          for(k in 1:R){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval+1
            vv <- vh + 0.5*h*ff$grad
          }
          qs <- cbind(qs,qq)
          vs <- cbind(vs,vv)
          Hs <- c(Hs,-ff$lp + 0.5*sum(vv^2))
        }
        fr <- ff
        I.new <- (b+1):(b+nstep)
      } else {
        # integrate backward in time
        qq <- qs[,1]
        vv <- -vs[,1]
        ff <- fl
        for(j in 1:nstep){
          for(k in 1:R){
            vh <- vv + 0.5*h*ff$grad
            qq <- qq + h*vh
            ff <- fun(qq)
            n.eval <- n.eval+1
            vv <- vh + 0.5*h*ff$grad
          }
          qs <- cbind(qq,qs)
          vs <- cbind(-vv,vs)
          Hs <- c(-ff$lp + 0.5*sum(vv^2),Hs)
          zero.ind <- zero.ind + 1L
          
        }
        I.new <- a - (nstep:1)
        fl <- ff
      }
      
      if(!all(is.finite(Hs))) return(list(orbit.good=FALSE,n.eval=n.eval))
      
      Uturn <- indicator.U.turn(a,b)
      subUturn <- indicator.sub.U.turn(at,bt)
      
      if(Uturn || subUturn){
        last.iter <- i
        break
      }
      
      # algorithm 3 of Durmus et alt
      wts <- exp(-Hs-max(-Hs))
      pib <- wts/sum(wts[zero.ind+I.new])
      
      if(all(is.finite(pib))){
        j.prop <- sample(x=I.new,size=1,replace = TRUE,prob=pib[zero.ind+I.new])
        
        alph.t <- min(1.0,1.0/sum(pib[zero.ind+(a:b)]))
        if(runif(1) < alph.t) J <- j.prop
      }
      a <- min(a,at)
      b <- max(b,bt)
    }
    
    wts <- exp(-Hs[zero.ind+a:b] - (max(-Hs[zero.ind+a:b])))
    wts <- wts/sum(wts)
    return(list(orbit.good=TRUE,a=a,b=b,qs=qs,vs=vs,Hs=Hs,J=J,
                zero.ind=zero.ind,wts=wts,last.iter=last.iter,
                n.eval=n.eval))
    
  }
  
  
  n.eval <- 0L
  Bis <- sample(x=c(0L,1L),size=M,replace=TRUE,prob=c(0.5,0.5))
  II <- Cmax
  for(c in 0:Cmax){ # step reduction loop
    
    orb <- buildOrbit(fun,q0,v0,c,H,M,Bis)
    n.eval <- n.eval + orb$n.eval
    if(orb$orbit.good){
      energy.err <- max(orb$Hs) - min(orb$Hs)
      if(debug) print(paste0("energy err : ",energy.err))
      
      if(is.finite(energy.err)){
        if(energy.err < energy.tol){
          II <- c
          break
        }
      }
    }
  } # step reduction 
  
  # sample trajectory used for proposal
  cc <- rng.disc.exp(1,II,probI) 
  lpf <- lp.disc.exp(cc,II,probI)
  if(cc>II){
    orb <- buildOrbit(fun,q0,v0,cc,H,M,Bis)
    n.eval <- n.eval + orb$n.eval
  }
  
  # get proposal
  J <- orb$J
  q.prop <- orb$qs[,orb$zero.ind+J]
  v.prop <- orb$vs[,orb$zero.ind+J]
  
  # find backward B
  at <- orb$a
  bt <- orb$b
  Bis.rev <- Bis
  ndoub <- log2(bt-at+1)
  
  for(i in ndoub:1){
    m <- floor((at+bt)/2)
    if(J<=m){
      Bis.rev[i] <- 0
      bt <- m
    } else {
      Bis.rev[i] <- 1
      at <- m+1
    }
  }
  
  
  check.max <- -1L
  II.rev <- 0L
  if((II==cc) && II>0){
    check.max <- II-1L
    II.rev <- II
  } else if(II>0){
    check.max <- Cmax
    II.rev <- Cmax
  }
  
  if(check.max > -0.5){
    for(c in 0:check.max){
      orb.rev <- buildOrbit(fun,q.prop,v.prop,c,H,M,Bis.rev)
      n.eval <- n.eval + orb.rev$n.eval
      if(orb.rev$orbit.good){
        energy.err <- max(orb.rev$Hs) - min(orb.rev$Hs)
        if(debug) print(paste0("energy err, reverse : ",energy.err))
        if(is.finite(energy.err)){
          if(energy.err < energy.tol){
            II.rev <- c
            break
          }
        }
      }
    }
  }
  
  lpr <- lp.disc.exp(cc,II.rev,probI)
  alph <- exp(min(0,lpr-lpf))
  if(runif(1) > alph){
    if(debug) print("reject")
    q.prop <- q0
    v.prop <- -v0
  }
  
  
  
  
  
  
  
  
  
  
  if(debug){
    print(paste0("n.eval : ",n.eval))
    print(paste0("II : ",II))
    print(paste0("II.rev : ",II.rev))
    print(paste0("cc : ",cc))
    print(paste0("last iter: ",orb$last.iter))
    print(paste0("J : ",orb$J))
    print(paste0("wts ess: ",1.0/(length(orb$wts)*sum(orb$wts^2))))
    par(mfrow=c(1,2))
    plot(orb$qs[1,],orb$qs[2,],type="l")
    points(orb$qs[1,],orb$qs[2,],col="black")
    points(orb$qs[1,orb$zero.ind+orb$a:orb$b],orb$qs[2,orb$zero.ind+orb$a:orb$b],col="red")
    points(orb$qs[1,orb$zero.ind],orb$qs[2,orb$zero.ind],col="green")
    
    
    plot((1:length(orb$Hs))-orb$zero.ind,orb$Hs,col="black")
    points(orb$a:orb$b,orb$Hs[orb$zero.ind+orb$a:orb$b],col="red")
    points(0,orb$Hs[orb$zero.ind],col="green")
    
  }
  return(list(q.new=q.prop,v.new=v.prop,
              J=orb$J,
              last.iter=orb$last.iter,
              II=II,
              cc=cc,
              n.eval=n.eval,
              II.rev=II.rev,
              wts.ess=1.0/(length(orb$wts)*sum(orb$wts^2))))
  
}

if(1==0){
  n.iter <- 50000
  fun <- lp.funnel
  qs <- matrix(0.0,2,n.iter)
  q <- rnorm(2)
  wts.ess <- numeric(n.iter)
  last.iter <- numeric(n.iter)
  
  for(iter in 1:n.iter){
    out <- fullOrbitDynStepSizeNUTS2(fun,q,rnorm(2),H=0.3)
    q <- out$q.new
    qs[,iter] <- q
    wts.ess[iter] <- out$wts.ess
    last.iter[iter] <- out$last.iter
  }
}
