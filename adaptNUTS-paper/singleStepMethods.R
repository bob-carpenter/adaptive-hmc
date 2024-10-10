#rm(list=ls())

ESS <- function(x){#return(round(coda::effectiveSize(coda::mcmc(x))))
  sims <- array(dim=c(nrow(x),1,ncol(x)))
  for(j in 1:ncol(x)) sims[,1,j] <- x[,j]
  return(rstan::monitor(sims,print=FALSE)$n_eff)
}

lp.std.gauss <- function(q){
  return(list(lp=-0.5*sum(q^2),grad=-q))
}


lp.corr.gauss <- function(q){
  corr <- 0.95
  f <- 1.0/(1.0-corr^2)
  lp <- -0.5*f*(q[1]^2+q[2]^2-2.0*corr*q[1]*q[2])
  g1 <- -f*(q[1]-corr*q[2])
  g2 <- -f*(q[2]-corr*q[1])
  return(list(lp=lp,grad=c(g1,g2)))
}

funnel.fac <- 3.0
lp.funnel <- function(q){
  return(list(lp=-0.5*q[1]^2 + dnorm(q[2],mean=0.0,sd=exp(0.5*funnel.fac*q[1]),log=TRUE),
              grad=c(-q[1]+0.5*q[2]^2*exp(-funnel.fac*q[1])*funnel.fac-0.5*funnel.fac,
                     -q[2]*exp(-funnel.fac*q[1]))))
}

lp.funnel10 <- function(q){
  return(list(lp=dnorm(q[1],mean=0.0,sd=3.0,log=TRUE) + 
                sum(dnorm(q[2:11],mean=0.0,sd=exp(0.5*q[1]),log=TRUE)),
              grad=c(-5.0-q[1]/9.0 + 0.5*exp(-q[1])*sum(q[2:11]^2),
                     -q[2:11]*exp(-q[1]))
              ))
}

lp.funnel10.std <- function(q){
  return(list(lp=dnorm(q[1],mean=0.0,sd=1.0,log=TRUE) + 
                sum(dnorm(q[2:11],mean=0.0,sd=exp(1.5*q[1]),log=TRUE)),
              grad=c(-15.0 - q[1] + 1.5*exp(-3*q[1])*sum(q[2:11]^2),
                     -q[2:11]*exp(-3*q[1]))
  ))
}



smile.sd <- 1.0
lp.smile <- function(q){
  return(list(lp=-0.5*q[1]^2 + dnorm(q[2],mean=q[1]^2,sd=smile.sd,log=TRUE),
              grad=c(-q[1]+2*q[1]*(q[2]-q[1]^2)/smile.sd^2,
                     -(q[2]-q[1]^2)/smile.sd^2)))
}

lp.dwell <- function(q){
  return(list(lp=-(1-q[1]^2)^2 - 0.5*(q[2]-q[1])^2,
              grad=c(4*(1-q[1]^2)*q[1]-q[1]+q[2],
                     q[1]-q[2]
              )))
}

lp.dwell.1d <- function(q){
  return(list(lp=-(1-q[1]^2)^2,
              grad=c(4*(1-q[1]^2)*q[1])))
}

lp.t4.1d <- function(q){
  return(list(lp=-2.5*log(1+0.25*q^2),
              grad=c(-(5/4)*q/(1+0.25*q^2))))
}

lp.mod.funnel <- function(q){
  x <- q[1]
  y <- q[2]
  t1 <- exp(-3 * x)
  t2 <- 1 + t1
  t3 <- 0.1e1 / t2
  t4 <- y ^ 2
  t5 <- -0.1e1 / 0.2e1
  
  return(list(lp=t5 * (t2 * t4 + log(t3) + x ^ 2),
              grad=c(0.3e1 / 0.2e1 * t1 * (t4 - t3) - x,-y * t2)))
}

leap.frog.energy <- function(fun,q0,v0,h,old.eval=NULL,tol=1.0e-2){
  if(is.null(old.eval)){
    ev0 <- fun(q0)
    n.eval <- 2L
  } else {
    ev0 <- old.eval
    n.eval <- 1L
  }
  
  vh <- v0 + 0.5*h*ev0$grad
  q <- q0 + h*vh
  ev1 <- fun(q)
  v <- vh + 0.5*h*ev1$grad
  
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  H1 <- -ev1$lp + 0.5*sum(v^2)
  return(list(q1=q,v1=v,q0=q0,v0=v0,err=abs(H0-H1)/tol,ev1=ev1,ev0=ev0,n.eval=n.eval,H0=H0,H1=H1,order=2))
}


leap.frog.euler <- function(fun,q0,v0,h,old.eval=NULL,tol=1.0e-2){
  if(is.null(old.eval)){
    ev0 <- fun(q0)
    n.eval <- 2L
  } else {
    ev0 <- old.eval
    n.eval <- 1L
  }
  
  vh <- v0 + 0.5*h*ev0$grad
  q <- q0 + h*vh
  ev1 <- fun(q)
  v <- vh + 0.5*h*ev1$grad
  
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  H1 <- -ev1$lp + 0.5*sum(v^2)
  
  q.f <- q0 + h*v0
  v.f <- v0 + h*ev0$grad
  #print(c(q,q.f))
  #print(c(v,v.f))
  
  err.f.q <- max(abs(q-q.f)/(tol*(1+abs(q))))
  err.f.v <- max(abs(v-v.f)/(tol*(1+abs(v))))
  
  q.b <- q - h*v
  v.b <- v - h*ev1$grad
  #print(c(q0,q.b))
  #print(c(v0,v.b))
  
  err.b.q <- max(abs(q0-q.b)/(tol*(1+abs(q0))))
  err.b.v <- max(abs(v0-v.b)/(tol*(1+abs(v0))))
  #print(c(err.f.q,err.f.v,err.b.q,err.b.v))
  err <- max(c(err.f.q,err.f.v,err.b.q,err.b.v))
  
  return(list(q1=q,v1=v,q0=q0,v0=v0,err=err,ev1=ev1,ev0=ev0,n.eval=n.eval,H0=H0,H1=H1,order=1))
}

leap.frog.RKN3 <- function(fun,q0,v0,h,old.eval=NULL,tol=1.0e-2){
  absTol <- tol
  relTol <- 0.0 #tol
  if(is.null(old.eval)){
    ev0 <- fun(q0)
    n.eval <- 3L
  } else {
    ev0 <- old.eval
    n.eval <- 2L
  }
  vh <- v0 + 0.5*h*ev0$grad
  q <- q0 + h*vh
  ev <- fun(q)
  v <- vh + 0.5*h*ev$grad
  H0 <- -ev0$lp + 0.5*sum(v0^2) 
  H1 <- -ev$lp + 0.5*sum(v^2)
  
  ev.mid <- fun(0.5*(q0+q)+(h/8)*(v0-v))
  mid.grad <- ev.mid$grad
  
  q.f <- q0 + h*v0 + h^2*((1/6)*ev0$grad + (1/3)*mid.grad)
  err.q.f <- max(abs(q.f-q)/(absTol+relTol*abs(q.f))) 
  v.f <- v0 + (h/6)*(ev0$grad + ev$grad + 4.0*mid.grad)
  err.v.f <- max(abs(v.f-v)/(absTol+relTol*abs(v.f)))
  
  q.b <- q - h*v + h^2*((1/6)*ev$grad + (1/3)*mid.grad)
  err.q.b <- max(abs(q.b-q0)/(absTol+relTol*abs(q.b))) 
  v.b <- -(-v + (h/6)*(ev$grad + ev0$grad + 4.0*mid.grad))
  err.v.b <- max(abs(v.b-v0)/(absTol+relTol*abs(v.b)))
  
  err <- max(c(err.q.f,err.q.b,err.v.f,err.v.b))
  
  return(list(q1=q,v1=v,q0=q0,v0=v0,h=h,ev1=ev,ev0=ev0,H0=H0,H1=H1,
              err=err,n.eval=n.eval,order=2))
}



yoshida.energy <- function(fun,q0,v0,h,old.eval=NULL,tol=1.0e-2){
  d1 <- 0.67560359597982881702
  d2 <- -0.17560359597982881702
  c1 <-  1.3512071919596576341
  c2 <- -1.7024143839193152681
  if(is.null(old.eval)){
    ev0 <- fun(q0)
    n.eval <- 4L
  } else {
    ev0 <- old.eval
    n.eval <- 3L
  }
  
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  
  v1 <- v0 + d1*h*ev0$grad
  q1 <- q0 + c1*h*v1
  ev1 <- fun(q1)
  v2 <- v1 + d2*h*ev1$grad
  q2 <- q1 + c2*h*v2
  ev2 <- fun(q2)
  v3 <- v2 + d2*h*ev2$grad
  q3 <- q2 + c1*h*v3
  ev3 <- fun(q3)
  v4 <- v3 + d1*h*ev3$grad
  
  H1 <- -ev3$lp + 0.5*sum(v4^2)
  return(list(q1=q3,v1=v4,q0=q0,v0=v0,err=abs(H0-H1)/tol,
              ev1=ev3,ev0=ev0,n.eval=n.eval,H0=H0,H1=H1,order=4))
}


yoshida.RKN5 <- function(fun,q0,v0,h,old.eval=NULL,tol=1.0e-2){
  d1 <- 0.67560359597982881702
  d2 <- -0.17560359597982881702
  c1 <-  1.3512071919596576341
  c2 <- -1.7024143839193152681
  if(is.null(old.eval)){
    ev0 <- fun(q0)
    n.eval <- 6L
  } else {
    ev0 <- old.eval
    n.eval <- 5L
  }
  
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  
  v1 <- v0 + d1*h*ev0$grad
  q1 <- q0 + c1*h*v1
  ev1 <- fun(q1)
  v2 <- v1 + d2*h*ev1$grad
  q2 <- q1 + c2*h*v2
  ev2 <- fun(q2)
  v3 <- v2 + d2*h*ev2$grad
  q3 <- q2 + c1*h*v3
  ev3 <- fun(q3)
  v4 <- v3 + d1*h*ev3$grad
  
  H1 <- -ev3$lp + 0.5*sum(v4^2)
  
  r1 <- 0.31397051103744487218
  aa40 <- 0.027770603407249854376
  aa41 <- -0.0031262329569712812993
  a0 <- 0.29832595886067916603
  a1 <- -0.052528319439979380288
  a2 <- -0.059198033714587409949
  a3 <- 0.17266612359248679131
  a4 <- 0.14073427070140083290
  # a5 <- 0
  b0 <- 0.12565983526819237485
  b1 <- -0.0039177971812320731387
  b2 <- -0.0039177971812320731704
  b3 <- 0.12565983526819237481
  b4 <- 0.37825796191303969836
  b5 <- 0.37825796191303969829
  
  
  fsum <- (aa40*ev0$grad + aa41*ev1$grad + aa41*ev2$grad + aa40*ev3$grad)
  q4 <- q0 + r1*h*v0 + h^2*fsum
  ev4 <- fun(q4)
  q5 <- q3 - r1*h*v4 + h^2*fsum
  ev5 <- fun(q5)
  
  qh.f <- q0 + h*v0 + h^2*(a0*ev0$grad + a1*ev1$grad + a2*ev2$grad + a3*ev3$grad + a4*ev4$grad)
  vh.f <- v0 + h*(b0*ev0$grad + b1*ev1$grad + b2*ev2$grad + b3*ev3$grad + b4*ev4$grad + b5*ev5$grad)
  
  qh.b <- q3 - h*v4 + h^2*(a0*ev3$grad + a1*ev2$grad + a2*ev1$grad + a3*ev0$grad + a4*ev5$grad)
  vh.b <- -v4 + h*(b0*ev3$grad + b1*ev2$grad + b2*ev1$grad + b3*ev0$grad + b4*ev5$grad + b5*ev4$grad)
  
  #print(-vh.b-v0)
  #print(v0)
  
  eq.f <- max(abs(qh.f-q3))/tol #/(tol*(1+abs(q3))))
  ev.f <- max(abs(vh.f-v4))/tol #/(tol*(1+abs(v4))))
  eq.b <- max(abs(qh.b-q0))/tol #/(tol*(1+abs(q0))))
  ev.b <- max(abs(-vh.b-v0))/tol #/(tol*(1+abs(v0))))
  err <- max(c(eq.f,ev.f,eq.b,ev.b))
  
  return(list(q1=q3,v1=v4,q0=q0,v0=v0,err=err,
              ev1=ev3,ev0=ev0,n.eval=n.eval,H0=H0,H1=H1,order=4,
              qh.f = qh.f,
              vh.f = vh.f,
              qh.b = qh.b,
              vh.b = vh.b))
}




fast.adapt.step.lam0 <- uniroot(f=function(l){return(exp(-l)*(l+1)-0.99)},interval = c(0,10))$root

fast.adapt.step <- function(fun,igr,q0,v0,h.ref=0.5,tol=1.0e-1,mu=0.99,sfac=1.5,old.ev0=NULL,
                            lambda.pri=0 #not used
                            ){
  if(is.null(old.ev0)){
    ev0 <- fun(q0)
    n.eval <- 1L
  } else {
    ev0 <- old.ev0
    n.eval <- 0L
  }
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  
  # do a single integration step
  first.step <- igr(fun,q0,v0,h=h.ref,old.eval = ev0,tol=tol)
  M.star <- (first.step$err)^(1.0/first.step$order)
  n.eval <- n.eval + first.step$n.eval
  
  if(is.finite(M.star)){
  if(M.star<1.0){
    lambda <- fast.adapt.step.lam0
    last.N.f <- 0L
  } else {
    last.N.f <- round(log(M.star))
    lambda <- sfac*M.star + lamW::lambertW0(-exp(-sfac*M.star))
  }
  } else {
    last.N.f <- log(128)
    lambda <- 128L
  }
  

  
  
  M <- max(1,rpois(1,lambda=lambda))
  #print(paste0("M : ",M))
  
  if(M>1){
    q <- q0
    v <- v0
    h <- h.ref/M
    ev <- ev0
    for(i in 1:M){
      vh <- v + 0.5*h*ev$grad
      q <- q + h*vh
      ev <- fun(q)
      v <- vh + 0.5*h*ev$grad
    }
    n.eval <- n.eval + M
    q.prop <- q
    v.prop <- v
    H1 <- -ev$lp + 0.5*sum(v^2)
    
    first.step.back <- igr(fun,q,-v,h=h.ref,old.eval = ev0,tol=tol)
    
    M.star.back <- (first.step.back$err)^(1.0/first.step.back$order)
    n.eval <- n.eval + first.step$n.eval
    
    
    
    if(is.finite(M.star.back)){
    if(M.star.back<1.0){
      lambda.back <- fast.adapt.step.lam0
    } else {
      lambda.back <- sfac*M.star.back + lamW::lambertW0(-exp(-sfac*M.star.back))
    }
    } else {
      lambda.back <- 128L
    }
    
    lalph.corr <- dpois(M,lambda=lambda.back,log=TRUE) - dpois(M,lambda = lambda,log=TRUE)
  } else {
    q.prop <- first.step$q1
    v.prop <- first.step$v1
    H1 <- first.step$H1
    lalph.corr <- 0.0
    M.star.back <- M.star
  }
  
  lalph <- H0-H1 + lalph.corr
  alpha <- exp(min(0,lalph))
  if(runif(1)<alpha){
    q.out <- q.prop
    v.out <- v.prop
  } else {
    q.out <- q0
    v.out <- -v0
  }
 
  #print(c(q.out,v.out)) 
  #print(paste0("h.ref = ",h.ref))
 
  return(list(q.out=q.out,v.out=v.out,
       last.N.f=last.N.f,last.N.b=M.star.back,
       lalph=lalph,NN=M,alpha=alpha,
       ev.out=fun(q.out),
       n.eval=n.eval,
       energy.err.last=abs(H0-H1),
       lalph.energy=H0-H1,
       lalph.correction=lalph.corr))
}


lp.disc.exp <- function(x,I=0,pI=0.99){
  l <- -log(1.0-pI)
  return(log(expm1(l)) + l*(I-x-1))
}
rng.disc.exp <- function(n,I=0,pI=0.99){
  l <- -log(1.0-pI)
  x.max <- ceiling((log(expm1(l)) + 35 + l*(I-1))/l)
  return(sample(x=I:x.max,size=n,replace = TRUE,prob=exp(lp.disc.exp(I:x.max,I=I,pI=pI))))
}


adapt.step <- function(fun,igr,q0,v0,h.ref=0.5,lambda.pri=0.05,I0.prob=0.99,tol=1.0e-1,
                       N.max=50L,err.summary=sum,
                       old.ev0=NULL,
                       NUT.q0=NULL,
                       N.0=0L){
  if(N.0>0){
    error("adapt step: N.0>0 not implemented")
  }
  d <- length(q0)
  Nmax <- max(N.max,qpois(1.0e-10,lambda=lambda.pri,lower.tail = FALSE))
  NUT.t <- -1.0
  if(is.null(old.ev0)){
    ev0 <- fun(q0)
    n.eval <- 1L
  } else {
    ev0 <- old.ev0
    n.eval <- 0L
  }
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  
  for(i in 0:Nmax){
    nstep <- 2^i
    hh <- h.ref/nstep
    err.ests <- numeric(nstep)
    q <- q0
    v <- v0
    ev.last <- ev0
    for(j in 1:nstep){
      igr.out <- igr(fun,q,v,h=hh,old.eval = ev.last,tol=tol)
      q <- igr.out$q1
      v <- igr.out$v1
      ev.last <- igr.out$ev1
      err.ests[j] <- igr.out$err
      n.eval <- n.eval + igr.out$n.eval
      if(!is.null(NUT.q0) && NUT.t<0.0){
        if(sum((q-NUT.q0)*v)<0.0) NUT.t <- j*hh
      }
    }
    
    if(all(is.finite(err.ests))){
      err <- err.summary(err.ests)
      
      if(err<1.0){
        energy.err <- abs(H0-(-igr.out$ev1$lp+0.5*sum(v^2))) 
        last.N <- i
        break
      }
    }
  }
  
  lp.N <- dpois(last.N:Nmax,lambda = lambda.pri,log=TRUE)
  lp.N <- lp.N - max(lp.N)
  p.N.f <- exp(lp.N)
  pp.sum <- sum(p.N.f)
  p.N.f <- (1.0/pp.sum)*p.N.f
  lp.N.f <- lp.N - log(pp.sum)
  #plot(last.N:Nmax,p.N.f)
  ###NN <- sample(x=last.N:Nmax,size = 1,prob = p.N.f)
  NN <- rng.disc.exp(1,I=last.N,pI=I0.prob)
  #print(paste0("last.N: ",last.N))
  #print(paste0("NN: ",NN))
  
  
  
  if(NN>last.N){
    nstep <- 2^NN
    hh <- h.ref/nstep
    err.ests <- numeric(nstep)
    q <- q0
    v <- v0
    ev.last <- ev0
    for(j in 1:nstep){
      igr.out <- igr(fun,q,v,h=hh,old.eval = ev.last,tol=tol)
      q <- igr.out$q1
      v <- igr.out$v1
      ev.last <- igr.out$ev1
      err.ests[j] <- igr.out$err
      n.eval <- n.eval + igr.out$n.eval
    }
  }
  # proposal
  q.prop <- q
  v.prop <- v
  H1 <- -igr.out$ev1$lp + 0.5*sum(v^2)
  #print(paste0("proposal energy error: ",H0-H1))
  #print(paste0("HMC acc prob: ",exp(min(0,H0-H1))))
  # now work out backward last.N
  ev0.back <- igr.out$ev1
  last.N.b <- last.N
  if(last.N>0){
    for(i in 0:(last.N-1)){
      nstep <- 2^i
      hh <- h.ref/nstep
      err.ests <- numeric(nstep)
      q <- q.prop
      v <- -v.prop
      ev.last <- ev0.back
      for(j in 1:nstep){
        igr.out <- igr(fun,q,v,h=hh,old.eval = ev.last,tol=tol)
        q <- igr.out$q1
        v <- igr.out$v1
        ev.last <- igr.out$ev1
        err.ests[j] <- igr.out$err
        n.eval <- n.eval + igr.out$n.eval
      }
      if(all(is.finite(err.ests))){
        err <- err.summary(err.ests)
        if(err<1.0){
          last.N.b <- i
          break
        }
      }
    }
  }
  correction <- 0.0
  if(last.N==last.N.b){
    lalph <- H0-H1
  } else {
    ###correction <- ppois(last.N-1,lambda=lambda.pri,log=TRUE,lower.tail = FALSE) -
    ###  ppois(last.N.b-1,lambda=lambda.pri,log=TRUE,lower.tail = FALSE)
    correction <- lp.disc.exp(NN,I=last.N.b,pI=I0.prob)-lp.disc.exp(NN,I=last.N,pI=I0.prob)
    lalph <- H0-H1+ correction
    
  }
  alpha <- exp(min(0.0,lalph))
  if(runif(1)<alpha){
    q.out <- q.prop
    v.out <- v.prop
    acc <- 1
  } else {
    q.out <- q0
    v.out <- -v0
    acc <- 0
  }
  
  return(list(q.out=q.out,v.out=v.out,
              last.N.f=last.N,last.N.b=last.N.b,
              lalph=lalph,NN=NN,alpha=alpha,
              ev.out=fun(q.out),
              n.eval=n.eval,
              energy.err.last=energy.err,
              lalph.energy=H0-H1,
              lalph.correction=correction,
              NUT.t=NUT.t,
              h.ref=h.ref))
  
}


adapt.step.energy <- function(fun,igr,q0,v0,h.ref=0.5,lambda.pri=0.05,I0.prob=0.99,tol=1.0e-1,
                       N.max=50L,
                       old.ev0=NULL,
                       NUT.q0=NULL,
                       N.0=0L){
  Nmax <- max(N.max,10L)
  d <- length(q0)
  NUT.t <- -1.0
  if(is.null(old.ev0)){
    ev0 <- fun(q0)
    n.eval <- 1L
  } else {
    ev0 <- old.ev0
    n.eval <- 0L
  }
  H0 <- -ev0$lp + 0.5*sum(v0^2)
  
  for(i in N.0:Nmax){
    nstep <- 2^i
    hh <- h.ref/nstep
    #err.ests <- numeric(nstep)
    q <- q0
    v <- v0
    ev.last <- ev0
    for(j in 1:nstep){
      igr.out <- igr(fun,q,v,h=hh,old.eval = ev.last,tol=tol)
      q <- igr.out$q1
      v <- igr.out$v1
      ev.last <- igr.out$ev1
      #err.ests[j] <- igr.out$err
      n.eval <- n.eval + igr.out$n.eval
      if(!is.null(NUT.q0) && NUT.t<0.0){
        if(sum((q-NUT.q0)*v)<0.0) NUT.t <- j*hh
      }
    }
    
    H1p <- -igr.out$ev1$lp+0.5*sum(v^2)
    err <- abs(H0-H1p)/tol
    if(is.finite(err)){
    
      
      if(err<1.0){
        energy.err <- err
        last.N <- i
        break
      }
    }
  }
  
  NN <- rng.disc.exp(1,I=last.N,pI=I0.prob)
  #print(paste0("last.N: ",last.N))
  #print(paste0("NN: ",NN))
  
  
  
  if(NN>last.N){
    nstep <- 2^NN
    hh <- h.ref/nstep
    #err.ests <- numeric(nstep)
    q <- q0
    v <- v0
    ev.last <- ev0
    for(j in 1:nstep){
      igr.out <- igr(fun,q,v,h=hh,old.eval = ev.last,tol=tol)
      q <- igr.out$q1
      v <- igr.out$v1
      ev.last <- igr.out$ev1
      #err.ests[j] <- igr.out$err
      n.eval <- n.eval + igr.out$n.eval
    }
  }
  # proposal
  q.prop <- q
  v.prop <- v
  H1 <- -igr.out$ev1$lp + 0.5*sum(v^2)
  #print(paste0("proposal energy error: ",H0-H1))
  #print(paste0("HMC acc prob: ",exp(min(0,H0-H1))))
  # now work out backward last.N
  ev0.back <- igr.out$ev1
  last.N.b <- last.N
  if(last.N>0){
    for(i in N.0:(last.N-1)){
      nstep <- 2^i
      hh <- h.ref/nstep
      #err.ests <- numeric(nstep)
      q <- q.prop
      v <- -v.prop
      ev.last <- ev0.back
      for(j in 1:nstep){
        igr.out <- igr(fun,q,v,h=hh,old.eval = ev.last,tol=tol)
        q <- igr.out$q1
        v <- igr.out$v1
        ev.last <- igr.out$ev1
        #err.ests[j] <- igr.out$err
        n.eval <- n.eval + igr.out$n.eval
      }
      H0p <- -igr.out$ev1$lp+0.5*sum(v^2)
      err <- abs(H0p-H1)/tol
      if(is.finite(err)){
        #err <- err.summary(err.ests)
        if(err<1.0){
          last.N.b <- i
          break
        }
      }
    }
  }
  correction <- 0.0
  if(last.N==last.N.b){
    lalph <- H0-H1
  } else {
    ###correction <- ppois(last.N-1,lambda=lambda.pri,log=TRUE,lower.tail = FALSE) -
    ###  ppois(last.N.b-1,lambda=lambda.pri,log=TRUE,lower.tail = FALSE)
    correction <- lp.disc.exp(NN,I=last.N.b,pI=I0.prob)-lp.disc.exp(NN,I=last.N,pI=I0.prob)
    lalph <- H0-H1+ correction
    
  }
  alpha <- exp(min(0.0,lalph))
  if(runif(1)<alpha){
    q.out <- q.prop
    v.out <- v.prop
    acc <- 1
  } else {
    q.out <- q0
    v.out <- -v0
    acc <- 0
  }
  
  return(list(q.out=q.out,v.out=v.out,
              last.N.f=last.N,last.N.b=last.N.b,
              lalph=lalph,NN=NN,alpha=alpha,
              ev.out=fun(q.out),
              n.eval=n.eval,
              energy.err.last=energy.err,
              lalph.energy=H0-H1,
              lalph.correction=correction,
              NUT.t=NUT.t,
              h.ref=h.ref))
  
}



fixed.sampler <- function(fun,q0,v0,n.iter=100000L,
                          h.ref=0.3,n.step=1L,refresh.mu=pi){
  d <- length(q0)
  q.samples <- matrix(0.0,n.iter+1,d)
  diagnostics <- matrix(0.0,n.iter,2)
  q <- q0
  v <- v0
  q.samples[1,] <- q0
  h <- h.ref/n.step
  refresh.lambda <- 1.0/refresh.mu
  for(iter in 1:n.iter){
    qq <- q
    vv <- v
    ev <- fun(qq)
    H0 <- -ev$lp + 0.5*sum(vv^2)
    for(i in 1:n.step){
      vh <- vv + 0.5*h*ev$grad
      qq <- qq + h*vh
      ev <- fun(qq)
      vv <- vh + 0.5*h*ev$grad
    }
    H1 <- -ev$lp + 0.5*sum(vv^2)
    
    if(runif(1)<exp(min(0.0,H0-H1))){
      q <- qq
      v <- vv
      acc <- 1
    } else {
      v <- -v
      acc <- 0
    }
    
    
    
    q.samples[iter+1,] <- q
    diagnostics[iter,1] <- exp(min(0.0,H0-H1))
    diagnostics[iter,2] <- acc
    
    if(rexp(1,rate=refresh.lambda)<h.ref) v <- rnorm(length(v))
    
  }
  
  return(list(q.samples=q.samples,diagnostics=diagnostics))
}




adapt.sampler <- function(fun,igr,q0,v0,n.iter=100000L,
                          h.init=0.3,
                          refresh.mu=pi,
                          lambda.pri=0.05,
                          I0.prob=0.99,
                          rho.refresh=1.0,
                          last.N.target=0.8,
                          energy.error.target=0.05,
                          step.type=adapt.step,
                          NUT.fac=2.0,
                          DA.kappa=0.85,
                          DA.gamma=0.05,
                          old.run=NULL,
                          adapt.mu=is.null(old.run),
                          adapt.h=is.null(old.run),
                          adapt.tol=is.null(old.run),
                          N.0=0){
  d <- length(q0)
  q.samples <- matrix(0.0,n.iter+1,d)
  v.samples <- matrix(0.0,n.iter+1,d)
  diagnostics <- matrix(0.0,n.iter,13)
  
  q.samples[1,] <- q0
  v.samples[1,] <- v0
  q <- q0
  v <- v0
  refresh.lambda <- 1.0/refresh.mu
  NUT.q0 <- q
  NUT.time <- 0.0
  NUT.done <- FALSE
  NUT.times <- c()
  NUT.censored.I <- c()
  old.eval <- fun(q0)
  
  is.not.energy <- !paste(deparse(step.type),collapse = "")==paste(deparse(adapt.step.energy),collapse = "")
  
  
  h.ref <- h.init
  tol <- energy.error.target
  H.h.sum <- 0.0
  if(is.not.energy) H.tol.sum <- 0.0
  
  if(!is.null(old.run)){
    n.iter.old <- length(old.run$diagnostics[,1])
    h.ref <- old.run$diagnostics[n.iter.old,5]
    tol <- old.run$diagnostics[n.iter.old,7]
    refresh.lambda <- 1.0/old.run$diagnostics[n.iter.old,13]
  }
  
  
  
  
  for(iter in 1:n.iter){
    
    s.out <- step.type(fun,igr,q,v,h.ref=h.ref,lambda.pri = lambda.pri,I0.prob=I0.prob,tol=tol,old.ev0=old.eval,NUT.q0=NUT.q0,N.0=N.0)  
    
    q <- s.out$q.out
    v <- s.out$v.out
    old.eval <- s.out$ev.out
    
    #v <- rho.refresh*v + sqrt(1.0-rho.refresh^2)*rnorm(length(v))
    
    refresh <- FALSE
    if(rexp(1,rate=refresh.lambda)<h.ref){
      refresh <- TRUE
      v <- rnorm(length(v))
      NUT.q0 <- q
    }
    
    q.samples[iter+1,] <- q
    v.samples[iter+1,] <- v
    diagnostics[iter,1] <- s.out$last.N.f
    diagnostics[iter,2] <- s.out$NN
    diagnostics[iter,3] <- s.out$alpha
    diagnostics[iter,4] <- s.out$n.eval
    diagnostics[iter,5] <- h.ref
    diagnostics[iter,6] <- s.out$energy.err.last
    diagnostics[iter,7] <- tol
    diagnostics[iter,8] <- s.out$last.N.b
    diagnostics[iter,9] <- s.out$lalph.energy
    diagnostics[iter,10] <- s.out$lalph.correction
    diagnostics[iter,11] <- s.out$NUT.t
    diagnostics[iter,12] <- refresh
    diagnostics[iter,13] <- 1.0/refresh.lambda
    
    # adaptation of tuning parameters
    # h adapted so that around last.N.target of steps only involves a single integration step
    H.h <-  last.N.target - as.integer(s.out$last.N.f==N.0) 
    H.h.sum <- H.h.sum + H.h
    if(iter>10 && iter<0.5*n.iter && iter %% 2==0 && adapt.h){
      l.h <- log(h.init) - (sqrt(iter)/0.5)*(DA.gamma/(iter+10))*H.h.sum
      eta <- iter^(-DA.kappa)
      l.h.bar <- eta*l.h + (1.0-eta)*log(h.ref)
      h.ref <- exp(l.h.bar)
    }
    
    # tol adapted for energy error
    if(is.not.energy && adapt.tol){
      H.tol <- s.out$energy.err.last/energy.error.target-1.0
      H.tol.sum <- H.tol.sum + H.tol
      if(iter>10 && iter<0.5*n.iter ){
        l.tol <- log(energy.error.target) - (sqrt(iter)/0.5)*(DA.gamma/(iter+10))*H.tol.sum
        eta <- iter^(-DA.kappa)
        l.tol.bar <- eta*l.tol + (1.0-eta)*log(tol)
        tol <- exp(l.tol.bar)
      }
    }
    
    # lambda.refresh adapted according to NUT criterion
    if(iter<0.5*n.iter && adapt.mu){
      if(!NUT.done){
        
        if(s.out$NUT.t>0.0){
          NUT.time <- NUT.time + s.out$NUT.t
          NUT.done <- TRUE
        } else {
          NUT.time <- NUT.time + s.out$h.ref
        }
      }
      if(refresh){
        NUT.times <- c(NUT.times,NUT.time)
        NUT.censored.I <- c(NUT.censored.I,NUT.done)
        NUT.time <- 0.0
        NUT.done <- FALSE
        if(sum(NUT.censored.I)>=10){
          NUT.mean <- sum(NUT.times)/sum(NUT.censored.I) # exponential MLE
          #print(paste0("NUT.mean : ",NUT.mean))
          refresh.lambda <- max(0.5*refresh.lambda,min(2*refresh.lambda,1.0/(NUT.fac*NUT.mean)))
        } else if(length(NUT.times) %% 100 == 0) {
          
          refresh.lambda <- 0.5*refresh.lambda
        }
      }
    } else if(!is.null(NUT.q0)){
      NUT.q0 <- NULL
    }
    #print(NUT.censored.I)
    #print(NUT.times)
    #print(refresh.lambda)
    #if(iter==300) stop("dsfd")
    #print(paste0("H.tol: ",H.tol))
    #print(s.out$energy.err.last)
    #print(h.ref)
    #print(tol)
    
    
    
  }
  
  return(list(q.samples=q.samples,
              v.samples=v.samples,
              diagnostics=diagnostics,
              NUT.times=NUT.times,
              NUT.censored.I=NUT.censored.I))
  
}




fun <- lp.smile #lp.corr.gauss #lp.dwell #lp.std.gauss #lp.funnel #
igr <- leap.frog.euler
q0 <- rnorm(2)
v0 <- rnorm(2)
h <- 0.5


#s.ret <- adapt.sampler(fun,igr,q0,v0,h.init = h)




# ret <- igr(fun=fun,q0=q0,v0=v0,h=h)
# ret.back <- igr(fun,q0=ret$q1,v0=-ret$v1,h=h)
# print(ret$err-ret.back$err)

# 
# adapt.step(fun,igr,q0,v0,h.ref=h)
# 


# hs <- exp(seq(from=log(1.0e-2),to=log(0.3),length.out=1000))
# err.en <- 0*hs
# err.eu <- 0*hs
# for(i in 1:length(hs)){
#   err.en[i] <- leap.frog.energy(fun,q0,v0,h=hs[i])$err
#   err.eu[i] <- leap.frog.euler(fun,q0,v0,h=hs[i])$err
# }
# 
# plot(hs,err.en,type="l",log="xy",ylim=c(min(c(err.en,err.eu)),max(c(err.en,err.eu))))
# lines(hs,err.eu,col="red")
# 
# 
# 
# 
# 
