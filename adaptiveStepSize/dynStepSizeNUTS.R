source("singleStepMethods.R")

dynStepSizeNUTS <- function(fun,q0,v0,
                            H=0.5,
                            R=4L,
                            NUTS.max=8L,
                            adaptStepSize.tol=0.05,
                            adaptStepSize.max=8L,
                            probI=0.99,
                            debug=FALSE){
  d <- length(q0)
  msize <- 2^(NUTS.max+2)+2
  qs <- matrix(0.0,d,msize) # states of the orbit (much too large!)
  vs <- matrix(0.0,d,msize) # states of the orbit
  Hs <- numeric(msize) # Hamiltonian
  Ifs <- numeric(msize) # I-s for forward integration
  Ibs <- numeric(msize) # I-s for backward integration
  cfs <- numeric(msize) # c-s for forward integration
  cbs <- numeric(msize) # c-s for backward integration
  
  os <- round(0.5*msize+1)
  dirs.forw <- rep(999,NUTS.max+1)
  fun0 <- fun(q0)
  Hs[os] <- -fun0$lp + 0.5*sum(v0^2)
  
  qs[,os] <- q0
  vs[,os] <- v0
  n.forw <- 0L
  n.back <- 0L
  fun.forw <- fun0
  fun.back <- fun0
  
  i.final <- NUTS.max
  for(i in 0L:NUTS.max){ # main orbit building loop
    nstep <- 2L^i
    back <- FALSE
    if(runif(1)<0.5){
      #integrate backward
      dirs.forw[i+1] <- 0L
      back <- TRUE
      qi <- qs[,os-n.back]
      vi <- -vs[,os-n.back]
      fi <- fun.back
      Hi <- Hs[os-n.back]
    } else {
      #integrate forward
      dirs.forw[i+1] <- 1L
      qi <- qs[,os+n.forw]
      vi <- vs[,os+n.forw]
      fi <- fun.forw
      Hi <- Hs[os+n.forw]
    }
    
    
    for(j in 1:nstep){ # loop over adaptive steps 
      I.forw <- adaptStepSize.max
      for(c in 0L:adaptStepSize.max){ #precision loop
        n.sub <- R*(2L^c)
        h <- H/n.sub
        qq <- qi
        vv <- vi
        ff <- fi
        Hsub <- numeric(n.sub+1)
        Hsub[1] <- Hi
        for(k in 1:n.sub){
          vh <- vv + 0.5*h*ff$grad
          qq <- qq + h*vh
          ff <- fun(qq)
          vv <- vh + 0.5*h*ff$grad
          Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
        }
        energy.err <- max(Hsub) - min(Hsub)
        
        if(energy.err<adaptStepSize.tol){ # stop if desired precision is found
          I.forw <- c
          break
        }
      } # loop over precisions
      
      # simulate precision
      cc <- rng.disc.exp(1L,I.forw,pI=probI)
      
      if(cc>I.forw){ # need to simulate with even higher precision
        n.sub <- R*(2L^cc)
        h <- H/n.sub
        qq <- qi
        vv <- vi
        ff <- fi
        Hsub <- numeric(n.sub+1)
        Hsub[1] <- Hi
        for(k in 1:n.sub){
          vh <- vv + 0.5*h*ff$grad
          qq <- qq + h*vh
          ff <- fun(qq)
          vv <- vh + 0.5*h*ff$grad
          Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
        }
      }
      
      
      # store progress
      if(back){
        n.back <- n.back + 1L
        cbs[n.back] <- cc
        Ibs[n.back] <- I.forw
        qs[,os-n.back] <- qq
        vs[,os-n.back] <- -vv
        Hs[os-n.back] <- -ff$lp + 0.5*sum(vv^2)
        fun.back <- ff
      } else {
        n.forw <- n.forw + 1L
        cfs[n.forw] <- cc
        Ifs[n.forw] <- I.forw
        qs[,os+n.forw] <- qq
        vs[,os+n.forw] <- vv
        Hs[os+n.forw] <- -ff$lp + 0.5*sum(vv^2)
        fun.forw <- ff
      }
      # prepare for next integration step
      qi <- qq
      vi <- vv
      fi <- ff
      Hi <- -ff$lp + 0.5*sum(vv^2)
    } # steps within NUTS iterations
    
    # NUT test
    test.forw <- sum(vs[,os+n.forw]*(qs[,os+n.forw]-qs[,os-n.back]))
    test.back <- sum(vs[,os-n.back]*(qs[,os+n.forw]-qs[,os-n.back]))
    if(debug) print(c(test.forw,test.back))
    if(test.forw<0.0 || test.back<0.0){
      i.final <- i
      break
    }
  } # doubling iteration
  
  # sample proposed state
  Hwts <- -Hs[(os-n.back):(os+n.forw)]
  Hwts <- exp(Hwts-max(Hwts))
  Hwts <- Hwts/sum(Hwts)
  J <- sample(x=(-n.back):n.forw,size=1,replace = TRUE,prob = Hwts)
  q.prop <- qs[,os+J]
  v.prop <- -vs[,os+J]
  
  
  # work out sequence of forward/backward integration that would lead to
  # the same simulated trajectory FROM the proposal 
  n.forw.rev <- n.back + J
  dirs.rev <- rep(999,NUTS.max+1)
  rem <- n.forw.rev
  for(i.rev in (i.final):0L){
    if(rem>=2^i.rev){
      dirs.rev[i.rev+1] <- 1L
      rem <- rem - 2^i.rev
    } else {
      dirs.rev[i.rev+1] <- 0L
    }
  }
  if(debug) print("---")
  # now run sequence of NUT tests from the proposal
  n.f.r <- 0L
  n.b.r <- 0L
  i.final.rev <- i.final
  alpha.nuts <- 1.0
  if(i.final>0){
    for(i.rev in 0:(i.final-1)){
      nstep <- 2^i.rev
      if(dirs.rev[i.rev+1]<0.5){
        n.b.r <- n.b.r + nstep
      } else {
        n.f.r <- n.f.r + nstep
      }
      test.forw <- sum(-vs[,os+J-n.f.r]*(qs[,os+J-n.f.r]-qs[,os+J+n.b.r]))
      test.back <- sum(-vs[,os+J+n.b.r]*(qs[,os+J-n.f.r]-qs[,os+J+n.b.r]))
      if(debug) print(c(test.forw,test.back))
      if(test.forw<0.0 || test.back<0.0){
        i.final.rev <- i.rev
        break
      }
    }
  }
  if(i.final.rev!=i.final){
    q.new <- q0
    v.new <- -v0
    alpha.nuts <- 0.0
    if(debug) print("reject")
  } else {
    q.new <- q.prop
    v.new <- v.prop
  }
  
  # calculate accept prob related to adaptive step sizes
  alpha.as <- 1.0
  if(alpha.nuts>1.0e-14 && abs(J) > 0.5){
    if(J>0){
      # consider only the forward integration
      cs <- cfs[1:J]
      If <- Ifs[1:J]
      lpf <- lp.disc.exp(cs,If,pI=probI)
      Ib <- If
      
      for(j in 1:J){
        qi <- qs[,os+j]
        vi <- -vs[,os+j]
        Hi <- Hs[os+j]
        fi <- fun(qi) # no need to count this grad eval
        maxc <- If[j]-1
        if(cs[j]>If[j]) maxc <- adaptStepSize.max
        if(!(If[j]==0 && cs[j]==0)){ # no need to check/calculate backward I if the crudest accuracy was used
          for(c in 0:maxc){
            n.sub <- R*(2L^c)
            h <- H/n.sub
            qq <- qi
            vv <- vi
            ff <- fi
            Hsub <- numeric(n.sub+1)
            Hsub[1] <- Hi
            for(k in 1:n.sub){
              vh <- vv + 0.5*h*ff$grad
              qq <- qq + h*vh
              ff <- fun(qq)
              vv <- vh + 0.5*h*ff$grad
              Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
            }
            energy.err <- max(Hsub) - min(Hsub)
            #print(paste0("energy err: ",energy.err))
            if(energy.err<adaptStepSize.tol){
              Ib[j] <- c
              break
            }
          }
        }
      }
      lpb <- lp.disc.exp(cs,Ib,pI=probI)
      
    } else {
      # consider only the backward integration
      cs <- cbs[1:(-J)]
      If <- Ibs[1:(-J)]
      lpf <- lp.disc.exp(cs,If,pI=probI)
      Ib <- If
      for(j in 1:(-J)){
        qi <- qs[,os-j]
        vi <- vs[,os-j]
        fi <- fun(qi) # no need to count this grad eval
        Hi <- Hs[os-j]
        maxc <- If[j]-1
        if(cs[j]>If[j]) maxc <- adaptStepSize.max
        if(!(If[j]==0 && cs[j]==0)){ # no need to check/calculate backward I if the crudest accuracy was used
          for(c in 0:maxc){
            n.sub <- R*(2L^c)
            h <- H/n.sub
            qq <- qi
            vv <- vi
            ff <- fi
            Hsub <- numeric(n.sub+1)
            Hsub[1] <- Hi
            for(k in 1:n.sub){
              vh <- vv + 0.5*h*ff$grad
              qq <- qq + h*vh
              ff <- fun(qq)
              vv <- vh + 0.5*h*ff$grad
              Hsub[k+1] <- -ff$lp + 0.5*sum(vv^2)
            }
            energy.err <- max(Hsub) - min(Hsub)
            #print(paste0("energy err: ",energy.err))
            if(energy.err<adaptStepSize.tol){
              Ib[j] <- c
              break
            }
          }
        }
      }
      lpb <- lp.disc.exp(cs,Ib,pI=probI)
    }
    
    alpha.as <- exp(min(0.0,sum(lpb-lpf)))
    if(runif(1)>alpha.as){
      q.new <- q0
      v.new <- -v0
      if(debug) print("reject from adaptive step sizes")
    }

  } # transition has not alread been rejection by the NUTS stuff
  
  
  if(debug){
    print("energy wts")
    print(Hwts)
    print(paste0("J = ",J))
    print(c(-n.back,n.forw))
    print(paste0("nforw.rev = ",n.forw.rev))
    print(dirs.forw)
    print(dirs.rev)
    if(n.forw>0){
      print("forward integration")
      print(rbind(Ifs[1:n.forw],cfs[1:n.forw]))
    }
    if(n.back>0){
      print("backward integration")
      print(rbind(Ibs[1:n.back],cbs[1:n.back]))
    }
    
    if(J!=0 && alpha.nuts>0.0){
      print("forward-backward accept prob")
      print(rbind(If,Ib))
      print(lpb-lpf)
    }
    
    plot(qs[1,(os-n.back):(os+n.forw)],qs[2,(os-n.back):(os+n.forw)],type="l")
    points(q0[1],q0[2],col="green",pch=12)
    points(qs[1,os+J],qs[2,os+J],col="red")
  }
  return(list(q.new=q.new,v.new=v.new,
              J=J,
              n.forw=n.forw,
              n.back=n.back,
              i.final=i.final,
              alpha.nuts=alpha.nuts,
              alpha.as=alpha.as,
              min.q=apply(qs[,(os-n.back):(os+n.forw)],1,min),
              max.q=apply(qs[,(os-n.back):(os+n.forw)],1,max),
              max.c=max(c(max(cfs[1:n.forw]),max(cbs[1:n.back]))),
              wts.ess=1.0/(sum(Hwts^2)*length(Hwts))))
  
}


#out <- dynStepSizeNUTS(lp.funnel,rnorm(2),rnorm(2),
 #                      H=0.5,R=4,NUTS.max = 5,debug=TRUE)
if(1==0){
  fun <-  lp.funnel # lp.corr.gauss #lp.smile #  lp.mod.funnel # lp.std.gauss #  
  n.iter <- 1000000
  qs <- matrix(0,2,n.iter)
  ifin <- numeric(n.iter)
  alph.n <- numeric(n.iter)
  alph.s <- numeric(n.iter)
  q <- c(rnorm(1),0.0001)
  for(i in 1:n.iter){
    #for(ii in 1:30){
      out <- dynStepSizeNUTS(fun,q,rnorm(2),H=0.5,R=4,NUTS.max = 12)
      q <- out$q.new
    #}
    qs[,i] <- q
    ifin[i] <- out$i.final
    alph.n[i] <- out$alpha.nuts
    alph.s[i] <- out$alpha.as
  }
}