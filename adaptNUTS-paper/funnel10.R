source("NUTS.R")
source("adaptNUTS.R")
set.seed(123)

if(1==1){
  fun <- lp.funnel10
  
  amins <- c(0.6,0.8,0.95)
  eerrs <- -log(amins)
  n.iter <- 1000000
  
  
  ret.a <- list()
  
  ret.n2 <- list()
  k <- 0
  for(ee in eerrs){
    print(ee)
    k <- k+1
    q0 <- rnorm(1,sd=3)
    q0 <- c(q0,sqrt(exp(q0))*rnorm(10))
    H.a <- adaptNUTS(fun,q0,n.iter=1000,n.warmup = 1000,energy.tol = ee)$diagnostics[1000,"H"]
    #H.w <- 0.5
    #H.a <- 0.5
    
    q0 <- rnorm(1,sd=3)
    q0 <- c(q0,sqrt(exp(q0))*rnorm(10))
    ret.a[[k]] <- adaptNUTS(fun,q0,n.iter=n.iter,n.warmup = 0,energy.tol = ee,H0=H.a)
    
    Ha.nuts <- mean(ret.a[[k]]$diagnostics[,"meanh"])
    
    ret.n2[[k]] <- NUTS(fun,q0,n.iter=n.iter,H0=Ha.nuts)
  }
  save.image("funnel10")
}

if(1==1){
  load("funnel10")



pdf("funnel_hist.pdf",width = 14,height = 7)
par(mfrow=c(2,3))
xg <- seq(from=-16,to=16,length.out=2000)
for(k in 1:length(amins)){
  par(mfg = c(1,k)) 
  mn <- paste0("adaptNUTS, $a_{min}=$",amins[k],
               ", $h$=",round(ret.a[[k]]$diagnostics[1000000,"H"],3))
  hist(ret.a[[k]]$samples[1,],probability = TRUE,xlim=c(-12,12),
       xlab=latex2exp::TeX("$\\omega$"),ylab="density",
       main=latex2exp::TeX(mn),
       cex.lab=1.6,
       cex.main=1.6)
  lines(xg,dnorm(xg,sd=3),col="red")
  par(mfg = c(2,k)) 
  mn <- paste0("NUTS, $h$=",round(ret.n2[[k]]$diagnostics[1000000,"H"],3))
  hist(ret.n2[[k]]$samples[1,],probability = TRUE,xlim=c(-12,12),
       xlab=latex2exp::TeX("$\\omega$"),ylab="density",
       main=latex2exp::TeX(mn),
       cex.lab=1.6,
       cex.main=1.6)
  lines(xg,dnorm(xg,sd=3),col="red")
}
dev.off()
}

