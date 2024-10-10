
source("NUTS.R")
source("adaptNUTS.R")

set.seed(1)


fun <- lp.std.gauss

ds <- 2^(10:15)
n.rep <- 50
n.per <- 50
res <- c()
if(1==0){
  for(d in ds){
    q0 <- rnorm(d)
    H.a <- adaptNUTS(fun=fun,q0=q0,n.iter = 1000,n.warmup = 1000,energy.tol = -log(0.6))$diagnostics[1000,"H"]
    q0 <- rep(0,d)
    for(rep in 1:n.rep){
      ret <- adaptNUTS(fun=fun,q0=q0,n.iter=n.per,n.warmup=0,energy.tol = -log(0.6),H0=H.a)
      rr <- c(apply(ret$samples^2,2,sum)[2:(n.per+1)],cumsum(ret$diagnostics[,"neval"]),ret$diagnostics[,"meanh"],ret$diagnostics[,"orbitLength"],H.a,1,d)
      res <- rbind(res,rr)
      ret <- NUTS(fun=fun,q0=q0,n.iter=n.per,H0=H.a)
      rr <- c(apply(ret$samples^2,2,sum)[2:(n.per+1)],cumsum(ret$diagnostics[,"neval"]),rep(H.a,n.per),ret$diagnostics[,"orbitLength"],H.a,0,d)
      res <- rbind(res,rr)
    }
  }
  
  colnames(res) <- c(paste0("dev",1:n.per),paste0("neval",1:n.per),paste0("meanh",1:n.per),paste0("orbitL",1:n.per),"H","method","d")
  
  save.image("gaussTransient")
}

if(1==1){
  dds <- c(ds[2],ds[4],ds[6])
  load("gaussTransient")
  pdf("gaussTransient.pdf",width=14,height = 14)
  
  par(mfrow=c(2,length(dds))) #length(ds)))
  coltr <- c(rgb(red = 1, green = 0, blue = 0, alpha = 0.4),
             rgb(red = 0, green = 1, blue = 0, alpha = 0.7),
             rgb(red = 0, green = 0, blue = 1, alpha = 0.4))
  colab <- c(rgb(red = 1, green = 0, blue = 0, alpha = 1),
             rgb(red = 0, green = 1, blue = 0, alpha = 1),
             rgb(red = 0, green = 0, blue = 1, alpha = 1))
  for(j in 1:length(dds)){ #length(ds))){
    par(mfg = c(1,j)) 
    plot(1e-100,1e100,xlim=c(1,n.per),
         ylim=c(0.01,1.1*qchisq(0.99,df=dds[j])),
         xlab="iteration #",
         main=paste0("d = ",dds[j]),
         ylab="deviance",
         cex.lab=1.4,cex.axis=1.6,cex.main=1.6)
    for(method in 0:1){
      dta <- res[res[,"method"]==method & res[,"d"]==dds[j],]
      for(k in 1:nrow(dta)){
        points(1:n.per,dta[k,1:n.per],col=coltr[method+1],pch=19,cex=0.5)
      }
    }
    legend("bottomright",legend = c("NUTS","adaptNUTS"),
           col=coltr[1:2],pch=rep(19,2))
    
    par(mfg = c(2,j)) 
    plot(1e-100,1e100,xlim=c(10,max(res[,(n.per+1):(2*n.per)])),
         ylim=c(0.01,1.1*qchisq(0.99,df=dds[j])),xlab="# gradient evals",log="x",
         main=paste0("d = ",dds[j]),
         ylab="deviance",
         cex.lab=1.4,cex.axis=1.6,cex.main=1.6)
    for(method in 0:1){
      dta <- res[res[,"method"]==method & res[,"d"]==dds[j],]
      for(k in 1:nrow(dta)){
        points(dta[k,(n.per+1):(2*n.per)],dta[k,1:n.per],col=coltr[method+1],pch=19,cex=0.2)
      }
      f1 <- log(as.vector(dta[,(n.per+1):(2*n.per)]))
      f2 <- as.vector(dta[,1:n.per])
      xp <- seq(from=min(f1),to=max(f1),length.out=1000)
      ks.out <- ksmooth(f1,f2,bandwidth = 1,x.points = xp)
      lines(exp(ks.out$x),ks.out$y,lwd=3,col=colab[method+1])
      lines(exp(ks.out$x),ks.out$y,lwd=1,col="black")
    }
    legend("bottomright",legend = c("NUTS","adaptNUTS"),
       col=coltr[1:2],pch=rep(19,2))
  }
  dev.off()
  pdf("gaussTransient_stepSize.pdf",width = 7,height = 7)
  par(mfrow=c(1,1))
  hplot1 <- matrix(0.0,3,length(ds))
  hplotn <- matrix(0.0,3,length(ds))
  for(method in 0:2){
    for(j in 1:length(ds)){
      dta <- res[res[,"method"]==method & res[,"d"]==ds[j],]
      hplot1[method+1,j] <- mean(dta[,c("meanh1","meanh2","meanh3")]) 
      hplotn[method+1,j] <- mean(dta[,"meanh50"])
    }
  }
  dgrid <- 2^(1:16)
  plot(ds,hplot1[2,],log="xy",ylim=c(min(hplot1[2,]),max(hplotn[2,])),pch=19,cex=2,
       xlab="dimension d",
       ylab="mean step size",
       main="adaptNUTS mean step size",
       cex.lab=1.4,
       cex.axis=1.4,
       cex.main=1.6)
  
  lines(dgrid,(hplot1[2,length(ds)])/(ds[length(ds)]^(-1/2))*dgrid^(-1/2))
  points(ds,hplotn[2,],pch=18,cex=2,col="red")
  
  lines(dgrid,(hplotn[2,length(ds)])/(ds[length(ds)]^(-1/4))*dgrid^(-1/4),col="red",lty=2)
  legend("bottomleft",legend = latex2exp::TeX(c("mean step size iteration # 1-3",
                                                "mean step size iteration # 50",
                                                "$\\propto d^{-1/2}$",
                                                "$\\propto d^{-1/4}$")),
         pch=c(19,18,NA,NA),col=c("black","red","black","red"),
         lty=c(NA,NA,1,2))
  
  dev.off()
}

