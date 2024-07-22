source("adaptRandomizedHMCStep.R")
set.seed(123)
fun <- lp.mod.funnel # lp.std.gauss # lp.smile # lp.corr.gauss # 
q <- rnorm(2)
v <- rnorm(2)

n.iter <- 100000L
res <- matrix(0.0,nrow=n.iter,ncol=9)
for(i in 1:n.iter){
  
  out <- adaptRandomizedHMCStep(fun,q,rnorm(2))
  res[i,1:2] <- out$q.new
  res[i,3] <- out$lalph
  res[i,4] <- max(abs(out$If-out$Ib))
  res[i,5] <- out$J
  res[i,6] <- min(out$Cs)
  res[i,7] <- max(out$Cs)
  res[i,8] <- out$q.min[1]
  res[i,9] <- out$q.max[1]
  q <- out$q.new
}

pdf("plots_modfunnel.pdf",height = 11,width = 8)
par(mfrow=c(3,2))
plot(res[,1],res[,2],pch=19,cex=0.1,xlab="q_1",ylab="q_2")
ts.plot(res[,1],main="q_1")
qqnorm(res[,1],col="grey",main="qqplot q_1")
abline(0,1)
acf(res[,1],main="ACF q_1")
plot(res[,8],res[,7],cex=0.1,pch=19,main="max c over trajectory",xlab="smallest q_1 visited during trajectory")
lines(ksmooth(res[,8],res[,7]),col="red")

plot(res[,8],exp(pmin(0.0,res[,3])),cex=0.1,pch=19,main="accept prob",xlab="smallest q_1 visited")

lines(ksmooth(res[,8],exp(pmin(0.0,res[,3]))),col="red")

dev.off()