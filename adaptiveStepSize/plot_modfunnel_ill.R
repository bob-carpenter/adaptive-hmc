source("adaptRandomizedHMCStep.R")
set.seed(123)
fun <- lp.mod.funnel # lp.std.gauss # lp.smile # lp.corr.gauss # 
q <- c(0,0.0)
v <- c(-1.0,0.4)

out <- adaptRandomizedHMCStep(fun,q,v,H=0.5,J.fixed=8,plot.stats=TRUE)
pdf("plot_modfunnel_ill.pdf",width=8,height = 4)
par(mfrow=c(1,2))
plot(out$q.big.step[1,],out$q.big.step[2,],cex=1.2,xlab="q_1",ylab="q_2",xlim=c(-2,0.5),ylim=c(-0.2,0.2))
points(q[1],q[2],col="green")
lines(out$q.intermediate[1,],out$q.intermediate[2,],lty=1,cex=0.3,col="grey")
points(out$q.intermediate[1,],out$q.intermediate[2,],pch=4,cex=0.3)



plot(out$t.trace,out$H.trace-out$H.trace[1],type="l",lty=1,cex=0.3,col="grey",
     xlab="time",ylab="energy error",ylim=c(-0.1,0.02))
points(out$t.trace,out$H.trace-out$H.trace[1],pch=4,cex=0.4)
nsteps <- 4*2^out$Cs
for(i in 0:8){
  lines(i*0.5*c(1,1),c(-10,10),col="red")
  if(i<8){
    text(x=0.5*i+0.25,y=-0.09,labels = paste0("N = ",nsteps[i+1]),cex=0.5)
  }
}



dev.off()