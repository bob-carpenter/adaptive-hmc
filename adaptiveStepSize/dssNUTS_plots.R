mdl <- "lp.funnel" # "lp.smile" #
fn <- paste0(mdl,"_",1000000)
load(fn)
print("load done")

subset <- seq(from=1,to=1000000,by=100)

pdf(paste0(fn,".pdf"),height=11.7,width=8.3)
par(mfrow=c(3,2))
plot(qs[1,subset],qs[2,subset],pch=19,cex=0.1,ylim=c(-30,30),
     main="scatter plot (thinned)",
     xlab="q_1",ylab="q_2")

#plot(subset,qs[1,subset],type="l",main="traceplot (thinned), q_1")

hist(qs[1,],probability = TRUE,breaks=51,main="histogram of q_1")
curve(dnorm,add=TRUE,col="red")

plot(min.q[subset],wts.ess[subset],pch=19,cex=0.1,ylim=c(0.9,1),
     main="energy wts ESS",
     xlab="smallest q_1 visited",xlim=c(min(min.q),max(min.q)))
lines(ksmooth(min.q,wts.ess,n.points = 1000),col="red")

plot(max.q[subset],ifin[subset],pch=19,cex=0.1,xlim=c(min(min.q),max(min.q)),
     main="K (i.e. number of NUTS iterations)",
     xlab="largest q_1 visited")
lines(ksmooth(max.q,ifin,n.points = 1000),col="red")


plot(min.q[subset],max.c[subset],pch=19,cex=0.1,
     xlab="smallest q_1 visited",
     ylab="largest c used",
     main="largest c used in orbit",
     xlim=c(min(min.q),max(min.q)))
lines(ksmooth(min.q,max.c,n.points = 1000),col="red")

plot(ksmooth(min.q,alph.n,n.points = 1000),type="l",ylim=c(0,1),lty=1,
     xlab="smallest q_1 visited",
     ylab="accept prob",
     main="mean accept probs, black=NUT, red=adaptive step size")


lines(ksmooth(min.q[alph.n>0.5],alph.s[alph.n>0.5],n.points = 1000),col="red")
dev.off()

