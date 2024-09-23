source("dynStepSizeNUTS.R")
set.seed(1)
mdl <- "lp.funnel" #"lp.smile" # lp.corr.gauss #  lp.mod.funnel # lp.std.gauss #  

fun <-  get(mdl) 
n.iter <- 1000000
qs <- matrix(0,2,n.iter)
ifin <- numeric(n.iter)
alph.n <- numeric(n.iter)
alph.s <- numeric(n.iter)
min.q <- numeric(n.iter)
max.q <- numeric(n.iter)
max.c <- numeric(n.iter)
wts.ess <- numeric(n.iter)
q <- c(rnorm(1),0.0001)

for(i in 1:n.iter){
  
  out <- dynStepSizeNUTS(fun,q,rnorm(2),H=0.5,R=4,NUTS.max = 12)
  q <- out$q.new

  qs[,i] <- q
  ifin[i] <- out$i.final
  alph.n[i] <- out$alpha.nuts
  alph.s[i] <- out$alpha.as
  min.q[i] <- out$min.q[1]
  max.q[i] <- out$max.q[1]
  max.c[i] <- out$max.c
  wts.ess[i] <- out$wts.ess
}

save.image(paste0(mdl,"_",n.iter))