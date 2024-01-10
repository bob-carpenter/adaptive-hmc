clear all;
%addpath ~nawaf/Dropbox/export_fig/

rng(123);

%% parameters

n_adapt_hmc_steps=1e6;   

nu=@(x) exp(-0.5*x.^2)/sqrt(2*pi);
force = @(x) -x;

% prep

h=1.9;
X0=2.0;
tol=1/32.0;
x_vec=zeros(n_adapt_hmc_steps,1);
ap_vec=zeros(n_adapt_hmc_steps,1);
tic
for oi=1:n_adapt_hmc_steps

%.. full velocity randomization

    V0=randn;
    V0star=V0;
    X0star=X0;               % current state of chain
    H0star=0.5*(V0^2+X0^2);  % initial energy

%.. Gibbs step 1: sample dt | (X0,V0)
     
    ii=0;
    while (1)
        [X1f,V1f,maxdH1f]=vverlet2(X0star,V0star,2^ii,h/2^ii);
        if maxdH1f<tol
            break;
        else
            ii=ii+1;
        end
    end   
    HH=h/2^ii; lambf=1/HH;

    UU=rand;
    dt=-log(rand)*HH;

%.. Gibbs step 2: hmc | dt

    [X1star,V1star,H1star]=vverlet(X0star,V0star,ceil(1.9/dt),dt);
    DeltaHstar=H1star-H0star;

    %.. compute lambb 
 
    ii=0;
    while (1)
        [X1f,V1f,maxdH1f]=vverlet2(X1star,-V1star,2^ii,h/2^ii);
        if maxdH1f<tol
            break;
        else
            ii=ii+1;
        end
    end   
    HH=h/2^ii; lambb=1/HH;

    alpha=min(1,exp(-DeltaHstar)*exp(-(lambb-lambf)*dt)*lambb/lambf);

    Bernoulli=(rand<alpha);
    X1=Bernoulli*X1star+(1-Bernoulli)*X0star;  %  actual next state

    ap_vec(oi)=alpha;

    x_vec(oi)=X1;
    dt_vec(oi)=dt;
    mean_dt_vec(oi)=1/lambf;
    X0=X1;
end
toc

disp([mean(ap_vec) mean(dt_vec)]);

%% graphical output

[n aHMC_em xout]=kde(x_vec(:),2^14,-8,8);
exact_im=nu(xout);
aHMC_em=aHMC_em/sum(aHMC_em(:));
exact_im=exact_im/sum(exact_im(:));

figure(5); hold on;
plot(xout,aHMC_em,'k','LineWidth',2);
plot(xout,exact_im,'k','LineWidth',2,'color',[0.75 0.75 0.75]);
%set(gca,'XTick',[0.25 0.5 0.75],'FontSize',16);
xlim([-2 2]);
xlabel('$x$','FontSize',16,'Interpreter','latex');
%title([' $\tau= $' num2str(h,'%3.2f')],'fontsize',20,'Interpreter','latex');
box on;
grid on;
set(gcf,'color',[1.0,1.0,1.0]);
legend({'empirical', 'exact'}, 'location', 'northeast', 'Interpreter','latex', 'fontsize',20, 'Orientation','vertical');
return
filename=['autoGHMC_asymp_bias_tau_' hstr '.pdf'];
export_fig(gcf,filename,'-pdf');
