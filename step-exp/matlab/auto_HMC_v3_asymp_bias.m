clear all;
%addpath ~nawaf/Dropbox/export_fig/

rng(123);

%% parameters

n_auto_hmc_steps=1e6;   
h=4.0; 

hstr = num2str(h,3);
Num = strfind(hstr,'.');
hstr(Num)='p';

nu=@(x) exp(-0.5*x.^2)/sqrt(2*pi);
force = @(x) -x;

% prep

X0=2.0;
tol=0.6;
x_vec=zeros(n_auto_hmc_steps,1);
ap_vec=zeros(n_auto_hmc_steps,1);
tic
for oi=1:n_auto_hmc_steps

    V0=randn;
    V0star=V0;
    X0star=X0;               % current state of chain
    H0star=0.5*(V0^2+X0^2);  % initial energy

%.. Gibbs step 1: sample II | (X0,V0)
    
    %.. compute lambf 

    ii=2;
    while (1)
        [X1f,V1f,maxdH1f]=pverlet2(X0star,V0star,2^ii,h/2^ii);
        [X1b,V1b,maxdH1b]=pverlet2(X1f,-V1f,2^ii,h/2^ii);

        if exp(-abs(maxdH1f))>tol && exp(-abs(maxdH1b))>tol
            break;
        else
            ii=ii+1;
        end
    end

    %disp([ii oi]); %pause(0.1); 
    lambf=1/ii; UU=rand;
    II=min(floor(log(UU)/log(1-lambf))+1,10);

%.. Gibbs step 2: hmc | II

    [X1star,V1star,H1star]=pverlet(X0star,V0star,2^II,h/2^II);
    DeltaHstar=H1star-H0star;

    %.. compute lambb 
 
    ii=2;
    while (1)
        [X1f,V1f,maxdH1f]=pverlet2(X1star,-V1star,2^ii,h/2^ii);
        [X1b,V1b,maxdH1b]=pverlet2(X1f,-V1f,2^ii,h/2^ii);

        if exp(-abs(maxdH1f))>tol && exp(-abs(maxdH1b))>tol
            break;
        else
            ii=ii+1;
        end
    end
    lambb=1/ii;


    alpha=min(1,exp(-DeltaHstar)*((1-lambb)/(1-lambf))^(II-1)*lambb/lambf);
    Bernoulli=(rand<alpha);
    X1=Bernoulli*X1star+(1-Bernoulli)*X0star;  %  actual next state

    ap_vec(oi)=alpha;
    %disp([alpha tol]); pause;

    x_vec(oi)=X1;
    X0=X1;
end
toc

disp(mean(ap_vec))

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
title([' $\tau= $' num2str(h,'%3.2f')],'fontsize',20,'Interpreter','latex');
box on;
grid on;
set(gcf,'color',[1.0,1.0,1.0]);
legend({'empirical', 'exact'}, 'location', 'northeast', 'Interpreter','latex', 'fontsize',20, 'Orientation','vertical');
return
filename=['autoGHMC_asymp_bias_tau_' hstr '.pdf'];
export_fig(gcf,filename,'-pdf');
