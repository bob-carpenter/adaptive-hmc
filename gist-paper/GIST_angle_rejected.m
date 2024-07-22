clear all;
addpath ~nawaf/Dropbox/export_fig/

rng(4);

%% parameters

n_gist_steps=5e4;   
dim=1e3; % dimension

nu=@(x) exp(-0.5*x.^2)/sqrt(2*pi);
force = @(x) -x;

% prep

sigma = (1:dim)/dim; sigma=sigma(:);

X0=ones(dim,1);
x_vec=zeros(n_gist_steps,1);

fun=@(t,x0,v0) sum(-1./sigma.*sin(t./sigma).*x0.*v0+cos(t./sigma).*v0.^2);

mean_ap=0.0;
tic
for oi=1:n_gist_steps

    Q0=X0;
    V0=randn(dim,1);

    % forward U-turn condition to find tau1

    tau10=atan(V0./Q0);        
    tau10=tau10+pi*(tau10<0);
    tau1 = fzero(@(t) fun(t,Q0,V0),mean(tau10));
    
    % backward U-turn condition to find tau2  

    alpha=rand*tau1;        
    Q1star=cos(alpha./sigma).*Q0+sigma.*sin(alpha./sigma).*V0;
    V1star=-1./sigma.*sin(alpha./sigma).*Q0+cos(alpha./sigma).*V0;

    tau20=atan(-V1star./Q1star);    
    tau20=tau20+pi*(tau20<0);
    tau2 = fzero(@(t) fun(t,Q1star,-V1star),mean(tau20));

    U=rand; alph=0;
    if U<(tau1/tau2)*(tau2>alpha)
        X1=Q1star;
        alph=1;
    else
        X1=Q0;
    end
    ap(oi)=alph;

    dt(oi)=alpha;
    SD(oi)=norm(X1-X0)^2;
    x_vec(oi)=X1(dim);

    if (oi==12884)
        t1=linspace(0,tau1,100);
        theta1=cos(t1./sigma(end)).*Q0(end)+sigma(end).*sin(t1./sigma(end)).*V0(end);
        rho1=-1./sigma(end).*sin(t1./sigma(end)).*Q0(end)+cos(t1./sigma(end)).*V0(end);
        t2=linspace(0,tau2,100);
        theta2=cos(t2./sigma(end)).*Q1star(end)-sigma(end).*sin(t2./sigma(end)).*V1star(end);
        rho2=-1./sigma(end).*sin(t2./sigma(end)).*Q1star(end)-cos(t2./sigma(end)).*V1star(end);

        figure(1); clf; hold on;
        plot(theta1,rho1,'k','LineWidth',2);
        plot(theta2,rho2,'k','LineWidth',2,'color',[0.75 0.75 0.75]);
        plot(Q0(end),V0(end),'ko','MarkerSize',12,'MarkerFaceColor','k');
        plot(Q1star(end),-V1star(end),'kv','MarkerSize',12,'MarkerFaceColor','k');
        plot(Q1star(end),V1star(end),'k^','MarkerSize',12,'MarkerFaceColor','k');
        set(gca,'FontSize',20);
        axis([-2 2 -2 2]);
        ylabel('$\rho^1$','FontSize',20,'Interpreter','latex');
        xlabel('$\theta^1$','FontSize',20,'Interpreter','latex');
        box on;
        grid on;
        set(gcf,'color',[1.0,1.0,1.0]);
        disp([alph tau1 alpha tau2])
        str = '$$(\theta_0^d,\rho_0^d)$$';
        text(Q0(end)-0.7,V0(end),str,'Interpreter','latex','FontSize',20,'backgroundcolor','w','edgecolor','k')
        str = '$$(\theta_{\alpha}^d,\rho_{\alpha}^d)$$';
        text(Q1star(end)-0.2,V1star(end)+0.35,str,'Interpreter','latex','FontSize',20,'backgroundcolor','w','edgecolor','k')
        str = '$$(\theta_{\alpha}^d,-\rho_{\alpha}^d)$$';
        text(Q1star(end)-0.2,-V1star(end)-0.35,str,'Interpreter','latex','FontSize',20,'backgroundcolor','w','edgecolor','k')
        break;
    end

    X0=X1;
end
toc

disp([mean(ap) mean(dt) mean(SD)])


%% graphical output
t1=linspace(0,tau1,100);
theta1=cos(t1./sigma(end)).*Q0(end)+sigma(end).*sin(t1./sigma(end)).*V0(end);
rho1=-1./sigma(end).*sin(t1./sigma(end)).*Q0(end)+cos(t1./sigma(end)).*V0(end);
t2=linspace(0,tau2,100);
theta2=cos(t2./sigma(end)).*Q1star(end)-sigma(end).*sin(t2./sigma(end)).*V1star(end);
rho2=-1./sigma(end).*sin(t2./sigma(end)).*Q1star(end)-cos(t2./sigma(end)).*V1star(end);

figure(1); hold on;
plot(Q0(end),V0(end),'ko','MarkerSize',10,'MarkerFaceColor','k');
plot(theta1,rho1,'k','LineWidth',2);
plot(theta2,rho2,'k','LineWidth',2,'color',[0.75 0.75 0.75]);
plot(Q0(end),V0(end),'ko','MarkerSize',10,'MarkerFaceColor','k');
plot(Q1star(end),-V1star(end),'kv','MarkerSize',10,'MarkerFaceColor','k');
plot(Q1star(end),V1star(end),'k^','MarkerSize',10,'MarkerFaceColor','k');

ylabel('$\rho^d$','FontSize',20,'Interpreter','latex');
xlabel('$\theta^d$','FontSize',20,'Interpreter','latex');
box on;
grid on;
set(gcf,'color',[1.0,1.0,1.0]);

filename=['GIST_angle_illustration_rejected.pdf'];
export_fig(gcf,filename,'-pdf');
