function [X1,V1,maxdH1] = vverlet2(X0,V0,num_steps,step_size)
%vverlet Summary of this function goes here
%   Detailed explanation goes here
force = @(x) -x;
hh=step_size; hh2=hh*hh;
H0=0.5*(V0^2+X0^2); maxdH1=0.0;
F0=force(X0);
for j=1:num_steps

    X1=X0+hh*V0+0.5*hh2*F0;
    F1=force(X1);
    V1=V0+0.5*hh*(F1+F0);
    dH1=0.5*(V1^2+X1^2)-H0; 
    maxdH1=max(maxdH1,dH1);
    X0=X1; V0=V1; F0=F1;

end
%H1=0.5*(V1^2+X1^2); % final energy
end