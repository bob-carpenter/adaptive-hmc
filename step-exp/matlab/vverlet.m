function [X1,V1,H1] = vverlet(X0,V0,num_steps,step_size)
%pverlet Summary of this function goes here
%   Detailed explanation goes here
force = @(x) -x;
hh=step_size; hh2=hh*hh;
F0=force(X0);
for j=1:num_steps

    X1=X0+hh*V0+0.5*hh2*F0;
    F1=force(X1);
    V1=V0+0.5*hh*(F1+F0);
    X0=X1; V0=V1; F0=F1;

end
H1=0.5*(V1^2+X1^2); % final energy
end