% Physical parameters
mb = 300;    % kg
mw = 60;     % kg
bs = 1000;   % N/m/s
ks = 16000 ; % N/m
kt = 190000; % N/m

% State matrices
Ac = [ 0 1 0 0; [-ks -bs ks bs]/mb ; ...
      0 0 0 1; [ks bs -ks-kt -bs]/mw];
Bc = [ 0 0; 0 10000/mb ; 0 0 ; [kt -10000]/mw];
Cc = [1 0 0 0; 1 0 -1 0; Ac(2,:)];
Dc = [0 0; 0 0; Bc(2,:)];

qcar = ss(Ac,Bc,Cc,Dc);
qcar.StateName = {'body travel (m)';'body vel (m/s)';...
          'wheel travel (m)';'wheel vel (m/s)'};
qcar.InputName = {'r';'fs'};
qcar.OutputName = {'xb';'sd';'ab'};

Ts = 0.01;
qcard=c2d(qcar,Ts);
A = qcard.A;
B = qcard.B(:,2);
F = qcard.B(:,1);
C1 = qcard.C(2:3,:);
Q = C1'*diag([10 1e-4])*C1;
rho = 1;
S = Q; R = rho;
N = 300;