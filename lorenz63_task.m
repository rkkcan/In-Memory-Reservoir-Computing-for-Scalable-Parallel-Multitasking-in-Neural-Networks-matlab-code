clc;
clear;
%lorenz63 task

dx = 3;                          %Input dimension           
dr = 100;                      %Reservoir dimensions
outputSize = 1;                 
a = 0.9;                         
C=0.1;
zk=1e-6;

%mamba weight
Wdara=load('origin1000q.txt');



%mamba weight
WinL =Wdara(1:dr,1:dx);
%random weight, the task will collapse
%WresL = randn(dr,dr) ;
WresL = Wdara(1:dr,dx+1:dx+dr) ;
bL=Wdara(1:dr,dx+dr+1:dx+dr+1);

%scaled mamba weight
Win =2*WinL;
Wres1 = Wdara(1:dr,dx+1:dx+dr) ;
rho = max(abs(eig(Wres1)));
Wres1 = 0.8*Wres1 / rho;
b=5*bL;

Wout1 = rand(outputSize, dr) - 0.5;


r=zeros(dr,1);
rL=zeros(dr,1);
rNL=zeros(dr,1);

%generation lorenz63 data
sigma = 10;
beta = 8/3;
rho = 28;
dt = 0.025;
T = 700;

t = 0:dt:T;
N = length(t);
x = zeros(1, N);
y = zeros(1, N);
z = zeros(1, N);

x(1) = 1;
y(1) = 1;
z(1) = 1;

% Four order Runge-Kutta
for i = 1:N-1
    k1 = dt * [sigma*(y(i)-x(i)); rho*x(i)-y(i)-x(i)*z(i); x(i)*y(i)-beta*z(i)];
    k2 = dt * [sigma*((y(i)+0.5*k1(2))-(x(i)+0.5*k1(1))); rho*(x(i)+0.5*k1(1))-(y(i)+0.5*k1(2))-(x(i)+0.5*k1(1))*(z(i)+0.5*k1(3)); (x(i)+0.5*k1(1))*(y(i)+0.5*k1(2))-beta*(z(i)+0.5*k1(3))];
    k3 = dt * [sigma*((y(i)+0.5*k2(2))-(x(i)+0.5*k2(1))); rho*(x(i)+0.5*k2(1))-(y(i)+0.5*k2(2))-(x(i)+0.5*k2(1))*(z(i)+0.5*k2(3)); (x(i)+0.5*k2(1))*(y(i)+0.5*k2(2))-beta*(z(i)+0.5*k2(3))];
    k4 = dt * [sigma*((y(i)+k3(2))-(x(i)+k3(1))); rho*(x(i)+k3(1))-(y(i)+k3(2))-(x(i)+k3(1))*(z(i)+k3(3)); (x(i)+k3(1))*(y(i)+k3(2))-beta*(z(i)+k3(3))];
    
    x(i+1) = x(i) + (k1(1)+2*k2(1)+2*k3(1)+k4(1))/6;
    y(i+1) = y(i) + (k1(2)+2*k2(2)+2*k3(2)+k4(2))/6;
    z(i+1) = z(i) + (k1(3)+2*k2(3)+2*k3(3)+k4(3))/6;
end
LOZ(1,:)=x;
LOZ(2,:)=y;
LOZ(3,:)=z;
meanL = mean(LOZ');  
MLOZ=LOZ-meanL';
varL=std(MLOZ');
MLOZ(1,:)=MLOZ(1,:)/varL(1);
MLOZ(2,:)=MLOZ(2,:)/varL(2);
MLOZ(3,:)=MLOZ(3,:)/varL(3);

AllData =MLOZ;
wormup=AllData(:,1:200);
inputData =AllData(:,201:10000);
targetData =AllData(:,202:10001);


output=[];
RR=[];   %state
RRL=[];
RRNL=[];

O=[];   %output
OL=[];
ONL=[];
% worming up
for i = 1:length(wormup)
    
    inputw = wormup(:,i);
    inputwL = wormup(:,i);
    
    u=Win*inputw;
    uL=WinL*inputwL;
    
    r_=u + Wres1 * r+b;
    rL_=uL + WresL * rL+bL;
    
    r =tanh(r_);
    rL=tanh(rL_);
   
    
end


% training
for i = 1:length(inputData)
    
    
    inputr = inputData(:,i);
    
    u=Win*inputr;
    uL=WinL*inputr;
    
    r_=u + Wres1 * r+b;
    rL_=uL + WresL * rL+bL;
    
    r =tanh(r_);
    rL=tanh(rL_);
    
    
    RR(:,i)=r;
    RRL(:,i)=rL;
    
end

II=eye(dr);
III=eye(dr);
%RC
RT=RR'*pinv(RR*RR'+zk*II);
Wout1=targetData*RT;
OTT=Wout1*RR;

%MCRC
RTL=RRL'*pinv(RRL*RRL'+zk*III);
WoutL=targetData*RTL;
OTTL=WoutL*RRL;

% test
testData =AllData(:,10002:20000);


outputt=Wout1*r;
outputtL=WoutL*rL;



for i = 1:length(testData)
    
    inputs = outputt;
    inputsL = outputtL;
    
    u=Win*inputs;
    uL=WinL*inputsL;
    
    r_=u + Wres1 * r+b;
    rL_=uL + WresL * rL+bL;
    
    rL=tanh(rL_); 
    r =tanh(r_);
    
    outputt = Wout1 * r;
    outputtL = WoutL * rL;
    
    O(:,i)=outputt;
    OL(:,i)=outputtL;
    
end


KKq=1:1:792;
figure('Position', [100 100 2000 200]); % Set window position and size
plot(O(1,1:721),  '--r', 'linewidth', 2);
hold on;
plot(OL(1,1:721),'--','Color', [0 0.7 0.8], 'linewidth',2);
plot(testData(1,1:792), 'k', 'linewidth',1); % Set the range of the x-axis
ylim([-2.5 2.5]); % Set the range of the y-axis to be the same as or similar to the range of the x-axis. Get the current tick positions.
xlim([0 721]);
xticks = 0:40:721; %Start from 0, setting a scale every 100. 
set(gca, 'XTick', xticks); % Set the position of the x-axis ticks

xticks_pos = get(gca, 'XTick');  
yticks_pos = get(gca, 'YTick');  
  
% 计算新的刻度标签值（乘以0.025）  
xticks_labels = arrayfun(@(x) num2str(x*0.025), xticks_pos, 'UniformOutput', false);  
% yticks_labels = arrayfun(@(y) num2str(y*0.025), yticks_pos, 'UniformOutput', false);  
  


% 设置新的刻度标签  
set(gca, 'XTickLabel', xticks_labels);  
% set(gca, 'YTickLabel', yticks_labels);
figure
plot3(testData(1,1:5000),testData(2,1:5000),testData(3,1:5000),'k')
figure
plot3(O(1,1:5000),O(2,1:5000),O(3,1:5000),'r')
figure
plot3(OL(1,1:5000),OL(2,1:5000),OL(3,1:5000),'Color', [0 0.7 0.8])
xlabel('x');  
ylabel('y');  
zlabel('z');  
  
% 设置坐标轴刻度  
set(gca, 'XTick', -2:1:2, 'YTick', -2:1:2, 'ZTick', -2:1:2); 
view(45, 20);