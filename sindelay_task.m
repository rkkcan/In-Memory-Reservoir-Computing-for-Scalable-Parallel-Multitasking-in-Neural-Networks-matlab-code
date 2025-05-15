clc;
clear;
% sindelay task

dx = 1;                          % Input dimension
dr = 50;                       % reservoir dimension
outputSize = 1;                  % Output dimension
zk=1e-6;
LE=30;  %Number of left movements
DO=30;  %Number of downward movements
RCscaled1=zeros(10,10);
RCscaled2=zeros(10,10);
RCrandom=zeros(10,10);

% Generate sindelay data
w=100000;
NRMSEKK=zeros(2,1);
k=3;
SS=(2*rand(1,w+k)-1);
kk=1;
ac = 1/kk;
s = kk*SS;
y=sin(ac*pi*s);
worm=20;
tra=10000;
tax=5000;
train=s(worm+1:worm+tra);
targetData=y(worm+1-k:worm+tra-k);
task=s(worm+tra+1:worm+tra+tax);
ttask=y(worm+tra+1-k:worm+tra+tax-k);
wormup=s(1:worm);

PNR1=zeros(1,64);
PNR2=zeros(1,64);
PNR3=zeros(1,64);

%readout mamba weights
for pu=0:63
    
    fileName = sprintf('layer%d.txt', pu);
    matrixData = readmatrix(fileName);
    Wdara=matrixData;

    for left=1:10
        
        for down=1:10

            %Generation Win
            Win =10*Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+2+LE*(left-1):2*dr+1+dx+LE*(left-1));
            b=Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+1+LE*(left-1));
            Wres1 = Wdara(1+DO*(down-1):dr+DO*(down-1),1+LE*(left-1):dr+LE*(left-1)) ;
            rho = max(abs(eig(Wres1)));
            Wres1=0.2*Wres1/rho;
            
            WinD=Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+2+LE*(left-1):2*dr+1+dx+LE*(left-1));
            bD=Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+1+LE*(left-1));
            Wres1D=Wdara(1+DO*(down-1):dr+DO*(down-1),1+LE*(left-1):dr+LE*(left-1)) ;
            
            WinDD=2*rand(dr, dx)-1;
            bDD=1*(2*rand(dr,1)-1);
            Wres1DD=(2*rand(dr, dr)-1);
            
            %generation Wout
            Wout1 = rand(outputSize, 2*dr) - 0.5;
            Wout2 = rand(outputSize, 2*dr) - 0.5;
            Woutc1 = 0.0*randn(outputSize, 2*dr);
            Woutc2 = 0.0*randn(outputSize, 2*dr);
            r=zeros(dr,1);
            r_=zeros(dr,1);
            rL=zeros(dr,1);
            rL_=zeros(dr,1);
            rLL=zeros(dr,1);
            rLL_=zeros(dr,1);
            
            RR=[];   %state
            RRL=[];
            RRNL=[];
            
            O=[];   %output
            OL=[];
            ONL=[];
            
            
            % worming up
            for i = 1:length(wormup)
                
                inputw = wormup(i);
                u=Win*inputw;
                uD=WinD*inputw;
                uDD=WinDD*inputw;
                r_=u + Wres1 * r+b;
                rL_=uD + Wres1D * rL+bD;
                rLL_=uDD + Wres1DD * rLL+bDD;
                r =tanh(r_);
                rL =tanh(rL_);
                rLL =tanh(rLL_);
                
                
            end
            
            
            % training
            for i = 1:length(train)
                
                inputr = train(i);
                
                u=Win*inputr;
                uD=WinD*inputr;
                uDD=WinDD*inputr;
                r_=u + Wres1 * r+b;
                rL_=uD + Wres1D * rL+bD;
                rLL_=uDD + Wres1DD * rLL+bDD;
                r =tanh(r_);
                rL =tanh(rL_);
                rLL =tanh(rLL_);
                
                RR(:,i)=[r_;r];
                RRL(:,i)=[rL_;rL];
                RRNL(:,i)=[rLL_;rLL];
                
            end
            
            II=eye(2*dr);
            III=eye(2*dr);
            IIII=eye(2*dr);
            
            %mamba weight
            RT=RR'*pinv(RR*RR'+zk*II);
            Wout1=targetData*RT;
            OTT=Wout1*RR;
            
            %scaled mamba weight
            RTL=RRL'*pinv(RRL*RRL'+zk*III);
            WoutL=targetData*RTL;
            OTTL=WoutL*RRL;
            
            %random
            RTLL=RRNL'*pinv(RRNL*RRNL'+zk*III);
            WoutLL=targetData*RTLL;
            OTTLL=WoutLL*RRNL;
            
            %test
            
            inputt=task(1);
            
            for i = 1:length(task)
                
                
                inputs=task(i);
                
                
                u=Win*inputs;
                uD=WinD*inputs;
                uDD=WinDD*inputs;
                r_=u + Wres1 * r+b;
                rL_=uD + Wres1D * rL+bD;
                rLL_=uDD + Wres1DD * rLL+bDD;
                r =tanh(r_);
                rL =tanh(rL_);
                rLL =tanh(rLL_);
                
                outputt = Wout1 * [r_;r];
                outputtL = WoutL * [rL_;rL];
                outputtLL = WoutLL * [rLL_;rLL];
                
                O(:,i)=outputt;
                OL(:,i)=outputtL;
                ONL(:,i)=outputtLL;
                
            end
            MSE1 = mean((  ttask-O).^2);
            EEE=sum(ttask.^2);
            EEEE=max(ttask)-min(ttask);
            % NRMSE mamba weight
            NRMSE1 = sqrt(MSE1) / EEEE;
            
            MSE2 = mean((  ttask-OL).^2);
            EEEM=sum(ttask.^2);
            EEEEM=max(ttask)-min(ttask);
            
            % NRMSE scaled mamba weight
            NRMSE2 = sqrt(MSE2) / EEEEM;
            
            MSE3 = mean((  ttask-ONL).^2);
            EEEMM=sum(ttask.^2);
            EEEEMM=max(ttask)-min(ttask);
            % NRMSE random weight
            NRMSE3 = sqrt(MSE3) / EEEEM;
            RCscaled1(left,down)=NRMSE1;
            RCscaled2(left,down)=NRMSE2;
            RCrandom(left,down)=NRMSE3;
            
        end
    end
    overall_mean1 = mean(RCscaled1(:));
    disp(['scaled: ', num2str(overall_mean1)]);
    
    overall_mean2 = mean(RCscaled2(:));
    disp(['RCrandom: ', num2str(overall_mean2)]);
    
    overall_mean3 = mean(RCrandom(:));
    disp(['ALLRCrandom: ', num2str(overall_mean3)]);
    
    PNR1(pu+1)=overall_mean1;
    PNR2(pu+1)=overall_mean2;
    PNR3(pu+1)=overall_mean3;
end
barColor1 = hex2rgb('#E6724B');
PNR3(31)=PNR3(30);
colors=[hex2rgb('#0072BD'); hex2rgb('#D95319'); hex2rgb('#ECAE18'); hex2rgb('#873E96'); hex2rgb('#75AB2D')];
plot(PNR2, 'LineWidth', 2 ,'MarkerSize', 4,'Color', colors(1,:), 'MarkerFaceColor', colors(1,:));
hold on; 
plot(PNR1, 'LineWidth', 2,'MarkerSize', 4,'Color', barColor1, 'MarkerFaceColor', barColor1);
hold on; 
plot(PNR3, 'LineWidth', 2,'MarkerSize', 4,'Color', [0.25 0.25 0.25], 'MarkerFaceColor', [0.25 0.25 0.25]);
hold on; 
ylim([0 0.8]);xlim([0 65])
