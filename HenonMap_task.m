clc;
clear;

%hehhonmap tesk
dx = 1;   %Input dimension                    
dr = 10;  %Reservoir dimensions
zk=1e-6;  %Ridge regression parameters  
LE=10;  %Number of left movements
DO=10;  %Number of downward movements

% generation henonmap data
step = 10000;
data = HenonMap(2*step+1);
AllData = data;
wormup=AllData(1:19);
inputData =AllData(20:10000);
targetData =AllData(21:10001);
testData =AllData(10001:20001);
RCscaled1=zeros(10,10);
RCscaled2=zeros(10,10);
RCrandom=zeros(10,10);

PNR1=zeros(1,64);
PNR2=zeros(1,64);
PNR3=zeros(1,64);

%readout mamba weight
qwer=0:63;
for pu=0:63
    
    fileName = sprintf('layer%d.txt', pu);
    matrixData = readmatrix(fileName);
    Wdara=matrixData;
%     Wdara=load('origin1000.txt');
    for left=1:10
        
        for down=1:10
            
            Win =Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+2+LE*(left-1):2*dr+1+dx+LE*(left-1));
            b=Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+1+LE*(left-1));
            Wres1 = Wdara(1+DO*(down-1):dr+DO*(down-1),1+LE*(left-1):dr+LE*(left-1)) ;
            %scaling
            rho = max(abs(eig(Wres1)));
            Wres1=0.1*Wres1/rho;
            
            WinD=Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+2+LE*(left-1):2*dr+1+dx+LE*(left-1));
            bD=Wdara(1+DO*(down-1):dr+DO*(down-1),2*dr+1+LE*(left-1));
            Wres1D=Wdara(1+DO*(down-1):dr+DO*(down-1),1+LE*(left-1):dr+LE*(left-1)) ;
           
            WinDD=2*rand(dr, dx)-1;
            bDD=1*(2*rand(dr,1)-1);
            Wres1DD=(2*rand(dr, dr)-1);

            r=zeros(dr,1);
            r_=zeros(dr,1);
            rL=zeros(dr,1);
            rL_=zeros(dr,1);
            rLL=zeros(dr,1);
            rLL_=zeros(dr,1);
            
            %             RR=zeros(2*dr,length(inputData));
            %             RRL=zeros(2*dr,length(inputData));
            %
            %             O=zeros(2*dr,length(testData));
            %             OL=zeros(2*dr,length(testData));
            
            RR=[];   %state
            RRL=[];
            RRNL=[];
            
            O=[];   %output
            OL=[];
            ONL=[];
            % worming up
            for i = 1:length(wormup)
                
                inputw = wormup(:,i);
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
            for i = 1:length(inputData)
                
                inputr = inputData(:,i);
                
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
            
            %scaled mamba weight
            RT=RR'*pinv(RR*RR'+zk*II);
            Wout1=targetData*RT;
            OTT=Wout1*RR;
            
            %mamba weight
            RTL=RRL'*pinv(RRL*RRL'+zk*III);
            WoutL=targetData*RTL;
            OTTL=WoutL*RRL;
            
            %random weight
            RTLL=RRNL'*pinv(RRNL*RRNL'+zk*III);
            WoutLL=targetData*RTLL;
            OTTLL=WoutLL*RRNL;
            
            
        
            
            Winz=Win;
            WinDz=WinD;
            Wres1z=Wres1;
            Wres1Dz=Wres1D;
            
            testData =AllData(10000:20000);
            
            % test
            for i = 1:length(testData)
                
               inputs = testData(:,i);
                inputsL = testData(:,i);
                
                
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
            
            NRMSE1 = sqrt(mean((O(10:10000)-testData(11:10001)).^2)./var(testData(11:10001)));
            NRMSE2 = sqrt(mean((OL(10:10000)-testData(11:10001)).^2)./var(testData(11:10001)));
            NRMSE3 = sqrt(mean((ONL(10:10000)-testData(11:10001)).^2)./var(testData(11:10001)));
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
    disp(['random RCrandom: ', num2str(overall_mean3)]);
    
    PNR1(pu+1)=overall_mean1;
    PNR2(pu+1)=overall_mean2;
    PNR3(pu+1)=overall_mean3;
    
end
barColor1 = hex2rgb('#E6724B');

colors=[hex2rgb('#0072BD'); hex2rgb('#D95319'); hex2rgb('#ECAE18'); hex2rgb('#873E96'); hex2rgb('#75AB2D')];
plot(PNR2, 'LineWidth', 2 ,'MarkerSize', 4,'Color', colors(1,:), 'MarkerFaceColor', colors(1,:));
hold on; 
plot(PNR1, 'LineWidth', 2,'MarkerSize', 4,'Color', barColor1, 'MarkerFaceColor', barColor1);
hold on; 
plot(PNR3, 'LineWidth', 2,'MarkerSize', 4,'Color', [0.25 0.25 0.25], 'MarkerFaceColor', [0.25 0.25 0.25]);
hold on; 
ylim([0 0.8]);xlim([0 65])


