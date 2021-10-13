function gc_dynamics_1d(periodic,N)


%----------------------------------------------
%network parameters
m = 4;              %CV = 1/sqrt(m)
x_prefs = (1:N)'/N; %inherited location preferences (m)


%FF input
beta_vel = 1.5;     %velocity gain
beta_0 = 70;        %uniform input
alpha = 1000;        %weight imbalance parameter
gamma = 1.05/100;    %Cennter Surround weight params
beta = 1/100;        %Cennter Surround weight params

%temporal parameters

T = 100;            %length of integration time blocks (s)
dt = 1/2000;        %step size of numerical integration (s)
tau_s = 30/1000;    %synaptic time constant (s)


%Graphing parameters
bins = linspace(0+.01,1-.01,50);
scrsz = get(0,'ScreenSize');
figure('Position',[500 scrsz(4)/2 scrsz(3)*(3/4) scrsz(4)/2])

% Trajectory Data (Sinusoidal)
x = (sin((dt:dt:T)*2*pi/10)+1)/2;
v= zeros(1,T/dt);
for i = 2:T/dt
    v(i) = (x(i)-x(i-1))/dt;
end

z = (-N/2:1:N/2-1);

% Feed forward network input
if periodic == 1
    % gaussian FF input for aperiodic network
    envelope = exp(-4*(z'./(N/2)).^2);
else
    envelope = ones(N,1);
end


s_prev = zeros(2*N,1);  %Population activity
spk = zeros(2*N,T/dt);  %Total spiking
spk_count = zeros(2*N,1); %Current spiking

% Weight setup
crossSection = alpha*(exp(-gamma*z.^2)-exp(-beta*z.^2));
crossSection = circshift(crossSection, [0 N/2 - 1]);

W_RR = zeros(N,N);
W_LL = zeros(N,N);
W_RL = zeros(N,N);
W_LR = zeros(N,N);

for i = 1:N
    W_RR(i,:) =  circshift(crossSection,[0 i - 1]); % Right neurons to Right neurons
    W_LL(i,:) =  circshift(crossSection,[0 i + 1]); % Left neurons to Left neurons
    W_RL(i,:) =  circshift(crossSection,[0 i]);     % Left neurons to Right neurons
    W_LR(i,:) =  circshift(crossSection,[0 i]);     % Right neurons to Left neurons
end



for t = 2:T/dt


        %LEFT population
        v_L = (1 - beta_vel*v(t));
        g_LL = W_LL*s_prev(1:N);                     %L->L
        g_LR = W_LR*s_prev(N+1:2*N);                 %R->L
        G_L = v_L*((g_LL + g_LR) + envelope*beta_0);              %input conductance into Left population


        %RIGHT population
        v_R = (1 + beta_vel*v(t));
        g_RR = W_RR*s_prev(N+1:2*N);                 %R->R
        g_RL = W_RL*s_prev(1:N);                     %L->R
        G_R = v_R*((g_RR + g_RL) + envelope*beta_0);              %input conductance into Right population

        G = [G_L;G_R];
        F = zeros(2*N,1) + G.*(G>=0);   %linear transfer function

        % subdivide interval m times
        spk_sub = poissrnd(repmat(F,1,m)*dt);
        spk_count = spk_count+sum(spk_sub,2);
        spk(:,t) = floor(spk_count/m);
        spk_count = rem(spk_count,m);

        %update population activity
        s_new = s_prev + spk(:,t) - s_prev*dt/tau_s;
        s_prev = s_new;


        if (mod(t,100)==0)%plot every 100 steps
            subplot(2,2,1), plot(x_prefs,W_RR(:,N/2),'r'), hold on, plot(x_prefs,W_LL(:,N/2),'b'), hold off, title('Intra Connections')
            subplot(2,2,2), plot(x_prefs,F(1:N)), hold off,  title('Population Response')
            subplot(2,2,3), plot(x_prefs,exp(-(x_prefs - x(t)).^2/.001^2)/max(exp(-(x_prefs - x(t)).^2/.001^2)),'g'), hold off, title('Position')
            subplot(2,2,4), plot(bins,histc(x(1:t).*spk(N/2,1:t),bins)/dt./histc(x(1:t),bins)), title('SN Response')
            drawnow
        end

end


