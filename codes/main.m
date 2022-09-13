%% load data
clear; close all; clc;
load("All_data.mat");
fs = 100;
%% histogram
% extract min and max voltage over all trials to use for bins in hist
max_amp = -inf; min_amp = inf;
for i = 1:316
    if max(x_train(:, :, i), [], "all") > max_amp; max_amp = max(x_train(:, :, i), [], "all"); end
    if min(x_train(:, :, i), [], "all") < min_amp; min_amp = min(x_train(:, :, i), [], "all"); end
end

p_index = find(y_train == 1);
n_index = find(y_train == 0);

numBins = 2:30;
J = zeros(size(numBins));
count = 1;
for nbins = numBins
% nbins = 20;
    bins = linspace(min_amp, max_amp, nbins);
    X_hist = zeros(316, 28*nbins);
    for i = 1:316
        for j = 1:28
            X_hist(i, 1+(j-1)*nbins:j*nbins) = hist(x_train(:, j, i), bins); %#ok<HIST> 
        end
    end
    mu0 = mean(X_hist);
    mu1 = mean(X_hist(p_index, :));
    mu2 = mean(X_hist(n_index, :));
    S1 = cov(X_hist(p_index, :));
    S2 = cov(X_hist(n_index, :));
    Sb = (mu1-mu0)*(mu1-mu0)' + (mu2-mu0)*(mu2-mu0)';
    Sw = S1 + S2;
    J(count) = trace(Sb) / trace(Sw);
    count = count + 1;
end
% plot(numBins, J)
% title('fisher score for different number of bins', 'Interpreter','latex')
% xlabel('number of bins', 'Interpreter','latex')
% ylabel('fisher score', 'Interpreter','latex')

[~, idx] = max(J);
nbins = numBins(idx);
bins = linspace(min_amp, max_amp, nbins);
X_hist = zeros(316, 28*nbins);
for i = 1:316
    for j = 1:28
        X_hist(i, 1+(j-1)*nbins:j*nbins) = hist(x_train(:, j, i), bins); %#ok<HIST> 
    end
end
%% form factor
X_FF = zeros(316, 28);
for i = 1:316
    for j = 1:28
        s = x_train(:, j, i);
        s_dot = diff(s);
        s_ddot = diff(s, 2);
        X_FF(i, j) = (var(s_ddot)/var(s_dot)) / (var(s_dot)/var(s));
    end
end
%% correlation
% 28 choose 2 -> 378
X_corr = zeros(316, 378);
for i = 1:316
    rho = corr(x_train(:, :, i));
    count = 1;
    for m = 1:28
        for n = 1:28
            if n <= m; continue; end
            X_corr(i, count) = rho(m, n);
            count = count + 1;
        end
    end
end
%% frequency characteristics
X_freq = zeros(316, 2*28);
for i = 1:316
    for j = 1:28
        signal = x_train(:, j, i)';
        X_freq(i, 1+(j-1)*2:2*j) = [meanfreq(signal, fs), medfreq(signal, fs)];
    end
end
%% CSP
x_train_zero_mean = zeros(size(x_train));
for i = 1:316
    x_train_zero_mean(:, :, i) = x_train(:, :, i) - mean(x_train(:, :, i)); 
end
C0 = zeros(28, 28);
C1 = zeros(28, 28);
for i = p_index
    s = x_train(:, :, i)';
    C1 = C1 + s*s'/trace(s*s');
end
for i = n_index
    s = x_train(:, :, i)';
    C0 = C0 + s*s'/trace(s*s');
end
[V, D] = eig(C1, C0);
[d, index] = sort(diag(D), "descend");
V = V(:, index);
F = 14;
W_CSP = [V(:, 1:F), V(:, end-F+1:end)];

X_CSP = zeros(316, 2*F);
for i = 1:316
    s = x_train(:, :, i)';
    Y = W_CSP'*s;
    X_CSP(i, :) = var(Y, [], 2)';
end
%% obw
X_obw = zeros(316, 28);
for i = 1:316
    for j = 1:28
        signal = x_train(:, j, i);
        X_obw(i, j) = obw(signal, fs);
    end
end
%% EEG toolbox
featureNames = ["min", "max", "ar", "md", "var", "sd", "am", "re", "le", "sh", "te", ...
                "lrssv", "mte", "me", "mcl", "n2d", "2d", "n1d", "1d", "kurt", "skew", ...
                "hc", "hm", "ha", "bpd", "bpt", "bpa", "bpb", "bpg", "rba"];

opts.alpha = 2;
opts.fs = fs;
opts.order = 4;
X_toolbox = zeros(316, 924);

count2 = 0;
for k = 1:30
    k %#ok<NOPTS> 
    for i = 1:316
        count = 1;
        for j = 1:28
            signal = x_train(:, j, i);
            if k == 3
                X_toolbox(i, count2 + count:count2+count+3) = jfeeg(featureNames(k), signal, opts);
                count = count + 4;
            else
                X_toolbox(i, count2 + count) = jfeeg(featureNames(k), signal, opts);
                count = count + 1;
            end
        end
    end
    if k == 3
        count2 = count2 + 4*28;
    else
        count2 = count2 + 28;
    end
end
%% total data
X = [X_hist, X_FF, X_corr, X_freq, X_CSP, X_obw, X_toolbox];
X = X';
y = y_train;
%% normalize data
[X_norm,PS] = mapstd(X);
%% clean data
% clean rows with NaNs and infs and zeros
[nanrows, ~] = find(isnan(X_norm));
[infrows, ~] = find(isinf(X_norm));
zerorows = [];
count = 1;
for i = 1:1554
    if all(X_norm(i, :) == 0)
        zerorows(count) = i; %#ok<SAGROW> 
        count = count + 1;
    end
end
junkrows = unique([nanrows; infrows; zerorows']);
idx = 1:length(X_norm);
idx(junkrows) = [];
X_norm_clean = X_norm(idx, :);
%% compute fisher scores 
clear; close all; clc;
load("X_norm_clean.mat")
load('labels.mat')
p_index = find(y == 1);
n_index = find(y == 0);

mu0 = mean(X_norm_clean, 2);
mu1 = mean(X_norm_clean(:, p_index), 2);
mu2 = mean(X_norm_clean(:, n_index), 2);
sigma1 = std(X_norm_clean(:, p_index), [], 2);
sigma2 = std(X_norm_clean(:, n_index), [], 2);
J = zeros(1, size(X_norm_clean, 1));
for i = 1:size(X_norm_clean, 1)
    J(i) = (norm(mu0(i)-mu1(i))^2 + norm(mu0(i)-mu2(i))^2) / (sigma1(i)^2 + sigma2(i)^2);
end
%%
nfeatures = 28*ones(1, 36);
nfeatures([1, 3, 4, 9, 15, 16]) = [84, 378, 56, 112, 2, 2];
figure();
bar(J)
% hold on
% start_idx = 1;
% for i = 1:length(nfeatures)
%     end_idx = start_idx + nfeatures(i) - 1;
%     line([end_idx, end_idx], [0, 0.07], 'Color', 'red')
%     start_idx = end_idx + 1;
% end
%%

best_features = find(J >= 0.03);
numN = 10:80;
acc = zeros(size(numN));
count = 1;
c = cvpartition(316, 'KFold', 5);
for N = numN
    for i = 1:5
        indexTrain = training(c, i);
        indexValid = test(c, i);
        XTrain = X_norm_clean(best_features, indexTrain);
        yTrain = y(indexTrain);
        XValid = X_norm_clean(best_features, indexValid);
        yValid = y(indexValid);
        net = patternnet(N);
        net = train(net,XTrain,yTrain);
        predict_y = net(XValid);
        p_TrainY = net(XTrain);
        [X,Y,T,~,OPTROCPT] = perfcurve(yTrain,p_TrainY,1);
        Thr = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
        predict_y = predict_y >= Thr ;
        acc(count) = acc(count) + sum(predict_y == yValid) / size(XValid, 2);
    end
    acc(count) = acc(count) / 5;
    count = count + 1;
end
%% train again
% load("MLP.mat")
[bestacc, idx] = max(acc);
bestN = numN(idx);
MLP = patternnet(bestN);
MLP = train(MLP, X_norm_clean(best_features, :), y);
yhat = MLP(X_norm_clean(best_features, :));
[X,Y,T,~,OPTROCPT] = perfcurve(y,yhat,1);
Thr_MLP = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
yhat = yhat >= Thr_MLP;  
acc1 = sum(yhat == y)/316;  
%% extract features for test data
% histogram
X_hist_test = zeros(100, 28*nbins);
for i = 1:100
    for j = 1:28
        X_hist_test(i, 1+(j-1)*nbins:j*nbins) = hist(x_test(:, j, i), bins); %#ok<HIST> 
    end
end
% form factor
X_FF_test = zeros(100, 28);
for i = 1:100
    for j = 1:28
        s = x_test(:, j, i);
        s_dot = diff(s);
        s_ddot = diff(s, 2);
        X_FF_test(i, j) = (var(s_ddot)/var(s_dot)) / (var(s_dot)/var(s));
    end
end
% correlation
X_corr_test = zeros(100, 378);
for i = 1:100
    rho = corr(x_test(:, :, i));
    count = 1;
    for m = 1:28
        for n = 1:28
            if n <= m; continue; end
            X_corr_test(i, count) = rho(m, n);
            count = count + 1;
        end
    end
end
% frequency characteristics
X_freq_test = zeros(100, 2*28);
for i = 1:100
    for j = 1:28
        signal = x_test(:, j, i)';
        X_freq_test(i, 1+(j-1)*2:2*j) = [meanfreq(signal, fs), medfreq(signal, fs)];
    end
end
% CSP
X_CSP_test = zeros(100, 2*F);
for i = 1:100
    s = x_test(:, :, i)';
    Y = W_CSP'*s;
    X_CSP_test(i, :) = var(Y, [], 2)';
end
% obw
X_obw_test = zeros(100, 28);
for i = 1:100
    for j = 1:28
        signal = x_test(:, j, i);
        X_obw_test(i, j) = obw(signal, fs);
    end
end
% EEG toolbox
featureNames = ["min", "max", "ar", "md", "var", "sd", "am", "re", "le", "sh", "te", ...
                "lrssv", "mte", "me", "mcl", "n2d", "2d", "n1d", "1d", "kurt", "skew", ...
                "hc", "hm", "ha", "bpd", "bpt", "bpa", "bpb", "bpg", "rba"];

opts.alpha = 2;
opts.fs = fs;
opts.order = 4;
X_toolbox_test = zeros(100, 924);

count2 = 0;
for k = 1:30
    k %#ok<NOPTS> 
    for i = 1:100
        count = 1;
        for j = 1:28
            signal = x_test(:, j, i);
            if k == 3
                X_toolbox_test(i, count2 + count:count2+count+3) = jfeeg(featureNames(k), signal, opts);
                count = count + 4;
            else
                X_toolbox_test(i, count2 + count) = jfeeg(featureNames(k), signal, opts);
                count = count + 1;
            end
        end
    end
    if k == 3
        count2 = count2 + 4*28;
    else
        count2 = count2 + 28;
    end
end
% total data
X_test = [X_hist_test, X_FF_test, X_corr_test, X_freq_test, X_CSP_test, X_obw_test, X_toolbox_test];
X_test = X_test';
% normalize data
X_norm_test = mapstd('apply',X_test, PS);
% clean data
% clean rows with NaNs and infs and zeros
X_test_norm_clean = X_norm_test(idx, :);
%% predict label of test data
% select best features
XTest = X_test_norm_clean(best_features, :);
yhat_Test_MLP = MLP(XTest);
yhat_Test_MLP = yhat_Test_MLP >= Thr_MLP;  
%% rbf
acc = zeros(1, 25);
spreadMat = [5, 6, 7, 8, 9] ;
NMat = [5,10,15,20,25] ;
c = cvpartition(316, 'KFold', 5);
count = 1;
for s = 1:5
    spread = spreadMat(s) ;
    for n = 1:5 
        Maxnumber = NMat(n) ;
        for i=1:5
            indexTrain = training(c, i);
            indexValid = test(c, i);
            XTrain = X_norm_clean(best_features, indexTrain);
            yTrain = y(indexTrain);
            XValid = X_norm_clean(best_features, indexValid);
            yValid = y(indexValid);
            net = newrb(XTrain,yTrain,10^-5,spread,Maxnumber) ;
            predict_y = net(XValid);
            p_TrainY = net(XTrain);
            [X,Y,T,~,OPTROCPT] = perfcurve(yTrain,p_TrainY,1);
            Thr = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
            predict_y = predict_y >= Thr ;
            acc(count) = acc(count) + sum(predict_y == yValid) / size(XValid, 2);
        end
        acc(count) = acc(count) / 5;
        count = count + 1;
    end
end
%% train again
% load("RBF.mat")
bestSpread = 9;
bestNMat = 25;
RBF = newrb(X_norm_clean(best_features, :), y_train, 10^-5, bestSpread, bestNMat);
yhat = RBF(X_norm_clean(best_features, :));
[X,Y,T,~,OPTROCPT] = perfcurve(y,yhat,1);
Thr_RBF = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
yhat = yhat >= Thr_RBF;
acc = sum(yhat == y)/316;
%% predict test labels
yhat_test_RBF = RBF(XTest);
yhat_test_RBF = yhat_test_RBF >= Thr_RBF;
%% Genetic Algorithm
clear; close all; clc;
load('X_norm_clean.mat')
load('X_test_norm_clean.mat')
load('All_data.mat')
% best_features = [812, 740, 814, 626, 620, 1175, 687, 1103, 365, 798, 1408, 680, 482, 651, 583];
best_features = unique([1408, 17, 437, 454, 1347, 1320, 852, 933, 810, 764, 1058, 255, 966, ...
                 1097, 1301, 764, 962, 137, 963, 846, 506, 765, 249, 967, 1189, 917, ...
                 141, 706, 1282, 380]);
%% hyperparameter tuning
numN = 10:80;
acc = zeros(size(numN));
count = 1;
c = cvpartition(316, 'KFold', 5);
for N = numN
    for i = 1:5
        indexTrain = training(c, i);
        indexValid = test(c, i);
        XTrain = X_norm_clean(best_features, indexTrain);
        yTrain = y_train(indexTrain);
        XValid = X_norm_clean(best_features, indexValid);
        yValid = y_train(indexValid);
        net = patternnet(N);
        net = train(net,XTrain,yTrain);
        predict_y = net(XValid);
        p_TrainY = net(XTrain);
        [X,Y,T,~,OPTROCPT] = perfcurve(yTrain,p_TrainY,1);
        Thr = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
        predict_y = predict_y >= Thr ;
        acc(count) = acc(count) + sum(predict_y == yValid) / size(XValid, 2);
    end
    acc(count) = acc(count) / 5;
    count = count + 1;
end
[bestacc, idx] = max(acc);
%% train again
bestN = numN(idx);
MLP = patternnet(bestN);
MLP = train(MLP, X_norm_clean(best_features, :), y_train);
yhat = MLP(X_norm_clean(best_features, :));
[X,Y,T,~,OPTROCPT] = perfcurve(y_train,yhat,1);
Thr_MLP = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
yhat = yhat >= Thr_MLP;  
acc1 = sum(yhat == y_train)/316;  
%% predict labels
XTest = X_test_norm_clean(best_features, :);
yhat_Test_MLP_GA = MLP(XTest);
yhat_Test_MLP_GA = yhat_Test_MLP_GA >= Thr_MLP;
%% hyperparameter tuning
acc = zeros(1, 25);
spreadMat = [5, 6, 7, 8, 9] ;
NMat = [5,10,15,20,25] ;
c = cvpartition(316, 'KFold', 5);
count = 1;
for s = 1:5
    spread = spreadMat(s) ;
    for n = 1:5 
        Maxnumber = NMat(n) ;
        for i=1:5
            indexTrain = training(c, i);
            indexValid = test(c, i);
            XTrain = X_norm_clean(best_features, indexTrain);
            yTrain = y_train(indexTrain);
            XValid = X_norm_clean(best_features, indexValid);
            yValid = y_train(indexValid);
            net = newrb(XTrain,yTrain,10^-5,spread,Maxnumber) ;
            predict_y = net(XValid);
            p_TrainY = net(XTrain);
            [X,Y,T,~,OPTROCPT] = perfcurve(yTrain,p_TrainY,1);
            Thr = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
            predict_y = predict_y >= Thr ;
            acc(count) = acc(count) + sum(predict_y == yValid) / size(XValid, 2);
        end
        acc(count) = acc(count) / 5;
        count = count + 1;
    end
end
bestacc = max(acc);
%% train again
bestSpread = 8;
bestNMat = 20;
RBF = newrb(X_norm_clean(best_features, :), y_train, 10^-5, bestSpread, bestNMat);
yhat = RBF(X_norm_clean(best_features, :));
[X,Y,T,~,OPTROCPT] = perfcurve(y_train,yhat,1);
Thr_RBF = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
yhat = yhat >= Thr_RBF;
acc = sum(yhat == y_train)/316;
%% predict labels
yhat_test_RBF_GA = RBF(XTest);
yhat_test_RBF_GA = yhat_test_RBF_GA >= Thr_RBF;













