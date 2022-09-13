function score = fitness_SVM(x)
load('X_norm_clean.mat')
load('MLP.mat')
load('y_train')
load('kfold')
selected_features = X_norm_clean(unique(x), :);
acc = 0;
for i = 1:5
    indexTrain = training(c, i);
    indexValid = test(c, i);
    XTrain = selected_features(:, indexTrain);
    yTrain = y_train(indexTrain);
    XValid = selected_features(:, indexValid);
    yValid = y_train(indexValid);
    SVMModel = fitcsvm(XTrain',yTrain');
%     net = newrb(XTrain, yTrain, 0, 9, 25);
%     predict_y = net(XValid);
%     p_TrainY = net(XTrain);
%     [X,Y,T,~,OPTROCPT] = perfcurve(yTrain,p_TrainY,1);
%     Thr = T(X==OPTROCPT(1) & Y==OPTROCPT(2));
%     predict_y = predict_y >= Thr ;
%     acc = acc + sum(predict_y == yValid) / size(XValid, 2);
    label = predict(SVMModel, XValid');
    acc = acc + sum(label == yValid') / length(yValid);
end
acc = acc / 5;
score = -acc*100;
















