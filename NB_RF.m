
%% Importing and formatting data

clear all
close all
clc

%import both csv file
mat = readtable('studentmat.csv');
por = readtable('studentpor.csv');

%combine both files
sp = [mat;por];

%catergoricals to nominals
names = mat.Properties.VariableNames;

[nrows, ncols] = size(sp);
category = false(1,ncols);
for i = 1:ncols
    if isa(sp.(names{i}),'cell') || isa(sp.(names{i}),'nominal')
        category(i) = true;
        sp.(names{i}) = grp2idx(sp.(names{i}));
    end
end

%% grouping continous attributes - continous to nominal 

for i = 1:nrows
    sp.G3(i) = ordinal(sp.G3(i),{'Fail','Pass'},...
                       [],[0,10,20]); 
    sp.G2(i) = ordinal(sp.G2(i),{'Fail','Pass'},...
                       [],[0,10,20]);    
    sp.G1(i) = ordinal(sp.G1(i),{'Fail','Pass'},...
                       [],[0,10,20]);                    
end

%selecting only social vars activities	romantic	famrel	freetime	goout	Dalc	Walc	health
spSOC = horzcat(sp(:,19),sp(:,23:29),sp(:,31:33));

%transforming table to array
spSOC1 = table2array(spSOC);

%% Basic stats

%Frequencies of each variable
disp('Activites frequencies (1 - yes, 2 - no)')
tabulate(spSOC.activities)
disp('romantic frequencies (1 - yes, 2 - no)')
tabulate(spSOC.romantic)
disp('famrel frequencies (from 1 - very bad to 5 - excellent)')
tabulate(spSOC.famrel)
disp('freetime frequencies (from 1 - very low to 5 - very high)')
tabulate(spSOC.freetime)
disp('goout frequencies (from 1 - very low to 5 - very high)')
tabulate(spSOC.goout)
disp('Dalc frequencies (from 1 - very low to 5 - very high)')
tabulate(spSOC.Dalc)
disp('Walc frequencies (from 1 - very low to 5 - very high)')
tabulate(spSOC.Walc)
disp('health frequencies (from 1 - very bad to 5 - very good)')
tabulate(spSOC.health)
disp('G1 frequencies (1 - fail, 2 - pass)')
tabulate(spSOC.G1)
disp('G2 frequencies (1 - fail, 2 - pass)')
tabulate(spSOC.G2)
disp('G3 frequencies (1 - fail, 2 - pass)')
tabulate(spSOC.G3)

%Correlation of variables
covmat = corrcoef(double(spSOC1));
figure
x = size(spSOC1, 2);
imagesc(covmat);
set(gca,'XTick',1:x);
set(gca,'YTick',1:x);
set(gca,'XTickLabel',spSOC.Properties.VariableNames);
set(gca,'YTickLabel',spSOC.Properties.VariableNames);
axis([0 x+1 0 x+1]);
grid;
colorbar;


%% creating test and training set
%Separating preditors and lables
X = double(spSOC1(:,1:10));
Y = double(spSOC1(:,11));
disp('Size of predictor matrix')
size(X)
disp('Size of class matrix')
size(Y)

%Partioning data into testing (10%) and training (90%) sets
c = cvpartition(Y,'holdout',.1);

%Create training and testing sets for predictors and labels
X_Train = X(training(c,1),:);
Y_Train = Y(training(c,1));
X_Test = X(test(c,1),:);
Y_Test = Y(test(c,1),:);

%Freq of classes in training and testing
disp('Training Set')
tabulate(Y_Train)
disp('Test Set')
tabulate(Y_Test)


%% Creating Naive Bayes model using 10 fold validation
rng(1);

%NB model applied to training set using multivariate multinomial dist
Nb = fitcnb(X_Train,Y_Train,'Distribution','mvmn')

%NB model cross validated using 10-fold - fuction default is 10-fold
CVNb = crossval(Nb)

%Estimating generalization error
disp('Error for each fold')
NbErIndv = kfoldLoss(CVNb,'mode','individual')
disp('Avearge error across all folds')
NbErAv = kfoldLoss(CVNb)

% Best model indentified
NbBM = CVNb.Trained{4}

%Access accuracy error for indiv fold
indiv = kfoldLoss(CVNb,'folds',4)
 
%Identified X model to have lowest error so will use this on test set
NbBMPred = NbBM.predict(X_Test);

%Confusion matrix - predicted class against actual class
disp('Confusion matrix')
[Nbcon, classorder] = confusionmat(Y_Test,NbBMPred);
Nbcon

%Confusion matrix for each class as a percentage of the true class
disp('Each class as % of true class')
NBconper = bsxfun(@rdivide,Nbcon,sum(Nbcon,2)) * 100

%Classification rate - accuracy 
disp('Accuracy NB')
NBAcc = trace(Nbcon)/sum(Nbcon(:))

%Misclassification rate - accuracy error
disp('Accuracy error NB')
NBError = 1 - trace(Nbcon)/sum(Nbcon(:))

%Precision - measure of classifiers exactness 
% number of True Positives divided by the number of True Positives and False Positives.
% low pre indicates large number of false positives - higher the better
disp('Precision NB')
NBpre = Nbcon(1,1)/(Nbcon(1,1)+Nbcon(1,2))

%Recall - measure of classifiers completeness
% number of True Positives divided by the number of True Positives and the number of False Negatives.
%low recall indicates many false negatives - higher the better
disp('Recall NB')
NBrec = Nbcon(1,1)/(Nbcon(1,1)+Nbcon(2,1))

%F1 score conveys the balance between the precision and the recall - higher the better
% 2*((precision*recall)/(precision+recall))
disp('F1 measure NB')
NBF1 = 2*((NBpre*NBrec)/(NBpre+NBrec))

% False positive rate
%how many we predicted to pass, actually failed 
%False positive can be more expesnive (cost)
disp('False positive rate NB')
NBFalse_positive_rate = (Nbcon(2,1))/ (Nbcon(1,2) + Nbcon(2,1))

%% Plotting k-fold errors
rng(1);
% empty set
errs = [];

%append each error to set
for i = 1:10
    new = kfoldLoss(CVNb,'folds',i);
    errs = [errs, new];
end

%final set - Y
errs

%number of folds - X
folds = [1:10];

figure
plot(folds,errs)
title 'K-th fold against accuracy error'
xlabel 'K-th fold'
ylabel 'Accuracy error'

%% Random forest
rng(1);

%Defining number of trees
ntrees = 150; 

%Bootstrapped-aggregated decision tree applied to training set
tic
b2 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'Method','classification'); 
T1_elasped = toc;
oobErrorBaggedEnsemble = oobError(b2);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of trees'
ylabel 'OObError'

%Predicting testing set using b1
%p function shows probability
[y_fit,p] = predict(b2,X_Test); 

%reformatting predictions 
d = cell2mat(y_fit);
RFpred= double(d);

%confusion matrix of predicited against actual classes
confusion_matrix = confusionmat(RFpred,Y_Test)

% Obtainging confusion matrix values
True_negative = confusion_matrix(3,1);
False_positive = confusion_matrix(4,1);
True_positive = confusion_matrix(4,2);
False_negative = confusion_matrix(3,2);

%Classification rate
disp('Classification accuracy RF')
Accuracy = (True_positive + True_negative)/ (True_negative + False_negative + True_positive + False_positive)

%Misclassification rate 
disp('Misclassification accuracy RF')
AccError = 1 - Accuracy

%Precision 
disp('Precision RF')
RFPrecision = (True_positive) / (True_positive + False_positive)

%Recall
disp('Recall RF')
RFRecall = (True_positive)/ (True_positive + False_negative)

%F-measure
disp ('F1 measure RF')
RFFmeasure = 2*((RFPrecision*RFRecall)/(RFPrecision+RFRecall))


%% ROC Curve for RF
rng(1);
% True positive rate
% how many we predicted to pass, acutally passed 
True_positive_rate = (True_positive)/ (True_positive + False_negative);  

% False positive rate
%how many we predicted to pass, actually failed 
%False positive can be more expesnive (cost)
False_positive_rate = (False_positive)/ (True_negative + False_positive); 

% Loop to change the cut off points 
cutoff_values = linspace(0,1,10);
for i = 1:10
    cut_value1 = double((p(:,2) >= cutoff_values(i)) +1);
    confusion_matrix1 = confusionmat(Y_Test,cut_value1);
    %Accuracy for cut of value 
    True_negative_1 = confusion_matrix1(1,1);
    False_positive_1 = confusion_matrix1(1,2);
    True_positive_1 = confusion_matrix1(2,2);
    False_negative_1 = confusion_matrix1(2,1);

    Accuracy_1 = (True_positive_1 + True_negative_1)/ (True_negative_1 + False_negative_1 + True_positive_1 + False_positive_1);
    % True positive rate value 
    table(i, 1) = (True_positive_1)/ (True_positive_1 + False_negative_1);
    % False positive rate value 
    table(i,2) = (False_positive_1)/ (True_negative_1 + False_positive_1);    
end

%ROC curve 
plot(table(:,2), table(:,1))
ylabel('Percentage of True Positive')
xlabel('Percentage of False Positive')


%% Evaluating number of predictors against accuracy error
% 1 predictor
tic
b1 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',1,...
    'Method','classification');
b1_elasped = toc;
oobErrorBaggedEnsemble_b1 = oobError(b1);

% 2 predictors
tic
b2 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',2,...
    'Method','classification');
b2_elasped = toc;
oobErrorBaggedEnsemble_b2 = oobError(b2);

% 3 predictors
tic
b3 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',3,...
    'Method','classification');
b3_elasped = toc;
oobErrorBaggedEnsemble_b3 = oobError(b3);

% 4 predictors
tic
b4 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',4,...
    'Method','classification');
b4_elasped = toc;
oobErrorBaggedEnsemble_b4 = oobError(b4);

% 5 predictors
tic
b5  = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',5,...
    'Method','classification');
b5_elasped = toc;
oobErrorBaggedEnsemble_b5 = oobError(b5);

% 6 predictors
tic
b6 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',6,...
    'Method','classification');
b6_elasped = toc;
oobErrorBaggedEnsemble_b6 = oobError(b6);

% 7 predictors
tic
b7 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',7,...
    'Method','classification');
b7_elasped = toc;
oobErrorBaggedEnsemble_b7 = oobError(b7);

% 8 predictors
tic
b8 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',8,...
    'Method','classification');
b8_elasped = toc;
oobErrorBaggedEnsemble_b8 = oobError(b8);

% 9 predictors
tic
b9 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',9,...
    'Method','classification');
b9_elasped = toc;
oobErrorBaggedEnsemble_b9 = oobError(b9);

% 10 predictors
tic
b10 = TreeBagger(ntrees,X_Train,Y_Train,'OOBPrediction','On',...
    'NumPredictorsToSample',10,...
    'Method','classification');
b10_elasped = toc;
oobErrorBaggedEnsemble_b10 = oobError(b10);

% joining tables
table1 = oobErrorBaggedEnsemble_b1; %n=1
table2 = oobErrorBaggedEnsemble_b2; %n=2
table3 = oobErrorBaggedEnsemble_b3; %n = 3
table4 =oobErrorBaggedEnsemble_b4;% n=4
table5 =oobErrorBaggedEnsemble_b5;% n=4
table6 = oobErrorBaggedEnsemble_b6;%n=6
table7 = oobErrorBaggedEnsemble_b7;%n=7
table8 = oobErrorBaggedEnsemble_b8;%n=8
table9 = oobErrorBaggedEnsemble_b9;% n= 9
table10 = oobErrorBaggedEnsemble_b10;%n=10

%table of the number of predictors
joint_table = [table1 table2 table3 table4 table5 table6 table7 table8 table9 table10]; 

% plotting number of errors against the number of trees and features
figure()
mesh(joint_table) 
xlabel('Number of features')
zlabel('oobError')
ylabel('Number of trees')

%% Minimum leaf size against time
%Per tree leaf this would give the minimum number of observation
rng(1);

for t = 1:100 % t= the number of minium leaf 
    tic
    Min_leaf1 = TreeBagger(50,X_Train,Y_Train,'OOBPrediction','On',...
    'Method','classification', 'MinLeafSize',t);
    leaf_time(t) = toc;   
end

rng(1)
figure()
plot(leaf_time)
title 'Min leaf size against time'
xlabel 'Min leaf size'
ylabel 'Time taken (seconds)'

% conclude leave the min leaf size at 1 because there is not much
% difference in size













