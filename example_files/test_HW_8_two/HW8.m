% William Walsh
% CPTS 534 - Homework Comparison
% HW4 (Linear Regression) vs. WEKA MLP Classifier
% 10/07/2025
clear; clc;
disp("++++ Program Start ++++")
disp("Comparing HW4 Linear Regression vs. WEKA MLP")






%% === HW4 Confusion Matrix (row-specific, true class on rows) ===
% Classes: 1=AB, 2=M, 6=PR
% Main script
Z = csvread('glass_data_HW6.csv');      % Columns 1-9 = features, col 10 = labels

% >>> first set
disp('-- from homework 4')
bin = [2, 1.5, 4];   % [defaultBin, thresholdLow, thresholdHigh]
fprintf('Bin limits used [%g, %g, %g] \n', bin(1),bin(2),bin(3))
HW6_classify(Z,bin);

[conf_HW4_alt,R2,totalClass,accClass] = HW6_classify(Z,bin);
fprintf('Coefficient of determination (R^2): %.4f\n', R2)
fprintf('\nConfusion Matrix:\n')
fprintf('                                   True class\n')
fprintf('                                %6s %6s %6s\n\n', '1','2','6')
fprintf('predictors assigned to class 1  %6d %6d %6d\n', conf_HW4_alt(1,1), conf_HW4_alt(1,2), conf_HW4_alt(1,3))
fprintf('predictors assigned to class 2  %6d %6d %6d\n', conf_HW4_alt(2,1), conf_HW4_alt(2,2), conf_HW4_alt(2,3))
fprintf('predictors assigned to class 6  %6d %6d %6d\n\n', conf_HW4_alt(3,1), conf_HW4_alt(3,2), conf_HW4_alt(3,3))

fprintf('Records per class: Class1=%d, Class2=%d, Class6=%d\n', totalClass(1,:), totalClass(2,:), totalClass(6,:));

fprintf('Accuracy Class 1: %.3f\n', accClass(1))
fprintf('Accuracy Class 2: %.3f\n', accClass(2))
fprintf('Accuracy Class 6: %.3f\n', accClass(6))
fprintf('Overall Accuracy: %.3f\n\n', accClass(7))

disp('-- end of homework 4')
%conf_HW4 = [43 26 0;    % True AB
%             19 57 0;    % True M
%              0  2 27];  % True PR

%records_per_class = [69 76 29];  % Total per true class
records_per_class = [totalClass(1,:), totalClass(2,:), totalClass(6,:)];
fprintf('records_per_class = [%g %g %g]\n\n', totalClass(1,:), totalClass(2,:), totalClass(6,:))
conf_HW4 = conf_HW4_alt;  % Assign confusion matrix from HW4 classification
%% === WEKA MLP Confusion Matrix ===
conf_MLP = [54 15 0;    % True AB
             16 58 2;    % True M
              2  2 25];  % True PR

%% === Compute per-class and overall accuracies ===
% Per-class accuracy = diagonal / row total
acc_HW4 = diag(conf_HW4) ./ records_per_class';
acc_MLP = diag(conf_MLP) ./ records_per_class';

% Overall accuracy = sum of diagonal / total records
overall_HW4 = sum(diag(conf_HW4)) / sum(records_per_class);
overall_MLP = sum(diag(conf_MLP)) / sum(records_per_class);

%% === Display Results ===
fprintf("\nPer-Class Accuracy Comparison:\n")
fprintf("----------------------------------------------------\n")
fprintf("%-10s %-15s %-15s %-10s\n", 'Class', 'HW4_Acc', 'MLP_Acc', 'Δ (MLP-HW4)')
fprintf("----------------------------------------------------\n")

class_labels = ["AB (Beer Bottle)"; "M (Medicine Bottle)"; "PR (Plate/Window)"];
for i = 1:3
    diff_acc = acc_MLP(i) - acc_HW4(i);
    fprintf("%-18s %-15.3f %-15.3f %+8.3f\n", class_labels(i), acc_HW4(i), acc_MLP(i), diff_acc)
end

fprintf("----------------------------------------------------\n")
fprintf("%-18s %-15.3f %-15.3f %+8.3f\n", 'Overall', overall_HW4, overall_MLP, overall_MLP - overall_HW4)
fprintf("----------------------------------------------------\n\n")

%% === Store results in a structure for later use ===
results.HW4.confusion = conf_HW4;
results.HW4.accuracy_class = acc_HW4;
results.HW4.accuracy_overall = overall_HW4;

results.MLP.confusion = conf_MLP;
results.MLP.accuracy_class = acc_MLP;
results.MLP.accuracy_overall = overall_MLP;

%% === Display confusion matrices neatly ===
disp("HW4 Confusion Matrix (True rows, Predicted cols):")
disp(array2table(conf_HW4, 'VariableNames', {'Pred_AB','Pred_M','Pred_PR'}, ...
    'RowNames', {'True_AB','True_M','True_PR'}))

disp("MLP Confusion Matrix (True rows, Predicted cols):")
disp(array2table(conf_MLP, 'VariableNames', {'Pred_AB','Pred_M','Pred_PR'}, ...
    'RowNames', {'True_AB','True_M','True_PR'}))

%% === Save results to file ===
save('HW4_vs_MLP_results.mat', 'results')
disp("Results saved to HW4_vs_MLP_results.mat")

disp("++++ Program End ++++")


% ====================================================================
function [confMat, R2, totalClass, accClass] = HW6_classify(Z,bin)
    [numSamples, ~ ] = size(Z);

    % -------------------------------
    % Build regression design matrix
    % -------------------------------
    X = ones(numSamples, 10);     % Bias + 9 features
    X(:,2:10) = Z(:,1:9);
    y = Z(:,10);                  % True labels
    
    % -------------------------------
    % Linear regression: w = (X^T X)^(-1) X^T y
    % -------------------------------
    w = (X' * X) \ (X' * y);      % Solve least squares
    y_pred = X * w;               % Fitted values
    
    % -------------------------------
    % Coefficient of determination R²
    % -------------------------------
    sse = sum((y - y_pred).^2);   % Sum of squared errors
    sst = sum((y - mean(y)).^2);  % Total variance
    R2 = 1 - sse/sst;
    %fprintf('Coefficient of determination (R^2): %.4f\n', R2)
    
    % -------------------------------
    % Bin assignment (discretize predictions)
    % -------------------------------
    predClass = bin(1) * ones(numSamples,1);  % Default bin = 2
    predClass(y_pred < bin(2)) = 1;           % Bin 1 if fit < low
    predClass(y_pred > bin(3)) = 6;           % Bin 6 if fit > high
    
    % -------------------------------
    % Confusion matrix (rows = predicted, cols = true)
    % -------------------------------
    classes = [1 2 6];
    confMat = zeros(3,3);
    
    for i = 1:length(classes)
        for j = 1:length(classes)
            confMat(i,j) = sum(predClass == classes(i) & y == classes(j));
        end
    end

    % Custom display
    % disp('>>> debug start <<<')
    % fprintf('y_pred range: %.2f to %.2f\n', min(y_pred), max(y_pred));
    % fprintf('Counts below 1.5: %d\n', sum(y_pred < 1.5));
    % fprintf('Counts between 1.5 and 2.5: %d\n', sum(y_pred >= 1.5 & y_pred <= 2.5));
    % fprintf('Counts between 2.5 and 4: %d\n', sum(y_pred > 2.5 & y_pred <= 4));
    % fprintf('Counts above 4: %d\n', sum(y_pred > 4));
    % fprintf('>>> debug end <<<\n')

    % fprintf('\nConfusion Matrix:\n')
    % fprintf('                                   True class\n')
    % fprintf('                                %6s %6s %6s\n\n', '1','2','6')
    % fprintf('predictors assigned to class 1  %6d %6d %6d\n', confMat(1,1), confMat(1,2), confMat(1,3))
    % fprintf('predictors assigned to class 2  %6d %6d %6d\n', confMat(2,1), confMat(2,2), confMat(2,3))
    % fprintf('predictors assigned to class 6  %6d %6d %6d\n\n', confMat(3,1), confMat(3,2), confMat(3,3))

    % -------------------------------
    % Class counts
    % -------------------------------
    totalClass = zeros(6,1);

    totalClass(1,:) = sum(y == 1);
    totalClass(2,:) = sum(y == 2);
    totalClass(6,:) = sum(y == 6);
    %fprintf('totalClass\n')
    %disp(totalClass)
    % -------------------------------
    % Accuracy per class & overall
    % -------------------------------
    accClass  = zeros(7,1);
    accClass(1) = confMat(1,1) / max(totalClass(1,:),1);
    accClass(2) = confMat(2,2) / max(totalClass(2,:),1);
    accClass(6) = confMat(3,3) / max(totalClass(6,:),1);
    accClass(7) = trace(confMat) / numSamples;
    
    % fprintf('Accuracy Class 1: %.3f\n', accClass1)
    % fprintf('Accuracy Class 2: %.3f\n', accClass2)
    % fprintf('Accuracy Class 6: %.3f\n', accClass6)
    % fprintf('Overall Accuracy: %.3f\n\n', accOverall)
end

