close all

%Test predictions
evaluated_predictions = 1*ones(10000, 1); %1 if correct, 0 if incorrect
n_correct = 0;
for i=1:length(y_pred)
    if y_pred(i) ~= y_test(i)
        evaluated_predictions(i) = 0;
    else
        n_correct = n_correct + 1;
    end
end
error_rate = 1 - n_correct/length(y_test);

%Confusion matrix
figure()
CMat = confusionmat(y_test, y_pred);
confusionchart(CMat, categorical({'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}), 'Normalization', 'row-normalized');
xlabel("Predicted number")
ylabel("True number")

%Find index of random correct and incorrect
%classifications of the number 'wanted_number'
random_incorrect_indx = 0;
random_correct_indx = 0;
wanted_number = 5;
max_tries = 1000000;
for j = 1:max_tries
    i = randi([1 length(y_pred)]);
    if evaluated_predictions(i) == 0 && y_test(i) == wanted_number
        random_incorrect_indx = i;
        break;
    end
end
for j = 1:max_tries
    i = randi([1 10000]);
    if evaluated_predictions(i) == 1 && y_test(i) == wanted_number
        random_correct_indx = i;
        break;
    end
end

%Plot correctly and incorrectly
%classified images

figure()
m = zeros(28,28);
m(:) = X_test(random_incorrect_indx,:);
image(flip(imrotate(m, 90)));
title("Misclassified")
set(gca,'xtick',[])
set(gca,'ytick',[])

figure()
m = zeros(28,28);
m(:) = X_test(random_correct_indx,:);
image(flip(imrotate(m, 90)));
title("Correctly classified")
set(gca,'xtick',[])
set(gca,'ytick',[])

