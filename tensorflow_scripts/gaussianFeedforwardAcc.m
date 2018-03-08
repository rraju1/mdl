fileID_acc = fopen('results/original_cost.txt','r');
fileID_norm = fopen('results/original_norm.txt','r');
fileID_acc_lam5 = fopen('results/lam5_cost.txt','r');
fileID_norm_lam5 = fopen('results/lam5_norm.txt','r');
fileID_acc_lam4 = fopen('results/lam4cost.txt','r');
fileID_norm_lam4 = fopen('results/lam4norm.txt','r');
formatSpec = '%f';

accuracy = fscanf(fileID_acc, formatSpec);
norm = fscanf(fileID_norm, formatSpec);
acc_lam5 = fscanf(fileID_acc_lam5, formatSpec);
norm_lam5 = fscanf(fileID_norm_lam5, formatSpec);
acc_lam4 = fscanf(fileID_acc_lam4, formatSpec);
norm_lam4 = fscanf(fileID_norm_lam4, formatSpec);

acc_data_points = 21;
num_trials = 40;
acc_trials = zeros(acc_data_points,1);
norm_trials = zeros(acc_data_points,1);
lam5_acc_trials = zeros(acc_data_points,1);
lam5_norm_trials = zeros(acc_data_points,1);
lam4_acc_trials = zeros(acc_data_points,1);
lam4_norm_trials = zeros(acc_data_points,1);


for i=1:num_trials
    for j=1:acc_data_points
        acc_trials(j) = acc_trials(j) + accuracy((num_trials - 1)*acc_data_points + j);
        norm_trials(j) = norm_trials(j) + norm((num_trials - 1)*acc_data_points + j);
        lam5_acc_trials(j) = lam5_acc_trials(j) + acc_lam5((num_trials - 1)*acc_data_points + j);
        lam5_norm_trials(j) = lam5_norm_trials(j) + norm_lam5((num_trials - 1)*acc_data_points + j);
        lam4_acc_trials(j) = lam4_acc_trials(j) + acc_lam4((num_trials - 1)*acc_data_points + j);
        lam4_norm_trials(j) = lam4_norm_trials(j) + norm_lam4((num_trials - 1)*acc_data_points + j);
    end
end
f_acc = figure;
f_norm = figure;

acc_trials = acc_trials/num_trials;
norm_trials = norm_trials/num_trials;
lam5_acc_trials = lam5_acc_trials/num_trials;
lam5_norm_trials = lam5_norm_trials/num_trials;
lam4_acc_trials = lam4_acc_trials/num_trials;
lam4_norm_trials = lam4_norm_trials/num_trials;
xpos = linspace(0,100,acc_data_points);
figure(f_acc);

plot(xpos, acc_trials,'-go', xpos, lam5_acc_trials, '-ro', xpos, lam4_acc_trials, '-bo');

title('Error vs. Accuracy')
xlabel('Error')
ylabel('Accuracy')
figure(f_norm)

plot(xpos, norm_trials,'-go', xpos,lam5_norm_trials, '-ro', xpos, lam4_norm_trials, '-bo');
title('Error vs. Norm')
xlabel('Error')
ylabel('Relative Norm')