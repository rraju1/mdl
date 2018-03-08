fileID_acc = fopen('test.txt','r');
fileID_norm = fopen('norm.txt','r');
f_cost_acc_lam5 = fopen('cost_1e-5.txt','r');
f_cost_norm_lam5 = fopen('norm_1e-5.txt', 'r');
f_cost_acc = fopen('cost_test.txt','r');
f_cost_norm = fopen('cost_norm.txt', 'r');
f_cost_acc_lam3 = fopen('cost_test_1e-3.txt', 'r');
f_norm_acc_lam3 = fopen('cost_norm_1e-3.txt', 'r');

formatSpec = '%f';

accuracy = fscanf(fileID_acc, formatSpec);
norm = fscanf(fileID_norm, formatSpec);
acc_cost = fscanf(f_cost_acc, formatSpec);
norm_cost = fscanf(f_cost_norm, formatSpec);
lamb3acc = fscanf(f_cost_acc_lam3, formatSpec);
lamb3norm = fscanf(f_norm_acc_lam3, formatSpec);
lamb5acc = fscanf(f_cost_acc_lam5, formatSpec);
lamb5norm = fscanf(f_cost_norm_lam5, formatSpec);

acc_data_points = 21;
num_trials = 10;
acc_trials = zeros(acc_data_points,1);
norm_trials = zeros(acc_data_points,1);
acc_cost_trials = zeros(acc_data_points,1);
norm_cost_trials = zeros(acc_data_points,1);
lam3acc_trials = zeros(acc_data_points,1);
lam3norm_trials = zeros(acc_data_points,1);
lam5acc_trials = zeros(acc_data_points,1);
lam5norm_trials = zeros(acc_data_points,1);


for i=1:num_trials
    for j=1:acc_data_points
        acc_trials(j) = acc_trials(j) + accuracy((num_trials - 1)*acc_data_points + j);
        norm_trials(j) = norm_trials(j) + norm((num_trials - 1)*acc_data_points + j);
        acc_cost_trials(j) = acc_cost_trials(j) + acc_cost((num_trials - 1)*acc_data_points + j);
        norm_cost_trials(j) = norm_cost_trials(j) + norm_cost((num_trials - 1)*acc_data_points + j);
        lam3acc_trials(j) = lam3acc_trials(j) + lamb3acc((num_trials - 1)*acc_data_points + j);
        lam3norm_trials(j) = lam3norm_trials(j) + lamb3norm((num_trials - 1)*acc_data_points + j);
        lam5acc_trials(j) = lam5acc_trials(j) + lamb5acc((num_trials - 1)*acc_data_points + j);
        lam5norm_trials(j) = lam5norm_trials(j) + lamb5norm((num_trials - 1)*acc_data_points + j);
    end
end
f_acc = figure;
f_norm = figure;

acc_trials = acc_trials/num_trials;
norm_trials = norm_trials/num_trials;
acc_cost_trials = acc_cost_trials/num_trials;
norm_cost_trials = norm_cost_trials/num_trials;
lam3acc_trials = lam3acc_trials/num_trials;
lam3norm_trials = lam3norm_trials/num_trials;
lam5acc_trials = lam5acc_trials/num_trials;
lam5norm_trials = lam5norm_trials/num_trials;
xpos = linspace(0,100,acc_data_points);
figure(f_acc);

plot(xpos, acc_trials,'-go',xpos,lam5acc_trials, '-bo',xpos, acc_cost_trials, '-ro',xpos,lam3acc_trials, '-mo');

title('Error vs. Accuracy')
xlabel('Error')
ylabel('Accuracy')
figure(f_norm)

plot(xpos, norm_trials,'-go',xpos,lam5norm_trials, '-bo',xpos, norm_cost_trials, '-ro',xpos,lam3norm_trials, '-mo');
title('Error vs. Norm')
xlabel('Error')
ylabel('Relative Norm')