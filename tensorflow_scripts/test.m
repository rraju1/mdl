filename = 'woutOri.csv';
M = csvread(filename);
filename = 'woutUpdated.csv';
M1 = csvread(filename);

newM = M - M1;
non_zero = nnz(newM);
fprintf('%d\n',norm(newM))