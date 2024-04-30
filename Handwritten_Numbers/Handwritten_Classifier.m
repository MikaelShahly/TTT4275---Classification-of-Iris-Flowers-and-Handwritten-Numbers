data_all = load('data_all.mat');
X_test = data_all.testv;
y_test = data_all.testlab;
X_train = data_all.trainv;
y_train = data_all.trainlab;

use_clusters = true;

if use_clusters
    X_train_tot = sortrows([X_train y_train], 28*28+1);
    M = 64; %Number of clusters per class
    
    %Clustering of each class
    [ind1, C1] = kmeans(X_train_tot(1:5923, 1:28*28), M);
    [ind2, C2] = kmeans(X_train_tot(5924:12665, 1:28*28), M);
    [ind3, C3] = kmeans(X_train_tot(5925:18623, 1:28*28), M);
    [ind4, C4] = kmeans(X_train_tot(18624:24754, 1:28*28), M);
    [ind5, C5] = kmeans(X_train_tot(24755:30596, 1:28*28), M);
    [ind6, C6] = kmeans(X_train_tot(30597:36017, 1:28*28), M);
    [ind7, C7] = kmeans(X_train_tot(36018:41935, 1:28*28), M);
    [ind8, C8] = kmeans(X_train_tot(41936:48200, 1:28*28), M);
    [ind9, C9] = kmeans(X_train_tot(48201:54051, 1:28*28), M);
    [ind10, C10] = kmeans(X_train_tot(54052:60000, 1:28*28), M);

    %Cluster matrix
    C = [C1;C2;C3;C4;C5;C6;C7;C8;C9;C10];

    %Class of each cluster
    y_clusters = [zeros(M, 1); ones(M, 1); 2*ones(M, 1); 3*ones(M, 1); 4*ones(M, 1); 5*ones(M, 1); 
        6*ones(M, 1); 7*ones(M, 1); 8*ones(M, 1); 9*ones(M, 1)];
end

%Classification
k = 7; %Number of nearest neighbors to compare (only when using clustering)
group_size = 1000; %Size of goups of test samples
y_pred = zeros(10000, 1);
if use_clusters
    for i = 0:length(X_test)/group_size-1
        d = dist(X_test(group_size*i+1:group_size*(i+1),:), C.');
        for j = 1:group_size
            k_nearest = zeros(k, 1); %Classes of nearest neighbors
            for n=1:k
                [minDist, indx] = min(d(j,:));
                d(j,indx) = inf; %Next neighbor must be different
                k_nearest(n) = y_clusters(indx);
            end
            y_pred(group_size*i+j) = mode(k_nearest); %Take most frequent class of neighbors
        end
    end
else
    for i = 0:length(X_test)/group_size-1
        d = dist(X_test(group_size*i+1:group_size*(i+1),:), X_train.');
        for j = 1:group_size
            [minDist, indx] = min(d(j,:));
            y_pred(group_size*i+j) = y_train(indx);
        end
    end
end