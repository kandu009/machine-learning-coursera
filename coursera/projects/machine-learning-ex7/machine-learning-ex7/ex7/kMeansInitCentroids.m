function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

% for i = 1:length(X)
%     minDist = Inf;
%     for j = 1:K
%         dist = norm(X(i,:) - centroids(j,:)) ^ 2;
%         if dist < minDist
%             minDist = dist;
%             centroids(i) = j;
%         end
%     end
% end

%Initializethecentroidstoberandomexamples
%Randomlyreordertheindicesofexamples
randidx = randperm(size(X, 1));
%TakethefirstKexamplesascentroids
centroids = X(randidx(1:K), :);

% =============================================================

end

