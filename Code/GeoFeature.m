function fea = GeoFeature(X,edge,get_knn,knn_dis)
[n, m] = size(X);
k = size(get_knn,2);
fea = zeros(length(edge), 4);
fea(:, 1) = sqrt(sum((X(edge(:,1),:)-X(edge(:,2),:)).^2, 2));

mean_knn_dis = mean(knn_dis,2);
fea(:, 2) = fea(:, 1)./min(mean_knn_dis(edge),[],2);

mean_vec = squeeze(mean(reshape(X(reshape(get_knn',[],1),:),k,n,m)))-X;
norm_vec = mean_vec./max(sqrt(sum(mean_vec.^2, 2)),eps);
edge_vec = (X(edge(:,2),:) - X(edge(:,1),:))./fea(:,1);
fea(:,3) = sum((norm_vec(edge(:,1),:)-norm_vec(edge(:,2),:)).*edge_vec,2);

for i=1:length(edge)
    fea(i,4) = length(intersect(get_knn(edge(i,1),:),get_knn(edge(i,2),:)));
end
fea(:,1) = [];
fea = zscore(fea);