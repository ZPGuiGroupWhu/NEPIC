function [cluster] = NEPIC(X,varargin)

paramNames = {'k','minPts'};
defaults   = {10,0};
[k, minPts] = internal.stats.parseArgs(paramNames, defaults, varargin{:});

[X, ~, orig_id] = unique(X, 'rows');
n = size(X, 1);

%% Search KNN
[get_knn, knn_dis] = knnsearch(X, X, 'k', k+1);
get_knn(:, 1) = [];
knn_dis(:, 1) = [];

%% Construct Delaunay Triangulation Network
tri = delaunay(X);
tri = sort(tri,2);
edge = [tri(:,1),tri(:,2);tri(:,1),tri(:,3);tri(:,2),tri(:,3)];
edge = unique(edge,'rows','stable');

%% Claculate geometric features
fea = GeoFeature(X,edge,get_knn,knn_dis);

%% Identify cross-cluster edges
ind = kmeans(fea, 2,"MaxIter",200,"Replicates",5);
intra_edge_id = ind==1;
cross_edge_id = ind==2;
plotDTN(X, edge, ind);
if(mean(fea(cross_edge_id,1)) < mean(fea(intra_edge_id,1)))
    intra_edge_id = ind==2;
    cross_edge_id = ind==1;
end
cross_edge = edge(cross_edge_id,:);
intra_edge = edge(intra_edge_id,:);
iso_pts = setdiff(1:n,unique(intra_edge));
% plotDTN(X, intra_edge, ones(length(intra_edge),1));

%% Generate connected components
G = graph([intra_edge(:,1)',iso_pts], [intra_edge(:, 2)',iso_pts]);
cluster = conncomp(G, 'Type', 'weak');

%% Identify boundary points
bou_pts_id = [];
for i=1:size(cross_edge, 1)
    pts_id = tri(any(tri == cross_edge(i,1), 2) & any(tri == cross_edge(i,2), 2),:);
    bou_pts_id = [bou_pts_id;unique(pts_id(:))];
end
bou_pts_id = unique(bou_pts_id);

clus_num = length(unique(cluster));
for i=1:clus_num
    clus_id = find(cluster==i);
    X_i = X(clus_id,:);
    bou_pts_mark = ismember(clus_id,bou_pts_id);
    if(~min(bou_pts_mark))
        sg = subgraph(G, clus_id);
        sg = rmnode(sg, find(bou_pts_mark));
        re_clus = conncomp(sg, 'Type', 'weak');
        if(max(re_clus) > 1)
            temp_clus = zeros(length(clus_id),1);
            temp_clus(bou_pts_mark==0) = re_clus;
            near_int_pts = knnsearch(X_i(bou_pts_mark==0,:),X_i(bou_pts_mark,:),'k',1);
            temp_clus(bou_pts_mark) = re_clus(near_int_pts);
            cluster(clus_id) = max(cluster) + temp_clus;
        end
    end
end

%% Detect noise points
clus_lab = unique(cluster);
for i=1:length(clus_lab)
    clus_id = find(cluster==clus_lab(i));
    if(length(clus_id) < minPts)
        cluster(clus_id) = 0;
    end
end
% plot(X(bou_pts_id,1),X(bou_pts_id,2),'ko');
% hold on;

%% Adjust the cluster id to continuous positive integer
mark_temp = 1;
storage = zeros(n,1);
for i=1:n
    if (cluster(i)~=0)
        if(ismember(cluster(i),storage)==0)
            storage(i) = cluster(i);
            cluster(i) = mark_temp;
            mark_temp = mark_temp+1;
        else
            cluster(i) = cluster(find(storage==cluster(i),1));
        end
    end
end

cluster = cluster(:,orig_id)';
end