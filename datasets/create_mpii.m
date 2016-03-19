% output format
% a .hdf5 file with dims to be [H,W,3,N]
%                           or [W,H,3,N]

addpath mpii; 

% load annotation
load('mpii/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat');  % named RELEASE

% only select images that satisfy all of these 
% 1. full joints annotation
% 2. single person
train_idx = find(RELEASE.img_train==1);
img_flags = vertcat(1,RELEASE.single_person{train_idx});
num_max = numel(img_flags);

data = zeros([64,64,3,num_max],'single');
num_added = 0;
tic;
for i = 1:numel(train_idx)
    img_id = train_idx(i);
    sing_per_idx = RELEASE.single_person{img_id};
    for j = 1:numel(sing_per_idx)
        anno = RELEASE.annolist(img_id).annorect(j);
        if isempty(anno.annopoints), continue; end
        xys = [[anno.annopoints.point.x];
               [anno.annopoints.point.y]];
        ids = [anno.annopoints.point.id];
        % inconsistency in annotation:
        % in_visibile can be [], 0, 1
        % vis = [anno.annopoints.point.is_visible]; 
        if numel(ids) < 16, continue; end

        img_name = RELEASE.annolist(img_id).image.name;
        img_path = fullfile('mpii/images/', img_name);
        im = imread(img_path);
        [h,w,~] = size(im);
        x1 = max(1,min(xys(1,:))); x2 = min(w,max(xys(1,:)));
        y1 = max(1,min(xys(2,:))); y2 = min(h,max(xys(2,:)));


        im = im(floor(y1):ceil(y2), floor(x1):ceil(x2), :);
        %hold on;
        %rectangle('position', [x1,y1,x2-x1+1,y2-y1+1]);
        %hold off;
        rim = imresize(im, [64,64]);
        data(:,:,:,num_added+1) = rim;
        num_added = num_added+1;

        if rand < 0.01
            imwrite(rim, fullfile('mpii_samples', img_name));
        end
    end
    if toc > 1
        fprintf('Processed %d/%d images.\n', i, numel(train_idx));
        tic;
    end
end

% remove non-used room
data = data(:,:,:,1:num_added);

% TODOO permute [2,1,3]

% convert to gray-scale for ease of training

% output
out_name = 'mpii_full_64';
out_path = [out_name '.hdf5'];
fprintf('Added %d images into %s.\n', num_added, out_path);

% write to hdf5 
h5create(out_path, ['/' out_name], [64,64,3,num_added]);
h5write(out_path, ['/' out_name], data);
