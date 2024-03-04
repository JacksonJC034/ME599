% Read files; normalize data
no_classes = 7;
data_name = {'anger','disgust','fear','happy','neutral','sadness','surprise'};
data = []; tfiles(1) = 0; class = {};datak = [];
 
for j = 1:no_classes
    image_folder = ['./KDEF_wavelet_data/',data_name{j} '/']; % Relative Path For Macbook
    % image_folder = ['.\KDEF_wavelet_data\',data_name{j} '\']; % Relative Path for Windows
    file_pattern = fullfile(image_folder, '*.png');
    image_files = dir(file_pattern);
    nfiles = length(image_files);
    nfs(j) = nfiles;
    files_rg = tfiles(j) + 1:tfiles(j) + nfiles;
    class(files_rg) = data_name(j);
    i = 0;
    for k = files_rg        
     i = i+1;filename = image_files(i).name;
     im = imread([image_folder, filename]); % Read i th image
     datak(:, k) = reshape(im, [numel(im), 1]); % Save image
    end
    tfiles(j+1) = tfiles(j) + nfiles;
 
end
data  = datak/255;
 
%%

% Selecting data for training and testing
 
ratio = .75; % Ratio of training to test data
rtrain = []; tp(1) = 0; 
load('seed');
rng(s)
% s = rng;
for j = 1:no_classes
    lmt(j) = floor(ratio*nfs(j));
    rp = tfiles(j) + randperm(nfs(j),lmt(j));
    tp(j+1) = tp(j)+ lmt(j);
    rtrain = cat(2,rtrain,rp);
end
rtest = setdiff(1:size(data,2),rtrain);
rtest= rtest';rtrain = rtrain';
 
 
train_data = data(:,rtrain);
train_label = class(rtrain);
test_data = data(:,rtest);
test_label = class(rtest);