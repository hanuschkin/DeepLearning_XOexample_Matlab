%
% Illustration of convolutional neural network (CNN) function.
% Two simple image categories: O and X images
% Matlab implementation
%
% Inspired by Brandon Rohrer's 'Data Science and Robots' blog
% http://brohrer.github.io/how_convolutional_neural_networks_work.html
%
% AH received a hardware grant from NVIDIA Corporation.
% We gratefully acknowledges the support of NVIDIA Corporation for our research.
% (c) A. Hanuschkin 2016, 2017

% The CNN consistes of:
% 3 convolutional layer with filter size of 10x10 and 16-32-32 features
%
%      1   ''   Image Input             116x116x1 images with 'zerocenter' normalization
%      2   ''   Convolution             16 10x10 convolutions with stride [1  1] and padding [4  4]
%      3   ''   ReLU                    ReLU
%      4   ''   Max Pooling             5x5 max pooling with stride [2  2] and padding [0  0]
%      5   ''   Convolution             32 10x10 convolutions with stride [1  1] and padding [4  4]
%      6   ''   ReLU                    ReLU
%      7   ''   Max Pooling             5x5 max pooling with stride [2  2] and padding [0  0]
%      8   ''   Convolution             32 10x10 convolutions with stride [1  1] and padding [2  2]
%      9   ''   ReLU                    ReLU
%     10   ''   Max Pooling             3x3 max pooling with stride [2  2] and padding [0  0]
%     11   ''   Fully Connected         2 fully connected layer
%     12   ''   Softmax                 softmax
%     13   ''   Classification Output   cross-entropy


% This is the result of Matlab training the network.
% A NVIDIA GPU Quadro K620 was used. NVIDIA TitanX does not perform much
% better because of the small MiniBatchSize and low complexity of the CNN
%
% First training run
% % |========================================================================================================|
% % |     Epoch    |   Iteration  | Time Elapsed | Time Elapsed |  Mini-batch  |  Mini-batch  | Base Learning|
% % |              |              |  (seconds)   |  TitanX (s)  |    Loss      |   Accuracy   |     Rate     |
% % |========================================================================================================|
% % |            3 |           50 |       107.22 |        91.45 |       0.6923 |       87.50% |     0.001000 |
% % |            5 |          100 |       212.47 |       180.74 |       0.6885 |       98.44% |     0.001000 |
% % |            8 |          150 |       318.29 |       271.31 |       0.6413 |       90.63% |     0.001000 |
% % |           10 |          200 |       421.80 |       360.76 |       0.0732 |       96.88% |     0.001000 |
% % |========================================================================================================|
%
% Performance on the validation set 0.9533
%
% Second training run (based on first training)
% % |=========================================================================================|
% % |     Epoch    |   Iteration  | Time Elapsed |  Mini-batch  |  Mini-batch  | Base Learning|
% % |              |              |  (seconds)   |     Loss     |   Accuracy   |     Rate     |
% % |=========================================================================================|
% % |            3 |           50 |       104.44 |       0.1243 |       95.31% |     0.001000 |
% % |            5 |          100 |       211.23 |       0.1279 |       95.31% |     0.001000 |
% % |            8 |          150 |       318.30 |       0.0138 |      100.00% |     0.001000 |
% % |           10 |          200 |       424.03 |       0.0080 |      100.00% |     0.001000 |
% % |=========================================================================================|
% Performance on the validation set 0.9933
% We have 4/600 wrong classifications on the validation set.

close all; clear variables; clear global;               % clean desk
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
doTraining              = true;
doHDF5_output           = false;                        % save training/validation data to HDF5 file for Caffe input 
doImageList_output      = false;                        % save training/validation data to list file for Caffe input
% Set this flags to plot (Note: optimized for screen resolution (1920x1200))
show.wrong_classified   = true;                         % wrong classified images
show.filter             = true;                         % filters(weights)
show.feature_maps       = true;                         % feature maps

% load the file data for training the CNN
IMDS = imageDatastore('training_data_sm\','IncludeSubfolders',true,'FileExtensions','.bmp','LabelSource','foldernames'); % use imageDatastore for loading the two image categories
example_image = readimage(IMDS,1);                      % read one example image
numChannels = size(example_image,3);                    % get color information
numImageCategories = size(categories(IMDS.Labels),1);   % get category labels
[trainingDS,validationDS] = splitEachLabel(IMDS,0.7,'randomize'); % generate training and validation set
LabelCnt = countEachLabel(IMDS);                        % load lable information
for cats=1:numImageCategories                           % print out how many images we have for each category
    fprintf('%s\t%d\n',LabelCnt.Label(cats),LabelCnt.Count(cats));
end

%%
if doTraining
    %% Setup of the CNN
    % Convolutional layer parameters
    filterSize = [10 10];
    numFilters = 16;
    
    inputLayer = imageInputLayer(size(example_image));  % input layer with no data augmentation
    
    middleLayers = [
        % The first convolutional layer has a bank of numFilters filters of size filterSize. A
        % symmetric padding of 4 pixels is added.
        convolution2dLayer(filterSize, numFilters, 'Padding', 4)
        % Next add the ReLU layer:
        reluLayer()
        % Follow it with a max pooling layer that has a 5x5 spatial pooling area
        % and a stride of 2 pixels. This down-samples the data dimensions.
        maxPooling2dLayer(5, 'Stride', 2)
        % % % Repeat the 3 core layers to complete the middle of the network.
        convolution2dLayer(filterSize, numFilters*2, 'Padding', 4)
        reluLayer()
        maxPooling2dLayer(5, 'Stride',2)
        convolution2dLayer(filterSize, 2 * numFilters, 'Padding', 2)
        reluLayer()
        maxPooling2dLayer(3, 'Stride',2)];
    
    finalLayers = [
        % % Add a fully connected layer with 2 output neurons.
        fullyConnectedLayer(numImageCategories)
        % Add the softmax loss layer and classification layer. The final layers use
        % the output of the fully connected layer to compute the categorical
        % probability distribution over the image classes. During the training
        % process, all the network weights are tuned to minimize the loss over this
        % categorical distribution.
        softmaxLayer
        classificationLayer
        ];
    
    layers = [
        inputLayer
        middleLayers
        finalLayers
        ];
    
    %Initialize the first convolutional layer weights using normally distributed random numbers with standard deviation of 0.0001. This helps improve the convergence of training.
    layers(2).Weights = 0.0001 * randn([filterSize numChannels numFilters]);
    
    % Set the network training options
    opts = trainingOptions('sgdm', ...
        'Momentum', 0.9, ...
        'InitialLearnRate', 0.001, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropFactor', 0.5, ...
        'LearnRateDropPeriod', 10, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs', 10, ...         % 10 for Quadro 
        'MiniBatchSize', 64, ...    % 64 for Quadro 
        'Verbose', true);
    % 'MiniBatchSize' reduced from 128 to 64 because GPU ran out of memory
        % for Quadro GPU
    % 'MiniBatchSize' increased from 128 to >512 -> TitanX GPU but than
        % more example images would be needed (too much averaging)
    
    % Train a network.
    rng('default');
    rng(123); % random seed
    XONet = trainNetwork(trainingDS, layers, opts);
    save('XONet_r008.mat','XONet');
    % Retrain the network
    rng(123); % random seed
    XONet2 = trainNetwork(trainingDS, XONet.Layers, opts);
    save('XONet_r008_R2.mat','XONet2');
else
    % Load pre-trained detector for the example.
    load('XONet_r008.mat');
    load('XONet_r008_R2.mat');
end

%% test the performance
% test network performance on validation set
[labels,~] = classify(XONet, validationDS, 'MiniBatchSize', 128);
confMat = confusionmat(validationDS.Labels, labels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
fprintf('Performance on validation set \t\t\t%.4f\n',mean(diag(confMat)));

% test network performance on validation set
[labels,~] = classify(XONet2, validationDS, 'MiniBatchSize', 128);
confMat = confusionmat(validationDS.Labels, labels);
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));
fprintf('Performance on validation set after retraining \t%.4f\n',mean(diag(confMat)));

%% plot wrong classsified images
if show.wrong_classified
    % % Find wrong classified examples
    idx = find(validationDS.Labels~=labels);
    idx2 = int16(rand(length(idx),1)*size(labels,1));
    fprintf('We have %d/%d wrong classifications\n',length(idx),size(labels,1));
    for jj = 1:length(idx)
        img = readimage(validationDS,idx(jj));
        figh=figure(1);clf;
        set(figh,'Outerposition',[1,41,450,360]);
        subplot(1,2,1)
        imagesc(img),
        axis square
        axis xy
        lab = sprintf('classified as %s',labels(idx(jj)));
        title(lab,'Color','Red');
        img = readimage(validationDS,idx2(jj));
        subplot(1,2,2)
        imagesc(img)
        axis square
        axis xy
        lab = sprintf('classified as %s',labels(idx2(jj)));
        title(lab,'Color','Green');
        pause
    end
end

%% plot the filter (weights)
if show.filter
    figh=figure(3);clf;
    set(figh,'Outerposition',[451,41,450,360]);
    for jj=1:16
        subplot(4,4,jj)
        imagesc(XONet2.Layers(2).Weights(:,:,1,jj))
        set(gca, 'XTickLabel', [])
        set(gca, 'YTickLabel', [])
        axis xy
        colormap(gray);
    end
    subplot(4,4,2)
    title('1 conv layer, weights of filter 1:16');
    
    %%
    for filter = 1:32
        figh=figure(20+filter);clf;
        set(figh,'Outerposition',[450*2+1+filter*2,41,450,360]);
        for jj=1:16
            subplot(4,4,jj)
            imagesc(XONet2.Layers(5).Weights(:,:,jj,filter))
            set(gca, 'XTickLabel', [])
            set(gca, 'YTickLabel', [])
            axis xy
            colormap(gray);
        end
        subplot(4,4,2)
        fname = sprintf('2 conv layer, weights of filter %d',filter);
        title(fname);
    end
    for filter = 1:32
        figh=figure(333+filter);clf;
        set(figh,'Outerposition',[450*3+1+filter*2,41,450,360]);
        for jj=1:32
            subplot(6,6,jj)
            imagesc(XONet2.Layers(8).Weights(:,:,jj,filter))
            set(gca, 'XTickLabel', [])
            set(gca, 'YTickLabel', [])
            axis xy
            colormap(gray);
        end
        subplot(6,6,2)
        fname = sprintf('3 conv layer, weights of filter %d',filter);
        title(fname);
    end
end

%% plot feature maps for a single examples of each category
if show.feature_maps
    for cats=1:2
        if cats==1
            fig_off = 0;  % figure id offset
            y_offset = 0; % y-position offset
            imds_test = imageDatastore(validationDS.Files(8));  % circle            
        else
            fig_off=100;
            y_offset = 400;
            imds_test = imageDatastore(validationDS.Files(end-1));  % cross            
        end
        figh = figure(65+fig_off);clf;
        set(figh,'Outerposition',[1,800-y_offset,450,400]);
        imagesc(readimage(imds_test,1))
        axis xy
        colormap(gray);
        features_conv1_maxpool = activations(XONet2,imds_test,4,'OutputAs','channels');
        figh= figure(66+fig_off);clf;
        set(figh,'Outerposition',[451,800-y_offset,450,400]);
        for jj = 1:16
            subplot(4,4,jj)
            imagesc(features_conv1_maxpool(:,:,jj))
            set(gca, 'XTickLabel', [])
            set(gca, 'YTickLabel', [])
            axis xy
        end
        features_conv1_maxpool2 = activations(XONet2,imds_test,7,'OutputAs','channels');
        figh=figure(67+fig_off);clf;
        set(figh,'Outerposition',[450*2+1,800-y_offset,450,400]);
        for jj = 1:32
            subplot(6,6,jj)
            imagesc(features_conv1_maxpool2(:,:,jj))
            set(gca, 'XTickLabel', [])
            set(gca, 'YTickLabel', [])
            axis xy
        end
        features_conv1_maxpool3 = activations(XONet2,imds_test,10,'OutputAs','channels');
        figh=figure(68+fig_off);clf;
        set(figh,'Outerposition',[450*3+1,800-y_offset,450,400]);
        for jj = 1:32
            subplot(6,6,jj)
            imagesc(features_conv1_maxpool3(:,:,jj))
            set(gca, 'XTickLabel', [])
            set(gca, 'YTickLabel', [])
            axis xy
        end
        features_fc = activations(XONet2,imds_test,11,'OutputAs','channels');
        figh=figure(69+fig_off);clf;
        set(figh,'Outerposition',[450*4+1,800-y_offset,130,400]);
        plot([1,2],[features_fc(1,1,1), features_fc(1,1,2)],'X','Color','red','Linewidth',5)
        %axis([0,3,-5,5])
    end
end

%% HDF5 output
if doHDF5_output
    fprintf('output training/validation data sets to HDF5\n');
    fprintf('training set to HDF5...\n');
    Img = zeros([size(example_image) 1 size(trainingDS.Files,1)]);
    Seg = zeros(1,size(trainingDS.Labels,1));
    for img_nr = 1:size(trainingDS.Files,1)
        Img(:,:,:,img_nr) = readimage(trainingDS,img_nr);
        if trainingDS.Labels(img_nr) == 'circles_sm'
            Seg(img_nr) = 0;
        else
            Seg(img_nr) = 1;
        end
    end
    hdf5write( 'training_set.h5','data', single(Img),'label', single(Seg));

    fprintf('validation set to HDF5...\n');
    Img = zeros([size(example_image) 1 size(validationDS.Files,1)]);
    Seg = zeros(1,size(validationDS.Labels,1));
    for img_nr = 1:size(validationDS.Files,1)
        Img(:,:,img_nr) = readimage(validationDS,img_nr);
        if validationDS.Labels(img_nr) == 'circles_sm'
            Seg(img_nr) = 0;
        else
            Seg(img_nr) = 1;
        end
    end
    hdf5write( 'validation_set.h5','data', single(Img),'label', single(Seg));
end

%% ImageList output
pattern = '\w+.bmp';
if doImageList_output
    fprintf('output training/validation data sets to List of Images\n');
    fprintf('training set to List of Images...\n');
    fileID = fopen('training_img_files_space.txt','w');
    for img_nr = 1:size(trainingDS.Files,1)
        d_c = regexp(trainingDS.Files{img_nr}, pattern, 'match');
        if trainingDS.Labels(img_nr) == 'circles_sm'
            d_l = 0;
        else
            d_l = 1;
        end
        fprintf(fileID,'%s %d\n',d_c{1},d_l);        
    end
    fclose(fileID);
    fprintf('validation set to List of Images...\n');
    fileID = fopen('validation_img_files_space.txt','w');
    for img_nr = 1:size(validationDS.Files,1)
        d_c = regexp(validationDS.Files{img_nr}, pattern, 'match');
        if validationDS.Labels(img_nr) == 'circles_sm'
            d_l = 0;
        else
            d_l = 1;
        end
        fprintf(fileID,'%s %d\n',d_c{1},d_l);
    end
    fclose(fileID);
end
