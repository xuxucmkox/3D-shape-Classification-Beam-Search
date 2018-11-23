rng('shuffle');
kernels;
addpath util;
addpath 3D;
addpath bp;
addpath generative;

debug = 0;

data_list = read_data_list(model.data_path, model.classnames, ....
    model.volume_size + 2 * model.pad_size, 'train', debug);

param = [];
param.epochs = 70; %modify
param.lr = 2;
param.weight_decay = 5*10^-4;
param.momentum = 0.9;
param.batch_size = 32;
param.snapshot_iter = 1;
param.snapshot_name = 'bp_finetune_iter';
param.test_iter = 1;
batch_size = param.batch_size;

fprintf('Begin discriminative funetuning the CDBN\n');
fprintf('lr = %f, wd = %d, momentum = %f\n', param.lr, param.weight_decay, param.momentum);

% prepare data and label
[new_list, label] = balance_data(data_list, batch_size);
n = length(new_list);
batch_size = n*0.8; % use 80% subset of samples
param.batch_size = batch_size;
batch_num = 1;
ntrees = 2; %modify

numLayer = model.numLayer;
model.layers{numLayer}.layerSize = [ntrees*7, 1]; %modify
model.layers{numLayer}.w = rand([prod(model.layers{numLayer-1}.layerSize), prod(model.layers{numLayer}.layerSize)], 'single');
model.layers{numLayer}.w = (model.layers{numLayer}.w - 0.5) * 2 * sqrt( 6 / (prod(model.layers{numLayer}.layerSize) + prod(model.layers{numLayer-1}.layerSize)));
model.layers{numLayer}.c = zeros([1, model.layers{numLayer}.layerSize], 'single');


for l = 2 : numLayer
    model.layers{l} = rmfield(model.layers{l},'b');
    model.layers{l}.grdw = zeros(size(model.layers{l}.w), 'single');
    model.layers{l}.grdc = zeros(size(model.layers{l}.c), 'single');
    model.layers{l}.histw = zeros(size(model.layers{l}.w), 'single');
    model.layers{l}.histc = zeros(size(model.layers{l}.c), 'single');
end


hist = zeros(ntrees,8,model.classes); %modify

for tree = 1:ntrees
    
    % run a subset of data ,collecting prob at each node
    
    shuffle_index = randperm(n);
    batch_index = shuffle_index(1 : batch_size);
    batch_label = label(batch_index,:);
    
    % compute histogram,maybe we can make samples here to estimate prob
    
    for iter = 1 : param.epochs
        batch = read_batch(model, new_list(batch_index), false); %reset the batch
        [model, activation] = bp_forward(model, batch); % this function modifies batch or not?
        for instance = 1:batch_size;
            index = batch_index(instance);
            prb = activation{numLayer}(instance,1+(tree-1)*7);  %prb for prob for routing direction
            if rand < prb
                prb = activation{numLayer}(instance,2+(tree-1)*7);
                if rand < prb
                    prb = activation{numLayer}(instance,4+(tree-1)*7);
                    if rand < prb
                        hist(tree,1,new_list(index).label) = hist(tree,1,new_list(index).label) + 1;
                    else
                        hist(tree,2,new_list(index).label) = hist(tree,2,new_list(index).label) + 1;
                    end
                else
                    prb = activation{numLayer}(instance,5+(tree-1)*7);
                    if rand < prb
                       hist(tree,3,new_list(index).label) = hist(tree,3,new_list(index).label) + 1;
                    else
                       hist(tree,4,new_list(index).label) = hist(tree,4,new_list(index).label) + 1;
                    end
                end
            else
                prb = activation{numLayer}(instance,3+(tree-1)*7);
                if rand < prb
                    prb = activation{numLayer}(instance,6+(tree-1)*7);
                    if rand < prb
                        hist(tree,5,new_list(index).label) = hist(tree,5,new_list(index).label) + 1;
                    else
                        hist(tree,6,new_list(index).label) = hist(tree,6,new_list(index).label) + 1;
                    end
                else
                    prb = activation{numLayer}(instance,7+(tree-1)*7);
                    if rand < prb
                       hist(tree,7,new_list(index).label) = hist(tree,7,new_list(index).label) + 1;
                    else
                       hist(tree,8,new_list(index).label) = hist(tree,8,new_list(index).label) + 1;
                    end
                end
            end
        end
        
        for node = 1:8  %modify
            if sum(hist(tree,node,:))~=0
                hist(tree,node,:) = hist(tree,node,:) / sum(hist(tree,node,:));
            end
            for class = 1:model.classes
                if hist(tree,node,class) == 0
                    hist(tree,node,class) = 0.0001;
                end
            end
            
        end
   
        
        % need to change activation in loss and error in bp_backward to hist value
        [model, loss] = bp_backward(model, activation, batch_label,hist,tree);
        model = bp_update(model, param);
        
        fprintf('iteration: %d, loss: %f\n', iter, loss);
        
    end
        
end

% after we have hist and trained model, we can route one sample through all
% trees to their leaves by looking up the act value in model, then compute
% the prob.

for l = 2 : numLayer
    model.layers{l} = rmfield(model.layers{l},'grdw');
    model.layers{l} = rmfield(model.layers{l},'grdc');
    model.layers{l} = rmfield(model.layers{l},'histw');
    model.layers{l} = rmfield(model.layers{l},'histc');
end

save('bp_finetuned_model', 'model');
save('prob','hist');
