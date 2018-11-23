
% Discriminative finetuning of CDBN.
% Input: generatively pretrained model.
% Output: discriminative CNN model.

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
param.epochs = 10;
param.lr = 0.1;
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
batch_size = n;
param.batch_size = batch_size;
batch_num = n / batch_size;
assert(batch_num == floor(batch_num));


% prepare model: replace the topmost layer
numLayer = model.numLayer;
numClass = 2;
model.layers{numLayer}.layerSize = [numClass, 1];
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

% start training
for iter = 1 : param.epochs
    shuffle_index = randperm(n);
    batch_index = shuffle_index(1:batch_size);
    batch = read_batch(model, new_list(batch_index), false);
    [model, activation] = bp_forward(model, batch);
    
    batch_class_label = label(batch_index,:);
    batch_label = single(zeros(9600,2));
    gini_impurity = 0;
    
    for c = 1:model.classes
        activation_class = activation{numLayer}(batch_class_label(:,c) == 1,:);
        left = 0;
        right = 0;
        for m = 1:size(activation_class,1)
            if activation_class(m,1)>=0.5
                left = left + 1;
            else
                right = right + 1;
            end
        end
        
        if left > right
            batch_label(batch_class_label(:,c)==1,1) = 1;
            batch_label(batch_class_label(:,c)==1,2) = 0;
        else
            batch_label(batch_class_label(:,c)==1,1) = 0;
            batch_label(batch_class_label(:,c)==1,2) = 1;
        end
        
        gini_impurity = gini_impurity + left*right/(left+right)^2;
    end
    
    [model, loss] = bp_backward(model, activation, batch_label);
    model = bp_update(model, param);
    
    fprintf('iteration: %d, gini impurity: %f\n', iter, gini_impurity);
    
    if mod(iter, param.snapshot_iter) == 0
        fprintf('snapshoting to %s_%d\n', param.snapshot_name, iter);
        snapshot_name = sprintf('%s_%d', param.snapshot_name, iter);
        save(snapshot_name, 'model');
  
    end
    

end

for l = 2 : numLayer
    model.layers{l} = rmfield(model.layers{l},'grdw');
    model.layers{l} = rmfield(model.layers{l},'grdc');
    model.layers{l} = rmfield(model.layers{l},'histw');
    model.layers{l} = rmfield(model.layers{l},'histc');
end

save('bp_finetuned_model', 'model');
