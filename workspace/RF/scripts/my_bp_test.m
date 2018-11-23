rng('shuffle');
kernels;
addpath util;
addpath 3D;
addpath bp;
addpath generative;

debug = 0;

data_list = read_data_list(model.data_path, model.classnames, ...
    model.volume_size + 2 * model.pad_size, 'test');

% create testing data list and labels
label = [];
new_list = repmat(struct('filename', '', 'label', 0), 1, 1);
curr = 1;
for i = 1 : model.classes
    cnt = length(data_list{i});
    label(curr:curr+cnt-1,1) = i;
    new_list(curr:curr+cnt-1,1) = data_list{i};
    curr = curr + cnt;
end
n = length(label);

% prepare data and label
test_index = [1:n];
batch_size = 32;
batch_num = ceil(n / batch_size);

accuracy = 0;
for b = 1 : batch_num
    idx_end = min(b*batch_size, n);
    batch_index = test_index((b-1)*batch_size + 1 : idx_end);
    batch = read_batch(model, new_list(batch_index), false);
    batch_label = label(batch_index,:);
    [model, activation] = bp_forward(model, batch);
    predict = zeros(batch_size,1);
    
    for instance = 1:batch_size;    
        postprb = zeros(1,model.classes);
        index = batch_index(instance);
        for tree = 1:size(hist,1)
            prb = activation{model.numLayer}(instance,1+(tree-1)*7);  %prb for prob for routing direction
            if rand < prb
                prb = activation{model.numLayer}(instance,2+(tree-1)*7);
                if rand < prb
                    prb = activation{model.numLayer}(instance,4+(tree-1)*7);
                    if rand < prb
                        postprb = postprb + reshape(hist(tree,1,:),[1,model.classes]);
                    else
                        postprb = postprb + reshape(hist(tree,2,:),[1,model.classes]);
                    end
                else
                    prb = activation{model.numLayer}(instance,5+(tree-1)*7);
                    if rand < prb
                        postprb = postprb + reshape(hist(tree,3,:),[1,model.classes]);
                    else
                        postprb = postprb + reshape(hist(tree,4,:),[1,model.classes]);
                    end
                end
            else
                prb = activation{model.numLayer}(instance,3+(tree-1)*7);
                if rand < prb
                    prb = activation{model.numLayer}(instance,6+(tree-1)*7);
                    if rand < prb
                        postprb = postprb + reshape(hist(tree,5,:),[1,model.classes]);
                    else
                        postprb = postprb + reshape(hist(tree,6,:),[1,model.classes]);
                    end
                else
                    prb = activation{model.numLayer}(instance,7+(tree-1)*7);
                    if rand < prb
                        postprb = postprb + reshape(hist(tree,7,:),[1,model.classes]);
                    else
                        postprb = postprb + reshape(hist(tree,8,:),[1,model.classes]);
                    end
                end
            end
        end
        [prob,predict_class] = max(postprb);
        predict(instance) = predict_class;
    end
    accuracy = accuracy + sum(predict == batch_label);
end

accuracy = accuracy / n;
fprintf('test accuracy: %f\n', accuracy);
