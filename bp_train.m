function [loss_all,accuracy] = bp_train(model,portion)

data_list = read_data_list(model.data_path, model.classnames, ...
    model.volume_size + 2 * model.pad_size, 'train');

% create training data list and labels
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

% randomly select a portion of samples to evaluate training accuracy
m = floor(portion*n);
perm = randperm(n);
label = label(perm(1:m));
new_list = new_list(perm(1:m));
n = m;

% prepare data and label
test_index = [1:n];
batch_size = 32;
batch_num = ceil(n / batch_size);

accuracy = 0;
loss_all = 0;
for b = 1 : batch_num
    idx_end = min(b*batch_size, n);
    batch_index = test_index((b-1)*batch_size + 1 : idx_end);
    batch = read_batch(model, new_list(batch_index), false);
    batch_label = label(batch_index,:);
    [model, activation] = bp_forward(model, batch);
    
    [~, predict] = max(activation{end}, [], 2);
    accuracy = accuracy + sum(predict == batch_label);
    loss = - sum( log(activation{end}(full(sparse(1:length(batch_index), batch_label, 1)) == 1) ) );
    loss_all = loss_all + loss;
end

accuracy = accuracy / n;
loss_all = loss_all / n;
fprintf('train accuracy: %f\n', accuracy);
fprintf('average training loss: %f\n', loss_all);
