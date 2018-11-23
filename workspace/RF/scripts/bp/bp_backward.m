function [model, loss] = bp_backward(model, activation, label,hist,tree)
% backprop backward used for discriminative finetuning

 global kConv_backward kConv_backward_c;
 global kConv_weight kConv_weight_c;

numLayer = model.numLayer;
batch_size = size(label,1);
numClass = size(label,2);
error = cell(numLayer, 1);

s_node = (tree-1)*7+1;
e_node = (tree-1)*7+7;
act = activation{numLayer}(:,s_node:e_node);

% compute mu for all data
mu = zeros(batch_size,size(hist,2));
for i = 1:batch_size
    mu(i,1) = act(i,1)*act(i,2)*act(i,4);
    mu(i,2) = act(i,1)*act(i,2)*(1-act(i,4));
    mu(i,3) = act(i,1)*(1-act(i,2))*act(i,5);
    mu(i,4) = act(i,1)*(1-act(i,2))*(1-act(i,5));
    mu(i,5) = (1-act(i,1))*act(i,3)*act(i,6);
    mu(i,6) = (1-act(i,1))*act(i,3)*(1-act(i,6));
    mu(i,7) = (1-act(i,1))*(1-act(i,3))*act(i,7);
    mu(i,8) = (1-act(i,1))*(1-act(i,3))*(1-act(i,7));
end

% compute product = pi*mu for all data
product = zeros(batch_size,size(hist,2),numClass);
% compute PT
predict = zeros(size(label));
for i = 1:batch_size
    for j = 1:numClass
        for k = 1:size(hist,2)
            product(i,k,j) = hist(tree,k,j)*mu(i,k);         
            predict(i,j) = predict(i,j) + product(i,k,j);
        end
    end
end

loss = -mean(log(predict(label == 1)));

error{numLayer} = zeros(size(activation{numLayer}));

for i = 1:batch_size
    lb = find(label(i,:));
        
    error{numLayer}(i,s_node-1+4) = act(i,4)*product(i,2,lb)/predict(i,lb) - (1-act(i,4))*product(i,1,lb)/predict(i,lb);
    error{numLayer}(i,s_node-1+5) = act(i,5)*product(i,4,lb)/predict(i,lb) - (1-act(i,5))*product(i,3,lb)/predict(i,lb);
    error{numLayer}(i,s_node-1+6) = act(i,6)*product(i,6,lb)/predict(i,lb) - (1-act(i,6))*product(i,5,lb)/predict(i,lb);
    error{numLayer}(i,s_node-1+7) = act(i,7)*product(i,8,lb)/predict(i,lb) - (1-act(i,7))*product(i,7,lb)/predict(i,lb);
    
    error{numLayer}(i,s_node-1+2) = act(i,2)*(product(i,3,lb)+product(i,4,lb))/predict(i,lb) - (1-act(i,2))*(product(i,1,lb)+product(i,2,lb))/predict(i,lb) ;
    error{numLayer}(i,s_node-1+3) = act(i,3)*(product(i,7,lb)+product(i,8,lb))/predict(i,lb) - (1-act(i,3))*(product(i,5,lb)+product(i,6,lb))/predict(i,lb);
    
    error{numLayer}(i,s_node-1+1) = act(i,1)*(product(i,5,lb)+product(i,6,lb)+product(i,7,lb)+product(i,8,lb))/predict(i,lb) - (1-act(i,1))*(product(i,1,lb)+product(i,2,lb)+product(i,3,lb)+product(i,4,lb))/predict(i,lb); 
end

for l = model.numLayer-1 : -1 : 2
    if l == 1
        error{l} = myConvolve(kConv_backward, error{l+1}, model.layers{l+1}.w, model.layers{l+1}.stride, 'backward');
    elseif strcmp(model.layers{l+1}.type, 'convolution')
        error{l} = myConvolve(kConv_backward_c, error{l+1}, model.layers{l+1}.w, model.layers{l+1}.stride, 'backward');
    else 
		error{l} = double(error{l+1}) * double(model.layers{l+1}.w');
        if strcmp(model.layers{l}.type, 'convolution')
            error{l} = reshape(error{l}, [batch_size, model.layers{l}.layerSize]);
        end
    end
    error{l} = error{l} .* double(activation{l}) .* double(( 1 - activation{l} ));% for sigmoid
end

% Compute the gradients for each layer
for l = 2 : numLayer
    if l == 2
        model.layers{l}.grdw = myConvolve(kConv_weight, activation{l-1}, error{l}, model.layers{l}.stride, 'weight')*size(error{l},2)^3;
        model.layers{l}.grdc = sum(reshape(error{l}, [], model.layers{l}.layerSize(4)),1)' ./ batch_size; 
    elseif strcmp(model.layers{l}.type, 'convolution')
        model.layers{l}.grdw = myConvolve(kConv_weight_c, activation{l-1}, error{l}, model.layers{l}.stride, 'weight')*size(error{l},2)^3;
        model.layers{l}.grdc = sum(reshape(error{l}, [], model.layers{l}.layerSize(4)),1)' ./ batch_size; 
    else
        activation{l-1} = reshape(activation{l-1}, batch_size, []);
        model.layers{l}.grdw = double(activation{l-1}') * error{l}./ batch_size;
        model.layers{l}.grdc = mean(error{l}, 1);       
    end
end
