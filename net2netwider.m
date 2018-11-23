% function [ newmodel ] = net2netwider( model,pos1,pos2,newWidth )

% -- net  = network
% -- pos1 = position at which one has to widen the output
% -- pos2 = position at which the next weight layer is present
% -- newWidth   = new width of the layer

function newmodel = net2netwider(model,pos1,pos2,newWidth)

if strcmp( model.layers{pos1}.type, 'convolution') || strcmp(model.layers{pos1}.type, 'fullconnected')
    if strcmp(model.layers{pos1}.type, 'convolution')
        model.layers{pos1}.w = permute(model.layers{pos1}.w,[1,5,2,3,4]); % rearrange dims
        model.layers{pos2}.w = permute(model.layers{pos2}.w,[1,5,2,3,4]); % rearrange dims
    end
    
    if strcmp(model.layers{pos1}.type, 'convolution') && strcmp(model.layers{pos2}.type, 'convolution')
        assert(size(model.layers{pos2}.w,2) == size(model.layers{pos1}.w,1), 'failed sanity check')
        oldWidth = size(model.layers{pos1}.w,1);
        assert(newWidth > oldWidth,'new width should be greater than old width');
    elseif strcmp(model.layers{pos1}.type, 'fullconnected') && strcmp(model.layers{pos2}.type, 'fullconnected')
        assert(size(model.layers{pos2}.w,1) == size(model.layers{pos1}.w,2), 'failed sanity check')
        oldWidth = size(model.layers{pos1}.w,2);
        assert(newWidth > oldWidth,'new width should be greater than old width');
    else
        save('model_error','model');
        save('pos2','pos2');
        save('pos1','pos1');
        error('fc layer after conv is not supported in net2netwider')
    end
    
    
    if strcmp( model.layers{pos1}.type, 'convolution')
        size1 = size(model.layers{pos1}.w);
        nw1 = zeros([newWidth,size1(2:5)]);
        size2 = size(model.layers{pos2}.w);
        nw2 = zeros([size2(1),newWidth,size2(3:5)]);
    elseif strcmp(model.layers{pos1}.type, 'fullconnected')
        size1 = size(model.layers{pos1}.w);
        nw1 = zeros([size1(1),newWidth]);
        size2 = size(model.layers{pos2}.w);
        nw2 = zeros([newWidth,size2(2)]);
    end
    
    if strcmp( model.layers{pos1}.type, 'convolution')
        nc1 = zeros(newWidth,1);
    elseif strcmp(model.layers{pos1}.type, 'fullconnected')
        nc1 = zeros(1,newWidth);
    end
    
    %     copy the original weights over
    if strcmp( model.layers{pos1}.type, 'convolution')        
        nw1(1:oldWidth,:,:,:,:) = model.layers{pos1}.w;
        nw2(:,1:oldWidth,:,:,:) = model.layers{pos2}.w;
        nc1(1:oldWidth,:) = model.layers{pos1}.c;
    elseif strcmp(model.layers{pos1}.type, 'fullconnected')
        nw1(:,1:oldWidth) = model.layers{pos1}.w;
        nw2(1:oldWidth,:) = model.layers{pos2}.w;
        nc1(:,1:oldWidth) = model.layers{pos1}.c;
    end
    
    %    now do random selection on new weights
    tracking = num2cell(1:oldWidth); %isempty for checking
    for i = (oldWidth + 1): newWidth
        j = randi([1, oldWidth]);
        tracking{j} = [tracking{j};i];
        
        % copy the weights
        if strcmp( model.layers{pos1}.type, 'convolution')
            nw1(i,:,:,:,:) = model.layers{pos1}.w(j,:,:,:,:);
            nw2(:,i,:,:,:) = model.layers{pos2}.w(:,j,:,:,:);
        elseif strcmp(model.layers{pos1}.type, 'fullconnected')
            nw1(:,i) = model.layers{pos1}.w(:,j);
            nw2(i,:) = model.layers{pos2}.w(j,:);
        end
        nc1(i) = model.layers{pos1}.c(j);
    end
    
    for i = tracking
        ii = cell2mat(i)';
        for j = ii
            if strcmp( model.layers{pos1}.type, 'convolution')
                nw2(:,j,:,:,:) = nw2(:,j,:,:,:) / size(ii,2);
            elseif strcmp(model.layers{pos1}.type, 'fullconnected')
                nw2(j,:) = nw2(j,:) / size(ii,2);
            end
        end
    end
    
    if strcmp( model.layers{pos1}.type, 'convolution')
        model.layers{pos1}.layerSize(end) = newWidth;
    elseif strcmp(model.layers{pos1}.type, 'fullconnected')
        model.layers{pos1}.layerSize(1) = newWidth;
    end
    
    
    model.layers{pos1}.w = squeeze(permute(nw1,[1,3,4,5,2]));
    model.layers{pos2}.w = squeeze(permute(nw2,[1,3,4,5,2]));
    % use squeeze to take care of case when one dim has length 1
    
    model.layers{pos1}.gradw = zeros(size(model.layers{pos1}.w));
    model.layers{pos2}.gradw = zeros(size(model.layers{pos2}.w));
    
    model.layers{pos1}.c = nc1;
    model.layers{pos1}.gradc = zeros(size(model.layers{pos1}.c));
    
else
    error('Only fullyconnected and convolution supported')
end

newmodel = model;
