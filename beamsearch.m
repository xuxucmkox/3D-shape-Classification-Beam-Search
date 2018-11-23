function [models,parent,actions,bs_last] = beamsearch(d_limit,k,smodel)
    % d_limit: the maximum depth search tree has
    % k: the number of best candidates to expand
    % model: starting model
    % bs_last is the beam size at last depth
    
    rng('shuffle');
    kernels;
    addpath 3D;
    addpath bp;
    addpath generative;
    addpath util;
    
    depth = 1;
    models = [smodel];
    f_action.type = 'no';
    f_action.layer = 0;
    f_action.increment = 0;
    % fake action, no use
    actions = [f_action];
    parent(1) = 0;
    models_e = smodel; % models to expand
    m = 1; % index for parent of model
    while depth <= d_limit   
        i = 1; % index for models_c
        actions_c = [];
        for model = models_e
            a = 1; % index for action candidates for this current model
            n = model.numLayer;
            for l = 2:(n-1)
                if strcmp(model.layers{l}.type, 'convolution')&& strcmp(model.layers{l+1}.type, 'convolution')
                    for j = [1] % number of filters must be multiples of 16
                        actions_e(a).type = 'wider';
                        actions_e(a).layer = l;
                        actions_e(a).increment = j;
                        a = a + 1;
                    end
                elseif strcmp(model.layers{l}.type, 'fullconnected')&& strcmp(model.layers{l+1}.type, 'fullconnected')
                    for j = [0.5,1] % 
                        actions_e(a).type = 'wider';
                        actions_e(a).layer = l;
                        actions_e(a).increment = j;
                        a = a + 1;
                    end
                end
                if (strcmp(model.layers{l}.type, 'convolution') && mod(model.layers{l}.kernelSize(1),2) == 1) || strcmp(model.layers{l}.type, 'fullconnected')
                    actions_e(a).type = 'deeper';
                    actions_e(a).layer = l;
                    actions_e(a).increment = 0;
                    a = a + 1;
                end
            end
            
            % compute scores for all candidate models
            for action = actions_e
                if strcmp(action.type,'wider')
                    if strcmp(model.layers{action.layer}.type, 'convolution')
                        layersize = model.layers{action.layer}.layerSize(end);
                    elseif strcmp(model.layers{action.layer}.type, 'fullconnected');
                        layersize = model.layers{action.layer}.layerSize(1);
                    end
                    models_c(i) = bp_finetune(net2netwider(model,action.layer,action.layer+1,layersize*(1+action.increment)),10,depth/d_limit); 
                    [losses(i),scores(i)] = bp_train(models_c(i),depth/d_limit); % use bp_train to evaluate accuracy
                elseif strcmp(action.type,'deeper')
                    models_c(i) = bp_finetune(net2netdeeper(model,action.layer,'sigmoid'),10,depth/d_limit);
                    [losses(i),scores(i)] = bp_train(models_c(i),depth/d_limit);
                end
                parent_c(i) = m; % parent info for candidate models
                i = i + 1;
            end
            m = m + 1;
	    
            actions_c = [actions_c,actions_e];
            clear actions_e;
        end
        
        [sorted,indices] = sort(scores,'descend');
        if size(indices,2)<k
            bs = size(indices,2);
        else
            bs = k;
        end
        models_e = models_c(indices(1:bs));
        models = [models,models_e];
        actions = [actions, actions_c(indices(1:bs))];
        parent = [parent,parent_c(indices(1:bs))];
        
        % reset the candidates sets
        clear actions_e actions_c parent_c models_c losses scores;
        
        depth = depth + 1;
    end
    bs_last = bs;
end
