function newmodel = beamsearch(d_limit,k,model)
    % d_limit: the maximum depth search tree has
    % k: the number of best candidates to expand
    % model: starting model
    depth = 1;
    models = [];
    actions = [];
    parent(1) = 0;
    models_e = model; % models to expand
    while depth <= d_limit   
        i = 1; % index for models_c
        for model = models_e
            n = model.numLayer;
            m = 1; % index for model in models_e
            a = 1; % index for action candidates
            for l = 2:(n-1)
                for i = [1,2] % number of filters must be multiples of 16
                    if strcmp(model.layers{l}.type, 'convolution')&& strcmp(model.layers{l+1}.type, 'convolution')
                        actions_c(a).type = 'wider';
                        actions_c(a).layer = l;
                        actions_c(a).increment = i;
                        a = a + 1;
                    elseif strcmp(model.layers{l}.type, 'fullconnected')&& strcmp(model.layers{l+1}.type, 'fullconnected')
                        actions_c(a).type = 'wider';
                        actions_c(a).layer = l;
                        actions_c(a).increment = i;
                        a = a + 1;
                    end
                end
                if (strcmp(model.layers{l}.type, 'convolution') && mod(model.layers{l}.kernelSize(1),2) == 1) || strcmp(model.layers{l}.type, 'fullconnected')
                    actions_c(a).type = 'deeper';
                    actions_c(a).layer = l;
                    actions_c(a).increment = 0;
                    a = a + 1;
                end
            end
            
            % compute scores for all candidate models
            for action = actions_c
                if strcmp(action.type,'wider')
                    if strcmp(model.layers{action.layer}.type, 'convolution')
                        layersize = model.layers{action.layer}.layerSize(end);
                    elseif strcmp(model.layers{action.layer}.type, 'fullconnected');
                        layersize = model.layers{action.layer}.layerSize(1);
                    end
                    models_c(i) = net2netwider(model,action.layer,action.layer+1,layersize*(1+action.increment));
                    [losses(i),scores(i)] = bp_train(models_c(i),depth/d_limit); % use bp_train to evaluate accuracy
                elseif strcmp(actions_c(i).type,'deeper')
                    models_c(i) = net2netdeeper(model,action.layer,'sigmoid');
                    [losses(i),scores(i)] = bp_train(models_c(i),depth/d_limit);
                end
                parent_c(i) = m; % parent info for candidate models
                i = i + 1;
            end
            m = m + 1;
        end
        
        [sorted,indices] = sort(scores,'descend');
        models_e = models_c(indices(1:k),:);
        models = [models;models_e];
        actions = [actions; actions_c(indices(1:k),:)];
        parent = [parent;parent_c];
        
        % reset the candidates sets
        clear actions_c parent_c models_c losses scores;
        
        depth = depth + 1;
    end
    % then backtrack to find the best dpeth 2 candidate

end