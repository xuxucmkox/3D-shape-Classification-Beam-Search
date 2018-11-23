D = 5;
K = 3;
acc_train = zeros(D,K);
acc_test= zeros(D,K);
time = zeros(D,K);

addpath bp;

for depth = 1:D
    for k = 1:K
        fprintf('the %d depth, %d width beam search begins:\n',depth,k);
        d = 1;
        load s_model;
        tic;
        record_actions = [];
        while d<=depth
            fprintf('the %d depth prior beam search begins:\n',d);
            [models,parent,actions,bs_last] = beamsearch(depth-d+1,k,model);
            record_actions = [record_actions,actions];
            
            for i =1:bs_last
                [loss,scores(i)] = bp_train(models(end-i+1),1);
            end
            [sorted,indices] = sort(scores(1:bs_last),'descend');
            model_index = indices(1)-bs_last+size(models,2);

            while parent(model_index) ~= 1
                model_index = parent(model_index);
            end
            fprintf('the best model to expand at %d depth prior beam search is found:\n',d);
            model = models(model_index);
            model = bp_finetune(model,10,1);
            d = d + 1;
            clear scores;
        end
        toc;
        
        save(strcat('actions for','depth:',int2str(depth),'width:',int2str(k)),'record_actions');
        
        elapsedtime = toc;
        time(depth,k) = elapsedtime;
        [~,accuracy] = bp_train(model,1);
        acc_train(depth,k) = accuracy;
        [~,accuracy] = bp_test(model);
        acc_test(depth,k) = accuracy;
        fprintf('the %d depth, %d width beam search is finished:\n',depth,k);
    end
end

save('acc_train','acc_train');
save('acc_test','acc_test');
save('time','time');

