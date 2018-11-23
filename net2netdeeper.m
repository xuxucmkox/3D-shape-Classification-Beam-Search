function newmodel = net2netdeeper(model,pos,nonlin)
% -- model  = network
% -- pos = position at which the layer has to be deepened
% -- nonlin = type of non-linearity to insert 'sigmoid' or 'relu'

model.numLayer = model.numLayer + 1;
% add a new empty layer to pos, existing layer positions are moved
% accordingly
for l = model.numLayer:-1:(pos+1)
    if l ~= pos+1;
        model.layers{l} = model.layers{l-1};
    else
        model.layers{l} = model.layers{pos}; % copy last layer temporarily, values will be changed after
    end
end

newlayer = model.layers{pos+1};

if strcmp( newlayer.type, 'fullconnected') == 1
    newlayer.actFun = nonlin;
    % newlayer.layerSize = newlayer.layerSize; layer size is not changed
    newlayer.w = single(eye(newlayer.layerSize));
    newlayer.c = single(zeros(1,newlayer.layerSize));
    
elseif strcmp( newlayer.type, 'convolution') == 1
    % m.kH % 2 == 1 and m.kW % 2 == 1, kernel height and width have to be
    % odd, so that kernel matrix has a center value
    padkz1 = (newlayer.kernelSize(1)-1)/2; % used as zero padding size
    padkz2 = (newlayer.kernelSize(2)-1)/2;
    padkz3 = (newlayer.kernelSize(3)-1)/2;
    
    newlayer.actFun = nonlin;
    Size = size(newlayer.w);
    newlayer.w = single(zeros(Size(1),Size(2),Size(3),Size(4),Size(1)));
    
    c1 = floor(newlayer.kernelSize(1) / 2) + 1;
    c2 = floor(newlayer.kernelSize(2) / 2) + 1;
    c3 = floor(newlayer.kernelSize(3) / 2) + 1;
    
    % the following line seems to be irrelevant, weight matrix size cant
    % only have two dimensions
    %     local restore = false
    %     if m2.weight:dim() == 2 then
    %         m2.weight = m2.weight:view(m2.weight:size(1), m2.nInputPlane, m2.kH, m2.kW)
    %         restore = true
    %     end
    
    for i = 1:Size
        newlayer.w(i,c1,c2,c3,i)=1;
    end
    
    %     if restore then
    %         m2.weight = m2.weight:view(m2.weight:size(1), m2.nInputPlane * m2.kH * m2.kW)
    %     end
    
    newlayer.c = single(zeros(size(newlayer.c)));
else
    error('Only fullyconnected and convolution supported')
end

newlayer.stride = 1;
newlayer.hasPadding = 1;
newlayer.paddingsize = [padkz1,padkz2,padkz3];
model.layers{pos+1} = newlayer;
newmodel = model;

end

%  isfield(model1.layers{pos+1,1},'hasPadding') used in bp_forward to add
%  paddings to the new added layer activation 
