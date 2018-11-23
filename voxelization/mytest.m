load('0.mat');
FV = struct('faces',surface.TRIV,'vertices',surface.VERT);
patch(FV,'FaceColor',[1 0 0]);
instance=polygon2voxel(FV,[30 30 30],'auto');
instance = int8(instance);
figure;
plot3D(instance);