function testModel(net, data, cfg)

for i = 1:5
    x_gt = data(:,:,:,i);

    y = forwardOperator(x_gt);
    y = y + cfg.noiseSigma * randn(size(y),'like',y);

    x_rec = admmSolverTest(y, net, cfg);


    figure;
    subplot(1,3,1), imshow(x_gt,[]), title("GT");
    subplot(1,3,2), imshow(imresize(y,[28 28]),[]), title("Input");
    subplot(1,3,3), imshow(x_rec,[]), title("Reconstruction");
end
end
