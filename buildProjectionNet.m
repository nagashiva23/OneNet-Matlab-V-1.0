function net = buildProjectionNet(cfg)

layers = [
    imageInputLayer(cfg.imageSize, Normalization="none")

    convolution2dLayer(3,64,Padding="same")
    reluLayer

    convolution2dLayer(3,64,Padding="same")
    reluLayer

    convolution2dLayer(3,1,Padding="same")
];

net = dlnetwork(layers);
end
