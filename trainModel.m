function net = trainModel(net, data, cfg)

trAvg = [];
trAvgSq = [];

N = size(data,4);

for e = 1:cfg.numEpochs
    for i = 1:N
        x = data(:,:,:,i);

        [loss, grads] = dlfeval(@modelLoss, net, x, cfg);


        [net, trAvg, trAvgSq] = adamupdate( ...
            net, grads, trAvg, trAvgSq, i, cfg.learningRate);
    end
    disp(["Epoch", e, "Loss", extractdata(loss)]);
end
end
