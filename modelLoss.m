function [loss, grads] = modelLoss(net, x_gt, cfg)

x_gt = dlarray(x_gt,'SSCB');

% Forward model (identity + noise for now)
y = x_gt + cfg.noiseSigma * randn(size(x_gt),'like',x_gt);

% DIFFERENTIABLE solver
x_hat = admmSolverTrain(y, net, cfg);

loss = mse(x_hat, x_gt);

grads = dlgradient(loss, net.Learnables);
end
