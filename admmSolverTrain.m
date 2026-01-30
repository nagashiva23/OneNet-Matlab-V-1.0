function x = admmSolverTrain(y, net, cfg)

% y is dlarray
x = y;
z = y;
u = zeros(size(y),'like',y);

rho = cfg.lambda;

for k = 1:cfg.maxIter
    % z-update (learned projection) — KEEP dlarray
    z = forward(net, x + u);

    % x-update (identity operator for stability)
    x = (y + rho*(z - u)) / (1 + rho);

    % dual update
    u = u + x - z;
end
end
