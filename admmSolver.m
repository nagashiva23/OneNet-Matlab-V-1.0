function x = admmSolver(y, net, cfg)

x = adjointOperator(y, cfg.imageSize);
z = x;
u = zeros(size(x));

rho = cfg.lambda;

for k = 1:cfg.maxIter
    dlZ = dlarray(x + u,'SSCB');
    z = extractdata(predict(net, dlZ));

    rhs = adjointOperator(y, cfg.imageSize) + rho*(z - u);
    x = rhs / (1 + rho);

    u = u + x - z;
end
end
