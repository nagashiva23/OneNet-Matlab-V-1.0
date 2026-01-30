function x = admmSolverTest(y, net, cfg)

x = y;
z = y;
u = zeros(size(y));

rho = cfg.lambda;

for k = 1:cfg.maxIter
    dlZ = dlarray(x + u,'SSCB');
    z = extractdata(predict(net, dlZ));

    x = (y + rho*(z - u)) / (1 + rho);
    u = u + x - z;
end
end
