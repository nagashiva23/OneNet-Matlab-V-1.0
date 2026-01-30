function x = adjointOperator(y, outSize)

x = zeros(outSize, 'like', y);
x(1:2:end, 1:2:end, :) = y;

h = fspecial('gaussian', 5, 1);
x = imfilter(x, h, 'circular');
end
