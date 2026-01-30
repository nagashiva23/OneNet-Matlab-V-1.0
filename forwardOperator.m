function y = forwardOperator(x)

h = fspecial('gaussian', 5, 1);
xBlur = imfilter(x, h, 'circular');

y = xBlur(1:2:end, 1:2:end, :);
end
