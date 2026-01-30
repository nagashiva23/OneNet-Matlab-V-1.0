function [trainData, testData] = loadDataset(cfg)

[XTrain, ~] = digitTrain4DArrayData;
[XTest,  ~] = digitTest4DArrayData;

trainData = XTrain;
testData  = XTest;

end
